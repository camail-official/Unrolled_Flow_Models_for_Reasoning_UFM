"""Latent flow with rollout training.

Training draws, per sample, a time grid t_0 < t_1 < ... < t_N, starts the answer
slot on the interpolant z(t_0) = (1 - t_0) * noise + t_0 * embed(answer), unrolls
N Euler steps of the x-prediction flow through the network, and applies a single
cross-entropy at the terminal state. Gradients flow through the whole rollout,
attenuated by `grad_decay` at every step. Clean positions (graph, question,
markers, padding) are reset to their embeddings after every step.

The supervised node follows a staged curriculum over the shortest root->target
path: at stage K the target is node v_{min(K+1, hops)} of the path (with
probability `replay_prob` an earlier node), so training starts at the first hop
and reaches the answer by stage `hops - 1`.

Inference starts from pure noise at t = 0 and integrates a uniform grid of `num_steps`
Euler steps, decoding the terminal state.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class LatentFlow:
    def __init__(self, vocab_size: int, rollout_steps: int, grad_decay: float, grid: str,
                 t0_max: float, t_max: float, fixed_tN: float, random_tN: bool,
                 t_clamp_eps: float, label_smoothing: float, bf16_acc32: bool,
                 max_stage: int, replay_prob: float):
        assert grid in ("stratified", "random")
        assert not (fixed_tN > 0.0 and random_tN), "pick one terminal-knot rule"
        self.vocab_size = vocab_size
        self.N = rollout_steps
        self.grad_decay = grad_decay
        self.grid = grid
        self.t0_max = t0_max              # start time t_0 ~ U[0, t0_max]
        self.t_max = t_max                # interior knots live in (t_0, t_max)
        self.fixed_tN = fixed_tN          # > 0: terminal knot fixed at this time
        self.random_tN = random_tN        # terminal knot ~ U(t_{N-1}, t_max)
        self.t_clamp_eps = t_clamp_eps    # floor on the (1 - t) Euler denominator
        self.label_smoothing = label_smoothing
        self.bf16_acc32 = bf16_acc32      # bf16 network calls, fp32 rollout state
        self.max_stage = max_stage
        self.replay_prob = replay_prob
        self.stage = 0                    # set by the trainer every epoch

    # ------------------------------------------------------------------ training
    def _interior_positions(self, B: int, device) -> torch.Tensor:
        """Relative positions u in (0, 1) of the N-1 interior knots, sorted, [B, N-1]."""
        N = self.N
        if self.grid == "stratified":                       # one knot per bin (k/N, (k+1)/N)
            kk = torch.arange(1, N, device=device).float()
            return (kk.unsqueeze(0) + torch.rand(B, N - 1, device=device)) / N
        return torch.sort(torch.rand(B, N - 1, device=device), dim=1)[0]   # i.i.d. uniform, sorted

    def _time_grid(self, t0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Per-sample knots [B, N+1]: t_0, N-1 interior knots, terminal knot."""
        B, N, device = t0.shape[0], self.N, t0.device
        knots = t0 + (self.t_max - t0) * u
        if self.fixed_tN > 0.0:
            tN = torch.full((B, 1), self.fixed_tN, device=device)
        elif self.random_tN:
            t_prev = knots[:, -1:] if N > 1 else t0
            tN = t_prev + torch.rand(B, 1, device=device) * (self.t_max - t_prev)
        else:
            tN = torch.full((B, 1), self.t_max, device=device)
        return torch.cat([t0, knots, tN], dim=1)

    def _curriculum_target(self, path: torch.Tensor, path_len: torch.Tensor) -> torch.Tensor:
        """One-hot [B, V] of the supervised path node at the current stage."""
        B, device = path.shape[0], path.device
        K = min(self.stage, self.N - 1)
        K_eff = torch.minimum(path_len - 1, torch.full_like(path_len, K))
        if self.replay_prob > 0.0:
            replay = torch.rand(B, device=device) < self.replay_prob
            r = (torch.rand(B, device=device) * (K_eff + 1).float()).long()
            K_eff = torch.where(replay, torch.minimum(r, K_eff), K_eff)
        i = torch.minimum(K_eff + 1, path_len).clamp(min=1)
        node = path.gather(1, i.clamp(max=path.shape[1] - 1).unsqueeze(1)).squeeze(1)
        return F.one_hot(node, self.vocab_size).float()

    def loss(self, model, batch: dict) -> torch.Tensor:
        x0, cond_mask, attn = batch["input_ids"], batch["cond_mask"], batch["attention_mask"]
        answer_idx, path, path_len = batch["answer_idx"], batch["path"], batch["path_len"]
        B, device = x0.shape[0], x0.device
        m = cond_mask.unsqueeze(-1).float()

        u = self._interior_positions(B, device)
        target = self._curriculum_target(path, path_len)
        t0 = self.t0_max * torch.rand(B, 1, device=device)
        clean = model.embed_ids(x0)
        noise = torch.randn_like(clean)
        z = m * clean + (1.0 - m) * ((1.0 - t0.view(B, 1, 1)) * noise + t0.view(B, 1, 1) * clean)
        t_grid = self._time_grid(t0, u)
        terminal_euler = self.fixed_tN > 0.0 or self.random_tN

        for j in range(self.N):
            if self.grad_decay < 1.0:                       # value unchanged, gradient scaled
                z = self.grad_decay * z + (1.0 - self.grad_decay) * z.detach()
            t_j = t_grid[:, j]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16_acc32):
                h = model.forward_d(z, t_j, attn)
            h = h.float()
            if j == self.N - 1 and not terminal_euler:
                z_next = h                                  # terminal state = the clean estimate
            else:
                dt = (t_grid[:, j + 1] - t_j).view(B, 1, 1).clamp(min=1e-6)
                beta = dt / (1.0 - t_j.view(B, 1, 1)).clamp(min=self.t_clamp_eps)
                z_next = (1.0 - beta) * z + beta * h        # Euler step of the x-prediction flow
            z = m * clean + (1.0 - m) * z_next

        if self.label_smoothing > 0.0:
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.vocab_size
        log_probs = model.decode(z[torch.arange(B, device=device), answer_idx])
        return -(target * log_probs).sum(dim=-1).mean()

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def sample(self, model, batch: dict, num_steps: int) -> torch.Tensor:
        """Token predictions [B, L] after `num_steps` Euler steps from pure noise."""
        x0, cond_mask, attn = batch["input_ids"], batch["cond_mask"], batch["attention_mask"]
        B, device = x0.shape[0], x0.device
        m = cond_mask.unsqueeze(-1).float()
        clean = model.embed_ids(x0)
        z = m * clean + (1.0 - m) * torch.randn_like(clean)
        t_vals = torch.linspace(0.0, self.t_max, num_steps + 1, device=device)
        for i in range(num_steps):
            t = t_vals[i].expand(B)
            h = model.forward_d(z, t, attn)
            if i == num_steps - 1:
                z = h
            else:
                dt = (t_vals[i + 1] - t_vals[i]).item()
                z = z + dt * (h - z) / (1.0 - t.view(B, 1, 1)).clamp(min=self.t_clamp_eps)
            z = m * clean + (1.0 - m) * z
        return model.decode(z).argmax(dim=-1)
