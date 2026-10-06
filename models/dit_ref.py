"""UFM model for Sudoku / Maze: a DiT that predicts the clean latent endpoint, plus the Euler
rollout it is trained through (see `_rollout`, `_step`, `forward`, `forward_eval`).

The paper runs use: block_type=lightning (RMSNorm + SwiGLU, adaLN-Zero, RoPE-2D), x_pred,
stable_w, psi_update (sphere retraction), no_time, rand_t0, train_N=24 / n_back=6,
bf16_acc32 (see config/*.yaml). Every other flag in DiTRefConfig is an ablation or a variant
that is NOT used by any paper configuration; the defaults keep them off.

Two block flavours via `block_type` (both keep adaLN-Zero + RoPE-2D + canonical time embedder):
  canonical : Peebles DiT/SiT block — LayerNorm(no affine, eps 1e-6) + plain MHSA (no QK-norm)
              + GELU MLP 4x.
  lightning : LightningDiT block — RMSNorm + SwiGLU. Both flavours feed t in [0,1] raw to the
              sinusoids. Dropout at 4 sites.
"""
from typing import Dict, List, Optional
import math

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from pydantic import BaseModel

from models.layers import rms_norm, CosSin, CastedEmbedding, CastedLinear, apply_rotary_pos_emb, _find_multiple
from models.sparse_emb import CastedSparseEmbedding


class DiTRefConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int = 1  # used when puzzle_sparse (ARC); else unused
    # ARC: per-(task,aug) sparse prefix (TRM scheme) instead of the shared learned prefix;
    # ndim-vector per identifier, zero-init, SignSGD-trained, padded to puzzle_emb_len tokens
    puzzle_sparse: bool = False
    # puzzle_id_remap: path to a JSON int list mapping per-(task,aug) id -> per-TASK id.
    # Collapses the sparse table to one aug-shared vector per task (blank id 0 maps to 0).
    puzzle_id_remap: str = ""
    puzzle_emb_ndim: int = 0
    # dyn_crop (ARC, pairs with size_bucketing): crop each batch's grids to the max content
    # extent (rounded up to a multiple of 6), shrinking both prompt and answer blocks;
    # rope/mask/noise/readout follow the cropped length; loss head crops labels via _crop_idx
    dyn_crop: bool = False
    causal: bool = False             # unused

    block_type: str = "canonical"    # canonical | lightning

    # ProsQA's DiTBlock reads AdaLN modulation from a GLOBALLY SHARED time embedding whose
    # width (`time_dim`) is independent of d_model -- e.g. time_dim=128 with d_model=768.
    # Ours ties the embedding width to hidden_size everywhere. 0 (default) preserves that
    # the default conditioning path is unchanged;
    # >0 narrows the shared embedding AND every block's/final layer's adaLN input to this
    # width, replicating ProsQA's bottleneck rather than approximating it with a per-block
    # adapter (see RefTimestepEmbedder / DiTRefBlock.adaLN / final_adaLN construction).
    adaln_dim: int = 0

    n_layers: int = 2
    hidden_size: int = 432
    num_heads: int = 12              # head_dim 36 (%4==0 -> 2D-RoPE safe)
    # head_dim override (0 = hidden_size // num_heads). >0 decouples the attention inner width
    # num_heads*head_dim from the residual width, e.g. a TRM-like 8x64 = 512 attention inside
    # a 128-wide residual stream (qkv: d -> 3*512, o: 512 -> d). RoPE uses this head_dim.
    head_dim: int = 0
    expansion: float = 4.0
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    # rope_mode: "2d" (default, unchanged) splits rotation pairs row/col by grid coordinate;
    # "1d" uses a single flat sequence-position axis (standard RoPE, no grid structure);
    # "none" zeroes the rotation (cos=1, sin=0 identity) -- no rotary positional signal at
    # all. (The boolean `rope_2d` field is accepted for compatibility and has no effect;
    # `rope_mode` is the switch.)
    rope_mode: str = "2d"
    grid_size: int = 9
    puzzle_emb_len: int = 0

    # Canonical DiT time embedder: 256 sinusoidal dims -> Linear-SiLU-Linear(d). t raw in [0,1].
    time_freq_dim: int = 256

    dropout: float = 0.0             # 4 sites: attn probs, attn out, MLP hidden, MLP out
    # vanilla_gpt2: plain pre-norm transformer block, NOTHING from DiT --
    #   x = x + attn(norm(x));  x = x + mlp(norm(x))
    # no adaLN, no shift/scale/gate, no time embedder, no final adaLN. Branches are
    # LIVE at init (the DiT block is identity-at-init). Norm kind still follows
    # block_type (canonical=LayerNorm, lightning=RMSNorm); mlp_kind overrides the MLP.
    # learned_cond: DELETE the time embedder entirely and feed adaLN a learned
    # d-vector instead. Functionally identical to no_time (cond was already a
    # constant), but drops 256*d+d + d*d+d params of dead sinusoidal machinery.
    learned_cond: bool = False
    # rand_N_min/max > 0: sample the rollout depth per batch, N ~ U{min..max}
    # (n_back still fixed). Not used by the paper configs.
    rand_N_min: int = 0
    rand_N_max: int = 0
    vanilla_gpt2: bool = False
    mlp_kind: str = ""               # "" = follow block_type | gelu | swiglu
    # mlp_t (TRM's "MLP on L instead of transformer", trm.py TinyRecursiveReasoningModel
    # ACTV1Block, mlp_t=True): replaces token-mixing self-attention with a SwiGLU applied
    # across the SEQUENCE axis (transpose to [B,D,L], SwiGLU(hidden_size=L), transpose
    # back) -- fixed-size, non-permutation-invariant, only valid for a FIXED total sequence
    # length (asserted against dyn_crop). Channel-mixing self.mlp is unchanged either way.
    # (TRM's Sudoku setting.) Not used by the paper configurations.
    mlp_t: bool = False

    # Flow engine 
    train_N: int = 12
    n_back: int = 6
    time_sampling: str = "stratified"
    eval_Ns: List[int] = [1, 8, 16, 64]
    x_pred: bool = False
    t_clamp_eps: float = 0.05
    noise_scale: float = 1.0
    normalize_z: bool = False
    # Reference-run knobs:
    # [t_min, t_max] grid support; rand_t0 hinted-interpolant start (t0_max scheduled by
    # pretrain.py); grad_decay leap on tracked BPTT steps. Init noise is RAW (not sphered).
    t_min: float = 0.0
    t_max: float = 1.0
    rand_t0: bool = False
    t0_start: float = 0.6
    t0_min: float = 0.2
    t0_decay_frac: float = 0.3
    # Target-content curriculum (a grid port of the ProsQA staged curriculum; unused here): stage the
    # LOSS TARGET, not the rollout depth or an intermediate forward pass. At stage k of
    # curr_stages, the loss counts a fixed row-major-ordered k/curr_stages fraction of the
    # "reasoning" cells -- defined dataset-agnostically as inputs != labels (true at every
    # blank Sudoku cell, and equally at every maze path cell) -- plus ALL "given" cells
    # (inputs == labels), which are free and never need staging. Exactly one CE per
    # rollout throughout; only which positions count toward it changes across EPOCHS, not
    # within a single forward pass -- this is why it is not deep supervision (see
    # models/losses.py::FlowLossHead._curriculum_mask for the enforcement of that
    # distinction and pretrain.py::update_curriculum_schedule for the epoch schedule,
    # which mirrors update_t0_schedule's t0_max decay exactly).
    curriculum: str = "none"      # "none" | "staged"
    curr_stages: int = 4          # number of cumulative reveal stages (1/4, 2/4, 3/4, 4/4)
    curr_frac: float = 0.3        # schedule completes (reaches stage curr_stages) by this
                                   # fraction of total epochs, then holds at "everything"
    grad_decay: float = 1.0
    # injected by pretrain.create_model from the top-level PretrainConfig; only the sparse
    # puzzle embedding needs it (one scratch slot per accumulation micro-batch).
    grad_accum: int = 1
    # exp_update tau-mode : interior steps
    # z_{k+1} = exp(-h) z_k + (1-exp(-h)) f (exact integrator of dz/dtau = f - z, convex,
    # bounded without normalize_z), grids on [0, tau_max], model conditioned on RAW tau,
    # final step = usual x-pred readout (CE on the last model call).
    exp_update: bool = False
    tau_max: float = 10.0
    # tau_uniform_t (exp_update only): keep the TAU clock and the exp/blend update, but place
    # the interior TRAINING knots equally spaced in t instead of equally spaced in tau, over
    # the SAME t-range. Isolates "was the exponential GRID the lever, or just the tau clock?"
    # uniform-in-tau  -> geometric in t, constant blend w per step (the default)
    # uniform-in-t    -> constant dt per step, blend w grows toward the end
    tau_uniform_t: bool = False
    # stable_w (t-mode only): interior blend weight w = dt/(1-t_k) computed through the
    # log-time identity w = 1 - e^{-(tau_{k+1}-tau_k)} = 1 - (1-t_{k+1})/(1-t_k), bounded
    # in (0,1] with no eps-clamp; model stays conditioned on NORMAL time t on [t_min, t_max]
    stable_w: bool = False
    # no_time: time-BLIND velocity f(z) — the embedder always sees t=0, so adaLN collapses
    # to learned constants and each step becomes the same autonomous operator (weight-tied
    # looped transformer / TRM-FPRM regime). Integrator grids/blends are unchanged.
    no_time: bool = False
    # carry training (REQUIRES no_time: cached states have no clock identity to violate).
    # Persist each sample's pre-readout state across visits; every optimizer step runs only
    # the n_back tracked window + readout (no warmup rollout). Effective depth accumulates
    # across visits (TRM/FPRM one-segment-per-step design). carry_pool = dataset example
    # count (buffer rows); carry_fresh_p = prob of restarting a sample from the usual
    # interpolant init (keeps early-flow coverage for from-noise eval).
    carry_pool: int = 0
    carry_fresh_p: float = 0.15
    # grid_geom_G > 0: TRAINING knots uniform in tau up to G (constant blend weight w),
    # aligning the train-time schedule with the geometric inference grid (spend steps only
    # in the late region). 0 = uniform-in-t knots (the paper setting).
    grid_geom_G: float = 0.0
    # t0_zero_p: with this probability a fresh start uses t0 = 0 (pure noise) instead of
    # the sampled t0 -- the "mixture" free-start distribution (50% noise / 50% U(0,cap))
    t0_zero_p: float = 0.0
    # terminal boundary mixture (mirror of t0_zero_p at the other end of the grid):
    # with prob t_end_p the final knot is t_end_val (<1) instead of 1; t_int_cap keeps
    # ordinary interior knots out of the near-terminal band in all cases
    t_end_p: float = 0.0
    t_end_val: float = 0.99
    t_int_cap: float = 1.0
    # t0_geom_G > 0: fresh-start t0 sampled under the EXPONENTIAL warp t0 = 1-e^{-G u},
    # u ~ U(0,1) (log-SNR-uniform training-time distribution)
    t0_geom_G: float = 0.0
    # ecld_lambda > 0 (Categorical Flow Maps, Roos/Davis et al.): endpoint-consistency
    # self-distillation for the TIME-BLIND model. Sample a jump fraction gamma via their
    # logit-normal (s,t) scheme; X = z + gamma*(f(z)-z); loss += lambda * CE(sg f(X), f(z)).
    # Teaches one big jump to decode like many small steps -> few-step (1-2 NFE) inference.
    ecld_lambda: float = 0.0
    # residual head: D = z + W_out(AdaNorm(h_answer, tau)); identity at init.
    residual_head: bool = False
    # prosqa_io (ProsQA-faithful I/O, use WITH head_norm_last): embeddings = Linear(V->d)
    # on one-hot WITH bias, default init (emb RMS ~0.1, no sqrt(d) scaling — noise is LOUDER
    # than embeddings, as in prosqa/model.py); attention qkv/o get biases (their MHSA);
    # decode = Linear(d->V)+bias (std 0.02) applied DIRECTLY on the norm-last output (no
    # extra decode norm). Time module intentionally NOT changed (stays 2-layer/freq-256).
    prosqa_io: bool = False
    # tie_embeddings (GPT2-style): share the embed_tokens/lm_head weight matrix instead of
    # two independent ones. Shapes already match exactly ([vocab_size, hidden_size] for
    # both CastedEmbedding.embedding_weight and CastedLinear(hidden_size, vocab_size).weight)
    # -- ties by parameter-sharing, not by copying. Only applies to the non-prosqa_io path
    # (prosqa_io's in_proj/lm_head have different shapes, one-hot-input vs hidden-dim-input).
    tie_embeddings: bool = False
    # vocab_state: the flow lives in R^V (vocabulary space) instead of R^d. The carried state
    # z is [B, L, V]; the clean endpoint is sqrt(V) * one_hot(label) (RMS 1, like the R^d
    # embedding target); noise is noise_scale * N(0, I_V); the readout v_head maps d -> V and
    # its output IS the logits (NO lm_head / decode: softmax-CE directly on the final state).
    # The transformer still runs in R^d: z enters through a linear map R^V -> R^d, which by
    # default (vocab_state_tie) is the token-embedding matrix itself, scaled so that the
    # one-hot endpoint maps EXACTLY to the reference model's embed_scale * E[label] -- i.e. the R^d
    # flow is the image of this R^V flow under the embedding (the theoretical equivalence).
    # vocab_state_tie=false uses a separate Linear(V, d) (LeCun init, std 1/sqrt(V)).
    vocab_state: bool = False
    vocab_state_tie: bool = True
    # state_dim (0 = hidden_size, the default R^d flow): carry the flow in a SMALLER R^{d_z}
    # while the transformer keeps its width d. The clean endpoint is a separate token table
    # state_embed [V, d_z] scaled to RMS 1 (same convention as embed_tokens); z enters the
    # network through z_in = Linear(d_z, d) (LeCun init: RMS-preserving); the readout v_head
    # maps d -> d_z; decode = lm_head(norm(z1)) with lm_head [d_z -> V] as before. Isolates the
    # NOISE/STATE dimension from model capacity (noise_scale should follow 1/sqrt(d_z)).
    state_dim: int = 0
    # head_norm_last (norm-last head): the FINAL NORM is the LAST operation —
    # output = FinalNorm(h_ans [+ z if input_skip]) * (1+scale(tau)) + shift(tau), NO v_head
    # after it. The only radial freedom is final_ada's small gain, so output-scale drift is
    # structurally impossible.
    head_norm_last: bool = False
    # input_skip (ProsQA use_input_skip): add the input state z to the trunk output BEFORE
    # the final norm — the safe residual (identity path is normalized, no BPTT compounding).
    # Only used with head_norm_last.
    input_skip: bool = False
    # target_rms: RMS-normalize the hint target emb(x1) (per token) -> target RMS == 1.
    target_rms: bool = False
    # target_std_prompt: standardize the hint target with the PROMPT's statistics:
    # emb(x1) <- (emb(x1) - mu_prompt) / sigma_prompt  (per-sample scalars over the
    # prompt embedding block) -> answer targets live in the prompt's distribution.
    target_std_prompt: bool = False
    # first_norm_rms: ONLY the very first norm the sequence hits (block 0, attention-branch
    # pre-norm) becomes RMS; every other norm keeps the block flavour. Isolates the
    # entry-point DC question from the deeper norms.
    first_norm_rms: bool = False
    # rezero (requires vanilla_gpt2): restores the TWO things the vanilla ablation removed
    # besides time conditioning, at ~0 parameter cost --
    #   (a) a zero-init learnable SCALAR gate per branch  ->  x = x + g*branch(norm(x)),
    #       so the block is the identity at init (ReZero/LayerScale). For a weight-tied
    #       operator composed 12-32x this is what makes the composition trainable at all;
    #       adaLN-Zero got it for free via its zero-init gate, vanilla_gpt2 had it nowhere.
    #   (b) per-channel AFFINE on every norm (_norm here has none -- so vanilla_gpt2 had no
    #       per-channel freedom anywhere in the block, which is NOT what GPT-2 does).
    # Isolates "identity-at-init + affine" from "adaLN modulation".
    # Pair with no_decay_groups so wd does not fight the zero-init gates (p.ndim < 2).
    rezero: bool = False
    # time_add (vanilla/rezero block only): the ReZero block has NO conditioning path, so
    # time re-enters the classic pre-DiT way -- the sinusoidal-MLP embedding is ADDED to the
    # residual stream once, before the blocks:  h <- h + time_emb(t).  With no_time false
    # this makes the operator NON-autonomous (v(z,t) instead of v(z)), i.e. no longer a
    # weight-tied looped transformer. Pairs with rezero to isolate "does time help at all"
    # from "does adaLN help", which the reference configuration confounds.
    time_add: bool = False
    # inject_arch (input-injection layout): NO separate answer block.
    # The flow lives ON THE PROMPT CELLS -- the sequence is [prefix | seq_len] (97 tokens
    # instead of 178) and each cell's input is prompt_emb + z, i.e. TRM-style input
    # injection at every step. ~1.8x fewer MLP tokens, ~3.4x fewer attention pairs, same
    # parameter count. Incompatible with prefix_attn (there is no prompt/answer split).
    inject_arch: bool = False
    # adapt_w: BUDGET-PRESERVING adaptive stepping (unused by the paper). The knot grid is
    # ignored except for its start t0; instead h_k = w_k * (1 - t_k) with w clamped in
    # [adapt_w_min, adapt_w_max] < 1, so sum(h) can never overshoot the remaining budget
    # and t -> 1 geometrically (floor 0.25 over 12 steps => t_end = 1 - 0.75^12 = 0.968).
    # w is scaled by the relative-motion rule on the realized step:
    #   r = rms(z_{k+1} - z_k) / rms(z_k);  r > adapt_target -> w *= adapt_shrink, else *= adapt_grow.
    # Controller inputs are detached (no gradient through the step sizes).
    adapt_w: bool = False
    adapt_target: float = 0.15
    adapt_grow: float = 1.5
    adapt_shrink: float = 0.9
    adapt_w_min: float = 0.25
    adapt_w_max: float = 0.9
    # adapt_w_eval: replay the same controller at eval (train/eval integrator parity).
    # false -> eval falls back to the plain uniform grid.
    adapt_w_eval: bool = True
    # psi_update: EXACT reformulation of normalize_z with the state never normalized.
    # The sphere appears once, as the z-OPERAND of the blend: u_{k+1} = (1-b)·sphere(u_k) + b·f,
    # except k=0 which uses the raw quiet u_0 (matching normalize_z, whose sphere acts only
    # AFTER step 0). Model always reads the raw carrier (pre-norm LN makes that identical to
    # reading the sphered state). Equivalent to normalize_z up to float noise.
    psi_update: bool = False
    # sphere_unit_norm: ablation of _sphere's radius. Default (False) divides by RMS,
    # matching every other norm in this network (rms_norm in layers.py) -- output L2 norm
    # is sqrt(d), the natural scale of an isotropic unit-variance Gaussian in d dims, and
    # matches the scale the blend partner f (unnormalized model output) lives at. True
    # divides by the raw L2 norm instead -- output norm is exactly 1, ~sqrt(d) smaller,
    # testing whether that extra shrinkage helps or (per rms_norm-parity argument) just
    # drowns z out relative to f in the (1-w)*sphere(z) + w*f blend.
    sphere_unit_norm: bool = False
    # normalize_input (psi(z_t) conditioning): the MODEL INPUT view
    # of the answer slots is sphere-normalized (RMS 1 per token) at every call, but the
    # integrated state z stays RAW (accumulator untouched; Euler/exp update uses raw z).
    # Isolates "what the network conditions on" from "what the integrator carries".
    normalize_input: bool = False
    # final_norm_plain: readout uses a PLAIN RMS norm before v_head instead of the
    # tau-conditioned AdaLN (no (1+scale)+shift modulation) — removes the last adaptive
    # normalization from the head; decode norm unchanged.
    final_norm_plain: bool = False
    # normalize_noise: per-step NOISE-COMPONENT normalization (state itself untouched).
    # After each Euler/exp step at time t', decompose the state with the current clean
    # estimate f:  n = (z - t'·f) / (1 - t'),  re-scale the noise part to the prior's RMS
    # (n <- noise_scale · sphere(n)) and rebuild  z = (1-t')·n + t'·f.  Pins only the
    # stochastic direction; the signal part stays free. (exp mode: t' = 1 - exp(-tau').)
    normalize_noise: bool = False
    # norm_output: sphere the model OUTPUT f once inside the head (RMS-1 per
    # token) — part of the architecture, NOT a state renormalization. With the convex/exp
    # update the state then stays inside the unit ball of f's for free; kills the free
    # output-scale drift without touching z or the integrator.
    norm_output: bool = False
    # prefix-LM attention: prompt queries attend only prompt keys.
    prefix_attn: bool = False
    # post-norm blocks (TRM/BERT style): the branch reads the RAW stream,
    # and Norm+AdaLN modulation is applied AFTER each residual add -> the residual stream is
    # re-normalized every sub-layer (pins the internal scale). Note: NOT identity at init
    # (zero gates -> block output = norm(x) at step 0), unlike pre-norm adaLN-Zero.
    post_norm: bool = False
    bf16_acc32: bool = False
    forward_dtype: str = "bfloat16"


def _norm(x: torch.Tensor, kind: str, eps: float) -> torch.Tensor:
    if kind == "canonical":
        return F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)
    return rms_norm(x, variance_epsilon=eps)


class RefAttention(nn.Module):
    """Plain MHSA with RoPE (no QK-norm), SDPA, attn-prob + output dropout."""

    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.output_size = head_dim * num_heads
        self.qkv_proj = CastedLinear(hidden_size, 3 * self.output_size, bias=bias)
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=bias)
        self.dropout = dropout

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, attn_mask=None,
                kv_states=None, kv_cos_sin=None) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(B, L, 3 * self.num_heads, self.head_dim)
        q = qkv[:, :, :self.num_heads]
        if kv_states is not None:
            # ER cross+self: queries from hidden_states, keys/values from kv_states
            kvq = self.qkv_proj(kv_states).view(B, kv_states.shape[1], 3 * self.num_heads, self.head_dim)
            k = kvq[:, :, self.num_heads:2 * self.num_heads]
            v = kvq[:, :, 2 * self.num_heads:]
            if cos_sin is not None:
                q, _ = apply_rotary_pos_emb(q, q, cos_sin[0], cos_sin[1])
                k, _ = apply_rotary_pos_emb(k, k, kv_cos_sin[0], kv_cos_sin[1])
        else:
            k = qkv[:, :, self.num_heads:2 * self.num_heads]
            v = qkv[:, :, 2 * self.num_heads:]
            if cos_sin is not None:
                cos, sin = cos_sin
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (q, k, v))
        out = scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=attn_mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        out = einops.rearrange(out, 'B H S D -> B S H D').reshape(B, L, self.output_size)
        return self.o_proj(out)


class GeluMLP(nn.Module):
    """Canonical DiT pointwise MLP: Linear -> GELU(tanh) -> drop -> Linear."""

    def __init__(self, hidden_size: int, expansion: float, dropout: float):
        super().__init__()
        inter = int(hidden_size * expansion)
        self.fc1 = CastedLinear(hidden_size, inter, bias=True)
        self.fc2 = CastedLinear(inter, hidden_size, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x), approximate="tanh")))


class RefSwiGLU(nn.Module):
    """LightningDiT FFN: SwiGLU with hidden dropout."""

    def __init__(self, hidden_size: int, expansion: float, dropout: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.drop(F.silu(gate) * up))


class DiTRefBlock(nn.Module):
    def __init__(self, config: DiTRefConfig, attn_norm_override: str | None = None,
                 seq_len_total: int | None = None) -> None:
        super().__init__()
        self.kind = config.block_type
        self.attn_kind = attn_norm_override or config.block_type  # first_norm_rms hook
        self.post_norm = config.post_norm
        self.norm_eps = config.norm_eps
        self.config = config
        if config.mlp_t:
            assert seq_len_total is not None, "mlp_t needs the fixed total sequence length"
            self.self_attn = None
            self.mlp_t_layer = RefSwiGLU(seq_len_total, config.expansion, config.dropout)
        else:
            self.self_attn = RefAttention(config.hidden_size, config.head_dim or config.hidden_size // config.num_heads,
                                          config.num_heads, config.dropout, bias=config.prosqa_io)
            self.mlp_t_layer = None
        mk = config.mlp_kind or ("gelu" if self.kind == "canonical" else "swiglu")
        if mk == "gelu":
            self.mlp = GeluMLP(config.hidden_size, config.expansion, config.dropout)
        else:
            self.mlp = RefSwiGLU(config.hidden_size, config.expansion, config.dropout)
        self.attn_drop = nn.Dropout(config.dropout)
        self.mlp_drop = nn.Dropout(config.dropout)
        # adaLN-Zero
        self.vanilla = config.vanilla_gpt2
        self.rezero = config.rezero
        if self.rezero:
            assert self.vanilla, "rezero replaces adaLN; it needs the vanilla block"
            d = config.hidden_size
            self.rz_a = nn.Parameter(torch.zeros(1))   # identity at init
            self.rz_m = nn.Parameter(torch.zeros(1))
            self.aff_a_w, self.aff_a_b = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
            self.aff_m_w, self.aff_m_b = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        if not self.vanilla:
            self.adaLN = CastedLinear(config.adaln_dim or config.hidden_size, 6 * config.hidden_size, bias=True)
            with torch.no_grad():
                self.adaLN.weight.zero_()
                self.adaLN.bias.zero_()

    def _mix(self, cos_sin: CosSin, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """Token-mixing: self-attention, or (mlp_t) a SwiGLU applied across the sequence
        axis instead (transpose to [B,D,L], SwiGLU(hidden_size=L), transpose back)."""
        if self.config.mlp_t:
            return self.mlp_t_layer(x.transpose(1, 2)).transpose(1, 2)
        return self.self_attn(cos_sin=cos_sin, hidden_states=x, attn_mask=attn_mask)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, cond: torch.Tensor, attn_mask=None,
                memory=None, full_cos_sin=None) -> torch.Tensor:
        if self.vanilla:
            if self.rezero:
                x = _norm(hidden_states, self.attn_kind, self.norm_eps) * self.aff_a_w + self.aff_a_b
                hidden_states = hidden_states + self.rz_a * self.attn_drop(
                    self._mix(cos_sin, x, attn_mask))
                y = _norm(hidden_states, self.kind, self.norm_eps) * self.aff_m_w + self.aff_m_b
                return hidden_states + self.rz_m * self.mlp_drop(self.mlp(y))
            if self.post_norm:
                # TRM / Looped-Flows block (vanilla + post_norm): NO adaLN, branch on the raw
                # stream, RMSNorm AFTER each residual add --
                #   h = norm(h + attn(h));  h = norm(h + mlp(h))     (TRM layers.py verbatim)
                # time enters once at the input via time_add (their additive time MLP).
                hidden_states = _norm(hidden_states + self.attn_drop(
                    self._mix(cos_sin, hidden_states, attn_mask)), self.kind, self.norm_eps)
                return _norm(hidden_states + self.mlp_drop(self.mlp(hidden_states)),
                             self.kind, self.norm_eps)
            x = _norm(hidden_states, self.attn_kind, self.norm_eps)
            hidden_states = hidden_states + self.attn_drop(
                self._mix(cos_sin, x, attn_mask))
            return hidden_states + self.mlp_drop(
                self.mlp(_norm(hidden_states, self.kind, self.norm_eps)))
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(F.silu(cond)).unsqueeze(1).chunk(6, dim=-1)
        if self.post_norm:
            # post_norm: branch on the RAW stream; Norm+AdaLN AFTER each residual add
            hidden_states = hidden_states + gate_a * self.attn_drop(
                self._mix(cos_sin, hidden_states, attn_mask))
            hidden_states = _norm(hidden_states, self.kind, self.norm_eps) * (1 + scale_a) + shift_a
            hidden_states = hidden_states + gate_m * self.mlp_drop(self.mlp(hidden_states))
            hidden_states = _norm(hidden_states, self.kind, self.norm_eps) * (1 + scale_m) + shift_m
            return hidden_states
        x = _norm(hidden_states, self.attn_kind, self.norm_eps) * (1 + scale_a) + shift_a
        if memory is not None:
            assert not self.config.mlp_t, "mlp_t is incompatible with memory/cross-attention"
            xm = _norm(memory, self.attn_kind, self.norm_eps) * (1 + scale_a) + shift_a
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=x, attn_mask=None,
                                      kv_states=torch.cat([xm, x], dim=1), kv_cos_sin=full_cos_sin)
        else:
            attn_out = self._mix(cos_sin, x, attn_mask)
        hidden_states = hidden_states + gate_a * self.attn_drop(attn_out)
        x = _norm(hidden_states, self.kind, self.norm_eps) * (1 + scale_m) + shift_m
        hidden_states = hidden_states + gate_m * self.mlp_drop(self.mlp(x))
        return hidden_states


class RefTimestepEmbedder(nn.Module):
    """Canonical DiT: 256 sinusoidal dims of RAW t -> Linear-SiLU-Linear(d)."""

    def __init__(self, freq_dim: int, out_dim: int, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.in_proj = CastedLinear(freq_dim, out_dim, bias=True)
        self.out_proj = CastedLinear(out_dim, out_dim, bias=True)
        half = freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.freqs = nn.Buffer(freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.to(torch.float32).unsqueeze(-1) * self.freqs
        feats = torch.cat((torch.cos(args), torch.sin(args)), dim=-1).to(self.cast_to)
        return self.out_proj(F.silu(self.in_proj(feats)))


class DiTRef(nn.Module):
    """Reference-DiT velocity/x-pred flow. Layout [prefix P][prompt L][answer L], RoPE-2D."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = DiTRefConfig(**config_dict)
        cfg = self.config
        assert cfg.block_type in ("canonical", "lightning")
        self.forward_dtype = getattr(torch, cfg.forward_dtype)

        self.embed_scale = math.sqrt(cfg.hidden_size)
        if cfg.prosqa_io:
            assert cfg.head_norm_last, "prosqa_io decode expects the norm-last output (no extra decode norm)"
            self.embed_tokens = None
            self.in_proj = nn.Linear(cfg.vocab_size, cfg.hidden_size)      # one-hot -> d, bias, default init
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)      # d -> V, bias
            with torch.no_grad():
                nn.init.normal_(self.lm_head.weight, std=0.02)
                nn.init.zeros_(self.lm_head.bias)
        else:
            self.embed_tokens = CastedEmbedding(cfg.vocab_size, cfg.hidden_size, init_std=1.0 / self.embed_scale, cast_to=self.forward_dtype)
            self.lm_head = CastedLinear(cfg.hidden_size, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.embed_tokens.embedding_weight
        # state dimension of the carried flow variable z (R^d by default, R^V for vocab_state,
        # R^{state_dim} for a decoupled small state)
        assert not (cfg.vocab_state and cfg.state_dim > 0), "vocab_state and state_dim are exclusive"
        self.state_dim = cfg.vocab_size if cfg.vocab_state else (cfg.state_dim or cfg.hidden_size)
        if cfg.state_dim > 0 and cfg.state_dim != cfg.hidden_size:
            for bad in ("prosqa_io", "residual_head", "input_skip", "head_norm_last", "inject_arch",
                        "tie_embeddings", "target_std_prompt"):
                assert not getattr(cfg, bad), f"state_dim is incompatible with {bad}"
            assert cfg.carry_pool == 0, "state_dim: carry buffer is sized by hidden_size"
            assert cfg.x_pred, "state_dim readout is the x-pred endpoint"
            dz = cfg.state_dim
            self.state_embed = CastedEmbedding(cfg.vocab_size, dz, init_std=1.0 / math.sqrt(dz), cast_to=self.forward_dtype)
            self.z_in = CastedLinear(dz, cfg.hidden_size, bias=False)      # LeCun: std 1/sqrt(dz), RMS-preserving
            self.lm_head = CastedLinear(dz, cfg.vocab_size, bias=False)    # decode from the small state
        if cfg.vocab_state:
            for bad in ("prosqa_io", "residual_head", "input_skip", "head_norm_last", "inject_arch",
                        "tie_embeddings", "target_std_prompt"):
                assert not getattr(cfg, bad), f"vocab_state is incompatible with {bad}"
            assert cfg.carry_pool == 0, "vocab_state: carry buffer is sized by hidden_size"
            assert cfg.x_pred, "vocab_state readout is the x-pred endpoint = logits"
            self.lm_head = None            # no decode: the final state IS the logits
            if not cfg.vocab_state_tie:
                self.z_in = CastedLinear(cfg.vocab_size, cfg.hidden_size, bias=False)  # LeCun: std 1/sqrt(V)

        self.prefix_len = cfg.puzzle_emb_len
        if self.prefix_len > 0:
            if cfg.puzzle_sparse:
                # TRM ARC scheme: per-(task,aug) vector (zero-init), trained by SignSGD
                # outside AdamW; padded to prefix_len tokens in _prompt_embeddings
                assert cfg.puzzle_emb_ndim > 0
                n_rows = cfg.num_puzzle_identifiers
                if cfg.puzzle_id_remap:
                    import json as _json
                    remap = _json.load(open(cfg.puzzle_id_remap))
                    self.register_buffer("_pid_remap", torch.tensor(remap, dtype=torch.int32),
                                         persistent=False)
                    n_rows = int(max(remap)) + 1
                self.puzzle_emb = CastedSparseEmbedding(
                    n_rows, cfg.puzzle_emb_ndim,
                    batch_size=cfg.batch_size, init_std=0, cast_to=self.forward_dtype,
                    grad_accum=getattr(cfg, "grad_accum", 1))
            else:
                self.puzzle_emb = nn.Parameter(torch.zeros(1, self.prefix_len, cfg.hidden_size))
                nn.init.normal_(self.puzzle_emb, std=0.02)

        # RoPE: mode-dependent construction (see rope_mode docstring above).
        head_dim = cfg.head_dim or cfg.hidden_size // cfg.num_heads
        if cfg.rope_mode == "2d":
            # first half of rotation pairs by row, second by col; prompt/answer cells share coords
            assert head_dim % 4 == 0, f"2D RoPE needs head_dim % 4 == 0, got {head_dim}"
            quarter = head_dim // 4
            freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))
            cell = torch.arange(cfg.seq_len, dtype=torch.float32)
            row, col = (cell // cfg.grid_size), (cell % cfg.grid_size)
            rows = torch.cat([row, row]); cols = torch.cat([col, col])
            angles = torch.cat([rows[:, None] * freqs, cols[:, None] * freqs], dim=-1)
            emb = torch.cat((angles, angles), dim=-1)
        elif cfg.rope_mode == "1d":
            # standard flat single-axis RoPE: one sequence-position index per cell;
            # prompt/answer cells share position index, matching the 2D branch's convention
            assert head_dim % 2 == 0, f"1D RoPE needs head_dim % 2 == 0, got {head_dim}"
            half = head_dim // 2
            freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
            pos = torch.arange(cfg.seq_len, dtype=torch.float32)
            pos = torch.cat([pos, pos])
            angles = pos[:, None] * freqs
            emb = torch.cat((angles, angles), dim=-1)
        elif cfg.rope_mode == "none":
            # identity rotation: cos=1, sin=0 everywhere -- no rotary positional signal
            emb = torch.zeros(2 * cfg.seq_len, head_dim)
        else:
            raise ValueError(f"Unknown rope_mode: {cfg.rope_mode}")
        cos, sin = emb.cos(), emb.sin()
        if self.prefix_len > 0:
            cos = torch.cat([torch.ones(self.prefix_len, cos.shape[1]), cos], dim=0)
            sin = torch.cat([torch.zeros(self.prefix_len, sin.shape[1]), sin], dim=0)
        if cfg.inject_arch:
            # 97-token layout: only [prefix | cells], no duplicated answer block
            assert not cfg.prefix_attn, "inject_arch has no prompt/answer split to mask"
            cos, sin = cos[:self.prefix_len + cfg.seq_len], sin[:self.prefix_len + cfg.seq_len]
        self.rope_cos = nn.Buffer(cos, persistent=False)
        self.rope_sin = nn.Buffer(sin, persistent=False)

        _ad = cfg.adaln_dim or cfg.hidden_size   # shared time-embedding / adaLN input width
        if cfg.vanilla_gpt2:
            self.time_emb = (RefTimestepEmbedder(cfg.time_freq_dim, _ad,
                                                 cast_to=self.forward_dtype)
                             if cfg.time_add else None)
        elif cfg.learned_cond:
            self.time_emb = None
            # MUST NOT be zero-init: adaLN is zero-init, so cond=0 gives
            #   dL/d(adaLN.weight) ∝ silu(cond) = 0   and   dL/d(cond_vec) ∝ adaLN.weight = 0,
            # a permanent deadlock in which only adaLN.bias ever moves -- i.e. the modulation
            # degenerates to 6 learned d-vectors instead of the full Linear(d,6d).
            # Start exactly where the removed embedder would have put it:
            # the sinusoidal-MLP output at t=0, then let it move freely.
            with torch.no_grad():
                _te = RefTimestepEmbedder(cfg.time_freq_dim, _ad, cast_to=torch.float32)
                init = _te(torch.zeros(1, device=next(_te.parameters()).device)).float().reshape(1, -1)
            self.cond_vec = nn.Parameter(init.clone())
        else:
            self.time_emb = RefTimestepEmbedder(cfg.time_freq_dim, _ad, cast_to=self.forward_dtype)
        if cfg.mlp_t:
            assert not cfg.dyn_crop, "mlp_t needs a fixed sequence length, incompatible with dyn_crop"
        self.blocks = nn.ModuleList([
            DiTRefBlock(cfg, attn_norm_override=("lightning" if cfg.first_norm_rms and i == 0 else None),
                        seq_len_total=self.rope_cos.shape[0])
            for i in range(cfg.n_layers)])

        self.final_adaLN = (None if cfg.vanilla_gpt2 else
                            CastedLinear(_ad, 2 * cfg.hidden_size, bias=True))
        if cfg.rezero:
            self.aff_out_w = nn.Parameter(torch.ones(cfg.hidden_size))
            self.aff_out_b = nn.Parameter(torch.zeros(cfg.hidden_size))
        # head_norm_last: NO v_head — the normed+modulated hidden IS the output (ProsQA head)
        self.v_head = None if cfg.head_norm_last else CastedLinear(cfg.hidden_size, self.state_dim, bias=False)
        assert not (cfg.head_norm_last and cfg.norm_output), "head_norm_last subsumes norm_output"
        assert not (cfg.input_skip and not cfg.head_norm_last), "input_skip is part of the norm-last head"
        with torch.no_grad():
            if self.final_adaLN is not None:
                self.final_adaLN.weight.zero_()
                self.final_adaLN.bias.zero_()
            if self.v_head is not None and not cfg.norm_output:
                # norm_output: keep v_head at standard init — zero-init would put the sphere
                # at its 0-singularity (gradient blow-up).
                self.v_head.weight.zero_()

        # prefix-LM mask: keep[q, k] = (q is not prompt) OR (k is prompt)
        if cfg.prefix_attn:
            total_len = self.prefix_len + 2 * cfg.seq_len
            cm = torch.zeros(total_len, dtype=torch.bool)
            cm[self.prefix_len: self.prefix_len + cfg.seq_len] = True
            keep = (~cm).unsqueeze(1) | cm.unsqueeze(0)
            self.attn_mask = nn.Buffer(keep.view(1, 1, total_len, total_len), persistent=False)
        else:
            self.attn_mask = None

        # rand_t0 hint cap, scheduled from outside (pretrain.py) per train iter
        self.t0_max = float(cfg.t0_start) if cfg.rand_t0 else 0.0
        # curriculum stage, updated once per epoch by pretrain.py::update_curriculum_schedule.
        # Starts at curr_stages (= "everything", i.e. curriculum inactive) so a model built
        # with curriculum="none" (all paper configurations) applies no masking.
        self.curr_stage = int(cfg.curr_stages)
        if cfg.residual_head:
            assert cfg.x_pred, "residual_head parameterizes the clean-answer estimate (x_pred)"

    # ---- flow engine ----
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        if self.config.prosqa_io:
            oh = F.one_hot(ids.to(torch.long), self.config.vocab_size).to(self.forward_dtype)
            return self.in_proj(oh)
        return self.embed_scale * self.embed_tokens(ids.to(torch.int32))

    # ---- vocab_state helpers (identity / unchanged path when vocab_state is off) ----
    def _x1(self, labels: torch.Tensor) -> torch.Tensor:
        """Clean endpoint of the flow for the hinted start: R^d embedding, or for vocab_state
        sqrt(V) * one_hot(label) (RMS 1 over the V state coordinates)."""
        if self.config.vocab_state:
            V = self.config.vocab_size
            return math.sqrt(V) * F.one_hot(labels.to(torch.long), V).to(self.forward_dtype)
        if self.state_dim != self.config.hidden_size:
            return math.sqrt(self.state_dim) * self.state_embed(labels.to(torch.int32))   # RMS 1
        return self._embed(labels)

    def _z_to_h(self, z: torch.Tensor) -> torch.Tensor:
        """State -> transformer input. vocab_state: linear R^V -> R^d, tied to the token
        embedding (scaled so sqrt(V)*one_hot(k) -> embed_scale * E[k]) or a free Linear."""
        if not self.config.vocab_state:
            return self.z_in(z) if self.state_dim != self.config.hidden_size else z
        if self.config.vocab_state_tie:
            E = self.embed_tokens.embedding_weight.to(z.dtype)          # [V, d]
            return (self.embed_scale / math.sqrt(self.config.vocab_size)) * (z @ E)
        return self.z_in(z)

    def _prompt_embeddings(self, inputs: torch.Tensor, puzzle_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[prefix?, prompt] embedding; the prefix rides inside prompt_emb from here on."""
        pe = self._embed(inputs)
        if self.prefix_len > 0:
            if self.config.puzzle_sparse:
                if self.config.puzzle_id_remap:
                    puzzle_ids = self._pid_remap.to(puzzle_ids.device)[puzzle_ids.to(torch.long)]
                p = self.puzzle_emb(puzzle_ids.to(torch.int32))
                pad = self.prefix_len * self.config.hidden_size - p.shape[-1]
                if pad > 0:
                    p = F.pad(p, (0, pad))
                p = p.view(-1, self.prefix_len, self.config.hidden_size).to(pe.dtype)
            else:
                p = self.puzzle_emb.to(pe.dtype).expand(pe.shape[0], -1, -1)
            pe = torch.cat([p, pe], dim=1)
        return pe

    def _velocity(self, cos_sin: Optional[CosSin], prompt_emb: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.vanilla_gpt2:
            h = torch.cat([prompt_emb, self._z_to_h(z)], dim=1)
            if self.config.time_add:
                te = self.time_emb(torch.zeros_like(t) if self.config.no_time else t)
                h = h + te.unsqueeze(1).to(h.dtype)
            for block in self.blocks:
                h = block(cos_sin=cos_sin, hidden_states=h, cond=None, attn_mask=self.attn_mask)
            h = h[:, self.prefix_len + (getattr(self, "_cur_seq", None) or self.config.seq_len):]
            hn = _norm(h, self.config.block_type, self.config.norm_eps)
            if self.config.rezero:  # affine on the readout norm too (adaLN had final_adaLN here)
                hn = hn * self.aff_out_w + self.aff_out_b
            out = self.v_head(hn)
            return z + out if self.config.residual_head else out
        if self.config.learned_cond:
            cond = self.cond_vec.to(z.dtype).expand(z.shape[0], -1)
        else:
            cond = self.time_emb(torch.zeros_like(t) if self.config.no_time else t)
        P = self.prefix_len
        if self.config.inject_arch:
            # in-place flow: prompt cells carry the state, z is injected every step
            h = torch.cat([prompt_emb[:, :P], prompt_emb[:, P:] + z], dim=1)
        else:
            h = torch.cat([prompt_emb, self._z_to_h(z)], dim=1)
        for block in self.blocks:
            h = block(cos_sin=cos_sin, hidden_states=h, cond=cond, attn_mask=self.attn_mask)
        h = h[:, P:] if self.config.inject_arch else \
            h[:, P + (getattr(self, "_cur_seq", None) or self.config.seq_len):]
        if self.config.final_norm_plain:
            # plain RMS before the head — no AdaLN modulation at the readout
            return self.v_head(rms_norm(h, variance_epsilon=self.config.norm_eps))
        shift, scale = self.final_adaLN(F.silu(cond)).unsqueeze(1).chunk(2, dim=-1)
        if self.config.head_norm_last:
            # norm-last head: norm is the LAST op; nothing free after it
            if self.config.input_skip:
                h = h + z
            return _norm(h, self.config.block_type, self.config.norm_eps) * (1 + scale) + shift
        h = _norm(h, self.config.block_type, self.config.norm_eps) * (1 + scale) + shift
        out = self.v_head(h)
        if self.config.residual_head:
            out = z + out  # D = z + r (identity at init, W_out zero-init)
        if self.config.norm_output:
            out = self._sphere(out)
        return out

    def _step(self, cos_sin, prompt_emb, z, times, k: int, N: int) -> torch.Tensor:
        t = times[:, k]
        z_in = self._sphere(z) if self.config.normalize_input else z  # psi(z): input view only
        if self.config.bf16_acc32:
            with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16):
                f = self._velocity(cos_sin, prompt_emb, z_in, t)
            f = f.to(z.dtype)
        else:
            f = self._velocity(cos_sin, prompt_emb, z_in, t)
        if not self.config.x_pred:
            if self.config.exp_update:
                # tau-mode: `times` holds TAU, so the plain difference is h, not dt.
                # Convert to the t-step:  dt = (1-t_k)(1-e^-h) = e^-tau_k (1-e^-h).
                # This makes the v-pred Euler step ALGEBRAICALLY IDENTICAL to the x-pred
                # convex blend toward the implied endpoint x1 = z + (1-t)v:
                #   z + dt*v = z + w(1-t_k)v = z + w(x1 - z) = (1-w)z + w*x1,  w = 1-e^-h
                # so v-pred gets the same constant-fractional-progress schedule, with the
                # network conditioned on raw tau exactly as the x-pred models are.
                h = (times[:, k + 1] - times[:, k]).to(self.forward_dtype)
                dt = torch.exp(-times[:, k].to(self.forward_dtype)) * (-torch.expm1(-h))
            else:
                dt = (times[:, k + 1] - times[:, k]).to(self.forward_dtype)
            return z + dt.view(-1, 1, 1) * f
        if k == N - 1:
            if getattr(self, "_want_pre_readout", False):
                self._pre_readout = z.detach()  # cache the state fed to the readout
            return f  # x-pred readout: loss/decode directly on the final model call
        if self.config.psi_update and k > 0:
            z = self._sphere(z)  # sphere the blend operand only (k=0 keeps the raw quiet start)
        if self.config.exp_update:
            # exact integrator of dz/dtau = f - z over h = tau_{k+1} - tau_k (convex, bounded)
            a = torch.exp(-(times[:, k + 1] - times[:, k])).view(-1, 1, 1).to(z.dtype)
            z_next = a * z + (1.0 - a) * f
        elif self.config.stable_w:
            # w = dt/(1-t_k) via exp of log-time deltas: same knots as plain t-Euler,
            # but bounded in (0,1] (equals the exp update expressed in normal time)
            lt = torch.log1p(-times[:, k].clamp(max=1 - 1e-7))
            ltn = torch.log1p(-times[:, k + 1].clamp(max=1 - 1e-7))
            w = (-torch.expm1(ltn - lt)).view(-1, 1, 1).to(z.dtype)
            z_next = (1.0 - w) * z + w * f
        else:
            dt = (times[:, k + 1] - times[:, k]).to(self.forward_dtype)
            denom = (1.0 - t).clamp(min=self.config.t_clamp_eps).to(self.forward_dtype)
            z_next = z + dt.view(-1, 1, 1) * (f - z) / denom.view(-1, 1, 1)
        if self.config.normalize_noise:
            tn = times[:, k + 1]
            if self.config.exp_update:
                tn = 1.0 - torch.exp(-tn)
            tv = tn.view(-1, 1, 1).to(z_next.dtype)
            den = (1.0 - tv).clamp(min=self.config.t_clamp_eps)
            n = (z_next - tv * f) / den
            n = self.config.noise_scale * self._sphere(n)   # noise part back to prior RMS
            z_next = (1.0 - tv) * n + tv * f
        return z_next

    def apply_crop(self, R: int, C: int, device) -> torch.Tensor:
        """Eager-side (train-loop) crop: set rope/length state, return the flat index.
        Keeps every sync and shape decision OUTSIDE the compiled forward."""
        cfg = self.config
        g = cfg.grid_size
        P, S = self.prefix_len, cfg.seq_len
        if R * C >= g * g:
            self._crop_idx = None
            self._cur_seq = cfg.seq_len
            self._cs = None
            return None
        idx = (torch.arange(R, device=device).unsqueeze(1) * g
               + torch.arange(C, device=device)).flatten()
        self._crop_idx = None   # loss head must NOT re-crop: batch arrives pre-cropped
        self._cur_seq = R * C
        if cfg.inject_arch:
            self._cs = (torch.cat([self.rope_cos[:P], self.rope_cos[P + idx]]),
                        torch.cat([self.rope_sin[:P], self.rope_sin[P + idx]]))
        else:
            self._cs = (torch.cat([self.rope_cos[:P], self.rope_cos[P + idx], self.rope_cos[P + S + idx]]),
                        torch.cat([self.rope_sin[:P], self.rope_sin[P + idx], self.rope_sin[P + S + idx]]))
        return idx

    def _set_crop(self, batch: Dict[str, torch.Tensor]) -> None:
        cfg = self.config
        g = cfg.grid_size
        if cfg.dyn_crop and batch["inputs"].shape[1] != cfg.seq_len:
            return  # pre-cropped by the train loop via apply_crop: keep its state
        self._crop_idx = None
        self._cur_seq = cfg.seq_len
        self._cs = None
        # EVAL NEVER CROPS: eval batches are unsorted (no savings) and per-rank divergent
        # crop shapes force eval-graph recompiles that stall DDP collectives past the
        # NCCL watchdog. One stable full-length eval graph.
        if not cfg.dyn_crop or "labels" not in batch or not self.training:
            return
        m = ((batch["inputs"] > 0) | (batch["labels"] > 0)).view(batch["inputs"].shape[0], g, g)
        rows, cols = m.any(2).any(0), m.any(1).any(0)
        R = int(rows.nonzero().max().item()) + 1 if bool(rows.any()) else 1
        C = int(cols.nonzero().max().item()) + 1 if bool(cols.any()) else 1
        R, C = min(g, -(-R // 6) * 6), min(g, -(-C // 6) * 6)  # round up: few compile shapes
        if R * C >= g * g:
            return
        idx = (torch.arange(R, device=m.device).unsqueeze(1) * g
               + torch.arange(C, device=m.device)).flatten()
        P, S = self.prefix_len, cfg.seq_len
        self._crop_idx = idx
        self._cur_seq = R * C
        if cfg.inject_arch:
            self._cs = (torch.cat([self.rope_cos[:P], self.rope_cos[P + idx]]),
                        torch.cat([self.rope_sin[:P], self.rope_sin[P + idx]]))
        else:
            self._cs = (torch.cat([self.rope_cos[:P], self.rope_cos[P + idx], self.rope_cos[P + S + idx]]),
                        torch.cat([self.rope_sin[:P], self.rope_sin[P + idx], self.rope_sin[P + S + idx]]))

    def _call_f(self, prompt_emb, z):
        t_dummy = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        if self.config.bf16_acc32 and z.device.type == "cuda":
            with torch.autocast(z.device.type, dtype=torch.bfloat16):
                f = self._velocity((self.rope_cos, self.rope_sin) if getattr(self, "_cs", None) is None else self._cs,
                                   prompt_emb, z, t_dummy)
            return f.float()
        return self._velocity((self.rope_cos, self.rope_sin) if getattr(self, "_cs", None) is None else self._cs,
                              prompt_emb, z, t_dummy)

    def _sphere(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.sphere_unit_norm:
            norm = z.float().pow(2).sum(-1, keepdim=True).clamp(min=1e-12).sqrt().to(z.dtype)
            return z / norm
        rms = z.float().pow(2).mean(-1, keepdim=True).clamp(min=1e-12).sqrt().to(z.dtype)
        return z / rms

    def _init_noise(self, batch_size: int, device) -> torch.Tensor:
        # RAW noise_scale * N(0, I), NOT sphere-projected at init
        # (normalize_z re-spheres only after each Euler step).
        L = getattr(self, "_cur_seq", None) or self.config.seq_len
        return self.config.noise_scale * torch.randn(
            batch_size, L, self.state_dim, dtype=self.forward_dtype, device=device)

    def _rollout(self, prompt_emb, z, times, grad_from: int) -> torch.Tensor:
        cos_sin = getattr(self, "_cs", None) or (self.rope_cos, self.rope_sin)
        N = times.shape[-1] - 1
        cfg = self.config
        gd = cfg.grad_decay
        ad = cfg.adapt_w and (self.training or cfg.adapt_w_eval)
        if ad:
            # keep only the grid start t0; every later knot is written by the controller
            times = times.clone()
            w = torch.full_like(times[:, 0], 1.0 / N)
        for k in range(N):
            if ad:
                times[:, k + 1] = times[:, k] + w * (1.0 - times[:, k])
                z_prev = z.detach()
            if k < grad_from:
                with torch.no_grad():
                    z = self._step(cos_sin, prompt_emb, z, times, k, N)
                    if cfg.normalize_z:
                        z = self._sphere(z)
            else:
                if gd < 1.0:
                    # leap: value unchanged, gradient through the carried state
                    # attenuated by gd each tracked step (gd^k over the chain)
                    z = gd * z + (1.0 - gd) * z.detach()
                z = self._step(cos_sin, prompt_emb, z, times, k, N)
                if cfg.normalize_z:
                    z = self._sphere(z)
            if ad and k < N - 1:
                with torch.no_grad():
                    r = ((z.detach() - z_prev).float().pow(2).mean((1, 2)).sqrt()
                         / z_prev.float().pow(2).mean((1, 2)).sqrt().clamp_min(1e-6))
                    fac = torch.where(r > cfg.adapt_target,
                                      torch.full_like(w, cfg.adapt_shrink),
                                      torch.full_like(w, cfg.adapt_grow))
                    w = (w * fac).clamp(cfg.adapt_w_min, cfg.adapt_w_max)
        return z

    def _grid_hi(self) -> float:
        return self.config.tau_max if self.config.exp_update else self.config.t_max

    def _sample_time_grid(self, t0col: torch.Tensor, N: int) -> torch.Tensor:
        """[B, N+1] sorted grid [t0, interior knots in (t0, hi), hi]; t0col: [B, 1].
        hi = t_max (t-mode) or tau_max (exp_update tau-mode)."""
        cfg = self.config
        batch_size, device = t0col.shape[0], t0col.device
        hi_val = self._grid_hi()
        hi = torch.full_like(t0col, hi_val)
        if cfg.t_end_p > 0 and self.training:
            # flow-maps-style boundary mixture on the TERMINAL end: with prob p the grid
            # ends at t_end_val (e.g. 0.99) instead of 1 -- trains near-terminal
            # refinement as its own regime (t-mode only; interior sampling is capped by
            # t_int_cap so ordinary knots never enter this band)
            assert not cfg.exp_update and cfg.grid_geom_G == 0, "t_end mixture: plain t-grid only"
            mix = torch.rand_like(t0col) < cfg.t_end_p
            hi = torch.where(mix, torch.full_like(hi, cfg.t_end_val), hi)
        if N <= 1:
            return torch.cat((t0col, hi), dim=-1)
        mode = cfg.time_sampling
        if mode == "uniform":
            u01 = torch.linspace(0.0, 1.0, N + 1, device=device)[1:-1].expand(batch_size, N - 1)
        elif mode == "random_sorted":
            u01 = torch.rand(batch_size, N - 1, device=device).sort(dim=-1).values
        elif mode == "stratified":
            offsets = torch.arange(N - 1, dtype=torch.float32, device=device)
            u01 = (offsets + torch.rand(batch_size, N - 1, device=device)) / (N - 1)
        else:
            raise ValueError(f"Unknown time_sampling: {mode}")
        if cfg.grid_geom_G > 0:
            # knots uniform in tau = -log(1-t): tau0 -> G, mapped back to t (endpoint stays hi)
            tau0 = -torch.log1p(-t0col.clamp(max=1 - 1e-6))
            tau_i = tau0 + (cfg.grid_geom_G - tau0) * u01
            interior = 1.0 - torch.exp(-tau_i)
            return torch.cat((t0col, interior, hi), dim=-1)
        eff_hi = torch.minimum(hi, torch.full_like(hi, cfg.t_int_cap))
        if cfg.exp_update and cfg.tau_uniform_t:
            # same [t0, t(eff_hi)] range, knots equally spaced in t rather than in tau
            t_lo = 1.0 - torch.exp(-t0col)
            t_hi = 1.0 - torch.exp(-eff_hi)
            tt = t_lo + (t_hi - t_lo) * u01
            interior = -torch.log1p(-tt.clamp(max=1 - 1e-7))
        else:
            interior = t0col + (eff_hi - t0col) * u01
        return torch.cat((t0col, interior, hi), dim=-1)

    def _decode(self, z1: torch.Tensor) -> torch.Tensor:
        if self.config.vocab_state:
            return z1  # the final state in R^V IS the logits: softmax/stablemax-CE applied directly
        if self.config.prosqa_io:
            return self.lm_head(z1)  # z1 is the norm-last output: logits directly, no extra norm
        return self.lm_head(_norm(z1, self.config.block_type, self.config.norm_eps))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.config
        self._set_crop(batch)
        inputs = batch["inputs"] if self._crop_idx is None else batch["inputs"][:, self._crop_idx]
        prompt_emb = self._prompt_embeddings(inputs, batch.get("puzzle_identifiers"))
        batch_size = prompt_emb.shape[0]
        device = prompt_emb.device

        z0 = self._init_noise(batch_size, device)
        lo = 0.0 if cfg.exp_update else cfg.t_min
        if cfg.rand_t0:
            # hint strength t0 ALWAYS sampled/scheduled in t-space (same as the reference runs);
            # exp mode only maps the grid start to tau0 = -ln(1-t0).
            if cfg.t0_geom_G > 0:
                u = torch.rand(batch_size, 1, device=device)
                t0col = (1.0 - torch.exp(-cfg.t0_geom_G * u)).clamp(max=1 - 1e-4)
            else:
                t0col = lo + (self.t0_max - lo) * torch.rand(batch_size, 1, device=device)
            if cfg.t0_zero_p > 0:
                t0col = torch.where(torch.rand(batch_size, 1, device=device) < cfg.t0_zero_p,
                                    torch.zeros_like(t0col), t0col)
            lab = batch["labels"] if self._crop_idx is None else batch["labels"][:, self._crop_idx]
            x1_emb = self._x1(lab.clamp_min(0))
            if cfg.target_rms:
                x1_emb = self._sphere(x1_emb)
            elif cfg.target_std_prompt:
                mu = prompt_emb.mean(dim=(1, 2), keepdim=True)
                sd = prompt_emb.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                x1_emb = (x1_emb - mu) / sd
            t0v = t0col.view(batch_size, 1, 1).to(z0.dtype)
            z0 = (1.0 - t0v) * z0 + t0v * x1_emb
            if cfg.exp_update:
                t0col = -torch.log1p(-t0col)
        else:
            t0col = torch.full((batch_size, 1), lo, device=device)
        N_roll = cfg.train_N
        if cfg.rand_N_max > 0 and self.training:
            N_roll = int(torch.randint(cfg.rand_N_min, cfg.rand_N_max + 1, (1,)).item())
        times = self._sample_time_grid(t0col, N_roll)
        self._aux_loss = None
        if cfg.ecld_lambda > 0 and self.training:
            assert cfg.no_time, "ECLD consistency implemented for the time-blind operator"
            zc = self._sphere(z0) if cfg.psi_update else z0
            f_s = self._call_f(prompt_emb, zc)
            # their (s,t) ~ logit-normal(-0.4, 1), s = min -> jump fraction gamma = (t-s)/(1-s)
            u = torch.sigmoid(-0.4 + torch.randn(batch_size, 2, device=device))
            s_t, t_t = u.min(dim=1).values, u.max(dim=1).values
            gamma = ((t_t - s_t) / (1.0 - s_t).clamp(min=0.05)).view(-1, 1, 1).to(z0.dtype)
            with torch.no_grad():
                x_jump = zc + gamma * (f_s - zc)
                x_jump = self._sphere(x_jump) if cfg.psi_update else x_jump
                teacher = F.softmax(self._decode(self._call_f(prompt_emb, x_jump)).float(), dim=-1)
            student_logp = F.log_softmax(self._decode(f_s).float(), dim=-1)
            self._aux_loss = -cfg.ecld_lambda * (teacher * student_logp).sum(-1).mean()
        if cfg.carry_pool > 0 and self.training:
            assert cfg.no_time, "carry training requires the time-blind operator"
            idx = batch["ex_idx"].to("cpu", dtype=torch.long)
            if not hasattr(self, "_carry_buf") or self._carry_buf is None:
                # CPU-pinned buffer to keep GPU memory free (not used by the paper configurations)
                self._carry_buf = torch.zeros(cfg.carry_pool, cfg.seq_len, cfg.hidden_size,
                                              dtype=torch.bfloat16,
                                              pin_memory=torch.cuda.is_available())
                self._carry_seen = torch.zeros(cfg.carry_pool, dtype=torch.bool)
            restart = (torch.rand(batch_size, device=device) < cfg.carry_fresh_p) \
                      | (~self._carry_seen[idx]).to(device)
            cached = self._carry_buf[idx].to(device=device, dtype=z0.dtype, non_blocking=True)
            z0 = torch.where(restart.view(-1, 1, 1), z0, cached)
            times = times[:, -(cfg.n_back + 1):]
            self._want_pre_readout = True
            z1 = self._rollout(prompt_emb, z0, times, grad_from=0)
            self._carry_buf[idx] = self._pre_readout.to("cpu", torch.bfloat16)
            self._carry_seen[idx] = True
            self._want_pre_readout = False
            return self._decode(z1)
        z1 = self._rollout(prompt_emb, z0, times, grad_from=max(N_roll - cfg.n_back, 0))
        return self._decode(z1)

    def forward_eval(self, batch: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        cfg = self.config
        self._set_crop(batch)
        inputs = batch["inputs"] if self._crop_idx is None else batch["inputs"][:, self._crop_idx]
        prompt_emb = self._prompt_embeddings(inputs, batch.get("puzzle_identifiers"))
        z0 = self._init_noise(prompt_emb.shape[0], prompt_emb.device)
        lo = 0.0 if cfg.exp_update else cfg.t_min
        out = {}
        for N in cfg.eval_Ns:
            times = torch.linspace(lo, self._grid_hi(), N + 1, device=prompt_emb.device).expand(prompt_emb.shape[0], N + 1)
            out[N] = self._decode(self._rollout(prompt_emb, z0, times, grad_from=N))
        return out
