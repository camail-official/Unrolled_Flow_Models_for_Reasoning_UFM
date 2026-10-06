"""Multiple rollouts per puzzle: Pass@K (a correct answer among K rollouts) and the margin
chooser of the paper, which picks ONE of the K rollouts per puzzle: the one with the largest
mean, over positions, of the gap between the largest and second-largest logit (raw logits).

  python select_rollouts.py checkpoints/<Project>/<run> data/<dataset>/test --set test --N 128 --K 100

Prints Pass@1 (first rollout), Pass@K for K in {1,2,5,10,25,50,100} (any rollout correct), and
the accuracy of the rollout selected by the margin, all on the same rollouts.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load, make_grid, DEV  # noqa: E402


@torch.inference_mode()
def rollouts(m, x, y, times, K, seed):
    """x, y: [B, L]. Returns exact [B, K] (bool) and margin [B, K]."""
    B = x.shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    xr = x.repeat_interleave(K, 0)
    pe = m._prompt_embeddings(xr)
    z = (m.config.noise_scale * torch.randn(B * K, m.config.seq_len, m.state_dim, generator=g,
                                            device=DEV, dtype=m.forward_dtype)).float()
    z = m._rollout(pe, z, times.expand(B * K, -1), grad_from=times.shape[-1])
    logits = m._decode(z).float()
    pred = logits.argmax(-1)
    exact = (pred == y.repeat_interleave(K, 0)).all(-1).view(B, K)
    top2 = logits.topk(2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).mean(-1).view(B, K)
    return exact, margin


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir"); ap.add_argument("data_dir")
    ap.add_argument("--set", default="test"); ap.add_argument("--ckpt-file", default="best")
    ap.add_argument("--N", type=int, default=128); ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--grid", default="uniform", choices=["uniform", "geom"]); ap.add_argument("--G", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=0, help="puzzles per batch (x K rollouts)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    data = args.data_dir.rstrip("/")
    meta = json.load(open(f"{data}/dataset.json"))
    seq, vocab = meta["seq_len"], meta["vocab_size"]
    B = args.batch or max(1, (8 if seq > 200 else 64) * 100 // args.K)
    m = load(args.ckpt_dir, B * args.K, seq, vocab, args.ckpt_file)
    inp = np.load(f"{data}/{args.set}__inputs.npy").astype(np.int64)
    lab = np.load(f"{data}/{args.set}__labels.npy").astype(np.int64)
    if args.limit:
        inp, lab = inp[:args.limit], lab[:args.limit]
    times = make_grid(args.grid, args.N, args.G)
    print(f"ckpt {args.ckpt_dir}/{args.ckpt_file}  set {args.set} ({len(inp)} puzzles)  N={args.N} grid={args.grid}  K={args.K}", flush=True)
    E, M = [], []
    for s in range(0, len(inp), B):
        x = torch.from_numpy(inp[s:s + B]).to(DEV); y = torch.from_numpy(lab[s:s + B]).to(DEV)
        e, mg = rollouts(m, x, y, times, args.K, args.seed + s)
        E.append(e.cpu()); M.append(mg.cpu())
        if (s // B) % 10 == 0:
            print(f"  {s + len(x)}/{len(inp)}", flush=True)
    E, M = torch.cat(E), torch.cat(M)          # [n, K]
    n = E.shape[0]
    print(f"Pass@1 (first rollout): {100 * E[:, 0].float().mean():.2f}")
    for k in (1, 2, 5, 10, 25, 50, 100):
        if k <= args.K:
            print(f"Pass@{k:<3d} (any of the first {k} correct): {100 * E[:, :k].any(-1).float().mean():.2f}")
    pick = M.argmax(-1)
    print(f"margin chooser (argmax margin over {args.K} rollouts): {100 * E[torch.arange(n), pick].float().mean():.2f}")
