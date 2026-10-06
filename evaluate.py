"""Final evaluation: exact-match accuracy of a trained checkpoint on a held-out set, averaged
over independent noise seeds (each seed = one full pass, i.e. one rollout per puzzle, K=1).

  python evaluate.py checkpoints/<Project>/<run> data/<dataset>/test --set test --N 128 --seeds 5 --grid uniform
  python evaluate.py checkpoints/<Project>/<run> data/<dataset>/test --set test --N 128 --seeds 5 --grid geom --G 8

--set: which set of <data>/test to score (test / val, or all for Maze). --ckpt-file: "best" (default,
the checkpoint selected during training) or another file saved in the run directory.
Grids: uniform knots on [0,1], or geometric t_j = 1 - exp(-jG/N) with the last knot at 1.
Prints per-seed accuracies, mean +/- std, and the 2-sigma binomial band for the set size.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dit_ref import DiTRef  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(ckpt_dir, batch, seq, vocab, ckpt_file="best"):
    cfg = yaml.safe_load(open(f"{ckpt_dir}/all_config.yaml"))
    a = dict(cfg["arch"]); a.pop("name"); a.pop("loss")
    m = DiTRef(dict(batch_size=batch, seq_len=seq, vocab_size=vocab, num_puzzle_identifiers=1, **a))
    sd = torch.load(f"{ckpt_dir}/{ckpt_file}", map_location="cpu")
    # keys are saved from torch.compile(FlowLossHead(DiTRef)): strip both wrappers; strict so a
    # config/checkpoint mismatch fails loudly instead of scoring a randomly initialised model
    m.load_state_dict({k.replace("_orig_mod.", "").replace("model.", "", 1): v for k, v in sd.items()}, strict=True)
    m.eval()
    return m.to(DEV)


def make_grid(kind, N, G):
    if kind == "uniform":
        return torch.linspace(0.0, 1.0, N + 1, device=DEV)
    t = 1.0 - torch.exp(-torch.linspace(0.0, G, N + 1, device=DEV))
    t[-1] = 1.0
    return t


@torch.inference_mode()
def run_seed(m, inp, lab, times, seed, B):
    ok = 0
    g = torch.Generator(device=DEV).manual_seed(seed)
    for s in range(0, len(inp), B):
        x = torch.from_numpy(inp[s:s + B]).to(DEV)
        y = torch.from_numpy(lab[s:s + B]).to(DEV)
        pe = m._prompt_embeddings(x)
        z = (m.config.noise_scale * torch.randn(x.shape[0], m.config.seq_len, m.state_dim,
                                                generator=g, device=DEV, dtype=m.forward_dtype)).float()
        z = m._rollout(pe, z, times.expand(x.shape[0], -1), grad_from=times.shape[-1])
        ok += (m._decode(z).argmax(-1) == y).all(-1).sum().item()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir"); ap.add_argument("data_dir")
    ap.add_argument("--set", default="test"); ap.add_argument("--ckpt-file", default="best")
    ap.add_argument("--N", type=int, default=128); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--grid", default="uniform", choices=["uniform", "geom"]); ap.add_argument("--G", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=0)
    args = ap.parse_args()
    data = args.data_dir.rstrip("/")
    meta = json.load(open(f"{data}/dataset.json"))
    seq, vocab = meta["seq_len"], meta["vocab_size"]
    B = args.batch or (64 if seq > 200 else 512)
    m = load(args.ckpt_dir, B, seq, vocab, args.ckpt_file)
    inp = np.load(f"{data}/{args.set}__inputs.npy").astype(np.int64)
    lab = np.load(f"{data}/{args.set}__labels.npy").astype(np.int64)
    times = make_grid(args.grid, args.N, args.G)
    print(f"ckpt {args.ckpt_dir}/{args.ckpt_file}  set {args.set} ({len(inp)} puzzles)  N={args.N} grid={args.grid}{'' if args.grid == 'uniform' else f' G={args.G:g}'}  seeds={args.seeds}", flush=True)
    accs = []
    for seed in range(args.seeds):
        ok = run_seed(m, inp, lab, times, seed, B)
        accs.append(ok / len(inp))
        print(f"  seed {seed}: {ok}/{len(inp)} = {100 * accs[-1]:.2f}%", flush=True)
    mu = float(np.mean(accs))
    print(f"exact match: {100 * mu:.2f} +/- {100 * float(np.std(accs)):.2f} (std over {args.seeds} seeds); "
          f"2-sigma sampling band at n={len(inp)}: +/- {200 * np.sqrt(mu * (1 - mu) / len(inp)):.2f} pp")
