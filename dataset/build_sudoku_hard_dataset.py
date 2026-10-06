"""Build the Sudoku-Hard dataset of the S-FLM paper in this repository's format.

Reproduces exactly what the S-FLM training code builds (sudoku_generator.generate_sudoku_dataset,
adapted from Ali Alp's generator): their generator with data seed 42, 48,000 training +
2,000 validation puzzles, same deterministic shuffle and split. Then:
  train/  set "all"  = their 48,000 training puzzles
  test/   set "test" = their 2,000 validation puzzles (the set their models are scored on)
          set "val"  = 2,000 EXTRA puzzles (seed 43, deduplicated against train and test),
                       used only for checkpoint selection (config eval_sets: [val])
Encoding: uint8, value = digit + 1 (blank 0 -> 1), vocab 11, seq_len 81.

Usage: python dataset/build_sudoku_hard_dataset.py --difficulty hard --output-dir data/sudoku-hard [--workers 8]
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sudoku_generator import DIFFICULTY_TO_CLUES, _generate_raw_grids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--difficulty", default="hard", choices=list(DIFFICULTY_TO_CLUES))
ap.add_argument("--output-dir", default="data/sudoku-hard")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

NUM_TRAIN, NUM_VALID, SEED = 48000, 2000, 42          # the S-FLM data config
NUM_OURVAL, SEED_OURVAL = 2000, 43
OUT = args.output_dir
clues = DIFFICULTY_TO_CLUES[args.difficulty]

# --- their train/validation, bit-identical to generate_sudoku_dataset() ---
puzzles, solutions = _generate_raw_grids(NUM_TRAIN + NUM_VALID, clues, SEED, args.workers)
rng = random.Random(SEED)
idx = list(range(NUM_TRAIN + NUM_VALID))
rng.shuffle(idx)
puzzles = [puzzles[i] for i in idx]
solutions = [solutions[i] for i in idx]
tr_p, tr_s = puzzles[:NUM_TRAIN], solutions[:NUM_TRAIN]
te_p, te_s = puzzles[NUM_TRAIN:], solutions[NUM_TRAIN:]

# --- extra validation set (different seed), deduplicated against their puzzles ---
seen = {tuple(map(tuple, p)) for p in puzzles}
extra_p, extra_s = [], []
seed = SEED_OURVAL * 1_000_000
while len(extra_p) < NUM_OURVAL:
    p, s = _generate_raw_grids(NUM_OURVAL - len(extra_p) + 64, clues, seed, args.workers)
    seed += 1_000_000
    for a, b in zip(p, s):
        k = tuple(map(tuple, a))
        if k not in seen:
            seen.add(k); extra_p.append(a); extra_s.append(b)
        if len(extra_p) == NUM_OURVAL:
            break
va_p, va_s = extra_p, extra_s


def to_set(ps, ss):
    x = np.array(ps, dtype=np.uint8).reshape(len(ps), 81)
    y = np.array(ss, dtype=np.uint8).reshape(len(ss), 81)
    assert x.max() <= 9 and y.min() >= 1 and y.max() <= 9
    assert ((x != 0).sum(1) == clues).all(), "clue count mismatch"
    n = len(ps)
    return {"inputs": x + 1, "labels": y + 1,
            "group_indices": np.arange(0, n + 1, dtype=np.int32),
            "puzzle_indices": np.arange(n + 1, dtype=np.int32),
            "puzzle_identifiers": np.zeros(n, dtype=np.int32)}


def save_split(split, sets):
    total = sum(len(r["group_indices"]) - 1 for r in sets.values())
    meta = {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": 11,
            "seq_len": 81, "num_puzzle_identifiers": 1, "total_groups": total,
            "mean_puzzle_examples": 1.0, "total_puzzles": total, "sets": list(sets)}
    d = os.path.join(OUT, split); os.makedirs(d, exist_ok=True)
    json.dump(meta, open(os.path.join(d, "dataset.json"), "w"))
    for name, r in sets.items():
        for k, v in r.items():
            np.save(os.path.join(d, f"{name}__{k}.npy"), v)


save_split("train", {"all": to_set(tr_p, tr_s)})
save_split("test", {"val": to_set(va_p, va_s), "test": to_set(te_p, te_s)})
json.dump(["<blank>"], open(os.path.join(OUT, "identifiers.json"), "w"))
print(f"{args.difficulty} ({clues} clues): train {len(tr_p)}  test(=S-FLM validation) {len(te_p)}  "
      f"val(extra, seed {SEED_OURVAL}) {len(va_p)}  -> {OUT}")
