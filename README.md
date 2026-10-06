# Unrolled Flow Models for Reasoning — reproducibility code

> **Work in progress.** This repository is being cleaned up and documented; code, configs and
> instructions may still change. Replication runs of the paper's results are ongoing.

Training and evaluation code for the Sudoku-Extreme, Maze-Hard and Sudoku-Hard results of the
paper. UFM: a 2-layer, 448-wide DiT (8.4M parameters) predicts the clean endpoint of a latent flow;
training unrolls a 24-step Euler rollout with a hinted random start, backpropagates through the
last 6 steps, and applies a single cross-entropy loss to the decoded terminal state. The carried
state is retracted to the sphere of radius sqrt(d) before each update from the second step on
(`psi_update`); the final readout is decoded directly.

The training framework, data loader and Sudoku/Maze builders derive from the public TRM code base
(MIT), which itself builds on HRM (Apache-2.0); see `LICENSE` and `LICENSE-APACHE-2.0`. The
model, rollout and losses are in `models/dit_ref.py` and `models/losses.py`.

## Setup

Python 3.10, CUDA 12.4. Exact versions used for the paper runs are in `requirements.txt`.

```
pip install -r requirements.txt
```

Weights & Biases is off by default (`use_wandb: false`); every metric is also printed.

## Data

All three datasets are built locally; the first two download the public CSVs from HuggingFace.

```
# Sudoku-Extreme: 1000 base puzzles x 1000 augmentations; val = 2000 held-out train-pool puzzles,
# test = 2000-puzzle random subsample of the official test set (the full test set is 422,786 puzzles)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
# full official test set (422,786 puzzles), used only for the final Sudoku-Extreme numbers: its test split
# keeps every puzzle (a subsample size >= 422,786 disables subsampling); its train/val splits are not used
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-full --subsample-size 1000 --num-aug 0 --test-subsample-size 500000

# Maze-Hard 30x30: 1000 training mazes, 1000 test mazes (this benchmark has no validation split)
python dataset/build_maze_dataset.py --output-dir data/maze-30x30-hard-1k

# Sudoku-Hard: the 48k/2k synthetic split of the S-FLM paper (30 givens), regenerated with the
# original generator and seed; plus 2000 extra puzzles (seed 43) used only for checkpoint selection
python dataset/build_sudoku_hard_dataset.py --difficulty hard --output-dir data/sudoku-hard
```

Encoding: uint8 tokens, digit + 1 (blank = 1), 81 cells for Sudoku, 900 cells for Maze.

## Training

```
# Sudoku-Extreme, one A100-40GB, ~22 h
python pretrain.py --config-name=sudoku_extreme

# Maze-Hard, two A100-40GB, ~21 h
torchrun --standalone --nproc-per-node=2 pretrain.py --config-name=maze_hard

# Sudoku-Hard at the S-FLM budget (batch 256, 19.7k steps), one A100-40GB, ~1.1 h
python pretrain.py --config-name=sudoku_hard

# Sudoku-Hard at our full budget (batch 768, 130k steps), one A100-40GB, ~22 h
python pretrain.py --config-name=sudoku_hard_full
```

Checkpoints are written to `checkpoints/<Project>/<run_name>/best` together with `all_config.yaml`.
During training the model is evaluated every `eval_interval` epochs on the sets listed in
`eval_sets` at N in {8, 16, 32, 64, 128}, and the checkpoint with the best `best_metric` is kept.

Checkpoint selection:
- Sudoku-Extreme and Sudoku-Hard: `eval_sets: [val]`, i.e. only the validation split is seen
  during training; the test set is never evaluated until `evaluate.py`.
- Maze-Hard ships with train and test only. Training-time evaluation and checkpoint selection
  therefore use the test mazes (this holds for every method reported in the paper).
  With `save_best_only: false` the run directory keeps a `step_<n>` file per evaluation; pass
  `--ckpt-file step_<n>` to `evaluate.py` for a selection-free number.

## Evaluation

Exact-match accuracy of one rollout per puzzle (Pass@1), averaged over 5 independent noise seeds:

```
# Sudoku-Extreme, full official test set (422,786 puzzles); uniform grid and geometric grid (G=8)
python evaluate.py checkpoints/Sudoku-extreme-1k-aug-1000-DiTFlow-torch/sudoku_extreme data/sudoku-extreme-full/test --set test --N 128 --grid uniform
python evaluate.py checkpoints/Sudoku-extreme-1k-aug-1000-DiTFlow-torch/sudoku_extreme data/sudoku-extreme-full/test --set test --N 128 --grid geom --G 8

# Maze-Hard, 1000 test mazes
python evaluate.py checkpoints/Maze-30x30-hard-1k-DiTFlow-torch/maze_hard data/maze-30x30-hard-1k/test --set all --N 128

# Sudoku-Hard, the 2000 reporting puzzles of the S-FLM paper
python evaluate.py checkpoints/Sudoku-hard-DiTFlow-torch/sudoku_hard data/sudoku-hard/test --set test --N 128
python evaluate.py checkpoints/Sudoku-hard-DiTFlow-torch/sudoku_hard_full data/sudoku-hard/test --set test --N 128
```

Expected results (exact match %, N = 128, mean +/- std over 5 noise seeds):

| benchmark | test population | uniform grid | geometric grid (G=8) |
|---|---|---|---|
| Sudoku-Extreme | 422,786 | 74.4 +/- 0.0 | 79.7 +/- 0.0 |
| Maze-Hard | 1,000 | 89.3 +/- 0.5 | — |
| Sudoku-Hard, S-FLM budget | 2,000 | 86.9 +/- 0.9 | — |
| Sudoku-Hard, full budget | 2,000 | 99.8 (paper: single seed; the command above averages 5) | — |

Multiple rollouts per puzzle: Pass@K and the margin chooser of the selection section, on the
same rollouts. For Sudoku-Extreme this runs on the 2,000-puzzle test subsample built above (the paper's
selection study uses that subsample); N = 128 and, for Sudoku-Extreme, the geometric grid G = 8 are the
settings used throughout the paper.

```
python select_rollouts.py checkpoints/Sudoku-extreme-1k-aug-1000-DiTFlow-torch/sudoku_extreme data/sudoku-extreme-1k-aug-1000/test --set test --N 128 --K 100
python select_rollouts.py checkpoints/Maze-30x30-hard-1k-DiTFlow-torch/maze_hard data/maze-30x30-hard-1k/test --set all --N 128 --K 100
```

Runs are not bit-reproducible across GPU types. Every paper number comes from a single training
seed; the +/- values above are over inference noise seeds only, and training-seed variance was not
measured.

## Baselines

FLM and S-FLM numbers in the paper come from the authors' released code
(https://github.com/jdeschena/s-flm, commit 30def79), with an exact-match readout on the held-out
puzzles added. They are not included here. Settings used:

- Sudoku-Hard: the released Sudoku scripts (`scripts/train/sudoku/{flm,sfm_truncated}.sh`: 8 layers,
  width 512, batch 256, lr 3e-4) with the cosine learning-rate schedule from the same code base in
  place of the constant one, trained for 60k steps; the reported numbers are the best checkpoints
  (30k steps) selected on the 2k reporting puzzles, and S-FLM uses ancestral noise removal at the
  final step. The released script run verbatim (constant lr, 20k steps, last checkpoint) gives
  45.5 (FLM) and 36.8 (S-FLM).
- Sudoku-Extreme and Maze-Hard: the same models with data adapters that write the datasets built
  above in their token format, trained with dropout 0.3, weight decay 1.0, peak lr 1e-4 with cosine
  decay, for 100k steps (Sudoku-Extreme) and up to 46k steps (Maze-Hard); best checkpoints on the
  validation split (Sudoku-Extreme) or the test mazes (Maze-Hard).

## Layout

```
pretrain.py                 training loop (Hydra configs in config/)
evaluate.py                 seed-averaged Pass@1 evaluation of a checkpoint
select_rollouts.py          K rollouts per puzzle: Pass@K and the margin chooser
prosqa/                     self-contained ProsQA code (own README)
puzzle_dataset.py           dataset loader
models/dit_ref.py           model, rollout, retraction
models/losses.py            stablemax cross-entropy loss head and metrics
config/cfg_pretrain.yaml    shared training defaults
config/arch/dit_ref_lit.yaml  architecture defaults
config/{sudoku_extreme,maze_hard,sudoku_hard,sudoku_hard_full}.yaml   the paper runs
dataset/                    data builders (Sudoku-Extreme, Maze-Hard, Sudoku-Hard + generator)
```
