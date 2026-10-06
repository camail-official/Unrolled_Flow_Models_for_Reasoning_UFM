# ProsQA: latent flow with rollout training

Self-contained code for the ProsQA graph-reachability results of the paper. A 2-layer,
768-wide DiT (15.7M parameters) reads the graph and the question as clean conditioning and
fills a single answer slot by integrating a latent flow. Training unrolls a 5-step Euler
rollout from a randomly timed start on the interpolant, backpropagates through all steps
with per-step gradient attenuation, and applies one cross-entropy at the terminal state;
the supervised node follows a staged curriculum along the shortest path. Inference starts
from noise and integrates `N` Euler steps; accuracy is reported as a function of `N`.

```
data.py            tokenizer (40 ids), sample formatting, collation
model.py           the DiT (embed_ids / forward_d / decode)
flow.py            rollout loss and Euler sampler
train.py           training loop, per-N evaluation, checkpoint selection
build_prosqa.py    builds the data splits from the public COCONUT release
configs/           the two paper runs
```

## Setup

Same environment as the parent directory (`pip install -r ../requirements.txt`).

## Data

ProsQA data of COCONUT (Hao et al., 2024), keeping the questions whose answer is at most
4 hops from the root (14,785 / 257 / 419 train / valid / test questions).

```
git clone https://github.com/facebookresearch/coconut
python build_prosqa.py --src coconut/data --out data
```

## Training

```
python train.py configs/fixed_tN.yaml     # main run: fixed terminal knot (t_N = 0.95)
python train.py configs/random_tN.yaml    # variant: random terminal knot
```

One GPU, about 3 hours per run. Every 10 epochs the EMA weights are evaluated on the
validation set at N in {1, ..., 8, 16}; `best.pt` is the checkpoint with the best validation
accuracy at N = 5, `best_N_<n>.pt` the best at each n. The test set is evaluated once, at the
end, as an N-sweep of `best.pt` (`results.json`); it plays no role in training or selection.

## Expected results

Answer accuracy of `best.pt`, N-sweep:

| configuration | val @N=5 | test N=1 | N=2 | N=4 | N=8 | N=16 | N=64 |
|---|---|---|---|---|---|---|---|
| fixed terminal knot (`fixed_tN.yaml`) | 1.000 | 0.477 | 0.771 | 0.993 | 0.998 | 0.998 | 0.998 |
| random terminal knot (`random_tN.yaml`) | 0.988 | 0.695 | 0.883 | 0.967 | 0.976 | 0.974 | 0.979 |

Accuracy rises from N = 1 to about N = 4 and is flat afterwards. Runs are not
bit-reproducible across GPU types; expect about one point of variation on the test set.

## What the model sees

- Input: the edge list, the two candidate answers and the root, all as node ids. The built
  data files contain only `edges`, `root`, `target`, `neg_target`; the natural-language
  questions and written-out reasoning steps of the release are not used.
- Training uses two things beyond the answer: the rollout may start on the interpolant
  between noise and the answer embedding (`t0_max`), and the supervised node follows the
  shortest path during the curriculum. Both are training-time only.
- At evaluation the answer slot is overwritten with the pad id before sampling; the
  prediction is decoded from noise given the graph and the question alone.

## Notes

- The tokenizer keeps four unused special ids so that `vocab_size = 40`.
- Training ends each rollout with an Euler step to the terminal knot (t_N = 0.95 in the main
  configuration); at inference the sampler integrates to `t_max` and reads out the denoiser's
  endpoint estimate, so the terminal knot is a training-only quantity.
- Run the commands from inside `prosqa/` (the scripts import each other by bare name).
- `bf16_acc32`: the network runs in bf16 autocast while the rollout state is carried in fp32.
  With the state itself in bf16 the rollout diverges.
- The N = 1 accuracy depends on how the first step is evaluated: `train.py` starts every
  rollout from pure noise at t = 0, which is the protocol of all numbers above.
