from typing import Optional, List
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    seed: int = 0  # seeds subsampling + augmentation

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0

    # Held-out sets, written to the test split as sets ["val", "test"]:
    # val  = train-pool examples NOT chosen for the training subsample (disjoint from train);
    #        the only set evaluated during training (config eval_sets: [val])
    # test = random subsample of the official test.csv; evaluated once by evaluate.py
    val_size: int = 2000
    test_subsample_size: Optional[int] = 2000


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def read_csv(set_name: str, config: DataProcessConfig):
    inputs = []
    labels = []

    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                assert len(q) == 81 and len(a) == 81

                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    return inputs, labels


def build_set(inputs, labels, num_augments: int):
    # Group/puzzle hierarchy IS used for Sudoku/Maze training: the sampler draws one
    # random puzzle (= one augmentation) per group (= base puzzle) per epoch.
    # Same semantics as the original TRM builder, but written into preallocated flat
    # arrays: a python list of 1M+ tiny arrays uses too much memory.
    per_puzzle = 1 + num_augments
    total = len(inputs) * per_puzzle

    inputs_arr = np.zeros((total, 81), dtype=np.uint8)
    labels_arr = np.zeros((total, 81), dtype=np.uint8)

    example_id = 0
    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(per_puzzle):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            inputs_arr[example_id] = inp.reshape(-1)
            labels_arr[example_id] = out.reshape(-1)
            example_id += 1

    assert np.all(inputs_arr <= 9) and np.all(labels_arr <= 9)

    return {
        "inputs": inputs_arr + 1,
        "labels": labels_arr + 1,

        "group_indices": np.arange(0, total + 1, per_puzzle, dtype=np.int32),
        "puzzle_indices": np.arange(total + 1, dtype=np.int32),
        # Puzzle identifiers are not used for Sudoku (all-blank, 0)
        # for Sudoku/Maze, but the dataloader still expects the array to exist.
        "puzzle_identifiers": np.zeros(total, dtype=np.int32),
    }


def save_split(split_name: str, sets: dict, config: DataProcessConfig):
    total_groups = sum(len(r["group_indices"]) - 1 for r in sets.values())

    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=total_groups,
        mean_puzzle_examples=1,
        total_puzzles=total_groups,
        sets=list(sets.keys())
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for set_name, results in sets.items():
        for k, v in results.items():
            np.save(os.path.join(save_dir, f"{set_name}__{k}.npy"), v)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    np.random.seed(config.seed)

    # Train pool: subsample train examples; val = examples NOT chosen for train
    train_inputs, train_labels = read_csv("train", config)
    total = len(train_inputs)

    train_idx = np.arange(total)
    if config.subsample_size is not None and config.subsample_size < total:
        train_idx = np.random.choice(total, size=config.subsample_size, replace=False)

    remaining = np.setdiff1d(np.arange(total), train_idx)
    assert len(remaining) >= config.val_size, f"Not enough held-out examples for val: {len(remaining)} < {config.val_size}"
    val_idx = np.random.choice(remaining, size=config.val_size, replace=False)

    # Extract subsets, then free the full train pool (~3.8M examples) before augmenting
    tr_in, tr_lab = [train_inputs[i] for i in train_idx], [train_labels[i] for i in train_idx]
    va_in, va_lab = [train_inputs[i] for i in val_idx],   [train_labels[i] for i in val_idx]
    del train_inputs, train_labels

    # Official test set: random subsample
    test_inputs, test_labels = read_csv("test", config)
    test_idx = np.arange(len(test_inputs))
    if config.test_subsample_size is not None and config.test_subsample_size < len(test_idx):
        test_idx = np.random.choice(len(test_idx), size=config.test_subsample_size, replace=False)
    te_in, te_lab = [test_inputs[i] for i in test_idx], [test_labels[i] for i in test_idx]
    del test_inputs, test_labels

    print(f"train: {len(train_idx)} x (1 + {config.num_aug} augs) | val: {len(va_in)} (train-pool, disjoint) | test: {len(te_in)}")

    save_split("train", {
        "all": build_set(tr_in, tr_lab, config.num_aug),
    }, config)

    save_split("test", {
        "val":  build_set(va_in, va_lab, 0),
        "test": build_set(te_in, te_lab, 0),
    }, config)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    cli()
