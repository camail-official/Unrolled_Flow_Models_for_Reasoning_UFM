import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int
    # Flat sampling: ignore the group/puzzle hierarchy and treat the train split as one big
    # flat dataset — every epoch is a shuffled pass over ALL examples (augmentations included),
    # instead of TRM's one-random-augmentation-per-base-puzzle-per-epoch. NOTE: an "epoch"
    # then means a full pass over ~1M rows, so configs must scale `epochs` accordingly.
    flat_sampling: bool = False
    # Size bucketing (ARC dyn_crop): order groups by content extent (with jitter) so each
    # batch holds similar-sized grids; the model then crops the batch to its max extent.
    size_bucketing: bool = False
    # Optional subset of set names to load (e.g. ["val"] keeps the test set out of training-time eval).
    sets: Optional[List[str]] = None

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=([n for n in prev_sets if n in self.config.sets] if self.config.sets else prev_sets)
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }


    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_flat_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            total = len(dataset["inputs"])
            for _epoch in range(self.config.epochs_per_iter):
                order = rng.permutation(total)
                start = 0
                while start + self.config.global_batch_size <= total:  # drop last partial batch
                    idx = order[start:start + self.config.global_batch_size]
                    start += self.config.global_batch_size

                    local = idx[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                    local = np.sort(local)  # sorted indices for faster mmap reads
                    puzzle_ids = np.searchsorted(dataset["puzzle_indices"], local, side="right") - 1
                    batch = self._collate_batch({
                        "inputs": dataset["inputs"][local],
                        "labels": dataset["labels"][local],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_ids]
                    })
                    yield set_name, batch, self.config.global_batch_size

    def _group_max_ext(self, dataset):
        """Per-group max grid extent, key = R*64 + C (computed once, cached in-dict)."""
        if "_group_ext" not in dataset:
            inp, lab = dataset["inputs"], dataset["labels"]
            n = len(inp)
            g = int(round(float(np.sqrt(inp.shape[1]))))
            ex = np.zeros(n, dtype=np.int32)
            CH = 65536
            for st in range(0, n, CH):
                m = (np.asarray(inp[st:st + CH]) > 0) | (np.asarray(lab[st:st + CH]) > 0)
                m = m.reshape(len(m), g, g)
                rows, cols = m.any(2), m.any(1)
                R = g - np.argmax(rows[:, ::-1], axis=1); R[~rows.any(1)] = 1
                C = g - np.argmax(cols[:, ::-1], axis=1); C[~cols.any(1)] = 1
                ex[st:st + CH] = R.astype(np.int32) * 64 + C.astype(np.int32)
            starts = dataset["puzzle_indices"][dataset["group_indices"][:-1]]
            dataset["_group_ext"] = np.maximum.reduceat(ex, starts)
        return dataset["_group_ext"]

    def _iter_train(self):
        if self.config.flat_sampling:
            yield from self._iter_flat_train()
            return
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            if self.config.size_bucketing:
                ext = self._group_max_ext(dataset).astype(np.float64)
                group_order = np.concatenate([
                    np.argsort(ext + rng.uniform(0.0, 6.0, size=ext.size), kind="stable")
                    for _i in range(self.config.epochs_per_iter)])
            else:
                group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                raw = {
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                }
                batch = self._collate_batch(raw)
                batch["ex_idx"] = torch.from_numpy(np.ascontiguousarray(batch_indices.astype(np.int64)))
                if self.config.size_bucketing:
                    g = int(round(float(np.sqrt(raw["inputs"].shape[1]))))
                    m = ((np.asarray(raw["inputs"]) > 0) | (np.asarray(raw["labels"]) > 0)).reshape(-1, g, g)
                    rows, cols = m.any(2).any(0), m.any(1).any(0)
                    R = int(np.max(np.nonzero(rows)[0]) + 1) if rows.any() else 1
                    C = int(np.max(np.nonzero(cols)[0]) + 1) if cols.any() else 1
                    R, C = min(g, -(-R // 6) * 6), min(g, -(-C // 6) * 6)
                    batch["crop_rc"] = torch.tensor([R, C], dtype=torch.int32)

                yield set_name, batch, global_effective_batch_size
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()

