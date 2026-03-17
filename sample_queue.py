"""
Class-balanced batch sampler for ImageNet training.

Produces mini-batches where each batch contains exactly N randomly chosen
classes, each with M samples. Batch size = N * M.

This is an alternative to the standard IID sampling (DistributedSampler),
adapted from the per-class queue pattern in drifting_model/.

Usage:
    sampler = DistributedClassBalancedBatchSampler(
        dataset=dataset_train,
        num_classes_per_batch=50,   # N classes per batch
        num_samples_per_class=2,    # M samples per class
        num_replicas=world_size,
        rank=global_rank,
    )
    data_loader = DataLoader(
        dataset_train,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
"""

import math
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler):
    """
    Batch sampler that yields batches of N classes × M samples per class.

    Each batch:
        1. Randomly picks N classes (without replacement) from all classes.
        2. For each chosen class, randomly samples M dataset indices.
        3. Returns a flat list of N*M indices.

    This ensures every mini-batch has balanced class representation,
    which is useful for contrastive / drifting-field style training.
    """

    def __init__(
        self,
        dataset,
        num_classes_per_batch: int = 50,
        num_samples_per_class: int = 2,
        num_batches: int = None,
        seed: int = 0,
        epoch: int = 0,
    ):
        """
        Args:
            dataset: A dataset with .targets attribute (e.g., ImageFolder).
            num_classes_per_batch: N — number of distinct classes per batch.
            num_samples_per_class: M — number of samples per class per batch.
            num_batches: Number of batches per epoch. If None, defaults to
                         len(dataset) // (N * M) to roughly match one epoch.
            seed: Base random seed for reproducibility.
            epoch: Current epoch (call set_epoch() to change).
        """
        self.num_classes_per_batch = num_classes_per_batch
        self.num_samples_per_class = num_samples_per_class
        self.seed = seed
        self.epoch = epoch

        # Build class-to-indices mapping
        self.class_to_indices = defaultdict(list)
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        for idx, label in enumerate(targets):
            self.class_to_indices[label].append(idx)

        self.all_classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.all_classes)

        assert num_classes_per_batch <= self.num_classes, (
            f"num_classes_per_batch ({num_classes_per_batch}) > "
            f"total classes ({self.num_classes})"
        )

        self.batch_size = num_classes_per_batch * num_samples_per_class

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = len(dataset) // self.batch_size

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling (mirrors DistributedSampler)."""
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        for _ in range(self.num_batches):
            # Step 1: randomly pick N classes
            chosen_classes = rng.choice(
                self.all_classes, size=self.num_classes_per_batch, replace=False
            )

            batch_indices = []
            for c in chosen_classes:
                pool = self.class_to_indices[c]
                # Step 2: randomly sample M indices from this class
                if len(pool) >= self.num_samples_per_class:
                    selected = rng.choice(
                        pool, size=self.num_samples_per_class, replace=False
                    )
                else:
                    # Fewer samples than M — sample with replacement
                    selected = rng.choice(
                        pool, size=self.num_samples_per_class, replace=True
                    )
                batch_indices.extend(selected.tolist())

            yield batch_indices

    def __len__(self):
        return self.num_batches


class DistributedClassBalancedBatchSampler(Sampler):
    """
    Distributed version of ClassBalancedBatchSampler.

    Each GPU/rank independently draws its own class-balanced batches
    using a rank-specific random seed, ensuring no duplication across ranks.
    Each rank produces the same number of batches per epoch.
    """

    def __init__(
        self,
        dataset,
        num_classes_per_batch: int = 50,
        num_samples_per_class: int = 2,
        num_batches: int = None,
        num_replicas: int = None,
        rank: int = None,
        seed: int = 0,
        epoch: int = 0,
    ):
        """
        Args:
            dataset: A dataset with .targets attribute (e.g., ImageFolder).
            num_classes_per_batch: N — number of distinct classes per batch.
            num_samples_per_class: M — number of samples per class per batch.
            num_batches: Number of batches per epoch per rank. If None, defaults
                         to len(dataset) // (N * M * num_replicas).
            num_replicas: Number of processes (world_size). Auto-detected if None.
            rank: Rank of current process. Auto-detected if None.
            seed: Base random seed.
            epoch: Current epoch (call set_epoch() to change).
        """
        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.num_classes_per_batch = num_classes_per_batch
        self.num_samples_per_class = num_samples_per_class
        self.seed = seed
        self.epoch = epoch

        # Build class-to-indices mapping
        self.class_to_indices = defaultdict(list)
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        for idx, label in enumerate(targets):
            self.class_to_indices[label].append(idx)

        self.all_classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.all_classes)

        assert num_classes_per_batch <= self.num_classes, (
            f"num_classes_per_batch ({num_classes_per_batch}) > "
            f"total classes ({self.num_classes})"
        )

        self.batch_size = num_classes_per_batch * num_samples_per_class

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            # Divide total dataset across ranks
            self.num_batches = len(dataset) // (self.batch_size * self.num_replicas)

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling (mirrors DistributedSampler)."""
        self.epoch = epoch

    def __iter__(self):
        # Each rank uses a different seed to get different batches
        rng = np.random.RandomState(
            self.seed + self.epoch * self.num_replicas + self.rank
        )

        for _ in range(self.num_batches):
            # Step 1: randomly pick N classes
            chosen_classes = rng.choice(
                self.all_classes, size=self.num_classes_per_batch, replace=False
            )

            batch_indices = []
            for c in chosen_classes:
                pool = self.class_to_indices[c]
                # Step 2: randomly sample M indices from this class
                if len(pool) >= self.num_samples_per_class:
                    selected = rng.choice(
                        pool, size=self.num_samples_per_class, replace=False
                    )
                else:
                    selected = rng.choice(
                        pool, size=self.num_samples_per_class, replace=True
                    )
                batch_indices.extend(selected.tolist())

            yield batch_indices

    def __len__(self):
        return self.num_batches
