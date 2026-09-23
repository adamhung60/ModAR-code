"""Short training sets are repeated so each epoch contains enough batches.

Datasets are indexed by demonstration. A small set can otherwise produce only
one or two batches per rank, which rebuilds the loader before it can pipeline.
Repeating a demonstration draws a fresh random window rather than replaying the
same sample, and validation loaders are left at their true length.
"""
import numpy as np
import pytest
import torch

from util.modality_forcing.data import (MIN_BATCHES_PER_EPOCH, MFStats,
                                        ModalityForcingDataset, _build_loader)

from tests.test_depth_decode_gating import make_cfg, write_pack


def make_ds(tmp_path, n_demos, random_window=True, epoch_repeat=1):
    demos = []
    for i in range(n_demos):
        d = tmp_path / "beat_block_hammer" / f"demo_{i:06d}"
        write_pack(d)
        demos.append(str(d))
    stats = MFStats(
        action_mean=torch.zeros(2), action_std=torch.ones(2),
        proprio_mean=torch.zeros(2), proprio_std=torch.ones(2),
        dino_mean=0.0, dino_std=1.0)
    return ModalityForcingDataset(
        demos, make_cfg(), stats, depth_mean=0.0, depth_std=1.0,
        random_window=random_window, epoch_repeat=epoch_repeat,
        task_to_id={"beat_block_hammer": 0})


def test_repeat_stretches_len_but_not_demo_count(tmp_path):
    ds = make_ds(tmp_path, 3, epoch_repeat=7)
    assert ds.n_demos == 3
    assert len(ds) == 21


def test_indices_past_the_demo_list_wrap(tmp_path):
    """The sampler draws over the stretched length, so __getitem__ must accept
    indices beyond the demo list rather than raising."""
    ds = make_ds(tmp_path, 2, random_window=False, epoch_repeat=5)
    for idx in range(len(ds)):
        sample = ds[idx]
        assert sample["dino"].shape[0] > 0
    # A wrapped index addresses the same demo as its base index.
    a, b = ds[0], ds[2]
    assert torch.equal(a["dino"], b["dino"])


def test_revisiting_a_demo_draws_a_fresh_window(tmp_path):
    """THE correctness claim behind stretching: a repeat is another epoch, not a
    duplicated sample. With random_window the tau draw comes from the ambient
    numpy RNG, so two visits to one demo must differ."""
    ds = make_ds(tmp_path, 1, random_window=True, epoch_repeat=64)
    np.random.seed(0)
    windows = {tuple(ds[i]["actions"].flatten()[:4].tolist())
               for i in range(len(ds))}
    assert len(windows) > 1, (
        "every visit returned the same window: the stretch would be training on "
        "duplicates inside an epoch")


def test_deterministic_window_is_unaffected(tmp_path):
    """Val-style pinning must still be deterministic; only the epoch got longer."""
    ds = make_ds(tmp_path, 1, random_window=False, epoch_repeat=4)
    first = ds[0]["actions"]
    for i in range(len(ds)):
        assert torch.equal(ds[i]["actions"], first)


def test_train_loader_gets_enough_batches_per_epoch(tmp_path):
    ds = make_ds(tmp_path, 4)
    loader = _build_loader(ds, batch_size=2, shuffle=True, num_workers=2,
                           persistent=True)
    assert len(loader) >= MIN_BATCHES_PER_EPOCH


def test_val_loader_is_not_stretched(tmp_path):
    """A val loader's length IS the eval set; stretching it would silently
    multiply the number of eval batches."""
    ds = make_ds(tmp_path, 4)
    _build_loader(ds, batch_size=2, shuffle=False, num_workers=2,
                  persistent=False)
    assert ds.epoch_repeat == 1
    assert len(ds) == 4


def test_plan_windows_all_covers_every_decision_frame(tmp_path):
    ds = make_ds(tmp_path, 2, random_window=False)
    plan = ds.plan_windows("all")
    lo, hi = ds._decision_range(0)
    assert len(plan) == 2 * (hi - lo + 1)
    assert len(ds) == len(plan)
    first = ds[0]
    last = ds[hi - lo]
    assert first["dino"].shape == last["dino"].shape
    assert not torch.equal(first["dino"], last["dino"])


def test_plan_windows_random_is_pinned_by_seed(tmp_path):
    ds = make_ds(tmp_path, 3, random_window=False)
    first = ds.plan_windows("random", seed=7, per_demo=1)
    assert ds.plan_windows("random", seed=7, per_demo=1) == first
    assert ds.plan_windows("random", seed=8, per_demo=1) != first
    assert len(first) == 3


def test_single_worker_loader_is_not_stretched(tmp_path):
    """num_workers=0 has no pipeline to keep warm, so there is nothing to buy."""
    ds = make_ds(tmp_path, 4)
    _build_loader(ds, batch_size=2, shuffle=True, num_workers=0,
                  persistent=False)
    assert ds.epoch_repeat == 1
