"""Depth PNG decode is skipped only when nothing can read it.

Depth is the one sensor the loader has to INFLATE (u16 PNG) rather than read:
dino tokens are precomputed and RGB decode was already gated on image/tracks. On
an arm that does not model depth that inflate bought nothing, which is expensive
precisely because these steps are loader-bound rather than GPU-bound.

The risk of the gate is not that it fails to help, it is that it silently feeds
some arm zeros where real depth belongs. So the arms that DO use depth are pinned
against independently recomputed values here, not merely checked for shape.
"""
import numpy as np
import pytest
import torch

import util.modality_forcing.data as data_mod
from util.modality_forcing.codec import (decode_depth_u16_png,
                                         encode_depth_u16_png, encode_rgb_jpeg)
from util.modality_forcing.config import MFConfig
from util.modality_forcing.data import MFStats, ModalityForcingDataset
from util.modality_forcing.model import ModalityForcingWAM
from util.depth_utils import normalize_depth_maps

N_KEYFRAMES = 15
KF_STRIDE = 8
TRAJ_LEN = 115
DEPTH_MAX_M = 10.0
# A value that is neither 0 nor 1, so a placeholder cannot pass for real depth
# and a missing normalization cannot pass either.
DEPTH_M = 2.5


def make_cfg(**overrides) -> MFConfig:
    kwargs = dict(
        modalities=("dino", "tracks", "depth", "action"),
        dim=384,
        depth_img_size=28, image_height=28, image_width=42,
        grid=2, grid_height=2, grid_width=3,
        track_grid=2, track_grid_height=2, track_grid_width=3,
        obs_history=1, obs_future=2, obs_stride=8, action_horizon=16,
        action_dim=2, proprio_dim=2,
        generation_order=("tracks", "dino", "depth", "action"),
        generated_modalities=("tracks", "dino", "depth"),
    )
    kwargs.update(overrides)
    return MFConfig(**kwargs)


def dino_only_cfg(**overrides) -> MFConfig:
    """The shape of the real_world v4 dino-only arm: shortest possible cascade."""
    return make_cfg(
        modalities=("dino", "action"),
        generated_modalities=("dino",),
        generation_order=("dino", "action"),
        history_modalities=("dino",),
        **overrides)


def write_pack(demo):
    demo.mkdir(parents=True)
    cfg = make_cfg()
    rgb = np.zeros((cfg.spatial_height, cfg.spatial_width, 3), np.uint8)
    depth = np.full((cfg.spatial_height, cfg.spatial_width), DEPTH_M, np.float32)
    native = torch.arange(TRAJ_LEN, dtype=torch.float32)[:, None].repeat(1, 2)
    torch.save({
        "pack_version": 2,
        "visual_only": False,
        "keyframe_stride": KF_STRIDE,
        "traj_len": TRAJ_LEN,
        "n_keyframes": N_KEYFRAMES,
        "image_height": cfg.spatial_height,
        "image_width": cfg.spatial_width,
        "dino_tokens": torch.arange(
            N_KEYFRAMES, dtype=torch.float16)[:, None, None].repeat(
                1, cfg.n_patches, cfg.dino_dim),
        "depths_png": [
            encode_depth_u16_png(depth, DEPTH_MAX_M) for _ in range(N_KEYFRAMES)],
        "images_jpg": [encode_rgb_jpeg(rgb, 95) for _ in range(N_KEYFRAMES)],
        "point_tracks": torch.zeros(
            (N_KEYFRAMES, 4, cfg.n_patches, 3), dtype=torch.float16),
        "track_valid": torch.ones(N_KEYFRAMES, dtype=torch.bool),
        "track_seed": torch.full((cfg.n_patches, 2), 0.5),
        "actions": native,
        "proprio": native.clone(),
    }, demo / "trajectory.pt")


def make_dataset(demo, cfg):
    stats = MFStats(
        action_mean=torch.zeros(2), action_std=torch.ones(2),
        proprio_mean=torch.zeros(2), proprio_std=torch.ones(2),
        dino_mean=0.0, dino_std=1.0)
    return ModalityForcingDataset(
        [str(demo)], cfg, stats, depth_mean=0.0, depth_std=1.0,
        depth_max_m=DEPTH_MAX_M, random_window=False,
        task_to_id={"beat_block_hammer": 0})


@pytest.fixture
def demo(tmp_path):
    d = tmp_path / "beat_block_hammer" / "demo_000000"
    write_pack(d)
    return d


def count_decodes(monkeypatch):
    """Count depth inflates through the name data.py actually calls."""
    calls = []
    real = data_mod.decode_depth_u16_png

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(data_mod, "decode_depth_u16_png", spy)
    return calls


def test_modelled_depth_is_decoded_and_correct(demo, monkeypatch):
    """The arms in flight: depth must survive the gate byte for byte."""
    cfg = make_cfg()
    ds = make_dataset(demo, cfg)
    assert ds.want_depth
    calls = count_decodes(monkeypatch)
    sample = ds[0]

    assert calls, "depth is modelled here; it must still be decoded"
    # Recomputed from the pack independently of the loader, so this catches a
    # dropped normalization as well as a dropped decode.
    raw = decode_depth_u16_png(
        torch.load(demo / "trajectory.pt", weights_only=False)["depths_png"][0],
        DEPTH_MAX_M)
    expected = normalize_depth_maps(
        torch.from_numpy(np.stack([raw], 0)), 0.0, 1.0, max_depth_m=DEPTH_MAX_M,
        mode=ds.depth_norm_mode)
    torch.testing.assert_close(sample["depth_maps"][:1], expected)
    assert sample["depth_maps"].abs().sum() > 0


def test_dino_only_skips_decode_entirely(demo, monkeypatch):
    cfg = dino_only_cfg()
    ds = make_dataset(demo, cfg)
    assert not ds.want_depth
    calls = count_decodes(monkeypatch)
    sample = ds[0]
    assert not calls, "nothing reads depth on this arm; it must not be inflated"


def test_skipped_depth_keeps_batch_schema(demo):
    """The key stays, with the shape/dtype the model's positional arg expects."""
    cfg = dino_only_cfg()
    full = make_dataset(demo, make_cfg())[0]["depth_maps"]
    skipped = make_dataset(demo, cfg)[0]["depth_maps"]
    assert skipped.shape == full.shape
    assert skipped.dtype == full.dtype
    assert torch.count_nonzero(skipped) == 0


def test_depth_as_history_only_still_decodes(demo, monkeypatch):
    """Naming depth as context without modelling it is a supported ablation, so
    the gate is deliberately over-inclusive: it must not strand that arm."""
    cfg = make_cfg(
        modalities=("dino", "action"),
        generated_modalities=("dino",),
        generation_order=("dino", "action"),
        history_modalities=("dino", "depth"))
    ds = make_dataset(demo, cfg)
    assert ds.want_depth
    calls = count_decodes(monkeypatch)
    ds[0]
    assert calls


def test_dino_only_arm_trains_on_the_placeholder(demo):
    """End to end: the zeros must not reach a loss, a grad, or a NaN."""
    cfg = dino_only_cfg(schedule_mode="modar")
    ds = make_dataset(demo, cfg)
    sample = ds[0]
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}

    torch.manual_seed(0)
    model = ModalityForcingWAM(cfg)
    model.set_action_stats(torch.zeros(cfg.action_dim), torch.ones(cfg.action_dim))
    model.train()
    out = model(**batch)
    loss = out["loss"]
    assert torch.isfinite(loss), "placeholder depth leaked into the loss"
    loss.backward()

    # Depth carries no parameters on this arm, so a depth grad would mean the
    # placeholder was routed into the graph rather than ignored.
    depth_grads = [n for n, p in model.named_parameters()
                   if "depth" in n and p.grad is not None and p.grad.abs().sum() > 0]
    assert not depth_grads, f"placeholder reached depth params: {depth_grads}"
