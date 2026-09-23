"""Every action of every demo must land in some training window.

The old window range required the obs future span, the whole packed track
horizon AND the whole action chunk to fit before a decision frame was legal.
That silently truncated action supervision: turning on tracks or a longer
obs_future cut the last ~20 native steps of every demo, so no arm ever learned
how a demo ends. These tests pin down the replacement, where the range depends
only on the demo length and everything past the end is padded and masked.
"""
import numpy as np
import pytest
import torch

from util.modality_forcing.codec import encode_depth_u16_png, encode_rgb_jpeg
from util.modality_forcing.config import MFConfig
from util.modality_forcing.data import MFStats, ModalityForcingDataset
from util.modality_forcing.model import ModalityForcingWAM

N_KEYFRAMES = 15
KF_STRIDE = 8
N_OFFSETS = 4
TRAJ_LEN = 115


def make_cfg(**overrides) -> MFConfig:
    kwargs = dict(
        modalities=("dino", "tracks", "depth", "action"),
        generation_order=("tracks", "dino", "depth", "action"),
        dim=384,
        depth_img_size=28, image_height=28, image_width=42,
        grid=2, grid_height=2, grid_width=3,
        track_grid=2, track_grid_height=2, track_grid_width=3,
        obs_history=1, obs_future=2, obs_stride=8, action_horizon=16,
        action_dim=2, proprio_dim=2,
    )
    kwargs.update(overrides)
    return MFConfig(**kwargs)


def write_pack(demo, traj_len=TRAJ_LEN, n_keyframes=N_KEYFRAMES,
               offset_valid=None):
    """Pack whose actions encode their own native index."""
    demo.mkdir(parents=True)
    cfg = make_cfg()
    rgb = np.zeros((cfg.spatial_height, cfg.spatial_width, 3), np.uint8)
    depth = np.ones((cfg.spatial_height, cfg.spatial_width), np.float32)
    native = torch.arange(traj_len, dtype=torch.float32)[:, None].repeat(1, 2)
    # Mirrors the real packs: only anchors whose full horizon fits are tracked.
    valid = torch.tensor([k * KF_STRIDE + N_OFFSETS * KF_STRIDE <= traj_len - 1
                          for k in range(n_keyframes)])
    pack = {
        "pack_version": 2,
        "visual_only": False,
        "keyframe_stride": KF_STRIDE,
        "traj_len": traj_len,
        "n_keyframes": n_keyframes,
        "image_height": cfg.spatial_height,
        "image_width": cfg.spatial_width,
        "dino_tokens": torch.arange(
            n_keyframes, dtype=torch.float16)[:, None, None].repeat(
                1, cfg.n_patches, cfg.dino_dim),
        "depths_png": [
            encode_depth_u16_png(depth, 10.0) for _ in range(n_keyframes)],
        "images_jpg": [encode_rgb_jpeg(rgb, 95) for _ in range(n_keyframes)],
        "point_tracks": torch.arange(
            N_OFFSETS, dtype=torch.float16)[None, :, None, None].repeat(
                n_keyframes, 1, cfg.n_patches, 3),
        "track_valid": valid,
        # Deliberately non-zero, so delta mode can tell "held at the seed" apart
        # from the raw zeros a pack stores for untracked offsets.
        "track_seed": torch.full((cfg.n_patches, 2), 0.5),
        "actions": native,
        "proprio": native.clone(),
    }
    if offset_valid is not None:
        pack["track_offset_valid"] = offset_valid
    torch.save(pack, demo / "trajectory.pt")


def make_dataset(demo, cfg):
    stats = MFStats(
        action_mean=torch.zeros(2), action_std=torch.ones(2),
        proprio_mean=torch.zeros(2), proprio_std=torch.ones(2),
        dino_mean=0.0, dino_std=1.0)
    return ModalityForcingDataset(
        [str(demo)], cfg, stats, depth_mean=0.0, depth_std=1.0,
        random_window=False, task_to_id={"beat_block_hammer": 0})


def covered_actions(dataset):
    """Union of native action indices reachable across every legal window."""
    cfg = dataset.cfg
    k_lo, k_hi = dataset._v2_k_range(
        dataset.lengths[0], dataset.n_keyframes[0], dataset.kf_strides[0],
        dataset.kf_ratios[0], dataset.visual_only[0], dataset.track_futures[0])
    seen = set()
    for k in range(k_lo, k_hi + 1):
        tau = k * dataset.kf_strides[0]
        seen.update(range(tau, min(tau + cfg.action_horizon, dataset.lengths[0])))
    return seen


def test_every_action_is_trainable(tmp_path):
    demo = tmp_path / "beat_block_hammer" / "demo_000000"
    write_pack(demo)
    dataset = make_dataset(demo, make_cfg())
    assert covered_actions(dataset) == set(range(TRAJ_LEN))


def test_legacy_mode_loses_the_end_of_the_demo(tmp_path):
    """Guards the regression itself: the old range stops ~20 steps early."""
    demo = tmp_path / "beat_block_hammer" / "demo_000000"
    write_pack(demo)
    dataset = make_dataset(demo, make_cfg(window_mode="legacy"))
    covered = covered_actions(dataset)
    assert max(covered) == 95
    assert len(set(range(TRAJ_LEN)) - covered) == 19


def test_window_range_is_identical_with_and_without_tracks(tmp_path):
    """The whole point: arms must not differ in which windows they train on."""
    demo = tmp_path / "beat_block_hammer" / "demo_000000"
    write_pack(demo)
    ranges = []
    for modalities, obs_future in [
            (("dino", "depth", "action"), 2),
            (("dino", "tracks", "depth", "action"), 2),
            (("dino", "depth", "action"), 4),
    ]:
        cfg = make_cfg(modalities=modalities, obs_future=obs_future)
        dataset = make_dataset(demo, cfg)
        ranges.append(dataset._v2_k_range(
            dataset.lengths[0], dataset.n_keyframes[0], dataset.kf_strides[0],
            dataset.kf_ratios[0], dataset.visual_only[0],
            dataset.track_futures[0]))
    assert len(set(ranges)) == 1


def test_tail_window_pads_and_masks(tmp_path):
    demo = tmp_path / "beat_block_hammer" / "demo_000000"
    write_pack(demo)
    dataset = make_dataset(demo, make_cfg())
    dataset.random_window = False
    # Force the very last decision frame: tau=112 with only 3 real actions left.
    dataset._v2_k_range = lambda *a, **k: (N_KEYFRAMES - 1, N_KEYFRAMES - 1)
    sample = dataset[0]
    assert sample["action_valid"].tolist() == [1.0] * 3 + [0.0] * 13
    # Padded steps repeat the final action rather than running off the array.
    actions = sample["actions"][:, 0]
    assert actions[:3].tolist() == [112.0, 113.0, 114.0]
    assert actions[3:].eq(114.0).all()
    # Future keyframes 15 and 16 do not exist, so both futures are padded.
    assert sample["obs_future_valid"].tolist() == [0.0, 0.0]
    dino = sample["dino"][:, 0, 0]
    assert dino.eq(float(N_KEYFRAMES - 1)).all()


def test_missing_track_offsets_repeat_the_last_valid_position(tmp_path):
    demo = tmp_path / "beat_block_hammer" / "demo_000000"
    # Offset 0 tracked, offsets 1..3 not, at every keyframe.
    offset_valid = torch.zeros(N_KEYFRAMES, N_OFFSETS, dtype=torch.bool)
    offset_valid[:, 0] = True
    write_pack(demo, offset_valid=offset_valid)
    dataset = make_dataset(demo, make_cfg(track_pred_mode="absolute"))
    sample = dataset[0]
    assert sample["track_future_valid"].tolist() == [1.0, 0.0]
    # The pack stores offset j as the value j, so reading it raw would give
    # [0.0, 1.0]. Holding the last tracked position gives [0.0, 0.0].
    assert sample["point_tracks"][:, 0, 0].tolist() == [0.0, 0.0]


def test_wholly_untracked_window_means_zero_displacement(tmp_path):
    """In delta mode the pack's zeros would otherwise read as -seed, a large
    bogus displacement fed to the action as clean teacher context."""
    demo = tmp_path / "beat_block_hammer" / "demo_000000"
    write_pack(demo, offset_valid=torch.zeros(
        N_KEYFRAMES, N_OFFSETS, dtype=torch.bool))
    dataset = make_dataset(demo, make_cfg(track_pred_mode="delta"))
    sample = dataset[0]
    assert sample["track_future_valid"].tolist() == [0.0, 0.0]
    assert sample["point_tracks"][..., :2].abs().max().item() == 0.0


def test_masked_loss_ignores_padded_frames():
    pred = torch.zeros(2, 2, 4, 3)
    target = torch.zeros(2, 2, 4, 3)
    target[:, 1] = 100.0                      # huge error, second future frame
    weight = torch.ones(2)
    mask = torch.ones(2)
    frame_w = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    masked = ModalityForcingWAM._masked_loss(pred, target, weight, mask, frame_w)
    assert masked.item() == pytest.approx(0.0)
    unmasked = ModalityForcingWAM._masked_loss(pred, target, weight, mask)
    assert unmasked.item() > 0.0


def test_masked_loss_matches_plain_mean_when_all_frames_are_valid():
    torch.manual_seed(0)
    pred, target = torch.randn(3, 2, 4, 3), torch.randn(3, 2, 4, 3)
    weight, mask = torch.rand(3), torch.ones(3)
    plain = ModalityForcingWAM._masked_loss(pred, target, weight, mask)
    gated = ModalityForcingWAM._masked_loss(
        pred, target, weight, mask, torch.ones(3, 2))
    assert gated.item() == pytest.approx(plain.item(), rel=1e-6)


def test_sample_with_no_valid_frames_leaves_the_denominator():
    """Otherwise an all-padded sample would dilute the batch with a 0/0."""
    pred, target = torch.zeros(2, 2, 4, 3), torch.ones(2, 2, 4, 3)
    weight, mask = torch.ones(2), torch.ones(2)
    frame_w = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    gated = ModalityForcingWAM._masked_loss(
        pred, target, weight, mask, frame_w)
    # Only the first sample counts, and its error is exactly 1.0.
    assert gated.item() == pytest.approx(1.0)
