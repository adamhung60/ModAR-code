from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from robotwin_manip.datagen.robotwin_wam_io import (
    depth_mm_to_metric,
    intrinsic_for_resize,
    resize_rgb,
)
from scripts.data.extract_dino_features import _load_rgb_tensor
from scripts.data.extract_point_tracks import process_demo, seed_grid_pixels
from scripts.data.pack_wam_target_5mod import pack_one


def test_uncropped_320x240_resize_and_intrinsics(tmp_path: Path):
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[:, 0] = (255, 0, 0)
    rgb[:, -1] = (0, 255, 0)
    resized = resize_rgb(rgb, 168, 224)
    assert resized.shape == (168, 224, 3)
    assert resized[:, 0, 0].max() > 0
    assert resized[:, -1, 1].max() > 0

    depth_mm = np.arange(240 * 320, dtype=np.float32).reshape(240, 320)
    depth_m, valid = depth_mm_to_metric(depth_mm, 168, 224)
    expected = cv2.resize(depth_mm, (224, 168), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(depth_m, expected / 1000.0)
    assert valid.shape == (168, 224)

    camera_k = np.array([
        [300.0, 0.0, 160.0],
        [0.0, 280.0, 120.0],
        [0.0, 0.0, 1.0],
    ])
    scaled = intrinsic_for_resize(camera_k, 240, 320, 168, 224)
    np.testing.assert_allclose(
        scaled, np.array([
            [210.0, 0.0, 112.0],
            [0.0, 196.0, 84.0],
            [0.0, 0.0, 1.0],
        ]))

def test_dino_loader_and_track_grid_are_12x16(tmp_path: Path):
    rgb_path = tmp_path / "rgb.png"
    cv2.imwrite(str(rgb_path), np.zeros((240, 320, 3), np.uint8))
    image = _load_rgb_tensor(str(rgb_path), 168, 224)
    assert image.shape == (3, 168, 224)

    seed = seed_grid_pixels(168, 224)
    assert seed.shape == (192, 2)
    np.testing.assert_array_equal(seed[0], [7, 7])
    np.testing.assert_array_equal(seed[-1], [217, 161])


class _StationaryTracker:
    def __call__(self, clips, queries):
        batch, time = clips.shape[:2]
        xy = queries[..., 1:].unsqueeze(1).expand(batch, time, -1, -1)
        visible = torch.ones(xy.shape[:-1], device=xy.device, dtype=torch.bool)
        return xy, visible


def test_track_processing_emits_192_points():
    frames = torch.zeros((33, 3, 168, 224), dtype=torch.uint8)
    tracks, valid, seed, offset_valid = process_demo(
        _StationaryTracker(), frames, anchor_batch=1, device="cpu",
        image_height=168, image_width=224)
    assert tracks.shape == (5, 4, 192, 3)
    assert valid.tolist() == [True, False, False, False, False]
    assert seed.shape == (192, 2)
    assert tracks[0, 0, 0, 0] == pytest.approx(7 / 223 * 2 - 1, abs=1e-3)
    assert tracks[0, 0, 0, 1] == pytest.approx(7 / 167 * 2 - 1, abs=1e-3)


def test_tail_anchors_keep_the_offsets_that_still_fit():
    """Anchors without the full 32-frame horizon used to be dropped entirely,
    which is what made the end of every demo unsamplable by the loader."""
    frames = torch.zeros((33, 3, 168, 224), dtype=torch.uint8)
    _, _, _, offset_valid = process_demo(
        _StationaryTracker(), frames, anchor_batch=1, device="cpu",
        image_height=168, image_width=224)
    # Keyframes at native 0, 8, 16, 24, 32; offsets +8/+16/+24/+32 must land
    # on or before frame 32.
    assert offset_valid.sum(axis=1).tolist() == [4, 3, 2, 1, 0]
    assert offset_valid[1].tolist() == [True, True, True, False]


def _write_rectangular_demo(root: Path) -> Path:
    demo = root / "task" / "demo_000000"
    demo.mkdir(parents=True)
    frames = []
    for i in range(9):
        stem = f"{i:04d}"
        rgb_name = f"rgb_{stem}.png"
        depth_name = f"depth_{stem}.npy"
        state_name = f"state_{stem}.json"
        cv2.imwrite(str(demo / rgb_name), np.zeros((168, 224, 3), np.uint8))
        np.save(demo / depth_name, np.ones((168, 224), np.float32))
        (demo / state_name).write_text(json.dumps({
            "action": [float(i)],
            "proprio": [float(i)],
        }))
        torch.save(torch.zeros((192, 384), dtype=torch.float16),
                   demo / f"dino_tokens_{stem}.pth")
        frames.append({
            "rgb": rgb_name, "depth_npy": depth_name,
            "state_file": state_name,
        })
    (demo / "metadata.json").write_text(json.dumps({
        "task": "task",
        "frames": frames,
        "action_dim": 1,
        "proprio_dim": 1,
    }))
    return demo


def test_packer_infers_rectangular_token_and_track_geometry(tmp_path: Path):
    src = _write_rectangular_demo(tmp_path / "src")
    tracks_demo = tmp_path / "tracks" / "task" / "demo_000000"
    tracks_demo.mkdir(parents=True)
    np.savez_compressed(
        tracks_demo / "tracks.npz",
        tracks=np.zeros((2, 4, 192, 3), np.float16),
        valid=np.array([True, False]),
        seed_xy=np.zeros((192, 2), np.float32),
        grid=12, grid_height=12, grid_width=16,
    )
    dst = tmp_path / "packed" / "task" / "demo_000000"
    assert pack_one(src, tracks_demo, dst, 10.0, 95, False) == (str(src), "ok")

    pack = torch.load(dst / "trajectory.pt", weights_only=False)
    assert pack["dino_tokens"].shape == (2, 192, 384)
    assert (pack["image_height"], pack["image_width"]) == (168, 224)
    assert (pack["grid_height"], pack["grid_width"]) == (12, 16)
    assert (pack["track_grid_height"], pack["track_grid_width"]) == (12, 16)
    assert pack["point_tracks"].shape == (2, 4, 192, 3)
