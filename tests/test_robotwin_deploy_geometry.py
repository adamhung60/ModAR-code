from types import SimpleNamespace

import numpy as np
import pytest
import torch

import robotwin_manip.eval.robotwin_deploy as deploy


class _DinoStub:
    def __init__(self):
        self.rgb = None

    def __call__(self, rgb):
        self.rgb = rgb
        return torch.zeros((192, 384))


def _checkpoint(monkeypatch, *, height=168, width=224, grid_h=12, grid_w=16):
    cfg = SimpleNamespace(
        spatial_height=height,
        spatial_width=width,
        grid_h=grid_h,
        grid_w=grid_w,
        action_horizon=16,
        obs_history=1,
        n_tasks=0,
    )
    model = SimpleNamespace(cfg=SimpleNamespace(generation_order=["action"]))
    stats = {
        "action_mean": torch.zeros(14),
        "action_std": torch.ones(14),
        "proprio_mean": torch.zeros(14),
        "proprio_std": torch.ones(14),
        "dino_mean": 0.0,
        "dino_std": 1.0,
    }
    data_cfg = {
        "depth_mean": 0.0,
        "depth_std": 1.0,
        "depth_max_m": 10.0,
        "depth_norm_mode": "log",
    }
    monkeypatch.setattr(
        deploy, "build_mf_from_checkpoint",
        lambda *_args, **_kwargs: (model, cfg, stats, data_cfg),
    )


def test_rollout_observation_uses_uncropped_224x168(monkeypatch):
    _checkpoint(monkeypatch)
    dino = _DinoStub()
    monkeypatch.setattr(deploy, "DinoEncoder", lambda *_args, **_kwargs: dino)
    policy = deploy.WAMPolicy("unused.pt", device="cpu")
    assert policy.exec_horizon == 16

    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:, 0, 0] = 255
    rgb[:, -1, 1] = 255
    observation = {
        "observation": {
            "head_camera": {
                "rgb": rgb,
                "depth": np.full((240, 320), 1000, dtype=np.float32),
            },
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }

    tokens, depth, image, proprio = policy._features(observation)
    assert dino.rgb.shape == (168, 224, 3)
    assert dino.rgb[:, 0, 0].max() > 0
    assert dino.rgb[:, -1, 1].max() > 0
    assert tokens.shape == (192, 384)
    assert depth.shape == (168, 224)
    assert image.shape == (3, 168, 224)
    expected_image = (
        torch.from_numpy(dino.rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    )
    assert torch.equal(image, expected_image)
    assert proprio.shape == (14,)


def test_rollout_exec_horizon_allows_diagnostic_override(monkeypatch):
    _checkpoint(monkeypatch)
    monkeypatch.setattr(deploy, "DinoEncoder", lambda *_args, **_kwargs: _DinoStub())
    policy = deploy.WAMPolicy("unused.pt", device="cpu", exec_horizon=8)
    assert policy.exec_horizon == 8


def test_rollout_rejects_obsolete_square_checkpoint(monkeypatch):
    _checkpoint(monkeypatch, height=224, width=224, grid_h=16, grid_w=16)
    monkeypatch.setattr(deploy, "DinoEncoder", lambda *_args, **_kwargs: _DinoStub())
    with pytest.raises(ValueError, match="obsolete"):
        deploy.WAMPolicy("unused.pt", device="cpu")
