"""Conditioning-only dynamics modalities (``generated_modalities``).

A single-target arm keeps every sensor in the observation context but denoises
only one dynamics modality, so the arms differ in world-model target and nothing
else. Restricting ``modalities`` cannot express that: it would also strip the
history those other sensors provide.
"""
from __future__ import annotations

import pytest
import torch

from util.modality_forcing.config import MFConfig
from util.modality_forcing.grad_budget import realized_shares, resolve_loss_coeffs
from util.modality_forcing.model import ModalityForcingWAM

FULL = ("dino", "depth", "image", "action")


def tiny_cfg(generate=None, mode="modar", modalities=FULL,
             history=("dino", "depth", "image")):
    return MFConfig(
        schedule_mode=mode,
        modalities=modalities,
        generated_modalities=generate,
        dim=32,
        n_heads=2,
        rope_split=(8, 4, 4),
        n_shared_layers=2,
        n_expert_layers=1,
        grid=2,
        depth_img_size=4,
        depth_patch_size=2,
        image_patch_size=2,
        track_grid=2,
        dino_dim=8,
        action_dim=3,
        proprio_dim=4,
        obs_history=1,
        obs_future=1,
        action_horizon=2,
        steps_per_phase=2,
        generation_order=("dino", "tracks", "image", "depth", "action"),
        history_modalities=history,
    )


def make_batch(cfg, batch_size=2):
    return {
        "dino": torch.randn(
            batch_size, cfg.n_obs_frames, cfg.n_patches, cfg.dino_dim),
        "depth_maps": torch.randn(
            batch_size, cfg.n_obs_frames,
            cfg.depth_img_size, cfg.depth_img_size),
        "images": torch.randn(
            batch_size, cfg.n_obs_frames, cfg.image_channels,
            cfg.depth_img_size, cfg.depth_img_size),
        "point_tracks": torch.randn(
            batch_size, cfg.obs_future, cfg.n_patches, cfg.track_dim),
        "actions": torch.randn(
            batch_size, cfg.action_horizon, cfg.action_dim),
        "proprio": torch.randn(batch_size, cfg.proprio_dim),
    }


def test_default_generates_every_present_dynamics_modality():
    cfg = tiny_cfg(generate=None)
    assert cfg.generated_modality_names() == cfg.grid_modalities()
    assert cfg.history_only_modalities() == []


def test_named_subset_splits_generated_from_conditioning_only():
    cfg = tiny_cfg(generate=("image",))
    assert cfg.generated_modality_names() == ["image"]
    assert cfg.history_only_modalities() == ["dino", "depth"]


def test_modar_order_covers_only_the_generated_modality():
    model = ModalityForcingWAM(tiny_cfg(generate=("image",)))
    order, query_names = model._causal_training_order(None, torch.device("cpu"))
    assert order == ["image", "action"]
    assert query_names == ["image", "action"]


def test_modar_completeness_check_ignores_conditioning_only_modalities():
    # ModAR rejects a *generated* modality missing from the order; a
    # conditioning-only one is absent by design and must not trip that check.
    model = ModalityForcingWAM(tiny_cfg(generate=("depth",)))
    order, _ = model._causal_training_order(None, torch.device("cpu"))
    assert order == ["depth", "action"]


def test_only_the_generated_modality_carries_loss():
    model = ModalityForcingWAM(tiny_cfg(generate=("image",)))
    out = model(**make_batch(model.cfg), stream=None)
    assert torch.isfinite(out["loss"])
    assert float(out["loss_image"]) > 0
    assert float(out["loss_action"]) > 0
    assert float(out["loss_dino"]) == 0
    assert float(out["loss_depth"]) == 0


def break_zero_init(model):
    """Make the DiT input-sensitive.

    The blocks use adaLN-zero, so a freshly built model emits a constant
    regardless of its input; any wiring test on it would pass vacuously.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    return model


def test_conditioning_only_history_still_reaches_the_forward():
    """The held-out sensors must actually condition the run, not just be loaded."""
    model = break_zero_init(
        ModalityForcingWAM(tiny_cfg(generate=("image",))).eval())
    batch = make_batch(model.cfg)
    other = dict(batch)
    other["dino"] = batch["dino"] + 5.0

    torch.manual_seed(0)
    base = model(**batch, stream=None)["loss"]
    torch.manual_seed(0)
    perturbed = model(**other, stream=None)["loss"]
    assert not torch.allclose(base, perturbed), (
        "dino is conditioning-only but changing it changed nothing, so its "
        "history never entered the sequence")


def test_history_blocks_include_conditioning_only_modalities():
    model = ModalityForcingWAM(tiny_cfg(generate=("image",)))
    batch = make_batch(model.cfg)
    data = model._prep_data(
        batch["dino"], batch["depth_maps"], batch["actions"],
        batch["images"], batch["point_tracks"])
    blocks = model._causal_history_blocks(data, training=True)
    assert {block["name"] for block in blocks} == {"dino", "depth", "image"}


def test_sampler_generates_only_the_named_modality():
    model = ModalityForcingWAM(tiny_cfg(generate=("image",))).eval()
    batch = make_batch(model.cfg)
    generated = model.sample(
        batch["dino"][:, :1], batch["depth_maps"][:, :1], batch["proprio"],
        image_hist=batch["images"][:, :1])
    assert "images" in generated and "actions" in generated
    assert "dino" not in generated and "depth_maps" not in generated


def test_tracks_target_keeps_the_full_visual_history():
    """Tracks are future-only, so a tracks arm has to borrow its history."""
    cfg = tiny_cfg(generate=("tracks",),
                   modalities=("dino", "depth", "image", "tracks", "action"))
    model = ModalityForcingWAM(cfg)
    order, query_names = model._causal_training_order(None, torch.device("cpu"))
    assert order == ["tracks", "action"]
    assert query_names == ["tracks", "action"]
    out = model(**make_batch(cfg), stream=None)
    assert float(out["loss_tracks"]) > 0
    assert float(out["loss_dino"]) == float(out["loss_image"]) == 0


def test_budget_holds_action_at_half_for_a_single_target():
    """The trainer must pass only supervised modalities to the resolver."""
    sources = [{"name": "action", "stream": "action", "weight": 1.0},
               {"name": "dyn", "stream": "dynamics", "weight": 1.0}]
    supervised = ("image", "action")
    coeffs = resolve_loss_coeffs("modar", supervised, sources)
    shares = realized_shares("modar", supervised, sources, coeffs)
    assert shares["action"] == pytest.approx(0.5)
    assert shares["image"] == pytest.approx(0.5)


def test_budget_would_drift_if_unsupervised_modalities_were_counted():
    """Why the trainer filters: unsupervised modalities silently inflate action."""
    sources = [{"name": "action", "stream": "action", "weight": 1.0},
               {"name": "dyn", "stream": "dynamics", "weight": 1.0}]
    coeffs = resolve_loss_coeffs("modar", FULL, sources)
    # Only image is actually denoised, so dino/depth contribute nothing and the
    # realized action share is 0.5 / (0.5 + 0.5/3), not 0.5.
    shares = realized_shares("modar", FULL, sources, coeffs)
    live = shares["action"] + shares["image"]
    assert shares["action"] / live == pytest.approx(0.75)


def test_unsupported_mode_is_rejected():
    with pytest.raises(AssertionError, match="only supported for ModAR"):
        tiny_cfg(generate=("image",), mode="unified")


def test_future_only_modality_cannot_be_conditioning_only():
    with pytest.raises(AssertionError, match="neither generated nor usable"):
        tiny_cfg(generate=("image",),
                 modalities=("dino", "depth", "image", "tracks", "action"))


def test_modality_absent_from_history_list_cannot_be_conditioning_only():
    with pytest.raises(AssertionError, match="neither generated nor usable"):
        tiny_cfg(generate=("image",), history=("dino",))


def test_unknown_generated_modality_is_rejected():
    with pytest.raises(AssertionError, match="not present grid modalities"):
        tiny_cfg(generate=("segmentation",))
