"""Context-robustness augmentations on the block-causal teacher blocks.

Three knobs sit on the same line of the pipeline -- the clean teacher block a
query conditions on -- and each is meant to break a different way the action
head can over-trust that block:

  cond_noise_label_t=False   corrupt the context but keep the t=1.0 label, so
                             the branch used at inference is the one trained
                             to be skeptical (Option B)
  token_noise_p              replace whole tokens with pure noise, so no single
                             patch or track can be leaned on
  modality_dropout_p         omit a whole future block from the ACTION query

cond_noise_beta defaults to 0.5 with label_t off (the deployment recipe).
token_noise_p and modality_dropout_p stay off unless a run asks for them.
"""
from __future__ import annotations

import torch

from util.modality_forcing.config import MFConfig
from util.modality_forcing.model import ModalityForcingWAM


def cfg(**kw):
    base = dict(
        schedule_mode="modar",
        modalities=("dino", "depth", "tracks", "action"),
        dim=32, n_heads=2, rope_split=(8, 4, 4),
        n_shared_layers=1, n_expert_layers=1,
        grid=2, depth_img_size=4, depth_patch_size=2, track_grid=2,
        dino_dim=8, action_dim=3, proprio_dim=4,
        obs_history=1, obs_future=1, action_horizon=2, steps_per_phase=1,
        generation_order=("dino", "depth", "tracks", "action"),
        history_modalities=("dino", "depth"),
    )
    base.update(kw)
    return MFConfig(**base)


def batch(c, b=2):
    return dict(
        dino=torch.randn(b, c.n_obs_frames, c.n_patches, c.dino_dim),
        depth_maps=torch.randn(b, c.n_obs_frames, c.depth_img_size,
                               c.depth_img_size),
        point_tracks=torch.randn(b, c.obs_future, c.n_patches, c.track_dim),
        actions=torch.randn(b, c.action_horizon, c.action_dim),
        proprio=torch.randn(b, c.proprio_dim),
    )


def clean_context(model, data, stream, seed=0):
    """The (name -> future tensor) actually handed to each clean teacher block.

    Intercepting at ``_causal_block`` reads the augmentation in the space it
    operates on, before embedding mixes tokens together and makes "this patch
    was replaced" unrecoverable.
    """
    seen = {}
    original = model._causal_block

    def spy(name, values, role, step):
        if role == "clean":
            seen[name] = values.detach().clone()
        return original(name, values, role, step)

    model._causal_block = spy
    try:
        torch.manual_seed(seed)
        model(**data, stream=stream)
    finally:
        model._causal_block = original
    return seen


def test_defaults_are_unlabelled_cond_noise():
    c = cfg()
    assert c.cond_noise_beta == 0.5 and c.cond_noise_label_t is False
    assert c.token_noise_p == 0.0 and c.modality_dropout_p == 0.0


def test_zero_beta_leaves_context_clean():
    c = cfg(cond_noise_beta=0.0)
    assert c.cond_noise_beta == 0.0 and c.token_noise_p == 0.0
    assert c.modality_dropout_p == 0.0
    model = ModalityForcingWAM(c).train()
    data = batch(c)
    seen = clean_context(model, data, "action")
    # Every dynamics future reaches the action query bit-for-bit intact.
    torch.testing.assert_close(seen["dino"], data["dino"][:, c.obs_history:])
    torch.testing.assert_close(seen["tracks"], data["point_tracks"])


def test_option_b_corrupts_context_without_relabelling_it():
    """beta alone must still corrupt; only the announced time changes."""
    data = batch(cfg())
    captured = {}
    for label in (True, False):
        c = cfg(cond_noise_beta=0.5, cond_noise_label_t=label)
        model = ModalityForcingWAM(c).train()
        blocks = []
        original = model._assemble_causal

        def spy(*a, **k):
            out = original(*a, **k)
            blocks.append(out[4])
            return out

        model._assemble_causal = spy
        captured[label] = (clean_context(model, data, "action"), blocks)

    for label in (True, False):
        seen, blocks = captured[label]
        # Corruption happens either way ...
        assert not torch.allclose(seen["dino"], data["dino"][:, 1:])
        # ... but only Option A hands the query the t_c that produced it.
        has_time = any("cond_time" in b for b in blocks[0]
                       if b["role"] == "clean")
        assert has_time is label


def test_token_noise_replaces_whole_tokens_with_pure_noise():
    c = cfg(cond_noise_beta=0.0, token_noise_p=0.5)
    model = ModalityForcingWAM(c).train()
    data = batch(c)
    seen = clean_context(model, data, "action")
    gt = data["dino"][:, c.obs_history:]
    # Per (frame, patch) token: either untouched, or entirely resampled -- a
    # partially-replaced token would mean the mask leaked into the feature axis.
    same = torch.isclose(seen["dino"], gt).all(dim=-1)
    touched = torch.isclose(seen["dino"], gt).any(dim=-1)
    assert bool((same == touched).all())
    assert bool(same.any()) and bool((~same).any())


def test_token_noise_at_p1_leaves_no_signal():
    c = cfg(cond_noise_beta=0.0, token_noise_p=1.0)
    model = ModalityForcingWAM(c).train()
    data = batch(c)
    seen = clean_context(model, data, "action")
    for name, gt in (("dino", data["dino"][:, c.obs_history:]),
                     ("tracks", data["point_tracks"])):
        assert not torch.isclose(seen[name], gt).any()


def test_modality_dropout_hits_action_stream_only():
    data = batch(cfg(cond_noise_beta=0.0))
    act = clean_context(
        ModalityForcingWAM(cfg(cond_noise_beta=0.0, modality_dropout_p=1.0)).train(),
        data, "action")
    assert act == {}
    # The dynamics cascade is exempt: at inference it always has every
    # preceding future, so dropping one there trains against nothing real.
    dyn = clean_context(
        ModalityForcingWAM(cfg(cond_noise_beta=0.0, modality_dropout_p=1.0)).train(),
        data, "dynamics")
    assert set(dyn) == {"dino", "depth"}
    torch.testing.assert_close(dyn["dino"], data["dino"][:, 1:])


def test_dropout_keeps_history_and_survives_losing_everything():
    c = cfg(cond_noise_beta=0.0, modality_dropout_p=1.0)
    model = ModalityForcingWAM(c).train()
    data = batch(c)
    names = []
    original = model._assemble_causal

    def spy(*a, **k):
        out = original(*a, **k)
        names.append([(b["name"], b["role"]) for b in out[4]])
        return out

    model._assemble_causal = spy
    out = model(**data, stream="action")
    assert torch.isfinite(out["loss"])
    # All futures gone, yet the query still reads the sensor history.
    assert names[0] == [("dino", "history"), ("depth", "history"),
                        ("action", "query")]


def test_all_augmentations_are_eval_time_no_ops():
    c = cfg(cond_noise_beta=0.5, cond_noise_label_t=False,
            token_noise_p=0.5, modality_dropout_p=0.5)
    model = ModalityForcingWAM(c).eval()
    data = batch(c)
    seen = clean_context(model, data, "action")
    assert set(seen) == {"dino", "depth", "tracks"}
    torch.testing.assert_close(seen["dino"], data["dino"][:, c.obs_history:])
    torch.testing.assert_close(seen["tracks"], data["point_tracks"])
