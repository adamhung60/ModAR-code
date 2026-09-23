from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from util.modality_forcing.config import MFConfig
from util.modality_forcing.dit import Attention
from util.modality_forcing.model import (
    ModalityForcingWAM,
    load_spatially_compatible_state_dict,
)
from util.modality_forcing.tokenizers import (
    patchify_depth,
    patchify_rgb,
    unpatchify_depth,
    unpatchify_rgb,
)


def tiny_cfg(use_cache=True, mode="modar",
             modalities=("dino", "depth", "action")):
    generation_order = tuple(m for m in modalities if m != "action")
    if "action" in modalities:
        generation_order += ("action",)
    return MFConfig(
        schedule_mode=mode,
        modalities=modalities,
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
        generation_order=generation_order,
        history_modalities=("dino", "depth"),
        use_kv_cache=use_cache,
    )


def make_batch(cfg, batch_size=2):
    return {
        "dino": torch.randn(
            batch_size, cfg.n_obs_frames, cfg.n_patches, cfg.dino_dim),
        "depth_maps": torch.randn(
            batch_size, cfg.n_obs_frames,
            cfg.depth_img_size, cfg.depth_img_size),
        "actions": torch.randn(
            batch_size, cfg.action_horizon, cfg.action_dim),
        "proprio": torch.randn(batch_size, cfg.proprio_dim),
        "images": torch.randn(
            batch_size, cfg.n_obs_frames, cfg.image_channels,
            cfg.depth_img_size, cfg.depth_img_size),
        "point_tracks": torch.randn(
            batch_size, cfg.obs_future, cfg.n_patches, cfg.track_dim),
    }


def rectangular_cfg():
    return MFConfig(
        schedule_mode="modar",
        modalities=("dino", "depth", "image", "tracks"),
        dim=32,
        n_heads=2,
        rope_split=(8, 4, 4),
        n_shared_layers=1,
        n_expert_layers=1,
        depth_patch_size=2,
        image_patch_size=2,
        depth_img_size=4,
        image_height=6,
        image_width=8,
        grid=2,
        grid_height=3,
        grid_width=4,
        track_grid=2,
        track_grid_height=3,
        track_grid_width=4,
        dino_dim=8,
        action_dim=3,
        proprio_dim=4,
        obs_history=1,
        obs_future=1,
        action_horizon=2,
        steps_per_phase=1,
        generation_order=("dino", "depth", "image", "tracks"),
        history_modalities=("dino", "depth", "image"),
    )


def test_rectangular_patchify_round_trip():
    depth = torch.randn(2, 3, 6, 8)
    rgb = torch.randn(2, 3, 3, 6, 8)
    depth_tokens = patchify_depth(depth, 2)
    rgb_tokens = patchify_rgb(rgb, 2)
    assert depth_tokens.shape == (2, 3, 12, 4)
    assert rgb_tokens.shape == (2, 3, 12, 12)
    torch.testing.assert_close(
        unpatchify_depth(depth_tokens, 2, 3, 4), depth)
    torch.testing.assert_close(
        unpatchify_rgb(rgb_tokens, 2, 3, 4, channels=3), rgb)


def test_rectangular_model_forward_and_sample():
    cfg = rectangular_cfg()
    model = ModalityForcingWAM(cfg)
    batch = {
        "dino": torch.randn(2, cfg.n_obs_frames, cfg.n_patches, cfg.dino_dim),
        "depth_maps": torch.randn(
            2, cfg.n_obs_frames, cfg.spatial_height, cfg.spatial_width),
        "images": torch.randn(
            2, cfg.n_obs_frames, 3, cfg.spatial_height, cfg.spatial_width),
        "point_tracks": torch.randn(
            2, cfg.obs_future, cfg.n_patches, cfg.track_dim),
        "actions": torch.zeros(2, cfg.action_horizon, cfg.action_dim),
        "proprio": torch.zeros(2, cfg.proprio_dim),
    }
    out = model(**batch, stream="dynamics")
    assert torch.isfinite(out["loss"])
    generated = model.sample(
        batch["dino"][:, :1], batch["depth_maps"][:, :1],
        batch["proprio"], image_hist=batch["images"][:, :1])
    assert generated["depth_maps"].shape[-2:] == (6, 8)
    assert generated["images"].shape[-2:] == (6, 8)
    assert generated["dino"].shape[-2] == 12
    assert generated["point_tracks"].shape[-2] == 12


def test_rectangular_model_loads_legacy_square_checkpoint():
    square = ModalityForcingWAM(tiny_cfg(modalities=("dino", "depth")))
    legacy = dict(square.state_dict())
    legacy["pos_t"] = square.pos_t.clone()
    legacy["pos_h"] = square.pos_h.clone()
    legacy["pos_w"] = square.pos_w.clone()
    rect_cfg = copy.deepcopy(square.cfg)
    rect_cfg.image_height = 4
    rect_cfg.image_width = 6
    rect_cfg.grid_height = 2
    rect_cfg.grid_width = 3
    rect = ModalityForcingWAM(rect_cfg)
    load_spatially_compatible_state_dict(rect, legacy)
    for name, value in square.state_dict().items():
        torch.testing.assert_close(rect.state_dict()[name], value)


def prepared(model, batch):
    return model._prep_data(
        batch["dino"], batch["depth_maps"], batch["actions"])


def per_stream_copies(module):
    """A trunk submodule's copies: one per MoT stream, or one when dense."""
    return (list(module.values()) if isinstance(module, nn.ModuleDict)
            else [module])


def enable_nonzero_adaln(model):
    with torch.no_grad():
        for block in list(model.dit.shared) + [
                b for stack in model.dit.experts.values() for b in stack]:
            for ada in per_stream_copies(block.adaLN):
                ada[-1].weight.normal_(std=0.03)
                ada[-1].bias.normal_(std=0.03)


def test_block_mask_matches_documented_matrix():
    roles = [
        ("history", -1), ("clean", 0), ("clean", 1),
        ("query", 0), ("query", 1), ("query", 2),
    ]
    blocks = []
    for index, (role, step) in enumerate(roles):
        blocks.append({
            "role": role,
            "step": step,
            "start": index,
            "end": index + 1,
            "tokens": torch.zeros(1, 1, 1),
        })
    actual = ModalityForcingWAM._build_causal_mask(blocks)
    expected = torch.tensor([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 1],
    ], dtype=torch.bool)
    assert torch.equal(actual, expected)
    assert actual.any(dim=1).all()


def test_token_mask_expands_unequal_blocks():
    lengths = [3, 2, 4, 1]
    roles = [("history", -1), ("clean", 0), ("query", 0), ("query", 1)]
    blocks, cursor = [], 0
    for length, (role, step) in zip(lengths, roles):
        blocks.append({
            "role": role, "step": step, "start": cursor,
            "end": cursor + length, "tokens": torch.zeros(1, length, 1),
        })
        cursor += length
    mask = ModalityForcingWAM._build_causal_mask(blocks)
    assert mask.shape == (sum(lengths), sum(lengths))
    assert mask[3:5, :5].all()                    # C1 sees H and itself
    assert not mask[5:9, 3:5].any()              # Q1 cannot see C1
    assert mask[9:10, :5].all()                  # Q2 sees H and C1
    assert not mask[9:10, 5:9].any()             # Q2 cannot see Q1
    assert mask[9:10, 9:10].all()


def test_modar_training_uses_fixed_full_prefix():
    model = ModalityForcingWAM(tiny_cfg(mode="modar")).train()
    device = torch.device("cpu")
    dynamics_order, dynamics_queries = model._causal_training_order(
        "dynamics", device)
    action_order, action_queries = model._causal_training_order("action", device)
    assert dynamics_order == ["dino", "depth"]
    assert dynamics_queries == ["dino", "depth"]
    assert action_order == ["dino", "depth", "action"]
    assert action_queries == ["action"]


def test_modar_trains_only_configured_generation_modalities():
    cfg = tiny_cfg(modalities=("dino", "depth", "tracks"))
    cfg.generated_modalities = ("tracks",)
    cfg.generation_order = ("tracks",)
    model = ModalityForcingWAM(cfg).train()
    order, queries = model._causal_training_order("dynamics", torch.device("cpu"))
    assert order == ["tracks"]
    assert queries == ["tracks"]


def test_double_sequence_matches_individual_cut_forwards():
    torch.manual_seed(4)
    model = ModalityForcingWAM(tiny_cfg()).eval()
    enable_nonzero_adaln(model)
    batch = make_batch(model.cfg)
    data = prepared(model, batch)
    order = ["dino", "depth", "action"]
    times, query_zs = {}, {}
    for name in order:
        times[name] = torch.rand(batch["proprio"].shape[0])
        target = model._causal_future(data, name)
        query_zs[name] = model.scheduler.add_noise(
            target, times[name], torch.randn_like(target))

    together = model._run_causal(
        data, batch["proprio"], order, order, query_zs, times)
    hist = {
        name: data[name][:, model.hist_local[name]]
        for name in model.grid_names if model.has_history[name]
    }
    generated = {}
    for name in order:
        separate = model._causal_uncached_prediction(
            hist, generated, name, query_zs[name], times[name],
            batch["proprio"], None)
        torch.testing.assert_close(together[name], separate, atol=2e-6, rtol=2e-6)
        generated[name] = model._causal_future(data, name)


def test_forbidden_blocks_have_zero_query_gradient():
    torch.manual_seed(9)
    model = ModalityForcingWAM(tiny_cfg()).eval()
    enable_nonzero_adaln(model)
    batch = make_batch(model.cfg, batch_size=1)
    data = prepared(model, batch)
    order = ["dino", "depth", "action"]
    times = {name: torch.rand(1) for name in order}
    query_zs = {
        name: torch.randn_like(model._causal_future(data, name))
        for name in order
    }
    (tokens, pt, ph, pw, blocks, mask,
     c_tokens, query_conds) = model._assemble_causal(
        data, batch["proprio"], order, order, query_zs, times)
    tokens = tokens.detach().requires_grad_(True)
    query_blocks = [b for b in blocks if b["role"] == "query"]
    segments = [(b["name"], b["start"], b["end"]) for b in query_blocks]
    out = model.dit(
        tokens, c_tokens, pt, ph, pw, segments, mask,
        segment_conds=query_conds)
    out["depth"].square().sum().backward()

    grad = tokens.grad.abs().sum(dim=(0, 2))
    for block in blocks:
        block_grad = grad[block["start"]:block["end"]].sum()
        allowed = (
            block["role"] == "history"
            or (block["role"] == "clean" and block["step"] < 1)
            or (block["role"] == "query" and block["name"] == "depth")
        )
        if allowed:
            assert block_grad > 0, block
        else:
            assert block_grad == 0, block


def test_attention_cached_query_matches_full_rectangular_mask():
    torch.manual_seed(13)
    cfg = tiny_cfg()
    attention = Attention(cfg).eval()
    prefix = torch.randn(2, 5, cfg.dim)
    query = torch.randn(2, 3, cfg.dim)
    pt = torch.tensor([0, 0, 1, 1, 2, 3, 3, 4])
    ph = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])
    pw = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    full = torch.cat((prefix, query), dim=1)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:5, :5] = True
    mask[5:, :] = True
    full_out = attention(full, pt, ph, pw, mask)[:, 5:]

    _, cache = attention.forward_cached(
        prefix, pt[:5], ph[:5], pw[:5], None, append=True)
    cached_out, unchanged = attention.forward_cached(
        query, pt[5:], ph[5:], pw[5:], cache, append=False)
    torch.testing.assert_close(full_out, cached_out, atol=1e-6, rtol=1e-6)
    assert unchanged is cache


def test_boolean_sdpa_mask_matches_manual_attention():
    torch.manual_seed(14)
    cfg = tiny_cfg()
    attention = Attention(cfg).eval()
    x = torch.randn(2, 6, cfg.dim)
    pt = torch.tensor([0, 0, 1, 2, 2, 3])
    ph = torch.tensor([0, 1, 0, 1, 0, 1])
    pw = torch.tensor([0, 0, 1, 0, 1, 1])
    mask = torch.tensor([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1],
    ], dtype=torch.bool)
    actual = attention(x, pt, ph, pw, mask)
    q, k, v = attention._project_qkv(x, pt, ph, pw)
    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
    logits = logits.masked_fill(~mask, float("-inf"))
    manual = torch.matmul(logits.softmax(dim=-1), v)
    manual = attention.proj(manual.transpose(1, 2).reshape(2, 6, cfg.dim))
    torch.testing.assert_close(actual, manual, atol=1e-6, rtol=1e-6)


def test_query_does_not_mutate_prefix_cache():
    torch.manual_seed(15)
    model = ModalityForcingWAM(tiny_cfg()).eval()
    enable_nonzero_adaln(model)
    batch = make_batch(model.cfg)
    hist = {
        "dino": batch["dino"][:, :1],
        "depth": prepared(model, batch)["depth"][:, :1],
    }
    cache = model._causal_prefill_history(hist, batch["proprio"], None)
    before = [(k.clone(), v.clone()) for k, v in cache]
    z = torch.randn(2, 1, model.cfg.n_patches, model.cfg.dino_dim)
    model._causal_query_prediction(
        "dino", z, torch.full((2,), 0.4), batch["proprio"], None, cache)
    for (old_k, old_v), (new_k, new_v) in zip(before, cache):
        assert torch.equal(old_k, new_k)
        assert torch.equal(old_v, new_v)


def test_shared_qkv_projects_prefix_only_during_prefill_and_append():
    torch.manual_seed(17)
    model = ModalityForcingWAM(tiny_cfg()).eval()
    batch = make_batch(model.cfg)
    data = prepared(model, batch)
    hist = {
        "dino": batch["dino"][:, :1],
        "depth": data["depth"][:, :1],
    }
    lengths = [[] for _ in model.dit.shared]
    hooks = []
    for index, block in enumerate(model.dit.shared):
        for qkv in per_stream_copies(block.attn.qkv):
            hooks.append(qkv.register_forward_pre_hook(
                lambda _module, args, i=index: lengths[i].append(args[0].shape[1])))

    cache = model._causal_prefill_history(hist, batch["proprio"], None)
    z = torch.randn(2, 1, model.cfg.n_patches, model.cfg.dino_dim)
    time = torch.full((2,), 0.3)
    model._causal_query_prediction(
        "dino", z, time, batch["proprio"], None, cache)
    model._causal_query_prediction(
        "dino", z, time, batch["proprio"], None, cache)
    cache = model._causal_append_clean(
        "dino", z, 0, batch["proprio"], None, cache)
    z_depth = torch.randn_like(model._causal_future(data, "depth"))
    model._causal_query_prediction(
        "depth", z_depth, time, batch["proprio"], None, cache)
    for hook in hooks:
        hook.remove()

    history_tokens = 2 * model.cfg.n_patches
    query_tokens = model.cfg.n_patches
    for observed in lengths:
        assert observed == [
            history_tokens, query_tokens, query_tokens,
            query_tokens, query_tokens,
        ]


def test_cached_and_uncached_rollouts_match():
    torch.manual_seed(21)
    cached = ModalityForcingWAM(tiny_cfg(use_cache=True)).eval()
    enable_nonzero_adaln(cached)
    batch = make_batch(cached.cfg)
    args = (
        batch["dino"][:, :1],
        batch["depth_maps"][:, :1],
        batch["proprio"],
    )
    for solver in ("euler", "heun"):
        cached.cfg.solver = solver
        uncached = copy.deepcopy(cached)
        uncached.cfg.use_kv_cache = False
        torch.manual_seed(22)
        expected = uncached.sample(*args)
        torch.manual_seed(22)
        actual = cached.sample(*args)
        assert expected.keys() == actual.keys()
        for name in expected:
            torch.testing.assert_close(
                actual[name], expected[name], atol=3e-6, rtol=3e-6)


def test_oracle_rollout_consumes_the_same_noise_as_self_generation():
    """Teacher-forced dynamics must not shift the action's noise draw.

    The oracle skips the dynamics ODEs, so without a matching (discarded) draw
    the action would start from different noise than the self-generated pass
    and oracle-minus-self-gen would measure noise on top of conditioning.
    Equal RNG state after both rollouts is that invariant.
    """
    torch.manual_seed(31)
    model = ModalityForcingWAM(tiny_cfg()).eval()
    batch = make_batch(model.cfg)
    hist_args = (batch["dino"][:, :1], batch["depth_maps"][:, :1])
    hist = model._build_history(*hist_args, None)
    data = model._prep_data(
        batch["dino"], batch["depth_maps"], batch["actions"])

    torch.manual_seed(40)
    model.sample(*hist_args, batch["proprio"])
    after_self_gen = torch.get_rng_state()

    torch.manual_seed(40)
    model.sample_action_oracle(hist, batch["proprio"], data)
    assert torch.equal(torch.get_rng_state(), after_self_gen)


def test_unified_oracle_action_starts_from_the_same_noise_as_self_generation():
    """Same invariant as above, for unified, which cannot state it as RNG parity.

    Pinning the dynamics to the GT flow trajectory needs its own ``randn_like``
    per modality, drawn *after* the latents, so the two paths necessarily end on
    different RNG states. The weaker sufficient condition is what the oracle
    comparison actually rests on: the action's z0 comes out of
    ``_init_full_latents``, which runs first in both paths, so the
    oracle-minus-self-gen difference is conditioning and not two noise draws.
    """
    torch.manual_seed(31)
    cfg = tiny_cfg(mode="unified")
    model = ModalityForcingWAM(cfg).eval()
    batch = make_batch(cfg)
    hist_args = (batch["dino"][:, :1], batch["depth_maps"][:, :1])
    hist = model._build_history(*hist_args, None)
    data = model._prep_data(
        batch["dino"], batch["depth_maps"], batch["actions"])

    seen = []
    original = model._init_full_latents

    def capture(*args, **kwargs):
        # Clone: both samplers write their ODE state back into this dict, so
        # holding the reference would record the FINAL action, not its z0.
        z = original(*args, **kwargs)
        seen.append(z["action"].clone())
        return z

    model._init_full_latents = capture
    torch.manual_seed(40)
    model.sample(*hist_args, batch["proprio"])
    torch.manual_seed(40)
    model.sample_action_oracle(hist, batch["proprio"], data)

    assert len(seen) == 2
    torch.testing.assert_close(seen[0], seen[1])


def test_cached_action_without_history_prefix_matches_reference():
    cfg = tiny_cfg(use_cache=True, modalities=("action",))
    cfg.history_modalities = ()
    cfg.generation_order = ("action",)
    cached = ModalityForcingWAM(cfg).eval()
    enable_nonzero_adaln(cached)
    uncached = copy.deepcopy(cached)
    uncached.cfg.use_kv_cache = False
    batch = make_batch(cfg)
    args = (
        batch["dino"][:, :1],
        batch["depth_maps"][:, :1],
        batch["proprio"],
    )
    torch.manual_seed(24)
    expected = uncached.sample(*args)
    torch.manual_seed(24)
    actual = cached.sample(*args)
    torch.testing.assert_close(
        actual["actions"], expected["actions"], atol=2e-6, rtol=2e-6)


def test_two_stream_loss_contracts():
    torch.manual_seed(30)
    model = ModalityForcingWAM(tiny_cfg()).train()
    batch = make_batch(model.cfg)
    # A joint source supervises each fixed-order dynamics cut and the action.
    joint = model(**batch, stream=None)
    assert joint["n_action"] == 2
    supervised = [name for name in ("dino", "depth", "action")
                  if float(joint[f"n_{name}"]) > 0]
    joint_expected = sum(
        model.cfg.lambda_of(name) * joint[f"loss_{name}"]
        for name in supervised) / len(supervised)
    torch.testing.assert_close(joint["loss"].detach(), joint_expected)

    dynamics = model(**batch, stream="dynamics")
    assert dynamics["n_dino"] == 2
    assert dynamics["n_depth"] == 2
    assert dynamics["n_action"] == 0
    expected = (
        model.cfg.lambda_dino * dynamics["loss_dino"]
        + model.cfg.lambda_depth * dynamics["loss_depth"]
    ) / 2
    torch.testing.assert_close(dynamics["loss"].detach(), expected)

    action = model(**batch, stream="action")
    assert action["n_dino"] == 0
    assert action["n_depth"] == 0
    assert action["n_action"] == 2
    torch.testing.assert_close(
        action["loss"].detach(),
        model.cfg.lambda_action * action["loss_action"])


@pytest.mark.parametrize("modalities", [
    ("dino", "depth", "action"),
    ("dino", "tracks", "depth", "action"),
    ("dino", "tracks", "depth", "image", "action"),
])
def test_three_four_and_five_modality_smoke_forwards(modalities):
    cfg = tiny_cfg(modalities=modalities)
    cfg.generation_order = modalities
    model = ModalityForcingWAM(cfg).train()
    batch = make_batch(cfg, batch_size=1)
    for stream in ("dynamics", "action"):
        out = model(**batch, stream=stream)
        assert torch.isfinite(out["loss"])


def test_disjoint_still_runs_one_reduced_sequence_expert():
    torch.manual_seed(31)
    model = ModalityForcingWAM(tiny_cfg(mode="disjoint")).train()
    out = model(**make_batch(model.cfg))
    active = [
        name for name in model.cfg.modalities if float(out[f"n_{name}"]) > 0]
    assert len(active) == 1



