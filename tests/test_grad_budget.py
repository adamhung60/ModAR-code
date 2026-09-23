"""The action/dynamics gradient budget is identical across schedule modes.

The resolver in util.modality_forcing.grad_budget only produces a correct budget
if its supervision-probability table matches what the model actually supervises.
The load-bearing test here is therefore empirical: drive a real tiny model and
compare observed supervision rates against the table.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from scripts.train.train_modality_forcing import (
    expand_source_configs,
    load_config,
)
from util.modality_forcing.config import MFConfig
from util.modality_forcing.grad_budget import (
    describe_loss_coeffs,
    realized_shares,
    resolve_loss_coeffs,
    supervision_probs,
)
from util.modality_forcing.model import ModalityForcingWAM

THREE = ("dino", "depth", "action")
FIVE = ("dino", "tracks", "depth", "image", "action")

# RoboTwin co-training: a small action pool plus the big dynamics pool, equal weight.
ROBOTWIN = [
    {"name": "action", "stream": "action", "weight": 1.0},
    {"name": "dynamics", "stream": "dynamics", "weight": 1.0},
]
# scratch_actiononly trains a single source on the action pool.
ACTION_ONLY = [{"name": "action", "stream": None, "weight": 1.0}]
# Real-world setup: two dynamics-only datasets plus a joint robot source.
REAL_WORLD = [
    {"name": "egodex", "stream": "dynamics", "weight": 0.3},
    {"name": "human", "stream": "dynamics", "weight": 1.0},
    {"name": "robot", "stream": None, "weight": 1.0},
]
REAL_WORLD_ROBOT = [{"name": "robot", "stream": None, "weight": 1.0}]
REAL_WORLD_HUMAN_ROBOT = [
    {"name": "human", "stream": "dynamics", "weight": 1.0},
    {"name": "robot", "stream": None, "weight": 1.0},
]
# A dynamics-only pretraining source has no action labels.
DYNAMICS_ONLY = [{"name": "egodex", "stream": "dynamics", "weight": 1.0}]


def tiny_cfg(mode, modalities=THREE, uniform_p=False):
    """Smallest config that still exercises every forward path."""
    overrides = {}
    if uniform_p:
        # Mirrors cotrain_disjoint.yaml, which spreads p_* evenly over dynamics.
        share = 1.0 / len([m for m in modalities if m != "action"])
        overrides = {f"p_{name}": share
                     for name in modalities if name != "action"}
    return MFConfig(
        schedule_mode=mode,
        modalities=modalities,
        generation_order=modalities,
        history_modalities=tuple(
            name for name in ("dino", "depth", "image") if name in modalities),
        dim=32,
        n_heads=2,
        rope_split=(8, 4, 4),
        n_shared_layers=1,
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
        **overrides,
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


def dynamics_of(modalities):
    return [name for name in modalities if name != "action"]


# ---------------------------------------------------------------------------
# The table matches the model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,stream", [
    ("unified", None),
    ("unified", "action"),
    ("unified", "dynamics"),
    ("independent", None),
    ("independent", "action"),
    ("independent", "dynamics"),
    ("modar", None),
    ("modar", "action"),
    ("modar", "dynamics"),
    ("disjoint", None),
    ("disjoint", "action"),
    ("disjoint", "dynamics"),
    ("action_only", None),
    ("action_only", "action"),
])
def test_supervision_table_matches_model(mode, stream):
    """Observed supervision rates match supervision_probs for every pair."""
    torch.manual_seed(0)
    cfg = tiny_cfg(mode)
    model = ModalityForcingWAM(cfg).train()
    batch = make_batch(cfg)
    expected = supervision_probs(mode, cfg.modalities, stream, cfg.p_of)

    trials = 240
    counts = {name: 0 for name in cfg.modalities}
    with torch.no_grad():
        for _ in range(trials):
            out = model(**batch, stream=stream)
            for name in cfg.modalities:
                counts[name] += float(out[f"n_{name}"]) > 0

    for name in cfg.modalities:
        observed = counts[name] / trials
        if expected[name] in (0.0, 1.0):
            assert observed == expected[name], (
                f"{mode}/{stream} {name}: expected a deterministic "
                f"{expected[name]}, saw {observed}")
        else:
            assert observed == pytest.approx(expected[name], abs=0.08), (
                f"{mode}/{stream} {name}: expected ~{expected[name]}, "
                f"saw {observed}")


def test_modar_joint_source_supervises_every_modality():
    for modalities in (THREE, FIVE):
        cfg = tiny_cfg("modar", modalities)
        model = ModalityForcingWAM(cfg).train()
        order, query_names = model._causal_training_order(None, "cpu")
        assert order == list(modalities)
        assert query_names == list(modalities)


# ---------------------------------------------------------------------------
# The budget lands on target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [
    "unified", "independent", "modar", "disjoint"])
@pytest.mark.parametrize("modalities", [THREE, FIVE])
@pytest.mark.parametrize("sources", [
    ROBOTWIN, REAL_WORLD, REAL_WORLD_ROBOT, REAL_WORLD_HUMAN_ROBOT])
def test_action_takes_half_the_budget(mode, modalities, sources):
    cfg = tiny_cfg(mode, modalities, uniform_p=True)
    coeffs = resolve_loss_coeffs(mode, modalities, sources, p_of=cfg.p_of)
    shares = realized_shares(mode, modalities, sources, coeffs, cfg.p_of)

    dyn = dynamics_of(modalities)
    assert shares["action"] == pytest.approx(0.5)
    for name in dyn:
        assert shares[name] == pytest.approx(0.5 / len(dyn))
    assert sum(shares.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["unified", "modar", "disjoint"])
def test_custom_action_share_is_honored(mode):
    coeffs = resolve_loss_coeffs(
        mode, FIVE, ROBOTWIN, action_share=0.25,
        p_of=tiny_cfg(mode, FIVE, uniform_p=True).p_of)
    shares = realized_shares(
        mode, FIVE, ROBOTWIN, coeffs,
        tiny_cfg(mode, FIVE, uniform_p=True).p_of)

    assert shares["action"] == pytest.approx(0.25)
    for name in dynamics_of(FIVE):
        assert shares[name] == pytest.approx(0.75 / 4)


def test_action_only_run_spends_everything_on_action():
    coeffs = resolve_loss_coeffs("action_only", FIVE, ACTION_ONLY)
    shares = realized_shares("action_only", FIVE, ACTION_ONLY, coeffs)

    assert coeffs["action"] == {"action": 1.0}
    assert shares["action"] == pytest.approx(1.0)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_dynamics_only_run_spends_everything_on_dynamics():
    """A pretrain with no action labels keeps total mass at one, not 0.5."""
    coeffs = resolve_loss_coeffs("modar", FIVE, DYNAMICS_ONLY)
    shares = realized_shares("modar", FIVE, DYNAMICS_ONLY, coeffs)

    assert "action" not in coeffs["egodex"]
    assert shares["action"] == 0.0
    for name in dynamics_of(FIVE):
        assert shares[name] == pytest.approx(0.25)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_dynamics_budget_splits_across_datasets_by_source_weight():
    """The dataset mixture of the dynamics half tracks the configured weights."""
    coeffs = resolve_loss_coeffs("modar", FIVE, REAL_WORLD)
    total_weight = 0.3 + 1.0 + 1.0
    per_modality = 0.5 / 4

    for name, weight in (("egodex", 0.3), ("human", 1.0), ("robot", 1.0)):
        prob = supervision_probs("modar", FIVE, dict(
            egodex="dynamics", human="dynamics", robot=None)[name])
        contribution = (weight / total_weight) * prob["dino"] * coeffs[name]["dino"]
        assert contribution == pytest.approx(per_modality * weight / total_weight)


# ---------------------------------------------------------------------------
# Modes whose objective must not move
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["unified", "modar"])
@pytest.mark.parametrize("modalities", [THREE, FIVE])
def test_two_stream_coeffs_reproduce_mean_aggregation(mode, modalities):
    """Split action/dynamics streams use the same balanced objective.

    It is ``lambda_a * L_a`` on the action stream and a plain mean over dynamics
    modalities on the dynamics stream.
    """
    dyn = dynamics_of(modalities)
    coeffs = resolve_loss_coeffs(mode, modalities, ROBOTWIN)

    assert coeffs["action"] == {"action": pytest.approx(1.0)}
    assert coeffs["dynamics"] == {
        name: pytest.approx(1.0 / len(dyn)) for name in dyn}


@pytest.mark.parametrize("modalities", [THREE, FIVE])
def test_disjoint_coeffs_reproduce_single_active_term(modalities):
    """Disjoint's historical objective was one unscaled active modality."""
    cfg = tiny_cfg("disjoint", modalities, uniform_p=True)
    coeffs = resolve_loss_coeffs("disjoint", modalities, ROBOTWIN, p_of=cfg.p_of)

    assert coeffs["action"] == {"action": pytest.approx(1.0)}
    for name in dynamics_of(modalities):
        assert coeffs["dynamics"][name] == pytest.approx(1.0)


def test_dynamics_only_pretrain_coeffs_reproduce_mean_aggregation():
    coeffs = resolve_loss_coeffs("modar", FIVE, DYNAMICS_ONLY)
    assert coeffs["egodex"] == {
        name: pytest.approx(0.25) for name in dynamics_of(FIVE)}


def test_real_world_joint_source_scales_action_up():
    """The robot source carries all the action mass at 0.435 of the total weight."""
    coeffs = resolve_loss_coeffs("modar", FIVE, REAL_WORLD)
    robot_weight = 1.0 / (0.3 + 1.0 + 1.0)

    assert coeffs["robot"]["action"] == pytest.approx(0.5 / robot_weight)
    assert coeffs["robot"]["dino"] == pytest.approx(0.125)
    assert coeffs["egodex"]["dino"] == pytest.approx(0.125)
    assert coeffs["human"]["dino"] == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# The coefficients reach the loss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,stream", [
    ("unified", "action"),
    ("unified", "dynamics"),
    ("independent", "dynamics"),
    ("modar", "dynamics"),
    ("modar", "dynamics"),
    ("modar", None),
    ("disjoint", "dynamics"),
    ("action_only", None),
])
def test_loss_is_the_coefficient_weighted_sum(mode, stream):
    torch.manual_seed(3)
    cfg = tiny_cfg(mode, FIVE, uniform_p=True)
    model = ModalityForcingWAM(cfg).train()
    batch = make_batch(cfg)
    coeffs = {"dino": 0.25, "tracks": 0.5, "depth": 0.75, "image": 1.25,
              "action": 2.0}

    out = model(**batch, stream=stream, loss_coeffs=coeffs)

    expected = sum(
        coeffs[name] * cfg.lambda_of(name) * out[f"loss_{name}"]
        for name in cfg.modalities if float(out[f"n_{name}"]) > 0)
    torch.testing.assert_close(out["loss"].detach(), expected)


def test_omitted_modality_contributes_no_gradient():
    """A modality left out of loss_coeffs is dropped, matching a zero mask."""
    torch.manual_seed(5)
    cfg = tiny_cfg("unified", THREE)
    model = ModalityForcingWAM(cfg).train()
    batch = make_batch(cfg)

    out = model(**batch, stream="action", loss_coeffs={"action": 1.0})

    torch.testing.assert_close(
        out["loss"].detach(), cfg.lambda_of("action") * out["loss_action"])


def test_missing_loss_coeffs_preserves_historical_aggregation():
    """Bare calls (viz, sampling, older tests) must not need a resolver."""
    torch.manual_seed(11)
    cfg = tiny_cfg("unified", FIVE)
    model = ModalityForcingWAM(cfg).train()
    batch = make_batch(cfg)

    out = model(**batch, stream="dynamics")

    expected = sum(cfg.lambda_of(name) * out[f"loss_{name}"]
                   for name in cfg.modalities)
    torch.testing.assert_close(out["loss"].detach(), expected)


# ---------------------------------------------------------------------------
# The shipped configs
# ---------------------------------------------------------------------------

def sources_from_config(path):
    """Rebuild the (name, stream, weight) triples the trainer budgets over.

    Mirrors the source construction in ``train_modality_forcing.main`` so these
    assertions cover the real configs without touching the datasets they point
    at -- several of the real-world packs only exist on the training cluster.
    """
    cfg = load_config(path, {})
    dcfg = cfg.data
    source_cfgs = expand_source_configs(dcfg)
    if source_cfgs:
        sources = []
        for source_cfg in source_cfgs:
            configured = source_cfg.get("stream", "joint")
            sources.append({
                "name": str(source_cfg.name),
                "stream": None if configured == "joint" else str(configured),
                "weight": float(source_cfg.get("weight", 1.0))})
        return cfg, sources
    cotrain = (dcfg.get("tasks", None) is not None
               and dcfg.get("n_action_train", None) is not None)
    if cotrain and dcfg.get("split_pool", None) != "action":
        return cfg, [{"name": "action", "stream": "action", "weight": 1.0},
                     {"name": "dynamics", "stream": "dynamics", "weight": 1.0}]
    return cfg, [
        {"name": str(dcfg.get("name", "main")), "stream": None, "weight": 1.0}]


@pytest.mark.parametrize("config,action_share", [
    ("conf/methods/modar.yaml", 0.5),
    ("conf/methods/unified.yaml", 0.5),
    ("conf/methods/disjoint.yaml", 0.5),
    ("conf/methods/independent_noise.yaml", 0.5),
    ("conf/methods/action_only.yaml", 1.0),
    ("conf/examples/real_data.yaml", 0.5),
])
def test_shipped_configs_resolve_to_the_intended_budget(config, action_share):
    cfg, sources = sources_from_config(config)
    mfcfg = MFConfig.from_dict(OmegaConf.to_container(cfg.model, resolve=True))
    coeffs = resolve_loss_coeffs(
        mfcfg.schedule_mode, mfcfg.modalities, sources, p_of=mfcfg.p_of)
    shares = realized_shares(
        mfcfg.schedule_mode, mfcfg.modalities, sources, coeffs, mfcfg.p_of)

    dyn = dynamics_of(mfcfg.modalities)
    assert shares.get("action", 0.0) == pytest.approx(action_share)
    for name in dyn:
        assert shares[name] == pytest.approx((1.0 - action_share) / len(dyn))
    assert sum(shares.values()) == pytest.approx(1.0)
    # A source with no coefficient would burn compute for no gradient.
    for source in sources:
        assert coeffs[source["name"]], f"{source['name']} contributes nothing"


# ---------------------------------------------------------------------------
# Misuse
# ---------------------------------------------------------------------------

def test_duplicate_source_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate source"):
        resolve_loss_coeffs("modar", FIVE, [
            {"name": "robot", "stream": None, "weight": 1.0},
            {"name": "robot", "stream": "dynamics", "weight": 1.0}])


def test_nonpositive_source_weight_is_rejected():
    with pytest.raises(ValueError, match="must be > 0"):
        resolve_loss_coeffs("modar", FIVE, [
            {"name": "robot", "stream": None, "weight": 0.0}])


def test_unknown_stream_is_rejected():
    with pytest.raises(ValueError, match="unknown stream"):
        resolve_loss_coeffs("modar", FIVE, [
            {"name": "robot", "stream": "both", "weight": 1.0}])


def test_out_of_range_action_share_is_rejected():
    with pytest.raises(ValueError, match="action_share"):
        resolve_loss_coeffs("modar", FIVE, ROBOTWIN, action_share=1.5)


def test_description_reports_realized_shares():
    coeffs = resolve_loss_coeffs("unified", FIVE, ROBOTWIN)
    text = describe_loss_coeffs("unified", FIVE, ROBOTWIN, coeffs)

    assert "action share=0.500" in text
    assert "total=1" in text
    for source in ("action", "dynamics"):
        assert source in text
