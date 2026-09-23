from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from util.modality_forcing.data import build_source_task_to_id
from scripts.train.train_modality_forcing import (
    evaluate_sources,
    expand_source_configs,
    load_config,
    normalize_source_weights,
    select_action_source,
    weighted_source_loss,
)


def test_source_task_aliases_share_global_embedding_rows():
    vocab = ["pour", "stack_cups"]
    egodex = build_source_task_to_id(
        ["stack_unstack_cups", "pour"],
        task_vocab=vocab,
        task_aliases={"stack_unstack_cups": "stack_cups"},
    )
    human = build_source_task_to_id(
        ["pour", "stack_cups"], task_vocab=vocab)
    robot = build_source_task_to_id(
        ["stack_cups", "pour"], task_vocab=vocab)
    assert egodex["stack_unstack_cups"] == human["stack_cups"]
    assert human["stack_cups"] == robot["stack_cups"]
    assert egodex["pour"] == human["pour"] == robot["pour"]


def test_source_task_aliases_reject_invalid_mappings():
    with pytest.raises(ValueError, match="not configured"):
        build_source_task_to_id(
            ["stack"], ["stack"], {"pour": "stack"})
    with pytest.raises(ValueError, match="absent"):
        build_source_task_to_id(
            ["stack_unstack"], ["stack"], {"stack_unstack": "unknown"})
    with pytest.raises(ValueError, match="multiple physical"):
        build_source_task_to_id(
            ["stack", "stack_unstack"], ["stack"],
            {"stack_unstack": "stack"})


def test_source_configs_inherit_shared_data_defaults():
    cfg = OmegaConf.create({
        "depth_mean": -0.5,
        "depth_std": 0.25,
        "val_ratio": 0.1,
        "sources": [
            {
                "name": "off_domain",
                "stream": "dynamics",
                "data_root": "/video",
                "tasks": ["pour"],
            },
            {
                "name": "robot",
                "stream": "action",
                "data_root": "/robot",
                "tasks": ["stack"],
                "val_ratio": 0.2,
            },
        ],
    })
    sources = expand_source_configs(cfg)
    assert [source.name for source in sources] == ["off_domain", "robot"]
    assert sources[0].depth_mean == -0.5
    assert sources[0].val_ratio == 0.1
    assert sources[1].val_ratio == 0.2
    assert "sources" not in sources[0]


def test_source_configs_resolve_root_level_interpolations_before_detaching():
    cfg = OmegaConf.create({
        "datasets": {
            "robot": {"root": "/packs/robot", "train_count": 45},
        },
        "data": {
            "depth_mean": -0.5,
            "depth_std": 0.25,
            "sources": [{
                "name": "robot",
                "data_root": "${datasets.robot.root}",
                "train_count": "${datasets.robot.train_count}",
            }],
        },
    })

    source = expand_source_configs(cfg.data)[0]

    assert source.data_root == "/packs/robot"
    assert source.train_count == 45


def test_source_weights_are_positive_and_normalized():
    sources = [{"weight": 1.0}, {"weight": 2.0}, {"weight": 1.0}]
    assert normalize_source_weights(sources) == [0.25, 0.5, 0.25]
    with pytest.raises(ValueError, match="must be > 0"):
        normalize_source_weights([{"weight": 0.0}])


def test_joint_robot_source_defines_action_stats_over_dynamics_sources():
    dynamics = {"name": "egodex", "stream": "dynamics"}
    robot = {"name": "robot", "stream": None}

    assert select_action_source([dynamics, robot]) is robot


def test_explicit_action_source_takes_priority_over_joint_source():
    joint = {"name": "joint", "stream": None}
    action = {"name": "robot", "stream": "action"}

    assert select_action_source([joint, action]) is action


def test_real_world_robot_pass_jointly_supervises_dynamics_and_action():
    cfg = load_config("conf/examples/real_data.yaml", {})
    robot = next(
        source for source in cfg.data.sources if source.name == "robot")
    assert robot.stream == "joint"
    assert robot.train_count == 100
    video = next(
        source for source in cfg.data.sources
        if source.name == "actionless_video")
    assert video.stream == "dynamics"
    assert video.train_count == 200


def test_real_world_runs_evaluate_integrated_action_error():
    for config in ("modar", "action_only"):
        cfg = load_config(config, {})
        assert cfg.train.integ_every_samples == 4_800_000


def test_weighted_source_loss_accumulates_expected_gradient():
    parameter = torch.tensor(1.0, requires_grad=True)
    sources = [{"weight": 1.0}, {"weight": 3.0}]
    weights = normalize_source_weights(sources)
    outputs = [
        (sources[0], {"loss": 2.0 * parameter}, weights[0]),
        (sources[1], {"loss": 4.0 * parameter}, weights[1]),
    ]
    loss = weighted_source_loss(outputs)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(3.5))
    torch.testing.assert_close(parameter.grad, torch.tensor(3.5))


class _FakeModel:
    def __init__(self):
        self.cfg = SimpleNamespace(modalities=("dino", "action"))
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, dino, stream=None, loss_coeffs=None):
        batch_size = dino.shape[0]
        zero = torch.tensor(0.0)
        if stream == "action":
            return {
                "loss": torch.tensor(2.0),
                "loss_dino": zero,
                "loss_action": torch.tensor(2.0),
                "n_dino": zero,
                "n_action": torch.tensor(float(batch_size)),
                "act_mm": torch.tensor(10.0),
                "act_deg": torch.tensor(5.0),
            }
        return {
            "loss": torch.tensor(4.0),
            "loss_dino": torch.tensor(4.0),
            "loss_action": zero,
            "n_dino": torch.tensor(float(batch_size)),
            "n_action": zero,
            "act_mm": zero,
            "act_deg": zero,
        }


def test_evaluate_sources_reports_namespaced_and_weighted_metrics():
    batch = {"dino": torch.zeros(2, 1)}
    sources = [
        {
            "name": "robot",
            "stream": "action",
            "weight": 1.0,
            "val_loader": [batch],
        },
        {
            "name": "video",
            "stream": "dynamics",
            "weight": 3.0,
            "val_loader": [batch],
        },
    ]
    metrics = evaluate_sources(_FakeModel(), sources, "cpu", max_batches=0)
    assert metrics["val/robot/loss"] == 2.0
    assert metrics["val/video/loss"] == 4.0
    assert metrics["val/loss"] == 3.5
    assert metrics["val/loss_action"] == 2.0
    assert metrics["val/loss_dino"] == 4.0
    assert metrics["val/mm_error_onestep"] == 10.0
