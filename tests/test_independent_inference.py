import pytest
import torch

from util.modality_forcing.config import MFConfig
from util.modality_forcing.model import ModalityForcingWAM


MODALITIES = ("dino", "tracks", "depth", "image", "action")
INFER_ORDER = ("tracks", "dino", "depth", "image", "action")


def independent_cfg(**overrides):
    values = dict(
        schedule_mode="independent",
        infer_schedule="autoregressive",
        modalities=MODALITIES,
        generation_order=INFER_ORDER,
        history_modalities=("dino", "depth", "image"),
        obs_history=1,
        obs_future=2,
        dim=384,
        n_shared_layers=1,
        n_expert_layers=0,
        n_expert_layers_dino=0,
        n_expert_layers_tracks=0,
        n_expert_layers_depth=0,
        n_expert_layers_image=0,
        n_expert_layers_action=0,
        expert_dim_dino=384,
        expert_dim_tracks=384,
        expert_dim_depth=384,
        expert_dim_image=384,
        expert_dim_action=384,
        steps_per_phase=8,
        steps_by_modality={
            "tracks": 2, "dino": 3, "depth": 4, "image": 5, "action": 6,
        },
        solver_by_modality={"tracks": "heun", "action": "heun"},
    )
    values.update(overrides)
    return MFConfig(**values)


def test_independent_cascade_honors_generation_order_and_per_modality_solver():
    model = ModalityForcingWAM(independent_cfg()).eval()
    cfg = model.cfg
    batch = 1
    hist = {}
    for name in ("dino", "depth", "image"):
        dim = model.specs[model.order[name]].data_dim
        hist[name] = torch.zeros(batch, cfg.obs_history, cfg.n_patches, dim)
    proprio = torch.zeros(batch, cfg.proprio_dim)

    calls = []

    def fake_run(z, times, _proprio, _task_id):
        active = [
            name for name in MODALITIES
            if torch.allclose(times[name], torch.full_like(times[name], 0.5))
        ]
        calls.append(active[0])
        return {
            name: torch.zeros_like(value)
            for name, value in z.items()
        }

    solves = []

    def fake_solve(denoise, z0, steps, solver):
        solves.append((steps, solver))
        denoise(z0, 0.5)
        return z0

    model._run = fake_run
    model.scheduler.ode_solve = fake_solve
    with torch.no_grad():
        output = model._sample_autoregressive(hist, proprio)

    assert calls == list(INFER_ORDER)
    assert solves == [
        (2, "heun"), (3, "heun"), (4, "heun"),
        (5, "heun"), (6, "heun"),
    ]
    assert set(output) == {
        "point_tracks", "dino", "depth_maps", "images", "actions",
    }


def test_independent_unified_inference_dispatches_to_joint_sampler():
    model = ModalityForcingWAM(
        independent_cfg(infer_schedule="unified")
    ).eval()
    sentinel = {"actions": torch.zeros(1, 1, model.cfg.action_dim)}
    model._sample_unified = lambda hist, proprio, task_id=None: sentinel

    output = model._sample_independent(
        {}, torch.zeros(1, model.cfg.proprio_dim)
    )

    assert output is sentinel


@pytest.mark.parametrize(
    "infer_schedule",
    ("typo", "futures_at:nope", "futures_at:-0.1", "futures_at:1.1"),
)
def test_independent_rejects_invalid_infer_schedule(infer_schedule):
    with pytest.raises(AssertionError, match="infer_schedule"):
        independent_cfg(infer_schedule=infer_schedule)


def test_independent_requires_every_future_and_action_last_in_generation_order():
    with pytest.raises(ValueError, match="omits generated modalities"):
        independent_cfg(generation_order=("dino", "depth", "image", "action"))
    with pytest.raises(ValueError, match="action last"):
        independent_cfg(
            generation_order=("tracks", "dino", "action", "depth", "image"))
