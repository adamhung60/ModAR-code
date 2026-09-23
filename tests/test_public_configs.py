from omegaconf import OmegaConf

from util.modality_forcing.config import MFConfig
from util.modality_forcing.config_paths import load_config
from util.modality_forcing.data import split_demos_cotrain


def _model(name: str) -> MFConfig:
    cfg = load_config(name)
    return MFConfig.from_dict(OmegaConf.to_container(cfg.model, resolve=True))


def test_public_method_names_resolve_to_paper_formulations():
    expected = {
        "modar": ("modar", "autoregressive"),
        "unified": ("unified", "autoregressive"),
        "disjoint": ("disjoint", "autoregressive"),
        "independent_noise": ("independent", "unified"),
        "action_only": ("action_only", "autoregressive"),
    }
    for name, (schedule_mode, infer_schedule) in expected.items():
        model = _model(name)
        assert model.schedule_mode == schedule_mode
        assert model.infer_schedule == infer_schedule


def test_modar_uses_published_generation_order():
    model = _model("modar")
    assert model.generated_modalities is None
    assert model.generated_modality_names() == [
        "dino", "tracks", "depth", "image"
    ]
    assert model.generation_order == (
        "tracks", "dino", "depth", "image", "action"
    )


def test_default_robotwin_split_uses_250_unique_training_demos_per_task():
    cfg = load_config("modar")
    data = cfg.data
    assert data.n_dyn_train == 250
    assert data.n_action_train == 50
    assert data.n_dyn_val == 50
    assert data.split_universe == 300

    demos = [f"demo_{index:06d}" for index in range(300)]
    pools = split_demos_cotrain(
        {"task": demos},
        n_action_train=data.n_action_train,
        n_action_val=data.n_action_val,
        n_dyn_val=data.n_dyn_val,
        n_dyn_train=data.n_dyn_train,
        seed=cfg.train.seed,
        split_universe=data.split_universe,
    )
    assert len(pools["dyn_train"]) == 250
    assert len(pools["action_train"]) == 50
    assert len(pools["dyn_val"]) == 50
    assert set(pools["action_train"]).issubset(pools["dyn_train"])
    assert set(pools["dyn_train"]).isdisjoint(pools["dyn_val"])


def test_real_data_example_trains_robot_dynamics():
    cfg = load_config("real_data")
    sources = {source.name: source for source in cfg.data.sources}
    assert sources["actionless_video"].stream == "dynamics"
    assert sources["robot"].stream == "joint"
