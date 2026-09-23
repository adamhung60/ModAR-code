"""Shared checkpoint, DINO, and held-out-pool helpers for evaluation."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from util.modality_forcing.config import MFConfig
from util.modality_forcing.data import (
    build_task_to_id,
    canonical_task_vocab,
)
from util.modality_forcing.model import (
    ModalityForcingWAM,
    load_spatially_compatible_state_dict,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DinoEncoder:
    """Frozen DINOv2 patch-token encoder matching data preprocessing."""

    def __init__(self, device, model_name: str = "dinov2_vits14",
                 image_size: int | tuple[int, int] = 224):
        self.device = device
        self.image_size = (
            (int(image_size), int(image_size))
            if isinstance(image_size, int)
            else (int(image_size[0]), int(image_size[1]))
        )
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.eval().to(device)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def __call__(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """Encode one ``(H, W, 3)`` uint8 image into DINO patch tokens."""
        image = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).float()
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        if tuple(image.shape[-2:]) != self.image_size:
            image = F.interpolate(
                image, size=self.image_size, mode="bilinear",
                align_corners=False)
        image = (image - self.mean) / self.std
        features = self.backbone.forward_features(image)["x_norm_patchtokens"]
        return features[0]


def build_mf_from_checkpoint(ckpt_path, device, use_ema: bool = True, *,
                             with_meta: bool = False):
    """Load ``(model, model_config, stats, data_config)`` from a checkpoint.

    ``with_meta`` appends the training bookkeeping (``step``, ``samples_seen``,
    ``global_batch``) as a fifth element, so a caller comparing two checkpoints
    can report and check which point in training each one is from without
    paying a second read of the state dict.
    """
    checkpoint = torch.load(
        os.fspath(ckpt_path), map_location=device, weights_only=False)
    config = checkpoint["cfg"]
    model_config = dict(config["model"])
    if int(model_config.get("n_tasks", 0)) <= 0:
        tasks = config.get("data", {}).get("tasks")
        if tasks:
            model_config["n_tasks"] = len(list(tasks))
        elif config.get("data", {}).get("task"):
            model_config["n_tasks"] = 1
    mf_config = MFConfig.from_dict(model_config)
    model = ModalityForcingWAM(mf_config).to(device)
    state_dict = (
        checkpoint["ema"]
        if use_ema and checkpoint.get("ema") is not None
        else checkpoint["model"]
    )
    load_spatially_compatible_state_dict(model, state_dict)
    model.eval()
    if with_meta:
        meta = {key: checkpoint.get(key)
                for key in ("step", "samples_seen", "global_batch")}
        return model, mf_config, checkpoint["stats"], config["data"], meta
    return model, mf_config, checkpoint["stats"], config["data"]


def task_to_id_from_data_cfg(
        data_config: dict,
        mf_config: MFConfig,
        checkpoint_task_to_id: dict | None = None,
) -> dict | None:
    """Resolve the exact task-embedding map used during training.

    New checkpoints store the map explicitly. Legacy checkpoints fall back to
    the trainer's canonical sorted vocabulary—not YAML list order.
    """
    if mf_config.n_tasks <= 0:
        return None
    if checkpoint_task_to_id is not None:
        mapping = {
            str(task): int(index)
            for task, index in dict(checkpoint_task_to_id).items()
        }
        expected = list(range(len(mapping)))
        if sorted(mapping.values()) != expected:
            raise ValueError(
                f"checkpoint task_to_id must use contiguous rows, got {mapping}")
        if len(mapping) != int(mf_config.n_tasks):
            raise ValueError(
                "checkpoint task_to_id/model n_tasks mismatch: "
                f"{len(mapping)} != {mf_config.n_tasks}")
        return mapping
    vocabulary = data_config.get("task_vocab")
    if vocabulary:
        return build_task_to_id(canonical_task_vocab(list(vocabulary)))
    tasks = data_config.get("tasks")
    if tasks:
        return build_task_to_id(canonical_task_vocab(list(tasks)))
    sources = data_config.get("sources")
    if sources:
        source_tasks = {
            str(task)
            for source in sources
            for task in source.get("tasks", [])
        }
        if source_tasks:
            return build_task_to_id(canonical_task_vocab(list(source_tasks)))
    task = data_config.get("task")
    if task:
        return build_task_to_id([task])
    raise ValueError(
        f"model has n_tasks={mf_config.n_tasks}, but checkpoint has no task "
        "vocabulary or explicit task_to_id mapping")


def discover_eval_pool_multi(
        data_config: dict, n_rollouts: int | None = None,
        n_per_task: int | None = None) -> list[Path]:
    """Build a deterministic, task-balanced held-out evaluation pool."""
    from util.modality_forcing.data import discover_demos_multi, split_demos

    tasks = list(data_config["tasks"])
    demos_by_task = discover_demos_multi(data_config["data_root"], tasks)
    demo_limit = data_config.get("demo_limit")
    if demo_limit is not None:
        demos_by_task = {
            task: demos[:demo_limit] for task, demos in demos_by_task.items()}
    val_ratio = float(data_config.get("val_ratio", 0.1))
    seed = int(data_config.get("seed", 42))
    by_task = {}
    for task in tasks:
        _, val_demos = split_demos(demos_by_task[task], val_ratio, seed)
        by_task[task] = sorted(val_demos)
    if n_per_task is not None:
        return [
            demo
            for task in tasks
            for demo in by_task[task][:n_per_task]
        ]
    if n_rollouts is None:
        raise ValueError(
            "discover_eval_pool_multi requires n_rollouts or n_per_task")
    iterators = [iter(by_task[task]) for task in tasks]
    output = []
    exhausted = [False] * len(tasks)
    while len(output) < n_rollouts and not all(exhausted):
        for index, iterator in enumerate(iterators):
            if exhausted[index]:
                continue
            demo = next(iterator, None)
            if demo is None:
                exhausted[index] = True
            else:
                output.append(demo)
            if len(output) >= n_rollouts:
                break
    return output
