#!/usr/bin/env python3
"""Validate RoboTwin-6 packs and write data-derived depth statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data.compute_depth_stats import compute_depth_stats  # noqa: E402


DEFAULT_PACK_ROOT = (
    Path(os.environ.get("MODAR_DATA_ROOT", "data"))
    / "robotwin6_packed"
)
DEFAULT_CONFIGS = (
    REPO / "conf/methods/common.yaml",
)
ROBOTWIN6_TASKS = (
    "dump_bin_bigbin",
    "pick_diverse_bottles",
    "place_bread_skillet",
    "put_bottles_dustbin",
    "stack_bowls_three",
    "turn_switch",
)
CONFIG_VALUE = {
    "depth_mean": re.compile(r"^(\s*depth_mean:\s*).*$", re.MULTILINE),
    "depth_std": re.compile(r"^(\s*depth_std:\s*).*$", re.MULTILINE),
}


def demo_packs(pack_root: Path, task: str) -> list[Path]:
    return sorted(
        path for path in (pack_root / task).glob("demo_*/trajectory.pt")
        if path.is_file()
    )


def validate_pack(path: Path) -> None:
    pack = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    expected = {
        "image_height": 168,
        "image_width": 224,
        "grid_height": 12,
        "grid_width": 16,
        "track_grid_height": 12,
        "track_grid_width": 16,
    }
    mismatches = {
        key: (pack.get(key), value)
        for key, value in expected.items()
        if int(pack.get(key, -1)) != value
    }
    if mismatches:
        raise ValueError(f"{path}: geometry mismatch {mismatches}")
    if int(pack.get("pack_version", -1)) != 2:
        raise ValueError(f"{path}: expected pack_version=2")
    if tuple(pack["dino_tokens"].shape[-2:]) != (192, 384):
        raise ValueError(
            f"{path}: DINO shape {tuple(pack['dino_tokens'].shape)}")
    if int(pack["point_tracks"].shape[2]) != 192:
        raise ValueError(
            f"{path}: track shape {tuple(pack['point_tracks'].shape)}")
    if int(pack["actions"].shape[1]) != 14:
        raise ValueError(f"{path}: action shape {tuple(pack['actions'].shape)}")
    if int(pack["proprio"].shape[1]) != 14:
        raise ValueError(
            f"{path}: proprio shape {tuple(pack['proprio'].shape)}")


def validate_dataset(
    pack_root: Path, tasks: list[str], expected_per_task: int
) -> dict[str, int]:
    counts = {}
    for task in tasks:
        packs = demo_packs(pack_root, task)
        counts[task] = len(packs)
        if len(packs) != expected_per_task:
            raise ValueError(
                f"{task}: found {len(packs)}/{expected_per_task} packs")
        for index in sorted({0, len(packs) // 2, len(packs) - 1}):
            validate_pack(packs[index])
    extra = sorted(
        path.name for path in pack_root.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
        and path.name not in set(tasks)
    )
    if extra:
        raise ValueError(f"unexpected task directories: {extra}")
    return counts


def update_depth_stats(config: Path, mean: float, std: float) -> None:
    contents = config.read_text()
    replacements = {"depth_mean": mean, "depth_std": std}
    for key, value in replacements.items():
        updated, count = CONFIG_VALUE[key].subn(
            rf"\g<1>{value:.6f}", contents)
        if count != 1:
            raise ValueError(
                f"{config}: expected one {key} field, replaced {count}")
        contents = updated
    temporary = config.with_suffix(".yaml.partial")
    temporary.write_text(contents)
    os.replace(temporary, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument(
        "--configs", type=Path, nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument(
        "--update-configs", action="store_true",
        help="write measured depth_mean/depth_std into --configs")
    parser.add_argument("--expected-per-task", type=int, default=300)
    parser.add_argument("--max-demos-per-task", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pack_root = args.pack_root.expanduser().resolve()
    configs = [config.expanduser().resolve() for config in args.configs]
    tasks = list(ROBOTWIN6_TASKS)
    counts = validate_dataset(pack_root, tasks, args.expected_per_task)
    stats = compute_depth_stats(
        [pack_root],
        num_workers=args.num_workers,
        max_demos_per_task=args.max_demos_per_task,
        seed=args.seed,
    )
    if args.update_configs:
        for config in configs:
            update_depth_stats(
                config, float(stats["depth_mean"]), float(stats["depth_std"]))
    else:
        print(
            "depth statistics were measured but configs were not changed; "
            "pass --update-configs to write them",
            file=sys.stderr,
        )
    config_digests = {
        str(config.relative_to(REPO)): hashlib.sha256(
            config.read_bytes()).hexdigest()
        for config in configs
    }
    public_config = Path("conf/methods/common.yaml")
    ready = {
        "schema_version": 1,
        "pack_root": str(pack_root),
        "tasks": len(tasks),
        "demos_per_task": args.expected_per_task,
        "total_demos": sum(counts.values()),
        "image_height": 168,
        "image_width": 224,
        "grid_height": 12,
        "grid_width": 16,
        "n_patches": 192,
        **stats,
        "configs_sha256": config_digests,
        "config_sha256": config_digests[str(public_config)],
    }
    marker = pack_root / "training_ready.json"
    temporary = marker.with_suffix(".partial")
    temporary.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    print(json.dumps(ready, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
