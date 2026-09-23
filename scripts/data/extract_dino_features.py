"""Pre-extract DINOv2 patch tokens for WAM-format datasets.

Supports either a single ``--dataset`` (+ optional ``--output_dir``) or
multiple ``--roots`` (tokens written next to each demo).

Skips frames that already have tokens (safe to resume).

Examples::

    python scripts/data/extract_dino_features.py \\
        --roots data/robotwin_wam

    python scripts/data/extract_dino_features.py \\
        --roots data/egodex_wam/part2
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, repo_root)

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _parse_task_filter(s: str) -> set[str] | None:
    tasks = [t.strip() for t in s.split(",") if t.strip()]
    return set(tasks) if tasks else None


# ViT-S/14 @224 half-precision patch tokens are ~190 KiB. Reject both truncated
# files and the old bug that torch.saved a full batch view (~25 MiB).
_MIN_DINO_TOKEN_BYTES = 10_000
_MAX_DINO_TOKEN_BYTES = 500_000


def _dino_token_ready(path: str | os.PathLike) -> bool:
    try:
        sz = os.path.getsize(path)
    except OSError:
        return False
    return _MIN_DINO_TOKEN_BYTES <= sz <= _MAX_DINO_TOKEN_BYTES


def _save_dino_token(tensor: torch.Tensor, out_path: str) -> None:
    """Write a single-frame token .pth (must clone: views retain batch storage)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f"{out.suffix}.tmp")
    torch.save(tensor.detach().contiguous().clone(), tmp)
    tmp.replace(out)


def _collect_wam_demo_dirs(data_root: Path, task_filter: set[str] | None) -> list[Path]:
    demos: list[Path] = []
    for task_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if task_dir.name.startswith("."):
            continue
        if task_filter is not None and task_dir.name not in task_filter:
            continue
        for demo_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if demo_dir.name.startswith("."):
                continue
            if (demo_dir / "metadata.json").is_file():
                demos.append(demo_dir)
    return demos


def _collect_jobs_colocated(roots: list[str]) -> list[tuple[str, str]]:
    """Walk roots and yield (rgb_path, token_path) with colocated tokens."""
    jobs: list[tuple[str, str]] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            print(f"  Skipping non-existent root: {root_path}")
            continue
        for task_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            for demo_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                meta_path = demo_dir / "metadata.json"
                if not meta_path.is_file():
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                for fr in meta.get("frames", []):
                    rgb_name = fr.get("rgb")
                    if not rgb_name:
                        continue
                    rgb_path = demo_dir / rgb_name
                    if not rgb_path.is_file():
                        continue
                    stem = Path(rgb_name).stem
                    parts = stem.split("_", 1)
                    if len(parts) < 2:
                        continue
                    token_path = demo_dir / f"dino_tokens_{parts[1]}.pth"
                    if not _dino_token_ready(token_path):
                        jobs.append((str(rgb_path), str(token_path)))
    return jobs


def _build_jobs_wam(
        dataset_root: str,
        output_root: str,
        task_filter: set[str] | None,
        *,
        num_shards: int = 1,
        shard_index: int = 0):
    root = Path(dataset_root)
    demos = _collect_wam_demo_dirs(root, task_filter)
    if num_shards > 1:
        demos = demos[shard_index::num_shards]
    jobs = []
    skipped_demos = 0
    for demo_dir in demos:
        task = demo_dir.parent.name
        meta = json.loads((demo_dir / "metadata.json").read_text(encoding="utf-8"))
        frames = meta.get("frames", [])
        # Fast path: if every frame already has a large-enough token file, skip
        # the demo without per-frame path joins in the hot loop below.
        out_demo = Path(output_root) / task / demo_dir.name
        if out_demo.is_dir():
            ready = 0
            with os.scandir(out_demo) as it:
                for ent in it:
                    if (ent.name.startswith("dino_tokens_")
                            and ent.name.endswith(".pth")
                            and ent.is_file()
                            and ent.stat().st_size >= _MIN_DINO_TOKEN_BYTES):
                        ready += 1
            if ready >= len(frames) and frames:
                skipped_demos += 1
                continue

        for fr in frames:
            rgb_name = fr.get("rgb")
            if not rgb_name:
                continue
            rgb_path = str(demo_dir / rgb_name)
            if not os.path.isfile(rgb_path):
                continue
            stem = Path(rgb_name).stem
            parts = stem.split("_", 1)
            if len(parts) < 2:
                continue
            idx_suffix = parts[1]
            out_path = os.path.join(
                output_root, task, demo_dir.name, f"dino_tokens_{idx_suffix}.pth")
            colocated = os.path.join(
                str(demo_dir), f"dino_tokens_{idx_suffix}.pth")
            if _dino_token_ready(out_path) or _dino_token_ready(colocated):
                continue
            jobs.append((rgb_path, out_path))
    return demos, jobs, skipped_demos


def _group_jobs_by_demo(jobs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Group (rgb, out) jobs by demo dir so disk I/O stays sequential."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rgb_path, out_path in jobs:
        groups[str(Path(out_path).parent)].append((rgb_path, out_path))
    return [groups[k] for k in sorted(groups)]


def _load_rgb_tensor(
    rgb_path: str,
    image_height: int,
    image_width: int | None = None,
) -> torch.Tensor:
    """Load RGB at ``(height, width)``; omitted width keeps legacy square use."""
    image_width = image_height if image_width is None else image_width
    img = Image.open(rgb_path).convert("RGB").resize(
        (image_width, image_height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract DINOv2 patch tokens for WAM datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single WAM dataset root (use with --output_dir)")
    parser.add_argument("--roots", nargs="+", default=None,
                        help="Dataset roots; tokens colocated in each demo dir")
    parser.add_argument(
        "--tasks", type=str, default="",
        help="Comma-separated task folders under --dataset (empty = all tasks)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output root (default: same as --dataset)")
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Thread workers for image decode")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split demos across GPUs (run one process per shard)")
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if (args.dataset is None) == (args.roots is None):
        parser.error("Specify exactly one of --dataset or --roots")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard-index must be in [0, {args.num_shards})")

    device = torch.device(args.device)
    if (args.image_height is None) != (args.image_width is None):
        parser.error("--image-height and --image-width must be set together")
    image_height = int(
        args.image_height if args.image_height is not None else args.image_size)
    image_width = int(
        args.image_width if args.image_width is not None else args.image_size)
    if image_height % 14 or image_width % 14:
        parser.error("DINO image dimensions must be divisible by patch size 14")
    torch.backends.cudnn.benchmark = True

    if args.roots is not None:
        jobs = _collect_jobs_colocated([os.path.normpath(r) for r in args.roots])
        print(f"Collected {len(jobs)} images from {len(args.roots)} root(s)",
              flush=True)
        demo_groups = _group_jobs_by_demo(jobs)
        if args.num_shards > 1:
            demo_groups = demo_groups[args.shard_index::args.num_shards]
    else:
        dataset_root = os.path.normpath(args.dataset)
        output_root = os.path.normpath(args.output_dir or args.dataset)
        task_filter = _parse_task_filter(args.tasks)
        shard_demos, jobs, skipped_demos = _build_jobs_wam(
            dataset_root, output_root, task_filter,
            num_shards=args.num_shards, shard_index=args.shard_index)
        filt = "all tasks" if task_filter is None else str(sorted(task_filter))
        print(f"Found {len(shard_demos)} WAM demos on this shard ({filt}); "
              f"skipped_complete={skipped_demos}", flush=True)
        demo_groups = _group_jobs_by_demo(jobs)

    n_imgs = sum(len(g) for g in demo_groups)
    if args.num_shards > 1:
        print(f"[shard {args.shard_index}/{args.num_shards}] "
              f"{len(demo_groups)} demos / {n_imgs} images", flush=True)

    print(f"Loading {args.dino_model}...")
    backbone = torch.hub.load("facebookresearch/dinov2", args.dino_model)
    backbone.eval().to(device)

    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print(f"{n_imgs} images to extract (skipping already-extracted) "
          f"bs={args.batch_size} workers={args.num_workers} "
          f"demos={len(demo_groups)}", flush=True)
    if not demo_groups:
        print("Done!")
        return

    n_workers = max(1, args.num_workers)
    load_pool = ThreadPoolExecutor(max_workers=n_workers)
    use_amp = device.type == "cuda"

    # Unbuffered progress when stdout is redirected to a log file.
    pbar = tqdm(total=n_imgs, unit="img", mininterval=1.0, file=sys.stdout)
    for group in demo_groups:
        for s in range(0, len(group), args.batch_size):
            chunk = group[s:s + args.batch_size]
            rgbs = [j[0] for j in chunk]
            outs = [j[1] for j in chunk]
            tensors = list(load_pool.map(
                _load_rgb_tensor, rgbs, itertools.repeat(image_height),
                itertools.repeat(image_width)))
            images_t = torch.stack(tensors).to(device, non_blocking=True)
            images_t = (images_t - img_mean) / img_std
            with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp):
                features = backbone.forward_features(images_t)
                patch_tokens = features["x_norm_patchtokens"]
            expected_patches = (image_height // 14) * (image_width // 14)
            if patch_tokens.shape[1] != expected_patches:
                raise ValueError(
                    f"DINO emitted {patch_tokens.shape[1]} patches, expected "
                    f"{image_height // 14}x{image_width // 14}="
                    f"{expected_patches}")
            tokens_cpu = patch_tokens.half().cpu()
            for i, path in enumerate(outs):
                _save_dino_token(tokens_cpu[i], path)
            pbar.update(len(chunk))
            pbar.refresh()
    pbar.close()
    load_pool.shutdown(wait=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
