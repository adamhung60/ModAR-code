#!/usr/bin/env python3
"""Compute log-depth mean and standard deviation for packed demonstrations.

Reads packed keyframe depths, decodes them to metres, and applies the same
clamp and log transform as the training loader.
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from util.depth_utils import clamp_depth_maps, transform_depth_maps  # noqa: E402
from util.modality_forcing.codec import decode_depth_u16_png  # noqa: E402


def _demo(args):
    path, max_m, mode = args
    pack = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    pngs = pack.get("depths_png")
    if not pngs:
        return 0.0, 0.0, 0
    s = ss = 0.0
    c = 0
    for b in pngs:
        d = torch.from_numpy(decode_depth_u16_png(b, float(max_m))).float()
        v = transform_depth_maps(clamp_depth_maps(d, max_depth_m=float(max_m)), mode=mode)
        v = v.numpy().reshape(-1)
        s += float(v.sum()); ss += float((v * v).sum()); c += v.size
    return s, ss, c


def _sample_paths(paths: list[str], max_n: int, seed: int) -> list[str]:
    """Cap demos per task. Paths look like <root>/<task>/demo_*/trajectory.pt."""
    if max_n <= 0:
        return paths
    by_task: dict[str, list[str]] = {}
    for p in paths:
        task = Path(p).parts[-3]  # .../<task>/demo_X/trajectory.pt
        by_task.setdefault(task, []).append(p)
    rng = np.random.default_rng(seed)
    out = []
    for task, pats in sorted(by_task.items()):
        pats = sorted(pats)
        if len(pats) > max_n:
            idx = rng.choice(len(pats), size=max_n, replace=False)
            pats = [pats[i] for i in sorted(idx.tolist())]
        out.extend(pats)
    return out


def compute_depth_stats(
    roots: list[Path],
    *,
    tasks: list[str] | None = None,
    depth_max_m: float = 10.0,
    depth_norm_mode: str = "log",
    num_workers: int = 24,
    max_demos_per_task: int = 50,
    seed: int = 0,
) -> dict[str, float | int]:
    work = []
    for root in roots:
        selected = tasks or ["*"]
        paths = sorted(
            path
            for task in selected
            for path in glob.glob(
                str(root / task / "demo_*" / "trajectory.pt"))
        )
        paths = _sample_paths(paths, max_demos_per_task, seed)
        work += [(path, depth_max_m, depth_norm_mode) for path in paths]
    if not work:
        raise ValueError(f"no v2 packs found under {[str(root) for root in roots]}")

    print(
        f"scanning {len(work)} demos over {len(roots)} roots "
        f"(max_demos_per_task={max_demos_per_task})",
        flush=True,
    )
    total_sum = total_sq_sum = 0.0
    total_pixels = 0
    with mp.Pool(num_workers) as pool:
        for index, (value_sum, sq_sum, count) in enumerate(
            pool.imap_unordered(_demo, work, chunksize=8)
        ):
            total_sum += value_sum
            total_sq_sum += sq_sum
            total_pixels += count
            if (index + 1) % 1000 == 0:
                print(f"  {index + 1}/{len(work)} demos", flush=True)
    if total_pixels == 0:
        raise ValueError("selected packs contain no encoded depth frames")
    mean = total_sum / total_pixels
    std = float(
        np.sqrt(max(total_sq_sum / total_pixels - mean * mean, 0.0))
    ) or 1.0
    return {
        "depth_mean": float(mean),
        "depth_std": std,
        "n_pixels": total_pixels,
        "n_demos": len(work),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument(
        "--tasks", nargs="+",
        help="Only include these task directories (default: every task)")
    ap.add_argument("--depth-max-m", type=float, default=10.0)
    ap.add_argument("--depth-norm-mode", default="log")
    ap.add_argument("--num-workers", type=int, default=24)
    ap.add_argument(
        "--max-demos-per-task", type=int, default=50,
        help="Cap demos sampled per task (0 = use all). Default 50 is enough "
             "for stable log-depth mean/std without scanning full 1k packs.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for per-task demo subsampling.")
    args = ap.parse_args()

    stats = compute_depth_stats(
        [Path(root) for root in args.roots],
        tasks=args.tasks,
        depth_max_m=args.depth_max_m,
        depth_norm_mode=args.depth_norm_mode,
        num_workers=args.num_workers,
        max_demos_per_task=args.max_demos_per_task,
        seed=args.seed,
    )
    print(f"\nn_pixels={stats['n_pixels']} n_demos={stats['n_demos']}")
    print(f"depth_mean={stats['depth_mean']:.6f} "
          f"depth_std={stats['depth_std']:.6f} "
          f"mode={args.depth_norm_mode} max_m={args.depth_max_m}")


if __name__ == "__main__":
    main()
