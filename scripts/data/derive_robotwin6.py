#!/usr/bin/env python3
"""Derive resumable 224x168 ModAR packs from native RoboTwin HDF5 episodes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data.finalize_robotwin6 import validate_pack  # noqa: E402


DATA_ROOT = Path(os.environ.get("MODAR_DATA_ROOT", "data"))
DEFAULT_RAW = DATA_ROOT / "robotwin6_raw"
DEFAULT_WAM = DATA_ROOT / "robotwin6_wam"
DEFAULT_TRACKS = DATA_ROOT / "robotwin6_tracks"
DEFAULT_PACKED = DATA_ROOT / "robotwin6_packed"
DEFAULT_TASK_CONFIG = "modar_robotwin6"
ROBOTWIN6_TASKS = (
    "dump_bin_bigbin",
    "pick_diverse_bottles",
    "place_bread_skillet",
    "put_bottles_dustbin",
    "stack_bowls_three",
    "turn_switch",
)
EXPECTED_HEIGHT = 168
EXPECTED_WIDTH = 224
_PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with _PRINT_LOCK:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


def count_files(root: Path, task: str, pattern: str) -> int:
    return len(list((root / task).glob(pattern)))


def raw_episodes(raw_root: Path, task: str, task_config: str) -> list[Path]:
    return sorted(
        (raw_root / task / task_config / "data").glob("episode*.hdf5"))


def packed_task_valid(packed_root: Path, task: str, expected: int) -> bool:
    packs = sorted((packed_root / task).glob("demo_*/trajectory.pt"))
    if len(packs) != expected:
        return False
    for index in sorted({0, len(packs) // 2, len(packs) - 1}):
        validate_pack(packs[index])
    return True


def run_command(
    command: list[str],
    *,
    log_file: Path,
    environment: dict[str, str] | None = None,
) -> None:
    with log_file.open("a") as output:
        output.write(f"\n$ {' '.join(command)}\n")
        output.flush()
        subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def validate_wam_task(wam_root: Path, task: str, expected: int) -> None:
    metadata = sorted((wam_root / task).glob("demo_*/metadata.json"))
    if len(metadata) != expected:
        raise ValueError(
            f"{task}: found {len(metadata)}/{expected} converted demos")
    for path in metadata:
        meta = json.loads(path.read_text())
        if meta.get("camera_render_size") != [
            EXPECTED_HEIGHT, EXPECTED_WIDTH
        ]:
            raise ValueError(
                f"{path}: camera_render_size={meta.get('camera_render_size')}")


def derive_task(
    task: str,
    gpu: str,
    args: argparse.Namespace,
    expected: int,
) -> str:
    if packed_task_valid(args.packed_root, task, expected):
        log(f"{task}: existing packs validated")
        return "reused"
    episodes = raw_episodes(args.raw_root, task, args.task_config)
    if len(episodes) != expected:
        raise ValueError(
            f"{task}: native archive has {len(episodes)}/{expected} episodes")
    if args.dry_run:
        return "would_derive"

    log_file = args.log_root / f"{task}.log"
    if count_files(args.wam_root, task, "demo_*/metadata.json") != expected:
        run_command([
            sys.executable, "-u", "-m",
            "robotwin_manip.datagen.robotwin_to_wam",
            "--robotwin_data", str(args.raw_root),
            "--task_config", args.task_config,
            "--tasks", task,
            "--out_root", str(args.wam_root),
            "--camera", "head_camera",
            "--num_points", "2000",
            "--image-height", str(EXPECTED_HEIGHT),
            "--image-width", str(EXPECTED_WIDTH),
            "--max_depth", "10.0",
            "--num-workers", str(args.convert_workers),
        ], log_file=log_file)
    validate_wam_task(args.wam_root, task, expected)

    gpu_environment = os.environ.copy()
    gpu_environment["CUDA_VISIBLE_DEVICES"] = gpu
    run_command([
        sys.executable, "-u", "scripts/data/extract_dino_features.py",
        "--dataset", str(args.wam_root),
        "--tasks", task,
        "--device", "cuda:0",
        "--batch_size", str(args.dino_batch_size),
        "--num-workers", str(args.convert_workers),
        "--image-height", str(EXPECTED_HEIGHT),
        "--image-width", str(EXPECTED_WIDTH),
    ], log_file=log_file, environment=gpu_environment)
    run_command([
        sys.executable, "-u", "scripts/data/extract_point_tracks.py",
        "--src-root", str(args.wam_root),
        "--out-root", str(args.tracks_root),
        "--tasks", task,
        "--anchor-batch", str(args.anchor_batch),
        "--num-workers", str(args.convert_workers),
        "--image-height", str(EXPECTED_HEIGHT),
        "--image-width", str(EXPECTED_WIDTH),
    ], log_file=log_file, environment=gpu_environment)
    run_command([
        sys.executable, "-u", "scripts/data/pack_wam_target_5mod.py",
        "--src-root", str(args.wam_root),
        "--tracks-root", str(args.tracks_root),
        "--out-root", str(args.packed_root),
        "--tasks", task,
        "--depth-max-m", "10.0",
        "--num-workers", str(args.pack_workers),
    ], log_file=log_file)
    if not packed_task_valid(args.packed_root, task, expected):
        raise ValueError(f"{task}: packed output failed validation")
    marker = args.packed_root / "_derived_complete" / f"{task}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "task": task,
        "episodes": expected,
        "image_height": EXPECTED_HEIGHT,
        "image_width": EXPECTED_WIDTH,
        "gpu": gpu,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2) + "\n")
    log(f"{task}: derived and validated {expected} packs")
    return "derived"


def worker(
    gpu: str,
    work: queue.Queue[str],
    args: argparse.Namespace,
    expected: int,
    results: dict[str, str],
    result_lock: threading.Lock,
) -> None:
    while True:
        try:
            task = work.get_nowait()
        except queue.Empty:
            return
        try:
            status = derive_task(task, gpu, args, expected)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            status = f"failed: {error}"
        with result_lock:
            results[task] = status
        work.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--wam-root", type=Path, default=DEFAULT_WAM)
    parser.add_argument("--tracks-root", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--packed-root", type=Path, default=DEFAULT_PACKED)
    parser.add_argument("--expected-per-task", type=int, default=300)
    parser.add_argument("--convert-workers", type=int, default=4)
    parser.add_argument("--pack-workers", type=int, default=8)
    parser.add_argument("--dino-batch-size", type=int, default=64)
    parser.add_argument("--anchor-batch", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = args.tasks or list(ROBOTWIN6_TASKS)
    unknown = sorted(set(tasks).difference(ROBOTWIN6_TASKS))
    if unknown:
        raise ValueError(f"unknown public RoboTwin-6 tasks: {unknown}")
    if len(args.gpus) != len(set(args.gpus)):
        raise ValueError("--gpus must be unique")
    for name in ("raw_root", "wam_root", "tracks_root", "packed_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.log_root = args.packed_root / "_derive_logs"
    for root in (
        args.wam_root, args.tracks_root, args.packed_root, args.log_root
    ):
        root.mkdir(parents=True, exist_ok=True)

    work: queue.Queue[str] = queue.Queue()
    for task in tasks:
        work.put(task)
    results: dict[str, str] = {}
    result_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu, work, args, args.expected_per_task, results, result_lock),
            name=f"gpu-{gpu}",
        )
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(json.dumps(dict(sorted(results.items())), indent=2))
    failures = {
        task: status for task, status in results.items()
        if status.startswith("failed:")
    }
    if failures:
        raise SystemExit(f"derivation failures: {failures}")


if __name__ == "__main__":
    main()
