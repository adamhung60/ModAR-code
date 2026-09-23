#!/usr/bin/env python3
"""Convert RoboTwin ``episodeN.hdf5`` demos into ModAR's on-disk format.

Reads the per-episode HDF5 files RoboTwin writes under
``<robotwin_data>/<task>/<task_config>/data/episode*.hdf5`` and reconstructs
each trajectory as a WAM demo directory (see ``robotwin_wam_io``). For every
frame we take:

  * RGB         <- ``/observation/<camera>/rgb``      (JPEG-decoded)
  * depth       <- ``/observation/<camera>/depth``    (mm -> metric metres)
  * point cloud <- ``/pointcloud``                    (world-frame xyz, FPS'd)
  * proprio     <- ``/joint_action/vector`` at t      (14-D dual-arm joints)
  * action      <- ``/joint_action/vector`` at t+1    (absolute next-step target)

The point cloud is RoboTwin's own ``pointcloud:true`` output (already
world-frame and cropped); we only slice xyz and FPS-downsample to
``--num_points``. RGB and depth always resize the complete native 320x240
camera frame without cropping to 224x168 (width x height).

Collect the source data first with depth + pointcloud enabled, e.g.::

    cd RoboTwin
    bash collect_data.sh beat_block_hammer wam_depth 0   # wam_depth.yml: depth+pcd on

Then convert::

    python -m robotwin_manip.datagen.robotwin_to_wam \
        --robotwin_data RoboTwin/data \
        --task_config wam_depth \
        --tasks beat_block_hammer place_shoe \
        --out_root data/robotwin_wam \
        --num_points 2000
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from robotwin_manip.datagen.robotwin_wam_io import (  # noqa: E402
    DEFAULT_MAX_DEPTH_M,
    depth_mm_to_metric,
    intrinsic_for_resize,
    joint_vectors_to_proprio_action,
    resize_rgb,
    write_wam_demo_dir,
)

IMAGE_HEIGHT = 168
IMAGE_WIDTH = 224


def _episode_files(task_data_dir: Path) -> list[Path]:
    """Sorted episode*.hdf5 under <task>/<task_config>/data, by index."""
    files = []
    for p in task_data_dir.glob("episode*.hdf5"):
        stem = p.stem  # episodeN
        n = stem[len("episode"):]
        if n.isdigit():
            files.append((int(n), p))
    files.sort()
    return [p for _, p in files]


def _decode_rgb_frame(raw_bytes) -> np.ndarray:
    """JPEG bytes -> (H, W, 3) uint8. RoboTwin stores cv2-encoded RGB frames."""
    import cv2

    buf = np.frombuffer(bytes(raw_bytes), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None (corrupt RGB frame)")
    return img


def convert_episode(
    hdf5_path: Path,
    out_root: Path,
    task: str,
    demo_id: str,
    *,
    camera: str,
    num_points: int,
    max_depth_m: float,
    spec_id: str,
    split: str,
    save_points: bool,
    save_depth: bool,
    save_rgb: bool,
    image_height: int = IMAGE_HEIGHT,
    image_width: int = IMAGE_WIDTH,
) -> tuple[str, str]:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        if "joint_action/vector" not in f:
            return str(hdf5_path), "missing joint_action/vector"
        vectors = np.asarray(f["joint_action/vector"][()], dtype=np.float32)
        T = vectors.shape[0]
        proprio, actions = joint_vectors_to_proprio_action(vectors)

        cam_grp_path = f"observation/{camera}"
        if cam_grp_path not in f:
            return str(hdf5_path), (
                f"camera {camera!r} not in /observation "
                f"(have {list(f.get('observation', {}).keys())})")
        cam_grp = f[cam_grp_path]

        out_h = int(image_height)
        out_w = int(image_width)
        if (out_h, out_w) != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(
                f"RoboTwin output must be {IMAGE_HEIGHT}x{IMAGE_WIDTH}, "
                f"got {out_h}x{out_w}")

        rgbs = None
        camera_K = None
        if save_rgb or save_depth:
            raw0 = _decode_rgb_frame(cam_grp["rgb"][0])
            src_h, src_w = raw0.shape[:2]
            if "intrinsic_cv" in cam_grp:
                K_raw = np.asarray(cam_grp["intrinsic_cv"][0]
                                   if cam_grp["intrinsic_cv"].ndim == 3
                                   else cam_grp["intrinsic_cv"][()])
                camera_K = intrinsic_for_resize(
                    K_raw, src_h, src_w, out_h, out_w)

        if save_rgb:
            rgbs = np.stack([
                resize_rgb(image, out_h, out_w)
                for image in (
                    _decode_rgb_frame(cam_grp["rgb"][t]) for t in range(T))
            ], axis=0)

        depths_per_t = None
        if save_depth:
            if "depth" not in cam_grp:
                return str(hdf5_path), (
                    f"depth requested but /observation/{camera}/depth absent "
                    "(set data_type.depth: true in the collection config)")
            depth_all = np.asarray(cam_grp["depth"][()])
            depths_per_t = [
                depth_mm_to_metric(
                    depth_all[t], out_h, out_w, max_depth_m)
                for t in range(T)
            ]

        points_per_t = None
        if save_points:
            if "pointcloud" not in f:
                return str(hdf5_path), (
                    "points requested but /pointcloud absent "
                    "(set data_type.pointcloud: true in the collection config)")
            pcd_all = np.asarray(f["pointcloud"][()])  # (T, N, 6) world xyz+rgb
            points_per_t = [pcd_all[t][:, :3].astype(np.float32)
                            for t in range(T)]

    # rgbs may be None when save_rgb=False but save_depth requires a T anchor.
    if rgbs is None:
        rgbs = np.zeros((T, 1, 1, 3), dtype=np.uint8)

    write_wam_demo_dir(
        out_root, task, demo_id,
        rgbs=rgbs,
        proprio_matrix=proprio,
        actions=actions,
        points_per_t=points_per_t if save_points else [],
        num_points=num_points,
        spec_id=spec_id,
        source_hdf5=str(hdf5_path.resolve()),
        source_episode_index=int(demo_id.split("_")[-1]),
        split=split,
        depths_per_t=depths_per_t,
        camera_K=camera_K,
        camera_render_size=(out_h, out_w),
        save_points=save_points,
        save_depth=save_depth,
        save_rgb=save_rgb,
    )
    return str(hdf5_path), "ok"


def _convert_episode_worker(
    hdf5_path: Path,
    out_root: Path,
    task: str,
    demo_id: str,
    camera: str,
    num_points: int,
    max_depth_m: float,
    spec_id: str,
    split: str,
    save_points: bool,
    save_depth: bool,
    save_rgb: bool,
    image_height: int,
    image_width: int,
) -> tuple[str, str]:
    return convert_episode(
        hdf5_path, out_root, task, demo_id,
        camera=camera, num_points=num_points,
        max_depth_m=max_depth_m, spec_id=spec_id, split=split,
        save_points=save_points, save_depth=save_depth, save_rgb=save_rgb,
        image_height=image_height, image_width=image_width,
    )


def convert_task(
    robotwin_data: Path,
    task: str,
    task_config: str,
    out_root: Path,
    args: argparse.Namespace,
) -> None:
    task_data_dir = robotwin_data / task / task_config / "data"
    if not task_data_dir.is_dir():
        raise SystemExit(
            f"No RoboTwin data dir: {task_data_dir}. Collect first with "
            f"`bash collect_data.sh {task} {task_config} <gpu>`.")
    eps = _episode_files(task_data_dir)
    if args.max_episodes > 0:
        eps = eps[:args.max_episodes]
    if not eps:
        raise SystemExit(f"No episode*.hdf5 under {task_data_dir}")

    print(f"[{task}] {len(eps)} episodes from {task_data_dir}", flush=True)
    work = [
        (ep, out_root, task, f"demo_{i:06d}", args.camera, args.num_points,
         args.max_depth, args.spec_id, args.split,
         not args.no_points, not args.no_depth, not args.no_rgb,
         args.image_height, args.image_width)
        for i, ep in enumerate(eps)
    ]
    n_ok = 0
    if args.num_workers == 1:
        results = map(lambda item: _convert_episode_worker(*item), work)
        for i, (label, status) in enumerate(results, 1):
            if status == "ok":
                n_ok += 1
            else:
                print(f"  SKIP {label}: {status}", flush=True)
            if i % args.print_every == 0:
                print(f"  [{task}] {i}/{len(eps)} (ok={n_ok})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(_convert_episode_worker, *item) for item in work]
            for i, future in enumerate(as_completed(futures), 1):
                label, status = future.result()
                if status == "ok":
                    n_ok += 1
                else:
                    print(f"  SKIP {label}: {status}", flush=True)
                if i % args.print_every == 0:
                    print(f"  [{task}] {i}/{len(eps)} (ok={n_ok})", flush=True)
    print(f"[{task}] done: {n_ok}/{len(eps)} demos written to "
          f"{out_root / task}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robotwin_data", type=str, required=True,
                    help="RoboTwin save_path root (the dir holding <task>/...).")
    ap.add_argument("--task_config", type=str, required=True,
                    help="RoboTwin task_config name used at collection time "
                         "(e.g. wam_depth).")
    ap.add_argument("--tasks", type=str, nargs="+", required=True)
    ap.add_argument("--out_root", type=str, required=True,
                    help="Destination WAM (unpacked) root, e.g. data/robotwin_wam.")
    ap.add_argument("--camera", type=str, default="head_camera")
    ap.add_argument("--num_points", type=int, default=2000)
    ap.add_argument("--image-height", type=int, default=IMAGE_HEIGHT)
    ap.add_argument("--image-width", type=int, default=IMAGE_WIDTH)
    ap.add_argument("--max_depth", type=float, default=DEFAULT_MAX_DEPTH_M)
    ap.add_argument("--spec_id", type=str, default="robotwin_v1")
    ap.add_argument("--split", type=str, default="robotwin")
    ap.add_argument("--max_episodes", type=int, default=0,
                    help="0 = all episodes.")
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=1,
                    help="Parallel HDF5 conversion workers per task.")
    ap.add_argument("--no_points", action="store_true")
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--no_rgb", action="store_true")
    args = ap.parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if (args.image_height, args.image_width) != (IMAGE_HEIGHT, IMAGE_WIDTH):
        ap.error(
            f"RoboTwin output is fixed at {IMAGE_HEIGHT}x{IMAGE_WIDTH}")

    robotwin_data = Path(args.robotwin_data).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        convert_task(robotwin_data, task, args.task_config, out_root, args)


if __name__ == "__main__":
    main()
