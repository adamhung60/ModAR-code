"""Write RoboTwin trajectories in the ModAR intermediate on-disk layout.

Per-demo layout consumed by ``scripts/data/pack_wam_target_5mod.py``:

    <data_root>/<task>/<demo_id>/
        metadata.json            # frame manifest, proprio keys, camera_K
        frame_0000.pth           # (num_points, 3) float32 world-frame point cloud
        depth_0000.npy           # (H, W) float32 metric depth (metres)
        depth_valid_0000.npy     # (H, W) bool valid-depth mask
        state_0000.json          # action (14), proprio (14)
        rgb_0000.png             # third-person RGB matching the point cloud
        ...

Closed-loop evaluation replays RoboTwin seeds directly, so these intermediates
do not store a simulator state bundle. Offline training reads ``metadata.json``.

The 14-D vector is the dual-arm joint state (6 arm joints + 1 gripper per arm).
``proprio[t]`` is the joint vector at frame ``t``, and ``action[t]`` is the
absolute joint target at frame ``t+1``. The last frame repeats the final pose.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


# Farthest-point sampling with no simulator dependency, so this module loads in
# both the data environment and the RoboTwin evaluation environment.
def _fps_indices(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    n = points.shape[0]
    if num_samples >= n:
        return torch.arange(n, dtype=torch.long)
    selected = torch.empty(num_samples, dtype=torch.long)
    distances = torch.full((n,), float("inf"))
    farthest = 0
    for i in range(num_samples):
        selected[i] = farthest
        diff = points - points[farthest]
        dist = (diff * diff).sum(dim=1)
        distances = torch.minimum(distances, dist)
        farthest = int(torch.argmax(distances))
    return selected


def fps_downsample_xyz(pts: np.ndarray, num_target: int) -> np.ndarray:
    """FPS on (N,3) -> (num_target, 3) float32 (pads by resampling if N<target)."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros((num_target, 3), dtype=np.float32)
    if pts.shape[0] > num_target:
        idx = _fps_indices(torch.from_numpy(pts).float(), num_target)
        return pts[idx.numpy()].astype(np.float32)
    if pts.shape[0] < num_target:
        pad = num_target - pts.shape[0]
        rng = np.random.default_rng(0)
        pad_i = rng.choice(pts.shape[0], size=pad, replace=True)
        return np.concatenate([pts, pts[pad_i]], axis=0).astype(np.float32)
    return pts.astype(np.float32)


# Fixed proprio key ordering persisted in metadata so train + eval agree.
PROPRIO_KEYS: list[str] = ["joint_action_vector"]

# Metric-depth ceiling for valid pixels (metres). RoboTwin manipulation scenes
# are tabletop-scale; anything beyond this is background / far-plane garbage.
DEFAULT_MAX_DEPTH_M: float = 10.0


def joint_vectors_to_proprio_action(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(T, D) joint vectors -> (proprio (T,D), action (T,D)).

    proprio[t] = vector[t]; action[t] = vector[t+1] (last repeats). This is the
    next-step absolute-joint-target convention used by the RoboTwin baselines.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected (T, D) joint vectors, got {vectors.shape}")
    proprio = vectors.copy()
    action = np.empty_like(vectors)
    action[:-1] = vectors[1:]
    action[-1] = vectors[-1]
    return proprio, action


def depth_mm_to_metric(
    depth_mm: np.ndarray,
    out_height: int,
    out_width: int,
    max_depth_m: float = DEFAULT_MAX_DEPTH_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a full RoboTwin depth frame and convert millimetres to metres.

    RoboTwin stores depth in millimetres with invalid pixels set to 0 (masked by
    the render alpha; see ``RoboTwin/envs/camera/camera.py:get_depth``). We
    nearest-resize the uncropped frame (depth must not be blended across
    discontinuities), then build the validity mask expected by the encoder.
    """
    import cv2

    d = np.asarray(depth_mm, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"expected depth (H, W), got {d.shape}")
    d = cv2.resize(
        d, (int(out_width), int(out_height)), interpolation=cv2.INTER_NEAREST)
    dm = (d / 1000.0).astype(np.float32)
    valid = np.isfinite(dm) & (dm > 1.0e-4) & (dm <= float(max_depth_m))
    return dm, valid


def resize_rgb(rgb: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Uniformly resize a full RGB frame with area interpolation."""
    import cv2

    im = np.asarray(rgb)
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError(f"expected RGB (H, W, 3), got {im.shape}")
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    im = cv2.resize(
        im, (int(out_width), int(out_height)), interpolation=cv2.INTER_AREA)
    return im


def intrinsic_for_resize(
    intrinsic_cv: np.ndarray,
    src_h: int,
    src_w: int,
    out_height: int,
    out_width: int,
) -> np.ndarray:
    """Scale a pinhole intrinsic for an uncropped full-frame resize."""
    K = np.asarray(intrinsic_cv, dtype=np.float64).copy()
    scale_x = int(out_width) / float(src_w)
    scale_y = int(out_height) / float(src_h)
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y
    return K


def write_wam_demo_dir(
    data_root: Path,
    task: str,
    demo_id: str,
    *,
    rgbs: np.ndarray,
    proprio_matrix: np.ndarray,
    actions: np.ndarray,
    points_per_t: list,
    num_points: int,
    spec_id: str,
    source_hdf5: str,
    source_episode_index: int,
    split: str,
    depths_per_t: list | None = None,
    camera_K: np.ndarray | None = None,
    camera_render_size: tuple[int, int] | None = None,
    save_points: bool = True,
    save_depth: bool = True,
    save_rgb: bool = True,
) -> Path:
    """Write one RoboTwin demo in WAM format. Returns the demo directory."""
    demo_dir = Path(data_root) / task / demo_id
    demo_dir.mkdir(parents=True, exist_ok=True)

    t_max = int(rgbs.shape[0]) if save_rgb else int(proprio_matrix.shape[0])
    if proprio_matrix.shape[0] != t_max or actions.shape[0] != t_max:
        raise ValueError(
            f"proprio/actions length mismatch vs T={t_max}: "
            f"{proprio_matrix.shape[0]} / {actions.shape[0]}")
    if save_points and len(points_per_t) != t_max:
        raise ValueError(f"points_per_t {len(points_per_t)} != T={t_max}")
    if save_depth:
        if depths_per_t is None or len(depths_per_t) != t_max:
            raise ValueError("save_depth requires depths_per_t aligned to T")

    frames_meta: list = []
    for t in range(t_max):
        idx_str = f"{t:04d}"
        entry = {"idx": t}

        if save_points:
            fps_pts = fps_downsample_xyz(points_per_t[t], num_points)
            frame_pth = f"frame_{idx_str}.pth"
            torch.save(torch.from_numpy(fps_pts), demo_dir / frame_pth)
            entry["frame_pth"] = frame_pth

        state = {
            "action": np.asarray(actions[t], float).tolist(),
            "proprio": np.asarray(proprio_matrix[t], float).tolist(),
        }
        state_name = f"state_{idx_str}.json"
        (demo_dir / state_name).write_text(json.dumps(state))
        entry["state_file"] = state_name

        if save_rgb:
            rgb_name = f"rgb_{idx_str}.png"
            iio.imwrite(demo_dir / rgb_name, rgbs[t])
            entry["rgb"] = rgb_name

        if save_depth:
            dm, valid_hw = depths_per_t[t]
            depth_name = f"depth_{idx_str}.npy"
            valid_name = f"depth_valid_{idx_str}.npy"
            np.save(demo_dir / depth_name, np.asarray(dm, dtype=np.float32))
            np.save(demo_dir / valid_name, np.asarray(valid_hw, dtype=bool))
            entry["depth_npy"] = depth_name
            entry["depth_valid_npy"] = valid_name

        frames_meta.append(entry)

    meta = {
        "num_frames": t_max,
        "num_points": int(num_points),
        "task": task,
        "spec_id": spec_id,
        # NOTE: stored under a RoboTwin-specific key (not "source_hdf5") so the
        # packer's EgoDex hand-sidecar path (build_hand_sidecar, triggered by
        # meta["source_hdf5"] pointing at a real file) is not invoked -- RoboTwin
        # HDF5s have no hand "transforms" dataset.
        "source_robotwin_hdf5": source_hdf5,
        "source_episode_index": int(source_episode_index),
        "split": split,
        "frames": frames_meta,
        "point_cloud_frame": "world",
        "proprio_keys": list(PROPRIO_KEYS),
        "action_dim": int(actions.shape[-1]),
        "proprio_dim": int(proprio_matrix.shape[-1]),
        "benchmark": "robotwin",
    }
    if camera_K is not None:
        meta["camera_K"] = np.asarray(camera_K, dtype=float).tolist()
    if camera_render_size is not None:
        meta["camera_render_size"] = [int(camera_render_size[0]),
                                      int(camera_render_size[1])]

    (demo_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return demo_dir
