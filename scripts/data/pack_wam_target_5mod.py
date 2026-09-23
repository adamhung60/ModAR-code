#!/usr/bin/env python3
"""Pack intermediate demonstrations into the five-modality keyframe format.

Dynamics modalities (dino / depth / image / tracks) are stored ONLY at stride-8
keyframes (~1/8 the frames); actions + proprio stay dense at every native frame.
Depth is quantized uint16 PNG, RGB is uint8 JPEG (see `util.modality_forcing.codec`).

Sources:
  - RGB/depth/DINO/state : <src_root>/<task>/<demo>/  (dense per-frame files)
  - point tracks         : <tracks_root>/<task>/<demo>/tracks.npz

Output (one per demo): <out_root>/<task>/<demo>/trajectory.pt with keys:
  pack_version=2, keyframe_stride, n_track_offsets, track_grid, depth_max_m,
  traj_len, n_keyframes, keyframes(Ts native idx),
  actions(T,A) fp32, proprio(T,P) fp32,          # dense
  dino_tokens(Ts,N,384) fp16,                  # keyframes
  depths_png[list Ts bytes], images_jpg[list Ts bytes],
  point_tracks(Ts,4,256,3) fp16, track_valid(Ts,) bool, track_seed(256,2) fp32.

Idempotent: skips demos already packed (unless --force). Demos without a
tracks.npz yet are skipped (extraction still running) unless --require-tracks.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from util.modality_forcing.codec import encode_depth_u16_png, encode_rgb_jpeg  # noqa: E402
from PIL import Image  # noqa: E402

PACK_VERSION = 2
STRIDE = 8


def pack_one(src_demo: Path, tracks_demo: Path | None, dst_demo: Path,
             depth_max_m: float, jpeg_quality: int, force: bool,
             require_tracks: bool = True,
             dino_demo: Path | None = None) -> tuple[str, str]:
    label = str(src_demo)
    out_pack = dst_demo / "trajectory.pt"
    if out_pack.is_file() and not force:
        return label, "skipped"

    tracks_npz = (tracks_demo / "tracks.npz") if tracks_demo is not None else None
    if require_tracks and (tracks_npz is None or not tracks_npz.is_file()):
        return label, "no-tracks"

    dino_dir = dino_demo if dino_demo is not None else src_demo

    meta = json.loads((src_demo / "metadata.json").read_text())
    frames = meta["frames"]
    T = len(frames)
    keyframes = list(range(0, T, STRIDE))
    Ts = len(keyframes)

    # dense actions + proprio from per-frame state json
    actions = np.zeros((T, meta["action_dim"]), np.float32)
    proprio = np.zeros((T, meta["proprio_dim"]), np.float32)
    for i, fr in enumerate(frames):
        st = json.loads((src_demo / fr["state_file"]).read_text())
        actions[i] = st["action"]
        proprio[i] = st["proprio"]

    # Keyframe dynamics. Infer rectangular geometry and token dimensions from
    # the artifacts rather than assuming the legacy 224x224 / 16x16 layout.
    dino_frames: list[np.ndarray] = []
    depths_png: list[bytes] = []
    images_jpg: list[bytes] = []
    image_height = image_width = grid_height = grid_width = None
    for j, kf in enumerate(keyframes):
        fr = frames[kf]
        stem = Path(fr["rgb"]).stem.split("_", 1)[1]  # "0000"
        tok_path = dino_dir / f"dino_tokens_{stem}.pth"
        if not tok_path.is_file():
            tok_path = src_demo / f"dino_tokens_{stem}.pth"
        try:
            tok = torch.load(tok_path, weights_only=True)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"failed to load DINO tokens {tok_path}: {error}"
            ) from error
        tok_np = tok.to(torch.float16).cpu().numpy()
        if tok_np.ndim != 2:
            raise ValueError(f"{tok_path}: expected (patches, dim), got {tok_np.shape}")
        depth = np.load(src_demo / fr["depth_npy"]).astype(np.float32)
        rgb = np.asarray(Image.open(src_demo / fr["rgb"]).convert("RGB"))
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"{src_demo}: depth {depth.shape} != RGB {rgb.shape[:2]}")
        if image_height is None:
            image_height, image_width = map(int, depth.shape)
            if image_height % 14 or image_width % 14:
                raise ValueError(
                    f"{src_demo}: image size {(image_height, image_width)} is "
                    "not divisible by DINO patch size 14")
            grid_height = image_height // 14
            grid_width = image_width // 14
        elif depth.shape != (image_height, image_width):
            raise ValueError(
                f"{src_demo}: inconsistent keyframe size {depth.shape}")
        expected_tokens = grid_height * grid_width
        if tok_np.shape[0] != expected_tokens:
            raise ValueError(
                f"{tok_path}: {tok_np.shape[0]} tokens != "
                f"{grid_height}x{grid_width}={expected_tokens}")
        dino_frames.append(tok_np)
        depths_png.append(encode_depth_u16_png(depth, depth_max_m))
        images_jpg.append(encode_rgb_jpeg(rgb, jpeg_quality))
    dino = np.stack(dino_frames, axis=0)

    pack = {
        "pack_version": PACK_VERSION,
        "keyframe_stride": STRIDE,
        "depth_max_m": float(depth_max_m),
        "traj_len": T,
        "n_keyframes": Ts,
        "image_height": image_height,
        "image_width": image_width,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "keyframes": np.asarray(keyframes, np.int32),
        "actions": torch.from_numpy(actions),
        "proprio": torch.from_numpy(proprio),
        "dino_tokens": torch.from_numpy(dino),
        "depths_png": depths_png,
        "images_jpg": images_jpg,
    }
    meta_out = {
        "task": meta["task"], "language": meta.get("language"),
        "packed_version": PACK_VERSION, "keyframe_stride": STRIDE,
        "traj_len": T, "n_keyframes": Ts,
        "image_height": image_height, "image_width": image_width,
        "grid_height": grid_height, "grid_width": grid_width,
        "pack_source": str(src_demo.resolve()),
    }

    if require_tracks:
        tr = np.load(tracks_npz)
        point_tracks = tr["tracks"].astype(np.float16)
        track_valid = tr["valid"].astype(bool)
        track_seed = tr["seed_xy"].astype(np.float32)
        if point_tracks.shape[0] != Ts:
            return label, f"track/keyframe mismatch {point_tracks.shape[0]} vs {Ts}"
        track_grid_height = int(
            tr["grid_height"] if "grid_height" in tr else tr["grid"])
        track_grid_width = int(
            tr["grid_width"] if "grid_width" in tr else tr["grid"])
        if point_tracks.shape[2] != track_grid_height * track_grid_width:
            return label, (
                f"track geometry mismatch {point_tracks.shape[2]} vs "
                f"{track_grid_height}x{track_grid_width}")
        if (track_grid_height, track_grid_width) != (grid_height, grid_width):
            return label, (
                f"track/token grid mismatch "
                f"{track_grid_height}x{track_grid_width} vs "
                f"{grid_height}x{grid_width}")
        if "H" in tr and "W" in tr:
            track_hw = (int(tr["H"]), int(tr["W"]))
            if track_hw != (image_height, image_width):
                return label, (
                    f"track/image size mismatch {track_hw} vs "
                    f"{(image_height, image_width)}")
        track_offsets = (
            tr["offsets"].astype(np.int32) if "offsets" in tr
            else np.arange(1, point_tracks.shape[1] + 1, dtype=np.int32))
        pack.update({
            "n_track_offsets": int(point_tracks.shape[1]),
            "track_grid": track_grid_height,
            "track_grid_height": track_grid_height,
            "track_grid_width": track_grid_width,
            "track_offsets": track_offsets,
            "point_tracks": torch.from_numpy(point_tracks),
            "track_valid": torch.from_numpy(track_valid),
            "track_seed": torch.from_numpy(track_seed),
        })
        # Per-(keyframe, offset) validity. Tail anchors have only their nearer
        # offsets tracked, which the all-or-nothing track_valid cannot express.
        if "offset_valid" in tr:
            pack["track_offset_valid"] = torch.from_numpy(
                tr["offset_valid"].astype(bool))
        if "stride" in tr:
            pack["track_stride"] = int(tr["stride"])
        if "horizon" in tr:
            pack["track_horizon"] = int(tr["horizon"])
        meta_out["tracks_source"] = str(tracks_npz.resolve())
        meta_out["track_grid_height"] = track_grid_height
        meta_out["track_grid_width"] = track_grid_width
        meta_out["n_track_offsets"] = int(point_tracks.shape[1])
        meta_out["track_offsets"] = track_offsets.tolist()

    dst_demo.mkdir(parents=True, exist_ok=True)
    tmp = out_pack.with_suffix(".tmp")
    torch.save(pack, tmp)
    tmp.replace(out_pack)

    (dst_demo / "metadata.json").write_text(json.dumps(meta_out, indent=2))
    return label, "ok"


def _worker(a):
    (src, trk, dst, dmax, q, force, require_tracks, dino) = a
    tracks = Path(trk) if trk else None
    dino_demo = Path(dino) if dino else None
    return pack_one(Path(src), tracks, Path(dst), dmax, q, force, require_tracks,
                    dino_demo=dino_demo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--tracks-root", required=True)
    ap.add_argument("--dino-root", default="",
                    help="Optional root for dino_tokens_*.pth (default: same as --src-root)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--depth-max-m", type=float, default=10.0)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap demos per task (debug)")
    ap.add_argument(
        "--no-tracks", action="store_true",
        help="Pack dino/depth/image/action only (no CoTracker). Matches the "
             "original robotwin_wam_5mod_packed layout used when modalities "
             "exclude tracks.")
    args = ap.parse_args()
    require_tracks = not args.no_tracks
    dino_root = Path(args.dino_root) if args.dino_root else None

    work = []
    for task in args.tasks:
        tdir = Path(args.src_root) / task
        if not tdir.is_dir():
            print(f"[warn] missing {tdir}", flush=True)
            continue
        names = sorted(p.name for p in tdir.iterdir()
                       if p.is_dir() and p.name.startswith("demo_"))
        if args.limit:
            names = names[:args.limit]
        for name in names:
            trk = str(Path(args.tracks_root) / task / name) if require_tracks else ""
            dino = str(dino_root / task / name) if dino_root is not None else ""
            work.append((
                str(tdir / name),
                trk,
                str(Path(args.out_root) / task / name),
                args.depth_max_m, args.jpeg_quality, args.force, require_tracks,
                dino,
            ))
    print(f"packing {len(work)} demos -> {args.out_root} "
          f"(tracks={'on' if require_tracks else 'off'}"
          f", dino_root={args.dino_root or 'src'})", flush=True)

    if args.num_workers <= 1:
        res = [_worker(w) for w in tqdm(work)]
    else:
        with mp.Pool(args.num_workers) as pool:
            res = list(tqdm(pool.imap_unordered(_worker, work, chunksize=4), total=len(work)))

    from collections import Counter
    c = Counter(s for _, s in res)
    print("results:", dict(c), flush=True)
    soft = {"ok", "skipped"}
    if require_tracks:
        soft.add("no-tracks")  # still extracting; not a hard error
    errs = [(l, s) for l, s in res if s not in soft]
    for l, s in errs[:20]:
        print("  ERROR", l, s, flush=True)


if __name__ == "__main__":
    main()
