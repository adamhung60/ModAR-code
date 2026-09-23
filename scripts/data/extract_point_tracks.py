#!/usr/bin/env python3
"""Extract CoTracker3 point tracks for ModAR trajectory packs.

For every demo we seed a patch-center grid of query points at the DINO/depth
patch centers and, for each stride-8 keyframe anchor `a`, track that grid
forward 32 native frames (clip `[a, a+32]`). We record the tracked positions
at the four future offsets `+8/+16/+24/+32` (the dynamics prediction stride),
storing absolute normalized image coords `(x, y) in [-1, 1]` plus a visibility
flag per point.

Output (one file per demo):
    <out_root>/<task>/<demo>/tracks.npz
      tracks   : (Ts, 4, N, 3) float16     -- [x_norm, y_norm, visible]
      valid    : (Ts,)           bool       -- anchor had a full 32-frame horizon
      seed_xy  : (N, 2)          float32    -- normalized grid seed (fixed)
      meta      (offsets, grid, stride, horizon, W, H)

`Ts` = number of stride-8 keyframes in the demo (a = 0, 8, 16, ...). Keyframes
whose horizon runs past the end of the clip get `valid=False` and zero tracks.

Parallelism: run one process per GPU with disjoint `--shard-index` over the
flattened (task, demo) work list; each process uses a multi-worker DataLoader
to decode the RGB PNGs while the GPU runs CoTracker.
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

STRIDE = 8
HORIZON = 32
OFFSETS = (8, 16, 24, 32)
PATCH = 14  # 224 / 16
IMG = 224
DEFAULT_CKPT = os.path.expanduser(
    os.environ.get(
        "COTRACKER_CHECKPOINT",
        "~/.cache/torch/hub/checkpoints/scaled_offline.pth",
    )
)


def seed_grid_pixels(
    image_height: int = IMG,
    image_width: int = IMG,
    patch_size: int = PATCH,
) -> np.ndarray:
    """Patch-center query grid in pixel ``(x, y)`` coordinates."""
    if image_height % patch_size or image_width % patch_size:
        raise ValueError("image dimensions must be divisible by patch size")
    xs = np.arange(image_width // patch_size) * patch_size + patch_size // 2
    ys = np.arange(image_height // patch_size) * patch_size + patch_size // 2
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float32)


def norm_xy(xy: torch.Tensor, w: int, h: int) -> torch.Tensor:
    out = torch.empty_like(xy)
    out[..., 0] = xy[..., 0] / (w - 1) * 2 - 1
    out[..., 1] = xy[..., 1] / (h - 1) * 2 - 1
    return out


class DemoFrames(Dataset):
    def __init__(self, demos):
        self.demos = demos  # list of (task, demo_name, demo_dir)

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, i):
        task, name, d = self.demos[i]
        paths = sorted(glob.glob(os.path.join(d, "rgb_*.png")))
        frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths], 0)
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # (T,3,H,W) uint8
        return task, name, t


def build_worklist(src_root, tasks, out_root, overwrite):
    demos = []
    for task in tasks:
        tdir = os.path.join(src_root, task)
        if not os.path.isdir(tdir):
            print(f"[warn] missing task dir {tdir}", flush=True)
            continue
        for name in sorted(os.listdir(tdir)):
            if not name.startswith("demo_"):
                continue
            d = os.path.join(tdir, name)
            out = os.path.join(out_root, task, name, "tracks.npz")
            if (not overwrite) and os.path.exists(out):
                continue
            demos.append((task, name, d))
    return demos


def query_grid(image_height, image_width, patch_size, device):
    """CoTracker query tensor (1,N,3) plus the seed grid in pixels."""
    seed_xy = seed_grid_pixels(image_height, image_width, patch_size)
    q = np.concatenate([np.zeros((seed_xy.shape[0], 1), np.float32), seed_xy], 1)
    return torch.from_numpy(q)[None].to(device), seed_xy


@torch.no_grad()
def tail_anchor_tracks(model, vid, keyframes, T, q_base, image_height,
                       image_width, tracks_out, offset_valid_out):
    """Track the anchors whose full horizon runs off the end of the demo.

    These used to be skipped outright, leaving the last few keyframes with no
    tracks at all. Since the loader can only place a decision frame where tracks
    exist, that is what stopped any tracks model from training on the end of a
    demo. Here each such anchor is tracked over the frames that remain and only
    the offsets that land inside the clip are marked valid. Clip lengths differ
    per anchor and there are at most (HORIZON - OFFSETS[0]) / STRIDE of them, so
    they run one at a time.
    """
    device = q_base.device
    for ki, a in enumerate(keyframes):
        if not a + OFFSETS[0] <= T - 1 < a + HORIZON:
            continue
        fits = [o for o in OFFSETS if a + o <= T - 1]
        tr, vis = model(vid[a:T].to(device)[None], queries=q_base)
        idx = torch.tensor(fits, device=device)
        trn = norm_xy(tr[0, idx], image_width, image_height)
        packed = torch.cat(
            [trn, vis[0, idx].float()[..., None]], -1).cpu().numpy()
        for j, o in enumerate(fits):
            tracks_out[ki, OFFSETS.index(o)] = packed[j].astype(np.float16)
            offset_valid_out[ki, OFFSETS.index(o)] = True


@torch.no_grad()
def process_demo(
    model,
    frames,
    anchor_batch,
    device,
    image_height: int = IMG,
    image_width: int = IMG,
    patch_size: int = PATCH,
):
    T = frames.shape[0]
    if tuple(frames.shape[-2:]) != (image_height, image_width):
        raise ValueError(
            f"frames are {tuple(frames.shape[-2:])}, expected "
            f"{(image_height, image_width)}")
    vid = frames.float()  # (T,3,H,W) on cpu
    keyframes = list(range(0, T, STRIDE))
    Ts = len(keyframes)
    q_base, seed_xy = query_grid(image_height, image_width, patch_size, device)
    n_points = seed_xy.shape[0]

    tracks_out = np.zeros((Ts, len(OFFSETS), n_points, 3), np.float16)
    offset_valid_out = np.zeros((Ts, len(OFFSETS)), bool)

    full = [(ki, a) for ki, a in enumerate(keyframes) if a + HORIZON <= T - 1]
    off = torch.tensor(OFFSETS, device=device)
    for s in range(0, len(full), anchor_batch):
        chunk = full[s:s + anchor_batch]
        clips = torch.stack([vid[a:a + HORIZON + 1] for _, a in chunk], 0).to(device)
        qb = q_base.repeat(len(chunk), 1, 1)
        tr, vis = model(clips, queries=qb)  # (B,33,N,2), (B,33,N)
        tr = tr[:, off]                     # (B,4,N,2)
        vis = vis[:, off].float()           # (B,4,N)
        trn = norm_xy(tr, image_width, image_height)
        packed = torch.cat([trn, vis[..., None]], -1).cpu().numpy().astype(np.float16)
        for j, (ki, _) in enumerate(chunk):
            tracks_out[ki] = packed[j]
            offset_valid_out[ki] = True

    tail_anchor_tracks(model, vid, keyframes, T, q_base, image_height,
                       image_width, tracks_out, offset_valid_out)

    valid_out = offset_valid_out.all(axis=1)
    seed_norm = seed_xy.copy()
    seed_norm[:, 0] = seed_norm[:, 0] / (image_width - 1) * 2 - 1
    seed_norm[:, 1] = seed_norm[:, 1] / (image_height - 1) * 2 - 1
    return (tracks_out, valid_out, seed_norm.astype(np.float32),
            offset_valid_out)


def main():
    from cotracker.predictor import CoTrackerPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--anchor-batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--image-height", type=int, default=None)
    ap.add_argument("--image-width", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=IMG,
                    help="Legacy square size when height/width are omitted.")
    ap.add_argument("--patch-size", type=int, default=PATCH)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        ap.error(
            f"CoTracker checkpoint not found: {checkpoint}. "
            "Set --checkpoint or COTRACKER_CHECKPOINT.")
    if (args.image_height is None) != (args.image_width is None):
        ap.error("--image-height and --image-width must be set together")
    image_height = int(
        args.image_height if args.image_height is not None else args.image_size)
    image_width = int(
        args.image_width if args.image_width is not None else args.image_size)
    if image_height % args.patch_size or image_width % args.patch_size:
        ap.error("image dimensions must be divisible by --patch-size")

    device = "cuda"
    model = CoTrackerPredictor(
        checkpoint=checkpoint, offline=True, v2=False, window_len=60
    ).to(device)
    model.eval()

    demos = build_worklist(args.src_root, args.tasks, args.out_root, args.overwrite)
    demos = demos[args.shard_index::args.num_shards]
    ds = DemoFrames(demos)
    dl = DataLoader(ds, batch_size=1, num_workers=args.num_workers,
                    collate_fn=lambda b: b[0], prefetch_factor=2 if args.num_workers else None)
    n = len(demos)
    print(f"[shard {args.shard_index}/{args.num_shards}] {n} demos to process", flush=True)

    t0 = time.time()
    done = 0
    for task, name, frames in dl:
        tracks, valid, seed, offset_valid = process_demo(
            model, frames, args.anchor_batch, device, image_height, image_width,
            args.patch_size)
        out_dir = os.path.join(args.out_root, task, name)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, "tracks.npz"),
            tracks=tracks, valid=valid, seed_xy=seed,
            offset_valid=offset_valid,
            offsets=np.array(OFFSETS, np.int32),
            grid=image_height // args.patch_size,
            grid_height=image_height // args.patch_size,
            grid_width=image_width // args.patch_size,
            stride=STRIDE, horizon=HORIZON,
            W=image_width, H=image_height, patch_size=args.patch_size,
        )
        done += 1
        if done % 25 == 0 or done == n:
            el = time.time() - t0
            rate = done / el
            eta = (n - done) / rate if rate > 0 else float("nan")
            print(f"[shard {args.shard_index}] {done}/{n}  "
                  f"{rate*60:.1f} demos/min  elapsed {el/60:.1f}m  ETA {eta/60:.1f}m  "
                  f"(last {task}/{name})", flush=True)
    print(f"[shard {args.shard_index}] DONE {done}/{n} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
