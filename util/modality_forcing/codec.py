"""Encode/decode helpers for the keyframe 5-modality pack.

Depth is stored as quantized uint16 PNG (lossless, compact for smooth metric
depth); RGB as uint8 JPEG. Shared by the packer (`scripts/data/pack_wam_target_5mod.py`)
and the training loader (`util.modality_forcing.data`) so encode/decode stay in
lockstep.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

U16_MAX = 65535


def encode_depth_u16_png(depth_m: np.ndarray, max_depth_m: float) -> bytes:
    """Quantize metric depth (H,W float, metres) to uint16 and PNG-encode.

    Reconstruction: ``depth_m = code / 65535 * max_depth_m``. Values beyond
    `max_depth_m` clamp to the far plane (65535)."""
    d = np.clip(depth_m.astype(np.float32), 0.0, max_depth_m)
    code = np.round(d / max_depth_m * U16_MAX).astype(np.uint16)
    ok, enc = cv2.imencode(".png", code)
    if not ok:
        raise RuntimeError("cv2 PNG encode failed for depth")
    return enc.tobytes()


def decode_depth_u16_png(buf: bytes, max_depth_m: float) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("cv2 PNG decode failed for depth")
    return arr.astype(np.float32) / U16_MAX * max_depth_m


def encode_rgb_jpeg(rgb_u8: np.ndarray, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode_rgb_jpeg(buf: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))  # (H,W,3) uint8
