"""Depth-map normalization.

Clamp metric depth into a valid range, apply a ``linear`` or ``log`` transform,
then z-score with statistics computed in that transformed space. ``log`` is the
RoboTwin default. Use :func:`orient_depth_map_for_display` before drawing a depth
heatmap next to RGB; leave maps used for geometry in their stored row order.
"""
from __future__ import annotations

import torch

MIN_DEPTH_M = 1.0e-4

VALID_DEPTH_NORM_MODES = ("linear", "log")


def clamp_depth_maps(
    depths: torch.Tensor,
    max_depth_m: float,
    min_depth_m: float = MIN_DEPTH_M,
) -> torch.Tensor:
    """Fill non-finite depth and clip to ``[min_depth_m, max_depth_m]``."""
    out = torch.nan_to_num(
        depths,
        nan=float(max_depth_m),
        posinf=float(max_depth_m),
        neginf=float(min_depth_m),
    )
    return out.clamp(min=float(min_depth_m), max=float(max_depth_m))


def transform_depth_maps(
    clamped: torch.Tensor,
    mode: str = "log",
) -> torch.Tensor:
    """Apply the value transform to already-clamped metric depth.

    Args:
        clamped: depth tensor already clamped to ``[min_depth_m, max_depth_m]``
            so all values are strictly positive (safe for ``log``).
        mode: ``"linear"`` (identity) or ``"log"`` (natural log).
    """
    if mode == "linear":
        return clamped
    if mode == "log":
        return torch.log(clamped)
    raise ValueError(
        f"depth_norm_mode must be one of {VALID_DEPTH_NORM_MODES}, got {mode!r}")


def inverse_transform_depth_maps(
    transformed: torch.Tensor,
    mode: str = "log",
) -> torch.Tensor:
    """Invert ``transform_depth_maps`` back to metric depth (for viz)."""
    if mode == "linear":
        return transformed
    if mode == "log":
        return torch.exp(transformed)
    raise ValueError(
        f"depth_norm_mode must be one of {VALID_DEPTH_NORM_MODES}, got {mode!r}")


def normalize_depth_maps(
    depths: torch.Tensor,
    mean: float,
    std: float,
    max_depth_m: float = 10.0,
    eps: float = 1e-6,
    mode: str = "log",
) -> torch.Tensor:
    """Clamp to valid metric range, apply value transform, then global z-score.

    Invalid / far-plane pixels are folded to ``max_depth_m`` before
    transform + normalization so the encoder sees a single depth channel (no
    mask). ``mean`` / ``std`` must be computed in the same ``mode`` space.
    """
    clamped = clamp_depth_maps(depths, max_depth_m=max_depth_m)
    transformed = transform_depth_maps(clamped, mode=mode)
    inv_std = 1.0 / max(float(std), eps)
    return (transformed - float(mean)) * inv_std


def normalize_depth_maps_single(
    depth_hw: torch.Tensor,
    mean: float,
    std: float,
    max_depth_m: float = 10.0,
    eps: float = 1e-6,
    mode: str = "log",
) -> torch.Tensor:
    """Normalize a single (H, W) depth map."""
    return normalize_depth_maps(
        depth_hw.unsqueeze(0),
        mean, std, max_depth_m=max_depth_m, eps=eps, mode=mode,
    ).squeeze(0)


def denormalize_depth_maps(
    normed: torch.Tensor,
    mean: float,
    std: float,
    mode: str = "log",
) -> torch.Tensor:
    """Invert ``normalize_depth_maps`` back to metric depth (metres).

    Useful for visualization: undo the z-score and the value transform so the
    map can be displayed on a metric colour scale. Note clamping is not undone
    (it is lossy), so values reflect the clamped metric range.
    """
    transformed = normed * float(std) + float(mean)
    return inverse_transform_depth_maps(transformed, mode=mode)


def orient_depth_map_for_display(depth: torch.Tensor | "np.ndarray") -> "np.ndarray":
    """Flip a bottom-up depth map so it matches RGB orientation for display.

    Apply before ``imshow`` / GIF / WandB panels shown next to RGB. Do **not**
    apply before pinhole back-projection geometry (see
    :func:`sample_rgb_at_depth_buffer_pixels` for RGB lookup instead).
    RoboTwin depth is already RGB-aligned -- skip this flip there
    (``data.depth_flip_for_display: false``).
    """
    import numpy as np

    d = np.asarray(depth)
    if d.ndim < 2:
        raise ValueError(
            f"orient_depth_map_for_display expects (H, W) or (..., H, W), "
            f"got shape {d.shape}")
    return np.flip(d, axis=-2)


def depth_buffer_row_to_rgb_row(
    row: "np.ndarray | int",
    height: int,
) -> "np.ndarray | int":
    """Map a depth sidecar row index to the matching RGB PNG row index."""
    import numpy as np

    h = int(height)
    if isinstance(row, np.ndarray):
        return (h - 1) - row
    return (h - 1) - int(row)


def sample_rgb_at_depth_buffer_pixels(
    rgb: "np.ndarray",
    row: "np.ndarray",
    col: "np.ndarray",
) -> "np.ndarray":
    """Sample display-oriented RGB at depth-buffer ``(row, col)`` indices."""
    import numpy as np

    rgb = np.asarray(rgb)
    row = np.asarray(row)
    col = np.asarray(col)
    rgb_row = depth_buffer_row_to_rgb_row(row, rgb.shape[0])
    return rgb[rgb_row, col]
