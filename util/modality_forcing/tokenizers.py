"""Per-modality tokenizers and readout heads for the Modality-Forcing WAM.

Each modality is embedded with a single linear layer (no bottleneck) into the
shared transformer width, plus a learned per-modality embedding. Readout heads
are linear x-prediction projections back to data space.

Depth maps are turned into a rectangular grid of non-overlapping patches (each
patch_size**2 values) so they align token-for-token with the DINO patch grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from util.modality_forcing.config import MFConfig


def patchify_depth(depth: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, F, H, W) metric/normalized depth -> (B, F, n_patches, patch*patch).

    Non-overlapping patches in row-major order (patch p -> row p//grid, col
    p%grid), each flattened to patch*patch values.
    """
    B, F, H, W = depth.shape
    if H % patch or W % patch:
        raise ValueError(f"depth shape {(H, W)} must be divisible by patch size {patch}")
    gh, gw = H // patch, W // patch
    x = depth.reshape(B * F, 1, H, W)
    # unfold into (B*F, patch*patch, n_patches)
    x = torch.nn.functional.unfold(x, kernel_size=patch, stride=patch)
    # -> (B*F, n_patches, patch*patch)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(B, F, gh * gw, patch * patch)


def unpatchify_depth(patches: torch.Tensor, patch: int, grid_height: int,
                     grid_width: int | None = None) -> torch.Tensor:
    """(B, F, n_patches, patch*patch) -> (B, F, H, W). Inverse of patchify_depth."""
    B, F, N, D = patches.shape
    grid_width = grid_height if grid_width is None else grid_width
    if N != grid_height * grid_width:
        raise ValueError(
            f"got {N} patches, expected {grid_height}x{grid_width}")
    H, W = grid_height * patch, grid_width * patch
    x = patches.reshape(B * F, N, D).transpose(1, 2)  # (B*F, patch*patch, N)
    x = torch.nn.functional.fold(
        x, output_size=(H, W), kernel_size=patch, stride=patch)
    return x.reshape(B, F, H, W)


def patchify_rgb(rgb: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, F, C, H, W) RGB -> (B, F, n_patches, C*patch*patch).

    Non-overlapping patches, row-major, channels interleaved per patch (matches
    torch.nn.functional.unfold's channel-major flatten so unpatchify_rgb inverts).
    """
    B, F, C, H, W = rgb.shape
    if H % patch or W % patch:
        raise ValueError(f"RGB shape {(H, W)} must be divisible by patch size {patch}")
    gh, gw = H // patch, W // patch
    x = rgb.reshape(B * F, C, H, W)
    x = torch.nn.functional.unfold(x, kernel_size=patch, stride=patch)  # (B*F, C*p*p, N)
    x = x.transpose(1, 2).contiguous()                                  # (B*F, N, C*p*p)
    return x.reshape(B, F, gh * gw, C * patch * patch)


def unpatchify_rgb(patches: torch.Tensor, patch: int, grid_height: int,
                   grid_width: int | None = None,
                   channels: int = 3) -> torch.Tensor:
    """(B, F, n_patches, C*patch*patch) -> (B, F, C, H, W). Inverse of patchify_rgb."""
    B, F, N, D = patches.shape
    grid_width = grid_height if grid_width is None else grid_width
    if N != grid_height * grid_width:
        raise ValueError(
            f"got {N} patches, expected {grid_height}x{grid_width}")
    H, W = grid_height * patch, grid_width * patch
    x = patches.reshape(B * F, N, D).transpose(1, 2)  # (B*F, C*p*p, N)
    x = torch.nn.functional.fold(
        x, output_size=(H, W), kernel_size=patch, stride=patch)
    return x.reshape(B, F, channels, H, W)


class ModalityEmbedders(nn.Module):
    """Linear embedders (data space -> dim) + learned modality embeddings, keyed
    by modality name and driven by the config modality registry."""

    def __init__(self, cfg: MFConfig):
        super().__init__()
        self.cfg = cfg
        specs = cfg.modality_specs()
        self.embed = nn.ModuleDict(
            {s.name: nn.Linear(s.data_dim, cfg.dim) for s in specs})
        self.order = {s.name: s.order for s in specs}
        self.modality_emb = nn.Parameter(torch.zeros(len(specs), cfg.dim))
        nn.init.normal_(self.modality_emb, std=0.02)

    def embed_one(self, name: str, z: torch.Tensor) -> torch.Tensor:
        """Embed one modality's data (data-space -> dim) + its modality embedding."""
        return self.embed[name](z) + self.modality_emb[self.order[name]]

    def forward(self, zs: dict) -> dict:
        """zs: {name: (B,...,data_dim)} -> {name: (B,...,dim)} with modality emb."""
        return {name: self.embed_one(name, z) for name, z in zs.items()}


class ReadoutHeads(nn.Module):
    """Linear x-prediction heads per modality, keyed by modality name. Each head
    reads from its modality's expert width, which may be narrower than the trunk."""

    def __init__(self, cfg: MFConfig):
        super().__init__()
        specs = cfg.modality_specs()
        self.head = nn.ModuleDict({
            s.name: nn.Linear(cfg.expert_dim_of(s.name), s.data_dim)
            for s in specs})

    def read_one(self, name: str, h: torch.Tensor) -> torch.Tensor:
        return self.head[name](h)

    def forward(self, hs: dict) -> dict:
        return {name: self.head[name](h) for name, h in hs.items()}
