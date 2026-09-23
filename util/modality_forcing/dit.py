"""Shared flow-matching DiT trunk for the Modality-Forcing WAM.

Standard adaLN-Zero diffusion transformer with full bidirectional self-attention
and axial RoPE (temporal frame index + 2D patch grid). The per-modality
denoising state and the global proprio condition are summed into a single adaLN
conditioning vector `c` (shape (B, dim)); the fixed token layout + temporal RoPE
let the model tell clean history apart from noisy future targets, so a single
global `c` suffices (no per-token modulation needed in v0).

"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.modality_forcing.config import MFConfig


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0
                       ) -> torch.Tensor:
    """Sinusoidal embedding of a continuous time t in [0, 1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.dim))


class ConditioningEmbed(nn.Module):
    """Per-modality time embeddings plus proprio/task conditioning."""

    def __init__(self, cfg: MFConfig):
        super().__init__()
        specs = cfg.modality_specs()
        self.modalities = tuple(s.name for s in specs)
        self.t = nn.ModuleDict({s.name: TimeEmbedder(cfg.dim) for s in specs})
        self.proprio = nn.Sequential(
            nn.Linear(cfg.proprio_dim, cfg.dim), nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim))
        self.task_emb = (nn.Embedding(cfg.n_tasks, cfg.dim)
                         if cfg.n_tasks > 0 else None)
        if self.task_emb is not None:
            nn.init.normal_(self.task_emb.weight, std=0.02)

    def _add_task(self, c, task_id):
        if self.task_emb is not None:
            c = c + self.task_emb(task_id)
        return c

    def _proprio(self, proprio):
        return self.proprio(proprio)

    def forward(self, times: dict, proprio, task_id=None):
        """times: {name: (B,)} for every active modality."""
        c = self._proprio(proprio)
        for name in self.modalities:
            c = c + self.t[name](times[name])
        return self._add_task(c, task_id)

    def disjoint(self, active: str, time: torch.Tensor, proprio,
                 task_id=None) -> torch.Tensor:
        """Condition a reduced disjoint pass on its sole active modality."""
        c = self._proprio(proprio) + self.t[active](time)
        return self._add_task(c, task_id)

    def causal_base(self, proprio, task_id=None) -> torch.Tensor:
        """Rollout-invariant conditioning shared by every causal block."""
        return self._add_task(self._proprio(proprio), task_id)

    def causal_block(self, name: str, time: torch.Tensor,
                     base: torch.Tensor) -> torch.Tensor:
        """Condition one history/clean/query block at its own diffusion time."""
        return base + self.t[name](time)


def _rope_apply_axis(x: torch.Tensor, pos: torch.Tensor, n: int,
                     offset: int, max_period: float = 10000.0) -> torch.Tensor:
    """Apply interleaved RoPE to dims [offset:offset+n] of x using positions pos.

    x: (..., head_dim). pos: (L,). Rotates consecutive pairs within the chunk.

    REFERENCE IMPLEMENTATION. AxialRoPE below computes the same thing an order of
    magnitude more cheaply; this stays as the definition the equivalence test in
    tests/test_rope_tables.py checks against.
    """
    if n == 0:
        return x
    half = n // 2
    inv_freq = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.float()[:, None] * inv_freq[None, :]      # (L, half)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    # broadcast over (..., L, n): x is (B, H, L, head_dim)
    chunk = x[..., offset:offset + n]
    x_even = chunk[..., 0::2]
    x_odd = chunk[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    rotated = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    return torch.cat([x[..., :offset], rotated, x[..., offset + n:]], dim=-1)


class AxialRoPE(nn.Module):
    """Axial RoPE over (temporal, height, width) with a per-axis dim split.

    ONE rotation over the whole head_dim against precomputed cos/sin tables,
    rather than three per-axis passes that each rebuild their trig from scratch.

    WHY THE AXES FUSE. `_rope_apply_axis` rotates interleaved pairs *inside* the
    chunk [offset, offset+n), and every split entry is even, so chunk k covers
    exactly global pairs [offset_k/2, (offset_k+n_k)/2) with no pair straddling a
    boundary. Viewing the head as x[..., 0::2] / x[..., 1::2] therefore lines the
    pairs up in axis order, and concatenating the three per-axis tables gives
    every pair the same frequency the per-axis code gave it. Same arithmetic, one
    pass: the three full-width `torch.cat` reassemblies (each a copy of the whole
    (B, H, L, head_dim) tensor) collapse into a single `stack`.

    WHY TABLES ARE SAFE HERE. cos/sin depend only on the position VALUE, and
    every position the layouts emit is drawn from cfg.obs_clock(),
    cfg.action_clock(), or the h/w patch grids -- all config-derived with small
    bounds, which is what makes a dense table indexed by position exact rather
    than an approximation. Sizing the tables from those same clocks is what keeps
    that true if the horizon config changes. Caching keyed on the position TENSOR
    would not be safe: the block-causal and disjoint paths rebuild their position
    vectors every forward, so an identity-keyed cache would never hit, would grow
    without bound, and could hand back another layout's table once an address got
    reused.

    Tables are built per device on first use and kept as plain attributes rather
    than buffers on purpose: they are deterministic constants, so they need no
    DDP broadcast, and registering ~6 of them on each of a dozen-odd attention
    modules would add that many tensors to every DDP buffer sync.
    """

    def __init__(self, cfg: MFConfig, max_period: float = 10000.0):
        super().__init__()
        self.split = tuple(cfg.rope_split)  # (nt, nh, nw)
        self.max_period = max_period
        # Extents come from the clocks the positions are actually drawn from.
        # Temporal covers both the obs keyframe clock and the dense action clock.
        self.extent = (
            max(max(cfg.obs_clock()), max(cfg.action_clock())) + 1,
            cfg.grid_h,
            cfg.grid_w,
        )
        self._tables: dict = {}

    def _tables_for(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin) of shape (extent_axis, n_axis/2) per axis, concatenated on
        demand by ``apply``. Built on ``device`` so the trig matches what the
        reference produced there."""
        key = str(device)
        got = self._tables.get(key)
        if got is not None:
            return got
        cos, sin = [], []
        for n, extent in zip(self.split, self.extent):
            half = n // 2
            if half == 0:
                continue
            inv_freq = torch.exp(
                -math.log(self.max_period)
                * torch.arange(half, device=device, dtype=torch.float32) / half)
            pos = torch.arange(extent, device=device, dtype=torch.float32)
            ang = pos[:, None] * inv_freq[None, :]
            cos.append(torch.cos(ang))
            sin.append(torch.sin(ang))
        got = (cos, sin)
        self._tables[key] = got
        return got

    def apply(self, x, pos_t, pos_h, pos_w):
        cos_tab, sin_tab = self._tables_for(x.device)
        if not cos_tab:
            return x
        pos = [p for p, n in zip((pos_t, pos_h, pos_w), self.split) if n > 0]
        cos = torch.cat([t[p] for t, p in zip(cos_tab, pos)], dim=-1)
        sin = torch.cat([t[p] for t, p in zip(sin_tab, pos)], dim=-1)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack([x_even * cos - x_odd * sin,
                            x_even * sin + x_odd * cos], dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    """Per-feature RMS normalization (used for JiT-style QK-norm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * self.weight



def _token_broadcast(x):
    return x.unsqueeze(1) if x.dim() == 2 else x


def modulate(x, shift, scale):
    return x * (1 + _token_broadcast(scale)) + _token_broadcast(shift)


class Attention(nn.Module):
    """Self-attention at an arbitrary width. head_dim is fixed by the config so
    every width shares one rope_split; only the head count varies."""

    def __init__(self, cfg: MFConfig, dim: Optional[int] = None):
        super().__init__()
        dim = cfg.dim if dim is None else dim
        self.head_dim = cfg.head_dim
        self.n_heads = dim // self.head_dim
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = AxialRoPE(cfg)
        self.dropout = cfg.dropout
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()

    def _project_qkv(self, x, pos_t, pos_h, pos_w):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope.apply(self.q_norm(q), pos_t, pos_h, pos_w)
        k = self.rope.apply(self.k_norm(k), pos_t, pos_h, pos_w)
        return q, k, v

    def forward(self, x, pos_t, pos_h, pos_w, attn_mask=None):
        B, L, D = x.shape
        q, k, v = self._project_qkv(x, pos_t, pos_h, pos_w)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))

    def forward_cached(self, x, pos_t, pos_h, pos_w,
                       prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
                       append: bool):
        """Attend a bidirectional block to an immutable cached prefix."""
        B, L, D = x.shape
        q, k, v = self._project_qkv(x, pos_t, pos_h, pos_w)
        if prefix_kv is None:
            full_k, full_v = k, v
        else:
            full_k = torch.cat((prefix_kv[0], k), dim=2)
            full_v = torch.cat((prefix_kv[1], v), dim=2)
        out = F.scaled_dot_product_attention(
            q, full_k, full_v,
            dropout_p=self.dropout if self.training else 0.0)
        out = self.proj(out.transpose(1, 2).reshape(B, L, D))
        return out, ((full_k, full_v) if append else prefix_kv)


class DiTBlock(nn.Module):
    """One adaLN-Zero block. ``dim`` is the block width; ``cond_dim`` is the
    shared conditioning width, which stays at the trunk width for a narrower block."""

    def __init__(self, cfg: MFConfig, dim: Optional[int] = None,
                 cond_dim: Optional[int] = None):
        super().__init__()
        dim = cfg.dim if dim is None else dim
        cond_dim = cfg.dim if cond_dim is None else cond_dim
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(cfg, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"),
            nn.Dropout(cfg.dropout), nn.Linear(hidden, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def _mods(self, c):
        return self.adaLN(c).chunk(6, dim=-1)

    def forward(self, x, c, pos_t, pos_h, pos_w, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._mods(c)
        attn_out = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            pos_t, pos_h, pos_w, attn_mask)
        x = x + _token_broadcast(gate_msa) * attn_out
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + _token_broadcast(gate_mlp) * mlp_out

    def forward_cached(self, x, c, pos_t, pos_h, pos_w, prefix_kv, append):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._mods(c)
        attn_out, new_kv = self.attn.forward_cached(
            modulate(self.norm1(x), shift_msa, scale_msa),
            pos_t, pos_h, pos_w, prefix_kv, append)
        x = x + _token_broadcast(gate_msa) * attn_out
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + _token_broadcast(gate_mlp) * mlp_out, new_kv


class FinalLayer(nn.Module):
    """adaLN-modulated final norm. Per-modality heads are applied after this."""

    def __init__(self, cfg: MFConfig, dim: Optional[int] = None,
                 cond_dim: Optional[int] = None):
        super().__init__()
        dim = cfg.dim if dim is None else dim
        cond_dim = cfg.dim if cond_dim is None else cond_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class ModalityForcingDiT(nn.Module):
    """adaLN-Zero DiT: a shared trunk followed by per-modality expert layers."""

    def __init__(self, cfg: MFConfig):
        super().__init__()
        self.cfg = cfg
        self.modalities = tuple(cfg.modalities)
        self.cond = ConditioningEmbed(cfg)
        self.shared = nn.ModuleList(
            [DiTBlock(cfg) for _ in range(cfg.n_shared_layers)])
        self.expert_in = nn.ModuleDict({
            m: (nn.Identity() if cfg.expert_dim_of(m) == cfg.dim
                else nn.Linear(cfg.dim, cfg.expert_dim_of(m)))
            for m in self.modalities
        })
        self.experts = nn.ModuleDict({
            m: nn.ModuleList(
                [DiTBlock(cfg, dim=cfg.expert_dim_of(m), cond_dim=cfg.dim)
                 for _ in range(cfg.n_expert_layers_of(m))])
            for m in self.modalities
        })
        self.finals = nn.ModuleDict({
            m: FinalLayer(cfg, dim=cfg.expert_dim_of(m), cond_dim=cfg.dim)
            for m in self.modalities})

    def make_cond(self, times: dict, proprio, task_id=None):
        return self.cond(times, proprio, task_id)

    def make_cond_disjoint(self, active, time, proprio, task_id=None):
        return self.cond.disjoint(active, time, proprio, task_id)

    def make_cond_causal_base(self, proprio, task_id=None):
        return self.cond.causal_base(proprio, task_id)

    def make_cond_causal_block(self, name, time, base):
        return self.cond.causal_block(name, time, base)

    def forward(self, tokens, c, pos_t, pos_h, pos_w, segments, attn_mask=None,
                segment_conds=None):
        """Run the shared trunk, then each modality's expert stack.

        ``segments`` is a list of (modality_name, start, end). ``attn_mask``
        (L, L bool, True=attend) restricts attention in the shared trunk only.
        """
        x = tokens
        for blk in self.shared:
            x = blk(x, c, pos_t, pos_h, pos_w, attn_mask)
        out = {}
        for name, s, e in segments:
            xm = self.expert_in[name](x[:, s:e])
            pt, ph, pw = pos_t[s:e], pos_h[s:e], pos_w[s:e]
            cm = c if segment_conds is None else segment_conds[name]
            for blk in self.experts[name]:
                xm = blk(xm, cm, pt, ph, pw)
            out[name] = self.finals[name](xm, cm)
        return out

    def forward_prefix(self, tokens, c_tokens, pos_t, pos_h, pos_w, cache=None):
        """Prefill or append one clean prefix block, returning per-layer K/V."""
        if cache is None:
            cache = [None] * len(self.shared)
        if len(cache) != len(self.shared):
            raise ValueError("cache layer count does not match shared trunk")
        x = tokens
        new_cache = []
        for blk, layer_cache in zip(self.shared, cache):
            x, kv = blk.forward_cached(
                x, c_tokens, pos_t, pos_h, pos_w, layer_cache, True)
            new_cache.append(kv)
        return x, new_cache

    def forward_query(self, tokens, c_tokens, pos_t, pos_h, pos_w,
                      name, c_query, cache):
        """Run an active query block without mutating the clean-prefix cache."""
        if cache is None:
            cache = [None] * len(self.shared)
        if len(cache) != len(self.shared):
            raise ValueError("cache layer count does not match shared trunk")
        x = tokens
        for blk, layer_cache in zip(self.shared, cache):
            x, _ = blk.forward_cached(
                x, c_tokens, pos_t, pos_h, pos_w, layer_cache, False)
        x = self.expert_in[name](x)
        for blk in self.experts[name]:
            x = blk(x, c_query, pos_t, pos_h, pos_w)
        return self.finals[name](x, c_query)
