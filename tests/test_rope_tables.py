"""AxialRoPE's table path must equal the per-axis reference exactly.

AxialRoPE was rewritten from three per-axis passes into one rotation against
precomputed cos/sin tables, because the old form rebuilt its trig on every call
and reassembled the full head with a `torch.cat` per axis -- thousands of tiny
kernels per step, on a model that is already starved for launch bandwidth. The
rewrite is only worth having if it is the SAME function, so that is what these
check: not "close enough", but equal, on the geometries the repo actually trains
and on the odd ones (rectangular grids, a zero-width axis) that would expose an
off-by-one in the pair-to-axis mapping.

The other thing worth pinning is the table extent. Tables are indexed BY position
value, so a table sized from the wrong clock would either throw or, worse, read a
neighbouring row. The extents come from cfg.obs_clock() / cfg.action_clock() /
the patch grid, and the test asserts every position the layouts can emit lands
inside them.
"""
from __future__ import annotations

import math

import pytest
import torch

from util.modality_forcing.config import MFConfig
from util.modality_forcing.dit import AxialRoPE, _rope_apply_axis
from util.modality_forcing.model import ModalityForcingWAM


def reference(x, pos_t, pos_h, pos_w, split):
    """The pre-rewrite implementation, axis by axis."""
    nt, nh, nw = split
    x = _rope_apply_axis(x, pos_t, nt, 0)
    x = _rope_apply_axis(x, pos_h, nh, nt)
    x = _rope_apply_axis(x, pos_w, nw, nt + nh)
    return x


def geometry(**over):
    base = dict(dim=384, n_heads=6, rope_split=(24, 20, 20),
                modalities=("dino", "depth", "action"),
                grid_height=12, grid_width=16,
                image_height=168, image_width=224,
                obs_history=1, obs_future=2, obs_stride=8, action_horizon=16)
    base.update(over)
    return MFConfig(**base)


def small(**over):
    """A narrow-head geometry; the depth readout must stay non-compressive, so
    the patch has to shrink with the width."""
    base = dict(dim=32, n_heads=2, modalities=("dino", "depth", "action"),
                grid=2, depth_img_size=4, depth_patch_size=2, image_patch_size=2,
                track_grid=2, dino_dim=8, action_dim=3, proprio_dim=4,
                obs_history=1, obs_future=1, obs_stride=4, action_horizon=2,
                n_shared_layers=2, n_expert_layers=1)
    base.update(over)
    return MFConfig(**base)


CASES = {
    # the published 12x16 RoboTwin geometry
    "rect_12x16": geometry(),
    # square 16x16, the legacy 224x224 layout
    "square_16": geometry(grid_height=16, grid_width=16,
                          image_height=224, image_width=224),
    # the small split the unit tests use, with a narrower head
    "small_head": small(rope_split=(8, 4, 4)),
    # a zero-width axis: the pair mapping must simply skip it
    "no_temporal": small(rope_split=(0, 8, 8)),
    # only a temporal axis, the other end of the same edge case
    "temporal_only": small(rope_split=(16, 0, 0)),
    # deeper history + longer stride pushes the temporal clock well past the grid
    "long_clock": geometry(obs_history=3, obs_future=3, obs_stride=12,
                           action_horizon=24),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_table_rope_equals_per_axis_reference(name):
    cfg = CASES[name]
    torch.manual_seed(0)
    L, B, heads = 197, 3, cfg.dim // cfg.head_dim
    x = torch.randn(B, heads, L, cfg.head_dim)
    # positions drawn from the same clocks the layout builders use
    clock = torch.tensor(cfg.obs_clock() + cfg.action_clock())
    pos_t = clock[torch.randint(0, clock.numel(), (L,))]
    pos_h = torch.randint(0, cfg.grid_h, (L,))
    pos_w = torch.randint(0, cfg.grid_w, (L,))

    want = reference(x, pos_t, pos_h, pos_w, cfg.rope_split)
    got = AxialRoPE(cfg).apply(x, pos_t, pos_h, pos_w)

    assert got.shape == want.shape
    assert got.dtype == want.dtype, (
        "the rewrite must not change the dtype the attention sees: fp32 cos/sin "
        "promote bf16 activations, and SDPA downcasts them again")
    assert torch.equal(got, want), (
        f"{name}: max|diff| = {(got - want).abs().max().item():.3e}")


@pytest.mark.parametrize("name", sorted(CASES))
def test_bf16_inputs_match(name):
    """Training runs under bf16 autocast, so q/k reach RoPE in bf16."""
    cfg = CASES[name]
    torch.manual_seed(0)
    L = 64
    x = torch.randn(2, cfg.dim // cfg.head_dim, L, cfg.head_dim,
                    dtype=torch.bfloat16)
    clock = torch.tensor(cfg.obs_clock() + cfg.action_clock())
    pos_t = clock[torch.randint(0, clock.numel(), (L,))]
    pos_h = torch.randint(0, cfg.grid_h, (L,))
    pos_w = torch.randint(0, cfg.grid_w, (L,))
    want = reference(x, pos_t, pos_h, pos_w, cfg.rope_split)
    got = AxialRoPE(cfg).apply(x, pos_t, pos_h, pos_w)
    assert torch.equal(got, want)


def test_gradients_match():
    """The backward is where most of the removed kernels lived."""
    cfg = CASES["rect_12x16"]
    torch.manual_seed(0)
    L = 96
    base = torch.randn(2, 6, L, cfg.head_dim)
    clock = torch.tensor(cfg.obs_clock() + cfg.action_clock())
    pos_t = clock[torch.randint(0, clock.numel(), (L,))]
    pos_h = torch.randint(0, cfg.grid_h, (L,))
    pos_w = torch.randint(0, cfg.grid_w, (L,))

    grads = []
    for fn in (lambda z: reference(z, pos_t, pos_h, pos_w, cfg.rope_split),
               lambda z: AxialRoPE(cfg).apply(z, pos_t, pos_h, pos_w)):
        z = base.clone().requires_grad_(True)
        fn(z).square().sum().backward()
        grads.append(z.grad.clone())
    assert torch.equal(grads[0], grads[1])


@pytest.mark.parametrize("name", sorted(CASES))
def test_table_extent_covers_every_layout_position(name):
    """A table indexed by position must be sized for the largest one emitted.

    Undersizing is not a silent error on CPU (it throws) but it would be a
    correctness landmine, so the bound is checked directly rather than trusted.
    """
    cfg = CASES[name]
    rope = AxialRoPE(cfg)
    ext_t, ext_h, ext_w = rope.extent
    assert max(cfg.obs_clock()) < ext_t
    assert max(cfg.action_clock()) < ext_t
    assert cfg.grid_h - 1 < ext_h
    assert cfg.grid_w - 1 < ext_w


def test_extent_covers_real_model_position_buffers():
    """End to end: the fixed layout a real model builds must index in range."""
    cfg = CASES["rect_12x16"]
    torch.manual_seed(0)
    model = ModalityForcingWAM(cfg)
    rope = AxialRoPE(cfg)
    ext_t, ext_h, ext_w = rope.extent
    assert int(model.pos_t.max()) < ext_t
    assert int(model.pos_h.max()) < ext_h
    assert int(model.pos_w.max()) < ext_w
    # and the tables actually accept them
    x = torch.randn(1, 6, model.pos_t.numel(), cfg.head_dim)
    got = rope.apply(x, model.pos_t, model.pos_h, model.pos_w)
    want = reference(x, model.pos_t, model.pos_h, model.pos_w, cfg.rope_split)
    assert torch.equal(got, want)


def test_tables_are_not_module_buffers():
    """Constants, so they must not join every DDP buffer broadcast."""
    rope = AxialRoPE(CASES["rect_12x16"])
    rope.apply(torch.randn(1, 6, 8, 64),
               torch.zeros(8, dtype=torch.long),
               torch.zeros(8, dtype=torch.long),
               torch.zeros(8, dtype=torch.long))
    assert list(rope.buffers()) == []
    assert rope.state_dict() == {}
