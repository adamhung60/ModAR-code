"""ModalityForcingWAM: the full world-action model (generic over modalities).

Assembles an ordered per-modality token sequence (clean in-context history +
noised future targets) driven by the config modality registry, runs the shared
adaLN DiT, and reads out per-modality x-predictions. Supports the legacy
3-modality set (dino, depth, action) and the extended 5-modality set
(dino, depth, image, tracks, action).

Modality kinds:
  - grid + history  (dino, depth, image): clean history frames + noised futures,
                     one rectangular token grid per present obs frame.
  - grid future-only (tracks): the seed is always the fixed grid, so no history/
                     current state is fed -- only the obs_future future frames are
                     denoised/predicted.
  - action         : dense per-step action tokens (no spatial grid).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from util.modality_forcing.config import MFConfig
from util.modality_forcing.dit import ModalityForcingDiT
from util.modality_forcing.scheduler import AutoregressiveScheduler
from util.modality_forcing.tokenizers import (
    ModalityEmbedders, ReadoutHeads, patchify_depth, unpatchify_depth,
    patchify_rgb, unpatchify_rgb)
# Robosuite OSC_POSE default per-unit output scales for the 6-DoF arm delta:
# position in meters, orientation (axis-angle) in radians. Used to report the
# action x-prediction error in physical units (mm / degrees).
_OSC_POS_SCALE_M = 0.05
_OSC_ROT_SCALE_RAD = 0.5

# Absolute JOINT-target action layouts (qpos control), keyed by action_dim:
# (arm dims in radians, gripper dims in their own native units). RoboTwin is
# the 14-D dual-arm case -- dims 0-5 / 7-12 are the two 6-DoF arms, 6 / 13 the
# grippers. The OSC mm/deg pair above reads dims 0:3 as a metres delta and
# 3:6 as an axis-angle delta, which is meaningless on this layout, so joint
# error gets its own readout rather than a reinterpretation of that one.
_JOINT_LAYOUTS = {
    14: ((0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12), (6, 13)),
    7: ((0, 1, 2, 3, 4, 5), (6,)),
}

_ALL_MODS = ("dino", "depth", "image", "tracks", "action")
_LEGACY_SPATIAL_BUFFERS = frozenset({"pos_t", "pos_h", "pos_w"})

# Qualified-name prefixes of ONE MODALITY'S PATHWAY: that modality's expert
# stack, its trunk-in projection, final norm and time embedder, its token
# embedder, and its readout head. Single source of truth for both
# freeze_to_experts_only() and the reinit-experts backbone loader, so the set
# that gets frozen is exactly the set that gets re-initialized. (The time
# embedder lives at dit.cond.t.<m>.*, NOT dit.t.<m>.*.)
_PATHWAY_PREFIX_TEMPLATES = (
    "dit.cond.t.{m}.",
    "dit.expert_in.{m}.",
    "dit.experts.{m}.",
    "dit.finals.{m}.",
    "embedders.embed.{m}.",
    "heads.head.{m}.",
)

_ACTION_PATHWAY_PREFIXES = tuple(
    t.format(m="action") for t in _PATHWAY_PREFIX_TEMPLATES)


def pathway_prefixes(modalities) -> tuple[str, ...]:
    """Qualified-name prefixes owned by ``modalities``, in template order."""
    return tuple(t.format(m=m)
                 for m in modalities for t in _PATHWAY_PREFIX_TEMPLATES)


def is_expert_pathway_key(name: str, modalities) -> bool:
    """True if a parameter name belongs to the pathway of any named modality."""
    return any(name.startswith(p) for p in pathway_prefixes(modalities))


def is_action_pathway_key(name: str) -> bool:
    """True if a parameter name belongs to the action pathway."""
    return is_expert_pathway_key(name, ("action",))


def spatially_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove only legacy derived spatial buffers and validate all real state."""
    cleaned = {
        key: value for key, value in state_dict.items()
        if key not in _LEGACY_SPATIAL_BUFFERS
    }
    expected = model.state_dict()
    missing = sorted(set(expected) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(cleaned)
        if expected[key].shape != cleaned[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "checkpoint state mismatch after spatial-buffer migration: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}")
    return cleaned


def load_spatially_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Strictly load learned state while regenerating geometry-dependent buffers."""
    model.load_state_dict(
        spatially_compatible_state_dict(model, state_dict), strict=True)


def _remap_modality_embedding(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    source_modalities,
) -> dict[str, torch.Tensor]:
    """Select learned modality-embedding rows by name for a subset warm start.

    Expert-pathway filtering handles removed modality-specific modules, but the
    shared ``embedders.modality_emb`` table has one row per active modality.
    Dropping image therefore changes its shape even though every retained row is
    directly reusable. Copy by name so action does not accidentally inherit the
    removed image row.
    """
    if not source_modalities:
        return state_dict
    key = "embedders.modality_emb"
    if key not in state_dict:
        return state_dict
    source_modalities = tuple(source_modalities)
    target_modalities = tuple(model.cfg.modalities)
    missing = [name for name in target_modalities if name not in source_modalities]
    if missing:
        raise RuntimeError(
            f"subset warm start target modalities absent from checkpoint: {missing}")
    rows = [source_modalities.index(name) for name in target_modalities]
    source = state_dict[key]
    if source.shape[0] != len(source_modalities):
        raise RuntimeError(
            "checkpoint modality embedding/config mismatch: "
            f"rows={source.shape[0]} modalities={list(source_modalities)}")
    remapped = dict(state_dict)
    remapped[key] = source[rows].clone()
    return remapped


def load_backbone_reinit_experts(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    modalities,
    *,
    source_modalities=None,
) -> list[str]:
    """Load the BACKBONE from a checkpoint while leaving the pathways of
    ``modalities`` at their fresh constructor init ("reinit experts"). Used by
    the warm-started capacity sweep and by the frozen-backbone / fresh-action
    finetune: the pretrained trunk is transplanted verbatim and the swept
    experts are trained from scratch on top of it.

    The swept modalities' keys (see is_expert_pathway_key) are dropped from the
    checkpoint so the model keeps its freshly initialized weights there, which
    also lets a differently *shaped* expert load a backbone whose expert shapes
    differ. Every model key that is NOT loaded must belong to a swept pathway;
    anything else missing, any unexpected key, or any shape mismatch on a loaded
    backbone key is a hard error (it would silently corrupt the backbone).

    Returns the sorted list of reinitialized (skipped) pathway keys.
    """
    modalities = tuple(modalities)
    cleaned = {k: v for k, v in state_dict.items()
               if k not in _LEGACY_SPATIAL_BUFFERS}
    cleaned = _remap_modality_embedding(
        model, cleaned, source_modalities=source_modalities)

    def swept(k: str) -> bool:
        return is_expert_pathway_key(k, modalities)

    load_sd = {k: v for k, v in cleaned.items() if not swept(k)}

    expected = model.state_dict()
    load_keys = set(load_sd)
    exp_keys = set(expected)
    missing = sorted(exp_keys - load_keys)
    unexpected = sorted(load_keys - exp_keys)
    mismatched = sorted(k for k in load_keys & exp_keys
                        if expected[k].shape != load_sd[k].shape)
    bad_missing = [k for k in missing if not swept(k)]
    if bad_missing or unexpected or mismatched:
        raise RuntimeError(
            f"backbone-only (reinit {list(modalities)}) load mismatch: "
            f"non-swept missing={bad_missing}, unexpected={unexpected}, "
            f"shape_mismatch={mismatched}")
    model.load_state_dict(load_sd, strict=False)
    return missing


def load_backbone_reinit_action(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> list[str]:
    return load_backbone_reinit_experts(model, state_dict, ("action",))


def backbone_reinit_ema_shadow(
    model: nn.Module,
    ckpt_ema: dict[str, torch.Tensor],
    modalities=("action",),
    *,
    source_modalities=None,
) -> dict[str, torch.Tensor]:
    """EMA shadow for a reinitialized-expert finetune.

    Backbone tensors come from the checkpoint. Reinitialized pathways keep the
    model's freshly initialized weights. Call after ``load_backbone_reinit_experts``.
    """
    modalities = tuple(modalities)
    shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema = {k: v for k, v in ckpt_ema.items() if k not in _LEGACY_SPATIAL_BUFFERS}
    ema = _remap_modality_embedding(
        model, ema, source_modalities=source_modalities)
    for k, v in ema.items():
        if (k in shadow and shadow[k].shape == v.shape
                and not is_expert_pathway_key(k, modalities)):
            shadow[k] = v.detach().clone().to(shadow[k])
    return shadow


class ModalityForcingWAM(nn.Module):
    def __init__(self, cfg: MFConfig):
        super().__init__()
        self.cfg = cfg
        self.embedders = ModalityEmbedders(cfg)
        self.dit = ModalityForcingDiT(cfg)
        self.heads = ReadoutHeads(cfg)
        self.scheduler = AutoregressiveScheduler(cfg)

        self.specs = cfg.modality_specs()
        self.mod_names = [s.name for s in self.specs]
        self.order = {s.name: s.order for s in self.specs}
        self.grid_names = [s.name for s in self.specs if s.kind == "grid"]
        # The dynamics modalities this run denoises. Usually every grid modality;
        # a single-target ablation names a subset and the rest stay clean history.
        self.gen_names = cfg.generated_modality_names()
        self.has_action = any(s.kind == "action" for s in self.specs)
        self.has_history = {s.name: s.has_history for s in self.specs}

        N, A = cfg.n_patches, cfg.action_horizon
        self.mode = cfg.schedule_mode
        # action_only drops the future obs frames (history-only context).
        self.obs_present = (cfg.obs_history if self.mode == "action_only"
                            else cfg.n_obs_frames)
        Fp = self.obs_present
        self.future_frames = ([] if self.mode == "action_only"
                              else list(range(cfg.obs_history, cfg.n_obs_frames)))
        self.n_hist_eff = cfg.obs_history

        # ---- fixed axial-RoPE building blocks (reused for causal assembly) ----
        obs_clock_full = torch.tensor(cfg.obs_clock(), dtype=torch.long)  # (F,)
        h_grid = torch.arange(N) // cfg.grid_w
        w_grid = torch.arange(N) % cfg.grid_w
        action_clock = torch.tensor(cfg.action_clock(), dtype=torch.long)
        self.register_buffer("obs_clock_full", obs_clock_full, persistent=False)
        self.register_buffer("h_grid", h_grid, persistent=False)
        self.register_buffer("w_grid", w_grid, persistent=False)
        self.register_buffer("action_clock", action_clock, persistent=False)

        # ---- per-modality frame layout ----
        # mod_frames[name]: global obs-frame indices this modality occupies (tensor
        #   order). hist_local / target_local: tensor-local indices of the clean
        #   history frames and the noised/supervised future frames.
        self.mod_frames = {}
        self.hist_local = {}
        self.target_local = {}
        for s in self.specs:
            if s.kind != "grid":
                continue
            if s.has_history:
                fr = list(range(Fp))
                self.hist_local[s.name] = list(range(cfg.obs_history))
                self.target_local[s.name] = [i for i in range(Fp)
                                             if i >= cfg.obs_history]
            else:  # future-only (tracks)
                fr = list(self.future_frames)
                self.hist_local[s.name] = []
                self.target_local[s.name] = list(range(len(fr)))
            self.mod_frames[s.name] = fr

        # ---- assemble the fixed token layout (segments + positions) ----
        self._build_generic_layout(N, A, obs_clock_full, h_grid, w_grid,
                                   action_clock)

        # No cross-modal attention masking in the shared trunk: 'disjoint' uses
        # the reduced causal path (one active modality per pass, no
        # future cross-conditioning), so it needs no full-sequence mask.
        self.register_buffer("_attn_mask", None, persistent=False)

        # Per-dim action normalization stats (for physical-unit error metrics).
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))

    # ---- layout builders --------------------------------------------------
    def _build_generic_layout(self, N, A, obs_clock_full, h_grid, w_grid,
                              action_clock):
        pt, ph, pw = [], [], []
        segments = []
        self.mod_nframes = {}
        cursor = 0
        for s in self.specs:
            if s.kind == "grid":
                fr = self.mod_frames[s.name]
                if len(fr) == 0:
                    continue                 # absent (e.g. tracks in action_only)
                self.mod_nframes[s.name] = len(fr)
                clk = obs_clock_full[torch.tensor(fr, dtype=torch.long)]
                pt.append(clk.repeat_interleave(N))
                ph.append(h_grid.repeat(len(fr)))
                pw.append(w_grid.repeat(len(fr)))
                blk = len(fr) * N
            else:  # action
                pt.append(action_clock)
                ph.append(torch.zeros(A, dtype=torch.long))
                pw.append(torch.zeros(A, dtype=torch.long))
                blk = A
            segments.append((s.name, cursor, cursor + blk))
            cursor += blk
        self.register_buffer("pos_t", torch.cat(pt), persistent=False)
        self.register_buffer("pos_h", torch.cat(ph), persistent=False)
        self.register_buffer("pos_w", torch.cat(pw), persistent=False)
        self.segments = segments
        self.seg_map = {name: (s, e) for name, s, e in segments}
        self.present_names = [name for name, _, _ in segments]
        self.L = cursor

    # ---- small helpers ----------------------------------------------------
    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.action_mean.copy_(mean.to(self.action_mean))
        self.action_std.copy_(std.to(self.action_std))

    def _action_phys_errors(self, x_action, actions, mask):
        std = self.action_std.float()
        mean = self.action_mean.float()
        pred = x_action.float() * std + mean
        gt = actions.float() * std + mean
        pos_mm = ((pred[..., :3] - gt[..., :3]) * _OSC_POS_SCALE_M).norm(dim=-1)
        rot_rad = ((pred[..., 3:6] - gt[..., 3:6]) * _OSC_ROT_SCALE_RAD).norm(dim=-1)
        pos_mm = pos_mm.mean(dim=-1) * 1000.0
        rot_deg = rot_rad.mean(dim=-1) * (180.0 / math.pi)
        denom = mask.sum().clamp(min=1)
        return (pos_mm * mask).sum() / denom, (rot_deg * mask).sum() / denom

    def _action_joint_errors(self, x_action, actions, mask):
        """Mean |error| on the revolute joints (degrees) and grippers (native).

        For qpos-control action spaces this is the physically readable number:
        it is denormalized, so it survives changes to action_mean/std, and it
        is directly comparable to how far the arm actually travels within a
        chunk. Returns ``(None, None)`` for layouts not in ``_JOINT_LAYOUTS``.
        """
        layout = _JOINT_LAYOUTS.get(int(x_action.shape[-1]))
        if layout is None:
            return None, None
        arm_idx, grip_idx = layout
        # The additive action_mean cancels in the difference; only the scale
        # matters, so this is (pred*std+mean) - (gt*std+mean) without the mean.
        diff = (x_action.float() - actions.float()) * self.action_std.float()
        arm = diff[..., arm_idx].abs().flatten(1).mean(dim=1) * (180.0 / math.pi)
        grip = diff[..., grip_idx].abs().flatten(1).mean(dim=1)
        denom = mask.sum().clamp(min=1)
        return (arm * mask).sum() / denom, (grip * mask).sum() / denom

    def _require_task_id(self, task_id):
        if self.cfg.n_tasks > 0 and task_id is None:
            raise ValueError("task_id is required when n_tasks > 0")
        return task_id

    def _stream_allowed(self, stream):
        if stream == "action":
            return {"action"}
        if stream == "dynamics":
            return set(self.grid_names)
        return None

    def _times(self, B, device, overrides=None, default=1.0):
        """Full {name: (B,)} time dict for every active modality (make_cond sums
        over all). default 1.0 = clean; pass overrides for active/noised ones."""
        d = {m: torch.full((B,), float(default), device=device)
             for m in self.mod_names}
        if overrides:
            d.update(overrides)
        return d

    # Reserved key under which `forward` stashes per-frame loss gates inside the
    # prepped `data` dict. Underscored so it can never collide with a modality.
    _FRAME_VALID_KEY = "_frame_valid"
    @staticmethod
    def _masked_loss(pred, target, weight, mask, frame_w=None):
        """Per-sample MSE reduced over the batch.

        ``mask`` (B,) gates whole samples for stream routing. ``frame_w`` (B,nf)
        additionally gates individual future frames, which is how windows near the
        end of a demo drop the frames the loader had to replicate-pad. A sample
        left with no valid frame leaves the batch denominator rather than
        contributing a 0/0.
        """
        se = (pred - target) ** 2
        if frame_w is None:
            err = se.flatten(1).mean(dim=1)
        else:
            w = frame_w.reshape(*frame_w.shape, *([1] * (se.dim() - 2)))
            w = w.to(se.dtype).expand_as(se)
            total = w.flatten(1).sum(dim=1)
            err = (se * w).flatten(1).sum(dim=1) / total.clamp(min=1e-6)
            mask = mask * (total > 0).to(mask.dtype)
        denom = mask.sum().clamp(min=1)
        return (err * weight * mask).sum() / denom

    def _denoise_xpred(self, name, z, t, net_out):
        """Convert network output to x-prediction (identity for x-param modes)."""
        if self.cfg.pred_type_of(name) != "v":
            return net_out
        if not torch.is_tensor(t):
            t = torch.full((z.shape[0],), float(t), device=z.device, dtype=z.dtype)
        while t.dim() < z.dim():
            t = t.unsqueeze(-1)
        return z + (1.0 - t) * net_out

    def _net_to_velocity(self, name, z, t_scalar, net_out, eps=1e-3):
        """Flow velocity for ODE integration from network output."""
        if self.cfg.pred_type_of(name) == "v":
            return net_out
        return (net_out - z) / max(1.0 - float(t_scalar), eps)

    def _action_x_for_pack(self, net_out, z_action, t_action):
        if net_out is None:
            return None
        return self._denoise_xpred("action", z_action, t_action, net_out)

    # ---- data prep --------------------------------------------------------
    def _prep_data(self, dino, depth_maps, actions, images=None,
                   point_tracks=None) -> dict:
        """Clean per-modality tensors, patchified for depth and RGB."""
        cfg = self.cfg
        Fp = self.obs_present
        data = {}
        for name in self.present_names:
            if name == "dino":
                data["dino"] = dino[:, :Fp]
            elif name == "depth":
                data["depth"] = patchify_depth(
                    depth_maps, cfg.depth_patch_size)[:, :Fp]
            elif name == "image":
                assert images is not None, "images modality active but not provided"
                data["image"] = patchify_rgb(
                    images, cfg.image_patch_size)[:, :Fp]
            elif name == "tracks":
                assert point_tracks is not None, (
                    "tracks modality active but point_tracks not provided")
                data["tracks"] = point_tracks
            elif name == "action":
                data["action"] = actions
        return data

    def _frame_valid(self, obs_future_valid, action_valid, track_future_valid):
        """Per-future-frame loss gates for windows running past the demo end.

        The loader replicate-pads those frames so tensor shapes stay fixed; this
        decides which of them carry gradient. Padded actions are supervised by
        default (``action_pad_mode='hold'`` teaches "reach the terminal pose and
        hold"), padded world-model targets never are.
        """
        cfg = self.cfg
        valid = {}
        for name in self.present_names:
            if name == "action":
                if cfg.action_pad_mode == "mask" and action_valid is not None:
                    valid["action"] = action_valid
                continue
            if cfg.obs_pad_supervise:
                continue
            src = track_future_valid if name == "tracks" else obs_future_valid
            if src is None:
                continue
            # action_only keeps history-only dynamics context, so those modalities
            # have an empty target set while the loader still emits obs_future
            # validity for the unused future slots.
            width = len(self.target_local[name])
            if width == 0:
                continue
            assert src.shape[1] == width, (
                f"{name} validity spans {src.shape[1]} frames but its target "
                f"spans {width}")
            valid[name] = src
        return valid

    def _effective_n(self, name, data, mask):
        """How many samples actually contributed to ``name``'s loss.

        A window whose futures are entirely replicate-padded carries no gradient
        for that modality. Counting it anyway would bias the n-weighted
        validation averages downward, since its loss is forced to zero.
        """
        frame_w = data.get(self._FRAME_VALID_KEY, {}).get(name)
        if frame_w is None:
            return mask.sum()
        return (mask * (frame_w.sum(dim=1) > 0).to(mask.dtype)).sum()

    # ---- core forward over the full (fixed-layout) sequence ---------------
    def _run(self, zs: dict, times: dict, proprio, task_id=None) -> dict:
        """zs: {name: latent}. grid -> (B,nf,N,d); action -> (B,A,d).
        times: {name: (B,)} for EVERY active modality. Returns {name: x_pred}."""
        B = proprio.shape[0]
        dim, N = self.cfg.dim, self.cfg.n_patches
        tok_blocks = []
        for name, _, _ in self.segments:
            t = self.embedders.embed_one(name, zs[name])
            tok_blocks.append(t.reshape(B, -1, dim) if name in self.grid_names else t)
        tokens = torch.cat(tok_blocks, dim=1)
        c = self.dit.make_cond(times, proprio, task_id)
        out = self.dit(tokens, c, self.pos_t, self.pos_h, self.pos_w,
                       self.segments, self._attn_mask)
        xs = {}
        for name, _, _ in self.segments:
            h = out[name]
            if name in self.grid_names:
                h = h.reshape(B, self.mod_nframes[name], N, -1)
            xs[name] = self.heads.read_one(name, h)
        return xs

    # ---- training dispatch ------------------------------------------------
    def forward(self, dino, depth_maps, actions, proprio, stream=None,
                task_id=None, images=None, point_tracks=None, loss_coeffs=None,
                obs_future_valid=None, action_valid=None,
                track_future_valid=None):
        """Training step (dispatches on schedule_mode).

        dino:(B,F,N,dino_dim); depth_maps:(B,F,H,W); actions:(B,A,act_dim);
        proprio:(B,proprio_dim); images:(B,F,C,H,W) [if image modality];
        point_tracks:(B,obs_future,N,track_dim) [if tracks modality]; all normalized.
        ``stream`` restricts supervision for two-stream training (action vs
        dynamics). ModAR uses the block-causal sequential engine.

        ``obs_future_valid`` (B,obs_future), ``track_future_valid`` (B,obs_future)
        and ``action_valid`` (B,action_horizon) mark which future frames are real
        rather than replicate-padded, for windows that run past the end of a demo.
        The loader emits them; they are absent for older callers, which then
        behave exactly as before.

        ``loss_coeffs`` maps modality -> loss coefficient for this data source (see
        :mod:`util.modality_forcing.grad_budget`), equalizing the action/dynamics
        gradient budget across schedule modes. ``None`` keeps each mode's own
        historical aggregation.
        """
        self._require_task_id(task_id)
        data = self._prep_data(
            dino, depth_maps, actions, images, point_tracks)
        data[self._FRAME_VALID_KEY] = self._frame_valid(
            obs_future_valid, action_valid, track_future_valid)
        if self.mode == "action_only":
            return self._forward_action_only(
                data, proprio, task_id, loss_coeffs)
        if self.mode == "unified":
            return self._forward_unified(
                data, proprio, stream, task_id, loss_coeffs)
        if self.mode == "independent":
            return self._forward_independent(
                data, proprio, stream, task_id, loss_coeffs)
        if self.mode == "disjoint":
            return self._forward_disjoint(
                data, proprio, stream, task_id, loss_coeffs)
        if self.mode == "modar":
            return self._forward_causal(
                data, proprio, stream, task_id, loss_coeffs)
        raise ValueError(f"unsupported schedule_mode {self.mode!r}")

    # ---- noising / loss helpers over the fixed layout ---------------------
    def _noise_futures(self, data, times, only=None):
        """Build zs from clean data by noising each present modality's targets at
        its time. ``only`` (set of names) restricts which modalities are noised
        (others are passed clean). Returns (zs, noises) where noises maps each
        noised modality to its epsilon draw."""
        zs, noises = {}, {}
        for name in self.present_names:
            if only is not None and name not in only:
                zs[name] = data[name]
                continue
            t = times[name]
            if name in self.grid_names:
                z = data[name].clone()
                tl = self.target_local[name]
                if tl:
                    noise = torch.randn_like(data[name][:, tl])
                    noises[name] = noise
                    z[:, tl] = self.scheduler.add_noise(
                        data[name][:, tl], t, noise)
                zs[name] = z
            else:  # action -> all tokens noised
                noise = torch.randn_like(data[name])
                noises[name] = noise
                zs[name] = self.scheduler.add_noise(data[name], t, noise)
        return zs, noises

    def _loss_for(self, name, xs, data, t, mask, noise=None):
        if name in self.grid_names:
            tl = self.target_local[name]
            pred = xs[name][:, tl]
            target_x = data[name][:, tl]
        else:
            pred = xs[name]
            target_x = data[name]
        if self.cfg.pred_type_of(name) == "v":
            assert noise is not None, f"v-pred for {name} requires noise"
            target = target_x - noise
            w = torch.ones_like(t)
        else:
            target = target_x
            w = self.scheduler.loss_weight(name, t)
        frame_w = data.get(self._FRAME_VALID_KEY, {}).get(name)
        return self._masked_loss(pred, target, w, mask, frame_w)

    def _loss_active(self, active, pred, data, t, mask, noise=None):
        """Loss on a future-only prediction from the causal path."""
        if active in self.grid_names:
            tl = self.target_local[active]
            target_x = data[active][:, tl]
        else:
            target_x = data["action"]
        if self.cfg.pred_type_of(active) == "v":
            assert noise is not None, f"v-pred for {active} requires noise"
            target = target_x - noise
            w = torch.ones_like(t)
        else:
            target = target_x
            w = self.scheduler.loss_weight(active, t)
        frame_w = data.get(self._FRAME_VALID_KEY, {}).get(active)
        return self._masked_loss(pred, target, w, mask, frame_w)

    def _combine_losses(self, losses, loss_coeffs, mean=False):
        """Reduce per-modality losses to the scalar training objective.

        ``loss_coeffs`` is the resolved gradient budget (see
        :mod:`util.modality_forcing.grad_budget`): a modality's coefficient is its
        target share of the objective divided by how often this stream actually
        supervises it, so every schedule mode spends the same fraction of its
        gradient on action. A modality absent from the mapping contributes nothing,
        which matches the zero masks the per-stream forwards already apply.

        ``loss_coeffs=None`` falls back to each mode's historical aggregation (a
        mean over supervised cuts for the block-causal path, a plain sum
        otherwise), so bare model calls -- viz, sampling, unit tests -- keep
        working without a resolver.
        """
        cfg = self.cfg
        if loss_coeffs is None:
            total = sum(cfg.lambda_of(n) * losses[n] for n in losses)
            return total / len(losses) if mean else total
        return sum(float(loss_coeffs.get(n, 0.0)) * cfg.lambda_of(n) * losses[n]
                   for n in losses)

    def _forward_independent(self, data, proprio, stream=None, task_id=None,
                             loss_coeffs=None):
        B = proprio.shape[0]
        device = proprio.device
        times = self.scheduler.sample_independent(B, device)
        zs, noises = self._noise_futures(data, times)
        xs = self._run(zs, times, proprio, task_id)
        ones = torch.ones(B, device=device)
        zeros = torch.zeros(B, device=device)
        losses, ns = {}, {}
        for name in self.present_names:
            if name == "action":
                m = ones if stream in (None, "action") else zeros
            else:
                m = ones if stream in (None, "dynamics") else zeros
            losses[name] = self._loss_for(
                name, xs, data, times[name], m, noise=noises.get(name))
            ns[name] = self._effective_n(name, data, m)
        total = self._combine_losses(losses, loss_coeffs)
        return self._pack(total, losses, ns, xs, data, device,
                          z_action=zs.get("action"), t_action=times.get("action"))

    def _forward_unified(self, data, proprio, stream=None, task_id=None,
                         loss_coeffs=None):
        B = proprio.shape[0]
        device = proprio.device
        t = self.scheduler.sample_unified(B, device)
        times = {m: t for m in self.mod_names}
        zs, noises = self._noise_futures(data, times)
        xs = self._run(zs, times, proprio, task_id)
        ones = torch.ones(B, device=device)
        zeros = torch.zeros(B, device=device)
        losses, ns = {}, {}
        for name in self.present_names:
            if name == "action":
                m = ones if stream in (None, "action") else zeros
            else:
                m = ones if stream in (None, "dynamics") else zeros
            losses[name] = self._loss_for(
                name, xs, data, t, m, noise=noises.get(name))
            ns[name] = self._effective_n(name, data, m)
        total = self._combine_losses(losses, loss_coeffs)
        return self._pack(total, losses, ns, xs, data, device,
                          z_action=zs.get("action"), t_action=times.get("action"))

    def _forward_disjoint(self, data, proprio, stream=None, task_id=None,
                          loss_coeffs=None):
        """Every modality (incl. action) is denoised in its own reduced-sequence
        pass from CLEAN HISTORY CONTEXT ONLY, never another modality's future."""
        return self._forward_disjoint_reduced(
            data, proprio, stream, task_id, loss_coeffs)

    def _forward_action_only(self, data, proprio, task_id=None,
                             loss_coeffs=None):
        """Clean obs history as context; denoise only the future action chunk."""
        B = proprio.shape[0]
        device = proprio.device
        ones = torch.ones(B, device=device)
        t_action = self.scheduler.sample_action(B, device)
        times = self._times(B, device, {"action": t_action}, default=1.0)
        zs, noises = self._noise_futures(data, times, only={"action"})
        xs = self._run(zs, times, proprio, task_id)
        losses = {"action": self._loss_for(
            "action", xs, data, t_action, ones, noise=noises.get("action"))}
        ns = {"action": self._effective_n("action", data, ones)}
        total = self._combine_losses(losses, loss_coeffs)
        return self._pack(total, losses, ns, xs, data, device,
                          z_action=zs.get("action"), t_action=t_action)

    # ---- loss packaging ---------------------------------------------------
    def _pack(self, total, losses, ns, xs, data, device, m_action=None,
              z_action=None, t_action=None):
        zero = torch.zeros((), device=device)
        x_action = xs.get("action")
        if x_action is not None and z_action is not None and t_action is not None:
            x_action = self._action_x_for_pack(x_action, z_action, t_action)
        actions = data.get("action")
        if x_action is not None and actions is not None:
            if m_action is None:
                m_action = ns.get("action", zero)
                B = actions.shape[0]
                m_action = (torch.ones(B, device=device)
                            if (torch.is_tensor(m_action) and float(m_action) > 0)
                            else torch.zeros(B, device=device))
            with torch.no_grad():
                act_mm, act_deg = self._action_phys_errors(x_action, actions, m_action)
        else:
            act_mm = act_deg = zero
        out = {"loss": total, "act_mm": act_mm, "act_deg": act_deg}
        for n in _ALL_MODS:
            lv = losses.get(n, zero)
            out[f"loss_{n}"] = lv.detach() if torch.is_tensor(lv) else lv
            nv = ns.get(n, zero)
            out[f"n_{n}"] = nv.detach() if torch.is_tensor(nv) else nv
        return out

    # ---- block-causal sequential autoregression over modalities -----------
    def _causal_block(self, name, values, role, step):
        """Embed one H/C/Q block and attach axial positions + mask metadata."""
        cfg = self.cfg
        if name in self.grid_names:
            B, nframes = values.shape[:2]
            tokens = self.embedders.embed_one(name, values).reshape(
                B, nframes * cfg.n_patches, cfg.dim)
            if role == "history":
                local = self.hist_local[name][:nframes]
            else:
                local = self.target_local[name][:nframes]
            globals_ = [self.mod_frames[name][j] for j in local]
            clock = self.obs_clock_full[
                torch.tensor(globals_, device=values.device, dtype=torch.long)]
            pt = clock.repeat_interleave(cfg.n_patches)
            ph = self.h_grid.repeat(nframes)
            pw = self.w_grid.repeat(nframes)
        else:
            tokens = self.embedders.embed_one(name, values)
            pt = self.action_clock
            ph = torch.zeros(cfg.action_horizon, dtype=torch.long,
                             device=values.device)
            pw = torch.zeros_like(ph)
        return {
            "name": name, "role": role, "step": step, "tokens": tokens,
            "pt": pt, "ph": ph, "pw": pw,
        }

    def _causal_history_blocks(self, source, training):
        """Clean history blocks; together they form one bidirectional superblock."""
        blocks = []
        history_names = set(self.cfg.history_modalities)
        for name in self.mod_names:
            if not self.has_history.get(name) or name not in history_names:
                continue
            if training and self.cfg.sensor_drop_prob > 0:
                if torch.rand(()).item() < self.cfg.sensor_drop_prob:
                    continue
            if name not in source:
                continue
            values = source[name][:, self.hist_local[name]]
            if values.shape[1] > 0:
                blocks.append(self._causal_block(
                    name, values, role="history", step=-1))
        return blocks

    def _causal_future(self, data, name):
        if name in self.grid_names:
            return data[name][:, self.target_local[name]]
        return data[name]

    @staticmethod
    def _causal_finalize_blocks(blocks):
        """Concatenate blocks and assign stable token ranges."""
        if not blocks:
            return None
        cursor = 0
        for block in blocks:
            length = block["tokens"].shape[1]
            block["start"], block["end"] = cursor, cursor + length
            cursor += length
        return (
            torch.cat([b["tokens"] for b in blocks], dim=1),
            torch.cat([b["pt"] for b in blocks]),
            torch.cat([b["ph"] for b in blocks]),
            torch.cat([b["pw"] for b in blocks]),
        )

    @staticmethod
    def _build_causal_mask(blocks, device=None):
        """Token mask for H/C/Q double-sequence teacher forcing."""
        if not blocks:
            raise ValueError("causal attention requires at least one block")
        length = blocks[-1]["end"]
        device = device or blocks[0]["tokens"].device
        mask = torch.zeros(length, length, dtype=torch.bool, device=device)
        histories = [b for b in blocks if b["role"] == "history"]
        cleans = [b for b in blocks if b["role"] == "clean"]

        def allow(row, key):
            mask[row["start"]:row["end"], key["start"]:key["end"]] = True

        for row in blocks:
            if row["role"] == "history":
                for key in histories:
                    allow(row, key)
            elif row["role"] == "clean":
                for key in histories:
                    allow(row, key)
                for key in cleans:
                    if key["step"] <= row["step"]:
                        allow(row, key)
            elif row["role"] == "query":
                for key in histories:
                    allow(row, key)
                for key in cleans:
                    if key["step"] < row["step"]:
                        allow(row, key)
                allow(row, row)
            else:
                raise ValueError(f"unknown causal block role {row['role']!r}")
        if not bool(mask.any(dim=1).all()):
            raise RuntimeError("causal attention mask contains an empty query row")
        return mask

    def _causal_token_conditioning(self, blocks, proprio, task_id, times):
        base = self.dit.make_cond_causal_base(proprio, task_id)
        token_conds, block_conds = [], {}
        ones = torch.ones(proprio.shape[0], device=proprio.device)
        for block in blocks:
            if block["role"] == "query":
                time = times.get(block["name"], ones)
            elif "cond_time" in block:
                time = block["cond_time"]
            else:
                time = ones
            cond = self.dit.make_cond_causal_block(block["name"], time, base)
            token_conds.append(cond[:, None].expand(
                -1, block["end"] - block["start"], -1))
            if block["role"] == "query":
                block_conds[block["name"]] = cond
        return torch.cat(token_conds, dim=1), block_conds

    def _causal_training_order(self, stream, device):
        configured = [name for name in self.cfg.generation_order
                      if name in self.mod_names]
        dynamics = [name for name in configured if name in self.gen_names]
        if not dynamics and stream != "action":
            raise ValueError("generation_order contains no trainable dynamics modality")
        if self.mode == "modar":
            missing = [name for name in self.gen_names if name not in dynamics]
            if missing:
                raise ValueError(
                    f"ModAR generation_order omits dynamics modalities {missing}")
            if stream == "dynamics":
                return dynamics, dynamics
            order = dynamics + (["action"] if self.has_action else [])
            if stream == "action":
                return order, ["action"]
            return order, order
        raise ValueError(f"unsupported causal schedule_mode {self.mode!r}")

    def _assemble_causal(self, data, proprio, order, query_names,
                         query_zs, times, task_id=None):
        """Assemble H, clean teacher blocks, and all supervised noisy queries."""
        blocks = self._causal_history_blocks(data, training=True)
        query_set = set(query_names)
        last_needed = max((order.index(name) for name in query_names), default=-1)
        beta = self.cfg.cond_noise_beta
        tok_p = self.cfg.token_noise_p
        drop_p = self.cfg.modality_dropout_p
        noise_cond = self.training and beta > 0.0
        tok_noise = self.training and tok_p > 0.0
        # Modality dropout only makes sense for the action query -- see
        # MFConfig.modality_dropout_p for why the dynamics cascade is exempt.
        drop_mod = (self.training and drop_p > 0.0
                    and list(query_names) == ["action"])
        for step, name in enumerate(order):
            if step >= last_needed:
                break
            if (drop_mod and name in self.gen_names
                    and torch.rand(()).item() < drop_p):
                continue
            fut = self._causal_future(data, name)
            cond_time = None
            if name in self.gen_names and (noise_cond or tok_noise):
                B = fut.shape[0]
                if noise_cond:
                    cond_time = 1.0 - beta * torch.rand(B, device=fut.device)
                    fut = self.scheduler.add_noise(
                        fut, cond_time, torch.randn_like(fut))
                if tok_noise:
                    # Applied AFTER the uniform corruption so a replaced token is
                    # pure noise (the t=0 marginal) rather than a mixture of two
                    # noises at t_c, which would be neither clean nor standard.
                    # fut is (B, frames, patches, D) for every generated
                    # modality, so dropping the feature axis leaves exactly one
                    # Bernoulli per attention token.
                    keep = torch.rand(fut.shape[:-1], device=fut.device) >= tok_p
                    fut = torch.where(keep.unsqueeze(-1), fut,
                                      torch.randn_like(fut))
            block = self._causal_block(name, fut, "clean", step)
            if cond_time is not None and self.cfg.cond_noise_label_t:
                block["cond_time"] = cond_time
            blocks.append(block)
        for step, name in enumerate(order):
            if name in query_set:
                blocks.append(self._causal_block(
                    name, query_zs[name], "query", step))
        packed = self._causal_finalize_blocks(blocks)
        tokens, pt, ph, pw = packed
        mask = self._build_causal_mask(blocks, tokens.device)
        c_tokens, query_conds = self._causal_token_conditioning(
            blocks, proprio, task_id, times)
        return tokens, pt, ph, pw, blocks, mask, c_tokens, query_conds

    def _run_causal(self, data, proprio, order, query_names,
                    query_zs, times, task_id=None):
        (tokens, pt, ph, pw, blocks, mask,
         c_tokens, query_conds) = self._assemble_causal(
            data, proprio, order, query_names, query_zs, times, task_id)
        query_blocks = [b for b in blocks if b["role"] == "query"]
        segments = [(b["name"], b["start"], b["end"]) for b in query_blocks]
        hidden = self.dit(
            tokens, c_tokens, pt, ph, pw, segments, mask,
            segment_conds=query_conds)
        predictions = {}
        B, N = proprio.shape[0], self.cfg.n_patches
        for block in query_blocks:
            name = block["name"]
            h = hidden[name]
            if name in self.grid_names:
                h = h.reshape(B, self.cfg.obs_future, N, -1)
            predictions[name] = self.heads.read_one(name, h)
        return predictions

    def _forward_causal(self, data, proprio, stream=None, task_id=None,
                        loss_coeffs=None):
        B, device = proprio.shape[0], proprio.device
        ones = torch.ones(B, device=device)
        order, query_names = self._causal_training_order(stream, device)
        # Self-forcing: build a self-generated CONTEXT copy so the query conditions
        # on the imperfect dynamics it sees at test time. Targets/supervision below
        # stay on the clean ``data``, so no modality is trained toward its own
        # generation. By default this is scoped to the ACTION stream: extending it
        # to the dynamics queries trains them to denoise toward a GT future while
        # conditioning on a self-gen rollout of a DIFFERENT future, which is
        # ill-posed because every modality describes the same future window.
        data_ctx = data
        if self.cfg.self_forcing and self.training and self.gen_names:
            # Scope by supervised query set: the action stream supervises only the
            # action query, so its dynamics context is pure conditioning and safe
            # to self-force. A pass that also supervises dynamics queries shares
            # one context, so action-only scoping cannot be honored there.
            if (not self.cfg.self_forcing_action_only
                    or list(query_names) == ["action"]):
                data_ctx = self._inject_selfgen(data, proprio, task_id)
        times, noises, query_zs = {}, {}, {}
        for name in query_names:
            target = self._causal_future(data, name)          # clean GT target
            times[name] = self.scheduler.sample_active_time(name, B, device)
            noises[name] = torch.randn_like(target)
            query_zs[name] = self.scheduler.add_noise(
                target, times[name], noises[name])
        predictions = self._run_causal(
            data_ctx, proprio, order, query_names, query_zs, times, task_id)
        losses, ns = {}, {}
        for name in query_names:
            losses[name] = self._loss_active(
                name, predictions[name], data, times[name], ones,
                noise=noises.get(name))
            ns[name] = self._effective_n(name, data, ones)
        total = self._combine_losses(losses, loss_coeffs, mean=True)
        action_active = "action" in query_names
        xs = {"action": predictions["action"]} if action_active else {}
        return self._pack(
            total, losses, ns, xs, data, device,
            m_action=(ones if action_active else torch.zeros_like(ones)),
            z_action=query_zs.get("action"), t_action=times.get("action"))

    def _inject_selfgen(self, data, proprio, task_id):
        """Roll out the dynamics futures with the model's
        OWN generation (no_grad, gt_set=(), inference behavior) and splice them
        into a shallow copy of ``data`` at each modality's target frames, so a
        trunk pass conditions on self-generated instead of GT dynamics. Returns a
        CONTEXT-only copy: the caller keeps clean ``data`` for the supervised
        targets. History frames are untouched. The caller decides which passes get
        this treatment (see cfg.self_forcing_action_only)."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                gen = self._causal_generate_raw(
                    data, proprio, data, task_id,
                    order=self.cfg.generation_order, gt_set=(),
                    stop_before_action=True)
        finally:
            if was_training:
                self.train()
        data = dict(data)
        for name, val in gen.items():
            full = data[name].clone()
            full[:, self.target_local[name]] = val
            data[name] = full
        return data

    def freeze_to_experts_only(self, modalities):
        """Freeze every parameter except the pathways of ``modalities`` (each
        expert stack, its trunk-in projection + final layer + time embedder, the
        token embedder and readout head). Returns the trainable parameter count.
        Used by the frozen-backbone finetunes (train.reinit_experts /
        train.reinit_action)."""
        n_train = 0
        for name, p in self.named_parameters():
            keep = is_expert_pathway_key(name, modalities)
            p.requires_grad = keep
            if keep:
                n_train += p.numel()
        return n_train

    def freeze_to_action_only(self):
        return self.freeze_to_experts_only(("action",))

    def expert_pathway_param_groups(self, modalities, lr, trunk_lr_scale):
        """Two AdamW param groups: the swept pathways at ``lr``, everything else
        at ``trunk_lr_scale * lr``. Low-LR trunk finetune (trunk_lr_scale > 0)
        keeps most of the warm start while letting the trunk adapt to a
        differently shaped expert, which a hard freeze cannot do -- the trunk is
        co-adapted to the original expert shapes, so freezing it biases the
        measured optimum toward the shape it was trained with."""
        swept, trunk = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            tgt = swept if is_expert_pathway_key(name, modalities) else trunk
            tgt.append(p)
        groups = [{"params": swept, "lr": lr}]
        if trunk:
            groups.append({"params": trunk, "lr": lr * trunk_lr_scale})
        return groups

    def _disjoint_present_local(self, name, state):
        """Tensor-local frames present in a reduced disjoint sequence."""
        if state == "absent":
            return []
        if self.has_history[name]:
            loc = list(range(self.cfg.obs_history))
            if state == "active":
                loc = loc + list(self.future_frames)
            return loc
        # future-only (tracks)
        return list(range(len(self.future_frames))) if state == "active" else []

    def _assemble_disjoint(self, states, zs):
        """Build a reduced sequence containing histories and one active modality.
        Returns tokens:(B,L,dim), pos_t/h/w:(L,), and segs:{name:(s,e)}."""
        cfg = self.cfg
        N, dim = cfg.n_patches, cfg.dim
        tok_blocks, pt, ph, pw, segs = [], [], [], [], {}
        cursor = 0
        for name in self.mod_names:
            st = states[name]
            if st == "absent":
                continue
            if name in self.grid_names:
                loc = self._disjoint_present_local(name, st)
                if not loc:
                    continue
                emb = self.embedders.embed_one(
                    name, zs[name][:, loc])  # (B,nl,N,dim)
                B = emb.shape[0]
                tok_blocks.append(emb.reshape(B, len(loc) * N, dim))
                globals_ = [self.mod_frames[name][j] for j in loc]
                clk = self.obs_clock_full[torch.tensor(globals_, device=self.pos_t.device)]
                pt.append(clk.repeat_interleave(N))
                ph.append(self.h_grid.repeat(len(loc)))
                pw.append(self.w_grid.repeat(len(loc)))
                blk = len(loc) * N
            else:  # action (present only if ctx/active)
                tok_blocks.append(self.embedders.embed_one("action", zs["action"]))
                A = cfg.action_horizon
                pt.append(self.action_clock)
                ph.append(torch.zeros(A, dtype=torch.long, device=self.pos_t.device))
                pw.append(torch.zeros(A, dtype=torch.long, device=self.pos_t.device))
                blk = A
            segs[name] = (cursor, cursor + blk)
            cursor += blk
        tokens = torch.cat(tok_blocks, dim=1)
        return tokens, torch.cat(pt), torch.cat(ph), torch.cat(pw), segs

    def _run_disjoint(self, active, states, times, zs, proprio, task_id=None):
        """Reduced-sequence trunk pass; only the ACTIVE modality's expert+head run.
        Returns the active modality's future x-prediction (grid ->
        (B,obs_future,N,d); action -> (B,A,act_dim))."""
        tokens, pt, ph, pw, segs = self._assemble_disjoint(states, zs)
        c = self.dit.make_cond_disjoint(
            active, times[active], proprio, task_id)
        s, e = segs[active]
        out = self.dit(tokens, c, pt, ph, pw, [(active, s, e)])
        h = out[active]
        B, N = tokens.shape[0], self.cfg.n_patches
        if active in self.grid_names:
            nframes = (e - s) // N
            nfut = self.cfg.obs_future
            h = h.reshape(B, nframes, N, -1)[:, nframes - nfut:]
            return self.heads.read_one(active, h)
        return self.heads.read_one("action", h)

    def _forward_disjoint_reduced(self, data, proprio, stream=None, task_id=None,
                                  loss_coeffs=None):
        cfg = self.cfg
        B = proprio.shape[0]
        device = proprio.device
        ones = torch.ones(B, device=device)
        combo = self.scheduler.sample_disjoint(
            B, device, allowed=self._stream_allowed(stream),
            sensor_drop=cfg.sensor_drop_prob)
        active, states, times = combo["active"], combo["states"], combo["times"]

        # Noise only the active future; all other futures are physically absent.
        zs, noises = {}, {}
        for name in self.present_names:
            if name in self.grid_names:
                z = data[name].clone()
                if states[name] == "active":
                    tl = self.target_local[name]
                    noise = torch.randn_like(data[name][:, tl])
                    noises[name] = noise
                    z[:, tl] = self.scheduler.add_noise(
                        data[name][:, tl], times[name], noise)
                zs[name] = z
            else:
                z = data[name].clone()
                if states[name] == "active":
                    noise = torch.randn_like(data[name])
                    noises[name] = noise
                    z = self.scheduler.add_noise(data[name], times[name], noise)
                zs[name] = z

        x = self._run_disjoint(active, states, times, zs, proprio, task_id)
        t_act = times[active]
        loss = self._loss_active(
            active, x, data, t_act, ones, noise=noises.get(active))
        losses = {active: loss}
        ns = {active: self._effective_n(active, data, ones)}
        total = self._combine_losses(losses, loss_coeffs)
        xs = {"action": x} if active == "action" else {}
        m_action = ones if active == "action" else torch.zeros(B, device=device)
        z_action = zs.get("action") if active == "action" else None
        t_action = t_act if active == "action" else None
        return self._pack(total, losses, ns, xs, data, device, m_action=m_action,
                          z_action=z_action, t_action=t_action)

    # ---- inference (receding-horizon rollout of one chunk) ----------------
    @torch.no_grad()
    def sample(self, dino_hist, depth_hist_maps, proprio, task_id=None,
               image_hist=None):
        """Generate the future chunk given clean history (dispatches on mode).

        History maps for grid+history modalities: dino_hist:(B,H,N,dino_dim);
        depth_hist_maps:(B,H,Hd,Wd); image_hist:(B,H,C,Hd,Wd) when the image
        modality is active. Tracks take no history input."""
        self._require_task_id(task_id)
        hist = self._build_history(dino_hist, depth_hist_maps, image_hist)
        if self.mode == "action_only":
            return self._sample_action_only(hist, proprio, task_id)
        if self.mode == "unified":
            return self._sample_unified(hist, proprio, task_id)
        if self.mode == "independent":
            return self._sample_independent(hist, proprio, task_id)
        if self.mode == "disjoint":
            return self._sample_disjoint(hist, proprio, task_id)
        if self.mode == "modar":
            return self._sample_causal(hist, proprio, task_id)
        raise ValueError(f"unsupported schedule_mode {self.mode!r}")

    def oracle_dynamics(self) -> list:
        """Dynamics modalities that can be teacher-forced ahead of the action,
        in inference order. Empty when the mode has no such regime: 'disjoint'
        and 'action_only' never put future dynamics tokens in the action's
        context, so conditioning on GT dynamics is not defined for them."""
        if not self.has_action or self.mode not in ("modar", "unified"):
            return []
        return [m for m in self.cfg.generation_order if m in self.gen_names]

    @torch.no_grad()
    def sample_action_oracle(self, hist, proprio, data, task_id=None):
        """Generate the action with the future dynamics held at GROUND TRUTH.

        Same history and same action ODE as ``sample``, but the dynamics the
        action conditions on are the true futures rather than the model's own.

        Causal modes teacher-force every dynamics block to its GT future;
        unified pins the dynamics latents to the GT flow trajectory at each ODE
        step (feeding clean GT would be off-distribution there -- see
        ``_sample_unified_oracle``)."""
        dyn = self.oracle_dynamics()
        if not dyn:
            raise ValueError(
                f"schedule_mode {self.mode!r} has no GT-dynamics action regime")
        self._require_task_id(task_id)
        if self.mode == "unified":
            return self._sample_unified_oracle(hist, proprio, data, task_id)
        return self.sample_causal_mixed(
            hist, proprio, data, task_id, order=[*dyn, "action"],
            gt_set=tuple(dyn))

    def _build_history(self, dino_hist, depth_hist_maps, image_hist):
        """Patchify + collect clean history tensors for grid+history modalities."""
        cfg = self.cfg
        hist = {}
        for name in self.grid_names:
            if not self.has_history[name]:
                continue
            if name == "dino":
                hist["dino"] = dino_hist
            elif name == "depth":
                hist["depth"] = patchify_depth(depth_hist_maps, cfg.depth_patch_size)
            elif name == "image":
                assert image_hist is not None, "image modality active; pass image_hist"
                hist["image"] = patchify_rgb(image_hist, cfg.image_patch_size)
        return hist

    def _init_full_latents(self, hist, B, device):
        """Full grid latents: history clean, future at noise; action at noise."""
        cfg = self.cfg
        F, N = cfg.n_obs_frames, cfg.n_patches
        z = {}
        for name in self.present_names:
            if name == "action":
                z["action"] = torch.randn(B, cfg.action_horizon, cfg.action_dim,
                                          device=device)
                continue
            d = self.specs[self.order[name]].data_dim
            if self.has_history[name]:
                zz = torch.randn(B, F, N, d, device=device)
                hl = self.hist_local[name]
                zz[:, hl] = hist[name][:, :len(hl)]
                z[name] = zz
            else:  # tracks: future-only
                z[name] = torch.randn(B, cfg.obs_future, N, d, device=device)
        return z

    def _grid_out(self, name, fut_latent):
        """Convert a grid modality's future latent to output space."""
        cfg = self.cfg
        if name == "depth":
            return unpatchify_depth(
                fut_latent, cfg.depth_patch_size, cfg.grid_h, cfg.grid_w)
        if name == "image":
            return unpatchify_rgb(
                fut_latent, cfg.image_patch_size, cfg.grid_h, cfg.grid_w,
                channels=cfg.image_channels)
        return fut_latent  # dino / tracks

    _OUT_KEY = {"dino": "dino", "depth": "depth_maps", "image": "images",
                "tracks": "point_tracks", "action": "actions"}

    def _output_key(self, name):
        return self._OUT_KEY[name]

    def _sample_autoregressive(self, hist, proprio, task_id=None):
        """Cascade full-layout modalities in the configured deployment order."""
        cfg = self.cfg
        B = proprio.shape[0]
        device = proprio.device
        z = self._init_full_latents(hist, B, device)
        grid = [
            name for name in cfg.generation_order if name in self.grid_names]
        missing = [name for name in self.grid_names if name not in grid]
        if missing:
            raise ValueError(
                f"generation_order omits generated modalities {missing}")
        generated = set()
        for m in grid:
            tl = self.target_local[m]

            def dn(z_fut, t, _m=m, _tl=tl):
                z[_m][:, _tl] = z_fut
                over = {}
                for name in grid:
                    over[name] = (torch.ones(B, device=device) if name in generated
                                  else (torch.full((B,), float(t), device=device)
                                        if name == _m else torch.zeros(B, device=device)))
                over["action"] = torch.zeros(B, device=device)
                times = self._times(B, device, over, default=1.0)
                net_out = self._run(z, times, proprio, task_id)[_m][:, _tl]
                return self._denoise_xpred(_m, z_fut, t, net_out)
            n_steps = int(cfg.steps_by_modality.get(m, cfg.steps_per_phase))
            solver = cfg.solver_by_modality.get(m, cfg.solver)
            z[m][:, tl] = self.scheduler.ode_solve(
                dn, z[m][:, tl], n_steps, solver)
            generated.add(m)

        if self.has_action:
            def dn_action(z_act, t):
                z["action"] = z_act
                over = {name: torch.ones(B, device=device) for name in grid}
                over["action"] = torch.full((B,), float(t), device=device)
                times = self._times(B, device, over, default=1.0)
                net_out = self._run(z, times, proprio, task_id)["action"]
                return self._denoise_xpred("action", z_act, t, net_out)
            n_steps = int(
                cfg.steps_by_modality.get("action", cfg.steps_per_phase))
            solver = cfg.solver_by_modality.get("action", cfg.solver)
            z["action"] = self.scheduler.ode_solve(
                dn_action, z["action"], n_steps, solver)

        return self._collect_outputs(z)

    def _collect_outputs(self, z):
        out = {}
        for name in self.present_names:
            if name == "action":
                if "action" in z:
                    out["actions"] = z["action"]
                continue
            tl = self.target_local[name]
            fut = z[name][:, tl] if self.has_history[name] else z[name]
            out[self._output_key(name)] = self._grid_out(name, fut)
        return out

    def _sample_unified(self, hist, proprio, task_id=None):
        return self._collect_outputs(
            self._sample_unified_z(hist, proprio, task_id))

    def _sample_unified_raw(self, hist, proprio, task_id=None):
        """Native unified co-denoising, returned as INTERNAL future latents.

        Same generation as ``_sample_unified``; only the output space differs.
        This is the space ``_causal_future`` returns and ``sample_unified_pinned``
        consumes, so the model's own futures can be fed back to it on the same
        footing as a donor's."""
        z = self._sample_unified_z(hist, proprio, task_id)
        return {n: (z[n][:, self.target_local[n]] if n in self.grid_names
                    else z["action"])
                for n in self.present_names}

    def _sample_unified_z(self, hist, proprio, task_id=None):
        cfg = self.cfg
        B = proprio.shape[0]
        device = proprio.device
        eps = 1e-3
        z = self._init_full_latents(hist, B, device)
        grid = self.grid_names

        def cur(name):
            return z[name][:, self.target_local[name]] if name in grid else z["action"]

        def setcur(name, val):
            if name in grid:
                z[name][:, self.target_local[name]] = val
            else:
                z["action"] = val

        def predict_net(t):
            times = self._times(B, device,
                                {m: torch.full((B,), float(t), device=device)
                                 for m in self.mod_names})
            xs = self._run(z, times, proprio, task_id)
            res = {}
            for name in self.present_names:
                res[name] = (xs[name][:, self.target_local[name]] if name in grid
                             else xs["action"])
            return res

        ts = torch.linspace(0.0, 1.0, cfg.steps_per_phase + 1, device=device)
        for i in range(cfg.steps_per_phase):
            t0, t1 = float(ts[i]), float(ts[i + 1])
            dt = t1 - t0
            net0 = predict_net(t0)
            v0 = {n: self._net_to_velocity(n, cur(n), t0, net0[n], eps)
                  for n in self.present_names}
            z_prev = {n: cur(n) for n in self.present_names}
            for n in self.present_names:
                setcur(n, z_prev[n] + dt * v0[n])
            if not (cfg.solver == "euler" or (1.0 - t1) <= eps):
                net1 = predict_net(t1)
                for n in self.present_names:
                    v1 = self._net_to_velocity(n, cur(n), t1, net1[n], eps)
                    setcur(n, z_prev[n] + dt * 0.5 * (v0[n] + v1))
        return z

    @torch.no_grad()
    def _sample_unified_oracle(self, hist, proprio, data, task_id=None):
        """Unified co-denoising with the dynamics futures PINNED to the
        ground-truth flow trajectory (oracle conditioning).

        At each ODE time ``t`` every generated dynamics latent is overwritten
        with ``add_noise(gt_future, t, eps)`` for a per-modality noise ``eps``
        drawn ONCE and held fixed across the trajectory. The dynamics tokens
        therefore trace the exact GT straight-line flow (pure noise at t=0 ->
        clean GT at t=1), matching the marginal the model saw in training while
        revealing the true future -- rather than the model's own co-generated
        dynamics. Only the action block is ODE-integrated. This is the unified
        analogue of autoregressive clean teacher-forcing; feeding clean GT at every step instead
        would be off-distribution, since unified expects all tokens at a shared
        noise level."""
        dyn = [m for m in self.present_names if m != "action"]
        return self.sample_unified_pinned(
            hist, proprio, {m: self._causal_future(data, m) for m in dyn},
            task_id)

    @torch.no_grad()
    def sample_unified_pinned(self, hist, proprio, pinned, task_id=None,
                              dyn_noise=None):
        """Unified co-denoising with the dynamics futures PINNED to ``pinned``.

        ``pinned`` maps each non-action modality to a CLEAN future latent in the
        internal space ``_causal_future`` returns. The futures it names are not
        co-generated: at each ODE time they are overwritten with
        ``add_noise(pinned[m], t, eps)``, so they trace the straight-line flow
        that would have landed on them, and only the action block is integrated.

        The pinned futures do not have to be ground truth. Passing another
        checkpoint's generated dynamics turns this into a cross-model stitch --
        "is model A's world model a more useful conditioning signal for model
        B's action head than B's own?" -- which is exactly the oracle
        measurement with the oracle replaced by a second world model.

        ``dyn_noise`` optionally supplies the epsilon each pinned future is
        carried along the flow with. It is drawn from the ambient RNG when
        omitted, which means a caller that reseeds to resample the ACTION also
        silently resamples the futures' trajectory -- fine for rollouts, but it
        confounds any measurement that tries to isolate the action's own
        sampling spread. Pass a fixed dict to hold the futures still."""
        cfg = self.cfg
        B = proprio.shape[0]
        device = proprio.device
        eps = 1e-3
        z = self._init_full_latents(hist, B, device)
        grid = self.grid_names
        dyn = [m for m in self.present_names if m != "action"]
        missing = [m for m in dyn if m not in pinned]
        if missing:
            raise ValueError(
                f"sample_unified_pinned needs a future for every dynamics "
                f"modality; missing {missing}")
        gt = {m: pinned[m] for m in dyn}
        noise = (dyn_noise if dyn_noise is not None
                 else {m: torch.randn_like(gt[m]) for m in dyn})

        def setcur(name, val):
            if name in grid:
                z[name][:, self.target_local[name]] = val
            else:
                z["action"] = val

        def pin_dyn(t):
            tt = torch.full((B,), float(t), device=device)
            for m in dyn:
                setcur(m, self.scheduler.add_noise(gt[m], tt, noise[m]))

        def action_net(t):
            times = self._times(B, device,
                                {m: torch.full((B,), float(t), device=device)
                                 for m in self.mod_names})
            return self._run(z, times, proprio, task_id)["action"]

        pin_dyn(0.0)
        if not self.has_action:
            pin_dyn(1.0)
            return self._collect_outputs(z)
        ts = torch.linspace(0.0, 1.0, cfg.steps_per_phase + 1, device=device)
        for i in range(cfg.steps_per_phase):
            t0, t1 = float(ts[i]), float(ts[i + 1])
            dt = t1 - t0
            pin_dyn(t0)
            v0 = self._net_to_velocity("action", z["action"], t0,
                                       action_net(t0), eps)
            z_prev = z["action"]
            z["action"] = z_prev + dt * v0
            if not (cfg.solver == "euler" or (1.0 - t1) <= eps):
                pin_dyn(t1)
                v1 = self._net_to_velocity("action", z["action"], t1,
                                           action_net(t1), eps)
                z["action"] = z_prev + dt * 0.5 * (v0 + v1)
        pin_dyn(1.0)
        return self._collect_outputs(z)

    def _sample_action_only(self, hist, proprio, task_id=None):
        cfg = self.cfg
        B = proprio.shape[0]
        device = proprio.device
        z = {name: hist[name] for name in hist}          # history-only grid context
        z["action"] = torch.randn(B, cfg.action_horizon, cfg.action_dim, device=device)

        def dn_action(z_act, t):
            z["action"] = z_act
            over = {"action": torch.full((B,), float(t), device=device)}
            times = self._times(B, device, over, default=1.0)
            net_out = self._run(z, times, proprio, task_id)["action"]
            return self._denoise_xpred("action", z_act, t, net_out)
        z["action"] = self.scheduler.ode_solve(
            dn_action, z["action"], cfg.steps_per_phase, cfg.solver)
        return {"actions": z["action"]}

    def _sample_disjoint(self, hist, proprio, task_id=None):
        """Deployment rollout: denoise ONLY the action chunk from clean history
        context (no future cross-conditioning), matching the disjoint training
        pass. Uses the reduced causal path with active=action."""
        generated = self._disjoint_rollout(hist, proprio, task_id, ["action"])
        return {"actions": generated["action"]}

    def _disjoint_rollout(self, hist, proprio, task_id, order):
        """Generate each requested modality independently from clean history."""
        cfg = self.cfg
        B = proprio.shape[0]
        device = proprio.device
        N = cfg.n_patches
        history_mods = set(cfg.history_modalities)
        generated = {}

        def build_states(active):
            st = {}
            for name in self.mod_names:
                if name == active:
                    st[name] = "active"
                elif self.has_history.get(name) and name in history_mods:
                    st[name] = "hist_only"
                else:
                    st[name] = "absent"
            return st

        def build_zs(active, active_future):
            zs = {}
            for name in self.mod_names:
                if name == "action":
                    z = torch.zeros(B, cfg.action_horizon, cfg.action_dim, device=device)
                    if active == "action":
                        z = active_future
                    zs["action"] = z
                    continue
                d = self.specs[self.order[name]].data_dim
                if self.has_history[name]:
                    z = torch.zeros(B, cfg.n_obs_frames, N, d, device=device)
                    if name in history_mods:
                        hl = self.hist_local[name]
                        z[:, hl] = hist[name][:, :len(hl)]
                    tl = self.target_local[name]
                    if active == name:
                        z[:, tl] = active_future
                else:  # tracks (future-only)
                    z = torch.zeros(B, cfg.obs_future, N, d, device=device)
                    if active == name:
                        z = active_future
                zs[name] = z
            return zs

        for m in order:
            if m not in self.mod_names:
                continue
            states = build_states(m)

            def dn(z_fut, t, _m=m, _st=states):
                zs = build_zs(_m, z_fut)
                times = self._times(B, device,
                                    {_m: torch.full((B,), float(t), device=device)},
                                    default=1.0)
                net_out = self._run_disjoint(_m, _st, times, zs, proprio, task_id)
                return self._denoise_xpred(_m, z_fut, t, net_out)

            if m in self.grid_names:
                d = self.specs[self.order[m]].data_dim
                z0 = torch.randn(B, cfg.obs_future, N, d, device=device)
            else:
                z0 = torch.randn(B, cfg.action_horizon, cfg.action_dim, device=device)
            generated[m] = self.scheduler.ode_solve(
                dn, z0, cfg.steps_per_phase, cfg.solver)
        return generated

    def _causal_query_prediction(self, name, z_query, time, proprio, task_id,
                                 cache):
        """Cached active-query pass; the immutable prefix cache is not modified."""
        block = self._causal_block(name, z_query, "query", step=0)
        tokens, pt, ph, pw = self._causal_finalize_blocks([block])
        c_tokens, query_conds = self._causal_token_conditioning(
            [block], proprio, task_id, {name: time})
        hidden = self.dit.forward_query(
            tokens, c_tokens, pt, ph, pw, name, query_conds[name], cache)
        if name in self.grid_names:
            hidden = hidden.reshape(
                proprio.shape[0], self.cfg.obs_future, self.cfg.n_patches, -1)
        return self.heads.read_one(name, hidden)

    def _causal_uncached_prediction(self, hist, generated, name, z_query, time,
                                    proprio, task_id):
        """Full-prefix correctness reference for one causal query."""
        blocks = self._causal_history_blocks(hist, training=False)
        for step, (ctx_name, value) in enumerate(generated.items()):
            blocks.append(self._causal_block(ctx_name, value, "clean", step))
        query_step = len(generated)
        blocks.append(self._causal_block(name, z_query, "query", query_step))
        tokens, pt, ph, pw = self._causal_finalize_blocks(blocks)
        mask = self._build_causal_mask(blocks, tokens.device)
        c_tokens, query_conds = self._causal_token_conditioning(
            blocks, proprio, task_id, {name: time})
        query = blocks[-1]
        hidden = self.dit(
            tokens, c_tokens, pt, ph, pw,
            [(name, query["start"], query["end"])], mask,
            segment_conds=query_conds)[name]
        if name in self.grid_names:
            hidden = hidden.reshape(
                proprio.shape[0], self.cfg.obs_future, self.cfg.n_patches, -1)
        return self.heads.read_one(name, hidden)

    def _causal_prefill_history(self, hist, proprio, task_id):
        blocks = self._causal_history_blocks(hist, training=False)
        if not blocks:
            return None
        tokens, pt, ph, pw = self._causal_finalize_blocks(blocks)
        c_tokens, _ = self._causal_token_conditioning(
            blocks, proprio, task_id, {})
        _, cache = self.dit.forward_prefix(
            tokens, c_tokens, pt, ph, pw, cache=None)
        return cache

    def _causal_append_clean(self, name, value, step, proprio, task_id, cache):
        block = self._causal_block(name, value, "clean", step)
        tokens, pt, ph, pw = self._causal_finalize_blocks([block])
        c_tokens, _ = self._causal_token_conditioning(
            [block], proprio, task_id, {})
        _, cache = self.dit.forward_prefix(
            tokens, c_tokens, pt, ph, pw, cache=cache)
        return cache

    def _sample_causal(self, hist, proprio, task_id=None):
        """Generate one modality block at a time with an exact clean-prefix cache."""
        cfg = self.cfg
        B, device = proprio.shape[0], proprio.device
        order = [m for m in cfg.generation_order
                 if m in self.gen_names or (m == "action" and self.has_action)]
        generated = {}
        use_cache = cfg.use_kv_cache
        cache = (self._causal_prefill_history(hist, proprio, task_id)
                 if use_cache else None)
        for step, name in enumerate(order):
            if name in self.grid_names:
                data_dim = self.specs[self.order[name]].data_dim
                z0 = torch.randn(
                    B, cfg.obs_future, cfg.n_patches, data_dim, device=device)
            else:
                z0 = torch.randn(
                    B, cfg.action_horizon, cfg.action_dim, device=device)

            def denoise(z_query, t, _name=name):
                time = torch.full((B,), float(t), device=device)
                if use_cache:
                    net_out = self._causal_query_prediction(
                        _name, z_query, time, proprio, task_id, cache)
                else:
                    net_out = self._causal_uncached_prediction(
                        hist, generated, _name, z_query, time, proprio, task_id)
                return self._denoise_xpred(_name, z_query, t, net_out)

            n_steps = int(cfg.steps_by_modality.get(name, cfg.steps_per_phase))
            solver = cfg.solver_by_modality.get(name, cfg.solver)
            generated[name] = self.scheduler.ode_solve(
                denoise, z0, n_steps, solver)
            if use_cache and step + 1 < len(order):
                cache = self._causal_append_clean(
                    name, generated[name], step, proprio, task_id, cache)
        return self._format_generated(generated)

    @torch.no_grad()
    def sample_causal_mixed(self, hist, proprio, data, task_id=None,
                            order=None, gt_set=()):
        """Causal generation where prefix modalities in ``gt_set`` are
        teacher-forced to their GROUND-TRUTH future (from ``data``) instead of
        self-generated. Action is always ODE-sampled last.

        ``gt_set = ()``           -> identical to ``_sample_causal`` (self-gen).
        ``gt_set`` = all dynamics -> action on perfect futures (full oracle).
        ``gt_set`` a strict subset -> a partial-oracle ladder rung.

        ``data`` is a ``_prep_data`` dict; ``order`` defaults to
        ``cfg.generation_order``. ModAR mode only."""
        return self._format_generated(self._causal_generate_raw(
            hist, proprio, data, task_id, order, gt_set))

    def _causal_generate_raw(self, hist, proprio, data, task_id=None,
                             order=None, gt_set=(), stop_before_action=False):
        """Core causal rollout shared by ``sample_causal_mixed`` and the
        self-forcing training hook. Returns the RAW ``generated`` dict (internal
        latent space, keyed by modality name -- the same space
        ``_causal_future`` returns), NOT the public ``_format_generated`` output.
        ``stop_before_action`` drops the action step so only dynamics are rolled
        out. A name in ``gt_set`` is taken from ``data``.
        NOT wrapped in ``no_grad`` -- callers decide."""
        cfg = self.cfg
        B, device = proprio.shape[0], proprio.device
        if order is None:
            order = cfg.generation_order
        order = [m for m in order
                 if m in self.gen_names or (m == "action" and self.has_action)]
        if stop_before_action:
            order = [m for m in order if m != "action"]
        gt_set = set(gt_set)
        generated = {}
        use_cache = cfg.use_kv_cache
        cache = (self._causal_prefill_history(hist, proprio, task_id)
                 if use_cache else None)
        for step, name in enumerate(order):
            # Drawn even when the modality is teacher-forced and the draw goes
            # unused: it keeps the RNG stream identical to the self-generated
            # rollout, so under a pinned seed the ACTION starts both rollouts
            # from the same noise. Otherwise oracle-minus-self-gen carries the
            # difference of two independent noise draws on top of the
            # conditioning effect it is supposed to isolate.
            if name in self.grid_names:
                data_dim = self.specs[self.order[name]].data_dim
                z0 = torch.randn(
                    B, cfg.obs_future, cfg.n_patches, data_dim, device=device)
            else:
                z0 = torch.randn(
                    B, cfg.action_horizon, cfg.action_dim, device=device)
            if name != "action" and name in gt_set:
                generated[name] = self._causal_future(data, name)
            else:
                def denoise(z_query, t, _name=name):
                    time = torch.full((B,), float(t), device=device)
                    if use_cache:
                        net_out = self._causal_query_prediction(
                            _name, z_query, time, proprio, task_id, cache)
                    else:
                        net_out = self._causal_uncached_prediction(
                            hist, generated, _name, z_query, time, proprio, task_id)
                    return self._denoise_xpred(_name, z_query, t, net_out)

                n_steps = int(
                    cfg.steps_by_modality.get(name, cfg.steps_per_phase))
                solver = cfg.solver_by_modality.get(name, cfg.solver)
                generated[name] = self.scheduler.ode_solve(
                    denoise, z0, n_steps, solver)
            if use_cache and step + 1 < len(order):
                cache = self._causal_append_clean(
                    name, generated[name], step, proprio, task_id, cache)
        return generated

    def _format_generated(self, generated):
        """Map raw generated futures to the public sample() output keys."""
        out = {}
        for name, val in generated.items():
            if name == "action":
                out["actions"] = val
            else:
                out[self._output_key(name)] = self._grid_out(name, val)
        return out

    def _sample_independent(self, hist, proprio, task_id=None):
        sched = getattr(self.cfg, "infer_schedule", "autoregressive")
        if sched == "autoregressive":
            return self._sample_autoregressive(hist, proprio, task_id)
        if sched == "unified":
            return self._sample_unified(hist, proprio, task_id)
        if sched == "futures_noise":
            return self._sample_indep_action(hist, proprio, future_t=0.0,
                                             task_id=task_id)
        if sched.startswith("futures_at:"):
            return self._sample_indep_action(
                hist, proprio, future_t=float(sched.split(":", 1)[1]), task_id=task_id)
        raise ValueError(f"unknown infer_schedule {sched!r}")

    def _sample_indep_action(self, hist, proprio, future_t, task_id=None):
        """Denoise the action with grid futures held at noise level ``future_t``."""
        cfg = self.cfg
        B = proprio.shape[0]
        device = proprio.device
        z = self._init_full_latents(hist, B, device)
        grid = self.grid_names
        if future_t > 0.0:
            gen = self._sample_autoregressive(hist, proprio, task_id)
            for name in grid:
                fut = self._out_to_latent(name, gen)
                tl = self.target_local[name]
                noised = self.scheduler.add_noise(
                    fut, torch.full((B,), float(future_t), device=device),
                    torch.randn_like(fut))
                if self.has_history[name]:
                    z[name][:, tl] = noised
                else:
                    z[name] = noised

        def dn_action(z_act, t):
            z["action"] = z_act
            over = {name: torch.full((B,), float(future_t), device=device)
                    for name in grid}
            over["action"] = torch.full((B,), float(t), device=device)
            times = self._times(B, device, over, default=1.0)
            net_out = self._run(z, times, proprio, task_id)["action"]
            return self._denoise_xpred("action", z_act, t, net_out)
        z["action"] = self.scheduler.ode_solve(
            dn_action, z["action"], cfg.steps_per_phase, cfg.solver)
        out = self._collect_outputs(z)
        out["actions"] = z["action"]
        return out

    def _out_to_latent(self, name, out):
        """Inverse of _grid_out: output-space future -> patch latent."""
        cfg = self.cfg
        if name == "depth":
            return patchify_depth(out["depth_maps"], cfg.depth_patch_size)
        if name == "image":
            return patchify_rgb(out["images"], cfg.image_patch_size)
        return out[self._output_key(name)]
