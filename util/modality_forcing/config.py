"""Configuration for the Modality-Forcing WAM.

A single flat dataclass describing the architecture, token layout, and ModAR
schedule. Kept import-light (no torch) so it can be constructed from
Hydra/CLI without pulling in heavy deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModalitySpec:
    """Lightweight description of one modality in the token sequence.

    kind:
      "grid"   -- spatial patch/point grid (dino, depth, image, tracks).
                  Occupies N tokens per present frame; uses axial (t,h,w) RoPE.
      "action" -- dense per-step action tokens (action_horizon of them, no grid).
    has_history: grid obs modalities that carry a clean history frame (dino,
      depth, image). Tracks are future-only (the seed is always the fixed grid,
      so no history/current state is fed -- only the future motion is denoised).
    """
    name: str
    kind: str
    data_dim: int
    has_history: bool
    order: int


@dataclass
class MFConfig:
    # ---- Active modalities (ordered; action must be last) ----
    # Default is the legacy 3-modality set so older configs keep working; the
    # 5-modality configs set this explicitly to include image + tracks.
    modalities: Tuple[str, ...] = ("dino", "depth", "action")

    # ---- Modality dims (data space) ----
    dino_dim: int = 384          # DINOv2 ViT-S patch feature dim
    depth_patch_size: int = 14   # depth patch is patch_size**2 values (single channel)
    depth_img_size: int = 224
    # Optional rectangular geometry. When omitted, legacy square configs continue
    # to use depth_img_size x depth_img_size and grid x grid.
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    action_dim: int = 12
    proprio_dim: int = 12
    # RGB image: patchified like depth but 3-channel (patch is 3*patch_size**2).
    image_patch_size: int = 14
    image_channels: int = 3
    # Point tracks: a spatial grid of query points seeded at the decision frame;
    # each token carries (x, y, visibility). Future-only (no history token).
    track_dim: int = 3           # (x, y, visibility)
    track_grid: int = 16         # sqrt(n_track_points); must equal grid to reuse RoPE
    track_grid_height: Optional[int] = None
    track_grid_width: Optional[int] = None
    # Prediction parameterization for tracks:
    #   delta    -- predict per-point displacement from the initial grid seed
    #   absolute -- predict the absolute (normalized) point position
    track_pred_mode: str = "delta"

    # ---- Spatial grid ----
    # Square fallback geometry; public RoboTwin configs set explicit 12x16 grids.
    grid: int = 16
    grid_height: Optional[int] = None
    grid_width: Optional[int] = None

    # ---- Token layout (clock units = native frame index) ----
    obs_history: int = 2         # clean history obs frames
    obs_future: int = 2          # generated future obs frames (DINO + depth)
    # Native-frame gap between obs frames. For keyframe packs this must be an
    # integer multiple of the pack's keyframe_stride (the loader subsamples).
    obs_stride: int = 8
    action_horizon: int = 16     # dense future actions (stride 1) starting at tau

    # ---- Training window range ----
    # Which decision frames a demo may be sampled at.
    #   full   -- every frame with a valid history, i.e. k in [(obs_history-1)*ratio,
    #             Ts-1]. Futures that run past the end of the demo are replicate-
    #             padded and excluded from the loss via per-frame validity masks.
    #             The range depends only on the demo length, so different future
    #             horizons train on the same windows.
    #   legacy -- also requires the full observation future, track horizon, and
    #             action chunk to fit. That drops the final actions of every demo.
    window_mode: str = "full"
    # How to fill action steps past the end of the demo (window_mode == "full"):
    #   hold -- repeat the final action and supervise it, teaching "reach the
    #           terminal pose and stay there". Demos end at task success, so this
    #           is the behaviour we actually want at the end of an episode.
    #   mask -- pad the tensor the same way but drop those steps from the loss.
    action_pad_mode: str = "hold"
    # Padded observation futures stay OUT of the loss by default. The asymmetry
    # with actions is deliberate: "hold the final pose" is a command we choose,
    # while "the scene is frozen" is a claim about reality we cannot verify.
    obs_pad_supervise: bool = False

    # Learned task embedding in global adaLN cond (summed with proprio/times).
    # Set from len(data.tasks) at train init; 0 disables (unit smokes only).
    n_tasks: int = 0

    # ---- Transformer ----
    dim: int = 384               # >= depth patch (196) and = DINO dim (lossless DINO)
    # Layers: a SHARED trunk (full cross-modal attention) followed by per-modality
    # EXPERT layers (DINO/depth/action; within-modality attention only) for output
    # specialization, a la Latent Forcing's output experts.
    n_shared_layers: int = 5
    # Expert depth is per modality: `n_expert_layers` is the fallback and
    # `n_expert_layers_<name>` overrides it for a single modality (None = inherit).
    n_expert_layers: int = 2
    n_expert_layers_dino: Optional[int] = None
    n_expert_layers_depth: Optional[int] = None
    n_expert_layers_image: Optional[int] = None
    n_expert_layers_tracks: Optional[int] = None
    n_expert_layers_action: Optional[int] = None
    # Expert WIDTH is also per modality (None = the shared trunk `dim`). A narrow
    # expert takes a linear projection down from the trunk width at the top of its
    # stack and reads out at its own width, so cheap low-dimensional readouts
    # (action, tracks) need not pay for full-width blocks. Must be a multiple of
    # head_dim, which keeps head_dim (and therefore rope_split) constant across
    # widths; the expert's head count is expert_dim // head_dim.
    expert_dim_dino: Optional[int] = None
    expert_dim_depth: Optional[int] = None
    expert_dim_image: Optional[int] = None
    expert_dim_tracks: Optional[int] = None
    expert_dim_action: Optional[int] = None
    # 384 / 6 gives a 64-dimensional attention head, matching the public configs
    # and a well-supported shape for fused attention kernels.
    n_heads: int = 6             # head_dim = 64 (even, for RoPE)
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # QK-RMSNorm on attention queries/keys (canonical JiT stabilizer for
    # pixel-space training; off by default to preserve older runs' arch).
    qk_norm: bool = False
    # head_dim axial RoPE split: (temporal, height, width); must sum to head_dim
    # and each be even. Sums to 64 to match the n_heads default above -- the two
    # are coupled by an assertion in __post_init__, so they have to move together.
    rope_split: Tuple[int, int, int] = (24, 20, 20)

    # ---- Schedule mode ----
    # modar       : fixed-order block-causal autoregression. One double-sequence
    #               forward supervises every dynamics cut; the action stream sees
    #               the full fixed clean prefix. Inference uses a prefix KV cache.
    # unified     : ONE shared diffusion time for all modalities; supervise the
    #               source's stream targets (all modalities for a joint source)
    # action_only : clean obs history as context, denoise only future actions
    # independent : INDEPENDENT per-modality diffusion times. Covers actions
    #               denoised while the other futures are still noise, so the
    #               action head learns not to over-trust generated futures.
    # disjoint    : one random active modality per reduced-sequence pass, conditioned
    #               only on clean sensor history. No generated futures are shared.
    schedule_mode: str = "modar"
    # Inference schedule for schedule_mode == "independent" (ignored otherwise):
    #   autoregressive: generate modalities in generation_order, feeding each
    #                   clean result to subsequent phases
    #   unified       : co-denoise every future and action at one shared ODE time
    #   futures_noise : hold future DINO/depth at pure noise; denoise only the action
    #                   (robust; world model unused at inference)
    #   futures_at:<t>: generate futures, re-noise them to level <t>, then denoise the
    #                   action treating the futures as uncertain (skeptical middle ground)
    infer_schedule: str = "autoregressive"

    # ---- Active-modality schedule (schedule_mode == "disjoint" only) ----
    # Per-step sampling probabilities for which modality is ACTIVE (supervised).
    # Renormalized over enabled `modalities`. Ignored by ModAR, which supervises
    # every requested cut of the fixed order. Image / tracks
    # default to 0 so legacy 3-modality configs are unaffected.
    p_dino: float = 0.4
    p_depth: float = 0.3
    p_action: float = 0.3
    p_image: float = 0.0
    p_tracks: float = 0.0
    # Per-modality logit-normal (mu, sigma) for the active modality's time.
    logitnorm_dino: Tuple[float, float] = (-1.0, 1.0)
    logitnorm_tracks: Tuple[float, float] = (-1.0, 1.0)
    logitnorm_depth: Tuple[float, float] = (-1.0, 1.0)
    logitnorm_image: Tuple[float, float] = (-1.0, 1.0)
    logitnorm_action: Tuple[float, float] = (-1.0, 1.0)
    # Shared time logit-normal for schedule_mode == "unified".
    logitnorm_unified: Tuple[float, float] = (-1.0, 1.0)
    # Latent Forcing sec 5.2: a logit-normal puts ~zero mass at t=0 (pure noise),
    # but that is exactly the generative regime that decides global structure and
    # object placement. So for EVERY actively-generated modality (and every
    # schedule mode), with
    # prob early_t_frac we draw the active time from U[0, early_t_max] instead of
    # the logit-normal. 'independent' samples U[0,1] so it is covered inherently.
    early_t_frac: float = 0.1
    early_t_max: float = 0.5
    # Conditioning-noise augmentation (Latent Forcing sec 5.2, restored into the
    # active block-causal engine). During TRAINING only, each clean teacher block
    # for a generated dynamics modality is corrupted to a per-sample, per-modality
    # time t_c ~ U[1-cond_noise_beta, 1] via add_noise. beta=0.0 reproduces the
    # clean-conditioning behavior exactly. Inference still conditions generated
    # futures at t=1.0. Default 0.5 is the RoboTwin ModAR recipe.
    cond_noise_beta: float = 0.5
    # Whether the corrupted block ALSO announces its own t_c (Option A) or keeps
    # the t=1.0 label the clean path uses (Option B). Option A is close to a
    # no-op for exposure bias: announcing t_c lets the model learn a t-INDEXED
    # trust policy, and since inference always labels generated futures t=1.0 it
    # simply selects the fully-trusting branch -- the one branch the augmentation
    # never trained to be skeptical. Option B (the default) makes the SAME label
    # cover both clean and corrupted context, so trust has to be amortized into
    # the branch deployment actually uses.
    cond_noise_label_t: bool = False
    # Token-level conditioning dropout. After the uniform corruption above, each
    # TOKEN of a clean dynamics teacher block (one patch of one future frame, or
    # one track cell) is independently replaced by PURE noise with this
    # probability. Uniform corruption lowers every token's SNR together, which a
    # query can undo by averaging; replacing whole tokens instead removes
    # specific evidence, so the query cannot lean on any one patch or track
    # being present and informative. 0.0 disables.
    token_noise_p: float = 0.0
    # Modality dropout: probability that a generated dynamics modality's clean
    # teacher block is omitted entirely from the ACTION query's context. History
    # blocks are untouched, the modality's position in generation_order is
    # unchanged, and every modality may be dropped at once.
    # Scoped to the action stream on purpose: at inference the cascade always
    # produces every preceding future, so dropping one from a DYNAMICS query's
    # context is a mismatch with no deployment counterpart, whereas the action
    # head is precisely the consumer that should not depend on any single
    # modality carrying the plan. Drawn once per batch (like
    # sensor_drop_prob), so ranks may disagree -- DDP runs with
    # find_unused_parameters=True. 0.0 disables.
    modality_dropout_p: float = 0.0
    # During training, replace clean future context with the model's own generated
    # futures. Supervision stays on the ground-truth targets. Off by default.
    self_forcing: bool = False
    # Apply self-forcing only when the action is the sole supervised query.
    self_forcing_action_only: bool = True
    # v-loss-weighting clip per modality (JiT Eq. 5).
    t_clip_dino: float = 0.05
    t_clip_depth: float = 0.05
    t_clip_action: float = 0.05
    t_clip_image: float = 0.05
    t_clip_tracks: float = 0.05
    # Per-modality loss weights (equalize magnitudes; 1.0 for v0).
    lambda_dino: float = 1.0
    lambda_depth: float = 1.0
    lambda_action: float = 1.0
    lambda_image: float = 1.0
    lambda_tracks: float = 1.0

    # ---- Modality-autoregressive generation ----
    # Per forward, each obs sensor (dino/depth) is dropped ENTIRELY (no history, no
    # future) with this probability -> trains missing-sensor robustness. 0 = never.
    sensor_drop_prob: float = 0.0
    # Fixed generation order and the observation modalities that provide clean
    # history context.
    generation_order: Tuple[str, ...] = ("dino", "depth", "action")
    history_modalities: Tuple[str, ...] = ("dino", "depth", "image")
    # Which dynamics modalities are GENERATED (noised, queried, supervised). None
    # means every present one.
    # Naming a subset makes the remaining present dynamics modalities
    # conditioning-only: they still supply clean history, but are never denoised
    # and carry no loss. That is what isolates one world-model target while every
    # arm keeps the same observation context -- restricting `modalities` instead
    # would also strip the history those other sensors provide.
    generated_modalities: Optional[Tuple[str, ...]] = None
    # Reuse clean history / generated-prefix K/V during sequential inference.
    # False keeps an exact full-prefix reference path for parity testing.
    use_kv_cache: bool = True

    # ---- Prediction parameterization (per modality) ----
    # x : network predicts the clean signal x; loss is x-MSE weighted by
    #     1/max(1-t, t_clip)^2 -- equivalent to canonical JiT's v-space loss with
    #     clamp_min(t_clip) (so t_clip plays JiT's t_eps role; JiT uses 0.05).
    # v : network predicts the flow-matching velocity v = x - noise directly;
    #     loss is plain (unweighted) v-MSE (standard rectified-flow / flow matching).
    pred_type_dino: str = "x"
    pred_type_depth: str = "x"
    pred_type_image: str = "x"
    pred_type_tracks: str = "x"
    pred_type_action: str = "x"

    # ---- Inference ODE ----
    solver: str = "heun"         # heun | euler
    steps_per_phase: int = 25
    # Optional test-time overrides for sequential generation. Empty mappings
    # preserve the global solver/step count above.
    solver_by_modality: dict[str, str] = field(default_factory=dict)
    steps_by_modality: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "MFConfig":
        """Build from a plain dict (e.g. OmegaConf model section), coercing the
        tuple-typed fields that round-trip through YAML as lists."""
        d = dict(d)
        for k in ("modalities", "rope_split", "logitnorm_dino", "logitnorm_depth",
                  "logitnorm_action", "logitnorm_image", "logitnorm_tracks",
                  "logitnorm_unified", "generation_order", "history_modalities",
                  "generated_modalities"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def n_obs_frames(self) -> int:
        return self.obs_history + self.obs_future

    @property
    def n_patches(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def spatial_height(self) -> int:
        return int(self.image_height if self.image_height is not None
                   else self.depth_img_size)

    @property
    def spatial_width(self) -> int:
        return int(self.image_width if self.image_width is not None
                   else self.depth_img_size)

    @property
    def grid_h(self) -> int:
        return int(self.grid_height if self.grid_height is not None else self.grid)

    @property
    def grid_w(self) -> int:
        return int(self.grid_width if self.grid_width is not None else self.grid)

    @property
    def track_grid_h(self) -> int:
        return int(self.track_grid_height if self.track_grid_height is not None
                   else self.track_grid)

    @property
    def track_grid_w(self) -> int:
        return int(self.track_grid_width if self.track_grid_width is not None
                   else self.track_grid)

    @property
    def depth_patch_dim(self) -> int:
        return self.depth_patch_size * self.depth_patch_size

    @property
    def image_patch_dim(self) -> int:
        return self.image_channels * self.image_patch_size * self.image_patch_size

    @property
    def n_track_points(self) -> int:
        return self.track_grid_h * self.track_grid_w

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    # ---- Modality registry ----
    _MOD_DATA_DIM = {"dino": "dino_dim", "depth": "depth_target_dim",
                     "image": "image_target_dim", "tracks": "track_dim",
                     "action": "action_dim"}
    _MOD_KIND = {"dino": "grid", "depth": "grid", "image": "grid",
                 "tracks": "grid", "action": "action"}
    # Grid obs modalities that carry a clean history frame (tracks are future-only).
    _MOD_HAS_HISTORY = {"dino": True, "depth": True, "image": True,
                        "tracks": False, "action": False}

    def modality_specs(self) -> list:
        """Ordered ModalitySpec list for the active `modalities`."""
        out = []
        for i, name in enumerate(self.modalities):
            if name not in self._MOD_KIND:
                raise ValueError(f"unknown modality {name!r}")
            data_dim = getattr(self, self._MOD_DATA_DIM[name])
            out.append(ModalitySpec(name, self._MOD_KIND[name], int(data_dim),
                                    self._MOD_HAS_HISTORY[name], i))
        return out

    @property
    def depth_target_dim(self) -> int:
        return self.depth_patch_dim

    @property
    def image_target_dim(self) -> int:
        return self.image_patch_dim

    def grid_modalities(self) -> list:
        """Names of the spatial-grid modalities in order (dino/depth/image/tracks)."""
        return [s.name for s in self.modality_specs() if s.kind == "grid"]

    def generated_modality_names(self) -> list:
        """Grid modalities that are denoised and supervised, in config order."""
        grid = self.grid_modalities()
        if self.generated_modalities is None:
            return grid
        return [name for name in grid if name in self.generated_modalities]

    def validated_generation_order(
            self, order: Optional[Tuple[str, ...]] = None) -> Tuple[str, ...]:
        """Validate and return a deployment order compatible with training."""
        resolved = tuple(self.generation_order if order is None else order)
        if len(resolved) != len(set(resolved)):
            raise ValueError(f"generation_order contains duplicates: {resolved}")
        known = {"dino", "depth", "image", "tracks", "action"}
        unknown = [name for name in resolved if name not in known]
        if unknown:
            raise ValueError(f"generation_order contains unknown modalities: {unknown}")
        missing = [
            name for name in self.generated_modality_names()
            if name not in resolved
        ]
        if missing:
            raise ValueError(
                f"generation_order omits generated modalities {missing}")
        if "action" in self.modalities:
            if "action" not in resolved:
                raise ValueError("generation_order omits the action modality")
            if resolved[-1] != "action":
                raise ValueError(
                    "causal inference requires action last in generation_order")
        return resolved

    def history_only_modalities(self) -> list:
        """Present grid modalities kept purely as clean conditioning context."""
        generated = set(self.generated_modality_names())
        return [name for name in self.grid_modalities() if name not in generated]

    def p_of(self, name: str) -> float:
        return float(getattr(self, f"p_{name}"))

    def logitnorm_of(self, name: str) -> Tuple[float, float]:
        return getattr(self, f"logitnorm_{name}")

    def t_clip_of(self, name: str) -> float:
        return float(getattr(self, f"t_clip_{name}"))

    def lambda_of(self, name: str) -> float:
        return float(getattr(self, f"lambda_{name}"))

    def pred_type_of(self, name: str) -> str:
        return getattr(self, f"pred_type_{name}")

    def n_expert_layers_of(self, name: str) -> int:
        override = getattr(self, f"n_expert_layers_{name}", None)
        return int(self.n_expert_layers if override is None else override)

    def expert_dim_of(self, name: str) -> int:
        override = getattr(self, f"expert_dim_{name}", None)
        return int(self.dim if override is None else override)

    def expert_n_heads_of(self, name: str) -> int:
        return self.expert_dim_of(name) // self.head_dim

    def obs_clock(self) -> list:
        """Clock positions of the obs (DINO/depth) frames, origin at frame 0."""
        return [i * self.obs_stride for i in range(self.n_obs_frames)]

    def tau_clock(self) -> int:
        """Clock position of the decision frame tau (the last history obs frame)."""
        return (self.obs_history - 1) * self.obs_stride

    def action_clock(self) -> list:
        """Clock positions of the action frames.

        tau is the current/decision frame = the last history obs frame, at clock
        (obs_history-1)*obs_stride. Actions are dense (stride 1) starting at tau
        through tau + action_horizon - 1. Future obs keyframes may extend beyond
        the action chunk when obs_future * obs_stride > action_horizon (decoupled
        dynamics horizon).
        """
        tau = self.tau_clock()
        return [tau + i for i in range(self.action_horizon)]

    def __post_init__(self):
        assert self.schedule_mode in (
            "modar", "unified", "action_only", "independent", "disjoint"), (
            f"unknown schedule_mode {self.schedule_mode}")
        if self.schedule_mode in ("modar", "independent"):
            self.validated_generation_order()
        if self.schedule_mode == "independent":
            valid_infer_schedule = self.infer_schedule in (
                "autoregressive", "unified", "futures_noise")
            if self.infer_schedule.startswith("futures_at:"):
                try:
                    future_t = float(self.infer_schedule.split(":", 1)[1])
                except ValueError:
                    future_t = -1.0
                valid_infer_schedule = 0.0 <= future_t <= 1.0
            assert valid_infer_schedule, (
                f"unknown independent infer_schedule {self.infer_schedule!r}")
            missing = [
                name for name in self.grid_modalities()
                if name not in self.generation_order
            ]
            assert not missing, (
                f"independent generation_order omits generated modalities "
                f"{missing}")
            if "action" in self.modalities:
                assert self.generation_order[-1] == "action", (
                    "independent autoregressive inference requires action last in "
                    "generation_order")
        assert self.window_mode in ("full", "legacy"), (
            f"unknown window_mode {self.window_mode!r}")
        assert self.action_pad_mode in ("hold", "mask"), (
            f"unknown action_pad_mode {self.action_pad_mode!r}")
        assert self.solver in ("heun", "euler"), f"unknown solver {self.solver}"
        assert all(solver in ("heun", "euler")
                   for solver in self.solver_by_modality.values()), (
            f"unknown per-modality solver in {self.solver_by_modality}")
        assert all(int(steps) > 0 for steps in self.steps_by_modality.values()), (
            f"per-modality steps must be positive: {self.steps_by_modality}")
        for name in self.modalities:
            pt = self.pred_type_of(name)
            assert pt in ("x", "v"), (
                f"unknown pred_type_{name} {pt!r} (expected 'x' or 'v')")
            assert self.n_expert_layers_of(name) >= 0, (
                f"n_expert_layers_{name} must be >= 0")
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert sum(self.rope_split) == self.head_dim, (
            f"rope_split {self.rope_split} must sum to head_dim {self.head_dim}")
        assert all(s % 2 == 0 for s in self.rope_split), (
            "each rope_split axis must be even")
        for name in self.modalities:
            width = self.expert_dim_of(name)
            assert width > 0 and width % self.head_dim == 0, (
                f"expert_dim_{name} {width} must be a positive multiple of "
                f"head_dim {self.head_dim} so rope_split applies unchanged")
        if "depth" in self.modalities:
            assert self.expert_dim_of("depth") >= self.depth_target_dim, (
                f"expert_dim_depth {self.expert_dim_of('depth')} must be >= "
                f"depth target dim {self.depth_target_dim} so the depth readout "
                "stays non-compressive")
        assert self.dim >= self.depth_target_dim, (
            f"dim {self.dim} must be >= depth target dim {self.depth_target_dim} so "
            "the depth target embedding is non-compressive (lossless)")
        assert self.spatial_height == self.grid_h * self.depth_patch_size, (
            "image height must equal grid height * depth_patch_size")
        assert self.spatial_width == self.grid_w * self.depth_patch_size, (
            "image width must equal grid width * depth_patch_size")
        assert self.track_pred_mode in ("delta", "absolute"), (
            f"unknown track_pred_mode {self.track_pred_mode}")
        assert "action" not in self.modalities or self.modalities[-1] == "action", (
            "'action' must be the last entry in `modalities` when present")
        assert len(set(self.modalities)) == len(self.modalities), (
            f"duplicate modality in {self.modalities}")
        if "tracks" in self.modalities:
            assert (self.track_grid_h, self.track_grid_w) == (
                self.grid_h, self.grid_w), (
                "track grid must equal the visual grid to reuse spatial RoPE")
        if self.generated_modalities is not None:
            grid = self.grid_modalities()
            unknown = [name for name in self.generated_modalities
                       if name not in grid]
            assert not unknown, (
                f"generated_modalities {unknown} are not present grid "
                f"modalities {grid}")
            assert self.generated_modality_names(), (
                "generated_modalities must name at least one grid modality")
            # Only the block-causal engine separates the generated set from the
            # conditioning set; every other mode noises whatever is present, so a
            # subset there would silently supervise modalities it claims to skip.
            assert self.schedule_mode == "modar", (
                f"generated_modalities is only supported for ModAR, "
                f"not {self.schedule_mode!r}")
            # A conditioning-only modality earns its tokens through history. Tracks
            # are future-only, so holding them out leaves them wholly unused while
            # still costing embedder/head parameters and a dataloader field.
            stranded = [name for name in self.history_only_modalities()
                        if not self._MOD_HAS_HISTORY[name]
                        or name not in self.history_modalities]
            assert not stranded, (
                f"modalities {stranded} are neither generated nor usable as "
                "history; drop them from `modalities` instead")
