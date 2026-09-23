"""Lean dataset/loader for the Modality-Forcing WAM.

Reads per-demo ``trajectory.pt`` packs containing configured visual and/or
action modalities and emits exactly the fixed window the model consumes:

  - DINO + depth at ``n_obs_frames`` frames around tau, stride ``obs_stride``
    (first ``obs_history`` are clean history, the rest are future targets).
    Keyframe packs store one fixed ``keyframe_stride``; ``obs_stride`` must be an
    integer multiple of it and the loader takes every ratio-th keyframe, so one
    pack can serve several obs layouts without being rebuilt;
  - ``action_horizon`` dense actions starting at tau;
  - proprio at tau.

Packs are loaded with ``mmap=True`` so each sample only pages in the handful of
frames it needs, not the whole multi-hundred-MB file.
"""
from __future__ import annotations

import glob
import json
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from util.depth_utils import normalize_depth_maps
from util.modality_forcing.codec import decode_depth_u16_png, decode_rgb_jpeg
from util.modality_forcing.config import MFConfig

# ---- discovery / splitting --------------------------------------------------

def canonical_task_vocab(tasks: List[str]) -> List[str]:
    """Return the single canonical task-embedding order used everywhere.

    Task IDs are part of the checkpoint contract.  Historically the trainer
    sorted the configured vocabulary while deployment preserved YAML order,
    silently permuting task embeddings.  Centralize the trainer's established
    ordering and reject duplicates so every producer/consumer resolves the same
    name -> row mapping.
    """
    names = [str(task) for task in tasks]
    if len(names) != len(set(names)):
        duplicates = sorted(
            name for name in set(names) if names.count(name) > 1)
        raise ValueError(f"task vocabulary contains duplicates: {duplicates}")
    return sorted(names)


def build_task_to_id(tasks: List[str]) -> Dict[str, int]:
    """Map an already-resolved vocabulary to contiguous embedding rows."""
    return {t: i for i, t in enumerate(tasks)}


def build_source_task_to_id(
    tasks: List[str],
    task_vocab: Optional[List[str]] = None,
    task_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    """Map physical source task directories onto canonical embedding rows."""
    aliases = dict(task_aliases or {})
    unknown_aliases = sorted(set(aliases) - set(tasks))
    if unknown_aliases:
        raise ValueError(
            f"task_aliases contains tasks not configured for this source: "
            f"{unknown_aliases}")
    canonical = [aliases.get(task, task) for task in tasks]
    vocab = canonical_task_vocab(
        list(task_vocab) if task_vocab is not None else canonical)
    canonical_to_id = build_task_to_id(vocab)
    unknown_targets = sorted(set(canonical) - set(canonical_to_id))
    if unknown_targets:
        raise ValueError(
            f"canonical source tasks are absent from data.task_vocab: "
            f"{unknown_targets}")
    if len(set(canonical)) != len(canonical):
        raise ValueError(
            f"multiple physical tasks map to the same canonical task: {canonical}")
    return {
        task: canonical_to_id[canonical_name]
        for task, canonical_name in zip(tasks, canonical)
    }


def discover_demos(data_root: str, task: str) -> List[str]:
    root = os.path.join(data_root, task)
    demos = sorted(glob.glob(os.path.join(root, "demo_*")))
    demos = [d for d in demos if os.path.exists(os.path.join(d, "trajectory.pt"))]
    if not demos:
        raise ValueError(f"No demos with trajectory.pt under {root}")
    return demos


def split_demos(demos: List[str], val_ratio: float, seed: int
                ) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(demos))
    n_val = int(round(len(demos) * val_ratio))
    val_idx = set(order[:n_val].tolist())
    train = [d for i, d in enumerate(demos) if i not in val_idx]
    val = [d for i, d in enumerate(demos) if i in val_idx]
    return train, val


def split_demos_train_count(demos: List[str], train_count: int, seed: int
                            ) -> Tuple[List[str], List[str]]:
    """Select an exact training count and use every remaining demo for val."""
    n = len(demos)
    if not 0 < train_count < n:
        raise ValueError(
            f"train_count must be in [1, {n - 1}], got {train_count} "
            f"for {n} demos")
    rng = np.random.RandomState(seed)
    train_idx = set(rng.permutation(n)[:train_count].tolist())
    train = [d for i, d in enumerate(demos) if i in train_idx]
    val = [d for i, d in enumerate(demos) if i not in train_idx]
    return train, val


def _resolve_train_count(train_count, task: str) -> Optional[int]:
    if train_count is None:
        return None
    if isinstance(train_count, Mapping):
        if task not in train_count:
            raise ValueError(f"train_count mapping has no task {task!r}")
        return int(train_count[task])
    return int(train_count)


def split_demos_manifest(
    demos_by_task: dict[str, List[str]],
    manifest_path: str | os.PathLike,
) -> tuple[dict[str, List[str]], dict[str, List[str]]]:
    """Load an explicit whole-demo split and verify it matches discovery."""
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text())
    configured = payload.get("tasks")
    if not isinstance(configured, dict):
        raise ValueError(f"{path}: split manifest must contain a tasks mapping")
    train_by_task, val_by_task = {}, {}
    for task, demos in demos_by_task.items():
        if task not in configured:
            raise ValueError(f"{path}: split manifest has no task {task!r}")
        entry = configured[task]
        train_names = list(entry.get("train", []))
        val_names = list(entry.get("val", []))
        if len(train_names) != len(set(train_names)):
            raise ValueError(f"{path}: duplicate {task} train demo")
        if len(val_names) != len(set(val_names)):
            raise ValueError(f"{path}: duplicate {task} val demo")
        overlap = sorted(set(train_names) & set(val_names))
        if overlap:
            raise ValueError(f"{path}: {task} split overlap: {overlap}")
        by_name = {Path(demo).name: demo for demo in demos}
        listed = set(train_names) | set(val_names)
        missing = sorted(listed - set(by_name))
        unassigned = sorted(set(by_name) - listed)
        if missing or unassigned:
            raise ValueError(
                f"{path}: {task} split mismatch; missing={missing}, "
                f"unassigned={unassigned}")
        train_by_task[task] = [by_name[name] for name in train_names]
        val_by_task[task] = [by_name[name] for name in val_names]
    extra_tasks = sorted(set(configured) - set(demos_by_task))
    if extra_tasks:
        raise ValueError(f"{path}: unconfigured manifest tasks: {extra_tasks}")
    return train_by_task, val_by_task


def discover_demos_multi(data_root: str, tasks: List[str]) -> dict:
    """Discover demos for several tasks; returns {task: [demo_dirs]}."""
    return {t: discover_demos(data_root, t) for t in tasks}


def _round_robin(lists: List[List[str]]) -> List[str]:
    """Interleave several lists round-robin: [a0,b0,c0,a1,b1,...], skipping
    exhausted lists. Any prefix spans all (still-nonempty) lists -- so an
    unshuffled loader's first batch covers every task instead of just one."""
    out = []
    for i in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if i < len(x):
                out.append(x[i])
    return out


def split_demos_modality(demos_by_task: dict, action_val_ratio: float,
                         dyn_val_ratio: float, seed: int) -> dict:
    """Nested, per-task (stratified) per-modality demo split.

    For each task we shuffle once and cut: ``dyn_train`` = first (1-dyn_val_ratio),
    ``action_train`` = first (1-action_val_ratio) -- a strict subset of dyn_train.
    ``action_val`` / ``dyn_val`` are the respective complements (they overlap: the
    dyn_val tail is also part of action_val). Pools are pooled across tasks.
    """
    per_task = {"action_train": [], "dyn_train": [], "action_val": [], "dyn_val": []}
    for ti, task in enumerate(sorted(demos_by_task)):
        demos = demos_by_task[task]
        rng = np.random.RandomState(seed + ti)        # per-task offset -> stratified
        order = rng.permutation(len(demos))
        demos_sh = [demos[i] for i in order]
        n = len(demos_sh)
        n_dyn = int(round(n * (1.0 - dyn_val_ratio)))
        n_act = min(int(round(n * (1.0 - action_val_ratio))), n_dyn)   # nested
        per_task["dyn_train"].append(demos_sh[:n_dyn])
        per_task["dyn_val"].append(demos_sh[n_dyn:])
        per_task["action_train"].append(demos_sh[:n_act])
        per_task["action_val"].append(demos_sh[n_act:])
    # Round-robin across tasks so an unshuffled loader's first batch (e.g. the
    # fixed val viz batch) spans every task rather than just the first one.
    return {k: _round_robin(v) for k, v in per_task.items()}


def split_demos_cotrain(demos_by_task: dict, n_action_train: int,
                        n_action_val: int, n_dyn_val: int,
                        n_dyn_train: Optional[int], seed: int,
                        split_universe: Optional[int] = None) -> dict:
    """Count-based NESTED per-task co-train split (big dynamics pool + small
    action pool from one pack).

    Per task, shuffle once with a fixed per-task seed (``seed + ti``, ti = the
    task's position in ``demos_by_task`` iteration order, i.e. the config task
    list), then cut::

        dyn_val      = sh[:n_dyn_val]
        action_val   = sh[:n_action_val]                        (subset of dyn_val)
        dyn_train    = sh[n_dyn_val : n_dyn_val + n_dyn_train]   (rest if None)
        action_train = sh[n_dyn_val : n_dyn_val + n_action_train](subset of dyn_train)

    Nesting makes both val pools leak-free by construction (action_val subset of
    dyn_val, and the val block precedes the train block, so no val demo is seen
    by either training stream). Pools are round-robin pooled across tasks. This
    replaces the ratio-based ``split_demos_modality`` with explicit counts so the
    big/small pool sizes are exact and independent of per-task demo counts.

    ``split_universe`` makes the split stable when a task's pack pool grows.
    ``rng.permutation(n)`` depends on n, so without it, adding demos reshuffles
    everything and the val pools of a 500-demo and a 2100-demo run share no
    demos. With it, only the first ``split_universe`` demos (sorted by id, which
    is the collection episode index) enter the original permutation; the rest are
    permuted separately and appended to the dyn_train tail. The val block and the
    head of dyn_train are then byte-identical to the smaller run, so a larger
    pool is a strict superset of the smaller one's training data.
    """
    if n_action_val > n_dyn_val:
        raise ValueError(f"n_action_val ({n_action_val}) > n_dyn_val ({n_dyn_val})")
    per_task = {"action_train": [], "dyn_train": [], "action_val": [], "dyn_val": []}
    for ti, (task, demos) in enumerate(demos_by_task.items()):
        head = demos if split_universe is None else demos[:int(split_universe)]
        tail = [] if split_universe is None else demos[int(split_universe):]
        sh = [head[i] for i in np.random.RandomState(seed + ti).permutation(len(head))]
        if tail:
            tail_rng = np.random.RandomState(seed + ti + 1_000_000)
            sh += [tail[i] for i in tail_rng.permutation(len(tail))]
        n = len(sh)
        n_dt = (n - n_dyn_val) if n_dyn_train is None else int(n_dyn_train)
        if n_dyn_val + n_dt > n:
            raise ValueError(
                f"{task}: {n} demos < n_dyn_val ({n_dyn_val}) + n_dyn_train ({n_dt})")
        if n_action_train > n_dt:
            raise ValueError(
                f"{task}: n_action_train ({n_action_train}) > dyn_train ({n_dt})")
        per_task["dyn_val"].append(sh[:n_dyn_val])
        per_task["action_val"].append(sh[:n_action_val])
        per_task["dyn_train"].append(sh[n_dyn_val:n_dyn_val + n_dt])
        per_task["action_train"].append(sh[n_dyn_val:n_dyn_val + n_action_train])
    return {k: _round_robin(v) for k, v in per_task.items()}


@lru_cache(maxsize=64)
def _load_pack(path: str) -> dict:
    """mmap-load a demo's trajectory.pt. Cached per-process (per dataloader worker)
    so repeated draws of the same demo skip re-parsing the zip archive; tensors are
    mmap-backed and read-only (callers clone the slices they keep), so sharing is
    safe. maxsize keeps the open-fd count bounded across workers."""
    return torch.load(os.path.join(path, "trajectory.pt"),
                      map_location="cpu", mmap=True, weights_only=False)


MIN_BATCHES_PER_EPOCH = 200


def _build_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int,
                  persistent: bool) -> DataLoader:
    """DataLoader factory that transparently shards TRAINING loaders across DDP
    ranks. When a process group is initialized we attach a DistributedSampler to
    the shuffled (train) loaders so each rank sees a disjoint 1/world_size shard;
    unshuffled (val) loaders are left un-sharded -- eval runs on rank 0 over the
    full split. batch_size stays per-rank, so the effective global batch is
    batch_size * world_size.

    Training epochs are stretched when the demonstration list would otherwise
    yield only a few batches. Revisiting a demonstration draws a fresh random
    window. Validation loaders keep their true length."""
    if shuffle and num_workers > 0 and hasattr(ds, "epoch_repeat"):
        world = (dist.get_world_size()
                 if dist.is_available() and dist.is_initialized() else 1)
        need = MIN_BATCHES_PER_EPOCH * max(1, batch_size) * world
        ds.epoch_repeat = max(1, -(-need // max(1, ds.n_demos)))
    kwargs = dict(batch_size=min(batch_size, len(ds)), num_workers=num_workers,
                  pin_memory=True, persistent_workers=persistent, drop_last=shuffle)
    if shuffle and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        return DataLoader(ds, sampler=sampler, **kwargs)
    return DataLoader(ds, shuffle=shuffle, **kwargs)


# ---- normalization stats ----------------------------------------------------

class MFStats:
    def __init__(self, action_mean, action_std, proprio_mean, proprio_std,
                 dino_mean, dino_std):
        self.action_mean = action_mean
        self.action_std = action_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std
        self.dino_mean = float(dino_mean)
        self.dino_std = float(dino_std)


def compute_stats(train_demos: List[str], cfg: MFConfig,
                  max_dino_demos: int = 30, dino_frames_per_demo: int = 8,
                  seed: int = 0) -> MFStats:
    first_pack = _load_pack(train_demos[0])
    visual_only = bool(first_pack.get("visual_only", False))
    # action / proprio: per-dim mean/std over all train demos (cheap tensors).
    a_sum = torch.zeros(cfg.action_dim, dtype=torch.float64)
    a_sq = torch.zeros(cfg.action_dim, dtype=torch.float64)
    p_sum = torch.zeros(cfg.proprio_dim, dtype=torch.float64)
    p_sq = torch.zeros(cfg.proprio_dim, dtype=torch.float64)
    n = 0
    if visual_only:
        a_mean, a_std = torch.zeros(cfg.action_dim), torch.ones(cfg.action_dim)
        p_mean, p_std = torch.zeros(cfg.proprio_dim), torch.ones(cfg.proprio_dim)
    else:
        for d in train_demos:
            pack = _load_pack(d)
            a = pack["actions"].double()
            p = pack["proprio"].double()
            a_sum += a.sum(0); a_sq += (a * a).sum(0)
            p_sum += p.sum(0); p_sq += (p * p).sum(0)
            n += a.shape[0]
        a_mean = (a_sum / n).float()
        a_std = ((a_sq / n - (a_sum / n) ** 2).clamp(min=1e-8).sqrt()).float()
        p_mean = (p_sum / n).float()
        p_std = ((p_sq / n - (p_sum / n) ** 2).clamp(min=1e-8).sqrt()).float()

    # dino: global scalar mean/std over a sample of demos/frames.
    rng = np.random.RandomState(seed)
    sample = train_demos if len(train_demos) <= max_dino_demos else \
        [train_demos[i] for i in rng.choice(len(train_demos), max_dino_demos, replace=False)]
    d_sum = 0.0; d_sq = 0.0; d_cnt = 0
    for d in sample:
        pack = _load_pack(d)
        dino = pack["dino_tokens"]
        T = dino.shape[0]
        idx = rng.choice(T, min(dino_frames_per_demo, T), replace=False)
        x = dino[idx].float()
        d_sum += x.sum().item(); d_sq += (x * x).sum().item(); d_cnt += x.numel()
    d_mean = d_sum / d_cnt
    d_std = (d_sq / d_cnt - d_mean ** 2) ** 0.5
    return MFStats(a_mean, a_std, p_mean, p_std, d_mean, d_std)


# ---- dataset ----------------------------------------------------------------

class ModalityForcingDataset(Dataset):
    def __init__(self, demos: List[str], cfg: MFConfig, stats: MFStats,
                 depth_mean: float, depth_std: float, depth_max_m: float = 10.0,
                 depth_norm_mode: str = "log", random_window: bool = True,
                 seed: int = 0, task_to_id: Optional[Dict[str, int]] = None,
                 epoch_repeat: int = 1):
        self.cfg = cfg
        self.stats = stats
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.depth_max_m = depth_max_m
        self.depth_norm_mode = depth_norm_mode
        self.random_window = random_window
        self.base_seed = seed
        self.task_to_id = task_to_id or {}
        # How many times the demo list is walked before the loader's iterator
        # ends. See _build_loader: with a few hundred demos split over 8 ranks a
        # true epoch is one or two batches, so the iterator restarts constantly
        # and the prefetch pipeline never fills. Revisiting a demo does NOT
        # repeat a sample -- _tau draws from the ambient numpy RNG, so each
        # visit is a fresh random window, exactly as a further epoch would be.
        self.epoch_repeat = max(1, int(epoch_repeat))
        # Optional explicit (demo_index, decision_k_or_tau) list. When set,
        # __len__/__getitem__ walk that plan instead of one middle/random
        # window per demo. Used by full-val generation eval so every checkpoint
        # sees the same pinned random windows.
        self.window_plan = None

        self.stride = cfg.obs_stride
        self.tau_lo = (cfg.obs_history - 1) * cfg.obs_stride
        # min trajectory length needed so a valid tau exists.
        self.span_future_obs = cfg.obs_future * cfg.obs_stride
        self.active = set(cfg.modalities)
        self.want_image = "image" in self.active
        self.want_tracks = "tracks" in self.active
        # Also decode RGB when tracks are active (even if image is NOT a modeled
        # modality): the track visualization overlays predicted/GT tracks on the
        # real RGB frame. The model ignores an `images` tensor when image is off.
        self.load_images = self.want_image or self.want_tracks
        # Depth is the only sensor whose bytes must be INFLATED to be read: dino
        # tokens are precomputed and RGB is gated above, so on an arm that does not
        # model depth this decode was the whole loader cost and bought nothing.
        # That matters because these steps are loader-bound, not GPU-bound.
        #
        # The condition includes history even though _prep_data keys off
        # `modalities` alone. Naming depth as context without modelling it is a
        # supported ablation (see MFConfig.history_modalities), and being
        # over-inclusive here can only cost a decode, whereas being under-inclusive
        # would silently feed the model zeros.
        self.want_depth = ("depth" in self.active
                           or "depth" in tuple(cfg.history_modalities or ()))
        # Keep demos that admit at least one valid decision frame. For the
        # keyframe pack (pack_version >= 2) the decision frame is a keyframe
        # index k; for the legacy dense pack it is a native tau.
        self.demos = []
        self.lengths = []       # T (dense action length)
        self.versions = []      # pack_version
        self.visual_only = []
        self.n_keyframes = []   # Ts (v2 only; 0 for v1)
        self.track_futures = []  # stored track horizon (v2 only; 0 when unused)
        self.kf_strides = []    # pack keyframe gap in native frames (v2 only; 0 for v1)
        self.kf_ratios = []     # obs_stride / keyframe_stride (v2 only; 1 for v1)
        for d in demos:
            if self.task_to_id:
                task_name = Path(d).parent.name
                if task_name not in self.task_to_id:
                    raise ValueError(
                        f"demo {d} task {task_name!r} not in task_to_id "
                        f"{list(self.task_to_id)}")
            pack = _load_pack(d)
            ver = int(pack.get("pack_version", 1))
            visual_only = bool(pack.get("visual_only", False))
            T = int(pack["traj_len"]) if visual_only else int(pack["actions"].shape[0])
            if ver >= 2:
                Ts = int(pack["n_keyframes"])
                self._validate_v2_geometry(pack, d)
                kf_stride = int(pack["keyframe_stride"])
                if cfg.obs_stride % kf_stride:
                    raise ValueError(
                        f"{d}: cfg.obs_stride {cfg.obs_stride} must be a multiple "
                        f"of the pack's keyframe_stride {kf_stride}")
                ratio = cfg.obs_stride // kf_stride
                track_future = int(pack["point_tracks"].shape[1]) if self.want_tracks else 0
                if self.want_tracks and cfg.obs_future * ratio > track_future:
                    raise ValueError(
                        f"requested track horizon {cfg.obs_future} frames at "
                        f"{ratio} keyframes each needs {cfg.obs_future * ratio} "
                        f"packed offsets but {d} has {track_future}")
                k_lo, k_hi = self._v2_k_range(
                    T, Ts, kf_stride, ratio, visual_only, track_future)
                if k_hi >= k_lo:
                    self.demos.append(d); self.lengths.append(T)
                    self.versions.append(ver); self.n_keyframes.append(Ts); self.visual_only.append(visual_only)
                    self.track_futures.append(track_future)
                    self.kf_strides.append(kf_stride); self.kf_ratios.append(ratio)
            else:
                tau_hi = min(T - 1 - self.span_future_obs, T - cfg.action_horizon)
                if tau_hi >= self.tau_lo:
                    self.demos.append(d); self.lengths.append(T)
                    self.versions.append(ver); self.n_keyframes.append(0); self.visual_only.append(False)
                    self.track_futures.append(0)
                    self.kf_strides.append(0); self.kf_ratios.append(1)
        if not self.demos:
            raise ValueError("No demos long enough for the configured window.")

    def _validate_v2_geometry(self, pack, demo):
        cfg = self.cfg
        expected_hw = (cfg.spatial_height, cfg.spatial_width)
        expected_n = cfg.n_patches
        dino = pack["dino_tokens"]
        if int(dino.shape[1]) != expected_n:
            raise ValueError(
                f"{demo}: DINO has {int(dino.shape[1])} patches, expected "
                f"{cfg.grid_h}x{cfg.grid_w}={expected_n}")
        packed_hw = (
            int(pack.get("image_height", expected_hw[0])),
            int(pack.get("image_width", expected_hw[1])),
        )
        if packed_hw != expected_hw:
            raise ValueError(
                f"{demo}: pack image size {packed_hw} != config {expected_hw}")
        if self.want_depth:
            depth_hw = decode_depth_u16_png(
                pack["depths_png"][0], self.depth_max_m).shape
            if tuple(depth_hw) != expected_hw:
                raise ValueError(
                    f"{demo}: depth size {tuple(depth_hw)} != config {expected_hw}")
        if self.load_images:
            image_hw = decode_rgb_jpeg(pack["images_jpg"][0]).shape[:2]
            if tuple(image_hw) != expected_hw:
                raise ValueError(
                    f"{demo}: RGB size {tuple(image_hw)} != config {expected_hw}")
        if self.want_tracks and int(pack["point_tracks"].shape[2]) != expected_n:
            raise ValueError(
                f"{demo}: tracks have {int(pack['point_tracks'].shape[2])} "
                f"points, expected {expected_n}")

    @property
    def n_demos(self):
        """Distinct demos, independent of epoch_repeat. Epoch accounting must use
        this: 'epochs' means passes over the DATA, and inflating it with the
        repeat factor would silently rescale a number runs are compared on."""
        return len(self.demos)

    def __len__(self):
        n = len(self.window_plan) if self.window_plan is not None else len(self.demos)
        return n * self.epoch_repeat

    def _decision_range(self, i):
        """Inclusive (lo, hi) decision-frame range for demo ``i``."""
        if self.versions[i] >= 2:
            return self._v2_k_range(
                self.lengths[i], self.n_keyframes[i], self.kf_strides[i],
                self.kf_ratios[i], self.visual_only[i], self.track_futures[i])
        T = self.lengths[i]
        if self.cfg.window_mode == "legacy":
            tau_hi = min(T - 1 - self.span_future_obs,
                         T - self.cfg.action_horizon)
        else:
            tau_hi = T - 1
        return self.tau_lo, tau_hi

    def plan_windows(self, policy: str, seed: int = 0, per_demo: int = 1):
        """Pin a (demo, decision-frame) plan for eval.

        ``middle`` is the val-loader default. ``random`` draws ``per_demo``
        uniform frames per demo from a dedicated RNG so two datasets with the
        same demo order and seed score the same windows. ``all`` walks every
        valid decision frame.
        """
        if policy not in ("middle", "random", "all"):
            raise ValueError(f"unknown window policy {policy!r}")
        rng = np.random.RandomState(int(seed))
        plan = []
        for i in range(self.n_demos):
            lo, hi = self._decision_range(i)
            if hi < lo:
                continue
            if policy == "middle":
                ks = [(lo + hi) // 2]
            elif policy == "all":
                ks = range(lo, hi + 1)
            else:
                n = hi - lo + 1
                take = min(max(1, int(per_demo)), n)
                ks = rng.choice(np.arange(lo, hi + 1), size=take, replace=False)
                ks = sorted(int(k) for k in ks)
            plan.extend((i, int(k)) for k in ks)
        self.window_plan = plan
        return plan

    def _v2_k_range(self, T, Ts, kf_stride, ratio, visual_only=False,
                    track_future=0):
        """Valid decision-keyframe range. ``ratio`` obs frames are ``ratio``
        keyframes apart, so the window spans ``ratio`` times as many keyframes."""
        cfg = self.cfg
        k_lo = (cfg.obs_history - 1) * ratio
        if cfg.window_mode != "legacy":
            # Only the history has to fit. Obs futures, packed tracks and the
            # action chunk are replicate-padded past the end of the demo and
            # masked out of the loss, so the range depends solely on the demo
            # length -- the same windows for every arm, and every action index
            # lands in at least one of them.
            #
            # Keyframes are cut from the RGB stream, which can run one frame
            # past the action/proprio arrays, so the decision frame still has to
            # be one the robot state actually covers.
            if visual_only:
                return k_lo, Ts - 1
            return k_lo, min(Ts - 1, (T - 1) // kf_stride)
        span = cfg.obs_future * ratio
        k_hi = Ts - 1 - span if visual_only else min(
            Ts - 1 - span, (T - cfg.action_horizon) // kf_stride)
        # Packs store tracks for one fixed maximum horizon and mark only those
        # decision frames valid. A shorter configured horizon can reuse a subset of
        # the packed offsets, but it must retain the pack's original valid k range.
        if track_future:
            k_hi = min(k_hi, Ts - 1 - track_future)
        return k_lo, k_hi

    def _tau(self, idx, T):
        if self.cfg.window_mode == "legacy":
            tau_hi = min(T - 1 - self.span_future_obs,
                         T - self.cfg.action_horizon)
        else:
            tau_hi = T - 1
        if self.random_window:
            return int(np.random.randint(self.tau_lo, tau_hi + 1))
        return (self.tau_lo + tau_hi) // 2

    def _future_valid(self, wanted, last):
        """Per-future-frame validity for indices ``wanted``, capped at ``last``."""
        return torch.tensor([float(j <= last) for j in wanted])

    def _action_window(self, pack, tau, T):
        """Action chunk from ``tau``, replicate-padded past the end of the demo.

        Padding repeats the final action, i.e. "hold the terminal pose". Whether
        those steps are supervised is decided by ``cfg.action_pad_mode``.
        """
        horizon = self.cfg.action_horizon
        idx = [min(tau + i, T - 1) for i in range(horizon)]
        valid = self._future_valid([tau + i for i in range(horizon)], T - 1)
        return pack["actions"][idx].float(), valid

    def _window_tracks(self, pack, k, ratio):
        """Future tracks for the window at ``k``, plus per-frame validity.

        Packs only hold tracks for offsets whose horizon fits inside the demo, so
        the tail of every demo has missing offsets stored as zeros. Those repeat
        the last valid ABSOLUTE position instead (falling back to the seed grid,
        i.e. no motion at all), which keeps the teacher block a coherent "the
        points stop moving" future rather than a jump to the image corner.
        """
        cfg = self.cfg
        sel = slice(ratio - 1, cfg.obs_future * ratio, ratio)
        tr = pack["point_tracks"][k].float().clone()        # (packed_offsets,N,3)
        offset_valid = pack.get("track_offset_valid", None)
        if offset_valid is None:
            # Legacy packs only record all-or-nothing validity per keyframe.
            valid = torch.full((tr.shape[0],),
                               float(bool(pack["track_valid"][k])))
        else:
            valid = offset_valid[k].float()
        tr, valid = tr[sel].clone(), valid[sel].clone()
        seed = pack["track_seed"].float()                   # (N,2) normalized
        held = torch.cat([seed, torch.ones_like(seed[:, :1])], dim=-1)
        for j in range(tr.shape[0]):
            if valid[j] > 0:
                held = tr[j]
            else:
                tr[j] = held
        if cfg.track_pred_mode == "delta":
            tr[..., :2] = tr[..., :2] - seed[None]
        return tr.contiguous(), valid

    def _norm_common(self, dino, depth, actions, proprio):
        dino = (dino - self.stats.dino_mean) / self.stats.dino_std
        if depth is not None:
            depth = normalize_depth_maps(
                depth, self.depth_mean, self.depth_std,
                max_depth_m=self.depth_max_m, mode=self.depth_norm_mode)
        actions = (actions - self.stats.action_mean) / self.stats.action_std
        proprio = (proprio - self.stats.proprio_mean) / self.stats.proprio_std
        return dino, depth, actions, proprio

    def _depth_placeholder(self):
        """Shape-correct stand-in for skipped depth.

        The key stays in the batch rather than being dropped the way `images` is,
        because ModalityForcingWAM.forward takes depth_maps POSITIONALLY and the
        viz path slices it before the model can decide it is unused. Nothing reads
        the values: _prep_data only patchifies depth when it is a present
        modality.
        """
        cfg = self.cfg
        return torch.zeros(
            (cfg.n_obs_frames, cfg.spatial_height, cfg.spatial_width))

    def __getitem__(self, idx):
        if self.window_plan is not None:
            demo_i, forced = self.window_plan[idx % len(self.window_plan)]
            if self.versions[demo_i] >= 2:
                return self._getitem_v2(demo_i, k=forced)
            return self._getitem_v1(demo_i, tau=forced)
        idx %= len(self.demos)
        if self.versions[idx] >= 2:
            return self._getitem_v2(idx)
        return self._getitem_v1(idx)

    def _getitem_v1(self, idx, tau=None):
        cfg = self.cfg
        d = self.demos[idx]
        T = self.lengths[idx]
        pack = _load_pack(d)
        tau = self._tau(idx, T) if tau is None else int(tau)

        h = cfg.obs_history
        want = [tau + (i - (h - 1)) * self.stride for i in range(cfg.n_obs_frames)]
        obs_native = [min(j, T - 1) for j in want]

        dino = pack["dino_tokens"][obs_native].float()              # (F,N,384)
        depth = (pack["depths"][obs_native].float()                 # (F,H,W)
                 if self.want_depth else None)
        actions, action_valid = self._action_window(pack, tau, T)   # (A,12)
        proprio = pack["proprio"][tau].float()                      # (12,)
        dino, depth, actions, proprio = self._norm_common(dino, depth, actions, proprio)

        out = {
            "dino": dino.clone(),
            "depth_maps": (depth.clone() if depth is not None
                           else self._depth_placeholder()),
            "actions": actions.clone(),
            "proprio": proprio.clone(),
            "obs_future_valid": self._future_valid(want[h:], T - 1),
            "action_valid": action_valid,
        }
        if self.task_to_id:
            task_name = Path(d).parent.name
            out["task_id"] = torch.tensor(self.task_to_id[task_name], dtype=torch.long)
        return out

    def _getitem_v2(self, idx, k=None):
        cfg = self.cfg
        d = self.demos[idx]
        T = self.lengths[idx]
        Ts = self.n_keyframes[idx]
        pack = _load_pack(d)
        visual_only = self.visual_only[idx]
        kf_stride, ratio = self.kf_strides[idx], self.kf_ratios[idx]
        k_lo, k_hi = self._v2_k_range(
            T, Ts, kf_stride, ratio, visual_only, self.track_futures[idx])
        if k is None:
            if self.random_window:
                k = int(np.random.randint(k_lo, k_hi + 1))
            else:
                k = (k_lo + k_hi) // 2
        else:
            k = int(k)

        h = cfg.obs_history
        want = [k + (i - (h - 1)) * ratio for i in range(cfg.n_obs_frames)]
        kf_idx = [min(j, Ts - 1) for j in want]

        dino = pack["dino_tokens"][kf_idx].float()                  # (F,N,384)
        depth = (torch.from_numpy(np.stack(
            [decode_depth_u16_png(pack["depths_png"][j], self.depth_max_m)
             for j in kf_idx], 0))                                  # (F,H,W)
            if self.want_depth else None)
        if visual_only:
            actions = torch.zeros((cfg.action_horizon, cfg.action_dim))
            proprio = torch.zeros(cfg.proprio_dim)
            action_valid = torch.zeros(cfg.action_horizon)
        else:
            tau = k * kf_stride
            actions, action_valid = self._action_window(pack, tau, T)
            proprio = pack["proprio"][tau].float()
        dino, depth, actions, proprio = self._norm_common(dino, depth, actions, proprio)

        out = {
            "dino": dino.clone(),
            "depth_maps": (depth.clone() if depth is not None
                           else self._depth_placeholder()),
            "actions": actions.clone(),
            "proprio": proprio.clone(),
            "obs_future_valid": self._future_valid(want[h:], Ts - 1),
            "action_valid": action_valid,
        }
        if self.load_images:
            imgs = np.stack([decode_rgb_jpeg(pack["images_jpg"][j]) for j in kf_idx], 0)
            img = torch.from_numpy(imgs).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
            out["images"] = img.contiguous()                        # (F,3,H,W) in [-1,1]
        if self.want_tracks:
            # Packed offset j is keyframe k+j+1, so the frames this window wants
            # (k+ratio, k+2*ratio, ...) live at indices ratio-1, 2*ratio-1, ...
            tracks, track_valid = self._window_tracks(pack, k, ratio)
            if cfg.window_mode == "legacy" and not bool(track_valid.all()):
                raise ValueError(f"Invalid track horizon in {d} at keyframe {k}")
            out["point_tracks"] = tracks
            out["track_future_valid"] = track_valid
        if self.task_to_id:
            task_name = Path(d).parent.name
            out["task_id"] = torch.tensor(self.task_to_id[task_name], dtype=torch.long)
        return out


def make_loaders(cfg: MFConfig, data_root: str, task: str, batch_size: int,
                 val_ratio: float, num_workers: int, seed: int,
                 depth_mean: float, depth_std: float, depth_max_m: float = 10.0,
                 depth_norm_mode: str = "log",
                 stats: Optional[MFStats] = None,
                 demo_limit: Optional[int] = None,
                 train_random_window: bool = True,
                 task_vocab: Optional[List[str]] = None,
                 task_aliases: Optional[Dict[str, str]] = None,
                 split_manifest: Optional[str] = None,
                 train_count: Optional[int] = None):
    demos = discover_demos(data_root, task)
    if demo_limit is not None:
        demos = demos[:demo_limit]
    if split_manifest and train_count is not None:
        raise ValueError("split_manifest and train_count are mutually exclusive")
    if split_manifest:
        train_by_task, val_by_task = split_demos_manifest(
            {task: demos}, split_manifest)
        train_demos, val_demos = train_by_task[task], val_by_task[task]
    elif train_count is not None:
        train_demos, val_demos = split_demos_train_count(
            demos, _resolve_train_count(train_count, task), seed)
    else:
        train_demos, val_demos = split_demos(demos, val_ratio, seed)
    if stats is None:
        stats = compute_stats(train_demos, cfg, seed=seed)
    task_to_id = build_source_task_to_id(
        [task], task_vocab=task_vocab, task_aliases=task_aliases)

    def _mk(ds_demos, shuffle, random_window):
        if not ds_demos:
            return None
        ds = ModalityForcingDataset(
            ds_demos, cfg, stats, depth_mean, depth_std, depth_max_m,
            depth_norm_mode, random_window=random_window, seed=seed,
            task_to_id=task_to_id)
        return _build_loader(ds, batch_size, shuffle, num_workers,
                             persistent=(num_workers > 0))

    # train_random_window=False pins each demo to its single deterministic
    # (middle) window -> demo_limit=N yields exactly N fixed samples, which is
    # what the overfit sanity configs rely on.
    train_loader = _mk(train_demos, True, train_random_window)
    val_loader = _mk(val_demos, False, False)
    return train_loader, val_loader, stats


def make_loaders_multi(cfg: MFConfig, data_root: str, tasks: List[str],
                       batch_size: int, val_ratio: float, num_workers: int,
                       seed: int, depth_mean: float, depth_std: float,
                       depth_max_m: float = 10.0, depth_norm_mode: str = "log",
                       stats: Optional[MFStats] = None,
                       demo_limit: Optional[int] = None,
                       train_random_window: bool = True,
                       task_vocab: Optional[List[str]] = None,
                       task_aliases: Optional[Dict[str, str]] = None,
                       split_manifest: Optional[str] = None,
                       train_count: Optional[int] = None):
    """Single-stream loaders pooling demos across several tasks.

    Returns a single (train, val) pool combining all ``tasks`` -- the multi-task
    analogue of ``make_loaders``. Each task is split by ``val_ratio`` independently
    (balanced val) then round-robin interleaved across tasks; ``task_id`` is
    inferred per-demo, so the learned task embedding covers all tasks. Interleaving
    matters for the val loader (shuffle=False): round-robin makes any prefix span
    every task."""
    demos_by_task = discover_demos_multi(data_root, tasks)
    if demo_limit is not None:
        demos_by_task = {t: d[:demo_limit] for t, d in demos_by_task.items()}
    if split_manifest and train_count is not None:
        raise ValueError("split_manifest and train_count are mutually exclusive")
    if split_manifest:
        train_split, val_split = split_demos_manifest(
            demos_by_task, split_manifest)
        train_by_task = [train_split[t] for t in tasks]
        val_by_task = [val_split[t] for t in tasks]
    elif train_count is not None:
        if isinstance(train_count, Mapping):
            extra_tasks = sorted(set(train_count) - set(tasks))
            if extra_tasks:
                raise ValueError(
                    f"train_count mapping has unconfigured tasks: {extra_tasks}")
        train_by_task, val_by_task = [], []
        for task_index, task in enumerate(tasks):
            tr, va = split_demos_train_count(
                demos_by_task[task], _resolve_train_count(train_count, task),
                seed + task_index)
            train_by_task.append(tr)
            val_by_task.append(va)
    else:
        train_by_task, val_by_task = [], []
        for task in tasks:
            tr, va = split_demos(demos_by_task[task], val_ratio, seed)
            train_by_task.append(tr)
            val_by_task.append(va)
    train_demos = _round_robin(train_by_task)
    val_demos = _round_robin(val_by_task)
    if stats is None:
        stats = compute_stats(train_demos, cfg, seed=seed)
    # task_vocab (sorted global index space) may be a superset of `tasks` so the
    # learned embedding lines up across pretrain/finetune; falls back to `tasks`.
    task_to_id = build_source_task_to_id(
        list(tasks), task_vocab=task_vocab, task_aliases=task_aliases)

    def _mk(ds_demos, shuffle, random_window):
        if not ds_demos:
            return None
        ds = ModalityForcingDataset(
            ds_demos, cfg, stats, depth_mean, depth_std, depth_max_m,
            depth_norm_mode, random_window=random_window, seed=seed,
            task_to_id=task_to_id)
        return _build_loader(ds, batch_size, shuffle, num_workers,
                             persistent=(num_workers > 0 and shuffle))

    train_loader = _mk(train_demos, True, train_random_window)
    val_loader = _mk(val_demos, False, False)
    return train_loader, val_loader, stats


def build_modality_loaders(cfg: MFConfig, data_root: str, tasks: List[str],
                           batch_size: int, action_val_ratio: float,
                           dyn_val_ratio: float, num_workers: int, seed: int,
                           depth_mean: float, depth_std: float,
                           depth_max_m: float = 10.0, depth_norm_mode: str = "log",
                           stats: Optional[MFStats] = None,
                           demo_limit: Optional[int] = None,
                           task_vocab: Optional[List[str]] = None):
    """Multi-task loaders with nested per-modality budgets (two-stream training).

    Returns ({action_train, dyn_train, action_val, dyn_val} -> DataLoader|None,
    MFStats, pools). Stats are computed over ``dyn_train`` (the 90% pool, which by
    nesting contains every action-train demo)."""
    demos_by_task = discover_demos_multi(data_root, tasks)
    if demo_limit is not None:
        demos_by_task = {t: d[:demo_limit] for t, d in demos_by_task.items()}
    pools = split_demos_modality(demos_by_task, action_val_ratio, dyn_val_ratio, seed)
    if stats is None:
        stats = compute_stats(pools["dyn_train"], cfg, seed=seed)
    # task_vocab (sorted global index space) may be a superset of `tasks` so the
    # learned embedding lines up across pretrain/finetune; falls back to `tasks`.
    task_to_id = build_task_to_id(list(task_vocab) if task_vocab is not None else list(tasks))

    def _mk(ds_demos, shuffle, random_window, nworkers):
        if not ds_demos:
            return None
        ds = ModalityForcingDataset(
            ds_demos, cfg, stats, depth_mean, depth_std, depth_max_m,
            depth_norm_mode, random_window=random_window, seed=seed,
            task_to_id=task_to_id)
        return _build_loader(ds, batch_size, shuffle, nworkers,
                             persistent=(nworkers > 0 and shuffle))

    val_workers = min(2, num_workers)
    loaders = {
        "action_train": _mk(pools["action_train"], True, True, num_workers),
        "dyn_train": _mk(pools["dyn_train"], True, True, num_workers),
        "action_val": _mk(pools["action_val"], False, False, val_workers),
        "dyn_val": _mk(pools["dyn_val"], False, False, val_workers),
    }
    return loaders, stats, pools


def build_modality_loaders_cotrain(
        cfg: MFConfig, data_root: str, tasks: List[str], batch_size: int,
        n_action_train: int, n_action_val: int, n_dyn_val: int,
        n_dyn_train: Optional[int], num_workers: int, seed: int,
        depth_mean: float, depth_std: float,
        depth_max_m: float = 10.0, depth_norm_mode: str = "log",
        stats: Optional[MFStats] = None, demo_limit: Optional[int] = None,
        task_vocab: Optional[List[str]] = None, pool: Optional[str] = None,
        split_universe: Optional[int] = None):
    """Loaders over the count-based nested co-train split (split_demos_cotrain).

    Default (``pool=None``): the full two-stream set {action_train, dyn_train,
    action_val, dyn_val}; stats over dyn_train (the big pool, which by nesting
    contains every action-train demo). ``pool="action"``: only the action
    train/val loaders (single-stream baseline on the small pool, e.g. scratch
    action-only); stats over action_train so the baseline never touches the big
    pool. Returns ({name -> DataLoader|None}, MFStats, pools)."""
    demos_by_task = discover_demos_multi(data_root, tasks)
    if demo_limit is not None:
        demos_by_task = {t: d[:demo_limit] for t, d in demos_by_task.items()}
    pools = split_demos_cotrain(demos_by_task, n_action_train, n_action_val,
                                n_dyn_val, n_dyn_train, seed,
                                split_universe=split_universe)
    if stats is None:
        stats_pool = pools["action_train"] if pool == "action" else pools["dyn_train"]
        stats = compute_stats(stats_pool, cfg, seed=seed)
    task_to_id = build_task_to_id(
        list(task_vocab) if task_vocab is not None else list(tasks))

    def _mk(ds_demos, shuffle, random_window, nworkers):
        if not ds_demos:
            return None
        ds = ModalityForcingDataset(
            ds_demos, cfg, stats, depth_mean, depth_std, depth_max_m,
            depth_norm_mode, random_window=random_window, seed=seed,
            task_to_id=task_to_id)
        return _build_loader(ds, batch_size, shuffle, nworkers,
                             persistent=(nworkers > 0 and shuffle))

    val_workers = min(2, num_workers)
    loaders = {
        "action_train": _mk(pools["action_train"], True, True, num_workers),
        "action_val": _mk(pools["action_val"], False, False, val_workers),
        "dyn_train": None,
        "dyn_val": None,
    }
    if pool != "action":
        loaders["dyn_train"] = _mk(pools["dyn_train"], True, True, num_workers)
        loaders["dyn_val"] = _mk(pools["dyn_val"], False, False, val_workers)
    return loaders, stats, pools
