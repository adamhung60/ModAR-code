"""Training entry point for ModAR and the published comparison methods.

Usage:
    python scripts/train/train_modality_forcing.py \
        config=modar [key=value ...]
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from util.modality_forcing.config import MFConfig          # noqa: E402
from util.modality_forcing.data import (                    # noqa: E402
    build_task_to_id,
    canonical_task_vocab,
    make_loaders,
    make_loaders_multi,
)
from util.modality_forcing.model import (  # noqa: E402
    ModalityForcingWAM,
    backbone_reinit_ema_shadow,
    load_backbone_reinit_experts,
    load_spatially_compatible_state_dict,
    spatially_compatible_state_dict,
)

from util.modality_forcing.config_paths import (           # noqa: E402
    resolve_config_path, load_config as _load_config)
from util.modality_forcing.budget import (                 # noqa: E402
    describe_budget, resolve_sample_budgets)
from util.modality_forcing.grad_budget import (            # noqa: E402
    describe_loss_coeffs, resolve_loss_coeffs)

DEFAULT_CONFIG = "modar"

# Zero workers decode the batch inside the training step. Refuse that unless the
# config explicitly opts in, and cap a larger request to the cores this rank has.
MIN_NUM_WORKERS = 4


def resolve_num_workers(tcfg, local_world_size, log=print):
    """Loader workers for this rank: refuse a starved pipeline, cap oversubscription.

    Zero workers raise unless ``allow_starved_loader`` is set. A request larger
    than the cores available to this rank is capped.
    """
    want = int(tcfg.get("num_workers", MIN_NUM_WORKERS))
    if want < MIN_NUM_WORKERS:
        if not bool(tcfg.get("allow_starved_loader", False)):
            raise ValueError(
                f"train.num_workers={want} is below the floor of "
                f"{MIN_NUM_WORKERS}. Zero workers decode every sample inside the "
                f"training step. Set train.allow_starved_loader=true to allow it.")
        log(f"[loader] WARNING num_workers={want} with allow_starved_loader: "
            f"data loading runs serially with the step")
        return want
    # sched_getaffinity, NOT os.cpu_count(): under a Slurm cgroup (or any cpuset)
    # cpu_count reports the whole NODE while the job may hold a small slice of it.
    # A cgroup can expose 12 CPUs on a 96-core machine. cpu_count would see 96 and
    # fail to cap the workers.
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:                       # not Linux
        cores = os.cpu_count() or want
    # Leave this rank's own compute thread out of the worker budget. The step is
    # CPU-launch-bound, so starving the main process to feed workers is a net loss.
    per_rank = max(MIN_NUM_WORKERS, cores // max(1, local_world_size) - 1)
    got = min(want, per_rank)
    if got != want:
        log(f"[loader] num_workers {want} -> {got} "
            f"({cores} usable cores / {local_world_size} local ranks)")
    return got


def resolve_reinit_experts(tcfg):
    """Modalities whose expert pathways are reinitialized, and whether the trunk freezes.

    ``train.reinit_action: true`` means ``reinit_experts: [action]``.
    """
    swept = tcfg.get("reinit_experts", None)
    if swept is None:
        swept = ["action"] if bool(tcfg.get("reinit_action", False)) else []
    elif isinstance(swept, str):
        swept = [swept]
    swept = tuple(str(m) for m in swept)
    hard_freeze = float(tcfg.get("trunk_lr_scale", 0.0)) == 0.0
    return swept, hard_freeze


def lr_lambda(step, warmup, total, cosine=False):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if not cosine:
        # constant LR after warmup (default): makes resuming/extending a run to a
        # longer horizon clean -- the LR doesn't depend on max_steps, so a warm
        # restart continues at the same rate instead of jumping around the cosine
        # tail. Set train.lr_cosine=true to restore cosine-to-zero decay.
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def n_demos_of(loader):
    """Distinct demos behind a loader, ignoring any epoch stretching."""
    if loader is None:
        return 0
    ds = loader.dataset
    return int(getattr(ds, "n_demos", len(ds)))


def cycle(loader):
    """Infinite iterator over a loader. If it carries a DistributedSampler, bump
    its epoch each pass so every rank reshuffles its shard (deterministically) run
    to run rather than repeating the same order forever.
    """
    epoch = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def fixed_metric_loader(src_loader, n_samples, batch_size, seed):
    """Deterministic middle-window cohort for comparable periodic metrics.

    ``n_samples <= 0`` means every unique demo.  Train datasets may have a large
    ``epoch_repeat`` for input-pipeline efficiency; metrics intentionally ignore
    it.  A fixed demo subset and ``random_window=False`` make every checkpoint
    and every method see the same examples without keeping materialized tensors
    resident for the whole training run.
    """
    if src_loader is None:
        return None
    from torch.utils.data import DataLoader, Subset

    ds = copy.copy(src_loader.dataset)
    ds.epoch_repeat = 1
    ds.random_window = False
    n_unique = int(getattr(ds, "n_demos", len(ds)))
    if n_unique <= 0:
        return None
    requested = int(n_samples)
    if requested <= 0 or requested >= n_unique:
        indices = list(range(n_unique))
    else:
        generator = torch.Generator().manual_seed(int(seed))
        indices = torch.randperm(n_unique, generator=generator)[:requested].tolist()
    subset = Subset(ds, indices)
    return DataLoader(
        subset, batch_size=min(int(batch_size), len(subset)), shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)


def setup_distributed(timeout_minutes=120):
    """Init the process group from torchrun's env vars. Returns
    (rank, world_size, local_rank, is_distributed). A plain ``python ...`` launch
    (no RANK/WORLD_SIZE) stays single-process."""
    timeout_minutes = int(timeout_minutes)
    if timeout_minutes <= 0:
        raise ValueError(
            f"train.ddp_timeout_minutes must be positive, got {timeout_minutes}")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl",
                                device_id=torch.device(f"cuda:{local_rank}"),
                                timeout=timedelta(minutes=timeout_minutes))
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def normalize_source_weights(sources):
    """Return positive per-source weights normalized to sum to one."""
    weights = [float(source["weight"]) for source in sources]
    if any(weight <= 0 for weight in weights):
        raise ValueError("every data source weight must be > 0")
    total = sum(weights)
    return [weight / total for weight in weights]


def select_action_source(sources):
    """Select the source whose normalization defines physical action units."""
    explicit = next((
        source for source in sources if source["stream"] == "action"), None)
    if explicit is not None:
        return explicit
    return next((
        source for source in sources if source["stream"] is None), None)


def expand_source_configs(data_cfg):
    """Overlay each optional data.sources entry on shared data defaults."""
    raw_sources = data_cfg.get("sources", None)
    if not raw_sources:
        return []
    defaults = OmegaConf.to_container(data_cfg, resolve=True)
    defaults.pop("sources", None)
    return [
        OmegaConf.merge(
            defaults, OmegaConf.to_container(source, resolve=True))
        for source in raw_sources
    ]


def weighted_source_loss(step_outputs):
    """Weighted-mean loss from ``(source, output, normalized_weight)`` tuples."""
    return sum(
        weight * source_out["loss"]
        for _, source_out, weight in step_outputs)


def pack_stats(stats):
    return {
        "action_mean": stats.action_mean, "action_std": stats.action_std,
        "proprio_mean": stats.proprio_mean, "proprio_std": stats.proprio_std,
        "dino_mean": stats.dino_mean, "dino_std": stats.dino_std,
    }


def n_eval_batches(requested, loader):
    """How many loader batches to score.

    ``requested <= 0`` means one pass over unique demos, not the
    ``epoch_repeat``-stretched train epoch used to keep the training iterator
    warm. Val loaders already have ``epoch_repeat == 1``, so this is the full
    val split. Train integ/eval at a 10k-step cadence must not walk thousands
    of random windows.
    """
    if loader is None:
        return 0
    ds = getattr(loader, "dataset", None)
    n_demos = getattr(ds, "n_demos", None)
    batch_size = max(1, int(getattr(loader, "batch_size", 1) or 1))
    unique = (max(1, -(-int(n_demos) // batch_size))
              if n_demos else max(1, len(loader)))
    if int(requested) <= 0:
        return unique
    return min(int(requested), max(1, len(loader)))


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches, stream=None,
             loss_coeffs=None):
    if val_loader is None:
        return {}
    model.eval()
    agg = {"loss": 0.0}
    n = 0
    mod_agg = {f"loss_{m}": 0.0 for m in model.cfg.modalities}
    mod_wsum = {f"loss_{m}": 0.0 for m in model.cfg.modalities}
    act_w = 0.0
    act_agg = {"mm_error_onestep": 0.0, "deg_error_onestep": 0.0}
    src = {"mm_error_onestep": "act_mm", "deg_error_onestep": "act_deg"}
    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break
        out = model(**to_device(batch, device), stream=stream,
                    loss_coeffs=loss_coeffs)
        batch_n = int(batch["dino"].shape[0])
        agg["loss"] += float(out["loss"]) * batch_n
        for m in model.cfg.modalities:
            lk, nk = f"loss_{m}", f"n_{m}"
            w = float(out[nk])
            if w > 0:
                mod_agg[lk] += float(out[lk]) * w
                mod_wsum[lk] += w
        w = float(out["n_action"])
        act_w += w
        for k in act_agg:
            act_agg[k] += float(out[src[k]]) * w
        n += batch_n
    model.train()
    metrics = {f"val/loss": agg["loss"] / max(1, n)}
    for lk, v in mod_agg.items():
        if mod_wsum[lk] > 0:
            metrics[f"val/{lk}"] = v / mod_wsum[lk]
    if act_w > 0:
        for k, v in act_agg.items():
            metrics[f"val/{k}"] = v / act_w
    return metrics


@torch.no_grad()
def evaluate_sources(model, sources, device, max_batches):
    """Validate every source and return namespaced plus weighted aggregate metrics."""
    source_weights = normalize_source_weights(sources)
    per_source = []
    metrics = {}
    for source, weight in zip(sources, source_weights):
        result = evaluate(
            model, source["val_loader"], device, max_batches, source["stream"],
            loss_coeffs=source.get("loss_coeffs"))
        if not result:
            continue
        per_source.append((result, weight))
        for key, value in result.items():
            suffix = key.removeprefix("val/")
            metrics[f"val/{source['name']}/{suffix}"] = value

    aggregate_keys = set().union(*(result for result, _ in per_source))
    for key in aggregate_keys:
        present = [(result[key], weight) for result, weight in per_source
                   if key in result]
        denom = sum(weight for _, weight in present)
        metrics[key] = sum(value * weight for value, weight in present) / denom
    return metrics


@torch.no_grad()
def evaluate_integrated(model, batches, device, steps_per_phase=None,
                        noise_seed=None):
    """Full ODE-integrated action error (mm/deg) vs GT -- the deployment-relevant
    signal, unlike the one-step x-pred error. ``batches`` is a fixed metric
    cohort shared with generation scoring; ragged batches are sample-weighted
    and ``noise_seed`` supplies common random numbers across checkpoints.

    """
    if not batches:
        return {}
    was_training = model.training
    model.eval()
    nhist = getattr(model, "n_hist_eff", model.cfg.obs_history)
    saved_spp = model.cfg.steps_per_phase
    cpu_rng = torch.get_rng_state() if noise_seed is not None else None
    cuda_rng = (torch.cuda.get_rng_state_all()
                if noise_seed is not None and torch.cuda.is_available() else None)
    if steps_per_phase is not None:
        model.cfg.steps_per_phase = int(steps_per_phase)
    totals = {"mm": 0.0, "deg": 0.0, "joint": 0.0, "grip": 0.0}
    weights = {"physical": 0, "joint": 0}
    try:
        for index, batch in enumerate(batches):
            if noise_seed is not None:
                torch.manual_seed(int(noise_seed) + index)
            b = to_device(batch, device)
            B = b["dino"].shape[0]
            ones = torch.ones(B, device=device)
            img_hist = b["images"][:, :nhist] if "images" in b else None
            samp = model.sample(
                b["dino"][:, :nhist], b["depth_maps"][:, :nhist],
                b["proprio"], b.get("task_id"), image_hist=img_hist)
            m, d = model._action_phys_errors(
                samp["actions"], b["actions"], ones)
            totals["mm"] += float(m) * B
            totals["deg"] += float(d) * B
            weights["physical"] += B
            j, g = model._action_joint_errors(
                samp["actions"], b["actions"], ones)
            if j is not None:
                totals["joint"] += float(j) * B
                totals["grip"] += float(g) * B
                weights["joint"] += B
    finally:
        model.cfg.steps_per_phase = saved_spp
        if was_training:
            model.train()
        if cpu_rng is not None:
            torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    metrics = {
        "mm_error_integ": totals["mm"] / weights["physical"],
        "deg_error_integ": totals["deg"] / weights["physical"],
    }
    if weights["joint"]:
        metrics["joint_mae_deg_integ"] = totals["joint"] / weights["joint"]
        metrics["gripper_mae_integ"] = totals["grip"] / weights["joint"]
    return metrics


def save_ckpt_atomic(ckpt: dict, path: Path):
    """Write a checkpoint atomically so readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


# Model fields that change what the weights MEAN without changing any parameter
# shape, so the strict state-dict load cannot catch a mismatch. n_heads/rope_split
# re-partition attention and re-assign RoPE frequency bands; the clock fields move
# every token's RoPE position.
_ARCH_CRITICAL_FIELDS = (
    "n_heads", "rope_split", "obs_history", "obs_future", "obs_stride",
    "action_horizon", "track_pred_mode")


def _as_list(value):
    """Normalize YAML sequences so [16,16,16] and (16,16,16) compare equal."""
    return list(value) if isinstance(value, (list, tuple)) else value


def check_resume_arch(ckpt_cfg, cfg, allow: bool = False):
    """Refuse to warm-start across a silent architecture change."""
    if not ckpt_cfg:
        return
    old = dict(ckpt_cfg.get("model", {}))
    new = OmegaConf.to_container(cfg.model, resolve=True)
    diffs = [(k, old[k], new.get(k)) for k in _ARCH_CRITICAL_FIELDS
             if k in old and _as_list(old[k]) != _as_list(new.get(k))]
    if not diffs:
        return
    detail = "; ".join(f"{k}: checkpoint={o!r} config={n!r}" for k, o, n in diffs)
    if allow:
        print(f"[resume] WARNING: architecture changed ({detail}); "
              "loading anyway per train.allow_arch_change", flush=True)
        return
    raise RuntimeError(
        f"resume checkpoint was trained with a different architecture ({detail}). "
        "Parameter shapes still match, so this would load silently and corrupt the "
        "run. Pin the old values in the run config, or set "
        "train.allow_arch_change=true if the reinterpretation is intended.")


def load_config(config_path, cli):
    """Load a variant config, merge over its ``base:`` chain, then apply CLI overrides."""
    return _load_config(config_path, cli)


def main():
    cli = OmegaConf.from_cli()
    config_path = cli.pop("config", DEFAULT_CONFIG)
    cfg = load_config(config_path, cli)

    tcfg = cfg.train
    # ---- distributed (DDP) init: no-op unless launched via torchrun ----------
    rank, world_size, local_rank, distributed = setup_distributed(
        tcfg.get("ddp_timeout_minutes", 120))
    is_main = rank == 0

    def log0(*a, **k):
        if is_main:
            print(*a, **k)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else tcfg.device)
    else:
        device = torch.device("cpu")
    torch.manual_seed(tcfg.seed)
    if distributed:
        log0(f"[ddp] world_size={world_size} | per-rank batch={tcfg.batch_size} | "
             f"effective global batch={tcfg.batch_size * world_size}")

    # Turn any sample-denominated budget into steps now that the global batch is
    # known, so the same config gets the same data budget, LR warmup and eval
    # grid on any batch/GPU layout. See util/modality_forcing/budget.py.
    # Resolve loader workers before any loader is built, so every source below
    # (and the val/eval loaders) sees one vetted number.
    tcfg.num_workers = resolve_num_workers(
        tcfg, int(os.environ.get("LOCAL_WORLD_SIZE", 1)), log0)
    log0(f"[loader] num_workers={tcfg.num_workers} per rank")

    global_batch = int(tcfg.batch_size) * world_size
    derived_budget = resolve_sample_budgets(tcfg, global_batch)
    for key, value in derived_budget.items():
        tcfg[key] = value
    log0("[budget] " + describe_budget(derived_budget, global_batch))

    mfcfg = MFConfig.from_dict(OmegaConf.to_container(cfg.model, resolve=True))
    dcfg = cfg.data
    demo_limit = dcfg.get("demo_limit", None)
    val_ratio = float(dcfg.get("val_ratio", 0.1))
    source_cfgs = expand_source_configs(dcfg)

    # Multi-task: one train/val pool per task (val_ratio). Single-task: data.task.
    tasks = dcfg.get("tasks", None)
    source_tasks = []
    source_canonical_sets = []
    for source_cfg in source_cfgs:
        configured = source_cfg.get("tasks", None)
        configured_list = []
        if configured is not None:
            configured_list = list(configured)
        elif source_cfg.get("task", None) is not None:
            configured_list = [str(source_cfg.task)]
        aliases = dict(source_cfg.get("task_aliases", {}) or {})
        unknown_aliases = sorted(set(aliases) - set(configured_list))
        if unknown_aliases:
            raise ValueError(
                f"source {source_cfg.name} task_aliases contains unconfigured "
                f"tasks: {unknown_aliases}")
        canonical = [aliases.get(task, task) for task in configured_list]
        if len(set(canonical)) != len(canonical):
            raise ValueError(
                f"source {source_cfg.name} maps multiple tasks to one canonical "
                f"identity: {canonical}")
        source_tasks.extend(canonical)
        source_canonical_sets.append((str(source_cfg.name), set(canonical)))
    # Optional global task vocabulary. It decouples the task-embedding index space
    # from the set of tasks actually trained on: pretrain and finetune share the
    # same sorted vocab (the 65 pretraining tasks), so each target task keeps its
    # pretrained embedding row and the checkpoint loads strictly even though
    # finetune only trains on the 6 target tasks. Sorted -> stable, order-independent.
    task_vocab = dcfg.get("task_vocab", None)
    task_vocab = (
        canonical_task_vocab(list(task_vocab))
        if task_vocab
        else (
            canonical_task_vocab(list(set(source_tasks)))
            if source_cfgs
            else None
        )
    )
    if task_vocab is not None:
        # Checkpoints must serialize the exact embedding-row order used by the
        # loaders, not the pre-resolution YAML order.
        cfg.data.task_vocab = list(task_vocab)
        configured_tasks = source_tasks if source_cfgs else (
            list(tasks) if tasks is not None else [])
        unknown_tasks = sorted(set(configured_tasks) - set(task_vocab))
        if unknown_tasks:
            raise ValueError(
                f"canonical data source tasks absent from data.task_vocab: "
                f"{unknown_tasks}")
        mfcfg.n_tasks = len(task_vocab)
    else:
        mfcfg.n_tasks = len(list(tasks)) if tasks is not None else 1
    cfg.model.n_tasks = mfcfg.n_tasks
    if bool(dcfg.get("require_matching_task_sets", False)) and source_canonical_sets:
        expected_name, expected_tasks = source_canonical_sets[0]
        for source_name, canonical_tasks in source_canonical_sets[1:]:
            if canonical_tasks != expected_tasks:
                raise ValueError(
                    "data sources must expose matching canonical task sets: "
                    f"{expected_name}={sorted(expected_tasks)}, "
                    f"{source_name}={sorted(canonical_tasks)}")
    log0("MFConfig:", mfcfg)

    # Legacy two-stream co-training is adapted below into the same source list as
    # data.sources. Every optimizer step takes one forward per source and combines
    # them as a normalized weighted mean. Two legacy split flavours produce the
    # {action,dyn}x{train,val} pools:
    #   * count-based nested (data.n_action_train/n_action_val/n_dyn_val
    #     [/n_dyn_train]) -- exact pool sizes via split_demos_cotrain.
    #     data.split_pool=action trains single-
    #     stream on just the small action pool (scratch action-only baseline).
    #   * ratio-based (data.action_val_ratio/dyn_val_ratio) -- legacy percentage
    #     split (split_demos_modality); still supported for older run configs.
    # Otherwise the single-stream unified-pool path below is used.
    cotrain = tasks is not None and dcfg.get("n_action_train", None) is not None
    cotrain_single = cotrain and dcfg.get("split_pool", None) == "action"
    ratio_two_stream = (tasks is not None
                        and dcfg.get("action_val_ratio", None) is not None)
    two_stream = ratio_two_stream or (cotrain and not cotrain_single)
    sources = []
    train_loader = val_loader = None
    action_train_loader = dyn_train_loader = None
    action_val_loader = dyn_val_loader = None
    if source_cfgs:
        names = [str(source_cfg.name) for source_cfg in source_cfgs]
        if len(names) != len(set(names)):
            raise ValueError("data.sources names must be unique")
        for index, source_cfg in enumerate(source_cfgs):
            configured_stream = source_cfg.get("stream", "joint")
            if configured_stream == "joint":
                stream = None
            elif configured_stream in ("action", "dynamics"):
                stream = str(configured_stream)
            else:
                raise ValueError(
                    f"data.sources[{index}].stream must be action, dynamics, "
                    "or joint")
            source_seed = int(source_cfg.get("seed", tcfg.seed))
            source_batch_size = int(source_cfg.get("batch_size", tcfg.batch_size))
            source_workers = int(source_cfg.get("num_workers", tcfg.num_workers))
            source_demo_limit = source_cfg.get("demo_limit", demo_limit)
            source_val_ratio = float(source_cfg.get("val_ratio", val_ratio))
            configured = source_cfg.get("tasks", None)
            if configured is not None:
                source_train, source_val, source_stats = make_loaders_multi(
                    mfcfg, source_cfg.data_root, list(configured),
                    source_batch_size, source_val_ratio, source_workers,
                    source_seed, source_cfg.depth_mean,
                    source_cfg.depth_std, source_cfg.depth_max_m,
                    source_cfg.depth_norm_mode, demo_limit=source_demo_limit,
                    train_random_window=source_cfg.get("random_window", True),
                    task_vocab=task_vocab,
                    task_aliases=source_cfg.get("task_aliases", None),
                    split_manifest=source_cfg.get("split_manifest", None),
                    train_count=source_cfg.get("train_count", None))
            else:
                source_train, source_val, source_stats = make_loaders(
                    mfcfg, source_cfg.data_root, source_cfg.task,
                    source_batch_size, source_val_ratio, source_workers,
                    source_seed, source_cfg.depth_mean,
                    source_cfg.depth_std, source_cfg.depth_max_m,
                    source_cfg.depth_norm_mode, demo_limit=source_demo_limit,
                    train_random_window=source_cfg.get("random_window", True),
                    task_vocab=task_vocab,
                    task_aliases=source_cfg.get("task_aliases", None),
                    split_manifest=source_cfg.get("split_manifest", None),
                    train_count=source_cfg.get("train_count", None))
            source = {
                "name": str(source_cfg.name),
                "stream": stream,
                "weight": float(source_cfg.get("weight", 1.0)),
                "batch_size": source_batch_size,
                "train_loader": source_train,
                "val_loader": source_val,
                "stats": source_stats,
                "data_cfg": source_cfg,
            }
            sources.append(source)
            log0(f"source {source['name']}: stream={stream} "
                 f"train={len(source_train.dataset)} "
                 f"val={len(source_val.dataset) if source_val else 0} "
                 f"weight={source['weight']}")
        action_sources = [source for source in sources
                          if source["stream"] == "action"]
        if len(action_sources) > 1:
            raise ValueError(
                "data.sources currently supports at most one action source")
        two_stream = len(sources) > 1
        action_stats_source = select_action_source(sources)
        stats = (
            action_stats_source["stats"]
            if action_stats_source is not None else sources[0]["stats"])
    elif cotrain:
        from util.modality_forcing.data import build_modality_loaders_cotrain
        n_dyn_train = dcfg.get("n_dyn_train", None)
        loaders, stats, pools = build_modality_loaders_cotrain(
            mfcfg, dcfg.data_root, list(tasks), tcfg.batch_size,
            int(dcfg.n_action_train), int(dcfg.n_action_val), int(dcfg.n_dyn_val),
            None if n_dyn_train is None else int(n_dyn_train),
            tcfg.num_workers, tcfg.seed, dcfg.depth_mean, dcfg.depth_std,
            dcfg.depth_max_m, dcfg.depth_norm_mode, demo_limit=demo_limit,
            task_vocab=task_vocab, pool=("action" if cotrain_single else None),
            split_universe=dcfg.get("split_universe", None))
        if cotrain_single:
            train_loader = loaders["action_train"]
            val_loader = loaders["action_val"]
            log0(f"tasks: {list(tasks)} | ACTION POOL (count-based co-train split)")
            log0(f"train demos: {len(pools['action_train'])} | "
                 f"val demos: {len(pools['action_val'])}")
        else:
            action_train_loader = loaders["action_train"]
            dyn_train_loader = loaders["dyn_train"]
            action_val_loader = loaders["action_val"]
            dyn_val_loader = loaders["dyn_val"]
            log0(f"tasks: {list(tasks)} | TWO-STREAM (count-based co-train split)")
            log0(f"action_train: {len(pools['action_train'])} | "
                 f"dyn_train: {len(pools['dyn_train'])} | "
                 f"action_val: {len(pools['action_val'])} | "
                 f"dyn_val: {len(pools['dyn_val'])}")
            sources = [
                {"name": "action", "stream": "action", "weight": 1.0,
                 "batch_size": int(tcfg.batch_size),
                 "train_loader": action_train_loader,
                 "val_loader": action_val_loader, "stats": stats,
                 "data_cfg": dcfg},
                {"name": "dynamics", "stream": "dynamics", "weight": 1.0,
                 "batch_size": int(tcfg.batch_size),
                 "train_loader": dyn_train_loader,
                 "val_loader": dyn_val_loader, "stats": stats,
                 "data_cfg": dcfg},
            ]
    elif ratio_two_stream:
        from util.modality_forcing.data import build_modality_loaders
        loaders, stats, pools = build_modality_loaders(
            mfcfg, dcfg.data_root, list(tasks), tcfg.batch_size,
            float(dcfg.action_val_ratio), float(dcfg.dyn_val_ratio),
            tcfg.num_workers, tcfg.seed, dcfg.depth_mean, dcfg.depth_std,
            dcfg.depth_max_m, dcfg.depth_norm_mode, demo_limit=demo_limit,
            task_vocab=task_vocab)
        action_train_loader = loaders["action_train"]
        dyn_train_loader = loaders["dyn_train"]
        action_val_loader = loaders["action_val"]
        dyn_val_loader = loaders["dyn_val"]
        log0(f"tasks: {list(tasks)} | TWO-STREAM (ratio-based split)")
        log0(f"action_train: {len(pools['action_train'])} | "
             f"dyn_train: {len(pools['dyn_train'])} | "
             f"action_val: {len(pools['action_val'])} | "
             f"dyn_val: {len(pools['dyn_val'])}")
        sources = [
            {"name": "action", "stream": "action", "weight": 1.0,
             "batch_size": int(tcfg.batch_size),
             "train_loader": action_train_loader,
             "val_loader": action_val_loader, "stats": stats, "data_cfg": dcfg},
            {"name": "dynamics", "stream": "dynamics", "weight": 1.0,
             "batch_size": int(tcfg.batch_size),
             "train_loader": dyn_train_loader,
             "val_loader": dyn_val_loader, "stats": stats, "data_cfg": dcfg},
        ]
    elif tasks is not None:
        train_loader, val_loader, stats = make_loaders_multi(
            mfcfg, dcfg.data_root, list(tasks), tcfg.batch_size, val_ratio,
            tcfg.num_workers, tcfg.seed, dcfg.depth_mean, dcfg.depth_std,
            dcfg.depth_max_m, dcfg.depth_norm_mode, demo_limit=demo_limit,
            train_random_window=dcfg.get("random_window", True),
            task_vocab=task_vocab)
        log0(f"tasks: {list(tasks)}")
        log0(f"train demos: {n_demos_of(train_loader)} | "
             f"val demos: {n_demos_of(val_loader)}")
        sources = [
            {"name": str(dcfg.get("name", "main")), "stream": None, "weight": 1.0,
             "batch_size": int(tcfg.batch_size), "train_loader": train_loader,
             "val_loader": val_loader, "stats": stats, "data_cfg": dcfg},
        ]
    else:
        train_loader, val_loader, stats = make_loaders(
            mfcfg, dcfg.data_root, dcfg.task, tcfg.batch_size, val_ratio,
            tcfg.num_workers, tcfg.seed, dcfg.depth_mean, dcfg.depth_std,
            dcfg.depth_max_m, dcfg.depth_norm_mode, demo_limit=demo_limit,
            train_random_window=dcfg.get("random_window", True))
        log0(f"train demos: {n_demos_of(train_loader)} | "
             f"val demos: {n_demos_of(val_loader)}")
        sources = [
            {"name": str(dcfg.get("name", "main")), "stream": None, "weight": 1.0,
             "batch_size": int(tcfg.batch_size), "train_loader": train_loader,
             "val_loader": val_loader, "stats": stats, "data_cfg": dcfg},
        ]
    if cotrain_single:
        sources = [
            {"name": str(dcfg.get("name", "main")), "stream": None, "weight": 1.0,
             "batch_size": int(tcfg.batch_size), "train_loader": train_loader,
             "val_loader": val_loader, "stats": stats, "data_cfg": dcfg},
        ]
    # Hard-frozen backbone (train.reinit_experts with trunk_lr_scale == 0): every
    # parameter outside the swept pathways is frozen, so a source whose stream
    # touches none of them produces a loss with no grad_fn and breaks backward.
    # Drop those sources. Under a low-LR trunk finetune nothing is frozen, so
    # every source still contributes and all of them are kept. Self-forcing does
    # NOT freeze the trunk either (it self-forces both streams with the whole
    # model adapting), so it never drops a source here.
    _swept, _hard_freeze = resolve_reinit_experts(tcfg)
    if _swept and _hard_freeze:
        keep_streams = {None}
        if "action" in _swept:
            keep_streams.add("action")
        if set(_swept) - {"action"}:
            keep_streams.add("dynamics")
        kept = [s for s in sources if s["stream"] in keep_streams]
        if not kept:
            raise ValueError(
                f"frozen-backbone finetune of {list(_swept)} requires a source "
                f"on one of the streams {sorted(map(str, keep_streams))}")
        dropped = [s["name"] for s in sources if s["stream"] not in keep_streams]
        sources = kept
        if dropped:
            log0(f"[reinit-experts] frozen backbone; dropped sources {dropped} "
                 "(no trainable parameters on their stream)")

    source_weights = normalize_source_weights(sources)
    for source, weight in zip(sources, source_weights):
        source["normalized_weight"] = weight
        source_stats = source["stats"]
        log0(f"{source['name']} dino stats: mean={source_stats.dino_mean:.4f} "
             f"std={source_stats.dino_std:.4f}")

    # Every schedule mode aggregated its per-modality losses differently and
    # supervises a different set per stream, so the weight action carried in the
    # objective used to depend on the method. Resolve an explicit
    # per-source coefficient instead, pinning action at action_grad_share and
    # splitting the rest evenly across dynamics modalities.
    # Only the supervised modalities enter the budget. A conditioning-only sensor
    # never produces a loss term, so counting it would divide the dynamics half
    # across modalities that contribute nothing and leave action above its target
    # (0.8 rather than 0.5 for a one-of-four single-target arm).
    generated = set(mfcfg.generated_modality_names())
    budget_modalities = tuple(
        name for name in mfcfg.modalities
        if name == "action" or name in generated)
    loss_coeffs = resolve_loss_coeffs(
        mfcfg.schedule_mode, budget_modalities, sources,
        action_share=float(tcfg.get("action_grad_share", 0.5)), p_of=mfcfg.p_of)
    for source in sources:
        source["loss_coeffs"] = loss_coeffs[source["name"]]
        if not source["loss_coeffs"]:
            raise ValueError(
                f"source {source['name']!r} would contribute no gradient: the "
                f"{mfcfg.schedule_mode}/{source['stream']} supervision table is "
                "empty")
    log0("[grad-budget] " + describe_loss_coeffs(
        mfcfg.schedule_mode, budget_modalities, sources, loss_coeffs,
        p_of=mfcfg.p_of))

    # raw_model is the underlying module used for state_dict/EMA/sampling/eval and
    # all `.cfg` access; `model` is the (optionally DDP-wrapped, optionally
    # compiled) callable used for the training forward/backward.
    raw_model = ModalityForcingWAM(mfcfg).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    log0(f"model params: {n_params/1e6:.2f}M")
    # Warm-started expert finetune: with trunk_lr_scale == 0 freeze everything
    # outside the swept pathways; otherwise leave the whole model trainable and
    # put the trunk on a scaled LR (param groups below). Self-forcing never
    # freezes the trunk -- it self-forces both streams with the whole model
    # trainable.
    reinit_experts, hard_freeze = resolve_reinit_experts(tcfg)
    if reinit_experts and hard_freeze:
        n_train = raw_model.freeze_to_experts_only(reinit_experts)
        log0(f"[reinit-experts] frozen-backbone finetune of "
             f"{list(reinit_experts)}: {n_train/1e6:.2f}M trainable "
             f"of {n_params/1e6:.2f}M")
    model = raw_model

    # Optional linear LR scaling, keyed to the effective global batch rather than
    # the GPU count: 2 ranks x 24 and 1 rank x 48 are the same optimization
    # problem and must get the same LR, which a world_size-based rule gets wrong.
    # Off unless train.lr_reference_batch names the batch train.lr was tuned at.
    lr_reference_batch = int(tcfg.get("lr_reference_batch", 0) or 0)
    if lr_reference_batch:
        tcfg.lr = float(tcfg.lr) * global_batch / lr_reference_batch
        log0(f"[lr] scaled to {tcfg.lr:.2e} for global batch {global_batch} "
             f"(tuned at {lr_reference_batch})")

    trunk_lr_scale = float(tcfg.get("trunk_lr_scale", 0.0))
    if reinit_experts and not hard_freeze and trunk_lr_scale != 1.0:
        opt_params = raw_model.expert_pathway_param_groups(
            reinit_experts, float(tcfg.lr), trunk_lr_scale)
        log0(f"[reinit-experts] low-LR trunk finetune: {list(reinit_experts)} "
             f"at lr={float(tcfg.lr):.2e}, trunk at "
             f"{trunk_lr_scale:g}x = {float(tcfg.lr) * trunk_lr_scale:.2e}")
    else:
        opt_params = [p for p in raw_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        opt_params, lr=tcfg.lr, betas=tuple(tcfg.betas),
        weight_decay=tcfg.weight_decay)
    lr_cosine = bool(tcfg.get("lr_cosine", False))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, tcfg.warmup_steps, tcfg.max_steps, lr_cosine))
    ema = EMA(raw_model, tcfg.ema_decay) if (tcfg.ema and is_main) else None

    # ---- optional warm-restart / resume ----------------------------------
    # train.resume = "auto" (use <save_dir>/last.pt if present, else start fresh)
    # or an explicit ckpt path (must exist -- fails loudly if missing).
    # Loads model + EMA weights and the step counter, then re-anchors the LR
    # schedule at that step against the (possibly extended) max_steps. The
    # optimizer state is NOT restored (checkpoints don't carry it), so this is a
    # warm restart -- fine for extending a converged run to a longer horizon.
    start_step = 0
    resume = tcfg.get("resume", None)
    is_auto = str(resume) == "auto"
    resume_path = (Path(tcfg.save_dir) / "last.pt") if is_auto else (Path(resume) if resume else None)
    # "auto" is tolerant: a fresh run (no checkpoint yet) simply starts from
    # scratch, so the same requeued job works on window 1 and warm-restarts after.
    if resume and is_auto and not resume_path.exists():
        log0(f"[resume] auto: no checkpoint at {resume_path}; starting fresh")
        resume = None
    if resume:
        rck = torch.load(resume_path, map_location=device, weights_only=False)
        check_resume_arch(rck.get("cfg"), cfg, allow=tcfg.get("allow_arch_change", False))
        if reinit_experts:
            # Warm-started expert finetune: transplant the backbone verbatim and
            # keep the swept pathways at their fresh init, so a differently
            # shaped expert trains from scratch on a pretrained trunk.
            source_modalities = (
                (rck.get("cfg") or {}).get("model", {}).get("modalities")
            )
            skipped = load_backbone_reinit_experts(
                raw_model, rck["model"], reinit_experts,
                source_modalities=source_modalities)
            log0(f"[reinit-experts] loaded backbone; reinitialized "
                 f"{len(skipped)} tensors on {list(reinit_experts)}")
            if ema is not None and rck.get("ema") is not None:
                ema.shadow = backbone_reinit_ema_shadow(
                    raw_model, rck["ema"], reinit_experts,
                    source_modalities=source_modalities)
        else:
            load_spatially_compatible_state_dict(raw_model, rck["model"])
            if ema is not None and rck.get("ema") is not None:
                ema.shadow = copy.deepcopy(
                    spatially_compatible_state_dict(raw_model, rck["ema"]))
        # reset_step: load weights but start a FRESH step counter + LR schedule from 0
        # (a distinct fine-tune stage on top of a converged base),
        # rather than a warm restart that extends the base run's horizon (which would
        # re-anchor the LR deep in the cosine tail). save_dir must differ from the
        # base so checkpoints don't collide.
        reset_step = bool(tcfg.get("reset_step", False))
        # Steps are not comparable across layouts, samples are. When the GPU count
        # or batch size changes, re-anchor progress on samples_seen.
        ckpt_batch = int(rck.get("global_batch", 0) or 0)
        ckpt_samples = rck.get("samples_seen", None)
        if reset_step:
            start_step = 0
        elif ckpt_samples is not None and ckpt_batch != global_batch:
            start_step = int(ckpt_samples) // global_batch
            log0(f"[resume] layout changed (global batch {ckpt_batch} -> "
                 f"{global_batch}); re-anchored {int(ckpt_samples)} samples to "
                 f"step {start_step}")
        else:
            start_step = int(rck.get("step", 0))
        for _ in range(start_step):
            sched.step()
        log0(f"[resume] loaded {resume_path} @ base step {int(rck.get('step', 0))}; "
             f"{'FRESH LR schedule from 0' if reset_step else 'warm restart'}; "
             f"training to {tcfg.max_steps} (lr now {sched.get_last_lr()[0]:.2e})")
    # Dataset normalization, not the resumed checkpoint, defines the units for
    # action metrics and the top-level checkpoint stats in this training stage.
    raw_model.set_action_stats(stats.action_mean, stats.action_std)

    # ---- optional torch.compile of the DiT trunk -----------------------------
    # The trunk (attention + MLP + adaLN + RoPE) has a fixed structure and no
    # python-level graph breaks, so compiling JUST it fuses the many small
    # elementwise kernels for a ~1.5x step speedup. Compiling the whole model is
    # not worth it: the schedule_mode assembly (dict/slice/clone) graph-breaks and
    # recompiles on the varying reduced-sequence shapes. A few early recompiles are
    # expected as distinct modality combinations appear.
    if bool(tcfg.get("compile", False)):
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
        # Inductor compiles on the FIRST CALL, not here, so a compute node with a
        # broken compiler raises during the first compiled step. Fall back to eager
        # execution so training can continue.
        _dynamo.config.suppress_errors = True
        # Compile the forward METHOD in place (not the module) so state_dict keys
        # stay clean -- torch.compile(module) would wrap it in an OptimizedModule and
        # rename params to `dit._orig_mod.*`, breaking EMA and checkpoint loading.
        #
        # dynamic=False suits this model: the schedule reveals a fixed set of
        # modality combinations, so sequence lengths take a handful of discrete
        # values that each want their own static kernels, rather than a continuum
        # that would justify paying for dynamic shapes. The cache limit above is
        # what keeps that handful from evicting each other.
        raw_model.dit.forward = torch.compile(raw_model.dit.forward, dynamic=False)
        log0("[compile] torch.compile(model.dit.forward) enabled "
             "(suppress_errors: falls back to eager rather than failing the run)")

    use_wandb = bool(tcfg.get("wandb", False)) and is_main
    if use_wandb:
        import wandb
        # Stable, deterministic run id so resuming continues the SAME W&B run (no new
        # line/duplicate). Defaults to a slug of the run name (falling back to the
        # save_dir basename); override explicitly with train.wandb_id. resume="allow"
        # attaches to the existing id if present and creates it otherwise.
        run_name = tcfg.wandb_run_name or None
        wandb_id = tcfg.get("wandb_id", None)
        if not wandb_id:
            base_id = run_name or Path(tcfg.save_dir).name
            wandb_id = "".join(c if (c.isalnum() or c in "_-.") else "_"
                               for c in str(base_id))
        wandb.init(project=tcfg.wandb_project, name=run_name,
                   id=wandb_id, resume="allow",
                   config=OmegaConf.to_container(cfg, resolve=True))
        wandb.define_metric("_step")
        # samples_seen is defined against _step so it can be selected as the
        # x-axis in the UI: curves from runs at different batch sizes only line
        # up on samples, never on steps.
        wandb.define_metric("samples_seen", step_metric="_step")
        for prefix in ("train", "val", "perf"):
            wandb.define_metric(f"{prefix}/*", step_metric="_step")
        wandb.define_metric("lr", step_metric="_step")

    # Integrated-action cohorts are deterministic middle windows.
    explicit_metric_cohorts = (
        "metric_train_samples" in tcfg or "metric_val_samples" in tcfg)
    metric_train_samples = int(tcfg.get("metric_train_samples", 64))
    metric_val_samples = int(tcfg.get("metric_val_samples", 64))
    metric_batch_size = int(
        tcfg.get("metric_batch_size", tcfg.batch_size))
    # Common random numbers: the sampler's initial noise is pinned to this
    # constant at every checkpoint and in every run. Fixed literal (not
    # tcfg.seed) on purpose -- runs that differ in seed should still be scored
    # against the same noise.
    gen_noise_seed = 20240719
    action_source = (
        select_action_source(sources) if "action" in mfcfg.modalities else None)

    integ_every = int(tcfg.get("integ_every", 0))
    eval_max_batches = int(tcfg.max_eval_batches) * world_size
    if integ_every and tcfg.eval_every and integ_every % int(tcfg.eval_every) != 0:
        raise ValueError(
            f"integ_every ({integ_every}) must be a multiple of eval_every "
            f"({tcfg.eval_every}); integrated metrics are computed on eval boundaries.")
    integ_loaders = {}
    if integ_every and is_main and action_source is not None:
        if explicit_metric_cohorts:
            for split in ("train", "val"):
                loader = fixed_metric_loader(
                    action_source[f"{split}_loader"],
                    (metric_train_samples if split == "train"
                     else metric_val_samples),
                    metric_batch_size, tcfg.seed)
                if loader is not None:
                    integ_loaders[split] = loader
        else:
            # Preserve legacy configs whose integrated metric was expressed as
            # a number of per-rank training batches.
            integ_n_batches = int(tcfg.get("integ_batches", 1)) * world_size
            for split in ("train", "val"):
                source_loader = action_source[f"{split}_loader"]
                n_batches = n_eval_batches(integ_n_batches, source_loader)
                if source_loader is not None:
                    iterator = iter(source_loader)
                    integ_loaders[split] = [
                        next(iterator) for _ in range(n_batches)]

    save_dir = Path(tcfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = bool(tcfg.get("bf16", True)) and device.type == "cuda"
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if use_bf16 else nullcontext())

    # ---- periodic checkpoints for external closed-loop evaluation ----
    sim_eval_every = int(tcfg.get("sim_eval_every", 10000))
    # Do not emit external-eval checkpoints before this step.
    sim_eval_start = int(tcfg.get("sim_eval_start", 0))
    # Emit step-stamped checkpoints (+ TRAINING_DONE) for an external evaluator.
    emit_step_ckpts = bool(tcfg.get("emit_step_ckpts", False))
    # Persist a kept (not-overwritten) step-stamped checkpoint every ckpt_every
    # steps, independent of sim_eval -- so we always have periodic snapshots for
    # later SR eval even when in-training closed-loop eval is off. 0 = disabled.
    ckpt_every = int(tcfg.get("ckpt_every", 0))
    # Overwrite last.pt more frequently for preemption recovery without retaining
    # extra snapshots or running evaluation. 0 preserves the legacy eval-only
    # last.pt cadence.
    last_every = int(tcfg.get("last_every", 0))
    if last_every < 0:
        raise ValueError(f"last_every must be non-negative, got {last_every}")
    if emit_step_ckpts and sim_eval_every % int(tcfg.eval_every or 1) != 0:
        raise ValueError(
            f"sim_eval_every ({sim_eval_every}) must be a multiple of eval_every "
            f"({tcfg.eval_every}); SR checkpoints are emitted on eval boundaries.")
    if emit_step_ckpts and sim_eval_start % int(tcfg.eval_every or 1) != 0:
        raise ValueError(
            f"sim_eval_start ({sim_eval_start}) must be a multiple of eval_every "
            f"({tcfg.eval_every}); SR checkpoints are emitted on eval boundaries.")
    if ckpt_every and ckpt_every % int(tcfg.eval_every or 1) != 0:
        raise ValueError(
            f"ckpt_every ({ckpt_every}) must be a multiple of eval_every "
            f"({tcfg.eval_every}); step-stamped checkpoints are saved on eval "
            f"boundaries.")
    done_file = save_dir / "TRAINING_DONE"
    if (emit_step_ckpts or ckpt_every) and is_main:
        if done_file.exists():
            done_file.unlink()
        (save_dir / "ckpts").mkdir(exist_ok=True)

    def training_ckpt(step):
        return {
            "model": raw_model.state_dict(),
            "ema": ema.shadow if ema is not None else None,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "task_to_id": (
                build_task_to_id(task_vocab) if task_vocab is not None else None
            ),
            "stats": pack_stats(stats),
            "source_stats": {
                source["name"]: pack_stats(source["stats"])
                for source in sources},
            "step": step,
            # Layout-independent progress: `step` only means something
            # alongside the global batch that produced it.
            "samples_seen": step * global_batch,
            "global_batch": global_batch,
        }

    # Decorrelate training-time RNG across DDP ranks. A bare manual_seed(seed) puts
    # every rank in RNG LOCKSTEP: identical active-modality draws, flow times, and
    # noise tensors each step (verified empirically), which collapses the effective
    # (modality, t, noise) diversity of the global batch to one rank's worth and
    # makes per-step gradients far noisier. Reseeding here (not earlier) keeps
    # everything above rank-identical: demo splits/stats take explicit seeds, and
    # DDP broadcasts rank-0 weights at wrap time. DataLoader worker base seeds are
    # drawn from this RNG at iterator creation, so workers decorrelate too.
    torch.manual_seed(tcfg.seed + 10_000 * rank)

    # find_unused_parameters=True is required because disjoint runs one expert,
    # sequential action streams route through only the action expert. ModAR can
    # also skip experts on some steps (tried False 2026-08-11;
    # DDP reduction error within one step despite the unused-param warning).
    #
    # Disjoint is different: every forward intentionally selects exactly one
    # expert, and a training step performs separate dynamics/action forwards.
    # DDP's unused-parameter traversal cannot safely coordinate those successive
    # backward passes. Give every trainable tensor a zero-valued graph edge and
    # use the normal reducer instead. The selected expert still receives the
    # only nonzero gradient.
    ddp_complete_graph = distributed and mfcfg.schedule_mode == "disjoint"
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(raw_model, device_ids=[local_rank],
                    find_unused_parameters=not ddp_complete_graph)
    else:
        model = raw_model

    def complete_ddp_graph(loss: torch.Tensor) -> torch.Tensor:
        if not ddp_complete_graph:
            return loss
        return loss + sum(
            parameter.reshape(-1)[0] * 0.0
            for parameter in raw_model.parameters()
            if parameter.requires_grad
        )

    source_iters = {
        source["name"]: cycle(source["train_loader"])
        for source in sources}
    # Epoch denominators. The dataset is indexed by demo and draws a fresh random
    # window per demo per pass, so an epoch counts how many distinct windows each
    # demo has contributed -- the number that says whether a sample budget is
    # over- or under-exposing this particular dataset. Reported per source
    # because the action and dynamics pools differ in size by an order of
    # magnitude; the text log shows the largest (the exposure bottleneck).
    # n_demos, not len(dataset): train loaders stretch their epoch by repeating
    # the demo list (see _build_loader), and counting the repeats as data would
    # divide every reported epoch number by that factor.
    epoch_denoms = {source["name"]: max(1, n_demos_of(source["train_loader"]))
                    for source in sources}
    primary_source = max(epoch_denoms, key=epoch_denoms.get)
    # Grad-accumulate one weighted forward per source and all-reduce only after
    # the final backward.
    no_sync = model.no_sync if distributed else nullcontext
    model.train()

    t0 = time.time()
    last_log_t, last_log_step = t0, start_step
    for step in range(start_step, tcfg.max_steps):
        opt.zero_grad(set_to_none=True)
        step_sources = sources
        step_weights = normalize_source_weights(step_sources)
        step_outputs = []
        for index, (source, weight) in enumerate(zip(step_sources, step_weights)):
            batch = to_device(next(source_iters[source["name"]]), device)
            sync_ctx = (
                no_sync() if index + 1 < len(step_sources) else nullcontext())
            with sync_ctx:
                with amp_ctx:
                    source_out = model(
                        **batch, stream=source["stream"],
                        loss_coeffs=source["loss_coeffs"])
                    weighted_loss = complete_ddp_graph(
                        source_out["loss"] * weight)
                weighted_loss.backward()
            step_outputs.append((source, source_out, weight))

        loss = weighted_source_loss(step_outputs).detach()
        out = {"loss": loss}
        zero = torch.zeros((), device=device)
        for modality in raw_model.cfg.modalities:
            present = [
                (source_out, weight)
                for _, source_out, weight in step_outputs
                if float(source_out[f"n_{modality}"]) > 0]
            if present:
                denom = sum(weight for _, weight in present)
                out[f"loss_{modality}"] = sum(
                    weight * source_out[f"loss_{modality}"]
                    for source_out, weight in present) / denom
                out[f"n_{modality}"] = sum(
                    source_out[f"n_{modality}"] for source_out, _ in present)
            else:
                out[f"loss_{modality}"] = zero
                out[f"n_{modality}"] = zero
        action_present = [
            (source_out, weight) for _, source_out, weight in step_outputs
            if float(source_out["n_action"]) > 0]
        for key in ("act_mm", "act_deg"):
            if action_present:
                denom = sum(weight for _, weight in action_present)
                out[key] = sum(
                    weight * source_out[key]
                    for source_out, weight in action_present) / denom
            else:
                out[key] = zero
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        sched.step()
        if ema is not None:
            ema.update(raw_model)

        if step % tcfg.log_every == 0 and is_main:
            now = time.time()
            d_steps = step - last_log_step
            # Windowed it/s. The fallback counts steps since THIS process
            # started, not since step 0: on a resume the two differ by the whole
            # history, and dividing 200k steps by the half second it took to reach
            # the first log reported 450,000 it/s.
            rate = (d_steps / (now - last_log_t) if d_steps > 0
                    else (step - start_step + 1) / (now - t0))
            last_log_t, last_log_step = now, step
            eta_h = (tcfg.max_steps - step) / max(rate, 1e-9) / 3600.0
            gpu_gb = (torch.cuda.max_memory_allocated(device) / 1e9
                      if device.type == "cuda" else 0.0)
            loss = loss.detach()
            active_mods = {m for m in raw_model.cfg.modalities
                           if float(out[f"n_{m}"]) > 0}
            mod_losses = {m: float(out[f"loss_{m}"]) for m in active_mods
                          if f"loss_{m}" in out}
            mod_str = " ".join(f"{m[:3]} {v:.4f}" for m, v in mod_losses.items())
            err_str = ""
            if "action" in active_mods:
                err_str = (f"| mm_os {float(out['act_mm']):.1f} "
                           f"deg_os {float(out['act_deg']):.2f} ")
            samples_seen = (step + 1) * global_batch
            epochs = samples_seen / epoch_denoms[primary_source]
            msg = (f"step {step:6d} | loss {float(loss):.4f} "
                   f"| {mod_str} "
                   f"{err_str}"
                   f"| lr {sched.get_last_lr()[0]:.2e} | {rate:.1f} it/s "
                   f"| eta {eta_h:.1f}h | peak {gpu_gb:.1f}GB "
                   f"| {samples_seen/1e6:.2f}M smp ({epochs:.0f} ep)")
            print(msg, flush=True)
            if use_wandb:
                log_d = {
                    "_step": step,
                    # Logged alongside every metric so any run can be replotted
                    # against samples or epochs instead of steps -- the only way
                    # two runs at different batch sizes are comparable.
                    "samples_seen": samples_seen,
                    "train/loss": float(loss),
                    "lr": sched.get_last_lr()[0],
                    "perf/steps_per_sec": rate,
                    "perf/samples_per_sec": (
                        rate * world_size
                        * sum(source["batch_size"] for source in step_sources)),
                    "perf/global_batch": global_batch,
                    "perf/step_ms": 1000.0 / max(rate, 1e-9),
                    "perf/eta_hours": eta_h,
                    "perf/elapsed_hours": (now - t0) / 3600.0,
                    "perf/gpu_mem_peak_gb": gpu_gb,
                }
                for name, denom in epoch_denoms.items():
                    log_d[f"perf/epochs/{name}"] = samples_seen / denom
                for source, source_out, _ in step_outputs:
                    log_d[f"train/{source['name']}/loss"] = float(
                        source_out["loss"])
                for m, v in mod_losses.items():
                    log_d[f"train/loss_{m}"] = v
                if "action" in active_mods:
                    log_d["train/mm_error_onestep"] = float(out["act_mm"])
                    log_d["train/deg_error_onestep"] = float(out["act_deg"])
                wandb.log(log_d, step=step)

        eval_due = bool(tcfg.eval_every and step > 0
                        and step % tcfg.eval_every == 0)
        if (is_main and last_every and step > 0
                and step % last_every == 0 and not eval_due):
            save_ckpt_atomic(training_ckpt(step), save_dir / "last.pt")

        if eval_due and is_main:
            metrics = evaluate_sources(
                raw_model, sources, device, eval_max_batches)
            if integ_every and step % integ_every == 0:
                for split, loader in integ_loaders.items():
                    for k, v in evaluate_integrated(
                            raw_model, loader, device,
                            noise_seed=gen_noise_seed).items():
                        metrics[f"{split}/{k}"] = v
            if metrics:
                print("  " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                      flush=True)
                if use_wandb:
                    wandb.log({"_step": step, **metrics}, step=step)
            ckpt = training_ckpt(step)
            save_ckpt_atomic(ckpt, save_dir / "last.pt")
            # Persist a kept step-stamped checkpoint when either (a) the SR worker
            # cadence fires (emit_step_ckpts every sim_eval_every, gating closed-loop
            # eval) or (b) the standalone ckpt_every snapshot cadence fires. Both land
            # on eval boundaries; a single save covers overlapping steps.
            emit_now = (emit_step_ckpts and step >= sim_eval_start
                        and step % sim_eval_every == 0)
            emit_now = emit_now or (ckpt_every and step % ckpt_every == 0)
            if emit_now:
                ckpt_dir = save_dir / "ckpts"
                ckpt_dir.mkdir(exist_ok=True)
                save_ckpt_atomic(ckpt, ckpt_dir / f"ckpt_{step:07d}.pt")

        # Resync ranks after the (rank-0-only) periodic eval/ckpt work so
        # the others don't sit blocked at the next all-reduce for the whole eval.
        if distributed and step > 0:
            did_rank0_work = bool(
                tcfg.eval_every and step % tcfg.eval_every == 0)
            if did_rank0_work:
                dist.barrier()

    # Final checkpoint: range(max_steps) never reaches max_steps, so save the
    # final weights and emit a step-stamped checkpoint for offline consumers.
    # (rank 0 only under DDP; all ranks hold identical weights post-all-reduce.)
    if is_main:
        final_ckpt = training_ckpt(int(tcfg.max_steps))
        save_ckpt_atomic(final_ckpt, save_dir / "last.pt")
        if emit_step_ckpts or ckpt_every:
            ckpt_dir = save_dir / "ckpts"
            ckpt_dir.mkdir(exist_ok=True)
            save_ckpt_atomic(final_ckpt, ckpt_dir / f"ckpt_{tcfg.max_steps:07d}.pt")

        # Signal offline checkpoint consumers whenever periodic snapshots were
        # enabled, regardless of whether their cadence came from SR or ckpt_every.
        if emit_step_ckpts or ckpt_every:
            done_file.touch()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    log0("done.")


if __name__ == "__main__":
    main()
