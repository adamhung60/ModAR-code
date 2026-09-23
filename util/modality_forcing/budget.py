"""Batch-invariant training budgets.

An optimizer step is not a portable unit of training: it means 16 samples in one
run and 48 in another. Denominating a run in steps therefore couples four things
that should be independent -- the data budget, the LR warmup length, the eval
grid, and the x-axis every curve is plotted against -- to whatever batch size and
GPU count the run happened to land on. Changing any of them silently rescales the
other three, which makes two runs incomparable in a way nothing in the config
reveals.

Configs instead declare budgets in samples (``max_samples``, ``warmup_samples``,
``eval_every_samples``, ...). The step counts are derived here, once, after the
global batch is known. Step-denominated keys still work and are left untouched,
so existing configs keep their current behaviour.

Samples are counted per stream: one optimizer step consumes ``batch_size *
world_size`` samples, and a co-training run draws that many from each of its
sources. That is the unit that makes ``batch_size`` comparisons mean what they
look like they mean.
"""
from __future__ import annotations

# (step-denominated key, sample-denominated key, must land on an eval boundary)
#
# The trainer requires the boundary-aligned cadences to divide eval_every,
# because their checkpoints and metrics are emitted from the eval branch. Naive
# rounding of each one independently would break that invariant, so they are
# snapped to the resolved eval_every below.
SAMPLE_BUDGETS = (
    ("max_steps", "max_samples", False),
    ("warmup_steps", "warmup_samples", False),
    ("eval_every", "eval_every_samples", False),
    # Rolling recovery checkpoints are independent of evaluation and therefore
    # must not be snapped to the eval grid.
    ("last_every", "last_every_samples", False),
    ("ckpt_every", "ckpt_every_samples", True),
    ("integ_every", "integ_every_samples", True),
    ("sim_eval_every", "sim_eval_every_samples", True),
    ("sim_eval_start", "sim_eval_start_samples", True),
)


def steps_for_samples(samples: int, global_batch: int) -> int:
    """Optimizer steps needed to consume ``samples`` at this global batch."""
    if global_batch <= 0:
        raise ValueError(f"global_batch must be positive, got {global_batch}")
    samples = int(samples)
    if samples <= 0:
        return 0
    return max(1, round(samples / global_batch))


def resolve_sample_budgets(train_cfg, global_batch: int) -> dict:
    """Derive step budgets from the ``*_samples`` keys present in ``train_cfg``.

    Returns only the keys that were given a sample budget, so a purely
    step-denominated config resolves to ``{}`` and is left alone. The caller
    applies the result and is responsible for logging it.
    """
    derived = {}
    for steps_key, samples_key, _ in SAMPLE_BUDGETS:
        samples = train_cfg.get(samples_key, None)
        if samples is None:
            continue
        derived[steps_key] = steps_for_samples(samples, global_batch)

    eval_every = int(derived.get("eval_every", train_cfg.get("eval_every", 0)) or 0)
    if eval_every:
        for steps_key, _, on_eval_boundary in SAMPLE_BUDGETS:
            value = derived.get(steps_key, 0)
            if on_eval_boundary and value:
                derived[steps_key] = max(1, round(value / eval_every)) * eval_every
    return derived


def describe_budget(derived: dict, global_batch: int) -> str:
    """One-line summary of a resolved budget, for the training log."""
    if not derived:
        return f"global batch {global_batch}; step-denominated config"
    parts = " ".join(f"{k}={v}" for k, v in sorted(derived.items()))
    return f"global batch {global_batch} -> {parts}"
