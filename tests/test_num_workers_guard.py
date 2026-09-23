"""Data loading cannot silently fall back to zero worker processes.

Zero workers run the input pipeline inside the training step. A request for zero
workers is rejected, a value larger than the cores available to the rank is
capped, and every shipped config asks for a real worker count.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from util.modality_forcing.config_paths import load_config

from scripts.train.train_modality_forcing import (
    MIN_NUM_WORKERS, resolve_num_workers)

REPO = Path(__file__).resolve().parents[1]


def tcfg(**over):
    base = {"num_workers": 8}
    base.update(over)
    return OmegaConf.create(base)


def usable_cores(monkeypatch, n):
    """Pin the cores this process may run on.

    Patches sched_getaffinity rather than cpu_count because the cap is about what
    this job is ALLOWED to use, which under a Slurm cgroup is far less than the
    node's core count.
    """
    monkeypatch.setattr(
        "scripts.train.train_modality_forcing.os.sched_getaffinity",
        lambda _pid: set(range(n)))


@pytest.mark.parametrize("workers", [0, 1, MIN_NUM_WORKERS - 1])
def test_starved_loader_is_refused(workers):
    with pytest.raises(ValueError, match="below the floor"):
        resolve_num_workers(tcfg(num_workers=workers), 1, log=lambda *_: None)


def test_starved_loader_can_be_opted_into_explicitly():
    """An escape hatch has to exist, but it has to be written down in the config."""
    got = resolve_num_workers(
        tcfg(num_workers=0, allow_starved_loader=True), 1, log=lambda *_: None)
    assert got == 0


def test_missing_key_falls_back_to_the_floor_not_zero():
    got = resolve_num_workers(OmegaConf.create({}), 1, log=lambda *_: None)
    assert got == MIN_NUM_WORKERS


def test_value_within_budget_is_untouched(monkeypatch):
    usable_cores(monkeypatch, 96)
    assert resolve_num_workers(tcfg(num_workers=8), 1,
                              log=lambda *_: None) == 8


def test_many_local_ranks_are_capped_not_zeroed(monkeypatch):
    """The case that motivated num_workers=0: eight ranks sharing one node.

    32 cores over 8 ranks leaves 3 after reserving one for this rank's own compute
    thread, which the floor lifts back to MIN_NUM_WORKERS -- capped, but still
    prefetching, which is the whole point.
    """
    usable_cores(monkeypatch, 32)
    got = resolve_num_workers(tcfg(num_workers=8), 8, log=lambda *_: None)
    assert got == MIN_NUM_WORKERS

    # a roomier node with the same rank count keeps more of the requested workers
    usable_cores(monkeypatch, 96)
    assert resolve_num_workers(tcfg(num_workers=8), 8, log=lambda *_: None) == 8


def test_cap_never_drops_below_the_floor(monkeypatch):
    """Even a badly oversubscribed node keeps some prefetching."""
    usable_cores(monkeypatch, 4)
    got = resolve_num_workers(tcfg(num_workers=16), 8, log=lambda *_: None)
    assert got == MIN_NUM_WORKERS


def test_cap_respects_a_cgroup_slice_of_a_big_node(monkeypatch):
    """A cgroup slice: 12 CPUs granted on a 96-core machine.

    os.cpu_count() reports 96 there, which would compute a cap of 95 and leave a
    12-CPU job running 12 workers plus its own compute thread -- every worker
    contending with the process it is meant to be feeding. The cap has to see 12.
    """
    monkeypatch.setattr(
        "scripts.train.train_modality_forcing.os.cpu_count", lambda: 96)
    usable_cores(monkeypatch, 12)
    assert resolve_num_workers(tcfg(num_workers=12), 1, log=lambda *_: None) == 11


DYNSCALE_CONFIGS = sorted((REPO / "conf").rglob("*.yaml"))


def test_no_shipped_config_requests_a_starved_loader():
    """A config asking for 0 would now crash the run at startup; none should."""
    offenders = []
    for path in DYNSCALE_CONFIGS:
        text = path.read_text()
        if "num_workers" not in text:
            continue
        cfg = load_config(str(path), OmegaConf.create({}))
        workers = cfg.get("train", {}).get("num_workers", None)
        if workers is not None and int(workers) < MIN_NUM_WORKERS:
            offenders.append(f"{path.relative_to(REPO)}: {workers}")
    assert not offenders, (
        "these configs would be refused at startup:\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("leaf", [
    "conf/methods/modar.yaml",
    "conf/methods/unified.yaml",
    "conf/methods/disjoint.yaml",
    "conf/methods/independent_noise.yaml",
    "conf/methods/action_only.yaml",
])
def test_public_methods_inherit_a_real_value(leaf):
    cfg = load_config(str(REPO / leaf), OmegaConf.create({}))
    workers = int(cfg.train.num_workers)
    assert workers >= MIN_NUM_WORKERS, f"{leaf} resolves to {workers}"
    # and it survives the resolver untouched on a roomy box
    assert resolve_num_workers(cfg.train, 1, log=lambda *_: None) >= MIN_NUM_WORKERS
