#!/usr/bin/env python
"""Closed-loop success-rate eval of WAM checkpoints in RoboTwin (SAPIEN).

Runs in the ``RoboTwin`` conda env (SAPIEN + our torch model). For each task we
reconstruct the held-out ``dyn_val`` demo pool exactly as training did
(``split_demos_cotrain`` over the checkpoint's own data config) and map each
val demo back to the expert-verified collection seed via ``seed.txt`` (demo_i
<-> episode_i <-> seed.txt[i]). Re-instantiating the env with that seed
reproduces a feasible, held-out initial condition -- no motion planner at eval,
identical ICs across every method (ModAR / disjoint / action-only), and no
leakage (dyn_val is disjoint from both training streams by the nested split).

Emits one JSONL record per checkpoint with step, per-task, and overall success
rates; optionally logs to W&B under ``robotwin/*``.

Example (single ckpt):
    python robotwin_manip/eval/run_sr.py \
        --ckpt outputs/modar/last.pt \
        --data-root /path/to/robotwin6_packed \
        --robotwin-data /path/to/RoboTwin/data \
        --robotwin-repo /path/to/RoboTwin \
        --task-config modar_robotwin6 \
        --n-per-task 50 --out outputs/modar/success_rate.jsonl
"""
from __future__ import annotations

import argparse
import builtins
import gc
import glob
import hashlib
import importlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from util.modality_forcing.data import (  # noqa: E402
    discover_demos_multi, split_demos_cotrain, split_demos_modality)
from robotwin_manip.eval.robotwin_deploy import (  # noqa: E402
    WAMPolicy, seed_inference_rng)

PROGRESS_SCHEMA_VERSION = 1


def _without_robotwin_step_prints(original_print):
    """Drop RoboTwin's per-action carriage-return progress spam."""
    def filtered_print(*args, **kwargs):
        is_step_progress = (
            bool(args)
            and isinstance(args[0], str)
            and args[0].startswith("step: \033[92m")
            and kwargs.get("end") == "\r"
        )
        if not is_step_progress:
            return original_print(*args, **kwargs)
        return None

    return filtered_print


# ---- held-out eval-seed derivation ------------------------------------------

def read_seed_txt(path: Path) -> list[int]:
    """Parse RoboTwin's space-separated seed.txt -> [seed for episode i]."""
    toks = path.read_text().split()
    return [int(t) for t in toks if t.lstrip("-").isdigit()]


def held_out_dyn_val(data_root: str, tasks: list[str], split: dict) -> list[str]:
    """dyn_val demo dirs from the training-time split (single source of truth).

    ``split`` selects the scheme, matching the trainer's loader selection:
      * count-based nested (``split_demos_cotrain``) when ``n_action_train`` is
        set -- exactly what ``build_modality_loaders_cotrain`` uses;
      * ratio-based (``split_demos_modality``) via action/dyn val ratios --
        legacy two-stream runs.
    """
    demos_by_task = discover_demos_multi(data_root, tasks)
    dl = split.get("demo_limit")
    if dl is not None:
        demos_by_task = {t: d[:dl] for t, d in demos_by_task.items()}
    if split.get("n_action_train") is not None:
        pools = split_demos_cotrain(
            demos_by_task, int(split["n_action_train"]),
            int(split["n_action_val"]), int(split["n_dyn_val"]),
            None if split.get("n_dyn_train") is None else int(split["n_dyn_train"]),
            int(split["seed"]),
            split_universe=(None if split.get("split_universe") is None
                            else int(split["split_universe"])))
    else:
        pools = split_demos_modality(
            demos_by_task, float(split["action_val_ratio"]),
            float(split["dyn_val_ratio"]), int(split["seed"]))
    # Default IC pool is dyn_val (held-out). ``split["pool"]`` selects a different
    # training-time pool (e.g. action_train) for train-split SR; the demo->seed
    # mapping in dyn_val_seeds_for_task is pool-agnostic (demo_<i> == seed.txt[i]).
    return pools[split.get("pool", "dyn_val")]


def dyn_val_seeds_for_task(task: str, data_root: str, tasks: list[str],
                           split: dict, seed_list: list[int],
                           n_per_task: int,
                           skip_unmapped: bool = False) -> list[tuple[str, int]]:
    """Held-out (demo_dir, seed) pairs for one task's dyn_val pool.

    The eval ICs are exactly the dyn-stream validation demos (disjoint from both
    training pools by the nested split). Maps each held-out demo back to its
    collection seed via ``demo_<i> == seed.txt[i]``.
    """
    dyn_val = held_out_dyn_val(data_root, tasks, split)
    out = []
    for demo_dir in dyn_val:
        if Path(demo_dir).parent.name != task:
            continue
        # demo_<i> dir name IS the episode index (converter enumerates sorted
        # episodes, packer preserves the id) == seed.txt line i.
        m = re.search(r"(\d+)$", Path(demo_dir).name)
        if not m:
            raise ValueError(f"cannot parse episode index from {demo_dir}")
        ep = int(m.group(1))
        if ep >= len(seed_list):
            if skip_unmapped:
                continue
            raise IndexError(f"{task}: episode {ep} beyond seed.txt "
                             f"({len(seed_list)} seeds)")
        out.append((ep, demo_dir, seed_list[ep]))
    out.sort(key=lambda x: x[0])                      # deterministic by episode idx
    if skip_unmapped and len(out) < n_per_task:
        raise ValueError(
            f"{task}: requested {n_per_task} held-out ICs but only {len(out)} "
            f"have entries in seed.txt ({len(seed_list)} seeds)")
    return [(d, s) for _ep, d, s in out[:n_per_task]]


# ---- RoboTwin env arg assembly (mirrors script/eval_policy.py main) ----------

def build_task_args(task_name: str, task_config: str) -> dict:
    """Assemble the setup_demo(**args) dict, matching eval_policy.py exactly.

    Must be called with cwd == RoboTwin repo root (relative ./task_config paths).
    """
    from envs import CONFIGS_PATH

    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), "r",
              encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = "wam"

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r",
              encoding="utf-8") as f:
        embodiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r",
              encoding="utf-8") as f:
        camera_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    def robot_file(kind):
        rf = embodiments[kind]["file_path"]
        if rf is None:
            raise ValueError(f"no embodiment file for {kind}")
        return rf

    def emb_config(rf):
        with open(os.path.join(rf, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    head_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_cfg[head_type]["h"]
    args["head_camera_w"] = camera_cfg[head_type]["w"]

    emb = args.get("embodiment")
    if len(emb) == 1:
        args["left_robot_file"] = robot_file(emb[0])
        args["right_robot_file"] = robot_file(emb[0])
        args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        args["left_robot_file"] = robot_file(emb[0])
        args["right_robot_file"] = robot_file(emb[1])
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")
    args["left_embodiment_config"] = emb_config(args["left_robot_file"])
    args["right_embodiment_config"] = emb_config(args["right_robot_file"])

    args["policy_name"] = "WAM"
    args["eval_mode"] = True          # loads per-task step_lim
    args["render_freq"] = 0
    return args


def load_task_env(task_name: str):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def initialize_eval_task_state(env, task_name: str) -> None:
    """Initialize task state that RoboTwin otherwise sets only in play_once()."""
    if task_name in {"place_object_scale", "put_object_cabinet"} \
            and not hasattr(env, "arm_tag"):
        env.arm_tag = "right" if env.object.get_pose().p[0] > 0 else "left"
    if task_name == "put_object_cabinet" and not hasattr(env, "origin_z"):
        env.origin_z = env.object.get_pose().p[2]


# ---- rollout ----------------------------------------------------------------

def run_task(policy: WAMPolicy, task_name: str, task_config: str,
             seed_pairs: list[tuple[str, int]],
             completed: list[dict] | None = None,
             on_episode=None, verbose: bool = True,
             replacement_pairs: list[tuple[str, int]] | None = None,
             samples_per_ic: int = 1,
             total_per_task: int | None = None) -> dict:
    """Closed-loop SR over the held-out ICs, one FRESH env per episode.

    Critical: we rebuild the whole env (new SAPIEN engine/scene) every episode
    instead of reusing one Base_Task across the loop. RoboTwin's
    close_env()+setup_demo() does NOT release the previous episode's PhysX scene
    state; reusing the instance lets that state accumulate and changes contact
    dynamics across episodes. Full teardown and collection preserve independent,
    reproducible trials (see haosulab/SAPIEN#174).
    """
    policy.set_task(task_name)
    args = build_task_args(task_name, task_config)
    completed = completed or []
    n_success = sum(int(r["success"]) for r in completed)
    # Per-replan inference latency: plan() runs the ODE sample synchronously and
    # syncs CUDA via the .cpu() on its return, so wall-clock around the call is
    # the model's inference time, isolated from SAPIEN stepping. Only the calls
    # made in THIS process are timed (resume loses earlier ones), but per-call
    # latency is stable so the average stays representative.
    plan_time_s = 0.0
    plan_calls = 0
    n_replacements = sum("replacement_seed" in r for r in completed)
    used_replacements = {
        (r["replacement_demo"], int(r["replacement_seed"]))
        for r in completed if "replacement_seed" in r
    }
    replacements = iter(
        pair for pair in (replacement_pairs or [])
        if (Path(pair[0]).name, int(pair[1])) not in used_replacements
    )
    # Flatten ICs into a rollout plan of (ic_index, sample, demo, seed). With
    # samples_per_ic==1 this is exactly the historic per-IC list, so the resume
    # prefix is unchanged. With K>1 each IC is rolled out K times on the SAME
    # scene seed (deterministic SAPIEN IC) while the policy-noise seed varies, so
    # the K outcomes are an unbiased sample of the policy's stochastic SR at that IC.
    K = max(1, int(samples_per_ic))
    if total_per_task is None:
        rollouts = [
            (i, s, demo, int(seed))
            for i, (demo, seed) in enumerate(seed_pairs)
            for s in range(K)
        ]
    else:
        if total_per_task < len(seed_pairs):
            raise ValueError(
                f"{task_name}: total_per_task={total_per_task} is smaller "
                f"than the {len(seed_pairs)} primary initial conditions")
        rollouts = []
        for rollout in range(total_per_task):
            i = rollout % len(seed_pairs)
            demo, seed = seed_pairs[i]
            rollouts.append(
                (i, rollout // len(seed_pairs), demo, int(seed)))
    total = len(rollouts)
    start = len(completed)
    if start and verbose:
        print(f"  [{task_name}] resume at {start}/{total} "
              f"| SR={n_success/start:.3f}", flush=True)
    for idx in range(start, total):
        i, s, demo_dir, seed = rollouts[idx]
        eval_demo, eval_seed = demo_dir, int(seed)
        setup_errors = []
        while True:
            env = load_task_env(task_name)
            try:
                env.setup_demo(
                    now_ep_num=i, seed=eval_seed, is_test=True, **args)
                # Some RoboTwin tasks initialize success-check state only in
                # play_once(), which closed-loop policy evaluation never calls.
                initialize_eval_task_state(env, task_name)
                break
            except Exception as error:
                if error.__class__.__name__ != "UnStableError":
                    raise
                setup_errors.append(
                    f"{Path(eval_demo).name}/seed={eval_seed}: {error}")
                try:
                    env.close_env()
                except Exception:
                    pass
                del env
                gc.collect()
                try:
                    eval_demo, eval_seed = next(replacements)
                    eval_seed = int(eval_seed)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"{task_name}: no held-out replacement IC remains after "
                        f"unstable setup for {setup_errors}") from exc
                if verbose:
                    print(
                        f"  [{task_name}] replace unstable "
                        f"{Path(demo_dir).name}/seed={seed} with "
                        f"{Path(eval_demo).name}/seed={eval_seed}",
                        flush=True)
        replaced = (eval_demo, eval_seed) != (demo_dir, int(seed))
        n_replacements += int(replaced)
        # Pin CPU/CUDA RNG after setup so policy sampling noise is reproducible
        # per rollout regardless of position in the loop -- see seed_inference_rng.
        # sample 0 keeps the historic seed (== scene seed) for exact continuity;
        # samples >0 offset it so the same scene draws independent policy noise.
        infer_seed = eval_seed if s == 0 else eval_seed + s * 2_000_003
        seed_inference_rng(infer_seed)
        while env.take_action_cnt < env.step_lim and not env.eval_success:
            obs = env.get_obs()
            t_plan = time.perf_counter()
            planned = policy.plan(obs)
            plan_time_s += time.perf_counter() - t_plan
            plan_calls += 1
            for action in planned:
                env.take_action(action, action_type="qpos")
                if env.eval_success:
                    break
        succ = bool(env.eval_success) or bool(env.check_success())
        n_success += int(succ)
        env.close_env()
        del env
        gc.collect()
        if on_episode is not None:
            episode_kw = (
                {"sample": s}
                if (total_per_task is None and K > 1) or s > 0
                else {}
            )
            if replaced:
                on_episode(
                    task_name, i, demo_dir, int(seed), succ,
                    replacement_demo=Path(eval_demo).name,
                    replacement_seed=eval_seed,
                    setup_errors=setup_errors, **episode_kw)
            else:
                on_episode(task_name, i, demo_dir, int(seed), succ,
                           **episode_kw)
        if verbose:
            sample_label = f" s={s}" if K > 1 else ""
            seed_label = (
                f"seed={eval_seed} (replacement for {seed})"
                if replaced else f"seed={seed}")
            print(f"  [{task_name}] {idx+1}/{total}{sample_label} {seed_label}: "
                  f"{'SUCCESS' if succ else 'fail'} | SR={n_success/(idx+1):.3f}",
                  flush=True)
    n = total
    result = {"n": n, "n_success": n_success,
              "success_rate": (n_success / n) if n else 0.0}
    if n_replacements:
        result["n_replacements"] = n_replacements
    if plan_calls:
        result["plan_time_s"] = round(plan_time_s, 3)
        result["plan_calls"] = plan_calls
        result["mean_plan_s"] = round(plan_time_s / plan_calls, 4)
    return result


def infer_step(ckpt_path: str, override: int | None) -> int:
    """Logged step: --step override, else ckpt_<step>.pt filename, else ckpt['step']."""
    if override is not None:
        return int(override)
    m = re.match(r"ckpt_(\d+)\.pt$", os.path.basename(ckpt_path))
    if m:
        return int(m.group(1))
    return int(torch.load(ckpt_path, map_location="cpu",
                          weights_only=False).get("step", 0))


def read_sample_progress(ckpt_path: str) -> dict:
    """Samples the checkpoint was trained on, when it records them.

    Two runs at different batch sizes reach the same step having seen different
    amounts of data, so a step-keyed SR curve cannot be compared across them.
    Checkpoints written since the sample-budget change carry samples_seen; older
    ones do not, and simply get no sample fields.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    out = {}
    for key in ("samples_seen", "global_batch"):
        if ckpt.get(key) is not None:
            out[key] = int(ckpt[key])
    return out


def _atomic_write_json(path: Path, value: dict) -> None:
    """Durably replace a JSON state file without exposing partial contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(dir_fd)
    os.close(dir_fd)


def _progress_path(out_path: Path, step: int,
                   order_key: str | None = None) -> Path:
    suffix = ""
    if order_key:
        digest = hashlib.sha256(order_key.encode("utf-8")).hexdigest()[:10]
        suffix = f".order_{digest}"
    return out_path.with_name(
        f"{out_path.stem}.step_{step:07d}{suffix}.progress.json")


def _final_record_exists(out_path: Path, step: int,
                         order_key: str | None = None) -> bool:
    if not out_path.is_file():
        return False
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if (int(rec["step"]) == int(step)
                and rec.get("order_key") == order_key):
            return True
    return False


def _trial_specs(seed_pairs_by_task: dict[str, list[tuple[str, int]]],
                 samples_per_ic: int = 1,
                 total_per_task: int | None = None) -> list[dict]:
    # K==1 keeps the historic 4-key schema byte-identical so pre-existing SR
    # sidecars resume unchanged. K>1 expands each IC into K trials, one per
    # policy-noise draw, ordered IC-outer / sample-inner within each task.
    if total_per_task is not None:
        specs = []
        for task, pairs in seed_pairs_by_task.items():
            if total_per_task < len(pairs):
                raise ValueError(
                    f"{task}: total_per_task={total_per_task} is smaller "
                    f"than the {len(pairs)} primary initial conditions")
            for rollout in range(total_per_task):
                i = rollout % len(pairs)
                sample = rollout // len(pairs)
                demo, seed = pairs[i]
                spec = {
                    "task": task,
                    "index": i,
                    "demo": Path(demo).name,
                    "seed": int(seed),
                }
                if sample > 0:
                    spec["sample"] = sample
                specs.append(spec)
        return specs
    if samples_per_ic <= 1:
        return [
            {"task": task, "index": i, "demo": Path(demo).name, "seed": int(seed)}
            for task, pairs in seed_pairs_by_task.items()
            for i, (demo, seed) in enumerate(pairs)
        ]
    return [
        {"task": task, "index": i, "demo": Path(demo).name, "seed": int(seed),
         "sample": s}
        for task, pairs in seed_pairs_by_task.items()
        for i, (demo, seed) in enumerate(pairs)
        for s in range(samples_per_ic)
    ]


def _progress_fingerprint(ckpt: str, step: int, tasks: list[str],
                          task_config: str, split: dict,
                          seed_pairs_by_task: dict[str, list[tuple[str, int]]],
                          replacement_pairs_by_task:
                          dict[str, list[tuple[str, int]]],
                          n_per_task: int, exec_horizon: int, steps_per_phase,
                          solver, use_ema: bool,
                          generation_order: list[str] | None,
                          samples_per_ic: int = 1,
                          total_per_task: int | None = None,
                          infer_schedule: str | None = None) -> dict:
    ckpt_path = Path(ckpt)
    fp = {
        "ckpt": ckpt_path.name,
        "ckpt_bytes": ckpt_path.stat().st_size,
        "step": int(step),
        "tasks": list(tasks),
        "task_config": task_config,
        "split": split,
        "trials": _trial_specs(
            seed_pairs_by_task,
            samples_per_ic,
            total_per_task=total_per_task,
        ),
        "replacement_trials": _trial_specs(replacement_pairs_by_task),
        "n_per_task": int(n_per_task),
        "exec_horizon": int(exec_horizon),
        "steps_per_phase": steps_per_phase,
        "solver": solver,
        "use_ema": bool(use_ema),
        "generation_order": generation_order,
    }
    # Only stamp the key for multi-sample runs so single-sample sidecars stay
    # byte-identical to the historic fingerprint and resume without a mismatch.
    if samples_per_ic > 1:
        fp["samples_per_ic"] = int(samples_per_ic)
    if total_per_task is not None:
        fp["total_per_task"] = int(total_per_task)
    if infer_schedule is not None:
        fp["infer_schedule"] = infer_schedule
    return fp


def _summarize_progress(state: dict) -> None:
    results = state["results"]
    state["completed"] = len(results)
    state["n_success"] = sum(int(r["success"]) for r in results)
    state["n_replacements"] = sum("replacement_seed" in r for r in results)
    state["success_rate"] = (
        state["n_success"] / state["completed"] if state["completed"] else 0.0)
    per_task = {}
    for task in state["fingerprint"]["tasks"]:
        task_results = [r for r in results if r["task"] == task]
        n_success = sum(int(r["success"]) for r in task_results)
        target_n = sum(
            1 for trial in state["fingerprint"]["trials"]
            if trial["task"] == task)
        per_task[task] = {
            "n": len(task_results),
            "target_n": target_n,
            "n_success": n_success,
            "n_replacements": sum(
                "replacement_seed" in r for r in task_results),
            "success_rate": n_success / len(task_results) if task_results else 0.0,
        }
    state["per_task"] = per_task


def _load_or_create_progress(
        path: Path,
        fingerprint: dict,
        initial_results: list[dict] | None = None,
        initial_eval_s: float = 0.0) -> dict:
    if path.is_file():
        state = json.loads(path.read_text())
        if state.get("schema_version") != PROGRESS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported progress schema in {path}: "
                f"{state.get('schema_version')}")
        if state.get("fingerprint") != fingerprint:
            # Progress files created before replacement ICs were supported have
            # the same primary trials but no replacement pool. Upgrade those
            # sidecars in place so long-running SR jobs resume safely.
            legacy_fingerprint = {
                key: value for key, value in fingerprint.items()
                if key != "replacement_trials"
            }
            if state.get("fingerprint") == legacy_fingerprint:
                state["fingerprint"] = fingerprint
            else:
                raise ValueError(
                    f"progress fingerprint mismatch for {path}; move or delete "
                    "the stale sidecar before running a different evaluation")
        if not isinstance(state.get("results"), list):
            raise ValueError(f"invalid progress results in {path}")
    else:
        state = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "status": "running",
            "fingerprint": fingerprint,
            "step": fingerprint["step"],
            "ckpt": fingerprint["ckpt"],
            "total": len(fingerprint["trials"]),
            "results": list(initial_results or []),
            "eval_s": float(initial_eval_s),
        }

    expected_by_task = {
        task: [
            trial for trial in fingerprint["trials"]
            if trial["task"] == task
        ]
        for task in fingerprint["tasks"]
    }
    completed_by_task = {task: 0 for task in fingerprint["tasks"]}
    for result in state["results"]:
        task = result.get("task")
        task_index = completed_by_task.get(task)
        if task_index is None or task_index >= len(expected_by_task[task]):
            raise ValueError(
                f"non-prefix progress result for task {task!r} in {path}; "
                "refusing unsafe resume")
        expected = expected_by_task[task][task_index]
        keys = ("task", "index", "demo", "seed", "sample") \
            if "sample" in expected else ("task", "index", "demo", "seed")
        identity = {k: result[k] for k in keys}
        if identity != expected:
            raise ValueError(
                f"non-prefix progress result for task {task!r} index "
                f"{task_index} in {path}; refusing unsafe resume")
        completed_by_task[task] += 1
    _summarize_progress(state)
    return state


def _completed_by_task(state: dict) -> dict[str, list[dict]]:
    completed = {task: [] for task in state["fingerprint"]["tasks"]}
    for result in state["results"]:
        completed[result["task"]].append(result)
    return completed


def eval_checkpoint(ckpt: str, tasks: list[str], task_config: str,
                    data_root: str | None,
                    robotwin_data: str, n_per_task: int, device: str,
                    exec_horizon: int | None, steps_per_phase, solver,
                    use_ema: bool,
                    step: int | None, action_val_ratio: float | None,
                    dyn_val_ratio: float | None, split_seed: int | None,
                    out_path: Path,
                    generation_order: list[str] | None = None,
                    infer_schedule: str | None = None,
                    policy=None,
                    extra_fingerprint: dict | None = None,
                    split_pool: str = "dyn_val",
                    samples_per_ic: int = 1,
                    split_tasks: list[str] | None = None,
                    skip_unmapped_seeds: bool = False,
                    initial_results: list[dict] | None = None,
                    initial_eval_s: float = 0.0,
                    total_per_task: int | None = None) -> tuple[dict, Path]:
    # Callers may inject a compatible pre-built policy. ``ckpt`` still identifies
    # the checkpoint that defines the held-out split and evaluation step, while
    # ``extra_fingerprint`` keeps custom evaluations in distinct progress files.
    if policy is None:
        policy = WAMPolicy(ckpt=ckpt, device=device, exec_horizon=exec_horizon,
                           use_ema=use_ema, steps_per_phase=steps_per_phase,
                           solver=solver, generation_order=generation_order,
                           infer_schedule=infer_schedule)
    dc = policy.data_cfg
    data_root = data_root or dc.get("data_root")
    if not data_root:
        raise ValueError(
            "checkpoint data config has no data_root; pass --data-root")
    data_root = os.path.abspath(
        os.path.expandvars(os.path.expanduser(str(data_root))))
    # Held-out ICs = the training-time dyn-stream val pool. Pull the split knobs
    # from the checkpoint's own data config (CLI overrides win) so eval matches
    # exactly how this run was trained: count-based nested (n_action_train...) or
    # legacy ratio-based (action_val_ratio/dyn_val_ratio).
    sseed = split_seed if split_seed is not None else int(dc.get("seed", 42))
    split = {"seed": sseed, "demo_limit": dc.get("demo_limit")}
    # Which training-time pool supplies the eval ICs. Default dyn_val (held-out,
    # the deployment-relevant metric); other pools (e.g. action_train) enable
    # in-distribution "train-split" SR. Part of the fingerprint so a train-split
    # eval never resumes onto a dyn_val sidecar.
    split["pool"] = split_pool
    if dc.get("n_action_train") is not None:
        split.update(
            n_action_train=dc["n_action_train"], n_action_val=dc["n_action_val"],
            n_dyn_val=dc["n_dyn_val"], n_dyn_train=dc.get("n_dyn_train"),
            split_universe=dc.get("split_universe"))
    else:
        avr = action_val_ratio if action_val_ratio is not None else dc.get("action_val_ratio")
        dvr = dyn_val_ratio if dyn_val_ratio is not None else dc.get("dyn_val_ratio")
        if avr is None or dvr is None:
            raise ValueError(
                "checkpoint data config has neither n_action_train nor "
                "action_val_ratio/dyn_val_ratio; pass --action-val-ratio and "
                "--dyn-val-ratio to define the held-out pool.")
        split.update(action_val_ratio=avr, dyn_val_ratio=dvr)
    ckpt_step = infer_step(ckpt, step)

    # The held-out split is computed over the WHOLE task list: split_demos_cotrain
    # seeds each task's shuffle with (seed + its index in the list), so evaluating
    # a subset of tasks with that subset as the task list would silently hand every
    # task after the first a different val pool. ``split_tasks`` keeps the full
    # training task list for split purposes while ``tasks`` selects what to roll
    # out, which is what makes it safe to shard one eval across processes by task.
    split_task_list = list(split_tasks) if split_tasks else list(tasks)
    unknown = [t for t in tasks if t not in split_task_list]
    if unknown:
        raise ValueError(f"tasks {unknown} are not in split_tasks {split_task_list}")

    seed_pairs_by_task = {}
    replacement_pairs_by_task = {}
    for task in tasks:
        seed_list = read_seed_txt(
            Path(robotwin_data) / task / task_config / "seed.txt")
        all_pairs = dyn_val_seeds_for_task(
            task, data_root, split_task_list, split, seed_list,
            n_per_task if skip_unmapped_seeds else len(seed_list),
            skip_unmapped=skip_unmapped_seeds)
        seed_pairs_by_task[task] = all_pairs[:n_per_task]
        replacement_pairs_by_task[task] = all_pairs[n_per_task:]

    order_key = ",".join(generation_order) if generation_order is not None else None
    if infer_schedule is not None:
        order_key = f"{order_key or 'default'}__{infer_schedule}"
    progress_path = _progress_path(out_path, ckpt_step, order_key)
    fingerprint = _progress_fingerprint(
        ckpt, ckpt_step, tasks, task_config, split, seed_pairs_by_task,
        replacement_pairs_by_task,
        n_per_task, policy.exec_horizon, steps_per_phase, solver, use_ema,
        generation_order, samples_per_ic=samples_per_ic,
        total_per_task=total_per_task,
        infer_schedule=infer_schedule)
    if extra_fingerprint:
        fingerprint = {**fingerprint, **extra_fingerprint}
    state = _load_or_create_progress(
        progress_path,
        fingerprint,
        initial_results=initial_results,
        initial_eval_s=initial_eval_s,
    )
    _atomic_write_json(progress_path, state)
    completed = _completed_by_task(state)
    resume_eval_s = float(state.get("eval_s", 0.0))
    resume_t0 = time.time()

    def save_episode(task_name, index, demo_dir, seed, success,
                     sample=None, **replacement):
        result = {
            "task": task_name,
            "index": int(index),
            "demo": Path(demo_dir).name,
            "seed": int(seed),
            "success": bool(success),
        }
        if sample is not None:
            result["sample"] = int(sample)
        result.update(replacement)
        task_completed = sum(
            result["task"] == task_name
            for result in state["results"]
        )
        task_trials = [
            trial for trial in fingerprint["trials"]
            if trial["task"] == task_name
        ]
        expected = task_trials[task_completed]
        keys = ("task", "index", "demo", "seed", "sample") \
            if "sample" in expected else ("task", "index", "demo", "seed")
        identity = {k: result[k] for k in keys}
        if identity != expected:
            raise ValueError(
                f"episode result order mismatch: got {identity}, expected {expected}")
        state["results"].append(result)
        state["eval_s"] = round(resume_eval_s + time.time() - resume_t0, 1)
        _summarize_progress(state)
        _atomic_write_json(progress_path, state)

    per_task = {}
    tot_n = tot_s = 0
    tot_plan_s = 0.0
    tot_plan_calls = 0
    for task in tasks:
        res = run_task(
            policy, task, task_config, seed_pairs_by_task[task],
            completed=completed[task], on_episode=save_episode,
            replacement_pairs=replacement_pairs_by_task[task],
            samples_per_ic=samples_per_ic,
            total_per_task=total_per_task)
        per_task[task] = res
        tot_n += res["n"]
        tot_s += res["n_success"]
        tot_plan_s += res.get("plan_time_s", 0.0)
        tot_plan_calls += res.get("plan_calls", 0)
    # bucket names the pool the ICs came from, so a train-split record is never
    # mistaken for a held-out one. Default --split dyn_val keeps the historic
    # "dyn_val_ic" string, so existing jsonl files stay comparable.
    rec = {"step": ckpt_step, "bucket": f"{split_pool}_ic",
           "success_rate": (tot_s / tot_n) if tot_n else 0.0,
           "n": tot_n, "n_success": tot_s, "per_task": per_task,
           # Per-episode outcomes, kept because the progress sidecar that held
           # them is deleted on completion. Two arms evaluated on the same ICs
           # are PAIRED, so retaining these turns an unpaired two-proportion
           # test (CI ~ +/-9pp at n=200) into McNemar on the discordant pairs.
           "results": list(state["results"])}
    if steps_per_phase is not None:
        rec["steps_per_phase"] = int(policy.model.cfg.steps_per_phase)
    if solver is not None:
        rec["solver"] = str(policy.model.cfg.solver)
    if tot_plan_calls:
        rec["mean_plan_s"] = round(tot_plan_s / tot_plan_calls, 4)
        rec["plan_calls"] = tot_plan_calls
    if samples_per_ic > 1:
        # Spread of the closed-loop SR under policy stochasticity: for each
        # policy-noise draw s, SR is measured over every IC/task, giving K
        # whole-eval SR point estimates whose mean equals the pooled
        # success_rate (equal per-sample counts). std is the +/-1 sigma band the
        # SR curve plots. Computed from state["results"] so a resumed eval folds
        # in episodes rolled out by earlier processes.
        _add_sample_spread(rec, state["results"], samples_per_ic)
    return rec, progress_path


def _add_sample_spread(rec: dict, results: list[dict], samples_per_ic: int) -> None:
    """Attach per-sample SR spread (overall + per-task) to an SR record."""
    def _by_sample(rows):
        counts: dict[int, list[int]] = {}
        for r in rows:
            s = int(r.get("sample", 0))
            c = counts.setdefault(s, [0, 0])
            c[0] += 1
            c[1] += int(r["success"])
        srs = [c[1] / c[0] for _, c in sorted(counts.items()) if c[0]]
        std = statistics.pstdev(srs) if len(srs) > 1 else 0.0
        return [round(x, 4) for x in srs], round(std, 4)

    srs, std = _by_sample(results)
    rec["samples_per_ic"] = int(samples_per_ic)
    rec["sr_by_sample"] = srs
    rec["sr_std"] = std
    per_task = rec.get("per_task", {})
    by_task: dict[str, list[dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    for task, rows in by_task.items():
        if task in per_task:
            t_srs, t_std = _by_sample(rows)
            per_task[task]["sr_by_sample"] = t_srs
            per_task[task]["sr_std"] = t_std


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="single checkpoint .pt")
    g.add_argument("--ckpt-dir", help="dir of ckpt_*.pt to sweep oldest->newest")
    ap.add_argument(
        "--data-root",
        help="local packed root used to reconstruct the held-out split; "
             "defaults to the checkpoint data config.")
    ap.add_argument("--robotwin-data", required=True,
                    help="RoboTwin data dir holding <task>/<cfg>/seed.txt.")
    ap.add_argument("--robotwin-repo", required=True,
                    help="RoboTwin repo root (cwd for env + task configs).")
    ap.add_argument("--task-config", default="modar_robotwin6")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="default: the checkpoint's own data.tasks")
    ap.add_argument("--split-tasks", nargs="+", default=None,
                    help="full task list the held-out split is computed over "
                         "(default: --tasks). Pass the checkpoint's whole task "
                         "list when sharding one eval across processes with "
                         "--tasks, so each shard sees the same val pool it "
                         "would in a single-process run.")
    ap.add_argument("--n-per-task", type=int, default=50)
    ap.add_argument(
        "--skip-unmapped-seeds", action="store_true",
        help="skip held-out demo IDs beyond seed.txt instead of failing. This "
             "supports enlarged packs whose first collection prefix retains "
             "the original verified seed map; requires at least --n-per-task "
             "mapped held-out demos per task.")
    ap.add_argument("--samples-per-ic", type=int, default=1,
                    help="rollouts per held-out IC. >1 replays each scene under "
                         "independent policy-noise seeds to measure the SR spread "
                         "(sr_std / sr_by_sample). Default 1 (historic behavior).")
    ap.add_argument("--action-val-ratio", type=float, default=None,
                    help="override held-out split (default: checkpoint cfg).")
    ap.add_argument("--dyn-val-ratio", type=float, default=None,
                    help="override held-out split (default: checkpoint cfg).")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="override split seed (default: checkpoint data.seed).")
    ap.add_argument("--split", dest="split_pool", default="dyn_val",
                    choices=["dyn_val", "dyn_train", "action_train", "action_val"],
                    help="training-time pool supplying the eval ICs. Default "
                         "dyn_val (held-out). Use action_train for in-distribution "
                         "train-split SR. Off by default; nothing wires it yet.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--exec-horizon", type=int, default=None,
        help="actions executed per replan (default: checkpoint "
             "model.action_horizon)")
    ap.add_argument("--steps-per-phase", type=int, default=None)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--step", type=int, default=None,
                    help="override logged step (single-ckpt, non-monotonic runs).")
    ap.add_argument("--generation-order", default=None,
                    help="comma-separated ModAR rollout override, e.g. "
                         "'tracks,dino,depth,image,action'. Default: checkpoint cfg.")
    ap.add_argument(
        "--infer-schedule",
        choices=["autoregressive", "unified", "futures_noise"],
        default=None,
        help="independent-model inference schedule override",
    )
    ap.add_argument("--out", required=True, help="JSONL output (appended).")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-id", default=None)
    ap.add_argument("--wandb-entity", default=None)
    args = ap.parse_args()

    # Resolve to absolute paths NOW: we os.chdir into the RoboTwin repo below, so
    # any relative --ckpt/--out/--data-root would otherwise break afterwards.
    if args.data_root is not None:
        args.data_root = os.path.abspath(args.data_root)
    args.out = os.path.abspath(args.out)
    if args.ckpt:
        ckpts = [os.path.abspath(args.ckpt)]
    else:
        ckpts = sorted(os.path.abspath(p)
                       for p in glob.glob(os.path.join(args.ckpt_dir, "ckpt_*.pt")))
        if not ckpts:
            raise SystemExit(f"no ckpt_*.pt under {args.ckpt_dir}")

    # Task list: default to the first checkpoint's own data.tasks.
    tasks = args.tasks
    if tasks is None:
        c0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        tasks = list(c0["cfg"]["data"]["tasks"])
    print(f"tasks: {tasks} | n_per_task={args.n_per_task} "
          f"| samples_per_ic={args.samples_per_ic} | ckpts={len(ckpts)}",
          flush=True)
    builtins.print = _without_robotwin_step_prints(builtins.print)

    # SAPIEN env needs to import `envs.<task>`, read ./task_config, and import
    # test_render from the RoboTwin repo root; then prime the renderer exactly as
    # script/eval_policy.py does before instantiating any task env.
    os.chdir(args.robotwin_repo)
    for p in ("./", "./policy", "./description/utils", "./script"):
        if p not in sys.path:
            sys.path.append(p)
    from test_render import Sapien_TEST  # noqa: E402
    Sapien_TEST()

    run = None
    if args.wandb_project or args.wandb_id:
        import wandb
        # wandb is a nice-to-have mirror; the jsonl/progress files on disk are the
        # SR eval's real output. A flaky network (wandb.init timing out) must NOT
        # abort the eval — otherwise a transient wandb outage produces no SR record
        # and the watcher retries forever. Init non-fatally (fail fast at 30s) and
        # simply skip wandb logging when it is unreachable.
        try:
            run = wandb.init(project=args.wandb_project, id=args.wandb_id,
                             entity=args.wandb_entity, resume="allow",
                             settings=wandb.Settings(init_timeout=30))
            # SR steps are ckpt steps, not trainer steps — keep a separate x-axis.
            wandb.define_metric("robotwin/ckpt_step")
            wandb.define_metric("robotwin/*", step_metric="robotwin/ckpt_step")
            wandb.define_metric("robotwin-per-task/*", step_metric="robotwin/ckpt_step")
        except Exception as e:
            print(f"[wandb] init failed ({type(e).__name__}: {e}); "
                  "continuing without wandb logging", flush=True)
            run = None

    generation_order_override = (args.generation_order.split(",") if args.generation_order else None)
    if generation_order_override is not None:
        print(f"generation_order override: {generation_order_override}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ckpt in ckpts:
        ckpt_step = infer_step(ckpt, args.step)
        order_key = ",".join(generation_order_override) if generation_order_override is not None else None
        if args.infer_schedule is not None:
            order_key = f"{order_key or 'default'}__{args.infer_schedule}"
        progress_path = _progress_path(out_path, ckpt_step, order_key)
        if _final_record_exists(out_path, ckpt_step, order_key):
            progress_path.unlink(missing_ok=True)
            print(f"[{os.path.basename(ckpt)}] step={ckpt_step} already complete; "
                  "skipping", flush=True)
            continue
        rec, progress_path = eval_checkpoint(
            ckpt, tasks, args.task_config, args.data_root, args.robotwin_data,
            args.n_per_task, args.device, args.exec_horizon, args.steps_per_phase,
            args.solver, not args.no_ema, args.step, args.action_val_ratio,
            args.dyn_val_ratio, args.split_seed, out_path,
            generation_order=generation_order_override, infer_schedule=args.infer_schedule,
            split_pool=args.split_pool,
            samples_per_ic=args.samples_per_ic, split_tasks=args.split_tasks,
            skip_unmapped_seeds=args.skip_unmapped_seeds)
        progress_state = json.loads(progress_path.read_text())
        rec["eval_s"] = round(float(progress_state["eval_s"]), 1)
        rec["ckpt"] = os.path.basename(ckpt)
        rec.update(read_sample_progress(ckpt))
        if generation_order_override is not None:
            rec["generation_order"] = list(generation_order_override)
        if args.infer_schedule is not None:
            rec["infer_schedule"] = args.infer_schedule
        if order_key is not None:
            rec["order_key"] = order_key
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        progress_path.unlink()
        order_tag = f" order={rec.get('order_key')}" if "order_key" in rec else ""
        print(f"[{rec['ckpt']}] step={rec['step']}{order_tag} "
              f"SR={rec['success_rate']:.3f} "
              f"({rec['n_success']}/{rec['n']}) in {rec['eval_s']}s", flush=True)
        if run is not None:
            log = {"robotwin/ckpt_step": rec["step"],
                   "robotwin/dyn_val_ic/success_rate": rec["success_rate"]}
            if "mean_plan_s" in rec:
                log["robotwin/mean_plan_s"] = rec["mean_plan_s"]
            for task, d in rec["per_task"].items():
                log[f"robotwin-per-task/{task}"] = d["success_rate"]
            # The resumed training run's internal W&B step is already beyond
            # early checkpoint numbers. Let W&B append at its next internal
            # step and use robotwin/ckpt_step as the declared SR x-axis.
            # The record is already fsynced to disk above, so a flaky-network
            # log failure must not lose it or crash the remaining checkpoints.
            try:
                run.log(log)
            except Exception as e:
                print(f"[wandb] log failed ({type(e).__name__}: {e}); "
                      "record already on disk", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
