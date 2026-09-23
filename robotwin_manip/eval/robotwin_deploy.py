"""Closed-loop WAM policy bridge for RoboTwin (SAPIEN).

Runs a Modality-Forcing WAM checkpoint as a RoboTwin policy: at each replan it
reproduces the training-time observation pipeline (head-camera RGB -> DINOv2
patch tokens, mm depth -> metric -> log+z-score, 14-D dual-arm joint proprio),
calls ``model.sample`` for a 16-step absolute-joint-target action chunk, and
executes the first ``action_exec_horizon`` targets via
``Base_Task.take_action(action, action_type='qpos')``.

Two entry points share this file:

  * module hooks ``get_model`` / ``reset_model`` / ``eval`` implement RoboTwin's
    ``script/eval_policy.py`` policy contract (the ``policy/WAM`` shim re-exports
    them). That path expert-gates seeds -- we do NOT use it for the SR sweep.
  * ``WAMPolicy`` is imported directly by ``robotwin_manip/eval/run_sr.py``,
    which drives the env over pre-verified held-out seeds without the planner.

The obs preprocessing MUST match ``robotwin_manip/datagen/robotwin_to_wam.py``
exactly: the complete native 320x240 frame is resized without cropping to
224x168 (width x height), yielding a 16x12 DINO/depth/track grid.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from util.depth_utils import normalize_depth_maps  # noqa: E402
from util.modality_forcing.data import build_task_to_id  # noqa: E402
from util.modality_forcing.rollout import (  # noqa: E402
    DinoEncoder, build_mf_from_checkpoint)
from robotwin_manip.datagen.robotwin_wam_io import (  # noqa: E402
    depth_mm_to_metric,
    resize_rgb,
)

IMAGE_HEIGHT = 168
IMAGE_WIDTH = 224
GRID_HEIGHT = 12
GRID_WIDTH = 16


def seed_inference_rng(seed: int) -> None:
    """Pin CPU/CUDA RNG to the IC seed for one closed-loop episode.

    ``sample()`` draws ``torch.randn`` noise on CUDA, so the policy is
    stochastic; without pinning, an IC's action noise depends on how much RNG
    was consumed earlier in the process (position in the eval loop). Reseeding
    both generators to the IC seed *after* ``setup_demo`` and *before* the first
    ``plan()`` makes the sampled actions reproducible per IC, independent of
    loop position. (This only pins the *policy*; the dominant eval artifact was
    SAPIEN physics state accumulating across a reused env -- see
    ``run_sr.run_task``, which rebuilds the env every episode.)
    """
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WAMPolicy:
    """Stateless (per-call replanning) WAM policy over RoboTwin observations."""

    def __init__(self, ckpt, device="cuda", exec_horizon=None, camera="head_camera",
                 use_ema=True, steps_per_phase=None, solver=None,
                 depth_mean=None, depth_std=None, depth_max_m=None,
                 depth_norm_mode=None, generation_order=None, infer_schedule=None,
                 image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH):
        self.device = torch.device(
            device if (str(device).startswith("cpu") or torch.cuda.is_available())
            else "cpu")
        self.model, self.mfcfg, self.stats, self.data_cfg = build_mf_from_checkpoint(
            ckpt, self.device, use_ema=use_ema)
        if steps_per_phase is not None:
            self.model.cfg.steps_per_phase = int(steps_per_phase)
        if solver is not None:
            self.model.cfg.solver = str(solver)
        if infer_schedule is not None:
            if self.model.cfg.schedule_mode != "independent":
                raise ValueError(
                    "infer_schedule override is only valid for independent models")
            if infer_schedule not in {"autoregressive", "unified", "futures_noise"}:
                raise ValueError(f"unsupported infer_schedule {infer_schedule!r}")
            self.model.cfg.infer_schedule = str(infer_schedule)
            print(
                f"[WAMPolicy] infer_schedule override -> {infer_schedule}",
                flush=True,
            )
        # Override the ModAR deployment rollout order. For example, ["action"] does
        # direct action generation from clean history and proprio only. By default,
        # the checkpoint's full generation_order supplies generated futures first.
        if generation_order is not None:
            order = list(generation_order)
            if "action" not in order:
                raise ValueError(f"generation_order must include 'action', got {order}")
            self.model.cfg.generation_order = order
            print(f"[WAMPolicy] generation_order override -> {order}", flush=True)

        self.camera = str(camera)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        expected_geometry = (
            IMAGE_HEIGHT, IMAGE_WIDTH, GRID_HEIGHT, GRID_WIDTH)
        checkpoint_geometry = (
            self.mfcfg.spatial_height, self.mfcfg.spatial_width,
            self.mfcfg.grid_h, self.mfcfg.grid_w)
        requested_geometry = (
            self.image_height, self.image_width, GRID_HEIGHT, GRID_WIDTH)
        if checkpoint_geometry != expected_geometry:
            raise ValueError(
                f"RoboTwin checkpoint geometry {checkpoint_geometry} is obsolete; "
                f"expected {expected_geometry}")
        if requested_geometry != expected_geometry:
            raise ValueError(
                f"RoboTwin rollout geometry {requested_geometry} != "
                f"{expected_geometry}")
        # By default, execute the full action chunk configured at training time.
        # This keeps model.action_horizon in the checkpoint as the single source
        # of truth while retaining an explicit override for diagnostics.
        self.exec_horizon = (
            int(self.mfcfg.action_horizon)
            if exec_horizon is None else int(exec_horizon))
        if self.exec_horizon > self.mfcfg.action_horizon:
            raise ValueError(
                f"exec_horizon ({self.exec_horizon}) > action_horizon "
                f"({self.mfcfg.action_horizon})")

        self.dino = DinoEncoder(
            self.device, image_size=(self.image_height, self.image_width))
        # Match training's index space EXACTLY: build_modality_loaders_cotrain uses
        # build_task_to_id(list(task_vocab)) in config-list order (NOT sorted), so
        # reuse the same map here rather than re-deriving it.
        vocab = self.data_cfg.get("task_vocab") or self.data_cfg.get("tasks")
        self.task_to_id = (build_task_to_id(list(vocab))
                           if (vocab and self.mfcfg.n_tasks > 0) else None)
        self.task_id_tensor = None

        # Depth normalization params: checkpoint's data config unless overridden.
        self.depth_mean = float(depth_mean if depth_mean is not None
                                else self.data_cfg["depth_mean"])
        self.depth_std = float(depth_std if depth_std is not None
                               else self.data_cfg["depth_std"])
        self.depth_max_m = float(depth_max_m if depth_max_m is not None
                                 else self.data_cfg.get("depth_max_m", 10.0))
        self.depth_norm_mode = str(depth_norm_mode if depth_norm_mode is not None
                                   else self.data_cfg.get("depth_norm_mode", "log"))

        self.a_mean = self.stats["action_mean"].to(self.device).float()
        self.a_std = self.stats["action_std"].to(self.device).float()
        self.p_mean = self.stats["proprio_mean"].to(self.device).float()
        self.p_std = self.stats["proprio_std"].to(self.device).float()
        self.d_mean = float(self.stats["dino_mean"])
        self.d_std = float(self.stats["dino_std"])

    def set_task(self, task_name):
        """Bind the task-embedding index for the multi-task model."""
        if self.task_to_id is not None and self.mfcfg.n_tasks > 0:
            if task_name not in self.task_to_id:
                raise KeyError(
                    f"task {task_name!r} not in checkpoint task vocab "
                    f"{sorted(self.task_to_id)}")
            self.task_id_tensor = torch.tensor(
                [self.task_to_id[task_name]], dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def _features(self, observation):
        """RoboTwin obs -> DINO, depth, normalized RGB, and proprio tensors."""
        cam = observation["observation"][self.camera]
        rgb = resize_rgb(
            np.asarray(cam["rgb"]), self.image_height, self.image_width)
        dm, _ = depth_mm_to_metric(
            np.asarray(cam["depth"]), self.image_height, self.image_width,
            self.depth_max_m)
        dino = (self.dino(rgb) - self.d_mean) / self.d_std
        depth = normalize_depth_maps(
            torch.from_numpy(dm).float().to(self.device), self.depth_mean,
            self.depth_std, max_depth_m=self.depth_max_m, mode=self.depth_norm_mode)
        image = (
            torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float()
            / 127.5 - 1.0
        )
        proprio = torch.from_numpy(
            np.asarray(observation["joint_action"]["vector"], np.float32)
        ).to(self.device)
        proprio = (proprio - self.p_mean) / self.p_std
        return dino, depth, image, proprio

    @torch.inference_mode()
    def plan(self, observation, return_streams: bool = False):
        """Return the next ``exec_horizon`` absolute 14-D joint targets (np).

        With ``return_streams=True``, also returns a plan dict for viz:
        conditioning RGB plus any predicted future streams the model
        generated (dino / depth / tracks / image).
        """
        dino, depth, image, proprio = self._features(observation)
        # obs_history is 1 for the RoboTwin configs; replicate the current frame
        # if a config ever sets H>1 (no persistent stride buffer is kept).
        h = int(self.mfcfg.obs_history)
        dino_hist = dino.unsqueeze(0).unsqueeze(0).expand(1, h, -1, -1)
        depth_hist = depth.unsqueeze(0).unsqueeze(0).expand(1, h, -1, -1)
        image_hist = image.unsqueeze(0).unsqueeze(0).expand(1, h, -1, -1, -1)
        proprio_b = proprio.unsqueeze(0)
        samp = self.model.sample(dino_hist, depth_hist, proprio_b,
                                 self.task_id_tensor, image_hist=image_hist)
        actions = samp["actions"][0] * self.a_std + self.a_mean          # (A,14)
        acts = actions[:self.exec_horizon].cpu().numpy().astype(np.float32)
        if not return_streams:
            return acts
        cam = observation["observation"][self.camera]
        plan_rgb = resize_rgb(
            np.asarray(cam["rgb"]), self.image_height, self.image_width)
        plan = {"rgb": plan_rgb,
                "dino_now": dino.detach().cpu(),
                "depth_now": depth.detach().cpu(),
                "qpos_now": np.asarray(observation["joint_action"]["vector"],
                                       np.float32),
                "generation_order": list(self.model.cfg.generation_order),
                "dino_mean": self.d_mean,
                "dino_std": self.d_std}
        if "dino" in samp:
            plan["dino_fut"] = samp["dino"][0].detach().cpu()
        if "depth_maps" in samp:
            plan["depth_fut"] = samp["depth_maps"][0].detach().cpu()
        if "point_tracks" in samp:
            plan["tracks_fut"] = samp["point_tracks"][0].detach().cpu()
        if "images" in samp:
            plan["image_fut"] = samp["images"][0].detach().cpu()
        return acts, plan

    @torch.inference_mode()
    def plan_oracle(self, observation, oracle_dino=None, oracle_depth=None,
                    oracle_tracks=None, force=None):
        """Action chunk with the world-model futures TEACHER-FORCED to oracle.

        This is the closed-loop analogue of Phase-1's open-loop oracle-action
        ceiling. Instead of the model GENERATING its obs futures (dino / depth /
        tracks) and conditioning the action on those self-predictions (as
        ``plan``/``sample`` do), we inject the *expert demonstrator's* actual
        future dynamics -- harvested by rolling the reactive procedural expert
        forward from the current sim state in a sandbox -- as clean context, and
        generate ONLY the action head on top of them. It answers: given perfect
        (expert-sourced) future conditioning, what is this action expert's SR
        ceiling? Comparing dino-only vs 3mod under this identical, perfect
        conditioning isolates the action head from the world model.

        The oracle futures are supplied in the model's own clean-target spaces
        (exactly what ``_causal_future`` reads):
          * ``oracle_dino``   : (n_tgt, N, D)  z-scored DINO patch tokens
          * ``oracle_depth``  : (n_tgt, H, W)  normalized depth MAPS (patchified
                                internally by ``_prep_data``)
          * ``oracle_tracks`` : (n_fut, N, 3)  (x,y,vis) in ``track_pred_mode``
                                space (delta vs anchor if delta-mode)
        where ``n_tgt == len(target_local[name])`` for grid+history modalities
        and ``n_fut == obs_future`` for future-only tracks (both == 2 here). Only
        the modalities that appear in ``generation_order[:-1]`` are consumed; pass
        ``None`` for the rest. Returns the first ``exec_horizon`` absolute 14-D
        joint targets (np.float32), same contract as ``plan``.

        ``force`` selects WHICH world-model modalities are teacher-forced. None
        (the default) forces all of ``generation_order[:-1]`` -- the full oracle.
        A strict subset is a PARTIAL-ORACLE rung: the listed modalities get the
        expert's futures, the rest are ODE-generated by the model exactly as in
        ``plan()``, and the cascade still runs in ``generation_order``, so a
        generated modality conditions on whatever (oracle or generated) came
        before it. Sweeping subsets attributes the oracle-vs-self-gen SR gap to
        individual modalities. Oracle tensors are required only for the forced
        modalities.
        """
        m = self.model
        order = list(m.cfg.generation_order)
        if order[-1] != "action":
            raise ValueError(f"plan_oracle target must be 'action', got {order}")
        dyn = order[:-1]
        forced = set(dyn) if force is None else set(force)
        if not forced <= set(dyn):
            raise ValueError(
                f"plan_oracle force={sorted(forced - set(dyn))} are not "
                f"generated modalities {dyn}")
        dino, depth, image, proprio = self._features(observation)
        h = int(self.mfcfg.obs_history)
        Fp = int(m.obs_present)
        dev = self.device

        # History maps (clean context frames) -- identical to plan().
        dino_hist = dino.unsqueeze(0).unsqueeze(0).expand(1, h, -1, -1)
        depth_hist = depth.unsqueeze(0).unsqueeze(0).expand(1, h, -1, -1)
        image_hist = image.unsqueeze(0).unsqueeze(0).expand(1, h, -1, -1, -1)
        proprio_b = proprio.unsqueeze(0)

        # dino clean-target tensor (1,Fp,N,D): fill target_local frames with the
        # oracle futures; history slot is unused by _causal_future for grids but
        # filled with the current frame for cleanliness. A modality that is NOT
        # forced keeps the replicated current frame as a placeholder -- it is
        # ODE-generated, so _causal_future never reads its clean value.
        dino_full = None
        if "dino" in m.present_names:
            dino_full = dino.unsqueeze(0).unsqueeze(0).expand(1, Fp, -1, -1).clone()
            if "dino" in forced:
                tl = m.target_local["dino"]
                if oracle_dino is None or oracle_dino.shape[0] != len(tl):
                    raise ValueError(
                        f"oracle_dino must be ({len(tl)},N,D); got "
                        f"{None if oracle_dino is None else tuple(oracle_dino.shape)}")
                dino_full[:, tl] = oracle_dino.to(dev).unsqueeze(0)

        # depth clean-target MAPS (1,Fp,H,W): same layout.
        depth_full = None
        if "depth" in m.present_names:
            depth_full = depth.unsqueeze(0).unsqueeze(0).expand(1, Fp, -1, -1).clone()
            if "depth" in forced:
                tl = m.target_local["depth"]
                if oracle_depth is None or oracle_depth.shape[0] != len(tl):
                    raise ValueError(
                        f"oracle_depth must be ({len(tl)},H,W); got "
                        f"{None if oracle_depth is None else tuple(oracle_depth.shape)}")
                depth_full[:, tl] = oracle_depth.to(dev).unsqueeze(0)

        # tracks are future-only: (1,n_fut,N,3) consumed whole by _causal_future.
        # _prep_data requires the tensor whenever the modality is present, so an
        # unforced track stream still needs a (never-read) placeholder.
        tracks_full = None
        if "tracks" in m.present_names:
            n_fut = len(m.target_local["tracks"])
            if "tracks" in forced:
                if oracle_tracks is None or oracle_tracks.shape[0] != n_fut:
                    raise ValueError(
                        f"oracle_tracks must be ({n_fut},N,3); got "
                        f"{None if oracle_tracks is None else tuple(oracle_tracks.shape)}")
                tracks_full = oracle_tracks.to(dev).unsqueeze(0)
            else:
                tracks_full = torch.zeros(
                    1, n_fut, m.cfg.n_track_points, m.cfg.track_dim, device=dev)

        images_full = None  # neither target ckpt has the image modality
        if "image" in m.present_names:
            images_full = image.unsqueeze(0).unsqueeze(0).expand(1, Fp, -1, -1, -1).clone()

        actions_dummy = torch.zeros(
            1, int(self.mfcfg.action_horizon), int(m.cfg.action_dim), device=dev)

        # sample_causal_mixed runs the SAME cascade as plan()/sample(), only with
        # the forced modalities swapped in for their clean oracle futures, so an
        # oracle rung and its self-gen baseline differ in nothing else.
        hist = m._build_history(dino_hist, depth_hist, image_hist)
        data = m._prep_data(dino_full, depth_full, actions_dummy, images_full,
                            tracks_full)
        out = m.sample_causal_mixed(hist, proprio_b, data,
                                    task_id=self.task_id_tensor, order=order,
                                    gt_set=forced)
        actions = out["actions"][0] * self.a_std + self.a_mean
        return actions[:self.exec_horizon].cpu().numpy().astype(np.float32)


# --- RoboTwin script/eval_policy.py plugin hooks (expert-gated path) ----------

def get_model(usr_args):
    policy = WAMPolicy(
        ckpt=usr_args["wam_checkpoint"],
        device=usr_args.get("device", "cuda"),
        exec_horizon=usr_args.get("action_exec_horizon"),
        camera=usr_args.get("camera", "head_camera"),
        image_height=int(usr_args.get("image_height", IMAGE_HEIGHT)),
        image_width=int(usr_args.get("image_width", IMAGE_WIDTH)),
        use_ema=not bool(usr_args.get("no_ema", False)),
        steps_per_phase=usr_args.get("steps_per_phase"),
        solver=usr_args.get("solver"),
        depth_mean=usr_args.get("depth_mean"),
        depth_std=usr_args.get("depth_std"),
        depth_max_m=usr_args.get("depth_max_m"),
        depth_norm_mode=usr_args.get("depth_norm_mode"))
    if usr_args.get("task_name"):
        policy.set_task(usr_args["task_name"])
    return policy


def reset_model(model):  # noqa: ARG001 - stateless; nothing to reset per episode
    return None


def eval(TASK_ENV, model, observation):
    """Execute one replan chunk; RoboTwin's loop refreshes obs and recalls this."""
    for action in model.plan(observation):
        TASK_ENV.take_action(action, action_type="qpos")
        if getattr(TASK_ENV, "eval_success", False):
            return
