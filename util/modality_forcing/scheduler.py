"""Flow-matching time schedules shared by Modality-Forcing WAM modes.

ModAR training supervises fixed-order query cuts conditioned on a clean teacher
prefix. Unified training shares one time across all targets.
Independent training supervises all present targets with separately sampled
uniform times. Disjoint training samples one active target via ``p_*`` and gives
it a per-modality logit-normal time. Inference uses the ODE helper below to
integrate a selected modality from noise (t=0) to clean data (t=1).
"""
from __future__ import annotations

from typing import Callable

import torch

from util.modality_forcing.config import MFConfig


class AutoregressiveScheduler:
    def __init__(self, cfg: MFConfig):
        self.cfg = cfg
        # Ordered active modalities + per-modality tables, driven by cfg.
        self.modalities = tuple(cfg.modalities)
        self.order = {m: i for i, m in enumerate(self.modalities)}
        self.has_history = {s.name: s.has_history for s in cfg.modality_specs()}
        self.p = torch.tensor([cfg.p_of(m) for m in self.modalities])
        self.logitnorm = {m: cfg.logitnorm_of(m) for m in self.modalities}
        self.t_clip = {m: cfg.t_clip_of(m) for m in self.modalities}

    def _logitnormal(self, modality: str, n: int, device) -> torch.Tensor:
        mu, sigma = self.logitnorm[modality]
        return torch.sigmoid(mu + sigma * torch.randn(n, device=device))

    def _mix_early_t(self, t: torch.Tensor, n: int, device) -> torch.Tensor:
        """LF sec 5.2 low-t mixture: with prob ``early_t_frac`` replace the
        (logit-normal) active time by U[0, early_t_max]. The logit-normal puts
        ~no mass near t=0, but that is exactly the generative regime (denoising
        from pure noise) that decides global structure / object placement -- so
        EVERY actively-generated modality needs it. Applied uniformly across all
        schedule modes (action_only, modar, unified, disjoint);
        'independent' already samples t ~ U[0,1] so it is covered inherently."""
        if self.cfg.early_t_frac <= 0:
            return t
        use_unif = torch.rand(n, device=device) < self.cfg.early_t_frac
        unif = self.cfg.early_t_max * torch.rand(n, device=device)
        return torch.where(use_unif, unif, t)

    def sample_active_time(self, modality: str, n: int, device) -> torch.Tensor:
        """Time for an actively-generated modality: logit-normal + low-t mixture."""
        return self._mix_early_t(self._logitnormal(modality, n, device), n, device)

    def sample_unified(self, n: int, device) -> torch.Tensor:
        """A single shared diffusion time (schedule_mode == 'unified')."""
        mu, sigma = self.cfg.logitnorm_unified
        t = torch.sigmoid(mu + sigma * torch.randn(n, device=device))
        return self._mix_early_t(t, n, device)

    def sample_action(self, n: int, device) -> torch.Tensor:
        """Action-modality time (schedule_mode == 'action_only')."""
        return self.sample_active_time("action", n, device)

    def sample_independent(self, n: int, device) -> dict:
        """Independent per-modality diffusion times (schedule_mode == 'independent').

        Following Latent Forcing's Multi-Schedule Model (sec 4.5): each modality's
        time is drawn UNIFORMLY on [0,1] and independently. A product of
        logit-normals would starve autoregressive inference trajectories of training
        signal; uniform covers every time combination. Returns {name: (n,)}."""
        return {m: torch.rand(n, device=device) for m in self.modalities}

    def sample_disjoint(self, batch_size: int, device, allowed=None,
                        sensor_drop: float = 0.0) -> dict:
        """Choose one active modality for a clean-history-only disjoint pass."""
        B = batch_size
        cand = [m for m in self.modalities if (allowed is None or m in allowed)]
        p = torch.tensor([self.p[self.order[m]] for m in cand])
        active = cand[int(torch.multinomial(p / p.sum(), 1).item())]

        states = {}
        for name in self.modalities:
            if name == active:
                states[name] = "active"
            elif self.has_history[name]:
                if torch.rand(()).item() < sensor_drop:
                    states[name] = "absent"
                else:
                    states[name] = "hist_only"
            else:
                states[name] = "absent"

        times = {}
        for name in self.modalities:
            st = states[name]
            if st == "active":
                times[name] = self.sample_active_time(name, B, device)
            else:
                times[name] = torch.ones(B, device=device)
        return {"active": active, "states": states, "times": times}

    def loss_weight(self, modality: str, t: torch.Tensor) -> torch.Tensor:
        """v-loss weighting 1 / max(1-t, t_clip)^2 (JiT Eq. 5), per sample."""
        clip = self.t_clip[modality]
        denom = torch.clamp(1.0 - t, min=clip)
        return 1.0 / (denom * denom)

    @staticmethod
    def add_noise(x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
                  ) -> torch.Tensor:
        """z = t*x + (1-t)*noise. t broadcasts over trailing dims of x."""
        while t.dim() < x.dim():
            t = t.unsqueeze(-1)
        return t * x + (1.0 - t) * noise

    def ode_solve(self, denoise_fn: Callable[[torch.Tensor, float], torch.Tensor],
                  z0: torch.Tensor, n_steps: int, solver: str = "heun",
                  eps: float = 1e-3, enable_grad: bool = False) -> torch.Tensor:
        """Integrate one modality's flow from t=0 (z0=noise) to t=1 (data).

        denoise_fn(z, t_scalar) -> x_pred (clean estimate) for the active modality.
        Uses the x-prediction -> velocity identity v = (x_pred - z)/(1-t).

        ``enable_grad`` keeps the rollout in the autograd graph (for the
        differentiable self-forcing path); the default disables grad (inference /
        stop-gradient generation), matching the previous @torch.no_grad() behavior.
        """
        ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with ctx:
            z = z0
            ts = torch.linspace(0.0, 1.0, n_steps + 1, device=z0.device)
            for i in range(n_steps):
                t0 = float(ts[i])
                t1 = float(ts[i + 1])
                dt = t1 - t0
                x0 = denoise_fn(z, t0)
                v0 = (x0 - z) / max(1.0 - t0, eps)
                z_e = z + dt * v0
                # On the final step (t1 -> 1) the x-pred velocity v=(x-z)/(1-t) is
                # singular, so the Heun corrector at t1 divides by ~0 and explodes.
                # The Euler predictor already lands on the clean x-estimate there, so
                # skip the corrector when 1-t1 is degenerate.
                if solver == "euler" or (1.0 - t1) <= eps:
                    z = z_e
                else:  # heun
                    x1 = denoise_fn(z_e, t1)
                    v1 = (x1 - z_e) / (1.0 - t1)
                    z = z + dt * 0.5 * (v0 + v1)
            return z
