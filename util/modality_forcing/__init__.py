"""Modality-autoregressive world-action model.

A flow-matching DiT that generates future point tracks, DINO features, depth,
and RGB, then robot actions, conditioned on clean observation history.
"""
from util.modality_forcing.config import MFConfig
from util.modality_forcing.model import ModalityForcingWAM
from util.modality_forcing.scheduler import AutoregressiveScheduler

__all__ = ["MFConfig", "ModalityForcingWAM", "AutoregressiveScheduler"]
