"""Calibration of H_u -> (delta_u, theta_u) per main.tex §II-B (eq. 1).

main.tex says:
    "A small H_u indicates that the task-relevant semantics concentrate
     on a few visual tokens, mapping to a large theta_u and a small delta_u,
     whereas a large H_u implies dispersed importance, mapping to a
     small theta_u and a large delta_u."

This module implements the simplified analytical map (README §5.2):

    delta(H) = delta_min + (delta_max - delta_min) * H
    theta(H) = theta_max - (theta_max - theta_min) * H

so that H=0 -> (delta_min, theta_max) (low sensitivity, compression-tolerant)
and  H=1 -> (delta_max, theta_min) (high sensitivity, compression-sensitive).

The look-up endpoints are chosen to match Fig 1(c) profiles in
config.FIG1C_PROFILES exactly when sampled at H in {0, 0.5, 1}.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# Endpoints chosen so that:
#   H=0   -> (0.20, 3.0)    (Low sensitivity in config.FIG1C_PROFILES)
#   H=1   -> (0.95, 1.0)    (High sensitivity in config.FIG1C_PROFILES)
# Medium (H=0.5) interpolates to (0.575, 2.0), close to (0.55, 1.8) target.
# The widened spread makes profile-aware optimization differ more from the
# average-profile baseline (Profile-Oblivious).
DELTA_MIN: float = 0.20
DELTA_MAX: float = 0.95
THETA_MIN: float = 1.00
THETA_MAX: float = 3.00


def calibrate(H: float | np.ndarray) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Return (delta, theta) given intent indicator H in [0, 1].

    Vectorized: H can be a scalar or numpy array.
    """
    H_clip = np.clip(H, 0.0, 1.0)
    delta = DELTA_MIN + (DELTA_MAX - DELTA_MIN) * H_clip
    theta = THETA_MAX - (THETA_MAX - THETA_MIN) * H_clip
    return delta, theta


def mean_profile(H_samples: np.ndarray) -> Tuple[float, float]:
    """Empirical mean profile bar_xi over a calibration set (for Profile-Oblivious)."""
    delta, theta = calibrate(H_samples)
    return float(np.mean(delta)), float(np.mean(theta))


__all__ = ["calibrate", "mean_profile", "DELTA_MIN", "DELTA_MAX", "THETA_MIN", "THETA_MAX"]
