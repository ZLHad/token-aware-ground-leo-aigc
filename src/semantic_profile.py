"""Request-level task-oriented semantic profile (main.tex §II-D, eq. 8-9).

Implements the two-parameter rate-distortion family

    D^sem(eta; xi) = delta * (1 - eta)**theta              # eq. (8)
    S^sem(eta; xi) = 1 - D^sem(eta; xi)
    Q^eff(eta, x)  = Q^mdl(x) * S^sem(eta; xi)             # eq. (9)

and the marginals used in Theorem 1 / Theorem 2:

    dS/deta        = delta * theta * (1-eta)**(theta-1)
    Q^mdl(x)       = (1-x)*Q_loc + x*Q_sat                 # main.tex §II-D para 1
"""
from __future__ import annotations

import numpy as np

from config import Q_SAT, Q_LOC


def Q_mdl(x: int | np.ndarray) -> float | np.ndarray:
    """Model-capability bound. x=1 -> sat, x=0 -> local."""
    return (1.0 - x) * Q_LOC + x * Q_SAT


def D_sem(eta: float | np.ndarray, delta: float, theta: float) -> float | np.ndarray:
    """Semantic distortion D^sem(eta; xi) = delta * (1-eta)^theta."""
    one_minus = np.clip(1.0 - eta, 0.0, 1.0)
    return delta * np.power(one_minus, theta)


def S_sem(eta: float | np.ndarray, delta: float, theta: float) -> float | np.ndarray:
    """Semantic sufficiency S^sem = 1 - D^sem."""
    return 1.0 - D_sem(eta, delta, theta)


def dS_sem_deta(eta: float | np.ndarray, delta: float, theta: float) -> float | np.ndarray:
    """dS^sem/deta = delta * theta * (1-eta)^(theta-1).

    Positive for eta in (0, 1), so S^sem is monotonically non-decreasing.
    """
    one_minus = np.clip(1.0 - eta, 0.0, 1.0)
    if np.isscalar(eta):
        if one_minus <= 0.0:
            return 0.0
        return delta * theta * one_minus ** (theta - 1.0)
    out = np.zeros_like(np.asarray(eta, dtype=float))
    mask = one_minus > 0.0
    out[mask] = delta * theta * one_minus[mask] ** (theta - 1.0)
    return out


def Q_eff(eta: float | np.ndarray, x: int,
          delta: float, theta: float) -> float | np.ndarray:
    """Effective generation quality Q^eff(eta, x; xi) (eq. 9)."""
    return Q_mdl(x) * S_sem(eta, delta, theta)


def quality_marginal_benefit(eta: float, x: int,
                              delta: float, theta: float,
                              w_q: float) -> float:
    """The Q(eta; xi) function defined just below main.tex (eq. 11):

        Q(eta; xi) = w_q * Q^sat * delta * theta * (1-eta)^(theta-1)
                     / [1 + Q^sat * S^sem(eta; xi)]

    Used inside the eta-stationarity equation. Note that the denominator
    is the derivative of w_q * log(1 + Q^eff) with respect to Q^eff and
    the numerator is w_q * Q^mdl * dS/deta.
    """
    Qm = Q_mdl(x)
    Ssem = S_sem(eta, delta, theta)
    dS = dS_sem_deta(eta, delta, theta)
    return w_q * Qm * dS / (1.0 + Qm * Ssem)


__all__ = [
    "Q_mdl", "D_sem", "S_sem", "dS_sem_deta", "Q_eff",
    "quality_marginal_benefit",
]
