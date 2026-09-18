"""KKT closed-form inner solution + 1-D bisection for eta*.

Implements Theorem 1 of main.tex (§III-B). For a user u with a fixed
(x_u, s_u) pair and strictly positive dual variables (mu^up, mu^dn, nu),
the inner subproblem max_{eta, B^up, B^dn, f} L_u admits:

    B^{dn,*}  = sqrt(w_1 * kappa_out / (mu_dn * log2(1+gamma_S2G)))            # eq. (13) -- max-TPOT, sat-dominant
    B^{up,*}(eta) = sqrt(w_1 * kappa_in * n_tilde(eta) / (mu_up * log2(1+gamma_G2S)))  # eq. (14)
    f^*(eta)  = sqrt((Phi_pf(eta) + Phi_dc(eta)) / nu)                         # eq. (15)
    eta^*     : unique root in [eta_min, 1] of Q(eta) = C(eta)                 # eq. (16)

where (derived by envelope theorem after substituting B^up*, f* into the
eta-stationarity of L_u):

    Q(eta; xi) = w_q * Q^mdl * delta * theta * (1-eta)^(theta-1)
                 / (1 + Q^mdl * S^sem(eta; xi))

    C(eta; nu, mu_up) = w_1 * n_vis * ( alpha_up / B^{up,*}(eta) + tau_up + zeta_pf / f^*(eta) )
                        + w_2 * n_vis * beta_1 / f^*(eta)

with the shorthand
    alpha_up = kappa_in / log2(1 + gamma_G2S)
    alpha_dn = kappa_out / log2(1 + gamma_S2G)
    tau_up   = kappa_in / R_ISL_eff(s_0, s_u)
    n_tilde(eta) = n_txt + eta * n_vis
    Phi_pf(eta)  = w_1 * zeta_pf * n_tilde(eta)
    Phi_dc(eta)  = w_2 * (beta_0 + beta_1 * (n_tilde(eta) + (n_out - 1) / 2))

For the local branch (x_u = 0), the user is served by f_loc with no shared
sat resources, so eta_loc^* solves:

    Q^loc(eta; xi) = w_1 * n_vis * zeta_pf_loc / f_loc + w_2 * n_vis * beta_1_loc / f_loc

(independent of dual variables and target satellite).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import brentq

import config as C
from src.data_gen import Request, Topology
from src.semantic_profile import (
    Q_mdl, S_sem, dS_sem_deta, Q_eff, quality_marginal_benefit,
)


# ---------------------------------------------------------------------------
#  Result holder
# ---------------------------------------------------------------------------
@dataclass
class InnerResult:
    """Solution of one (u, s, x) inner subproblem."""
    eta: float
    B_up: float
    B_dn: float
    f: float
    L: float           # value of Lagrangian L_u at the optimum
    QoE: float
    TTFT: float
    TPOT: float
    Q_eff: float


# ---------------------------------------------------------------------------
#  Per-user constants (independent of decisions)
# ---------------------------------------------------------------------------
def alpha_up(req: Request) -> float:
    """alpha_up = kappa_in / log2(1 + gamma_G2S)."""
    return C.KAPPA_IN / math.log2(1.0 + req.gamma_G2S)


def alpha_dn(req: Request) -> float:
    """alpha_dn = kappa_out / log2(1 + gamma_S2G)."""
    return C.KAPPA_OUT / math.log2(1.0 + req.gamma_S2G)


def tau_up(req: Request, topo: Topology, s_u: int) -> float:
    """tau_up = kappa_in / R_ISL_eff(s_0, s_u). 0 if s_u == s_0(u)."""
    if s_u == req.s0:
        return 0.0
    R = topo.R_ISL_eff[req.s0, s_u]
    if math.isinf(R) or R <= 0.0:
        return 0.0
    return C.KAPPA_IN / R


def tau_dn(req: Request, topo: Topology, s_u: int) -> float:
    """tau_dn = kappa_out / R_ISL_eff(s_u, s_0). 0 if s_u == s_0(u). Used in TTFT only."""
    if s_u == req.s0:
        return 0.0
    R = topo.R_ISL_eff[s_u, req.s0]
    if math.isinf(R) or R <= 0.0:
        return 0.0
    return C.KAPPA_OUT / R


def n_tilde(req: Request, eta: float) -> float:
    """n_tilde(eta) = n_txt + eta * n_vis  (eq. 2 in main.tex)."""
    return req.n_txt + eta * req.n_vis


def n_out_eff(req: Request) -> float:
    """(n_out - 1) / 2 — appears inside the decode TPOT."""
    return 0.5 * (req.n_out - 1.0)


def eta_min_req(req: Request, eta0: float = C.ETA_MIN) -> float:
    """Per-request compression floor: eta_min^u = eta_0 (technical floor).

    The task-dependent quality loss is carried by the calibrated profile
    (delta_u, theta_u), not by a floor. A single visual token is kept in full.
    """
    return 1.0 if int(req.n_vis) <= 1 else float(eta0)


def T_pre(req: Request) -> float:
    """User-side scoring pass (vision encoder + shallow backbone) latency."""
    return C.ZETA_SC * req.n_vis / C.F_LOC


# ---------------------------------------------------------------------------
#  Closed-form sat-side resource allocations (KKT)
# ---------------------------------------------------------------------------
def B_dn_star(req: Request, mu_dn: float, w_1: float) -> float:
    """B^{dn,*} (eq. 13) under bottleneck-max TPOT model.

    Since TPOT = max{TPOT_sat, T_ISL_dn, T_S2G} reduces to TPOT_sat in
    the satellite-decoding-dominant regime, B_dn enters only via T_S2G
    in TTFT, giving a w_1-only KKT coefficient.
    """
    denom = mu_dn * math.log2(1.0 + req.gamma_S2G)
    return math.sqrt(w_1 * C.KAPPA_OUT / denom)


def B_up_star(req: Request, eta: float, mu_up: float, w_1: float) -> float:
    """B^{up,*}(eta) (eq. 14)."""
    denom = mu_up * math.log2(1.0 + req.gamma_G2S)
    return math.sqrt(w_1 * C.KAPPA_IN * n_tilde(req, eta) / denom)


def Phi_pf_sat(req: Request, eta: float, w_1: float) -> float:
    return w_1 * C.ZETA_PF_SAT * n_tilde(req, eta)


def Phi_dc_sat(req: Request, eta: float, w_2: float) -> float:
    return w_2 * (C.BETA_0_SAT + C.BETA_1_SAT * (n_tilde(req, eta) + n_out_eff(req)))


def f_star_sat(req: Request, eta: float, nu: float, w_1: float, w_2: float) -> float:
    """f^*(eta) (eq. 15) on the satellite branch."""
    Phi = Phi_pf_sat(req, eta, w_1) + Phi_dc_sat(req, eta, w_2)
    return math.sqrt(Phi / nu)


# ---------------------------------------------------------------------------
#  Latency marginal cost C(eta) on the sat branch (after B_up*, f* substituted)
# ---------------------------------------------------------------------------
def C_eta_sat(req: Request, topo: Topology, s_u: int, eta: float,
              mu_up: float, nu: float, w_1: float, w_2: float) -> float:
    """C(eta; nu, mu_up) — latency marginal cost in eta (sat branch).

    Derived from envelope theorem after substituting B^{up,*}(eta) and f^*(eta).
    """
    Bup = B_up_star(req, eta, mu_up, w_1)
    fs  = f_star_sat(req, eta, nu, w_1, w_2)
    a_up = alpha_up(req)
    t_up = tau_up(req, topo, s_u)
    cost_up_path = a_up / Bup + t_up + C.ZETA_PF_SAT / fs
    cost_pertok  = C.BETA_1_SAT / fs
    return w_1 * req.n_vis * cost_up_path + w_2 * req.n_vis * cost_pertok


# ---------------------------------------------------------------------------
#  Solve eta* by bisection / brentq
# ---------------------------------------------------------------------------
def _solve_eta_sat(req: Request, topo: Topology, s_u: int,
                   mu_up: float, nu: float,
                   w_q: float, w_1: float, w_2: float,
                   eta_min: float = C.ETA_MIN, eta_max: float = C.ETA_MAX,
                   mu_dn: float = 1e-9) -> float:
    """Find eta* on the sat branch via Q(eta) = C(eta) (eq. 16)."""

    def F(eta: float) -> float:
        Q = quality_marginal_benefit(eta, x=1,
                                     delta=req.delta, theta=req.theta, w_q=w_q)
        Cc = C_eta_sat(req, topo, s_u, eta, mu_up, nu, w_1, w_2)
        return Q - Cc

    if eta_min >= eta_max - 1e-9:
        return eta_max
    F_lo, F_hi = F(eta_min), F(eta_max - 1e-9)
    cands = [eta_min, eta_max]
    if F_lo > 0.0 and F_hi < 0.0:
        cands.append(brentq(F, eta_min, eta_max - 1e-9,
                            xtol=C.ETA_BISECT_TOL, maxiter=C.ETA_BISECT_MAX_ITER))
    # best among stationary and boundary points (Lagrangian value at B_up*, f*)
    def L_of(eta: float) -> float:
        Bup = B_up_star(req, eta, mu_up, w_1); fs = f_star_sat(req, eta, nu, w_1, w_2)
        TTFT, TPOT = _ttft_tpot_sat(req, topo, s_u, eta, Bup, B_dn_star(req, mu_dn, w_1), fs)
        Qe = Q_eff(eta, x=1, delta=req.delta, theta=req.theta)
        return w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT - mu_up * Bup - nu * fs
    return max(cands, key=L_of)


def _solve_eta_loc(req: Request,
                   w_q: float, w_1: float, w_2: float,
                   eta_min: float = C.ETA_MIN, eta_max: float = C.ETA_MAX) -> float:
    """Find eta* on the local branch (no dual coupling, f = f_loc constant)."""

    def F(eta: float) -> float:
        Q = quality_marginal_benefit(eta, x=0,
                                     delta=req.delta, theta=req.theta, w_q=w_q)
        cost = (w_1 * req.n_vis * C.ZETA_PF_LOC / C.F_LOC
                + w_2 * req.n_vis * C.BETA_1_LOC / C.F_LOC)
        return Q - cost

    if eta_min >= eta_max - 1e-9:
        return eta_max
    F_lo, F_hi = F(eta_min), F(eta_max - 1e-9)
    cands = [eta_min, eta_max]
    if F_lo > 0.0 and F_hi < 0.0:
        cands.append(brentq(F, eta_min, eta_max - 1e-9,
                            xtol=C.ETA_BISECT_TOL, maxiter=C.ETA_BISECT_MAX_ITER))
    def QoE_of(eta: float) -> float:
        TTFT, TPOT = _ttft_tpot_loc(req, eta)
        return w_q * math.log(1.0 + Q_eff(eta, x=0, delta=req.delta, theta=req.theta)) - w_1 * TTFT - w_2 * TPOT
    return max(cands, key=QoE_of)


# ---------------------------------------------------------------------------
#  Full inner solve for one (u, s, x)
# ---------------------------------------------------------------------------
def _ttft_tpot_sat(req: Request, topo: Topology, s_u: int,
                    eta: float, B_up: float, B_dn: float, f: float) -> Tuple[float, float]:
    """TTFT and TPOT under the sat branch (eq. 6, 7).

    TPOT follows the bottleneck-max model (eq. 7). In the
    satellite-decoding-dominant regime (TPOT_sat ~ 10 ms vs
    T_S2G_per, T_ISL_per ~ sub-ms), TPOT ~= TPOT_sat numerically;
    we still compute the explicit max so that any future regime
    shift is exposed instead of hidden.
    """
    nti = n_tilde(req, eta)
    T_G2S = (C.KAPPA_IN * nti) / (B_up * math.log2(1.0 + req.gamma_G2S))
    T_S2G_per = C.KAPPA_OUT / (B_dn * math.log2(1.0 + req.gamma_S2G))
    if s_u == req.s0:
        T_isl_up = 0.0
        T_isl_dn_first = 0.0
        T_isl_dn_per = 0.0
        T_prop_isl = 0.0
    else:
        T_isl_up = C.KAPPA_IN * nti / topo.R_ISL_eff[req.s0, s_u]
        T_isl_dn_first = C.KAPPA_OUT / topo.R_ISL_eff[s_u, req.s0]
        T_isl_dn_per = T_isl_dn_first
        T_prop_isl = float(topo.T_prop_path[req.s0, s_u]) if topo.T_prop_path is not None else 0.0
    TTFT_sat = C.ZETA_PF_SAT * nti / f
    TPOT_sat = (C.BETA_0_SAT + C.BETA_1_SAT * (nti + n_out_eff(req))) / f
    # constant terms: user-side scoring pass, access-link propagation (up and
    # down), ISL propagation along the path (up and down)
    T_const = T_pre(req) + 2.0 * C.T_PROP_G2S + 2.0 * T_prop_isl
    TTFT = T_const + T_G2S + T_isl_up + TTFT_sat + T_isl_dn_first + T_S2G_per
    TPOT = max(TPOT_sat, T_isl_dn_per, T_S2G_per)
    return TTFT, TPOT


def _ttft_tpot_loc(req: Request, eta: float) -> Tuple[float, float]:
    """TTFT and TPOT on the local branch (eq. 6, 7, x=0)."""
    nti = n_tilde(req, eta)
    TTFT = T_pre(req) + C.ZETA_PF_LOC * nti / C.F_LOC     # encoder charged once, in T_pre
    TPOT = (C.BETA_0_LOC + C.BETA_1_LOC * (nti + n_out_eff(req))) / C.F_LOC
    return TTFT, TPOT


def solve_inner_sat(req: Request, topo: Topology, s_u: int,
                    mu_up: float, mu_dn: float, nu: float,
                    w_q: float, w_1: float, w_2: float,
                    eta_min: float = C.ETA_MIN,
                    eta_override: float | None = None) -> InnerResult:
    """Solve the inner subproblem under (x=1, s=s_u) with optional eta override.

    If eta_override is given, skip the bisection and use the supplied eta
    (used by Fixed-eta baseline).
    """
    eff_eta_min = eta_min_req(req, eta_min)
    eta = eta_override if eta_override is not None else _solve_eta_sat(
        req, topo, s_u, mu_up, nu, w_q, w_1, w_2, eta_min=eff_eta_min, mu_dn=mu_dn)
    Bup = B_up_star(req, eta, mu_up, w_1)
    Bdn = B_dn_star(req, mu_dn, w_1)
    fs  = f_star_sat(req, eta, nu, w_1, w_2)
    TTFT, TPOT = _ttft_tpot_sat(req, topo, s_u, eta, Bup, Bdn, fs)
    Qe = Q_eff(eta, x=1, delta=req.delta, theta=req.theta)
    QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
    L = QoE - mu_up * Bup - mu_dn * Bdn - nu * fs
    return InnerResult(eta=eta, B_up=Bup, B_dn=Bdn, f=fs,
                       L=L, QoE=QoE, TTFT=TTFT, TPOT=TPOT, Q_eff=Qe)


def solve_inner_loc(req: Request,
                    w_q: float, w_1: float, w_2: float,
                    eta_min: float = C.ETA_MIN,
                    eta_override: float | None = None) -> InnerResult:
    """Solve the inner subproblem under x=0 (independent of (mu, nu, s_u))."""
    eff_eta_min = eta_min_req(req, eta_min)
    eta = eta_override if eta_override is not None else _solve_eta_loc(
        req, w_q, w_1, w_2, eta_min=eff_eta_min)
    TTFT, TPOT = _ttft_tpot_loc(req, eta)
    Qe = Q_eff(eta, x=0, delta=req.delta, theta=req.theta)
    QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
    L = QoE        # no dual penalty when x=0
    return InnerResult(eta=eta, B_up=0.0, B_dn=0.0, f=C.F_LOC,
                       L=L, QoE=QoE, TTFT=TTFT, TPOT=TPOT, Q_eff=Qe)


# ---------------------------------------------------------------------------
#  Public solver dispatch
# ---------------------------------------------------------------------------
def solve_inner(req: Request, topo: Topology, s_u: int, x: int,
                mu_up: float, mu_dn: float, nu: float,
                w_q: float = C.W_Q, w_1: float = C.W_1, w_2: float = C.W_2,
                eta_min: float = C.ETA_MIN,
                eta_override: float | None = None) -> InnerResult:
    """Dispatch to sat-branch / local-branch inner solve."""
    if x == 1:
        return solve_inner_sat(req, topo, s_u, mu_up, mu_dn, nu,
                               w_q, w_1, w_2, eta_min=eta_min,
                               eta_override=eta_override)
    return solve_inner_loc(req, w_q, w_1, w_2,
                           eta_min=eta_min, eta_override=eta_override)


__all__ = [
    "InnerResult",
    "alpha_up", "alpha_dn", "tau_up", "tau_dn",
    "n_tilde", "n_out_eff", "eta_min_req", "T_pre",
    "B_up_star", "B_dn_star", "f_star_sat",
    "Phi_pf_sat", "Phi_dc_sat", "C_eta_sat",
    "solve_inner_sat", "solve_inner_loc", "solve_inner",
]
