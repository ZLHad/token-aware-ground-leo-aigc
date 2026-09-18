"""Unit tests for inner_solve.py — verify that the closed-form KKT solutions
match brute-force numerical optimization of the Lagrangian.

Run from repo root:
    python -m pytest code/tests/test_inner_solve.py -v
or:
    python code/tests/test_inner_solve.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import config as C
from src.data_gen import Request, ring_topology
from src.inner_solve import (
    solve_inner_sat, B_up_star, B_dn_star, f_star_sat,
    _ttft_tpot_sat, n_tilde, n_out_eff, eta_min_req,
)
from src.semantic_profile import Q_eff


def make_req(seed: int = 0) -> Request:
    rng = np.random.default_rng(seed)
    return Request(
        u=0,
        n_txt=int(rng.integers(*C.N_TXT_RANGE)),
        n_vis=int(rng.integers(*C.N_VIS_RANGE)),
        n_out=int(rng.integers(*C.N_OUT_RANGE)),
        H=float(rng.uniform()),
        delta=float(rng.uniform(0.3, 0.9)),
        theta=float(rng.uniform(1.2, 2.5)),
        s0=0,
        gamma_G2S=float(rng.uniform(5.0, 50.0)),
        gamma_S2G=float(rng.uniform(5.0, 50.0)),
    )


def lagrangian_sat(eta: float, B_up: float, B_dn: float, f: float,
                   req: Request, topo, s_u: int,
                   mu_up: float, mu_dn: float, nu: float,
                   w_q: float, w_1: float, w_2: float) -> float:
    """L_u (sat branch) computed directly from physical primitives."""
    TTFT, TPOT = _ttft_tpot_sat(req, topo, s_u, eta, B_up, B_dn, f)
    Qe  = Q_eff(eta, x=1, delta=req.delta, theta=req.theta)
    QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
    return QoE - mu_up * B_up - mu_dn * B_dn - nu * f


# ---------------------------------------------------------------------------
#  Test: closed-form (B_up*, B_dn*, f*) is a local maximum -- 1D scans over each
#  variable (others fixed at closed-form) must all peak at the closed-form value.
# ---------------------------------------------------------------------------
def test_closed_form_is_local_max():
    for seed in range(5):
        req = make_req(seed)
        topo = ring_topology(S=3)
        s_u = (req.s0 + 1) % 3
        mu_up, mu_dn, nu = 1e-8, 5e-9, 0.3
        w_q, w_1, w_2 = C.W_Q, C.W_1, C.W_2

        res = solve_inner_sat(req, topo, s_u, mu_up, mu_dn, nu, w_q, w_1, w_2)

        def L_at(eta=None, B_up=None, B_dn=None, f=None) -> float:
            return lagrangian_sat(
                eta if eta is not None else res.eta,
                B_up if B_up is not None else res.B_up,
                B_dn if B_dn is not None else res.B_dn,
                f   if f   is not None else res.f,
                req, topo, s_u, mu_up, mu_dn, nu, w_q, w_1, w_2,
            )

        L_star = res.L
        # ---- 1D sweep for each variable, ±factor scan ----
        scan_factors = np.linspace(0.5, 2.0, 21)   # 50% ~ 200% of closed-form
        for name, sweeper in [
            ("B_up", lambda x: L_at(B_up=res.B_up * x)),
            ("B_dn", lambda x: L_at(B_dn=res.B_dn * x)),
            ("f",    lambda x: L_at(f=res.f * x)),
        ]:
            sweep = np.array([sweeper(x) for x in scan_factors])
            argmax = scan_factors[np.argmax(sweep)]
            L_best = float(np.max(sweep))
            # Closed-form should be ≥ any value on the scan within tiny eps
            assert L_star >= L_best - 1e-6, (
                f"seed={seed} {name}: closed-form L={L_star:.6f} < scan max L={L_best:.6f} "
                f"at factor={argmax:.3f}"
            )
        # ---- 1D sweep for eta on a finer grid (it has its own bisection logic) ----
        eta_grid = np.linspace(max(eta_min_req(req), res.eta - 0.2),
                                min(1.0 - 1e-6, res.eta + 0.2), 41)
        eta_sweep = np.array([L_at(eta=e) for e in eta_grid])
        eta_best = float(np.max(eta_sweep))
        assert L_star >= eta_best - 1e-4, (
            f"seed={seed} eta: closed-form L={L_star:.6f} < scan max L={eta_best:.6f}"
        )
        print(f"  seed={seed}: L*={L_star:.4f}  3D scan max={eta_best:.4f}  diff={L_star-eta_best:+.2e}")


# ---------------------------------------------------------------------------
#  Test: B_up* / B_dn* / f* satisfy their first-order conditions analytically
# ---------------------------------------------------------------------------
def test_first_order_conditions():
    req = make_req(seed=42)
    topo = ring_topology(S=3)
    s_u = (req.s0 + 1) % 3
    mu_up, mu_dn, nu = 2e-8, 1e-8, 0.3
    w_q, w_1, w_2 = C.W_Q, C.W_1, C.W_2

    eta = 0.6
    Bup = B_up_star(req, eta, mu_up, w_1)
    Bdn = B_dn_star(req, mu_dn, w_1)
    fs  = f_star_sat(req, eta, nu, w_1, w_2)

    # FOC for B_up: w_1 * alpha_up * n_tilde / B^2 = mu_up
    alpha_up_ = C.KAPPA_IN / math.log2(1 + req.gamma_G2S)
    foc_Bup = w_1 * alpha_up_ * n_tilde(req, eta) / Bup**2
    assert abs(foc_Bup - mu_up) / mu_up < 1e-9, f"FOC B_up: {foc_Bup} vs {mu_up}"

    # FOC for B_dn under bottleneck-max TPOT: w_1 * alpha_dn / B^2 = mu_dn
    # (B_dn enters only via T_S2G in TTFT; per-token decoding decoupled.)
    alpha_dn_ = C.KAPPA_OUT / math.log2(1 + req.gamma_S2G)
    foc_Bdn = w_1 * alpha_dn_ / Bdn**2
    assert abs(foc_Bdn - mu_dn) / mu_dn < 1e-9, f"FOC B_dn: {foc_Bdn} vs {mu_dn}"

    # FOC for f: (Phi_pf + Phi_dc) / f^2 = nu
    nti = n_tilde(req, eta)
    Phi = (w_1 * C.ZETA_PF_SAT * nti
           + w_2 * (C.BETA_0_SAT + C.BETA_1_SAT * (nti + n_out_eff(req))))
    foc_f = Phi / fs**2
    assert abs(foc_f - nu) / nu < 1e-9, f"FOC f: {foc_f} vs {nu}"

    print(f"  KKT residuals:  B_up={abs(foc_Bup-mu_up)/mu_up:.2e}  "
          f"B_dn={abs(foc_Bdn-mu_dn)/mu_dn:.2e}  f={abs(foc_f-nu)/nu:.2e}")


# ---------------------------------------------------------------------------
#  Test: eta_min and eta_max boundary projection
# ---------------------------------------------------------------------------
def test_eta_boundary_projection():
    req = make_req(seed=7)
    topo = ring_topology(S=3)
    w_q, w_1, w_2 = C.W_Q, C.W_1, C.W_2

    # Very high nu -> compute very expensive -> eta -> eta_min
    res_hi_nu = solve_inner_sat(req, topo, req.s0, mu_up=1e-9, mu_dn=1e-9,
                                 nu=1e3, w_q=w_q, w_1=w_1, w_2=w_2)
    assert abs(res_hi_nu.eta - C.ETA_MIN) < 1e-6, f"high nu: eta = {res_hi_nu.eta}"

    # Very low nu -> compute nearly free -> eta -> 1
    res_lo_nu = solve_inner_sat(req, topo, req.s0, mu_up=1e-12, mu_dn=1e-12,
                                 nu=1e-6, w_q=w_q, w_1=w_1, w_2=w_2)
    assert res_lo_nu.eta > 0.99, f"low nu: eta = {res_lo_nu.eta}"

    print(f"  boundary projection:  high-nu eta={res_hi_nu.eta:.4f}  "
          f"low-nu eta={res_lo_nu.eta:.4f}")


if __name__ == "__main__":
    print("test_closed_form_is_local_max")
    test_closed_form_is_local_max()
    print("\ntest_first_order_conditions")
    test_first_order_conditions()
    print("\ntest_eta_boundary_projection")
    test_eta_boundary_projection()
    print("\nAll inner_solve tests passed.")
