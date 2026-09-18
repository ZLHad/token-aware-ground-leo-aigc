"""Unit tests for Theorem 2 — verify the sign of d eta* / d (nu, delta, theta).

main.tex eq. (18):
    d eta* / d nu_{s_u*}  <= 0       (compute price up   -> eta* down)
    d eta* / d delta_u    >= 0       (more sensitive D   -> eta* up   = less compression)
    d eta* / d theta_u    <= 0       (sharper exponent   -> eta* down = more compression)
                                       under the regularity theta * ln(1-eta*) < -1

Test: finite-difference sweep across (nu, delta, theta) values.
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
from src.inner_solve import _solve_eta_sat


def make_test_req(delta: float, theta: float) -> Request:
    return Request(
        u=0,
        n_txt=30,
        n_vis=512,
        n_out=60,
        H=float("nan"),
        delta=delta,
        theta=theta,
        s0=0,
        gamma_G2S=10 ** (C.GAMMA_G2S_DB_MEAN / 10.0),
        gamma_S2G=10 ** (C.GAMMA_S2G_DB_MEAN / 10.0),
    )


def eta_star_at(delta: float, theta: float, nu: float,
                mu_up: float = 1e-9) -> float:
    req = make_test_req(delta, theta)
    topo = ring_topology(S=1)
    return _solve_eta_sat(req, topo, s_u=0, mu_up=mu_up, nu=nu,
                          w_q=C.W_Q, w_1=C.W_1, w_2=C.W_2,
                          eta_min=C.ETA_MIN)


# ---------------------------------------------------------------------------
#  d eta* / d nu <= 0
# ---------------------------------------------------------------------------
def test_eta_monotone_in_nu():
    failures = 0
    for delta, theta, _ in C.FIG1C_PROFILES:
        nu_grid = np.logspace(-3, 0, 30)
        etas = np.array([eta_star_at(delta, theta, nu) for nu in nu_grid])
        # All increments must be non-positive (up to numerical noise)
        max_inc = float(np.max(np.diff(etas)))
        ok = max_inc <= 1e-6
        if not ok:
            failures += 1
        print(f"  delta={delta:.2f}, theta={theta:.2f}: max increment = {max_inc:+.2e}  "
              f"{'OK' if ok else 'FAIL'}")
    assert failures == 0, f"{failures} profile(s) violated d eta*/d nu <= 0"


# ---------------------------------------------------------------------------
#  d eta* / d delta >= 0 (at fixed (nu, theta))
# ---------------------------------------------------------------------------
def test_eta_monotone_in_delta():
    failures = 0
    for nu in (0.05, 0.1, 0.3):
        for theta in (1.5, 2.0, 2.5):
            delta_grid = np.linspace(0.2, 0.95, 16)
            etas = np.array([eta_star_at(d, theta, nu) for d in delta_grid])
            # Restrict to non-boundary region (eta in (eta_min + eps, 1 - eps))
            interior = (etas > C.ETA_MIN + 1e-3) & (etas < 1.0 - 1e-3)
            if interior.sum() < 3:
                continue   # not enough interior points to assess monotonicity
            min_inc = float(np.min(np.diff(etas[interior])))
            ok = min_inc >= -1e-6
            if not ok:
                failures += 1
            print(f"  nu={nu:.2f}, theta={theta:.2f}: min increment in delta = {min_inc:+.2e}  "
                  f"({interior.sum()} interior pts)  {'OK' if ok else 'FAIL'}")
    assert failures == 0, f"{failures} (nu, theta) combo violated d eta*/d delta >= 0"


# ---------------------------------------------------------------------------
#  d eta* / d theta <= 0 in the non-extreme compression regime
#  (theta * ln(1-eta*) < -1, i.e., eta* not too close to 0)
# ---------------------------------------------------------------------------
def test_eta_monotone_in_theta_regular_regime():
    failures = 0
    for nu in (0.05, 0.1, 0.3):
        for delta in (0.3, 0.5, 0.7):
            theta_grid = np.linspace(1.2, 3.0, 16)
            etas = np.array([eta_star_at(delta, t, nu) for t in theta_grid])
            # Regularity: theta * ln(1-eta*) < -1
            regular = []
            for t, e in zip(theta_grid, etas):
                if e >= 1.0 - 1e-9 or e <= C.ETA_MIN + 1e-9:
                    regular.append(False)
                    continue
                regular.append(t * math.log(1.0 - e) < -1.0)
            regular = np.array(regular)
            if regular.sum() < 3:
                continue
            # In the regular regime, eta* must be non-increasing in theta
            reg_etas = etas[regular]
            reg_thetas = theta_grid[regular]
            max_inc = float(np.max(np.diff(reg_etas))) if len(reg_etas) > 1 else 0.0
            ok = max_inc <= 1e-5
            if not ok:
                failures += 1
            print(f"  nu={nu:.2f}, delta={delta:.2f}: max increment in theta "
                  f"(regular regime, {regular.sum()} pts) = {max_inc:+.2e}  "
                  f"{'OK' if ok else 'FAIL'}")
    assert failures == 0, f"{failures} (nu, delta) combo violated d eta*/d theta <= 0"


# ---------------------------------------------------------------------------
#  Sanity: the three FIG1C profiles must produce *visually distinct* curves
#  (high sensitivity strictly above medium strictly above low across some nu)
# ---------------------------------------------------------------------------
