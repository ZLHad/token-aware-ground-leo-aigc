"""Algorithm 1 (main.tex §III-D) — outer dual loop + per-user enumeration.

    repeat (t = 1, 2, ...)
        for each user u:
            for each (s, x) in S x {0, 1}:
                evaluate closed-form inner solution (Theorem 1)
                record V_u*(s, x)
            pick (s_u*, x_u*) = argmax_{s,x} V_u*(s, x)
        D^(t) = sum_u V_u* + sum_s (mu^up B^up,max + mu^dn B^dn,max + nu F^max)   # dual value (upper bound)
        z_hat^(t) = per-satellite budget rescaling of the raw allocation        # primal recovery
        P^(t) = sum_u QoE_u(z_hat^(t))                                          # feasible objective
        update duals (mu^up, mu^dn, nu) by log-domain sub-gradient steps
    until  min_t D^(t) - max_t P^(t) <= eps_g * |min_t D^(t)|  or  t = I_max
    return the best recovered point and the certified gap

Decisions are returned per user. Resource usage / violations are aggregated
per access satellite (for B^up, B^dn) or per compute satellite (for f).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as C
from src.data_gen import Request, Topology, SlotData
from src.inner_solve import InnerResult, solve_inner


# ---------------------------------------------------------------------------
#  Decision and per-slot result
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    u: int
    x: int                  # 0 (local) or 1 (sat)
    s: int                  # target compute satellite (only meaningful if x=1)
    eta: float
    B_up: float
    B_dn: float
    f: float
    QoE: float
    TTFT: float
    TPOT: float
    Q_eff: float
    L: float = 0.0          # Lagrangian value V_u*(s, x) at the duals used (raw solve)


@dataclass
class SlotResult:
    decisions: List[Decision]          # best recovered (feasible) allocation
    iters: int
    primal_dual_gap: float             # certified relative gap (D_min - P_max) / |D_min|
    sum_QoE: float                     # P_max = sum_u QoE at the recovered point
    wall_time_s: float
    duals: Dict[str, np.ndarray]
    gap_history: List[float] = field(default_factory=list)      # relative gap per iteration
    sum_QoE_history: List[float] = field(default_factory=list)  # P^(t) (feasible) per iteration
    dual_value_history: List[float] = field(default_factory=list)  # D^(t) per iteration
    certificate: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Resource accounting
# ---------------------------------------------------------------------------
def compute_usage(decisions: List[Decision], S: int, requests: List[Request]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-satellite resource usage from a list of decisions.

    Returns (B^up_used, B^dn_used, f_used), each of shape (S,).
    B^up/B^dn are indexed by access satellite s_0(u); f by target s_u (only x=1).
    """
    B_up = np.zeros(S)
    B_dn = np.zeros(S)
    f    = np.zeros(S)
    req_by_u = {r.u: r for r in requests}
    for d in decisions:
        s0 = req_by_u[d.u].s0
        B_up[s0] += d.B_up
        B_dn[s0] += d.B_dn
        if d.x == 1:
            f[d.s] += d.f
    return B_up, B_dn, f


# ---------------------------------------------------------------------------
#  Main solver
# ---------------------------------------------------------------------------
def solve_per_slot(slot: SlotData,
                   w_q: float = C.W_Q, w_1: float = C.W_1, w_2: float = C.W_2,
                   eta_min: float = C.ETA_MIN,
                   max_iter: int = C.MAX_DUAL_ITER,
                   tol: float = C.GAP_TOL,
                   viol_tol: float = C.VIOL_TOL,
                   eps_floor_mu: float = C.EPS_FLOOR_MU,
                   eps_floor_nu: float = C.EPS_FLOOR_NU,
                   rho_mu: float = C.RHO_MU_INIT,
                   rho_nu: float = C.RHO_NU_INIT,
                   rho_decay: float = C.RHO_DECAY,
                   decay_iters: Optional[int] = None,
                   tail_step: float = 0.05,
                   warm_start: bool = True,
                   demote_tol: Optional[float] = None,
                   polish: bool = True,
                   viol_clip: float = C.VIOL_CLIP,
                   dual_avg_window: int = C.DUAL_AVG_WINDOW,
                   B_up_budget: float = C.B_UP_MAX,
                   B_dn_budget: float = C.B_DN_MAX,
                   F_budget: float = C.F_SAT_MAX,
                   x_force: Optional[int] = None,
                   s_force: Optional[Dict[int, int]] = None,
                   eta_force: Optional[float] = None,
                   return_history: bool = False,
                   ) -> SlotResult:
    """Run Algorithm 1 on one slot.

    Dual update uses log-space (geometric) projected sub-gradient ascent on
    budget-normalized violations:

        viol_norm = (B_used - B_budget) / B_budget          (dimensionless)
        mu_up    <- max(eps_floor_mu, mu_up * exp(rho_mu * clip(viol_norm, -clip, clip)))

    Equivalent to sub-gradient ascent on log(mu); the step decays as
    rho * rho_decay**it. Multipliers are warm-started by a common scale
    factor (bisection on log kappa) at which the raw demand meets the budgets.

    Stopping (certificate): every iteration evaluates the dual value D^(t)
    (weak-duality upper bound on the optimum of P) and a feasible point
    recovered by proportional budget scaling with objective P^(t). The loop
    exits when (min_t D^(t) - max_t P^(t)) / |min_t D^(t)| <= tol or after
    max_iter iterations; the best recovered point and its gap are returned.
    After the loop, the trailing-window mean of the last `dual_avg_window`
    multiplier iterates is evaluated as one more candidate, and (with
    polish=True) the compute prices of over-subscribed satellites are
    refined by coordinate-wise bisection starting from the multipliers of
    the best point; both can only tighten the certificate.

    x_force / s_force / eta_force are introspection hooks used by baselines:
      x_force      -- if 0 or 1, lock x_u = x_force for all u
      s_force[u]   -- if provided, lock s_u for user u (bypass enumeration)
      eta_force    -- if provided, lock eta_u to this value (skip bisection)
    """
    t0 = time.perf_counter()
    S = slot.topology.S
    requests = slot.requests

    # Warm-start at order-of-magnitude estimates of the shadow prices.
    # mu_* ~ O(U / B_max) so that initial mu*B is in the QoE range
    # without forcing all users to local on iter 0; nu ~ O(U / F_max).
    U = len(requests)
    mu_up = np.full(S, max(1.0, U / S) / B_up_budget, dtype=float)
    mu_dn = np.full(S, max(1.0, U / S) / B_dn_budget, dtype=float)
    nu    = np.full(S, max(1.0, U / S) / F_budget,    dtype=float)

    # Dual-averaging buffer (sliding window) for read-out stability
    mu_up_buf: List[np.ndarray] = []
    mu_dn_buf: List[np.ndarray] = []
    nu_buf:    List[np.ndarray] = []

    decisions: List[Decision] = []
    gap_history: List[float] = []
    sum_QoE_history: List[float] = []

    converged_at = -1

    def per_user_subproblems(mu_up_vec: np.ndarray,
                              mu_dn_vec: np.ndarray,
                              nu_vec: np.ndarray) -> List[Decision]:
        """Solve the per-user subproblem for all users given duals."""
        out: List[Decision] = []
        for req in requests:
            best: Optional[Tuple[InnerResult, int, int]] = None
            x_candidates = [x_force] if x_force is not None else [0, 1]
            for x in x_candidates:
                if x == 0:
                    res = solve_inner(req, slot.topology, s_u=req.s0, x=0,
                                      mu_up=mu_up_vec[req.s0],
                                      mu_dn=mu_dn_vec[req.s0],
                                      nu=nu_vec[req.s0],
                                      w_q=w_q, w_1=w_1, w_2=w_2,
                                      eta_min=eta_min,
                                      eta_override=eta_force)
                    if best is None or res.L > best[0].L:
                        best = (res, req.s0, 0)
                else:
                    s_iter = [s_force[req.u]] if s_force is not None else list(range(S))
                    for s in s_iter:
                        res = solve_inner(req, slot.topology, s_u=s, x=1,
                                          mu_up=mu_up_vec[req.s0],
                                          mu_dn=mu_dn_vec[req.s0],
                                          nu=nu_vec[s],
                                          w_q=w_q, w_1=w_1, w_2=w_2,
                                          eta_min=eta_min,
                                          eta_override=eta_force)
                        if best is None or res.L > best[0].L:
                            best = (res, s, 1)
            assert best is not None
            res, s_star, x_star = best
            out.append(Decision(
                u=req.u, x=x_star, s=s_star,
                eta=res.eta, B_up=res.B_up, B_dn=res.B_dn, f=res.f,
                QoE=res.QoE, TTFT=res.TTFT, TPOT=res.TPOT, Q_eff=res.Q_eff, L=res.L,
            ))
        return out

    def dual_value(decs: List[Decision], mu_up_vec, mu_dn_vec, nu_vec) -> float:
        """D(mu, nu) = sum_u V_u* + sum_s (mu^up B^up,max + mu^dn B^dn,max + nu F^max): weak-duality upper bound."""
        return float(sum(d.L for d in decs)
                     + np.sum(mu_up_vec) * B_up_budget + np.sum(mu_dn_vec) * B_dn_budget + np.sum(nu_vec) * F_budget)

    def recover(decs: List[Decision]):
        rec = _demote_to_budgets(decs, requests, S, B_up_budget, B_dn_budget, F_budget, slot.topology,
                                 w_q, w_1, w_2, eta_min, over_tol=demote_tol) if demote_tol is not None else decs
        rec = _project_to_budgets(rec, requests, S, B_up_budget, B_dn_budget, F_budget, slot.topology, w_q, w_1, w_2)
        return rec, float(sum(d.QoE for d in rec))

    D_min = np.inf; P_max = -np.inf; best_rec: List[Decision] = []; best_duals = None; best_it = -1
    dual_value_history: List[float] = []
    gap_now = np.inf

    # ---- Warm start: scale the initial prices by a common factor kappa so that
    # the largest budget-normalized over-subscription of the raw allocation is
    # close to zero (bisection on log kappa; a dozen inner sweeps).
    if warm_start:
        base = (mu_up.copy(), mu_dn.copy(), nu.copy())
        def max_viol(log_k: float) -> float:
            k = float(np.exp(log_k))
            decs = per_user_subproblems(base[0] * k, base[1] * k, base[2] * k)
            bu, bd, fu = compute_usage(decs, S, requests)
            return float(max((bu / B_up_budget).max(), (bd / B_dn_budget).max(), (fu / F_budget).max()) - 1.0)
        lo, hi = -3.0, 6.0
        if max_viol(lo) > 0.0:
            if max_viol(hi) > 0.0:
                lo = hi
            else:
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    if max_viol(mid) > 0.0: lo = mid
                    else: hi = mid
                lo = hi          # feasible side
        k0 = float(np.exp(lo))
        mu_up, mu_dn, nu = base[0] * k0, base[1] * k0, base[2] * k0
    for it in range(max_iter):
        # ---- Per-user inner subproblem at the current duals ----
        decisions = per_user_subproblems(mu_up, mu_dn, nu)
        D_t = dual_value(decisions, mu_up, mu_dn, nu)
        rec_t, P_t = recover(decisions)
        D_min = min(D_min, D_t)
        if P_t > P_max:
            P_max, best_rec, best_it = P_t, rec_t, it
            best_duals = (mu_up.copy(), mu_dn.copy(), nu.copy())
        gap_now = (D_min - P_max) / max(abs(D_min), 1e-9)
        dual_value_history.append(D_t); sum_QoE_history.append(P_t); gap_history.append(float(gap_now))
        if gap_now <= tol and it >= 1:
            converged_at = it
            break
        # Certificate-scaled step (Polyak-type): large while the current gap is
        # large, shrinking with the gap, with a slow geometric decay as safeguard.
        # Two-phase schedule: geometric decay for the first `decay_iters`
        # iterations, then a constant small step (`tail_step`) that lets the
        # per-satellite prices keep separating and re-balance the lumpy
        # integer assignment; the best recovered point is tracked throughout.
        if decay_iters is None or it < decay_iters:
            scale = float(rho_decay ** it)
        else:
            scale = tail_step
        step_mu = rho_mu * scale
        step_nu = rho_nu * scale

        # ---- Aggregate raw usage and update duals (log-space sub-gradient) ----
        Bup_used, Bdn_used, f_used = compute_usage(decisions, S, requests)
        viol_up = (Bup_used - B_up_budget) / B_up_budget
        viol_dn = (Bdn_used - B_dn_budget) / B_dn_budget
        viol_f  = (f_used   - F_budget)    / F_budget
        mu_up = np.maximum(eps_floor_mu, mu_up * np.exp(step_mu * np.clip(viol_up, -viol_clip, viol_clip)))
        mu_dn = np.maximum(eps_floor_mu, mu_dn * np.exp(step_mu * np.clip(viol_dn, -viol_clip, viol_clip)))
        nu    = np.maximum(eps_floor_nu, nu    * np.exp(step_nu * np.clip(viol_f,  -viol_clip, viol_clip)))

        mu_up_buf.append(mu_up.copy()); mu_dn_buf.append(mu_dn.copy()); nu_buf.append(nu.copy())
        if len(mu_up_buf) > dual_avg_window:
            mu_up_buf.pop(0); mu_dn_buf.pop(0); nu_buf.pop(0)

    # ---- One extra candidate: the trailing-window averaged duals ----
    if mu_up_buf:
        mu_up_avg = np.mean(np.stack(mu_up_buf, axis=0), axis=0)
        mu_dn_avg = np.mean(np.stack(mu_dn_buf, axis=0), axis=0)
        nu_avg    = np.mean(np.stack(nu_buf,    axis=0), axis=0)
        decs_avg = per_user_subproblems(mu_up_avg, mu_dn_avg, nu_avg)
        D_min = min(D_min, dual_value(decs_avg, mu_up_avg, mu_dn_avg, nu_avg))
        rec_a, P_a = recover(decs_avg)
        if P_a > P_max:
            P_max, best_rec, best_it = P_a, rec_a, it + 1
            best_duals = (mu_up_avg.copy(), mu_dn_avg.copy(), nu_avg.copy())
        gap_now = (D_min - P_max) / max(abs(D_min), 1e-9)
    if best_duals is None:
        best_duals = (mu_up.copy(), mu_dn.copy(), nu.copy())

    # ---- Price polishing: coordinate-wise bisection of the compute price of
    # each over-subscribed satellite (others fixed) so that the raw demand meets
    # the budget from below; keeps the best recovered point found.
    if polish:
        mu_u, mu_d, nu_p = (b.copy() for b in best_duals)
        for _ in range(2):
            for s_idx in range(S):
                decs = per_user_subproblems(mu_u, mu_d, nu_p)
                _, _, f_used = compute_usage(decs, S, requests)
                if f_used[s_idx] <= F_budget * 1.02:
                    continue
                lo, hi = nu_p[s_idx], nu_p[s_idx] * 8.0
                for _b in range(10):
                    mid = float(np.sqrt(lo * hi)); trial = nu_p.copy(); trial[s_idx] = mid
                    decs = per_user_subproblems(mu_u, mu_d, trial)
                    _, _, f_t = compute_usage(decs, S, requests)
                    if f_t[s_idx] > F_budget: lo = mid
                    else: hi = mid
                    D_min = min(D_min, dual_value(decs, mu_u, mu_d, trial))
                    rec_b, P_b = recover(decs)
                    if P_b > P_max:
                        P_max, best_rec = P_b, rec_b; best_duals = (mu_u.copy(), mu_d.copy(), trial.copy())
                nu_p[s_idx] = hi
        gap_now = (D_min - P_max) / max(abs(D_min), 1e-9)

    # ---- Certificate at the returned point ----
    Bup_r, Bdn_r, f_r = compute_usage(best_rec, S, requests)
    cert = {
        "D_min": float(D_min), "P_max": float(P_max), "gap": float(gap_now), "best_iter": int(best_it),
        "n_iter": int(converged_at + 1 if converged_at >= 0 else max_iter),
        "max_util_up": float(np.max(Bup_r) / B_up_budget), "max_util_dn": float(np.max(Bdn_r) / B_dn_budget),
        "max_util_f": float(np.max(f_r) / F_budget),
    }
    return SlotResult(
        decisions=best_rec,
        iters=cert["n_iter"],
        primal_dual_gap=float(gap_now),
        sum_QoE=float(P_max),
        wall_time_s=time.perf_counter() - t0,
        duals={"mu_up": best_duals[0], "mu_dn": best_duals[1], "nu": best_duals[2]},
        gap_history=gap_history if return_history else [],
        sum_QoE_history=sum_QoE_history if return_history else [],
        dual_value_history=dual_value_history if return_history else [],
        certificate=cert,
    )


# ---------------------------------------------------------------------------
#  Primal recovery, step 1: demotion. While a satellite's raw compute (or
#  bandwidth) demand exceeds (1 + over_tol) x budget, the offloaded user on
#  that satellite with the smallest offloading gain QoE_sat - QoE_loc is moved
#  to its local branch. The remaining excess (<= over_tol) is rescaled by
#  _project_to_budgets.
# ---------------------------------------------------------------------------
def _demote_to_budgets(decisions: List[Decision], requests: List[Request], S: int,
                       B_up_budget: float, B_dn_budget: float, F_budget: float,
                       topo: Topology, w_q: float, w_1: float, w_2: float,
                       eta_min: float, over_tol: float = 0.05) -> List[Decision]:
    from src.inner_solve import solve_inner_loc
    req_by_u = {r.u: r for r in requests}
    decs = list(decisions)
    loc_cache: Dict[int, Decision] = {}

    def local_decision(d: Decision) -> Decision:
        if d.u not in loc_cache:
            r = solve_inner_loc(req_by_u[d.u], w_q, w_1, w_2, eta_min=eta_min)
            loc_cache[d.u] = Decision(u=d.u, x=0, s=req_by_u[d.u].s0, eta=r.eta, B_up=0.0, B_dn=0.0, f=r.f,
                                      QoE=r.QoE, TTFT=r.TTFT, TPOT=r.TPOT, Q_eff=r.Q_eff, L=r.L)
        return loc_cache[d.u]

    for _ in range(len(decs)):
        Bup_used, Bdn_used, f_used = compute_usage(decs, S, requests)
        over_f  = f_used   / F_budget   - 1.0
        over_up = Bup_used / B_up_budget - 1.0
        over_dn = Bdn_used / B_dn_budget - 1.0
        worst = max(over_f.max(), over_up.max(), over_dn.max())
        if worst <= over_tol:
            break
        # satellite and resource with the largest excess
        if over_f.max() >= max(over_up.max(), over_dn.max()):
            s_bad = int(np.argmax(over_f)); users = [i for i, d in enumerate(decs) if d.x == 1 and d.s == s_bad]
        elif over_up.max() >= over_dn.max():
            s_bad = int(np.argmax(over_up)); users = [i for i, d in enumerate(decs) if d.x == 1 and req_by_u[d.u].s0 == s_bad]
        else:
            s_bad = int(np.argmax(over_dn)); users = [i for i, d in enumerate(decs) if d.x == 1 and req_by_u[d.u].s0 == s_bad]
        if not users:
            break
        i_min = min(users, key=lambda i: decs[i].QoE - local_decision(decs[i]).QoE)
        decs[i_min] = local_decision(decs[i_min])
    return decs


# ---------------------------------------------------------------------------
#  Budget projection (re-scales B_up, B_dn, f per satellite to enforce caps,
#  then recomputes per-user TTFT/TPOT/QoE consistently)
# ---------------------------------------------------------------------------
def _project_to_budgets(decisions: List[Decision], requests: List[Request], S: int,
                        B_up_budget: float, B_dn_budget: float, F_budget: float,
                        topo: Topology, w_q: float, w_1: float, w_2: float
                        ) -> List[Decision]:
    """Scale per-user resource allocations so each satellite stays within budget,
    then recompute QoE/TTFT/TPOT with the projected values."""
    import math

    from src.inner_solve import (n_tilde, n_out_eff, _ttft_tpot_sat, _ttft_tpot_loc)
    from src.semantic_profile import Q_eff

    req_by_u = {r.u: r for r in requests}

    # ---- Aggregate ----
    Bup_used, Bdn_used, f_used = compute_usage(decisions, S, requests)

    scale_up = np.ones(S)
    scale_dn = np.ones(S)
    scale_f  = np.ones(S)
    for s in range(S):
        if Bup_used[s] > B_up_budget and Bup_used[s] > 0:
            scale_up[s] = B_up_budget / Bup_used[s]
        if Bdn_used[s] > B_dn_budget and Bdn_used[s] > 0:
            scale_dn[s] = B_dn_budget / Bdn_used[s]
        if f_used[s]   > F_budget   and f_used[s]   > 0:
            scale_f[s]  = F_budget   / f_used[s]

    new_decisions: List[Decision] = []
    for d in decisions:
        req = req_by_u[d.u]
        if d.x == 0:
            new_decisions.append(d)
            continue
        Bup = d.B_up * scale_up[req.s0]
        Bdn = d.B_dn * scale_dn[req.s0]
        fs  = d.f    * scale_f[d.s]
        TTFT, TPOT = _ttft_tpot_sat(req, topo, d.s, d.eta, Bup, Bdn, fs)
        Qe  = Q_eff(d.eta, x=1, delta=req.delta, theta=req.theta)
        QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
        new_decisions.append(Decision(
            u=d.u, x=d.x, s=d.s, eta=d.eta,
            B_up=Bup, B_dn=Bdn, f=fs,
            QoE=QoE, TTFT=TTFT, TPOT=TPOT, Q_eff=Qe, L=d.L,
        ))
    return new_decisions


__all__ = ["Decision", "SlotResult", "solve_per_slot", "compute_usage"]
