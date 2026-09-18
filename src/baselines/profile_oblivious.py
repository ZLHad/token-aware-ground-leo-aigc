"""Profile-Oblivious baseline (main.tex §IV-A, second baseline).

Description from main.tex:
    "Profile-Oblivious employs the same dual decomposition but ignores the
     request-level semantic profile by setting xi_u = bar_xi uniformly across
     users, where bar_xi denotes the empirical mean profile over the
     calibration set."

Implementation:
  1. Run the main optimizer on a perturbed SlotData where every request has
     (delta, theta) overwritten to (delta_bar, theta_bar). The decisions
     (x, s, eta, B^up, B^dn, f) are oblivious to per-user semantic profile.
  2. Re-evaluate QoE / TTFT / TPOT under the *true* (delta_u, theta_u) at
     fixed decisions -- so we measure the true cost of being profile-oblivious.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import List

import config as C
from src.data_gen import SlotData
from src.inner_solve import _ttft_tpot_sat, _ttft_tpot_loc
from src.semantic_profile import Q_eff
from src.optimizer import Decision, SlotResult, solve_per_slot


# bar_xi = the single H-/task-oblivious profile fitted on the calibration set
# (data/calib_table.json: global_profile), i.e. what a request-oblivious
# scheduler assumes for every request.
import numpy as np
from src.data_gen import global_profile
DELTA_BAR, THETA_BAR = global_profile()


def _reevaluate_under_true_xi(decisions: List[Decision], slot: SlotData,
                               w_q: float, w_1: float, w_2: float) -> List[Decision]:
    """Recompute QoE / TTFT / TPOT under the *true* (delta_u, theta_u) at
    fixed decisions. Used to penalize the profile-oblivious baseline for
    its mis-specified objective."""
    req_by_u = {r.u: r for r in slot.requests}
    out: List[Decision] = []
    for d in decisions:
        req = req_by_u[d.u]
        if d.x == 0:
            TTFT, TPOT = _ttft_tpot_loc(req, d.eta)
        else:
            TTFT, TPOT = _ttft_tpot_sat(req, slot.topology, d.s, d.eta,
                                         d.B_up, d.B_dn, d.f)
        Qe = Q_eff(d.eta, x=d.x, delta=req.delta, theta=req.theta)
        QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
        out.append(Decision(u=d.u, x=d.x, s=d.s, eta=d.eta,
                            B_up=d.B_up, B_dn=d.B_dn, f=d.f,
                            QoE=QoE, TTFT=TTFT, TPOT=TPOT, Q_eff=Qe))
    return out


def solve_per_slot_profile_oblivious(slot: SlotData,
                                      delta_bar: float = DELTA_BAR,
                                      theta_bar: float = THETA_BAR,
                                      w_q: float = C.W_Q, w_1: float = C.W_1,
                                      w_2: float = C.W_2,
                                      **kwargs) -> SlotResult:
    """Run the main optimizer with each request's (delta, theta) overwritten by
    the calibration-set mean profile, then re-evaluate QoE under the true
    per-user (delta, theta) -- this captures the actual cost of being
    profile-oblivious."""
    oblivious_reqs = [replace(r, delta=delta_bar, theta=theta_bar) for r in slot.requests]
    oblivious_slot = SlotData(topology=slot.topology, requests=oblivious_reqs)
    result = solve_per_slot(oblivious_slot, w_q=w_q, w_1=w_1, w_2=w_2, **kwargs)
    # Re-evaluate at the true (delta, theta)
    decisions = _reevaluate_under_true_xi(result.decisions, slot, w_q, w_1, w_2)
    return SlotResult(
        decisions=decisions,
        iters=result.iters,
        primal_dual_gap=result.primal_dual_gap,
        sum_QoE=float(sum(d.QoE for d in decisions)),
        wall_time_s=result.wall_time_s,
        duals=result.duals,
        gap_history=result.gap_history,
        sum_QoE_history=result.sum_QoE_history,
    )


__all__ = ["solve_per_slot_profile_oblivious", "DELTA_BAR", "THETA_BAR"]
