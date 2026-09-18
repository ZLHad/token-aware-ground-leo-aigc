"""Fixed-eta Dijkstra baseline (main.tex §IV-A, first baseline).

Description from main.tex:
    "Fixed-eta Dijkstra disables adaptive compression by fixing a nominal
     compression ratio eta_u = 0.5 and selects the target satellite through
     latency-only Dijkstra routing with equally split bandwidth and compute."

Concretely:
    x_u  = 1 for all u (sat branch; this baseline does not consider local fallback)
    s_u  = s_0(u)  (latency-shortest path is the access satellite itself)
    eta_u = 0.5
    B_up_u = B_up_max / (# users with s_0 == s)
    B_dn_u = B_dn_max / (# users with s_0 == s)
    f_u    = F_max   / (# users with s_u == s)

The per-user TTFT/TPOT/QoE are then evaluated directly from the model.
"""
from __future__ import annotations

import math
import time
from typing import List

import numpy as np

import config as C
from src.data_gen import SlotData, Request, Topology
from src.inner_solve import _ttft_tpot_sat
from src.semantic_profile import Q_eff
from src.optimizer import Decision, SlotResult, compute_usage


FIXED_ETA: float = 0.2   # 90%-retention point of the request-oblivious (global) profile


def solve_per_slot_fixed_eta(slot: SlotData,
                              w_q: float = C.W_Q, w_1: float = C.W_1, w_2: float = C.W_2,
                              fixed_eta: float = FIXED_ETA,
                              B_up_budget: float = C.B_UP_MAX,
                              B_dn_budget: float = C.B_DN_MAX,
                              F_budget: float = C.F_SAT_MAX,
                              ) -> SlotResult:
    """Fixed-eta + Dijkstra-shortest target satellite (= s_0) + equal split."""
    t0 = time.perf_counter()
    S = slot.topology.S
    requests = slot.requests

    # Count users per access satellite (used for both B-split and f-split,
    # since s_u = s_0(u) for this baseline).
    n_per_sat = np.zeros(S, dtype=int)
    for req in requests:
        n_per_sat[req.s0] += 1

    decisions: List[Decision] = []
    for req in requests:
        s = req.s0
        U_s = max(n_per_sat[s], 1)
        Bup = B_up_budget / U_s
        Bdn = B_dn_budget / U_s
        fs  = F_budget   / U_s
        TTFT, TPOT = _ttft_tpot_sat(req, slot.topology, s, fixed_eta, Bup, Bdn, fs)
        Qe = Q_eff(fixed_eta, x=1, delta=req.delta, theta=req.theta)
        QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
        decisions.append(Decision(
            u=req.u, x=1, s=s, eta=fixed_eta,
            B_up=Bup, B_dn=Bdn, f=fs,
            QoE=QoE, TTFT=TTFT, TPOT=TPOT, Q_eff=Qe,
        ))

    return SlotResult(
        decisions=decisions,
        iters=1,
        primal_dual_gap=0.0,
        sum_QoE=float(sum(d.QoE for d in decisions)),
        wall_time_s=time.perf_counter() - t0,
        duals={"mu_up": np.zeros(S), "mu_dn": np.zeros(S), "nu": np.zeros(S)},
        gap_history=[],
    )


__all__ = ["solve_per_slot_fixed_eta", "FIXED_ETA"]
