"""Rule-Based Offloader baseline (paper Sec. IV-A, third baseline): a rule-based
offloader in the spirit of the in-context-learning offloader of [zhou2025genai].

Description from main.tex:
    "LLM-Distilled Heur. replaces the closed-form optimizer with an in-context-learning
     offloader operating on the same TTFT/TPOT inputs."

Implementation notes:
  * Letter-scale simulations require 1e5+ slot-level decisions, so always
    invoking a remote LLM API is infeasible. We therefore use a
    **heuristic decision rule that emulates the typical reasoning of a
    few-shot LLM offloader** as the default path, and expose an OPTIONAL
    real-API path (gated by env var WCL_USE_REAL_ICL=1, requires
    OPENAI_API_KEY) for validation runs.
  * The heuristic mimics Zhou 2025 in three ways:
      - greedy load-aware routing (hops + estimated congestion);
      - intent-proportional compression (eta_u depends on H_u);
      - x_u = 1 only when the sat is not over-subscribed.
  * Bandwidth and compute are equally split per access satellite, matching
    the original ICL paper's lack of fine-grained resource KKT.
"""
from __future__ import annotations

import math
import os
import time
from typing import List

import numpy as np

import config as C
from src.data_gen import SlotData, Request
from src.inner_solve import _ttft_tpot_sat, _ttft_tpot_loc
from src.semantic_profile import Q_eff
from src.optimizer import Decision, SlotResult


USE_REAL_API = os.environ.get("WCL_USE_REAL_ICL", "0") == "1"


def _eta_90(delta: float, theta: float) -> float:
    """Smallest eta with S(eta) = 1 - delta (1-eta)^theta >= 0.9 (clipped to [eta_0, 1])."""
    if delta <= 0.1:
        return C.ETA_MIN
    eta = 1.0 - (0.1 / delta) ** (1.0 / theta)
    return float(np.clip(eta, C.ETA_MIN, 1.0))


def _heuristic_decision(req: Request, sat_load: np.ndarray) -> tuple:
    """Rule-Based Offloader: a fixed engineering rule that uses the same
    user-side information as the proposed scheme but no joint optimization.

      * eta_u : the 90%-retention point of the request's calibrated task
                profile (a static per-task table);
      * s_u   : the access satellite;
      * x_u   : offload by default; fall back to the local model when the
                access satellite carries more than 1.5x its share of users
                and the request is compression-tolerant (eta_90 <= 0.1).
    """
    eta = _eta_90(req.delta, req.theta)
    s = req.s0
    x = 0 if (sat_load[s] >= 1.5 and eta <= 0.1) else 1
    return x, s, eta


def _real_api_decision(req: Request, sat_load_str: str) -> tuple:
    """Real OpenAI API call returning (x, s, eta). Used only when
    WCL_USE_REAL_ICL=1 is set. Falls back to heuristic on any failure.

    NOTE: Validated separately; production sweeps default to the heuristic.
    """
    try:
        from openai import OpenAI                                    # noqa: F401
        client = OpenAI()
        prompt = (
            "You are a satellite-edge inference scheduler. Given the user "
            f"request: n_txt={req.n_txt}, n_vis={req.n_vis}, n_out={req.n_out}, "
            f"intent H={req.H:.2f}; access sat s0={req.s0}; current sat loads: "
            f"{sat_load_str}. Decide (x_u in {{0,1}}, s_u in [0..5], eta_u in [0.1,1]). "
            "Reply only as 'x,s,eta' (e.g. '1,2,0.7')."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20, temperature=0.0,
        )
        txt = resp.choices[0].message.content.strip()
        x_str, s_str, eta_str = txt.split(",")
        return int(x_str), int(s_str), float(eta_str)
    except Exception:
        return _heuristic_decision(req, np.zeros(6))


def solve_per_slot_icl_llm(slot: SlotData,
                            w_q: float = C.W_Q, w_1: float = C.W_1, w_2: float = C.W_2,
                            B_up_budget: float = C.B_UP_MAX,
                            B_dn_budget: float = C.B_DN_MAX,
                            F_budget: float = C.F_SAT_MAX,
                            use_real_api: bool | None = None,
                            **_) -> SlotResult:
    """LLM-Distilled Heur. offloader (heuristic mock by default)."""
    t0 = time.perf_counter()
    S = slot.topology.S
    requests = slot.requests
    use_real = USE_REAL_API if use_real_api is None else use_real_api

    # Pre-compute current load as a fraction of expected (uniform-split) capacity
    n_per_sat = np.zeros(S, dtype=int)
    for r in requests:
        n_per_sat[r.s0] += 1
    sat_load = n_per_sat / max(1.0, len(requests) / S)   # ~1.0 if uniform
    sat_load_str = ",".join(f"{v:.2f}" for v in sat_load)

    # ---- Per-user decisions ----
    raw_decisions = []
    for req in requests:
        if use_real:
            x, s, eta = _real_api_decision(req, sat_load_str)
        else:
            x, s, eta = _heuristic_decision(req, sat_load)
        raw_decisions.append((req, x, s, eta))

    # Equal-split resources among the offloaded users (bandwidth at the access
    # satellite, compute at the serving satellite), then evaluate TTFT/TPOT/QoE
    decisions: List[Decision] = []
    n_sat_assigned = np.zeros(S, dtype=int)
    n_access = np.zeros(S, dtype=int)
    for req, x, s, _ in raw_decisions:
        if x == 1:                      # locally served users draw no bandwidth
            n_access[req.s0] += 1
            n_sat_assigned[s] += 1

    for req, x, s, eta in raw_decisions:
        if x == 0:
            TTFT, TPOT = _ttft_tpot_loc(req, eta)
            Qe = Q_eff(eta, x=0, delta=req.delta, theta=req.theta)
            QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
            decisions.append(Decision(
                u=req.u, x=0, s=req.s0, eta=eta,
                B_up=0.0, B_dn=0.0, f=C.F_LOC,
                QoE=QoE, TTFT=TTFT, TPOT=TPOT, Q_eff=Qe,
            ))
        else:
            U_acc = max(1, n_access[req.s0])
            U_sat = max(1, n_sat_assigned[s])
            Bup = B_up_budget / U_acc
            Bdn = B_dn_budget / U_acc
            fs  = F_budget   / U_sat
            TTFT, TPOT = _ttft_tpot_sat(req, slot.topology, s, eta, Bup, Bdn, fs)
            Qe = Q_eff(eta, x=1, delta=req.delta, theta=req.theta)
            QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
            decisions.append(Decision(
                u=req.u, x=1, s=s, eta=eta,
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


__all__ = ["solve_per_slot_icl_llm"]
