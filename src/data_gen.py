"""Data generation: requests, channels, satellite topology.

Symbols follow main.tex (§II-A, §II-B, §II-C).

Topology (main.tex §II-A): ring of S satellites, ISL graph G=(S,E),
shortest-path hop count -> effective end-to-end ISL rate
    R^{ISL}_{s_0,s} = R_ISL_single_hop / max(1, hops(s_0, s))

Access satellite assignment (main.tex §II-A): users are spread uniformly
along the ground track of the ring and associate with the nearest (largest
elevation) satellite; the association is fixed within the slot and is not an
optimization variable.

Request tuple (main.tex §II-B): (n_txt, n_vis, n_out, task, H_u) is resampled
from the calibration table (config.CALIB_TABLE: 440 real requests scored by
the 2B model); xi_u = (delta_u, theta_u) is the calibrated profile of the
request's task class. n_out ~ Uniform(N_OUT_RANGE).

Propagation delays: constant per access link and per ISL hop (config).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

import json
import os

import config as C


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------
@dataclass
class Request:
    """One user's request at one slot."""
    u: int
    n_txt: int
    n_vis: int
    n_out: int
    H: float
    delta: float
    theta: float
    s0: int                  # access satellite (nearest / largest elevation)
    gamma_G2S: float         # linear SNR
    gamma_S2G: float         # linear SNR
    task: str = ""           # task class from the calibration table


@dataclass
class Topology:
    """Satellite ring topology + Dijkstra shortest-path hop count matrix."""
    S: int
    hops: np.ndarray         # (S, S) integer hop count, 0 on diagonal
    R_ISL_eff: np.ndarray    # (S, S) per-session path rate r^ISL/hops; diag = +inf (no ISL needed)
    T_prop_path: np.ndarray = None  # (S, S) propagation delay along the path (s); diag = 0


# ---------------------------------------------------------------------------
#  Topology
# ---------------------------------------------------------------------------
def ring_topology(S: int, R_single_hop: float = C.R_ISL) -> Topology:
    """Ring of S satellites; hop count = min(|i-j|, S-|i-j|)."""
    idx = np.arange(S)
    diff = np.abs(idx[:, None] - idx[None, :])
    hops = np.minimum(diff, S - diff).astype(int)
    # Effective end-to-end rate (store-and-forward over hops)
    R = np.full((S, S), np.inf, dtype=float)
    nondiag = hops > 0
    R[nondiag] = R_single_hop / hops[nondiag]
    T_prop = hops.astype(float) * C.T_PROP_ISL_HOP
    return Topology(S=S, hops=hops, R_ISL_eff=R, T_prop_path=T_prop)


# ---------------------------------------------------------------------------
#  Calibration table (loaded once)
# ---------------------------------------------------------------------------
_CALIB = None


def load_calib_table():
    """Load config.CALIB_TABLE (relative to the code/ directory) once."""
    global _CALIB
    if _CALIB is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, C.CALIB_TABLE)) as f:
            _CALIB = json.load(f)
    return _CALIB


def profile_of(task: str):
    p = load_calib_table()["profiles"][task]
    return float(p["delta"]), float(p["theta"])


def global_profile():
    p = load_calib_table()["global_profile"]
    return float(p["delta"]), float(p["theta"])


# ---------------------------------------------------------------------------
#  Request generator
# ---------------------------------------------------------------------------
def sample_snr_lognormal(rng: np.random.Generator, n: int,
                          mu_db: float, sigma_db: float) -> np.ndarray:
    """Sample linear SNR from a lognormal distribution defined in dB."""
    snr_db = rng.normal(loc=mu_db, scale=sigma_db, size=n)
    return np.power(10.0, snr_db / 10.0)


def generate_requests(U: int, S: int,
                      rng: np.random.Generator) -> List[Request]:
    """Generate one slot of U requests: real (n_txt, n_vis, H, task) resampled
    from the calibration table, nearest-satellite association, lognormal SNRs."""
    table = load_calib_table()["requests"]
    picks = rng.integers(0, len(table), size=U)
    n_txt = np.array([table[i]["n_txt"] for i in picks])
    n_vis = np.array([table[i]["n_vis"] for i in picks])
    H = np.array([table[i]["H"] for i in picks])
    tasks = [table[i]["task"] for i in picks]
    prof = [profile_of(t) for t in tasks]
    delta = np.array([p[0] for p in prof]); theta = np.array([p[1] for p in prof])
    n_out = rng.integers(C.N_OUT_RANGE[0], C.N_OUT_RANGE[1] + 1, size=U)
    # users uniform along the ground track; access = nearest of S equally spaced satellites
    phi = rng.uniform(0.0, 2.0 * np.pi, size=U)
    s0 = np.rint(phi / (2.0 * np.pi / S)).astype(int) % S
    gamma_G2S = sample_snr_lognormal(rng, U, C.GAMMA_G2S_DB_MEAN, C.GAMMA_G2S_DB_STD)
    gamma_S2G = sample_snr_lognormal(rng, U, C.GAMMA_S2G_DB_MEAN, C.GAMMA_S2G_DB_STD)

    return [
        Request(
            u=int(u),
            n_txt=int(n_txt[u]),
            n_vis=int(n_vis[u]),
            n_out=int(n_out[u]),
            H=float(H[u]),
            delta=float(delta[u]),
            theta=float(theta[u]),
            s0=int(s0[u]),
            gamma_G2S=float(gamma_G2S[u]),
            gamma_S2G=float(gamma_S2G[u]),
            task=tasks[u],
        )
        for u in range(U)
    ]


# ---------------------------------------------------------------------------
#  One-shot scenario bundle
# ---------------------------------------------------------------------------
@dataclass
class SlotData:
    """Everything one solver needs for a single scheduling slot."""
    topology: Topology
    requests: List[Request]


def make_slot(U: int, S: int, rng: np.random.Generator) -> SlotData:
    return SlotData(topology=ring_topology(S), requests=generate_requests(U, S, rng))


__all__ = [
    "Request", "Topology", "SlotData",
    "ring_topology", "sample_snr_lognormal", "load_calib_table", "profile_of", "global_profile",
    "generate_requests", "make_slot",
]
