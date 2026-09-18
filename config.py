"""Global configuration for the TVT correspondence simulation (resubmission, 2026-09).

Symbol naming follows main.tex (see code/README.md §2.3). All physical units
are SI unless stated otherwise:

    bandwidth        : Hz
    rate / capacity  : bits per second
    compute          : TFLOPS  (10^12 FLOPs / s)
    time             : seconds
    token            : count
    kappa_*          : bits per token
    zeta_pf, beta_*  : seconds * TFLOPS per token  (so that *_/f_u gives seconds)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
#  Token byte-rate (source-coded payload size per token)
# ---------------------------------------------------------------------------
KAPPA_IN: float = 4_000.0    # bits per input token (visual token after source coding)
KAPPA_OUT: float = 200.0     # bits per output token (textual)


# ---------------------------------------------------------------------------
#  Model TTFT/TPOT coefficients, set from model size (2 FLOP per parameter
#  per token) so that zeta * n / f is seconds when f is in TFLOPS:
#    sat  : Qwen3-VL-32B  -> 2 x 32e9 = 64 GFLOP/token = 6.4e-2 s*TFLOPS/token
#    local: Qwen3-VL-2B backbone (1.4e9 non-embedding params) -> 2.8e-3;
#           the vision encoder is charged once in the user-side scoring pass
#           (ZETA_SC below) and its output is reused by the local model.
#  Decode is memory-bound in practice; beta_0 sets the per-token service time
#  at the reference compute share (structure measured in
#  calibration/data/timing_sweep.json: prefill linear in tokens, decode
#  per-token time almost independent of context length).
# ---------------------------------------------------------------------------
ZETA_PF_SAT: float = 6.4e-2      # s * TFLOPS / token   (prefill, 32B)
BETA_0_SAT: float  = 0.9         # s * TFLOPS           (decode per-token, ~45 ms at f=20 TFLOPS)
BETA_1_SAT: float  = 1e-4        # s * TFLOPS / token   (context-length slope, small)

ZETA_PF_LOC: float = 2.8e-3      # backbone-only prefill (post-encoder)
BETA_0_LOC: float  = 0.16        # ~40 ms/token at f_loc = 4 TFLOPS
BETA_1_LOC: float  = 1e-5

# User-side scoring pass (vision encoder + first 12 backbone layers of the 2B
# model): FLOPs ratio r = 0.70 of the full local prefill incl. the encoder
# (5.5 GFLOP per visual token), i.e. 3.8 GFLOP per visual token.
ZETA_SC: float     = 3.8e-3      # s * TFLOPS / visual token


# ---------------------------------------------------------------------------
#  Compute budgets
# ---------------------------------------------------------------------------
F_SAT_MAX: float   = 100.0       # TFLOPS effective serving throughput per satellite
F_LOC: float       = 4.0         # TFLOPS effective per UE (mobile NPU, FP16 transformer prefill)


# ---------------------------------------------------------------------------
#  Bandwidth budgets (Hz)
# ---------------------------------------------------------------------------
B_UP_MAX: float    = 100e6       # 100 MHz total per access satellite
B_DN_MAX: float    = 50e6        # 50 MHz total per access satellite


# ---------------------------------------------------------------------------
#  ISL: per-session per-hop slice and link capacity (occupancy is reported)
# ---------------------------------------------------------------------------
R_ISL: float       = 100e6       # per-hop per-session slice r^ISL (bits/s)
R_ISL_CAP: float   = 1e9         # link capacity R_e^max (bits/s) -> 10 sessions per link


# ---------------------------------------------------------------------------
#  Propagation delays (constant under fixed association and route)
# ---------------------------------------------------------------------------
T_PROP_G2S: float  = 4e-3        # access link, one way (LEO slant range ~1200 km)
T_PROP_ISL_HOP: float = 20e-3    # per ISL hop in the 6-satellite ring (~6000 km)


# ---------------------------------------------------------------------------
#  Model quality bounds
# ---------------------------------------------------------------------------
Q_SAT: float       = 1.00
Q_LOC: float       = 0.55


# ---------------------------------------------------------------------------
#  QoE weights (w_q log(1+Q) - w_1 TTFT - w_2 TPOT)
# ---------------------------------------------------------------------------
W_Q: float         = 5.0
W_1: float         = 1.0
W_2: float         = 10.0


# ---------------------------------------------------------------------------
#  Compression bounds. eta_0 = 0.05 is a technical floor only: the calibrated
#  profiles (calib_table.json) carry the task-dependent quality loss, and the
#  marginal-balance condition decides how far each request is compressed.
# ---------------------------------------------------------------------------
ETA_MIN: float     = 0.05
ETA_MAX: float     = 1.0


# ---------------------------------------------------------------------------
#  Request tuple distribution: (n_txt, n_vis, H, task) are resampled from the
#  calibration table (440 real requests over VQAv2 / TextVQA / ChartQA /
#  DocVQA scored by Qwen3-VL-2B); the profile (delta, theta) is looked up by
#  task class from the same table. n_out is drawn uniformly (short-answer
#  benchmarks do not represent AIGC output lengths).
# ---------------------------------------------------------------------------
CALIB_TABLE: str               = "data/calib_table.json"
N_TXT_RANGE: Tuple[int, int]   = (4, 40)       # observed range (state normalization only)
N_VIS_RANGE: Tuple[int, int]   = (256, 1024)
N_OUT_RANGE: Tuple[int, int]   = (20, 100)
LAMBDA_ARRIVAL: float          = 0.8     # Poisson arrival rate per user per slot


# ---------------------------------------------------------------------------
#  Channel SNR (lognormal in dB)
# ---------------------------------------------------------------------------
GAMMA_G2S_DB_MEAN: float       = 15.0
GAMMA_G2S_DB_STD: float        = 3.0
GAMMA_S2G_DB_MEAN: float       = 12.0
GAMMA_S2G_DB_STD: float        = 3.0


# ---------------------------------------------------------------------------
#  Dual-decomposition hyperparameters
#
#  Updates are performed in log-space (geometric step) on budget-normalized
#  violations, which is mathematically equivalent to a sub-gradient ascent on
#  log(mu) and remains numerically stable when initial violations span many
#  orders of magnitude:
#
#      mu_up <- max(eps_floor_mu, mu_up * exp(rho_mu * clip(viol_norm, -clip, +clip)))
#      nu    <- max(eps_floor_nu, nu    * exp(rho_nu * clip(viol_norm, -clip, +clip)))
#
#  Natural scales: mu_*  ~ O(1e-8) for B in 1e8 Hz; nu ~ O(0.1-1) for f in O(1) TFLOPS.
# ---------------------------------------------------------------------------
EPS_FLOOR_MU: float       = 1e-12   # strict-positivity floor for mu^up, mu^dn
EPS_FLOOR_NU: float       = 1e-4    # strict-positivity floor for nu
MAX_DUAL_ITER: int        = 600     # I_max
GAP_TOL: float            = 2e-2    # eps_g: stop when (D_min - P_max)/|D_min| <= eps_g
PRIMAL_DUAL_TOL: float    = GAP_TOL # backward-compat alias
VIOL_TOL: float           = 2e-2    # unused; kept for backward-compat
RHO_MU_INIT: float        = 0.50    # log-space step for mu^up, mu^dn (alpha_0)
RHO_NU_INIT: float        = 0.50    # log-space step for nu
RHO_DECAY: float          = 0.985   # step decay omega per iteration
VIOL_CLIP: float          = 3.0     # clip normalized violation magnitude
DUAL_AVG_WINDOW: int      = 8       # trailing-window size for dual averaging
                                     # also used for convergence-window check

# Backward-compat alias (small floor for legacy paths that don't distinguish)
EPS_FLOOR: float          = EPS_FLOOR_MU


# ---------------------------------------------------------------------------
#  eta bisection tolerance
# ---------------------------------------------------------------------------
ETA_BISECT_TOL: float     = 1e-6
ETA_BISECT_MAX_ITER: int  = 100


# ---------------------------------------------------------------------------
#  Default network sizes (per main.tex §IV-A)
# ---------------------------------------------------------------------------
U_DEFAULT: int            = 30
S_DEFAULT: int            = 6


# ---------------------------------------------------------------------------
#  Simulation horizon
# ---------------------------------------------------------------------------
N_SLOTS: int              = 100
N_SEEDS: int              = 5


# ---------------------------------------------------------------------------
#  Fig 1(c) — three semantic profiles with widened (delta, theta) spread
#  to make the three curves visually well-separated under eta_min=0.3.
# ---------------------------------------------------------------------------
FIG1C_PROFILES: Tuple[Tuple[float, float, str], ...] = (
    (0.20, 3.0, "Low sensitivity"),
    (0.55, 1.8, "Medium sensitivity"),
    (0.95, 1.5, "High sensitivity"),
)
FIG1C_NU_LOG_RANGE: Tuple[float, float, int] = (-3.0, 1.0, 22)  # log10(nu), wider range


# ---------------------------------------------------------------------------
#  Convenience bundle for a single "scenario" (used by data_gen)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """One macro-scenario passed around the optimizer/baselines."""
    U: int = U_DEFAULT
    S: int = S_DEFAULT
    n_slots: int = N_SLOTS
    seed: int = 0
    w_q: float = W_Q
    w_1: float = W_1
    w_2: float = W_2
    eta_min: float = ETA_MIN


__all__ = [
    "KAPPA_IN", "KAPPA_OUT",
    "ZETA_PF_SAT", "BETA_0_SAT", "BETA_1_SAT",
    "ZETA_PF_LOC", "BETA_0_LOC", "BETA_1_LOC",
    "ZETA_SC", "F_SAT_MAX", "F_LOC", "B_UP_MAX", "B_DN_MAX", "R_ISL", "R_ISL_CAP",
    "T_PROP_G2S", "T_PROP_ISL_HOP", "CALIB_TABLE", "GAP_TOL",
    "Q_SAT", "Q_LOC", "W_Q", "W_1", "W_2",
    "ETA_MIN", "ETA_MAX",
    "N_TXT_RANGE", "N_VIS_RANGE", "N_OUT_RANGE", "LAMBDA_ARRIVAL",
    "GAMMA_G2S_DB_MEAN", "GAMMA_G2S_DB_STD",
    "GAMMA_S2G_DB_MEAN", "GAMMA_S2G_DB_STD",
    "EPS_FLOOR", "EPS_FLOOR_MU", "EPS_FLOOR_NU",
    "MAX_DUAL_ITER", "PRIMAL_DUAL_TOL", "VIOL_TOL",
    "RHO_MU_INIT", "RHO_NU_INIT", "RHO_DECAY", "VIOL_CLIP", "DUAL_AVG_WINDOW",
    "ETA_BISECT_TOL", "ETA_BISECT_MAX_ITER",
    "U_DEFAULT", "S_DEFAULT", "N_SLOTS", "N_SEEDS",
    "FIG1C_PROFILES", "FIG1C_NU_LOG_RANGE",
    "Scenario",
]
