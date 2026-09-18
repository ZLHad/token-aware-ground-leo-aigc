"""Bottleneck margin of the TPOT max-model and recovery-step bookkeeping (R3-2, R1-4).

For the Proposed scheme, at the returned (recovered) point of every slot:
  * ratio_u = max(T_isl_dn_per, T_S2G_per) / TPOT_sat for each offloaded user -> how far the
    return stages are from taking over the max in eq. (7);
  * whether the max-argument changed between the raw dual point and the recovered point
    (a "bottleneck switch" caused by scaling f or B_dn);
  * the per-satellite scaling factors c_s applied by the proportional recovery.
Output: figures/data/aux_margins.json
Usage: python -m experiments.aux_margins [--U 10,...] [--seeds 5] [--slots 10] [--procs 8]
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import src.optimizer as OPT
from src.data_gen import make_slot
from src.inner_solve import n_tilde, n_out_eff

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(HERE, "figures", "data", "aux_margins.json")

# ---- record every (raw -> recovered) pair of the projection step; the returned point is one of them
_PAIRS: list = []
_orig_project = OPT._project_to_budgets
def _recording_project(decisions, *a, **k):
    out = _orig_project(decisions, *a, **k); _PAIRS.append((out, decisions)); return out
OPT._project_to_budgets = _recording_project

def stages(req, topo, d):
    """Per-token return-stage delays and the decoding interval for an offloaded decision."""
    nti = n_tilde(req, d.eta)
    T_S2G = C.KAPPA_OUT / (d.B_dn * math.log2(1.0 + req.gamma_S2G))
    T_isl = 0.0 if d.s == req.s0 else C.KAPPA_OUT / topo.R_ISL_eff[d.s, req.s0]
    TPOT_sat = (C.BETA_0_SAT + C.BETA_1_SAT * (nti + n_out_eff(req))) / d.f
    return T_S2G, T_isl, TPOT_sat

def run_task(args):
    U, seed, n_slots, S = args
    rng = np.random.default_rng(seed); rows = []
    for k in range(n_slots):
        slot = make_slot(U, S, rng); _PAIRS.clear()
        res = OPT.solve_per_slot(slot)
        pre = next((p for (o, p) in _PAIRS if o is res.decisions), None)
        req = {r.u: r for r in slot.requests}; topo = slot.topology
        pre_by_u = {d.u: d for d in pre} if pre is not None else {}
        ratios, switches, n_off, argmax_post = [], 0, 0, {"sat": 0, "s2g": 0, "isl": 0}
        c_f = {}; c_up = {}; c_dn = {}
        for d in res.decisions:
            if d.x != 1: continue
            n_off += 1; r = req[d.u]
            s2g, isl, tp = stages(r, topo, d); ratios.append(max(s2g, isl) / tp)
            am = "sat" if tp >= max(s2g, isl) else ("s2g" if s2g >= isl else "isl"); argmax_post[am] += 1
            if d.u in pre_by_u:
                p = pre_by_u[d.u]; s2g0, isl0, tp0 = stages(r, topo, p)
                am0 = "sat" if tp0 >= max(s2g0, isl0) else ("s2g" if s2g0 >= isl0 else "isl")
                if am0 != am: switches += 1
                if p.f > 0: c_f[d.s] = d.f / p.f
                if p.B_up > 0: c_up[r.s0] = d.B_up / p.B_up
                if p.B_dn > 0: c_dn[r.s0] = d.B_dn / p.B_dn
        rows.append(dict(U=U, seed=seed, slot=k, n_off=n_off, ratio_max=float(max(ratios)) if ratios else None,
                         ratio_p99=float(np.percentile(ratios, 99)) if ratios else None, ratio_mean=float(np.mean(ratios)) if ratios else None,
                         switches=switches, argmax_post=argmax_post, pre_found=pre is not None,
                         c_f_min=float(min(c_f.values())) if c_f else 1.0, c_up_min=float(min(c_up.values())) if c_up else 1.0,
                         c_dn_min=float(min(c_dn.values())) if c_dn else 1.0,
                         n_sat_scaled=int(sum(v < 0.999 for v in c_f.values())), gap=res.certificate.get("gap")))
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--U", default="10,15,20,25,30,35,40"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--slots", type=int, default=10); ap.add_argument("--procs", type=int, default=8); a = ap.parse_args()
    U_list = [int(x) for x in a.U.split(",")]
    tasks = sorted([(U, s, a.slots, C.S_DEFAULT) for U in U_list for s in range(a.seeds)], key=lambda t: -t[0])
    t0 = time.time(); rows = []
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(run_task, tasks)):
            rows += r; print(f"  [{i+1}/{len(tasks)}] U={r[0]['U']} seed={r[0]['seed']} ({time.time()-t0:.0f}s)", flush=True)
    summ = {}
    for U in U_list:
        R = [r for r in rows if r["U"] == U]
        summ[U] = dict(n_slots=len(R), n_off=int(sum(r["n_off"] for r in R)),
                       ratio_max=float(max(r["ratio_max"] for r in R if r["ratio_max"] is not None)),
                       ratio_p99=float(np.percentile([r["ratio_p99"] for r in R if r["ratio_p99"] is not None], 99)),
                       ratio_mean=float(np.mean([r["ratio_mean"] for r in R if r["ratio_mean"] is not None])),
                       switches=int(sum(r["switches"] for r in R)), non_sat_argmax=int(sum(r["argmax_post"]["s2g"] + r["argmax_post"]["isl"] for r in R)),
                       pre_found=float(np.mean([r["pre_found"] for r in R])),
                       c_f_min=float(min(r["c_f_min"] for r in R)), c_f_p5=float(np.percentile([r["c_f_min"] for r in R], 5)), c_f_median=float(np.median([r["c_f_min"] for r in R])),
                       c_up_min=float(min(r["c_up_min"] for r in R)), c_dn_min=float(min(r["c_dn_min"] for r in R)),
                       frac_slots_scaled=float(np.mean([r["n_sat_scaled"] > 0 for r in R])))
    json.dump(dict(config=dict(U_list=U_list, seeds=a.seeds, slots=a.slots), summary=summ, rows=rows), open(OUT, "w"), indent=1)
    print(f"\n{'U':>3s} {'off':>5s} {'ratio_max':>9s} {'p99':>7s} {'mean':>7s} {'switch':>6s} {'nonsat':>6s} {'c_f_min':>7s} {'c_f_p5':>7s} {'c_f_med':>7s} {'c_up_min':>8s} {'c_dn_min':>8s} {'scaled':>6s}")
    for U in U_list:
        v = summ[U]; print(f"{U:3d} {v['n_off']:5d} {v['ratio_max']:9.4f} {v['ratio_p99']:7.4f} {v['ratio_mean']:7.4f} {v['switches']:6d} {v['non_sat_argmax']:6d} {v['c_f_min']:7.3f} {v['c_f_p5']:7.3f} {v['c_f_median']:7.3f} {v['c_up_min']:8.3f} {v['c_dn_min']:8.3f} {v['frac_slots_scaled']:6.2f}")
    print("->", OUT)
