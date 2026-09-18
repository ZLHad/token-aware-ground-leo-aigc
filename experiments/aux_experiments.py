"""Auxiliary experiments for Fig. 2(b)-(d) and the ISL sensitivity sentence.
  python -m experiments.aux_experiments w1     -> figures/data/aux_w1.json      (Q_eff & QoE vs w_1, U=30)
  python -m experiments.aux_experiments eta    -> figures/data/aux_eta_nu.json  (eta* vs nu for the four task profiles)
  python -m experiments.aux_experiments dual   -> figures/data/aux_dual.json    (D^(t), P^(t) per iteration, U in {15,30,40})
  python -m experiments.aux_experiments isl    -> figures/data/aux_isl.json     (r_ISL in {100,50,20} Mbps, U=30)
"""
from __future__ import annotations
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from src.data_gen import make_slot, ring_topology, profile_of, Request, SlotData
from src.inner_solve import solve_inner_sat, _ttft_tpot_sat
from src.optimizer import solve_per_slot
from src.baselines.fixed_eta_dijkstra import solve_per_slot_fixed_eta
from src.baselines.profile_oblivious import solve_per_slot_profile_oblivious
from src.baselines.icl_llm import solve_per_slot_icl_llm
from src.baselines.td3_rl import solve_per_slot_td3
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(HERE, "figures", "data"); os.makedirs(DATA, exist_ok=True)
SCHEMES = {"Proposed": solve_per_slot, "Profile-Oblivious": solve_per_slot_profile_oblivious, "Rule-Based": solve_per_slot_icl_llm,
           "TD3-RL": solve_per_slot_td3, "Fixed-eta": solve_per_slot_fixed_eta}

def _w1_task(args):
    name, w1, seed, n_slots, U = args
    rng = np.random.default_rng(seed); q, qe, tt = [], [], []
    for _ in range(n_slots):
        slot = make_slot(U, C.S_DEFAULT, rng); res = SCHEMES[name](slot, w_1=w1)
        q += [d.QoE for d in res.decisions]; qe += [d.Q_eff for d in res.decisions]; tt += [d.TTFT for d in res.decisions]
    return dict(scheme=name, w1=w1, seed=seed, QoE=float(np.mean(q)), Qeff=float(np.mean(qe)), TTFT=float(np.mean(tt)))

def run_w1(U=30, seeds=3, n_slots=20, procs=10):
    w1s = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    tasks = [(n, w, s, n_slots, U) for n in SCHEMES for w in w1s for s in range(seeds)]
    with Pool(procs) as p: rows = p.map(_w1_task, tasks)
    json.dump(dict(U=U, w1s=w1s, rows=rows), open(os.path.join(DATA, "aux_w1.json"), "w"))
    for n in SCHEMES:
        print(f"{n:18s} " + " ".join(f"w1={w}: Qeff {np.mean([r['Qeff'] for r in rows if r['scheme']==n and r['w1']==w]):.3f}" for w in w1s))

def run_eta():
    """eta* on the satellite branch vs compute price nu, for a representative request of each task
    (median n_txt/n_vis of that task; mean SNR), mu fixed at their typical converged values."""
    from src.data_gen import load_calib_table
    tab = load_calib_table()["requests"]; topo = ring_topology(C.S_DEFAULT)
    nus = np.logspace(-3, 1, 25); out = {}
    for t in ["vqav2", "textvqa", "docvqa", "chartqa"]:
        rs = [r for r in tab if r["task"] == t]; d, th = profile_of(t)
        req = Request(u=0, n_txt=int(np.median([r["n_txt"] for r in rs])), n_vis=int(np.median([r["n_vis"] for r in rs])), n_out=60,
                      H=float(np.mean([r["H"] for r in rs])), delta=d, theta=th, s0=0, gamma_G2S=10 ** 1.5, gamma_S2G=10 ** 1.2, task=t)
        etas = [solve_inner_sat(req, topo, 0, mu_up=1e-8, mu_dn=2e-8, nu=float(nu), w_q=C.W_Q, w_1=C.W_1, w_2=C.W_2, eta_min=C.ETA_MIN).eta for nu in nus]
        out[t] = dict(delta=d, theta=th, n_vis=req.n_vis, etas=etas)
        print(f"{t:8s} (δ,θ)=({d:.2f},{th:.1f}) n_vis={req.n_vis}: eta* from {etas[0]:.3f} (nu={nus[0]:.0e}) to {etas[-1]:.3f} (nu={nus[-1]:.0e})")
    json.dump(dict(nus=nus.tolist(), curves=out), open(os.path.join(DATA, "aux_eta_nu.json"), "w"))

def _dual_task(args):
    U, seed = args; slot = make_slot(U, C.S_DEFAULT, np.random.default_rng(100 + seed))
    res = solve_per_slot(slot, return_history=True)
    return dict(U=U, seed=seed, D=res.dual_value_history, P=res.sum_QoE_history, gap=res.gap_history, iters=res.iters, wall=res.wall_time_s, cert=res.certificate)

def run_dual(seeds=5, procs=10):
    tasks = [(U, s) for U in (15, 30, 40) for s in range(seeds)]
    with Pool(procs) as p: rows = p.map(_dual_task, tasks)
    json.dump(rows, open(os.path.join(DATA, "aux_dual.json"), "w"))
    for U in (15, 30, 40):
        rr = [r for r in rows if r["U"] == U]
        print(f"U={U}: gap mean {np.mean([r['cert']['gap'] for r in rr])*100:.2f}% max {np.max([r['cert']['gap'] for r in rr])*100:.2f}%  iters {np.mean([r['iters'] for r in rr]):.0f}  wall {np.mean([r['wall'] for r in rr]):.1f}s")

def _isl_task(args):
    r_isl, seed, n_slots, U = args
    C.R_ISL = r_isl                          # per-hop slice (module-level, used by ring_topology and tau_up)
    rng = np.random.default_rng(seed); q, share, hops = [], [], []
    for _ in range(n_slots):
        slot = SlotData(topology=ring_topology(C.S_DEFAULT, R_single_hop=r_isl), requests=make_slot(U, C.S_DEFAULT, rng).requests)
        res = solve_per_slot(slot); req = {r.u: r for r in slot.requests}
        for d in res.decisions:
            q.append(d.QoE)
            if d.x == 1 and d.s != req[d.u].s0:
                nti = req[d.u].n_txt + d.eta * req[d.u].n_vis
                share.append(C.KAPPA_IN * nti / slot.topology.R_ISL_eff[req[d.u].s0, d.s] / d.TTFT); hops.append(int(slot.topology.hops[req[d.u].s0, d.s]))
    return dict(r_isl=r_isl, seed=seed, QoE=float(np.mean(q)), isl_share=float(np.mean(share)) if share else 0.0, multi_hop_frac=float(len(share) / len(q)))

def run_isl(U=30, seeds=3, n_slots=20, procs=9):
    tasks = [(r, s, n_slots, U) for r in (100e6, 50e6, 20e6) for s in range(seeds)]
    with Pool(procs) as p: rows = p.map(_isl_task, tasks)
    json.dump(rows, open(os.path.join(DATA, "aux_isl.json"), "w"))
    for r in (100e6, 50e6, 20e6):
        rr = [x for x in rows if x["r_isl"] == r]
        print(f"r_ISL={r/1e6:.0f} Mbps: QoE {np.mean([x['QoE'] for x in rr]):.3f}  ISL-upload share of TTFT {np.mean([x['isl_share'] for x in rr])*100:.1f}%  routed-via-ISL frac {np.mean([x['multi_hop_frac'] for x in rr]):.2f}")

if __name__ == "__main__":
    {"w1": run_w1, "eta": run_eta, "dual": run_dual, "isl": run_isl}[sys.argv[1]]()
