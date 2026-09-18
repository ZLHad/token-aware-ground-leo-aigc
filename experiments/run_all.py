"""E5 main sweep: five schemes x U-list x seeds, common random numbers per (U, seed).

Collects per-user QoE / TTFT / TPOT / Q_eff / eta / x per task, fairness statistics,
the Proposed scheme's certificate (gap, iterations, wall-clock) and ISL link occupancy.
Output: figures/data/run_all.json
Usage: python -m experiments.run_all [--U 10,15,...] [--seeds 5] [--slots 50] [--procs 10]
"""
from __future__ import annotations
import argparse, json, os, sys, time, collections
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from src.data_gen import make_slot
from src.inner_solve import solve_inner_loc
from src.optimizer import solve_per_slot
from src.baselines.fixed_eta_dijkstra import solve_per_slot_fixed_eta
from src.baselines.profile_oblivious import solve_per_slot_profile_oblivious
from src.baselines.icl_llm import solve_per_slot_icl_llm
from src.baselines.td3_rl import solve_per_slot_td3

SCHEMES = {"Proposed": solve_per_slot, "Profile-Oblivious": solve_per_slot_profile_oblivious,
           "Rule-Based": solve_per_slot_icl_llm, "TD3-RL": solve_per_slot_td3, "Fixed-eta": solve_per_slot_fixed_eta}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(HERE, "figures", "data", "run_all.json")

def ring_links(s0: int, s: int, S: int):
    """Directed links (i -> j) on the shorter ring path from s0 to s (ties: clockwise)."""
    if s0 == s: return []
    cw = (s - s0) % S; ccw = (s0 - s) % S
    step = 1 if cw <= ccw else -1; links = []; i = s0
    while i != s:
        j = (i + step) % S; links.append((i, j)); i = j
    return links

def run_task(args):
    name, U, seed, n_slots, S = args
    fn = SCHEMES[name]; rng = np.random.default_rng(seed)
    per_user = []   # dicts
    gaps, iters, walls, link_max = [], [], [], 0
    for k in range(n_slots):
        slot = make_slot(U, S, rng)
        res = fn(slot)
        loc = {r.u: solve_inner_loc(r, C.W_Q, C.W_1, C.W_2, eta_min=C.ETA_MIN).QoE for r in slot.requests}
        req = {r.u: r for r in slot.requests}
        occ = collections.Counter()
        for d in res.decisions:
            r = req[d.u]
            per_user.append(dict(task=r.task, QoE=d.QoE, TTFT=d.TTFT, TPOT=d.TPOT, Qeff=d.Q_eff, eta=d.eta, x=d.x,
                                 QoE_loc=loc[d.u], hops=int(slot.topology.hops[r.s0, d.s]) if d.x == 1 else 0))
            if d.x == 1:
                for l in ring_links(r.s0, d.s, S): occ[l] += 1
        link_max = max(link_max, max(occ.values()) if occ else 0)
        if name == "Proposed":
            gaps.append(res.certificate.get("gap", float("nan"))); iters.append(res.iters); walls.append(res.wall_time_s)
    return dict(scheme=name, U=U, seed=seed, per_user=per_user, gaps=gaps, iters=iters, walls=walls, link_max=link_max)

def summarize(rows):
    out = {}
    for row in rows:
        key = (row["scheme"], row["U"]); out.setdefault(key, []).append(row)
    summary = {}
    for (name, U), lst in out.items():
        pu = [d for row in lst for d in row["per_user"]]
        q = np.array([d["QoE"] for d in pu]); jain = float(q.sum() ** 2 / (len(q) * (q ** 2).sum())) if (q ** 2).sum() > 0 else float("nan")
        seed_means = [np.mean([d["QoE"] for d in row["per_user"]]) for row in lst]
        summary[f"{name}|{U}"] = dict(
            scheme=name, U=U, n_users=len(pu),
            QoE_mean=float(q.mean()), QoE_std_seeds=float(np.std(seed_means)), QoE_p5=float(np.percentile(q, 5)), QoE_min=float(q.min()), jain=jain,
            below_local=float(np.mean([d["QoE"] < d["QoE_loc"] - 1e-9 for d in pu])),
            TTFT_mean=float(np.mean([d["TTFT"] for d in pu])), TPOT_mean=float(np.mean([d["TPOT"] for d in pu])), Qeff_mean=float(np.mean([d["Qeff"] for d in pu])),
            offload_frac=float(np.mean([d["x"] for d in pu])),
            eta_by_task={t: float(np.mean([d["eta"] for d in pu if d["task"] == t])) for t in ["vqav2", "textvqa", "docvqa", "chartqa"]},
            QoE_by_task={t: float(np.mean([d["QoE"] for d in pu if d["task"] == t])) for t in ["vqav2", "textvqa", "docvqa", "chartqa"]},
            gap_mean=float(np.mean([g for row in lst for g in row["gaps"]])) if lst[0]["gaps"] else None,
            gap_max=float(np.max([g for row in lst for g in row["gaps"]])) if lst[0]["gaps"] else None,
            iters_mean=float(np.mean([i for row in lst for i in row["iters"]])) if lst[0]["iters"] else None,
            wall_mean=float(np.mean([w for row in lst for w in row["walls"]])) if lst[0]["walls"] else None,
            link_max=int(max(row["link_max"] for row in lst)),
        )
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--U", default="10,15,20,25,30,35,40"); ap.add_argument("--seeds", type=int, default=C.N_SEEDS)
    ap.add_argument("--slots", type=int, default=50); ap.add_argument("--procs", type=int, default=10); ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--out", default=OUT); a = ap.parse_args()
    U_list = [int(x) for x in a.U.split(",")]; names = a.schemes.split(",")
    tasks = [(n, U, s, a.slots, C.S_DEFAULT) for n in names for U in U_list for s in range(a.seeds)]
    # heavy tasks first
    tasks.sort(key=lambda t: (t[0] not in ("Proposed", "Profile-Oblivious"), -t[1]))
    print(f"{len(tasks)} tasks on {a.procs} procs; slots={a.slots} seeds={a.seeds}", flush=True); t0 = time.time()
    with Pool(a.procs) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(run_task, tasks)):
            rows.append(r); print(f"  [{i+1}/{len(tasks)}] {r['scheme']:18s} U={r['U']:2d} seed={r['seed']}  ({time.time()-t0:.0f}s)", flush=True)
    summ = summarize(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(config=dict(U_list=U_list, seeds=a.seeds, slots=a.slots, F=C.F_SAT_MAX, f_loc=C.F_LOC, w=(C.W_Q, C.W_1, C.W_2), eta0=C.ETA_MIN),
                   summary=summ, raw=[dict(scheme=r["scheme"], U=r["U"], seed=r["seed"], gaps=r["gaps"], iters=r["iters"], walls=r["walls"], link_max=r["link_max"],
                                          per_user=r["per_user"]) for r in rows]), open(a.out, "w"))
    print("\n" + f"{'scheme':18s} {'U':>3s} {'QoE':>7s} {'p5':>7s} {'Jain':>6s} {'<loc':>6s} {'TTFT':>6s} {'TPOT':>6s} {'offl':>5s} {'gap%':>6s} {'it':>4s}")
    for k in sorted(summ, key=lambda k: (summ[k]["U"], list(SCHEMES).index(summ[k]["scheme"]))):
        v = summ[k]; print(f"{v['scheme']:18s} {v['U']:3d} {v['QoE_mean']:7.3f} {v['QoE_p5']:7.3f} {v['jain']:6.3f} {v['below_local']:6.2f} {v['TTFT_mean']:6.2f} {v['TPOT_mean']*1000:6.1f} {v['offload_frac']:5.2f} " + (f"{v['gap_mean']*100:6.2f} {v['iters_mean']:4.0f}" if v['gap_mean'] is not None else ""))
    print("->", a.out)
