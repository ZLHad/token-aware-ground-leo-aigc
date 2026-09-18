"""Export the tables of the paper and of the supplementary material as CSV files.

Reads figures/data/run_all.json(.gz), figures/data/aux_w1.json, figures/data/aux_isl.json,
and data/calib_table.json, and writes results/*.csv (plus a flat per-user record file).
Usage: python -m experiments.export_tables
"""
from __future__ import annotations
import csv, gzip, json, os, collections
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "figures", "data"); OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
ORDER = ["Proposed", "Profile-Oblivious", "Rule-Based", "TD3-RL", "Fixed-eta"]
CLASSES = ["vqav2", "textvqa", "docvqa", "chartqa"]
U0 = 30

p = os.path.join(D, "run_all.json")
run = json.load(open(p)) if os.path.exists(p) else json.load(gzip.open(p + ".gz", "rt"))
S, raw, U_list = run["summary"], run["raw"], run["config"]["U_list"]

def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"{name}: {len(rows)} rows")

def per_user(scheme, U):
    return [pu for r in raw if r["scheme"] == scheme and r["U"] == U for pu in r["per_user"]]

# Table II
base = S[f"Proposed|{U0}"]["QoE_mean"]; rows = []
for s in ORDER:
    m = S[f"{s}|{U0}"]
    rows.append([s, f"{m['QoE_mean']:.3f}", f"{m['QoE_p5']:.3f}", f"{m['below_local']:.2f}", f"{m['TTFT_mean']*1e3:.0f}", f"{m['TPOT_mean']*1e3:.1f}",
                 "" if s == "Proposed" else f"{(base/m['QoE_mean']-1)*100:.1f}"])
write("table2_performance_U30.csv", ["scheme", "QoE", "QoE_5pct", "below_local_share", "TTFT_ms", "TPOT_ms", "gain_of_proposed_pct"], rows)

# Gains by load
rows = [[U] + [f"{(S[f'Proposed|{U}']['QoE_mean']/S[f'{s}|{U}']['QoE_mean']-1)*100:.1f}" for s in ORDER[1:]] for U in U_list]
write("gains_by_U.csv", ["U"] + [f"gain_over_{s}_pct" for s in ORDER[1:]], rows)

# Table S-IV: per class at U0
calib = json.load(open(os.path.join(HERE, "data", "calib_table.json")))
delta = {c: (calib.get("task_profiles", calib).get(c, {}) or {}).get("delta") for c in CLASSES}
lk = json.load(open(os.path.join(HERE, "calibration", "data", "profile_lookup_final.json")))["profiles"]
rows = []
for c in CLASSES:
    pr = [pu for pu in per_user("Proposed", U0) if pu["task"] == c]; po = [pu for pu in per_user("Profile-Oblivious", U0) if pu["task"] == c]
    q1, q0 = np.mean([x["QoE"] for x in pr]), np.mean([x["QoE"] for x in po])
    rows.append([c, f"{lk[f'task:{c}']['delta']:.2f}", f"{lk[f'task:{c}']['theta']:.1f}", f"{np.mean([x['eta'] for x in pr]):.2f}", f"{q1:.3f}",
                 f"{np.mean([x['eta'] for x in po]):.2f}", f"{q0:.3f}", f"{(q1/q0-1)*100:.1f}", len(pr)])
write("tableS4_per_class_U30.csv", ["class", "delta", "theta", "eta_mean_proposed", "QoE_proposed", "eta_mean_profile_oblivious", "QoE_profile_oblivious", "gain_pct", "n_users"], rows)

# Table S-V: ISL slice sensitivity (+ peak sessions from the main sweep)
isl = json.load(open(os.path.join(D, "aux_isl.json"))); by = collections.defaultdict(list)
for r in isl: by[float(r["r_isl"])].append(r)
peak_U0 = max(r["link_max"] for r in raw if r["scheme"] == "Proposed" and r["U"] == U0)
peak_all = max(r["link_max"] for r in raw if r["scheme"] == "Proposed")
rows = [[f"{k/1e6:.0f}", f"{np.mean([float(r['QoE']) for r in v]):.3f}", f"{np.mean([float(r['isl_share']) for r in v])*100:.1f}",
         f"{np.mean([float(r['multi_hop_frac']) for r in v])*100:.0f}", peak_U0 if k == 100e6 else "", peak_all if k == 100e6 else ""] for k, v in sorted(by.items(), reverse=True)]
write("tableS5_isl_slice_U30.csv", ["r_isl_Mbps", "QoE", "isl_share_of_TTFT_pct", "forwarded_requests_pct", "peak_sessions_per_link_U30", "peak_sessions_per_link_all_loads"], rows)

# Table S-VI: tail and share below local fallback
rows = []
for U in U_list:
    for s in ORDER:
        rows.append([U, s, f"{S[f'{s}|{U}']['QoE_p5']+0:.2f}".replace('-0.00','0.00'), f"{S[f'{s}|{U}']['below_local']:.2f}"])
    rows.append([U, "Local fallback", f"{np.percentile([pu['QoE_loc'] for pu in per_user('Proposed', U)], 5):.2f}", ""])
write("tableS6_tail_by_U.csv", ["U", "scheme", "QoE_5pct", "below_local_share"], rows)

# Table S-VII: certified gap at termination (Proposed)
rows = []
for U in U_list:
    rr = [r for r in raw if r["scheme"] == "Proposed" and r["U"] == U]
    g = np.concatenate([r["gaps"] for r in rr]) * 100; it = np.concatenate([r["iters"] for r in rr])
    rows.append([U, f"{np.median(g):.1f}", f"{g.mean():.1f}", f"{np.percentile(g, 95):.1f}", f"{g.max():.1f}", f"{(g > 2).mean()*100:.0f}", f"{np.median(it):.0f}", f"{it.mean():.0f}", len(g)])
write("tableS7_certified_gap.csv", ["U", "gap_median_pct", "gap_mean_pct", "gap_p95_pct", "gap_max_pct", "share_above_2pct", "iterations_median", "iterations_mean", "n_slots"], rows)

# Table S-VIII: Q_eff against w1 at U0
w1 = json.load(open(os.path.join(D, "aux_w1.json"))); by = collections.defaultdict(list)
for r in w1["rows"]: by[(r["scheme"], float(r["w1"]))].append(r)
rows = [[s] + [f"{np.mean([r['Qeff'] for r in by[(s, w)]]):.3f}" for w in w1["w1s"]] for s in ORDER]
write("tableS8_qeff_vs_w1_U30.csv", ["scheme"] + [f"w1={w:g}" for w in w1["w1s"]], rows)

# Flat per-user records of the main sweep
with gzip.open(os.path.join(OUT, "per_user_records.csv.gz"), "wt", newline="") as f:
    w = csv.writer(f); w.writerow(["scheme", "U", "seed", "task", "QoE", "QoE_local", "Qeff", "TTFT_s", "TPOT_s", "eta", "offloaded", "hops"]); n = 0
    for r in raw:
        for pu in r["per_user"]:
            w.writerow([r["scheme"], r["U"], r["seed"], pu["task"], f"{pu['QoE']:.4f}", f"{pu['QoE_loc']:.4f}", f"{pu['Qeff']:.4f}", f"{pu['TTFT']:.4f}", f"{pu['TPOT']:.5f}", f"{pu['eta']:.3f}", pu["x"], pu["hops"]]); n += 1
print(f"per_user_records.csv.gz: {n} rows")
