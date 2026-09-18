"""Fig. 2 (five panels, one row, print size) + Table I from the E5 outputs.
(a) QoE vs U  (b) Q_eff vs w_1  (c) eta* vs nu per task profile  (d) dual bound / recovered objective vs iteration  (e) calibrated retention curves
"""
import json, os, sys, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(HERE, "figures", "data"); OUTDIR = os.path.join(HERE, "figures")
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 5.8, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "lines.linewidth": 1.0, "lines.markersize": 3, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":", "legend.framealpha": 0.85})
ORDER = ["Proposed", "Profile-Oblivious", "Rule-Based", "TD3-RL", "Fixed-eta"]
LABEL = {"Proposed": "Proposed", "Profile-Oblivious": "Profile-Oblivious", "Rule-Based": "Rule-Based Offloader", "TD3-RL": "TD3-RL", "Fixed-eta": r"Fixed-$\eta$ Dijkstra"}
STY = {"Proposed": dict(color="C3", marker="o", ls="-"), "Profile-Oblivious": dict(color="C0", marker="s", ls="--"), "Rule-Based": dict(color="C2", marker="^", ls="-."),
       "TD3-RL": dict(color="C1", marker="D", ls=":"), "Fixed-eta": dict(color="C7", marker="v", ls="--")}
TASK = {"vqav2": "VQAv2 (scene)", "textvqa": "TextVQA (scene text)", "docvqa": "DocVQA (document)", "chartqa": "ChartQA (chart)"}
TCOL = {"vqav2": "C0", "textvqa": "C1", "docvqa": "C3", "chartqa": "C2"}

import gzip
_p = os.path.join(D, "run_all.json"); run = json.load(open(_p)) if os.path.exists(_p) else json.load(gzip.open(_p + ".gz", "rt")); S = run["summary"]; U_list = run["config"]["U_list"]

# ================= Fig. 2: four simulation panels, legend below =================
fig, ax = plt.subplots(1, 4, figsize=(7.16, 2.1))
for _a in ax: _a.set_box_aspect(1)
# (a) QoE vs U
for n in ORDER:
    m = [S[f"{n}|{U}"]["QoE_mean"] for U in U_list]; sd = [S[f"{n}|{U}"]["QoE_std_seeds"] for U in U_list]
    ax[0].errorbar(U_list, m, yerr=sd, capsize=1.5, elinewidth=0.6, label=LABEL[n], **STY[n])
ax[0].set_xlabel("number of users $U$\n(a)"); ax[0].set_ylabel("average per-user QoE")
# (b) Q_eff vs w_1
w = json.load(open(os.path.join(D, "aux_w1.json"))); w1s = w["w1s"]
for n in ORDER:
    ax[1].plot(w1s, [np.mean([r["Qeff"] for r in w["rows"] if r["scheme"] == n and r["w1"] == x]) for x in w1s], **STY[n])
ax[1].set_xscale("log"); ax[1].set_xlabel("latency weight $w_1$\n(b)"); ax[1].set_ylabel(r"effective quality $Q^{\mathrm{eff}}$")
# (c) eta* vs nu
e = json.load(open(os.path.join(D, "aux_eta_nu.json")))
for t in ["chartqa", "docvqa", "textvqa", "vqav2"]:
    c = e["curves"][t]; ax[2].plot(e["nus"], c["etas"], color=TCOL[t], label=TASK[t].split(" ")[0])
ax[2].set_xscale("log"); ax[2].set_xlabel("compute shadow price $\\nu_{s_u^*}$\n(c)"); ax[2].set_ylabel(r"optimal retention $\eta_u^*$")
ax[2].legend(loc="upper right", handlelength=1.4)
# (d) certified relative gap at termination vs U: distribution over all slots (seeds x slots) of the main sweep
raw = run["raw"]; gaps_by_U = {U: [g * 100 for r in raw if r["scheme"] == "Proposed" and r["U"] == U for g in r["gaps"]] for U in U_list}
bp = ax[3].boxplot([gaps_by_U[U] for U in U_list], positions=U_list, widths=3.2, whis=(5, 95), showfliers=False, patch_artist=True,
                   medianprops=dict(color="C3", lw=1.0), boxprops=dict(facecolor="#dbe9f6", edgecolor="C0", lw=0.7), whiskerprops=dict(color="C0", lw=0.7), capprops=dict(color="C0", lw=0.7))
ax[3].plot(U_list, [np.mean(gaps_by_U[U]) for U in U_list], "o", color="C0", ms=2.5, zorder=3)
ax[3].axhline(2.0, color="k", lw=0.7, ls=":"); ax[3].text(U_list[-1] + 4.2, 2.08, r"$\epsilon_g$ = 2%", fontsize=6, va="bottom", ha="right")
ax[3].set_xlim(U_list[0] - 3.5, U_list[-1] + 5.0); ax[3].set_ylim(0, 4.2); ax[3].set_xticks(U_list); ax[3].set_xticklabels([str(u) for u in U_list])
ax[3].set_xlabel("number of users $U$\n(d)"); ax[3].set_ylabel("certified gap [%]")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 0.995), handlelength=2.4, columnspacing=1.4)
plt.tight_layout(pad=0.3, w_pad=0.7, rect=(0, 0, 1, 0.95))
for ext in ("pdf", "png"): plt.savefig(os.path.join(OUTDIR, f"fig1_combined.{ext}"), dpi=300)
print("Fig. 3 of the paper (4 panels) -> figures/fig1_combined.pdf/.png")

# ================= Fig. 3: calibration curves, single column =================
# Points: held-out questions only (split by image content); curves: task profiles fitted on the
# calibration split (profile_lookup_final.json). Bootstrap resamples questions and recomputes the
# ratio mean acc(eta) / mean acc(1) jointly, so the CI collapses to zero at eta = 1.
cal_rows = [json.loads(l) for l in open(os.path.join(HERE, "calibration", "data", "answers_final.jsonl"))]
qm = {q["question_id"]: q for q in json.load(open(os.path.join(HERE, "calibration", "data", "question_ids.json")))}
byq = collections.defaultdict(dict)
for r in cal_rows: byq[r["question_id"]][(r["eta"], r["mode"])] = r["acc"]
etas = sorted({e for (e, m) in {k for d in byq.values() for k in d} if m == "topk_L12"})
rand_etas = sorted({e for (e, m) in {k for d in byq.values() for k in d} if m == "random"})
lk = json.load(open(os.path.join(HERE, "calibration", "data", "profile_lookup_final.json")))
def ratio(keys, e, mode="topk_L12"):
    a1 = np.mean([byq[q][(1.0, "topk_L12")] for q in keys]); return np.mean([byq[q][(e, mode)] for q in keys]) / a1
fig3, a3 = plt.subplots(figsize=(3.5, 2.1)); rng = np.random.default_rng(0); n_held = {}
for t in ["vqav2", "textvqa", "docvqa", "chartqa"]:
    keys = [q for q in byq if qm[q]["source"] == t and qm[q]["split"] == "heldout" and all((e, "topk_L12") in byq[q] for e in etas)]
    n_held[t] = len(keys)
    ret = np.array([ratio(keys, e) for e in etas])
    boots = np.array([[ratio([keys[i] for i in idx], e) for e in etas] for idx in (rng.integers(0, len(keys), len(keys)) for _ in range(300))])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    p = lk["profiles"][f"task:{t}"]; xx = np.linspace(0.02, 1, 200)
    a3.errorbar(etas, ret, yerr=[np.maximum(ret - lo, 0), np.maximum(hi - ret, 0)], fmt="o", ms=3, capsize=1.5, elinewidth=0.6, color=TCOL[t])
    a3.plot(xx, 1 - p["delta"] * (1 - xx) ** p["theta"], "-", color=TCOL[t], label=f"{TASK[t]}: ($\\delta$, $\\theta$)=({p['delta']:.2f}, {p['theta']:.1f})")
    rk = [q for q in keys if all((e, "random") in byq[q] for e in rand_etas)]
    if rk: a3.plot(rand_etas, [ratio(rk, e, "random") for e in rand_etas], "o", ms=3, mfc="none", mec=TCOL[t], mew=0.7, ls="none")
a3.plot([], [], "o", ms=3, mfc="none", mec="k", mew=0.7, ls="none", label="random selection")
a3.set_xlabel(r"retention ratio $\eta$"); a3.set_ylabel(r"quality retention $S^{\mathrm{sem}}(\eta)$"); a3.set_ylim(0, 1.12); a3.set_xlim(0, 1.02)
a3.legend(loc="lower right", fontsize=5.4, handlelength=1.6)
plt.tight_layout(pad=0.3)
print("Fig. 3 held-out n per task:", n_held)
for ext in ("pdf", "png"): fig3.savefig(os.path.join(OUTDIR, f"fig_calib.{ext}"), dpi=300)
print("Fig. 2 of the paper (calibration, single column) -> figures/fig_calib.pdf/.png")

# ---- Table I at U=30 ----
U0 = 30; rows = []
for n in ORDER:
    v = S[f"{n}|{U0}"]; rows.append((LABEL[n], v["QoE_mean"], v["QoE_p5"], v["below_local"], v["TTFT_mean"] * 1000, v["TPOT_mean"] * 1000))
base = S[f"Proposed|{U0}"]["QoE_mean"]
print(f"\nTable II (U={U0}):"); print(f"{'Scheme':22s} {'QoE':>6s} {'QoE_5%':>7s} {'<local':>7s} {'TTFT[ms]':>9s} {'TPOT[ms]':>9s}  gain of Proposed")
for name, q, p5, bl, tt, tp in rows: print(f"{name:22s} {q:6.3f} {p5:7.3f} {bl:7.2f} {tt:9.0f} {tp:9.1f}  {'' if name=='Proposed' else f'{(base/q-1)*100:+.1f}%'}")
gains = {U: {n: (S[f"Proposed|{U}"]["QoE_mean"] / S[f"{n}|{U}"]["QoE_mean"] - 1) * 100 for n in ORDER[1:]} for U in U_list}
print("\ngain over each baseline by U:"); [print(f"  U={U}: " + "  ".join(f"{n}: {g:+.1f}%" for n, g in gains[U].items())) for U in U_list]
pr = S[f"Proposed|{U0}"]; print(f"\nProposed @U={U0}: gap mean {pr['gap_mean']*100:.2f}% max {pr['gap_max']*100:.2f}%, iters {pr['iters_mean']:.0f}, wall {pr['wall_mean']:.1f}s, max ISL link sessions {pr['link_max']}, offload {pr['offload_frac']:.2f}")
print("eta by task @U=30:", {t: round(v, 3) for t, v in pr["eta_by_task"].items()})
