"""Final calibration analysis for a run tag.
  python final_analysis.py <tag> [rule_mode]
Outputs: calibration/results/CALIBRATION_RESULTS_<tag>.md, ../figures/calib_curves_<tag>.{png,pdf}, data/profile_lookup_<tag>.json
"""
import json, os, sys, collections, numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
tag = sys.argv[1] if len(sys.argv) > 1 else "full"; RULE = sys.argv[2] if len(sys.argv) > 2 else "ranked"; HKEY = sys.argv[3] if len(sys.argv) > 3 else "H"
def Hof(rec): return rec["H"] if HKEY == "H" else rec["H_var"][HKEY]
os.makedirs(os.path.join(HERE, "results"), exist_ok=True); OUT_MD = os.path.join(HERE, "results", f"CALIBRATION_RESULTS_{tag}.md"); FIG = os.path.join(HERE, "..", "figures", f"calib_curves_{tag}")
TASK = {"vqav2": "VQAv2 (global)", "textvqa": "TextVQA (localized)", "chartqa": "ChartQA (distributed)", "docvqa": "DocVQA (distributed)"}
rows = [json.loads(l) for l in open(os.path.join(DATA, f"answers_{tag}.jsonl"))]
sc = dict(np.load(os.path.join(DATA, f"scores_{tag}.npz"), allow_pickle=True)["records"].item())
qmeta = {json.loads(l)["question_id"]: json.loads(l) for l in open(os.path.join(DATA, "questions_full.jsonl"))}
byq = collections.defaultdict(dict)
for r in rows: byq[r["question_id"]][(r["eta"], r["mode"])] = r["acc"]
etas = sorted({e for (e, m) in {k for d in byq.values() for k in d} if m == RULE}); rand_etas = sorted({e for (e, m) in {k for d in byq.values() for k in d} if m == "random"})
Q = {q: d for q, d in byq.items() if all((e, RULE) in d for e in etas) and q in sc and q in qmeta}
def acc(q, e, m=RULE): return Q[q][(e, m)]
def S(eta, d, t): return 1 - d * (1 - eta) ** t
def curve(keys, e_list=etas, m=RULE):
    a1 = np.mean([acc(q, 1.0) for q in keys]); return np.array([np.mean([acc(q, e, m) for q in keys]) / a1 if a1 > 0 else np.nan for e in e_list]), a1
def boot_ci(keys, e_list=etas, B=300, seed=0):
    rng = np.random.default_rng(seed); keys = list(keys); out = []
    for _ in range(B):
        k = [keys[i] for i in rng.integers(0, len(keys), len(keys))]; out.append(curve(k, e_list)[0])
    return np.nanpercentile(np.array(out), [2.5, 97.5], axis=0)
def fit(x, y, w=None):
    try: (d, t), _ = curve_fit(S, np.array(x), np.array(y), p0=(0.5, 2.0), bounds=([0.0, 1.0], [1.0, 12.0]), sigma=(None if w is None else 1 / np.sqrt(np.array(w)))); return float(d), float(t)
    except Exception: return float("nan"), float("nan")
def rmse(y, yhat): return float(np.sqrt(np.nanmean((np.array(y) - np.array(yhat)) ** 2)))
# ---- grouping keys: task; task x H tercile (cuts on the calib split, global) ----
calib = [q for q in Q if qmeta[q]["split"] == "calib"]; held = [q for q in Q if qmeta[q]["split"] == "heldout"]
Hc = np.array([Hof(sc[q]) for q in calib]); cuts = np.quantile(Hc, [1 / 3, 2 / 3])
def terc(q): h = Hof(sc[q]); return 0 if h < cuts[0] else (1 if h < cuts[1] else 2)
med = {s: float(np.median([Hof(sc[q]) for q in calib if qmeta[q]["source"] == s])) for s in {qmeta[q]["source"] for q in calib}}
def half(q): return "Hlo" if Hof(sc[q]) < med[qmeta[q]["source"]] else "Hhi"
keyfun = {"global": lambda q: "all", "task": lambda q: qmeta[q]["source"], "task_x_H2": lambda q: f"{qmeta[q]['source']}|{half(q)}", "task_x_H": lambda q: f"{qmeta[q]['source']}|T{terc(q)+1}"}
L = []; L.append(f"# 校准结果（tag={tag}，规则={RULE}，H={HKEY}）\n"); L.append(f"题数（完整 {len(etas)} 档）：{len(Q)}；校准 {len(calib)} / 留出 {len(held)}；η 网格 {etas}；随机对照 η {rand_etas}；H 三分位切点（校准集）{cuts.round(3).tolist()}\n")
# ---- per-task curves with CI, ranked vs random ----
L.append("## 1. 各任务 retention（全部题，均值 [95% bootstrap CI]）\n"); L.append("| 任务 | n | acc@1 | " + " | ".join(f"η={e}" for e in etas if e < 1) + " |"); L.append("|---|---|---|" + "---|" * (len(etas) - 1))
task_curves = {}
for s in ["vqav2", "textvqa", "chartqa", "docvqa"]:
    keys = [q for q in Q if qmeta[q]["source"] == s]
    if not keys: continue
    c, a1 = curve(keys); ci = boot_ci(keys); task_curves[s] = (keys, c, ci, a1)
    L.append(f"| {TASK[s]} | {len(keys)} | {a1:.3f} | " + " | ".join(f"{c[i]:.2f} [{ci[0][i]:.2f},{ci[1][i]:.2f}]" for i, e in enumerate(etas) if e < 1) + " |")
if rand_etas:
    L.append("\n### 按重要性 vs 随机（retention）\n"); L.append("| 任务 | " + " | ".join(f"{RULE}@{e} / random@{e}" for e in rand_etas) + " |"); L.append("|---|" + "---|" * len(rand_etas))
    for s, (keys, c, ci, a1) in task_curves.items():
        cells = []
        for e in rand_etas:
            kr = [q for q in keys if (e, "random") in Q[q]]
            if kr: cr, _ = curve(kr, [e], RULE); rr, _ = curve(kr, [e], "random"); cells.append(f"{cr[0]:.2f} / {rr[0]:.2f}")
            else: cells.append("–")
        L.append(f"| {TASK[s]} | " + " | ".join(cells) + " |")
# ---- fits at three lookup levels: fit on calib, evaluate on heldout ----
L.append("\n## 2. 三级查表的拟合（校准集拟合，留出集评估）\n"); lookup = {"cuts": cuts.tolist(), "task_H_median": med, "etas": etas, "profiles": {}}; summary = {}
for level, kf in keyfun.items():
    groups = collections.defaultdict(lambda: {"calib": [], "heldout": []})
    for q in Q: groups[kf(q)][qmeta[q]["split"]].append(q)
    L.append(f"### {level}\n"); L.append("| 组 | n_calib | n_held | δ | θ | RMSE_calib | RMSE_held |"); L.append("|---|---|---|---|---|---|---|")
    rc, rh, wc, wh = [], [], [], []
    for g, sp in sorted(groups.items()):
        if len(sp["calib"]) < 5: continue
        yc, _ = curve(sp["calib"]); d, t = fit(etas, yc); yhat = S(np.array(etas), d, t); e_c = rmse(yc, yhat)
        e_h = float("nan")
        if len(sp["heldout"]) >= 5: yh, _ = curve(sp["heldout"]); e_h = rmse(yh, yhat); rh.append(e_h); wh.append(len(sp["heldout"]))
        rc.append(e_c); wc.append(len(sp["calib"])); lookup["profiles"][f"{level}:{g}"] = {"delta": d, "theta": t, "n_calib": len(sp["calib"]), "rmse_calib": e_c, "rmse_held": e_h}
        L.append(f"| {g} | {len(sp['calib'])} | {len(sp['heldout'])} | {d:.2f} | {t:.2f} | {e_c:.3f} | {e_h:.3f} |")
    # weighted RMSE of held-out group curves against this level's profiles; for 'global' evaluate against per-task held-out curves too
    summary[level] = (np.average(rc, weights=wc) if rc else np.nan, np.average(rh, weights=wh) if rh else np.nan)
# global profile evaluated on per-task held-out curves (what a request-oblivious scheduler would assume)
gd, gt = lookup["profiles"]["global:all"]["delta"], lookup["profiles"]["global:all"]["theta"]
per_task_held = []
for s, (keys, c, ci, a1) in task_curves.items():
    kh = [q for q in keys if qmeta[q]["split"] == "heldout"]
    if len(kh) >= 5: yh, _ = curve(kh); per_task_held.append((len(kh), rmse(yh, S(np.array(etas), gd, gt))))
g_on_task = np.average([e for _, e in per_task_held], weights=[n for n, _ in per_task_held]) if per_task_held else float("nan")
L.append("\n### 汇总（留出集 RMSE，按组大小加权）\n"); L.append("| 查表级别 | RMSE_calib | RMSE_held（同级分组） | 全局曲线对各任务留出曲线的 RMSE |"); L.append("|---|---|---|---|")
for level in keyfun: L.append(f"| {level} | {summary[level][0]:.3f} | {summary[level][1]:.3f} | {'%.3f' % g_on_task if level == 'global' else '–'} |")
# ---- per-question Spearman with H at low eta ----
L.append("\n## 3. 逐题退化 vs H（Spearman，acc@1>0 的题）\n"); L.append("| η | 全部 | " + " | ".join(TASK[s] for s in task_curves) + " |"); L.append("|---|---|" + "---|" * len(task_curves))
for e in [x for x in etas if x <= 0.2]:
    cells = []
    for keys in [list(Q)] + [k for k, *_ in task_curves.values()]:
        v = [q for q in keys if acc(q, 1.0) > 0]
        if len(v) >= 10: rho, pv = spearmanr([Hof(sc[q]) for q in v], [acc(q, 1.0) - acc(q, e) for q in v]); cells.append(f"{rho:+.2f} (p={pv:.2f}, n={len(v)})")
        else: cells.append("–")
    L.append(f"| {e} | " + " | ".join(cells) + " |")
json.dump(lookup, open(os.path.join(DATA, f"profile_lookup_{tag}.json"), "w"), indent=1)
# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4)); ax = axes[0]; x = np.array(etas)
for s, (keys, c, ci, a1) in task_curves.items():
    kc = [q for q in keys if qmeta[q]["split"] == "calib"]; d, t = fit(etas, curve(kc)[0]) if len(kc) >= 5 else (np.nan, np.nan)
    ln = ax.errorbar(x, c, yerr=[c - ci[0], ci[1] - c], fmt="o", ms=3.5, capsize=2, label=f"{TASK[s]} (δ={d:.2f}, θ={t:.1f})")
    if np.isfinite(d): xx = np.linspace(0.02, 1, 200); ax.plot(xx, S(xx, d, t), "-", color=ln[0].get_color(), lw=1.2)
ax.set_xlabel("retention ratio η"); ax.set_ylabel("quality retention S(η)"); ax.set_ylim(0, 1.08); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right"); ax.set_title("(a) by task", fontsize=9)
ax = axes[1]
for s, (keys, c, ci, a1) in task_curves.items():
    for ti in range(3):
        kk = [q for q in keys if terc(q) == ti]
        if len(kk) >= 8: cc, _ = curve(kk); ax.plot(x, cc, ["--", "-.", ":"][ti], marker="o", ms=2.5, lw=1, label=f"{s} T{ti+1}")
ax.set_xlabel("retention ratio η"); ax.set_ylim(0, 1.08); ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2); ax.set_title("(b) by task × H tercile (T1 = low H)", fontsize=9)
plt.tight_layout(); os.makedirs(os.path.dirname(FIG), exist_ok=True); plt.savefig(FIG + ".png", dpi=200); plt.savefig(FIG + ".pdf")
L.append(f"\n图：`code/figures/calib_curves_{tag}.png`；查表：`code/calibration/data/profile_lookup_{tag}.json`\n")
os.makedirs(os.path.dirname(OUT_MD), exist_ok=True); open(OUT_MD, "w").write("\n".join(L)); print("\n".join(L)); print("->", OUT_MD)
