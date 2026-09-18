"""Pilot analysis: retention curves (task / indicator terciles), ranked-vs-random control,
per-question degradation vs. user-side indicators (Spearman), profile fit S=1-delta(1-eta)^theta."""
import json, sys, os, numpy as np, collections
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
tag = sys.argv[1] if len(sys.argv) > 1 else "pilot"
rows = [json.loads(l) for l in open(os.path.join(DATA, f"answers_{tag}.jsonl"))]
sc = dict(np.load(os.path.join(DATA, f"scores_{tag}.npz"), allow_pickle=True)["records"].item())
byq = collections.defaultdict(dict)
for r in rows: byq[r["question_id"]][(r["eta"], r.get("mode", "ranked"))] = r
arms = sorted({(r["eta"], r.get("mode", "ranked")) for r in rows}); etas = sorted({e for e, m in arms if m == "ranked"})
qs = {q: d for q, d in byq.items() if all(a in d for a in arms)}
print(f"{len(qs)} questions complete over arms {arms}")
def acc(q, eta, mode="ranked"): return qs[q][(eta, mode)]["acc"]
def S(eta, delta, theta): return 1 - delta * (1 - eta) ** theta
def report(name, groups):
    print(f"\n=== retention by {name} ===")
    for g, keys in sorted(groups.items()):
        a1 = np.mean([acc(q, 1.0) for q in keys]); ret = {e: np.mean([acc(q, e) for q in keys]) / a1 if a1 > 0 else np.nan for e in etas}
        rnd = {e: np.mean([acc(q, e, "random") for q in keys]) / a1 for e, m in arms if m == "random"} if a1 > 0 else {}
        try:
            (d, t), _ = curve_fit(S, np.array(etas), np.array([ret[e] for e in etas]), p0=(0.5, 1.5), bounds=([0, 0.2], [1, 6])); fit = f"delta={d:.2f} theta={t:.2f}"
        except Exception as ex: fit = "fit n/a"
        print(f"  {g:<22} n={len(keys):<4} acc@1={a1:.3f} | " + " ".join(f"r({e})={ret[e]:.2f}" for e in etas) + " | " + " ".join(f"rand({e})={v:.2f}" for e, v in rnd.items()) + f" | {fit}")
groups = collections.defaultdict(list)
for q in qs: groups[f"{qs[q][(1.0,'ranked')]['source']}/{qs[q][(1.0,'ranked')]['answer_type']}"].append(q)
report("task", groups)
groups = collections.defaultdict(list)
for q in qs: groups[qs[q][(1.0,'ranked')]['source']].append(q)
report("source", groups)
# indicator terciles
indicators = {"H(L2,q)": lambda q: sc[q]["H"], "top10": lambda q: sc[q]["top10"], "n_vis": lambda q: sc[q]["n_vis"]}
for k in sorted(sc[next(iter(sc))]["H_var"]): indicators[k] = (lambda kk: (lambda q: sc[q]["H_var"][kk]))(k)
for name, f in indicators.items():
    v = np.array([f(q) for q in qs]); cuts = np.quantile(v, [1/3, 2/3]); g = collections.defaultdict(list)
    for q in qs: g["T1-low" if f(q) < cuts[0] else ("T2-mid" if f(q) < cuts[1] else "T3-high")].append(q)
    report(f"{name} tercile (cuts {cuts.round(3)})", g)
# per-question degradation vs indicators
print("\n=== Spearman: per-question drop acc(1)-acc(eta) vs indicator (questions with acc@1>0) ===")
valid = [q for q in qs if acc(q, 1.0) > 0]
for e in [x for x in etas if x < 1]:
    drop = np.array([acc(q, 1.0) - acc(q, e) for q in valid])
    line = f"  eta={e}: n={len(valid)} mean drop={drop.mean():.3f} | "
    for name, f in indicators.items():
        rho, pv = spearmanr([f(q) for q in valid], drop); line += f"{name}: rho={rho:+.2f}(p={pv:.2f})  "
    print(line)
    if (e, "random") in arms:
        dr = np.array([acc(q, 1.0) - acc(q, e, "random") for q in valid]); print(f"           ranked mean drop {drop.mean():.3f} vs random {dr.mean():.3f}")
