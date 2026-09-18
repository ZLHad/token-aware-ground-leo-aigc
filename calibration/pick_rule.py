"""Pick the token-selection rule from the seltest (calib subset). Writes data/selected_rule.json."""
import json, os, collections, numpy as np
DATA = "data"; MODES = ["topk", "random", "stratified", "hybrid50", "adaptive", "topk_L12"]; SIMPLICITY = {"topk": 0, "random": 9, "stratified": 1, "adaptive": 2, "hybrid50": 3, "topk_L12": 4}
full = [json.loads(l) for l in open(os.path.join(DATA, "answers_full.jsonl"))]
sel = [json.loads(l) for l in open(os.path.join(DATA, "answers_seltest.jsonl"))]
qids = {r["question_id"] for r in sel}; byq = collections.defaultdict(dict)
for r in full:
    if r["question_id"] in qids: byq[r["question_id"]][(r["eta"], "topk" if r["mode"] == "ranked" else "random")] = r["acc"]
for r in sel: byq[r["question_id"]][(r["eta"], r["mode"])] = r["acc"]
src = collections.defaultdict(set)
for r in sel: src[r["source"]].add(r["question_id"])
ret = {}   # (mode, task, eta) -> retention
for s, keys in src.items():
    keys = sorted(keys); a1 = np.mean([byq[q][(1.0, "topk")] for q in keys])
    for m in MODES:
        for eta in (0.1, 0.2):
            v = [byq[q][(eta, m)] for q in keys if (eta, m) in byq[q]]
            ret[(m, s, eta)] = (np.mean(v) / a1) if (v and a1 > 0) else np.nan
ok = {}
for m in MODES:
    if m == "random": continue
    worse = [ret[(m, s, e)] < ret[("random", s, e)] - 0.03 for s in src for e in (0.1, 0.2)]
    score = np.nanmean([ret[(m, s, e)] for s in src for e in (0.1, 0.2)])
    ok[m] = (not any(worse), score)
    print(f"{m:11s} mean={score:.3f}  never-worse-than-random={not any(worse)}  " + " ".join(f"{s}@{e}:{ret[(m,s,e)]:.2f}" for s in sorted(src) for e in (0.1, 0.2)))
cands = [m for m, (good, _) in ok.items() if good] or list(ok)
best = sorted(cands, key=lambda m: (-round(ok[m][1], 2), SIMPLICITY[m]))[0]
json.dump({"rule": best, "scores": {m: ok[m][1] for m in ok}, "table": {f"{m}|{s}|{e}": ret[(m, s, e)] for (m, s, e) in ret}}, open(os.path.join(DATA, "selected_rule.json"), "w"), indent=1)
print("SELECTED RULE:", best)
