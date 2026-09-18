import json, os, collections, numpy as np
DATA = "data"
full = [json.loads(l) for l in open(os.path.join(DATA, "answers_full.jsonl"))]
sel = [json.loads(l) for l in open(os.path.join(DATA, "answers_seltest.jsonl"))]
qids = {r["question_id"] for r in sel}
byq = collections.defaultdict(dict)
for r in full:
    if r["question_id"] in qids: byq[r["question_id"]][(r["eta"], "topk" if r["mode"] == "ranked" else "random")] = r["acc"]
for r in sel: byq[r["question_id"]][(r["eta"], r["mode"])] = r["acc"]
src = collections.defaultdict(list)
for r in sel: src[r["source"]].append(r["question_id"])
modes = ["topk", "random", "stratified", "adaptive", "topk_L6", "topk_L12"]
for eta in (0.1, 0.2):
    print(f"\n=== eta={eta}: retention vs acc@1 (calib subset) ===\n{'task':9s} {'n':>3s} " + " ".join(f"{m:>10s}" for m in modes))
    allrows = []
    for s, keys in sorted(src.items()):
        keys = sorted(set(keys)); a1 = np.mean([byq[q][(1.0, "topk")] for q in keys])
        row = []
        for m in modes:
            v = [byq[q][(eta, m)] for q in keys if (eta, m) in byq[q]]
            row.append(np.mean(v) / a1 if v else np.nan)
        allrows.append(row); print(f"{s:9s} {len(keys):3d} " + " ".join(f"{x:10.2f}" for x in row))
    print(f"{'mean':9s}     " + " ".join(f"{x:10.2f}" for x in np.nanmean(np.array(allrows), axis=0)))
