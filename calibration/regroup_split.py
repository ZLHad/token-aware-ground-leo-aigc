"""Make the calibration/held-out split consistent by image *content*.

ChartQA rows carry a synthetic per-row image_id, so the same chart can appear
under two ids; `data.py` split by id and let three such pairs straddle the two
sides. Rule (fixed before looking at any result): every question whose image
hash matches takes the split of the member with the smallest image index.
Rewrites data/questions_full.jsonl in place (backup: .pre_hash).
"""
import json, hashlib, os, re, collections, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, "data", "questions_full.jsonl")
Q = [json.loads(l) for l in open(P)]
h = {}
for q in Q:
    k = (q["source"], q["image_id"])
    if k not in h: h[k] = hashlib.md5(open(os.path.join(HERE, q["image_path"]), "rb").read()).hexdigest()
groups = collections.defaultdict(list)
for k, v in h.items(): groups[v].append(k)
def idx(k): m = re.search(r"(\d+)$", k[1]); return int(m.group(1)) if m else 0
target = {}
for v, ks in groups.items():
    if len(ks) < 2: continue
    lead = min(ks, key=idx); split = next(q["split"] for q in Q if (q["source"], q["image_id"]) == lead)
    for k in ks: target[k] = split
moved = []
for q in Q:
    k = (q["source"], q["image_id"])
    if k in target and q["split"] != target[k]: moved.append((q["question_id"], k, q["split"], target[k])); q["split"] = target[k]
if moved:
    shutil.copy(P, P + ".pre_hash")
    with open(P, "w") as f:
        for q in Q: f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"{len(Q)} questions, {len(h)} image ids, {len(groups)} distinct images, {sum(len(k) > 1 for k in groups.values())} duplicate groups")
for m in moved: print("moved", m)
print("split counts:", collections.Counter(q["split"] for q in Q))
