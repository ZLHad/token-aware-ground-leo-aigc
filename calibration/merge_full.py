"""Merge pilot (VQAv2+TextVQA) and dense (ChartQA+DocVQA) sets into questions_full.jsonl,
re-split by image (60/40, seed 0), and seed the full-run score/answer files from the pilot ones."""
import json, os, random, shutil, collections, numpy as np
D = "data"; rows = []
for f in ["questions.jsonl", "questions_dense.jsonl"]:
    rows += [json.loads(l) for l in open(os.path.join(D, f))]
seen = set(); items = []
for r in rows:
    k = (r["source"], r["image_id"], r["question"])
    if k in seen: continue
    seen.add(k); items.append(r)
imgs = sorted({(r["source"], r["image_id"]) for r in items}); random.Random(0).shuffle(imgs)
calib = set(imgs[: int(round(0.6 * len(imgs)))])
with open(os.path.join(D, "questions_full.jsonl"), "w") as f:
    for r in items:
        r["split"] = "calib" if (r["source"], r["image_id"]) in calib else "heldout"; f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"questions_full: {len(items)} questions / {len(imgs)} images;", dict(collections.Counter(r['source'] for r in items)),
      "| calib/heldout:", dict(collections.Counter(r['split'] for r in items)))
# seed full-run files from pilot (score records + answered arms are reused; runner resumes on (qid, eta, mode))
if not os.path.exists(os.path.join(D, "scores_full.npz")): shutil.copy(os.path.join(D, "scores_pilot.npz"), os.path.join(D, "scores_full.npz"))
if not os.path.exists(os.path.join(D, "answers_full.jsonl")): shutil.copy(os.path.join(D, "answers_pilot.jsonl"), os.path.join(D, "answers_full.jsonl"))
print("seeded scores_full.npz / answers_full.jsonl from pilot")
