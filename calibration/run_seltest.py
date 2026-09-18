"""Selection-rule test on a calibration-split subset: 40 ChartQA + 20 TextVQA + 20 VQAv2, eta in {0.1, 0.2}.
Modes: stratified / hybrid50 / adaptive(H) / topk_L12. (topk & random come from answers_full.jsonl.)"""
import json, os, sys, time, numpy as np
from PIL import Image
from selection import select
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
QUOTA = {"chartqa": 40, "textvqa": 20, "vqav2": 20}; ETAS = [0.1, 0.2]; MODES = ["topk_L6"]

def subset():
    qs = [json.loads(l) for l in open(os.path.join(DATA, "questions_full.jsonl"))]
    got = {k: 0 for k in QUOTA}; out = []
    for q in qs:
        s = q["source"]
        if q["split"] == "calib" and s in QUOTA and got[s] < QUOTA[s]: out.append(q); got[s] += 1
    return out

def stage_score(qs):
    from scorer import Scorer
    out = os.path.join(DATA, "scores_seltest_v2.npz")
    if os.path.exists(out): return dict(np.load(out, allow_pickle=True)["records"].item())
    s = Scorer(); rec = {}
    for i, q in enumerate(qs):
        rec[q["question_id"]] = s.score(Image.open(os.path.join(HERE, q["image_path"])).convert("RGB"), q["question"], time_it=False, layers=(2, 6, 12))
        if (i + 1) % 20 == 0: print(f"[score] {i+1}/{len(qs)}", flush=True)
    np.savez(out, records=np.array(rec, dtype=object)); return rec

def stage_answer(qs, sc):
    from pruned_vlm import PrunedVLM
    from vqa_metric import score as metric_score
    out = os.path.join(DATA, "answers_seltest.jsonl")
    done = {(r["question_id"], r["eta"], r["mode"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    v = PrunedVLM(); f = open(out, "a"); t0 = time.time(); n = 0
    for i, q in enumerate(qs):
        qid = q["question_id"]; r = sc[qid]; img = Image.open(os.path.join(HERE, q["image_path"])).convert("RGB")
        for eta in ETAS:
            for mode in MODES:
                if (qid, eta, mode) in done: continue
                if mode.startswith("topk_L"): keep = select(r["p_var"][f"L{int(mode[6:])}_q"], r["grid_thw"], eta, "topk")
                else: keep = select(r["p"], r["grid_thw"], eta, mode, H=r["H"])
                try: pred, meta = v.answer(img, q["question"], eta=eta, keep=keep)
                except Exception as e: print(f"ERROR {qid} {eta} {mode}: {e}", flush=True); continue
                acc = metric_score(q["source"], pred, q["answers"]); n += 1
                f.write(json.dumps(dict(question_id=qid, source=q["source"], eta=eta, mode=mode, pred=pred, acc=acc, H=r["H"], n_vis=meta["n_vis"], n_kept=len(keep)), ensure_ascii=False) + "\n"); f.flush()
                print(f"[{i+1}/{len(qs)}] eta={eta} {mode:<10} acc={acc:.2f} pred={pred!r:<20} ({(time.time()-t0)/n:.1f}s/gen)", flush=True)
    f.close()

if __name__ == "__main__":
    qs = subset(); print("subset:", {s: sum(q["source"] == s for q in qs) for s in QUOTA}, flush=True)
    sc = stage_score(qs); stage_answer(qs, sc); print("SELTEST FINISHED", flush=True)
