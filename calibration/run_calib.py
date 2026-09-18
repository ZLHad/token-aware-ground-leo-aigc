"""Two-stage calibration runner (resume-safe).

  python run_calib.py score  --tag pilot                 # 2B scoring pass -> data/scores_<tag>.npz
  python run_calib.py answer --tag pilot --etas 0.2,0.5,1  # 32B answers per eta -> data/answers_<tag>.jsonl
"""
import argparse, json, os, sys, time, numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")

def load_questions(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def stage_score(args):
    from scorer import Scorer
    qs = load_questions(os.path.join(DATA, args.questions)); out = os.path.join(DATA, f"scores_{args.tag}.npz")
    done = dict(np.load(out, allow_pickle=True)["records"].item()) if os.path.exists(out) else {}
    s = Scorer(layer_k=args.layer); print("analytic FLOPs ratio r =", round(s.flops_ratio(), 3), flush=True)
    t0 = time.time()
    for i, q in enumerate(qs):
        qid = q["question_id"]
        if qid in done: continue
        r = s.score(Image.open(os.path.join(HERE, q["image_path"])).convert("RGB"), q["question"], time_it=(i % 5 == 0))
        done[qid] = r
        if (i + 1) % 10 == 0 or i == len(qs) - 1:
            np.savez(out, records=np.array(done, dtype=object))
            el = time.time() - t0; print(f"[score] {i+1}/{len(qs)}  H={r['H']:.3f} n_vis={r['n_vis']} t_full={r['t_full']:.2f}s  ({el/ (i+1):.2f}s/q)", flush=True)
    np.savez(out, records=np.array(done, dtype=object)); print("scores ->", out)

def stage_answer(args):
    from pruned_vlm import PrunedVLM
    from vqa_metric import score as metric_score
    from selection import select
    qs = load_questions(os.path.join(DATA, args.questions))
    scores = dict(np.load(os.path.join(DATA, f"scores_{args.tag}.npz"), allow_pickle=True)["records"].item())
    rule = args.mode
    etas = [(float(x), rule) for x in args.etas.split(",")] + [(float(x), "random") for x in args.random_etas.split(",") if x]
    out = os.path.join(DATA, f"answers_{args.tag}.jsonl")
    done = {(r["question_id"], r["eta"], r.get("mode", "ranked")) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    rng = np.random.default_rng(args.seed)
    v = PrunedVLM(); f = open(out, "a"); t0 = time.time(); n = 0
    for i, q in enumerate(qs):
        qid = q["question_id"]
        if qid not in scores: print("no score for", qid); continue
        img = Image.open(os.path.join(HERE, q["image_path"])).convert("RGB"); p = scores[qid]["p"]
        for eta, mode in etas:
            if (qid, eta, mode) in done: continue
            try:
                if mode == "random": keep = select(p, scores[qid]["grid_thw"], eta, "random", rng=rng)
                elif mode in ("ranked", "topk"): keep = select(p, scores[qid]["grid_thw"], eta, "topk")
                elif mode.startswith("topk_L"): keep = select(scores[qid]["p_var"][f"L{int(mode[6:])}_q"], scores[qid]["grid_thw"], eta, "topk")
                else: keep = select(p, scores[qid]["grid_thw"], eta, mode, H=scores[qid]["H"])
                pred, meta = v.answer(img, q["question"], eta=eta, keep=(None if eta >= 1.0 else keep), max_new_tokens=args.max_new)
            except Exception as e:
                print(f"ERROR {qid} eta={eta}: {type(e).__name__}: {e}", flush=True); continue
            if meta["n_vis"] != scores[qid]["n_vis"]:
                print(f"GRID MISMATCH {qid}: 32B n_vis={meta['n_vis']} vs 2B {scores[qid]['n_vis']}", flush=True)
            acc = metric_score(q["source"], pred, q["answers"])
            rec = dict(question_id=qid, source=q["source"], answer_type=q["answer_type"], split=q["split"], eta=eta, mode=mode,
                       pred=pred, acc=acc, H=scores[qid]["H"], n_vis=meta["n_vis"], n_kept=meta["n_kept"], n_txt=scores[qid]["n_txt"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); n += 1
            print(f"[{i+1}/{len(qs)}] eta={eta:<4} {mode:<6} acc={acc:.2f} pred={pred!r:<22} H={rec['H']:.2f} kept {meta['n_kept']}/{meta['n_vis']}  ({(time.time()-t0)/max(n,1):.1f}s/gen)", flush=True)
    f.close(); print("answers ->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["score", "answer"])
    ap.add_argument("--tag", default="pilot"); ap.add_argument("--questions", default="questions.jsonl")
    ap.add_argument("--etas", default="0.2,0.5,1"); ap.add_argument("--random_etas", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mode", default="ranked"); ap.add_argument("--layer", type=int, default=2); ap.add_argument("--max_new", type=int, default=10)
    a = ap.parse_args(); (stage_score if a.stage == "score" else stage_answer)(a)
