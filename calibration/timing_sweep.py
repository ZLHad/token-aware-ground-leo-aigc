"""E3b: latency-coefficient structure on the calibration hardware (Qwen3-VL-32B 4-bit, M3 Max).
Measures prefill time vs. input tokens and per-token decode time. Writes data/timing_sweep.json."""
import json, os, numpy as np
from PIL import Image
from pruned_vlm import PrunedVLM
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
qs = [json.loads(l) for l in open(os.path.join(DATA, "questions_full.jsonl"))]
sc = dict(np.load(os.path.join(DATA, "scores_full.npz"), allow_pickle=True)["records"].item())
big = [q for q in qs if sc.get(q["question_id"], {}).get("n_vis", 0) >= 900][:12]     # large images -> wide token range
v = PrunedVLM(); recs = []
v.answer_timed(Image.open(os.path.join(HERE, big[0]["image_path"])).convert("RGB"), big[0]["question"], eta=0.3, max_new_tokens=5)  # warm-up
for q in big:
    img = Image.open(os.path.join(HERE, q["image_path"])).convert("RGB")
    for eta in (0.05, 0.15, 0.3, 0.6, 1.0):
        tp, steps, n_seq, n_vis = v.answer_timed(img, q["question"], eta=eta, max_new_tokens=20)
        recs.append(dict(qid=q["question_id"], eta=eta, n_seq=n_seq, n_vis=n_vis, t_prefill=tp, decode_ms=[1000 * s for s in steps]))
        print(f"{q['question_id']:>16s} eta={eta:<4} n_seq={n_seq:4d} prefill={tp:.3f}s decode={np.mean(steps)*1000:.1f} ms/tok", flush=True)
x = np.array([r["n_seq"] for r in recs]); y = np.array([r["t_prefill"] for r in recs]); b, a = np.polyfit(x, y, 1)
dec = np.array([np.mean(r["decode_ms"][2:]) for r in recs]); dslope, dint = np.polyfit(x, dec, 1)
summary = dict(prefill_fit=dict(intercept_s=float(a), per_token_s=float(b), r2=float(np.corrcoef(x, y)[0, 1] ** 2)),
               decode_ms_per_token=dict(mean=float(dec.mean()), std=float(dec.std()), slope_ms_per_ctx_token=float(dslope)), n=len(recs))
json.dump(dict(records=recs, summary=summary), open(os.path.join(DATA, "timing_sweep.json"), "w"), indent=1)
print(json.dumps(summary, indent=1)); print("TIMING DONE")
