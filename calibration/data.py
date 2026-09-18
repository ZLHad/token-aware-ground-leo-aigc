"""Build the calibration question set: VQAv2 (stratified by answer_type) + TextVQA.

Output: data/questions.jsonl (+ data/images/*.jpg). Split by image into calib / heldout.
Usage: python data.py --n_vqa 100 --n_text 100 --seed 0
"""
import argparse, json, os, random, collections
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); IMG = os.path.join(DATA, "images")

def take_vqav2(n, seed, buffer=3000):
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True).shuffle(seed=seed, buffer_size=buffer)
    # stratify by answer_type in original proportion (yes/no 38%, number 13%, other 49%)
    quota = {"yes/no": round(0.38*n), "number": round(0.13*n)}; quota["other"] = n - sum(quota.values())
    got = collections.Counter(); out = []
    for ex in ds:
        t = ex["answer_type"]
        if got[t] >= quota.get(t, 0): 
            if sum(got.values()) >= n: break
            continue
        out.append(dict(source="vqav2", image_id=str(ex["image_id"]), question_id=str(ex["question_id"]),
                        question=ex["question"], answers=[a["answer"] for a in ex["answers"]],
                        answer_type=t, question_type=ex["question_type"], image=ex["image"]))
        got[t] += 1
        if sum(got.values()) >= n: break
    return out

def take_textvqa(n, seed, buffer=1500):
    ds = load_dataset("lmms-lab/textvqa", split="validation", streaming=True).shuffle(seed=seed, buffer_size=buffer)
    out = []
    for ex in ds:
        out.append(dict(source="textvqa", image_id=str(ex["image_id"]), question_id=str(ex["question_id"]),
                        question=ex["question"], answers=list(ex["answers"]), answer_type="ocr",
                        question_type="textvqa", image=ex["image"]))
        if len(out) >= n: break
    return out

def take_chartqa(n, seed, buffer=1500):
    ds = load_dataset("lmms-lab/ChartQA", split="test", streaming=True).shuffle(seed=seed, buffer_size=buffer)
    out = []
    for i, ex in enumerate(ds):
        if ex.get("type", "human_test") != "human_test": continue   # natural human-written questions only
        out.append(dict(source="chartqa", image_id=f"c{seed}_{i}", question_id=f"chart_{seed}_{i}",
                        question=ex["question"], answers=[str(ex["answer"])], answer_type="chart",
                        question_type="chartqa", image=ex["image"]))
        if len(out) >= n: break
    return out

def take_docvqa(n, seed, buffer=800):
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True).shuffle(seed=seed, buffer_size=buffer)
    out = []
    for ex in ds:
        out.append(dict(source="docvqa", image_id=f"d{ex['docId']}_{ex['ucsf_document_page_no']}", question_id=f"doc_{ex['questionId']}",
                        question=ex["question"], answers=list(ex["answers"]), answer_type="doc",
                        question_type="|".join(ex.get("question_types") or []), image=ex["image"]))
        if len(out) >= n: break
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_vqa", type=int, default=100)
    ap.add_argument("--n_text", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_chart", type=int, default=0); ap.add_argument("--n_doc", type=int, default=0)
    ap.add_argument("--calib_frac", type=float, default=0.6); ap.add_argument("--out", default="questions.jsonl")
    a = ap.parse_args(); os.makedirs(IMG, exist_ok=True)
    items = (take_vqav2(a.n_vqa, a.seed) if a.n_vqa else []) + (take_textvqa(a.n_text, a.seed) if a.n_text else []) \
          + (take_chartqa(a.n_chart, a.seed) if a.n_chart else []) + (take_docvqa(a.n_doc, a.seed) if a.n_doc else [])
    # split by image (all questions of one image stay on the same side)
    rng = random.Random(a.seed); images = sorted({(it["source"], it["image_id"]) for it in items}); rng.shuffle(images)
    calib = set(images[: int(round(a.calib_frac * len(images)))])
    with open(os.path.join(DATA, a.out), "w") as f:
        for it in items:
            path = os.path.join(IMG, f"{it['source']}_{it['image_id']}.jpg")
            if not os.path.exists(path): it["image"].convert("RGB").save(path, quality=92)
            rec = {k: v for k, v in it.items() if k != "image"}
            rec["image_path"] = os.path.relpath(path, HERE); rec["split"] = "calib" if (it["source"], it["image_id"]) in calib else "heldout"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    c = collections.Counter((it["source"], it["answer_type"]) for it in items)
    print(f"wrote {len(items)} questions over {len(images)} images -> data/{a.out}"); print(dict(c))
if __name__ == "__main__": main()
