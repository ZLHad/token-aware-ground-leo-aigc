#!/bin/zsh
# usage: ./launch_final.sh <rule>   (rule in: stratified | hybrid50 | adaptive | topk_L12)
RULE="$1"; cd "$(dirname "$0")" && source ../.venv/bin/activate
python - "$RULE" <<'PY'
import json, sys, shutil, os
rule = sys.argv[1]; D = "data"
shutil.copy(os.path.join(D, "scores_full.npz"), os.path.join(D, "scores_final.npz"))
with open(os.path.join(D, "answers_final.jsonl"), "w") as f:
    for l in open(os.path.join(D, "answers_full.jsonl")):
        r = json.loads(l)
        if r["mode"] == "random": f.write(l)
        elif r["eta"] == 1.0 and r["mode"] == "ranked": r["mode"] = rule; f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("seeded answers_final.jsonl (random arms + eta=1 rows) for rule", rule)
PY
if [ "$RULE" = "topk_L12" ]; then rm -f data/scores_final.npz; python run_calib.py score --tag final --questions questions_full.jsonl; fi
python run_calib.py answer --tag final --questions questions_full.jsonl --mode "$RULE" --etas 0.05,0.1,0.15,0.2,0.3,0.5,0.75,1 --random_etas ""
python final_analysis.py final "$RULE" L12_q
echo "FINAL RUN FINISHED"
