#!/bin/zsh
# Full calibration run: score new questions (2B), then answer all missing arms (32B).
cd "$(dirname "$0")" && source ../.venv/bin/activate
python merge_full.py && \
python run_calib.py score  --tag full --questions questions_full.jsonl && \
python run_calib.py answer --tag full --questions questions_full.jsonl --etas 0.05,0.1,0.15,0.2,0.3,0.5,0.75,1 --random_etas 0.1,0.2
echo "FULL RUN FINISHED"
