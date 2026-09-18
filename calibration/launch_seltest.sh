#!/bin/zsh
cd "$(dirname "$0")" && source ../.venv/bin/activate
until grep -q "FULL RUN FINISHED" logs/full_run.log; do sleep 30; done
python run_seltest.py > logs/seltest.log 2>&1
python analyze_seltest.py >> logs/seltest.log 2>&1
