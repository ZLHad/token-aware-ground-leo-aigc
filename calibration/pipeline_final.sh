#!/bin/zsh
cd "$(dirname "$0")" && source ../.venv/bin/activate
./launch_final.sh topk_L12 > logs/final_run.log 2>&1
python timing_sweep.py > logs/timing_sweep.log 2>&1
python final_analysis.py full ranked H > logs/final_analysis_full.log 2>&1      # keep the L2 run's report for comparison
echo "PIPELINE FINISHED" >> logs/final_run.log
