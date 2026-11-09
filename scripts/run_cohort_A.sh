#!/bin/sh
# GSE182939: all five parameterisations, fit on all data + leave-one-time-point-out. Run from the repository root.
# Usage: sh scripts/run_cohort_A.sh [seed] ["variants"]     e.g. sh scripts/run_cohort_A.sh 1 "full table1 uniform"
SEED=${1:-1}; VARIANTS=${2:-"full zone uniform table1 nograph"}
export NW=600 NS=600 SEED INPUTS=inputs TPS=4h,12h,48h,6wk
mkdir -p results logs
for v in $VARIANTS; do
  [ -f results/result_${v}_s${SEED}n600.json ] || { python3 real_fit.py $v > logs/A_${v}_s${SEED}.txt 2>&1; mv result_${v}_s${SEED}n600.json results/ 2>/dev/null; }
  for h in 4h 12h 48h 6wk; do
    [ -f results/result_${v}_hold${h}_s${SEED}n600.json ] || { python3 real_fit.py $v $h > logs/A_${v}_h${h}_s${SEED}.txt 2>&1; mv result_${v}_hold${h}_s${SEED}n600.json results/ 2>/dev/null; }
  done
  echo "done $v (seed $SEED)"
done
