#!/bin/sh
# GSE269622: all five parameterisations, fit on all data + leave-one-time-point-out, seeds 1-3. Run from the repository root.
# Usage: sh scripts/run_cohort_B.sh ["variants"] ["seeds"]
VARIANTS=${1:-"full zone uniform table1 nograph"}; SEEDS=${2:-"1 2 3"}
export NW=1000 NS=1000 INPUTS=inputs_B TPS=4h,12h,48h
mkdir -p results logs
for SEED in $SEEDS; do export SEED
  for v in $VARIANTS; do
    [ -f results/result_${v}_B_s${SEED}n1000.json ] || { python3 real_fit.py $v > logs/B_${v}_s${SEED}.txt 2>&1; mv result_${v}_B_s${SEED}n1000.json results/ 2>/dev/null; }
    for h in 4h 12h 48h; do
      [ -f results/result_${v}_hold${h}_B_s${SEED}n1000.json ] || { python3 real_fit.py $v $h > logs/B_${v}_h${h}_s${SEED}.txt 2>&1; mv result_${v}_hold${h}_B_s${SEED}n1000.json results/ 2>/dev/null; }
    done
    echo "done $v (seed $SEED)"
  done
done
