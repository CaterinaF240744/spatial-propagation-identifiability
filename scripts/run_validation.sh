#!/bin/sh
# Validation analyses. Run from the folder that contains real_fit.py, inputs/ and inputs_B/.
#  (1) prior sensitivity: full model with prior s.d. 0.25 and 1.0 (default 0.5) -> does the posterior follow the prior?
#  (2) graph permutation: uniform model with NPERM randomly relabelled graphs, every time point held out in turn
#      -> does the real graph predict held-out data better than the same graph attached to the wrong compartments?
# Usage: sh scripts/run_validation.sh B [NPERM]     (cohort B, fast)      sh scripts/run_validation.sh A [NPERM]
C=${1:-B}; NPERM=${2:-20}
if [ "$C" = "B" ]; then export INPUTS=inputs_B TPS=4h,12h,48h NW=1000 NS=1000; HOLDS="4h 12h 48h"; SFX=_B; NN=1000
else export INPUTS=inputs TPS=4h,12h,48h,6wk NW=600 NS=600; HOLDS="4h 12h 48h 6wk"; SFX=; NN=600; fi
export SEED=1
for p in 0.25 1; do
  [ -f result_full_psd${p}${SFX}_s1n${NN}.json ] || PRIOR_SD=$p python3 real_fit.py full > log_val_psd${p}${SFX}.txt 2>&1
done
echo "prior sensitivity done ($C)"
i=1
while [ $i -le $NPERM ]; do
  for h in $HOLDS; do
    [ -f result_uniform_hold${h}_perm${i}${SFX}_s1n${NN}.json ] || PERM=$i python3 real_fit.py uniform $h > log_val_perm${i}_${h}${SFX}.txt 2>&1
  done
  echo "permutation $i/$NPERM done ($C)"
  i=$((i+1))
done
echo "FINITO validation $C"
