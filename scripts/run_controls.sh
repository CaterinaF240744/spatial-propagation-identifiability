#!/bin/sh
# Controls: synthetic data on the GSE182939 design, score without Vcam1/Ccl2, design analysis, joint fit of both cohorts.
mkdir -p results logs
SYN=null    python3 real_fit.py full > logs/syn_null.txt 2>&1
SYN=table1  python3 real_fit.py full > logs/syn_table1.txt 2>&1
SCORE=noVC  python3 real_fit.py full > logs/noVC_A.txt 2>&1
SCORE=noVC INPUTS=inputs_B TPS=4h,12h,48h NW=1000 NS=1000 python3 real_fit.py full > logs/noVC_B.txt 2>&1
for t in table1 null; do for f in 0.5 0.25; do for k in 4 8; do
  python3 design_power.py $t $f $k > logs/design_${t}_${f}_${k}.txt 2>&1
done; done; done
NW=500 NS=500 python3 joint_fit.py full    > logs/joint_full.txt 2>&1
NW=500 NS=500 python3 joint_fit.py uniform > logs/joint_uniform.txt 2>&1
mv result_*.json results/ 2>/dev/null; echo "controls done"
