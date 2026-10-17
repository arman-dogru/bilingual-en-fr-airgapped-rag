#!/usr/bin/env bash
# 0.5B ablation grid (1.5B C run stalled under box memory pressure). Greedy decoding; bootstrap CIs in audit.
set -e; cd "$(dirname "$0")"; P=.venv/bin/python; mkdir -p logs
run(){ name=$1; shift; env "$@" OUT=results/answers_$name.json $P src/rag.py > logs/$name.log 2>&1; IN=results/answers_$name.json OUT=results/audit_$name.json $P src/audit.py >> logs/$name.log 2>&1; echo "done $name"; }
M=LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
run D_0.5b_same_basic $M RETR=same PROMPT=basic
run E_0.5b_same_fewshot $M RETR=same PROMPT=fewshot
run F_0.5b_mixrr_fewshot $M RETR=mixed_rr PROMPT=fewshot
y=G_0.5b_mixrr_fewshot_posthoc
IN=results/answers_F_0.5b_mixrr_fewshot.json OUT=results/answers_$y.json $P src/attribute.py > logs/$y.log 2>&1
IN=results/answers_$y.json OUT=results/audit_$y.json $P src/audit.py >> logs/$y.log 2>&1; echo "done $y"
