#!/usr/bin/env bash
# Ablation grid on the fixed 52x2 question set. Greedy decoding (deterministic); uncertainty via bootstrap CIs.
set -e; cd "$(dirname "$0")"; P=.venv/bin/python
run(){ name=$1; shift; env "$@" OUT=results/answers_$name.json $P src/rag.py > logs/$name.log 2>&1; IN=results/answers_$name.json OUT=results/audit_$name.json $P src/audit.py >> logs/$name.log 2>&1; echo "done $name"; }
mkdir -p logs
#run A_1.5b_same_basic   LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct RETR=same PROMPT=basic
#run B_1.5b_same_fewshot LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct RETR=same PROMPT=fewshot
run C_1.5b_mixrr_fewshot LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct RETR=mixed_rr PROMPT=fewshot
run E_0.5b_mixrr_fewshot LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct RETR=mixed_rr PROMPT=fewshot
for x in C_1.5b_mixrr_fewshot E_0.5b_mixrr_fewshot; do y=${x}_posthoc
  IN=results/answers_$x.json OUT=results/answers_$y.json $P src/attribute.py > logs/$y.log 2>&1
  IN=results/answers_$y.json OUT=results/audit_$y.json $P src/audit.py >> logs/$y.log 2>&1; echo "done $y"; done
