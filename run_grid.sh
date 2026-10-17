#!/usr/bin/env bash
# Regenerates exactly the configs reported in the README (A B D E F G H) plus the determinism reruns.
# Each step is skipped if its output already exists (resumable); FORCE=1 redoes everything.
# Greedy decoding, fp32, THREADS=4 fixed (see src/rag.py). Uncertainty: paired, question-clustered bootstrap (src/compare.py).
set -euo pipefail; cd "$(dirname "$0")"; P=${PYTHON:-.venv/bin/python}; mkdir -p logs results
M05=Qwen/Qwen2.5-0.5B-Instruct; M15=Qwen/Qwen2.5-1.5B-Instruct
step(){ out=$1; shift; if [[ -s $out && ${FORCE:-0} != 1 ]]; then echo "skip $out"; return; fi; echo "run  $out"; env "$@" > "logs/$(basename "$out" .json).log" 2>&1; }
gen(){ name=$1; shift; step results/answers_$name.json "$@" OUT=results/answers_$name.json $P src/rag.py; }
posthoc(){ step results/answers_$2.json IN=results/answers_$1.json OUT=results/answers_$2.json $P src/attribute.py; }
gen D_0.5b_same_basic      LLM_MODEL=$M05 RETR=same     PROMPT=basic
gen E_0.5b_same_fewshot    LLM_MODEL=$M05 RETR=same     PROMPT=fewshot
gen F_0.5b_mixrr_fewshot   LLM_MODEL=$M05 RETR=mixed_rr PROMPT=fewshot
gen D_0.5b_same_basic_rerun   LLM_MODEL=$M05 RETR=same PROMPT=basic THREADS=4   # same settings: must match exactly
gen D_0.5b_same_basic_threads8 LLM_MODEL=$M05 RETR=same PROMPT=basic THREADS=8  # thread-count sensitivity
gen A_1.5b_same_basic      LLM_MODEL=$M15 RETR=same     PROMPT=basic
gen B_1.5b_same_fewshot    LLM_MODEL=$M15 RETR=same     PROMPT=fewshot
posthoc F_0.5b_mixrr_fewshot G_0.5b_mixrr_fewshot_posthoc
posthoc B_1.5b_same_fewshot  H_1.5b_same_fewshot_posthoc
[[ ${GEN_ONLY:-0} == 1 ]] && { echo "generation done"; exit 0; }
for f in results/answers_*.json; do n=$(basename "$f" .json); n=${n#answers_}
  step results/audit_$n.json IN=$f OUT=results/audit_$n.json $P src/audit.py; done
$P src/compare.py && $P src/abstain.py && $P src/summarize.py > /dev/null && echo "grid done"
