#!/usr/bin/env bash
# Regenerates exactly the configs reported in the README (A B D E F G H) plus determinism reruns.
# Resumable: each step is skipped if its output exists; FORCE=1 redoes everything. GEN_ONLY=1 stops after generation.
# Greedy decoding, THREADS=4 fixed; 0.5B in fp32, 1.5B in bf16 (memory). Uncertainty: paired question-clustered bootstrap (src/compare.py).
set -euo pipefail; cd "$(dirname "$0")"; P=${PYTHON:-.venv/bin/python}; mkdir -p logs results
M05=Qwen/Qwen2.5-0.5B-Instruct; M15=Qwen/Qwen2.5-1.5B-Instruct; EVAL=data/qa.json
# step OUT IN cmd...: skipped when OUT exists and is newer than its input file IN ("-" = no input file).
# Code changes do not invalidate outputs; use FORCE=1 (everything) or delete the specific output file.
step(){ out=$1; in=$2; shift 2
  if [[ -s $out && ${FORCE:-0} != 1 && ( $in == - || $out -nt $in ) ]]; then echo "skip $out"; return; fi
  echo "run  $out $(date +%H:%M)"; env "$@" > "logs/$(basename "$out" .json).log" 2>&1; }
gen(){ name=$1; shift; step results/answers_$name.json - "$@" OUT=results/answers_$name.json $P src/rag.py; }
posthoc(){ step results/answers_$2.json results/answers_$1.json IN=results/answers_$1.json OUT=results/answers_$2.json $P src/attribute.py; }
gen D_0.5b_same_basic      LLM_MODEL=$M05 RETR=same     PROMPT=basic
gen F_0.5b_mixrr_fewshot   LLM_MODEL=$M05 RETR=mixed_rr PROMPT=fewshot
gen E_0.5b_same_fewshot    LLM_MODEL=$M05 RETR=same     PROMPT=fewshot
gen A_1.5b_same_basic      LLM_MODEL=$M15 RETR=same     PROMPT=basic   DTYPE=bfloat16
gen B_1.5b_same_fewshot    LLM_MODEL=$M15 RETR=same     PROMPT=fewshot DTYPE=bfloat16
posthoc F_0.5b_mixrr_fewshot G_0.5b_mixrr_fewshot_posthoc
posthoc B_1.5b_same_fewshot  H_1.5b_same_fewshot_posthoc
# determinism: identical settings must reproduce byte-for-byte; a different thread count measures sensitivity
gen rerun_D_0.5b_same_basic          QA=$EVAL LLM_MODEL=$M05 RETR=same PROMPT=basic THREADS=4
gen rerun_D_0.5b_same_basic_threads8 QA=$EVAL LLM_MODEL=$M05 RETR=same PROMPT=basic THREADS=8
gen rerun_B_1.5b_same_fewshot        QA=$EVAL LLM_MODEL=$M15 RETR=same PROMPT=fewshot DTYPE=bfloat16
[[ ${GEN_ONLY:-0} == 1 ]] && { echo "generation done"; exit 0; }
for f in results/answers_[A-H]_*.json; do n=$(basename "$f" .json); n=${n#answers_}
  step results/audit_$n.json $f IN=$f OUT=results/audit_$n.json $P src/audit.py; done
step results/calibration.json data/calibration_set.json $P src/calibrate.py
$P src/determinism.py && $P src/compare.py && $P src/abstain.py && $P src/summarize.py > /dev/null && echo "grid done"
