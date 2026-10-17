#!/usr/bin/env bash
# Full pipeline with networking disabled (user + network namespace: only loopback, no routes).
set -euo pipefail; cd "$(dirname "$0")"
unshare -rn bash -c '
  set -e; echo "== network check"; ip -o link 2>/dev/null | cut -d: -f2 | tr "\n" " "; echo
  if curl -s -m 5 https://huggingface.co >/dev/null; then echo "NETWORK REACHABLE - abort"; exit 1; else echo "no network: OK"; fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv/bin/python src/retrieval.py >/dev/null && echo "retrieval ok"
  OUT=results/offline_answers.json .venv/bin/python src/rag.py >/dev/null && echo "rag ok"
  IN=results/offline_answers.json OUT=results/offline_audit.json .venv/bin/python src/audit.py | grep -E "faithfulness\"|language"
' 2>&1 | grep -v -iE "warn|loading weights"
