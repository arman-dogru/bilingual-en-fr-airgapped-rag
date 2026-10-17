#!/usr/bin/env bash
# Air-gapped check: inside a user+network namespace (loopback only) run retrieval, config F generation
# (mixed-language retrieval + reranker + few-shot), post-hoc attribution (-> G) and the citation audit, then verify
# the offline answers are byte-identical to the committed online run. Full logs: logs/offline_*.log
set -euo pipefail; cd "$(dirname "$0")"; mkdir -p logs
command -v curl >/dev/null || { echo "curl is required for the network check"; exit 2; }
command -v unshare >/dev/null || { echo "unshare is required"; exit 2; }
unshare -rn bash -c '
  set -euo pipefail; P=.venv/bin/python
  echo "== interfaces: $(ip -o link 2>/dev/null | cut -d: -f2 | tr "\n" " ")"
  rc=0; curl -s -m 5 https://huggingface.co >/dev/null || rc=$?
  case $rc in 6|7) echo "no network: OK (curl rc=$rc: could not resolve/connect)";; 0) echo "NETWORK REACHABLE - abort"; exit 1;; *) echo "unexpected curl rc=$rc - abort"; exit 1;; esac
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  $P -m src.retrieval > logs/offline_retrieval.log 2>&1 && echo "retrieval ok"
  QA=data/qa.json LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct RETR=mixed_rr PROMPT=fewshot OUT=results/offline_answers_F.json $P -m src.rag > logs/offline_rag.log 2>&1 && echo "rag (F) ok"
  IN=results/offline_answers_F.json OUT=results/offline_answers_G.json $P -m src.attribute > logs/offline_attr.log 2>&1 && echo "post-hoc attribution ok"
  IN=results/offline_answers_G.json OUT=results/offline_audit_G.json $P -m src.audit > logs/offline_audit.log 2>&1 && echo "audit ok"
  $P - <<PY
import json
a = {(x["id"], x["lang"]): x["answer"] for x in json.load(open("results/offline_answers_F.json"))["answers"]}
b = {(x["id"], x["lang"]): x["answer"] for x in json.load(open("results/answers_F_0.5b_mixrr_fewshot.json"))["answers"]}
same = sum(a[k] == b.get(k) for k in a); print(f"offline vs online F answers identical: {same}/{len(a)}")
g1 = {(x["id"], x["lang"]): x["answer"] for x in json.load(open("results/offline_answers_G.json"))["answers"]}
g2 = {(x["id"], x["lang"]): x["answer"] for x in json.load(open("results/answers_G_0.5b_mixrr_fewshot_posthoc.json"))["answers"]}
print(f"offline vs online G answers identical: {sum(g1[k] == g2.get(k) for k in g1)}/{len(g1)}")
s = json.load(open("results/offline_audit_G.json"))["summary"]["all_answerable"]; o = json.load(open("results/audit_G_0.5b_mixrr_fewshot_posthoc.json"))["summary"]["all_answerable"]
keys = [k for k in s if not k.endswith("ci95")]
print("offline G citation faithfulness:", s["citation_faithfulness"], s["citation_faithfulness_ci95"], "| audit summary identical to online:", all(s[k] == o[k] for k in keys))
PY
' 2>&1 | tee results/offline_run.log
