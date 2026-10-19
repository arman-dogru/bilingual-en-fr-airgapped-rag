#!/usr/bin/env bash
# Capture real src.ask output for the README terminal figure -> docs/demo/ask_demo.txt
# (needs the pinned models in the HF cache; ~1 min on CPU once cached). The figure itself is drawn by scripts/make_figures.py.
# Each "$ ..." line in the capture is exactly the command that produced the line below it (stderr: model-loading progress, dropped).
set -euo pipefail; cd "$(dirname "$0")/.."
F='{lang, answer, top_score, tau, abstained, cites: [.citation_check[]? | {verdict, entail}], source_1: .sources[0]?.url}'
out=docs/demo/ask_demo.txt; mkdir -p docs/demo; : > "$out"
for q in "How long do EI sickness benefits last?" \
         "Combien de temps durent les prestations de maladie de l'assurance-emploi ?" \
         "What is the population of Tokyo?"; do
  printf '$ .venv/bin/python -m src.ask "%s" 2>/dev/null | jq -c '"'"'%s'"'"'\n' "$q" "$F" >> "$out"
  .venv/bin/python -m src.ask "$q" 2>/dev/null | jq -c "$F" >> "$out"
  echo >> "$out"
done
cat "$out"
