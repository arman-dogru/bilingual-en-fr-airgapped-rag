# Results: bilingual EN/FR air-gapped RAG with citation audit

Run on 2026-10-16 on an 8-core CPU with no GPU.

## Setup
- **Corpus:** 20 bilingual Canada.ca page pairs covering EI, CPP/OAS/GIS, CCB, GST/HST, RRSP, TFSA, dental care, passports, citizenship, study and work permits, Express Entry, student aid, CDB and EI fishing. They are released under the Open Government Licence – Canada, and the URLs are in `data/corpus.json`. Each page is split into 3-paragraph windows that overlap by 1 paragraph.
- **Q&A set:** 26 questions, each written in both EN and FR (52 queries). Every question has a gold document and an evidence pattern (`data/qa.json`).
- **Embedder:** `intfloat/multilingual-e5-small`. **LLMs:** `Qwen2.5-0.5B-Instruct` and `Qwen2.5-1.5B-Instruct`, both bf16 on CPU. **Judge:** `mDeBERTa-v3-base-xnli` NLI with entailment ≥ 0.5.

## Retrieval recall (passage-level = gold doc AND chunk contains the evidence)
| Direction | R@1 | R@3 | R@5 | doc R@1 | doc R@5 |
|---|---|---|---|---|---|
| EN→EN | 0.92 | 0.96 | 1.00 | 0.96 | 1.00 |
| FR→FR | 0.85 | 0.96 | 0.96 | 0.96 | 1.00 |
| EN→FR | 0.65 | 0.81 | 0.89 | 0.81 | 0.96 |
| FR→EN | 0.69 | 0.96 | 0.96 | 0.89 | 1.00 |

Cross-lingual retrieval mostly costs precision at rank 1. By k=5 it nearly catches up with same-language retrieval.

## Generation + citation audit (same-language retrieval, top-3)
| Model | Claims | Cited | Faithfulness (supported/all) | Faithful when cited | Grounded in any retrieved source | Language match |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 85 | 4.7% | 0.02 | 0.50 | 0.75 | 0.98 |
| Qwen2.5-1.5B | 98 | 35.7% | 0.17 | 0.49 | 0.65 | 1.00 |

Faithfulness for the 1.5B model is 0.29 in EN and 0.13 in FR.

## Error analysis
- **Citation compliance is the main failure.** Small models skip the `[n]` format. The 0.5B model cites almost nothing. The 1.5B model often writes "[1], [2] et [3]" in a lead-in phrase or uses vague phrasing like "according to the first link".
- **The NLI judge undercounts support.** Claims such as "According to Source, you need … 7 consecutive days" are correct but score 0.35 because of the attribution prefix. Stripping attribution phrases before the NLI check is the next fix.
- **FR answers paraphrase more and add unsupported hedges**, for example "La durée exacte peut varier…".
- **Retrieval misses** come from short or nav-like chunks (Express Entry "3 programs" splits away from its list), the EN→FR RRSP-at-71 question, and the citizenship "1 095 days" number format in FR.

## Offline run
`./run_offline.sh` runs retrieval, generation and audit inside `unshare -rn`, which leaves only loopback, and confirms that `curl huggingface.co` fails first. The log is in `results/offline_run.log`, which shows the whole pipeline passing with no network. The offline numbers differ slightly from the online run (faithfulness 0.012 vs 0.024) because bf16 CPU generation is not bit-deterministic across thread scheduling.

## Next
1. Normalize claims before the NLI check (strip attribution prefixes) and hand-label about 30 claims to calibrate the judge.
2. Enforce citations by constrained or few-shot prompting, or by post-hoc citation attribution (assign each sentence its highest-entailment source), then compare.
3. Retrieve across both languages together (a single mixed index) and add a reranker.
4. Grow the Q&A set to 40–60 per language.
