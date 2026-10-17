# Bilingual EN/FR air-gapped RAG with citation audit

An offline question-answering system over open Government of Canada pages in English and French. After a one-time download it runs **fully air-gapped on CPU**, and it scores how faithful its citations are using a calibrated NLI judge.

## Motivation
Public-sector and regulated customers need LLM systems that run on-prem or fully disconnected (as with Cohere North), work in both official languages, and can show where every answer came from. This project asks how far a small local open model can get on retrieval across EN↔FR and on answers whose citations actually support their claims, and which interventions help most.

## Approach
| Component | Choice |
|---|---|
| Corpus | 20 EN/FR page pairs from Canada.ca (EI, CPP/OAS/GIS, CCB, GST/HST, RRSP, TFSA, dental, passports, citizenship, permits, Express Entry, student aid, CDB, EI fishing) under the Open Government Licence – Canada. URLs are in `data/corpus.json`. |
| Chunking | 3-paragraph windows that overlap by 1 paragraph (358 EN and 383 FR chunks). |
| Retriever | `intfloat/multilingual-e5-small` with indexes per language and one mixed EN+FR index. |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (top-20 → top-k). |
| Generator | `Qwen2.5-0.5B/1.5B-Instruct`, bf16 on CPU, greedy decoding, top-3 sources, inline `[n]` citations. |
| Citation forcing | Few-shot demo plus a "cite every sentence" rule, and/or **post-hoc attribution**: each uncited sentence gets the source most similar to it in e5 embeddings. That is a different model from the judge, so the judge isn't grading its own choices. |
| Judge | `mDeBERTa-v3-base-xnli` entailment ≥ 0.5 between the cited chunk and the claim, after stripping attribution prefixes ("According to the source, …"). Fragments under 4 words are dropped. |

## Evaluation protocol
- **Fixed held-out set:** `data/qa.json` has 52 questions, each written in both EN and FR (104 queries). Each has a gold document and an evidence regex, checked to occur in both language versions. A passage hit means the gold document *and* a chunk containing the evidence. Nothing was tuned on this set, apart from choosing the judge threshold on a separate hand-labelled claim sample.
- **Retrieval:** passage and document recall@1/3/5 in 4 directions plus the mixed index, with and without reranking. Uncertainty is a 95% Wilson CI.
- **Generation:** citation faithfulness (supported cited claims / all claims), uncited rate, faithfulness when cited, grounding in any retrieved source, and answer-language match. Uncertainty is a 95% bootstrap CI, resampling whole answers (1,000 draws).
- **Seeds:** decoding is greedy and therefore deterministic, so variance comes from the question sample, which the CIs capture rather than seed repeats.
- **Judge calibration:** 30 claims sampled with seed 0 and hand-labelled (21 were factual claims), in `data/calibration_labels.json`.

## Results

### Retrieval (n=52 per direction)
| Direction | Rerank | passage R@1 [95% CI] | R@5 | doc R@1 |
|---|---|---|---|---|
| EN→EN | no / yes | 0.90 [0.79, 0.96] / 0.90 | 0.98 / 1.00 | 0.98 / 0.96 |
| FR→FR | no / yes | 0.88 [0.77, 0.95] / 0.88 | 1.00 / 0.98 | 0.96 / 0.94 |
| EN→FR | no / yes | 0.62 [0.48, 0.73] / **0.77** [0.64, 0.86] | 0.92 / 0.92 | 0.73 / 0.87 |
| FR→EN | no / yes | 0.64 [0.50, 0.75] / **0.90** [0.79, 0.96] | 0.96 / 0.96 | 0.81 / 0.92 |
| EN→mixed | no / yes | 0.90 / 0.90 | 0.98 / 1.00 | 0.98 / 0.96 |
| FR→mixed | no / yes | 0.88 / 0.88 | 1.00 / 0.98 | 0.96 / 0.94 |

The reranker closes most of the cross-lingual top-1 gap (+15 points EN→FR, +27 points FR→EN). The mixed index behaves exactly like same-language retrieval: e5 always ranks chunks in the query's language first, so the mixed index gives no extra recall on this corpus.

### Generation and citation audit (104 answers; ablations)
| Config | Uncited | Citation faithfulness [95% CI] | Faithful when cited | Grounded in any source | Lang match |
|---|---|---|---|---|---|
| 1.5B · same-lang · basic prompt | 0.68 | 0.19 [0.14, 0.24] | 0.58 | 0.72 | 1.00 |
| 1.5B · same-lang · **few-shot** | 0.25 | **0.50** [0.40, 0.60] | 0.66 | 0.78 | 1.00 |
| 0.5B · same-lang · basic | 0.92 | 0.06 [0.02, 0.11] | 0.71 | 0.72 | 0.98 |
| 0.5B · same-lang · few-shot | 0.72 | 0.18 [0.12, 0.24] | 0.64 | 0.75 | 1.00 |
| 0.5B · mixed+rerank · few-shot | 0.74 | 0.15 [0.10, 0.21] | 0.58 | 0.73 | 0.99 |
| 0.5B · mixed+rerank · few-shot · **post-hoc attribution** | 0.00 | **0.58** [0.49, 0.66] | 0.58 | 0.73 | 0.99 |

Takeaways:
1. **Missing citations were the main problem, not hallucination.** About 72–78% of claims are entailed by *some* retrieved chunk in every configuration.
2. **Few-shot prompting** roughly triples citation faithfulness on both model sizes.
3. **Post-hoc attribution** gets the 0.5B model to 0.58, above the 1.5B few-shot model, at almost no cost.
4. **Precision of cited claims stays at about 0.6.** Roughly 40% of cited sentences are still unsupported, so grounding is the next bottleneck.
5. **The mixed index plus reranker doesn't help generation** here (0.15 vs 0.18, overlapping CIs), which matches the retrieval finding above.

### Judge calibration (21 factual claims)
| Setting @0.5 | Accuracy | Precision | Recall |
|---|---|---|---|
| raw claims | 0.86 | 0.88 | 0.94 |
| prefix-normalized | **0.90** | 0.89 | **1.00** |

Thresholds of 0.7 and 0.9 lose recall sharply (see `results/tables.md`).

## Error analysis (qualitative)
- **Wrong number, still "entailed":** "il faut compter au moins 365 jours de présence effective…" (gold: 1 095 days). The judge scored it 0.97 because a retrieved chunk mentions "365 days" for temporary-resident credit. This NLI false positive is a known weakness with numbers.
- **Entity confusion:** "persons with disabilities, their caregivers, and the CRA are eligible for the CDB…" mixes a navigation link into the answer. The normalized judge correctly rejects it (0.26).
- **Deflection instead of answering:** "Pour obtenir ces détails, vous pouvez consulter le site web [Mesures temporaires…]" is supported by the source but doesn't answer the question. Faithfulness metrics don't penalize this.
- **Cross-lingual retrieval misses:** "RRSP options at 71" EN→FR retrieves GIS pages. The Express Entry "3 programs" sentence is split from its list. Short nav-style chunks rank high without reranking.
- **Before prefix-stripping**, answers starting "According to Source, …" scored 0.35 even when correct. This was the main source of false negatives.

## Setup and reproduction
```bash
uv venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python sentence-transformers transformers beautifulsoup4 requests numpy
.venv/bin/python src/fetch_corpus.py       # online: corpus (data/corpus.json is committed)
.venv/bin/python src/retrieval.py          # retrieval table (downloads embedder + reranker)
./run_grid.sh                              # 1.5B configs (A, B)
./run_grid05.sh                            # 0.5B ablations (D–G) incl. post-hoc attribution
NORMALIZE=0 IN=results/answers_qwen1.5b.json OUT=results/calib_norm0.json .venv/bin/python src/audit.py
NORMALIZE=1 IN=results/answers_qwen1.5b.json OUT=results/calib_norm1.json .venv/bin/python src/audit.py
.venv/bin/python src/calibrate.py && .venv/bin/python src/summarize.py   # -> results/tables.md
./run_offline.sh                           # full pipeline with networking disabled
```
Environment variables: `LLM_MODEL`, `EMB_MODEL`, `RERANKER`, `NLI_MODEL`, `RETR` (`same` or `mixed_rr`), `PROMPT` (`basic` or `fewshot`), `NORMALIZE`, `IN`/`OUT`.

**Air-gapped run:** `run_offline.sh` runs inside `unshare -rn`, a network namespace with only loopback. It first asserts that `curl huggingface.co` fails, then runs with `HF_HUB_OFFLINE=1`. It was verified with the 0.5B pipeline (`results/offline_run.log`).

## Limitations
- The corpus is small (20 page pairs) and some pages are mostly navigation. The Q&A set and gold labels were written by one annotator.
- The judge was calibrated on only 21 claims, and NLI is weak on numbers. Faithfulness is an estimate (accuracy about 0.9), not ground truth.
- Because of RAM pressure on the shared box, the mixed+rerank and post-hoc configs ran on 0.5B only. The 1.5B + few-shot + post-hoc combination is still untested.
- The faithfulness metrics don't measure answer correctness or completeness against a gold answer.
- bf16 on CPU isn't bit-deterministic across thread counts, so offline and online numbers differ slightly.

## Next steps
1. Run 1.5B (and a ~3B Q4 GGUF via llama.cpp) with few-shot plus post-hoc attribution.
2. Add a numeric-consistency check (extract numbers and compare them with the cited chunk) to the judge.
3. Score answer correctness against gold answers and penalize deflections.
4. Grow the calibration set to 100+ claims and add a second annotator.
5. Use structure-aware chunking that keeps a heading with its list.

## Data and licence
Source pages: canada.ca under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada). No hosted APIs are used at any stage.
