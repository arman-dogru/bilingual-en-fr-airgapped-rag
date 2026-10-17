# Bilingual EN/FR air-gapped RAG with citation audit

An offline question-answering system over open Government of Canada pages in English and French. After a one-time download it runs **fully air-gapped on CPU**, and it scores how faithful its citations are using an NLI judge (calibration is indicative, see below). It also abstains on out-of-corpus questions.

## Motivation
Public-sector and regulated customers need LLM systems that run on-prem or fully disconnected (as with Cohere North), work in both official languages, and can show where every answer came from. This project asks how far a small local open model can get on retrieval across EN↔FR and on answers whose citations actually support their claims, and which interventions help most.

## Approach
| Component | Choice |
|---|---|
| Corpus | 20 EN/FR page pairs from Canada.ca (EI, CPP/OAS/GIS, CCB, GST/HST, RRSP, TFSA, dental, passports, citizenship, permits, Express Entry, student aid, CDB, EI fishing) under the Open Government Licence – Canada. URLs are in `data/corpus.json`. |
| Chunking | 3-paragraph windows that overlap by 1 paragraph (310 EN and 339 FR chunks). |
| Retriever | `intfloat/multilingual-e5-small` with indexes per language and one mixed EN+FR index. |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (top-20 → top-k). |
| Generator | `Qwen2.5-0.5B-Instruct` (fp32) / `Qwen2.5-1.5B-Instruct` (bf16) on CPU, greedy decoding with 4 fixed threads, top-3 sources wrapped in `<source>` delimiters with a "treat as data" instruction, inline `[n]` citations, and a fixed refusal sentence when the sources don't answer. |
| Citation forcing | Few-shot demo plus a "cite every sentence" rule, and/or **post-hoc attribution**: each uncited sentence gets the source most similar to it in e5 embeddings. That is a different model from the judge, so the judge isn't grading its own choices. |
| Judge | `mDeBERTa-v3-base-xnli` entailment ≥ 0.5 between the cited chunk and the claim, after stripping attribution prefixes ("According to the source, …"). Sentences under 4 words are not scored unless they contain a number (`Until age 85 [1].` is scored). They are still kept in the answer text, and citations such as `[1, 2]`, `[1-3]` and `.[1]` are parsed. The judge sees exactly the source text the generator saw. |

## Evaluation protocol
- **Splits.** `data/qa.json` has 52 questions, each in EN and FR (104 queries), with a gold document and an evidence regex that occurs in both language versions (`tests/test_data.py` checks this).
  - **DEV = q01–q26.** These were written first. The few-shot prompt, the attribution-prefix regex, the fragment filter, the judge threshold and every calibration label were developed on outputs for these questions, so DEV numbers are optimistic.
  - **TEST = q27–q52.** These were added later and never used to tune anything. The same author (the Developer agent) wrote both splits, so they're disjoint but not independently sourced. `tests/test_data.py` asserts the IDs and question texts don't overlap. **Headline numbers are TEST.**
  - **UNANSWERABLE = u01–u25** (`data/unanswerable.json`, EN+FR). These are out-of-corpus questions (other jurisdictions, general knowledge) used to evaluate abstention. u01–u12 are dev (used to pick the gate threshold) and u13–u25 are test.
- **Retrieval:** passage and document recall@1/3/5 in 4 directions plus the mixed index, with and without reranking, with a 95% Wilson CI. Evidence regexes are word-bounded and answer-specific (`ev` field).
- **Generation:** citation faithfulness (supported cited claims / all claims), uncited rate, faithfulness when cited, grounding in any retrieved source, and answer-language match (`und` when undetermined, with no default to EN). CIs are a **question-clustered bootstrap**: the EN and FR answers to a question are resampled together, zero-claim answers stay in, and the result is computed with `numpy.percentile`.
- **Config comparisons:** a **paired** question-clustered bootstrap over the same resampled questions for both configs (5,000 draws), with a two-sided p, in `src/compare.py` and `results/compare.json`. A delta is called a difference only when its 95% CI excludes 0.
- **Determinism:** greedy decoding, fixed seeds, `torch.set_num_threads(4)`, and pinned model revisions (`src/config.py`). The 0.5B model runs in fp32. The 1.5B model runs in bf16 because fp32 1.5B doesn't fit in RAM next to the other jobs on the 16 GB box. Each config is rerun once with the same settings, and the 0.5B model once with 8 threads. `results/determinism.json` reports the byte-identical answer rate.
- **Judge calibration** (`data/calibration_set.json`, `src/calibrate.py`): items are keyed by **(id, lang, claim text)**, and `calibrate.py` fails if a generated claim no longer occurs in its source answer. The set has three kinds of item:
  - 21 generated claims from the dev run: 16 supported, 5 not.
  - 16 **perturbed** hard negatives: each supported claim with one number, polarity or entity changed.
  - 16 **source-swap** easy negatives: each supported claim judged against the sources of an unrelated question.
  
  All labels and perturbations were written by the Developer agent (an LLM) that also built the pipeline. That makes it one annotator, with no human check, and every item comes from dev questions. Treat the judge as **indicative**, not validated.
- **Abstention:** `llm` means the generator says the sources don't contain the answer. `gate` means the system refuses before generating when the top retrieval score is below τ, with τ chosen on dev to maximize balanced accuracy. Results are reported on test as precision and recall for "should abstain", plus the false-refusal rate on answerable questions.

## Results
All tables are generated by `src/summarize.py` into [`results/tables.md`](results/tables.md). The numbers below are copied from it.

### Retrieval (n=52 questions per direction; full table in `results/tables.md`)
| Direction | passage R@1, no rerank [95% CI] | passage R@1, rerank [95% CI] | R@5 (no / rerank) |
|---|---|---|---|
| EN→EN | 0.90 [0.79, 0.96] | 0.90 [0.79, 0.96] | 0.98 / 1.00 |
| FR→FR | 0.88 [0.77, 0.95] | 0.88 [0.77, 0.95] | 1.00 / 0.98 |
| EN→FR | 0.62 [0.48, 0.74] | **0.77** [0.64, 0.86] | 0.92 / 0.92 |
| FR→EN | 0.62 [0.48, 0.74] | **0.90** [0.79, 0.96] | 0.96 / 0.96 |

The reranker closes most of the cross-lingual top-1 gap. The mixed EN+FR index behaves exactly like same-language retrieval, because e5 always ranks chunks in the query's language first.

### Generation and citation audit — TEST split (q27–q52, 52 answers per config; 95% CI clustered by question)
| Config | Uncited | Citation faithfulness [95% CI] | Faithful when cited | Grounded in any source [95% CI] | Lang match |
|---|---|---|---|---|---|
| A · 1.5B · same-lang · basic | 0.47 | 0.37 [0.24, 0.52] | 0.68 | 0.73 [0.61, 0.85] | 0.94 |
| B · 1.5B · same-lang · few-shot | 0.12 | 0.59 [0.44, 0.75] | 0.67 | 0.78 [0.66, 0.89] | 1.00 |
| H · 1.5B · same-lang · few-shot · post-hoc | 0.00 | **0.66** [0.50, 0.79] | 0.66 | 0.78 [0.66, 0.89] | 1.00 |
| D · 0.5B · same-lang · basic | 0.98 | 0.00 [0.00, 0.00] | – | 0.68 [0.53, 0.82] | 0.92 |
| E · 0.5B · same-lang · few-shot | 0.51 | 0.30 [0.19, 0.43] | 0.62 | 0.79 [0.69, 0.88] | 0.98 |
| F · 0.5B · mixed+rerank · few-shot | 0.65 | 0.20 [0.11, 0.33] | 0.58 | 0.74 [0.63, 0.85] | 0.98 |
| G · 0.5B · mixed+rerank · few-shot · post-hoc | 0.00 | 0.57 [0.47, 0.68] | 0.57 | 0.74 [0.63, 0.85] | 0.98 |

The DEV-split table is in `results/tables.md`. DEV values are similar, except that post-hoc on 1.5B looks larger there (H 0.74 vs B 0.57).

### Paired deltas in citation faithfulness (TEST; paired bootstrap over questions, 5,000 draws)
| Comparison | Δ | 95% CI | Reading |
|---|---|---|---|
| Few-shot vs basic, 1.5B (B−A) | +0.22 | [+0.08, +0.37] | better |
| Few-shot vs basic, 0.5B (E−D) | +0.30 | [+0.19, +0.44] | better |
| Mixed+rerank vs same-lang retrieval, 0.5B (F−E) | −0.10 | [−0.21, +0.01] | comparable (dev+test: −0.06 [−0.12, −0.01]) |
| Post-hoc attribution, 0.5B (G−F) | +0.37 | [+0.26, +0.48] | better |
| Post-hoc attribution, 1.5B (H−B) | +0.07 | [0.00, +0.17] | comparable (dev+test: +0.12 [+0.05, +0.19]) |
| 1.5B vs 0.5B, same setup (B−E) | +0.29 | [+0.18, +0.40] | better |
| 0.5B + post-hoc vs 1.5B few-shot (G−B) | −0.01 | [−0.15, +0.11] | comparable |
| 0.5B vs 1.5B, both post-hoc (G−H) | −0.08 | [−0.21, +0.04] | comparable (dev+test: −0.12 [−0.20, −0.04]) |

### Run-to-run determinism (dev+test, byte-identical answers)
| Rerun | dtype | Threads (ref → rerun) | Identical answers |
|---|---|---|---|
| D · 0.5B, same settings | fp32 | 4 → 4 | 104/104 (1.000) |
| D · 0.5B, different thread count | fp32 | 4 → 8 | 104/104 (1.000) |
| B · 1.5B, same settings | bf16 | 4 → 4 | 99/104 (0.952). Citation faithfulness is unchanged (0.577 dev+test, 0.586 test, same 130 claims). |
The old pipeline (bf16, thread count = `os.cpu_count()`) matched on only 61/104 answers in the reviewer's rerun. That drift (about 0.03 absolute) was as large as several ablation deltas. With fp32, the 0.5B answers are byte-identical, even across thread counts. The 1.5B bf16 run still varies on 5% of answers, but the metric doesn't move. Every delta above therefore reflects question-sampling uncertainty, and the CIs capture that.

### Abstention on out-of-corpus questions (TEST: 13 unanswerable × 2 languages vs 26 answerable × 2; τ tuned on DEV)
| Config | Mechanism | Precision | Recall | False-refusal rate | Unanswerable answered |
|---|---|---|---|---|---|
| A · 1.5B basic | LLM refusal | 0.67 | 0.85 | 0.21 | 4/26 |
| B · 1.5B few-shot | LLM refusal | 1.00 | 0.12 | 0.00 | 23/26 |
| E · 0.5B few-shot | LLM refusal | 0.00 | 0.00 | 0.00 | 26/26 |
| B / E (same-lang) | retrieval gate (e5 cosine < 0.886) | 0.68 | 1.00 | 0.23 | 0/26 |
| F / G (mixed+rerank) | retrieval gate (reranker logit < 2.27) | **0.77** | **1.00** | **0.15** | 0/26 |

### Judge calibration (53 items, all on DEV questions, LLM-labelled by the Developer agent, single annotator: indicative only)
| Subset @0.5, normalized | n | Accuracy [95% CI] | Recall [95% CI] | Specificity [95% CI] |
|---|---|---|---|---|
| Generated claims only | 21 | 0.90 [0.71, 0.97] | 1.00 [0.81, 1.00] | 0.60 [0.23, 0.88] |
| + perturbed hard negatives | 37 | 0.95 [0.82, 0.99] | 1.00 [0.81, 1.00] | 0.90 [0.71, 0.97] |
| + source-swap easy negatives (all) | 53 | 0.91 [0.80, 0.96] | 1.00 [0.81, 1.00] | 0.86 [0.72, 0.94] |

Raw (un-normalized) claims at 0.5 reach 0.89 accuracy over all 53 items, with recall 0.94. Thresholds of 0.7 and 0.9 trade recall for specificity (see `results/tables.md`). All 3 false positives on source-swap negatives are French.

### Takeaways
1. **Few-shot prompting is the biggest single lever** for both model sizes (+0.22 and +0.30 on TEST, both CIs exclude 0). With the basic prompt, the 0.5B model almost never cites.
2. **Post-hoc attribution is a large gain for 0.5B (+0.37)** and a smaller one for 1.5B (+0.07, comparable on TEST, significant on dev+test). It moves claims from "uncited" to "cited" but doesn't make them more faithful. Faithfulness when cited stays at about 0.57–0.68 in every config, so roughly a third of cited claims aren't entailed by their source.
3. **0.5B + post-hoc is comparable to 1.5B few-shot** (Δ −0.01 [−0.15, +0.11]). That isn't a like-for-like test. Like for like, 0.5B + post-hoc vs 1.5B + post-hoc is also comparable on TEST (−0.08 [−0.21, +0.04]), but lower on dev+test.
4. **Mixed-language retrieval with the reranker doesn't help generation.** It's comparable on TEST and slightly worse on dev+test.
5. **The models don't abstain reliably on their own.** The few-shot demo teaches them to always answer: 23–26 of 26 unanswerable test questions get an answer. A **retrieval-score gate** refuses all of them, at the cost of refusing 15% (reranker) to 23% (cosine) of answerable questions.
6. **Generation is now reproducible.** 0.5B fp32 answers are byte-identical (104/104 at 4 and at 8 threads). 1.5B bf16 matches on 99/104 answers with identical faithfulness.

## Error analysis (qualitative; all examples are from the DEV split / calibration run)
- **Wrong number, still "entailed":** "il faut compter au moins 365 jours de présence effective…" (gold: 1 095 days). The judge scored it 0.97 because a retrieved chunk mentions "365 days" for temporary-resident credit. NLI is known to be weak with numbers.
- **Entity confusion:** "persons with disabilities, their caregivers, and the CRA are eligible for the CDB…" mixes a navigation link into the answer. The normalized judge rejects it.
- **Fabricated detail inside a deflection:** "Pour obtenir ces détails, vous pouvez consulter le site web [Mesures temporaires…](http://www.mesures-temporelles-assuranceemploi.gouv.fr/)" has a made-up URL. It was relabelled unsupported on re-check, and the judge still accepts it (0.77), which is one of the 2 false positives among generated claims.
- **Judge false positives on unrelated French sources:** a correct ACE definition scored 0.85 entailment against passport text. Cross-lingual NLI seems less reliable in French.
- **Cross-lingual retrieval misses:** "RRSP options at 71" EN→FR retrieves GIS pages. The Express Entry "3 programs" sentence is split from its list. Short navigation-style chunks rank high without reranking.
- **Attribution prefixes:** answers starting "According to Source, …" were scored low even when correct. That's why claims are prefix-normalized before judging.


## Setup and reproduction
Python 3.13.5 (`.python-version`). The exact package set that produced `results/` is in `requirements.lock`. Model revisions are pinned in `src/config.py` and recorded in every results JSON (`env`).
```bash
uv venv --python 3.13.5 .venv
uv pip install --python .venv/bin/python -r requirements.lock --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
.venv/bin/python -m pytest -q tests                # parser, split, data and validation tests
OUT=data/corpus.refetch.json .venv/bin/python src/fetch_corpus.py   # optional: refetch; never overwrites data/corpus.json (sha256-checked)
.venv/bin/python src/retrieval.py                  # retrieval table
./run_grid.sh                                      # A B D E F G H + determinism reruns, audits, calibration, deltas, abstention, tables (resumable)
./run_offline.sh                                   # air-gapped: retrieval + F + post-hoc G + audit; checks offline == online answers
.venv/bin/python src/ask.py "How long do EI sickness benefits last?"     # one question; --lang en|fr optional
```
Configuration is read from environment variables, and invalid values are rejected: `LLM_MODEL`, `RETR` (`same` or `mixed_rr`), `PROMPT` (`basic` or `fewshot`), `DTYPE`, `THREADS`, `QA`, `IN`/`OUT`, and `NORMALIZE`. Results from the previous iteration, which used the old parser and nondeterministic decoding, are kept for reference in `results/legacy_445992f/`.

**Air-gapped run:** `run_offline.sh` runs inside `unshare -rn`, a network namespace with only loopback. It requires `curl` to fail with "could not resolve" or "could not connect" (rc 6/7) and aborts on any other outcome. Then, with `HF_HUB_OFFLINE=1`, it runs retrieval, config F, post-hoc attribution and the audit, and compares the offline answers byte-for-byte with the committed online run (`results/offline_run.log`). **Last run (PR #10): offline F answers were identical to the online run on 104/104, and offline G citation faithfulness was 0.577 [0.50, 0.66], equal to the online G value on dev+test.**

## Limitations
- The corpus is small (20 page pairs) and some pages are mostly navigation. The Q&A set, splits, gold labels, unanswerable set and calibration labels all come from one author, the Developer agent. The test split is disjoint by construction but wasn't written independently.
- The judge calibration has 53 items, all on dev questions and labelled by an LLM agent. NLI is weak on numbers ("365 jours" vs a gold value of 1 095 is scored as entailed). A human-labelled sample is the next step.
- Faithfulness metrics don't measure answer correctness or completeness against a gold answer.
- Prompt-injection hardening is limited to delimiting sources and adding a "treat as data" instruction. It hasn't been evaluated beyond the reviewer's 2-probe check.

## Next steps
1. A human-labelled calibration sample (100+ claims, against the cited chunk, with a second annotator and κ).
2. A numeric-consistency check in the judge: extract numbers and compare them with the cited chunk.
3. Combine the retrieval gate with an answerability classifier to cut false refusals. Add a prompt-injection eval set.
4. Score answer correctness against gold answers, and use structure-aware chunking that keeps a heading with its list.

## Data and licence
Source pages: canada.ca under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada). No hosted APIs are used at any stage.
