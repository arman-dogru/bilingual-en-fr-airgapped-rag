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

RESULTS_PLACEHOLDER

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

**Air-gapped run:** `run_offline.sh` runs inside `unshare -rn`, a network namespace with only loopback. It requires `curl` to fail with "could not resolve" or "could not connect" (rc 6/7) and aborts on any other outcome. Then, with `HF_HUB_OFFLINE=1`, it runs retrieval, config F, post-hoc attribution and the audit, and compares the offline answers byte-for-byte with the committed online run (`results/offline_run.log`).

## Limitations
- The corpus is small (20 page pairs) and some pages are mostly navigation. The Q&A set, splits, gold labels, unanswerable set and calibration labels all come from one author, the Developer agent. The test split is disjoint by construction but wasn't written independently.
- The judge calibration has 53 items, all on dev questions and labelled by an LLM agent. NLI is weak on numbers ("365 jours" vs a gold value of 1 095 is scored as entailed). A human-labelled sample is the next step.
- Faithfulness metrics don't measure answer correctness or completeness against a gold answer.
- Prompt-injection hardening is limited to delimiting sources and adding a "treat as data" instruction. It hasn't been evaluated beyond the reviewer's 2-probe check.

## Data and licence
Source pages: canada.ca under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada). No hosted APIs are used at any stage.
