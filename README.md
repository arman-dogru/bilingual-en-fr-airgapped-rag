# Bilingual EN/FR air-gapped RAG with citation audit

An offline retrieval-augmented QA system over open Government of Canada pages in English and French. Once the models and corpus are downloaded, it runs fully air-gapped on CPU. It measures EN↔FR cross-lingual retrieval and how faithful the answer citations are. The project is aimed at sovereign or air-gapped LLM deployment, as in Cohere North.

See **[REPORT.md](REPORT.md)** for results.

## Pipeline
| Step | Script | Output |
|---|---|---|
| Fetch 20 EN/FR Canada.ca pairs (OGL-Canada) | `src/fetch_corpus.py` | `data/corpus.json` |
| Chunk, embed (multilingual-e5-small) and recall@k in 4 directions | `src/retrieval.py` | `results/retrieval.json` |
| RAG with inline `[n]` citations (local Qwen2.5) | `src/rag.py` | `results/answers_*.json` |
| Citation audit (mDeBERTa XNLI) and language match | `src/audit.py` | `results/audit_*.json` |
| Full run with networking disabled | `run_offline.sh` | `results/offline_run.log` |

## Setup (online, once)
```bash
uv venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python sentence-transformers transformers beautifulsoup4 requests numpy
.venv/bin/python src/fetch_corpus.py
.venv/bin/python src/retrieval.py                       # downloads embedder
.venv/bin/python src/rag.py                             # downloads LLM
.venv/bin/python src/audit.py                           # downloads NLI judge
```
Environment variables: `LLM_MODEL` (default `Qwen/Qwen2.5-0.5B-Instruct`), `EMB_MODEL`, `NLI_MODEL`, and `OUT`/`IN` for result paths.

## Air-gapped run
```bash
./run_offline.sh   # unshare -rn: network namespace with only loopback; HF_HUB_OFFLINE=1
```

## Data and licence
The source pages come from canada.ca under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada). Each document's EN and FR URLs are recorded in `data/corpus.json`. No hosted APIs are used.
