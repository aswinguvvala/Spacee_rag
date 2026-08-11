# CLAUDE.md

## Project
Provenance-First Verifiable RAG Assistant — a portfolio project proving RAG answers are trustworthy, not just fluent. Every generated sentence is traced to its exact source chunk, scored for entailment via a local NLI/cross-encoder model, and the system explicitly abstains when the corpus doesn't support an answer. Primary audience: technical recruiters and hiring managers reviewing the GitHub repo and the live demo.

## Stack
- Python 3.11
- sentence-transformers (embeddings), FAISS or Chroma (vector index)
- A small local NLI/cross-encoder model for entailment scoring (e.g. `cross-encoder/nli-deberta-v3-xsmall`)
- An external LLM API for generation only (provider swappable behind one interface, default Anthropic)
- Streamlit for `app.py`
- pytest for `tests/`

## Commands
- Run app: `streamlit run app.py`
- Run tests: `pytest tests/ -v`
- Build/refresh index: `python -m src.ingestion`
- Run full eval: `python -m src.evaluate`
- Lint: `ruff check src/`
- Install deps: `pip install -r requirements.txt`

## Conventions
- Type hints on every function signature
- Google-style docstrings on all public functions/classes
- `logging` module, never `print`, in `src/`
- try/except with specific exception types and actionable messages around all I/O and API calls
- Every local model choice must stay CPU-friendly and small enough for a free hosting tier (no local model over ~500MB) — the only thing allowed to call an external service is the LLM generation step
- Single-responsibility modules: `ingestion.py` never does generation; `verification.py` never does retrieval
- New features get a test in `tests/` before being considered done, especially anything touching `verification.py`

## Architecture
- `src/ingestion.py` — load, chunk, embed, index the corpus
- `src/retrieval.py` — dense (+ optional hybrid) retrieval
- `src/generation.py` — LLM call producing answers with per-sentence chunk-id citations; must abstain when unsupported
- `src/verification.py` — entailment scoring per sentence against its cited chunk; flags unsupported claims and hallucinated citation IDs
- `src/evaluate.py` — MRR/NDCG/Recall@k with bootstrap CI, groundedness distribution, abstention precision/recall
- `app.py` — Streamlit demo with sentence-level color-coded grounding; must run standalone from a fresh clone with only an API key set

## Do not
- Don't use copyrighted or non-permissively-licensed documents in `corpus/`
- Don't skip the unanswerable-question subset in `eval/qa_set.jsonl` — the abstention benchmark is the point of this project, not optional
- Don't hardcode API keys — read from `.env`, never commit it
- Don't add a dependency without pinning its version in `requirements.txt`

## Definition of done
Code runs without errors, has tests (especially for entailment flagging and abstention), has type hints and docstrings, logs meaningfully instead of printing, and `results/metrics.json` contains real numbers the README can quote.