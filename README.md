# Provenance-First "Verifiable" RAG Assistant

> Work in progress — being built phase by phase. Retrieval, generation, entailment
> verification, and the Streamlit demo are done; evaluation numbers (`results/metrics.json`)
> and a revised abstention benchmark are still to come.

**Live demo:** _add your Streamlit Community Cloud URL here once deployed_

## What this is

A retrieval-augmented QA system over an 84-article space-exploration corpus (NASA, ESA,
Roscosmos, ISRO, CNSA, JAXA, and more — see [`corpus/SOURCES.md`](corpus/SOURCES.md)) where
every sentence of every generated answer is traced to the exact source chunk it came from,
scored by a local NLI model for whether that chunk actually entails the sentence, and the
system explicitly abstains when the corpus doesn't support an answer instead of
hallucinating one.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env           # then fill in an API key -- see below
streamlit run app.py
```

The FAISS index is pre-built and committed under `corpus/index/`, so no build step is
required before the first run. To rebuild it from scratch (e.g. after editing the corpus):

```bash
python -m src.ingestion
```

### API key

`.env.example` documents two options:

- `LLM_PROVIDER=anthropic` — paid, highest quality (get a key at
  [console.anthropic.com](https://console.anthropic.com/settings/keys))
- `LLM_PROVIDER=openrouter` — **free**, no payment method required (get a key at
  [openrouter.ai/keys](https://openrouter.ai/keys)). Note: not every `:free` OpenRouter model
  supports the forced tool-calling this project relies on for structured, cited output --
  `nvidia/nemotron-3-super-120b-a12b:free` is confirmed to work.

## Architecture

- `src/ingestion.py` — load, chunk, embed, index the corpus
- `src/retrieval.py` — dense + BM25 hybrid retrieval (Reciprocal Rank Fusion)
- `src/generation.py` — LLM call producing answers with per-sentence chunk-id citations;
  abstains when unsupported
- `src/verification.py` — local NLI cross-encoder scores each sentence against its cited
  chunk; flags unsupported claims and hallucinated citation ids
- `app.py` — Streamlit demo with sentence-level color-coded grounding
- `scripts/fetch_wikipedia_corpus.py` — reproducible corpus fetcher

## Tests

```bash
pytest tests/ -v
```

## Known gaps (tracked, not yet done)

- `eval/qa_set.jsonl`'s unanswerable-question subset predates the corpus expansion to 84
  articles and needs revision -- see the "Deliberately excluded" section of
  `corpus/SOURCES.md`.
- `src/evaluate.py` and `results/metrics.json` (MRR/NDCG/Recall@k, groundedness
  distribution, abstention precision/recall) haven't been built yet.
