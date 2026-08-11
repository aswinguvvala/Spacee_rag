# Provenance-First "Verifiable" RAG Assistant

> Work in progress — being built phase by phase. This README is a placeholder until
> Phase 8, when it will carry the full problem statement, architecture diagram, real
> evaluation numbers, and run-locally instructions.

## What this is

A retrieval-augmented QA system over a small space-exploration corpus (NASA and other
national/commercial space programs) where every sentence of every generated answer is
traced to the exact source chunk it came from, scored for whether that chunk actually
entails the sentence, and the system explicitly abstains when the corpus doesn't support
an answer instead of hallucinating one.

## Quickstart (dev, in progress)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env           # then fill in ANTHROPIC_API_KEY
```
