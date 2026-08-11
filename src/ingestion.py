"""Load the corpus, chunk it, embed the chunks, and build a persisted vector index.

Corpus choice
-------------
The corpus is ~84 English Wikipedia articles about space exploration broadly:
national and commercial space agencies, crewed and uncrewed missions,
spacecraft/hardware, and key figures, spanning the US, Soviet/Russian,
European, Indian, Chinese, and Japanese programs (see ``corpus/SOURCES.md``
for the full list, licensing/attribution, and the re-runnable fetch script
at ``scripts/fetch_wikipedia_corpus.py``). A single coherent domain (space
exploration, not "everything") was chosen deliberately: it gives a dense
factual record (dates, names, distances, outcomes) that can be verified
sentence-by-sentence, and it makes "unanswerable" questions constructible in
a principled way (real questions about adjacent topics this corpus does not
cover -- see ``corpus/SOURCES.md``'s "Deliberately excluded" section) rather
than arbitrary nonsense questions, which is what makes the abstention
benchmark in ``eval/qa_set.jsonl`` meaningful.

Chunking strategy
------------------
Fixed-size sliding window over whitespace-tokenized words (default 180 words,
40-word overlap), rather than semantic/embedding-based segmentation. This is a
deliberate simplicity trade-off for a project whose novelty is the
verification layer, not the chunker:

- Deterministic and trivially testable (no model in the loop, no
  nondeterminism from a segmentation model's boundary decisions).
- No extra model to load, keeping the ingestion path CPU-and-free-tier
  friendly.
- The word-count overlap means a fact that would otherwise be split across a
  chunk boundary has a good chance of appearing whole in at least one chunk,
  which is the main failure mode fixed-size chunking without overlap has.

The trade-off this accepts: a chunk boundary can land mid-sentence. That is
fine here because every citation in this project points at a *chunk id*, not
a sentence id — the verification layer (Phase 4) checks whether the retrieved
chunk's text entails the generated sentence, and a chunk that starts/ends
mid-sentence still contains recoverable factual content either side of the
cut.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import CORPUS_DIR, INDEX_DIR, get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE_WORDS = 180
DEFAULT_OVERLAP_WORDS = 40

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_METADATA_PATH = INDEX_DIR / "chunks.json"


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of text.

    Attributes:
        chunk_id: Globally unique id, ``"{doc_id}::chunk{n:04d}"``. This is
            the identifier the LLM cites in Phase 3 and that the verification
            layer checks against in Phase 4.
        doc_id: Source document id (filename stem under ``corpus/``).
        text: The chunk's raw text.
    """

    chunk_id: str
    doc_id: str
    text: str


class IngestionError(Exception):
    """Raised when the corpus cannot be loaded or the index cannot be built/persisted."""


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> dict[str, str]:
    """Load every ``.txt`` document in ``corpus_dir``.

    Args:
        corpus_dir: Directory containing one ``.txt`` file per source document.
            Non-``.txt`` files (e.g. ``SOURCES.md``) are ignored.

    Returns:
        Mapping of ``doc_id`` (filename stem) to the document's raw text.

    Raises:
        IngestionError: If the directory doesn't exist or contains no ``.txt``
            files, or if a file can't be read/decoded.
    """
    if not corpus_dir.is_dir():
        raise IngestionError(f"Corpus directory not found: {corpus_dir}")

    txt_paths = sorted(corpus_dir.glob("*.txt"))
    if not txt_paths:
        raise IngestionError(f"No .txt documents found in {corpus_dir}")

    documents: dict[str, str] = {}
    for path in txt_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise IngestionError(f"Failed to read corpus file {path}: {exc}") from exc
        if not text.strip():
            logger.warning("Corpus file %s is empty, skipping", path.name)
            continue
        documents[path.stem] = text

    logger.info("Loaded %d documents from %s", len(documents), corpus_dir)
    return documents


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split ``text`` into overlapping fixed-size word chunks.

    Args:
        text: Raw document text.
        chunk_size: Number of whitespace-tokenized words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        List of chunk strings, in order. Empty input returns an empty list.

    Raises:
        ValueError: If ``overlap >= chunk_size`` or either is non-positive,
            which would cause zero or negative forward progress.
    """
    if chunk_size <= 0 or overlap < 0:
        raise ValueError(f"chunk_size must be > 0 and overlap >= 0, got {chunk_size=}, {overlap=}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def build_chunks(
    documents: dict[str, str],
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Chunk every document and assign each chunk a globally unique id.

    Args:
        documents: Mapping of ``doc_id`` to raw text, as returned by
            :func:`load_corpus`.
        chunk_size: Passed through to :func:`chunk_text`.
        overlap: Passed through to :func:`chunk_text`.

    Returns:
        Flat list of :class:`Chunk` across all documents.
    """
    all_chunks: list[Chunk] = []
    for doc_id, text in documents.items():
        pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunk_id = f"{doc_id}::chunk{i:04d}"
            all_chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=piece))
        logger.info("Chunked %s into %d chunks", doc_id, len(pieces))

    logger.info("Built %d total chunks from %d documents", len(all_chunks), len(documents))
    return all_chunks


def embed_chunks(chunks: list[Chunk], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Embed chunk texts with a small local sentence-transformer model.

    Args:
        chunks: Chunks to embed.
        model_name: HuggingFace model id. Defaults to a ~90MB CPU-friendly
            model, well under the project's ~500MB local-model budget.

    Returns:
        Float32 array of shape ``(len(chunks), embedding_dim)``, L2-normalized
        so that inner product equals cosine similarity.

    Raises:
        IngestionError: If ``chunks`` is empty or the model fails to load.
    """
    if not chunks:
        raise IngestionError("Cannot embed an empty chunk list")

    try:
        model = SentenceTransformer(model_name)
    except (OSError, ValueError) as exc:
        raise IngestionError(f"Failed to load embedding model {model_name}: {exc}") from exc

    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    logger.info("Embedded %d chunks into vectors of dim %d", *embeddings.shape)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an exact flat inner-product FAISS index.

    An exact ``IndexFlatIP`` (rather than an approximate index) is used
    because the corpus is small (low thousands of chunks at most): exact
    search is fast enough and removes a whole class of recall bugs that
    approximate indexes can introduce.

    Args:
        embeddings: L2-normalized float32 array of shape ``(n, dim)``.

    Returns:
        A populated FAISS index.

    Raises:
        IngestionError: If ``embeddings`` is empty or not 2-dimensional.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise IngestionError(f"Expected a non-empty 2D embeddings array, got shape {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS IndexFlatIP with %d vectors of dim %d", index.ntotal, dim)
    return index


def save_index(index: faiss.Index, chunks: list[Chunk], index_dir: Path = INDEX_DIR) -> None:
    """Persist the FAISS index and chunk metadata to disk.

    Args:
        index: The FAISS index to serialize.
        chunks: Chunk metadata, saved in the same order the embeddings were
            added to ``index`` so that FAISS row ``i`` maps to ``chunks[i]``.
        index_dir: Directory to write ``faiss.index`` and ``chunks.json`` into.

    Raises:
        IngestionError: If the directory can't be created or files can't be written.
    """
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / "faiss.index"))
        with (index_dir / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
    except (OSError, ValueError) as exc:
        raise IngestionError(f"Failed to save index to {index_dir}: {exc}") from exc

    logger.info("Saved FAISS index and %d chunks to %s", len(chunks), index_dir)


def load_index(index_dir: Path = INDEX_DIR) -> tuple[faiss.Index, list[Chunk]]:
    """Load a previously persisted FAISS index and its chunk metadata.

    Args:
        index_dir: Directory containing ``faiss.index`` and ``chunks.json``.

    Returns:
        Tuple of ``(index, chunks)`` where ``chunks[i]`` corresponds to row
        ``i`` of ``index``.

    Raises:
        IngestionError: If either file is missing or fails to parse.
    """
    index_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        raise IngestionError(
            f"Index not found in {index_dir}. Run `python -m src.ingestion` to build it first."
        )

    try:
        index = faiss.read_index(str(index_path))
        with chunks_path.open("r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        chunks = [Chunk(**c) for c in raw_chunks]
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        raise IngestionError(f"Failed to load index from {index_dir}: {exc}") from exc

    logger.info("Loaded FAISS index (%d vectors) and %d chunks from %s", index.ntotal, len(chunks), index_dir)
    return index, chunks


def build_and_save_index(
    corpus_dir: Path = CORPUS_DIR,
    index_dir: Path = INDEX_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> None:
    """Run the full ingestion pipeline: load, chunk, embed, index, persist.

    Args:
        corpus_dir: Directory of source ``.txt`` documents.
        index_dir: Directory to persist the FAISS index and chunk metadata to.
        chunk_size: Words per chunk.
        overlap: Overlap words between consecutive chunks.
    """
    documents = load_corpus(corpus_dir)
    chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    save_index(index, chunks, index_dir)


if __name__ == "__main__":
    build_and_save_index()
