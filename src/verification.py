"""Per-sentence entailment scoring against cited source chunks.

Grounding, not fluency
----------------------
Generation (``src/generation.py``) produces sentences with citations, but an
LLM can still cite a real chunk_id for a sentence that chunk doesn't actually
support -- or invent a chunk_id that was never retrieved at all. This module
is the check: for every generated sentence, does its cited chunk *entail* it,
scored by a local NLI cross-encoder rather than trusted from the LLM's own
citation.

Two independent failure modes are checked:

1. Hallucinated citation: the cited chunk_id doesn't exist in this index at
   all. Checked with a plain lookup (``get_chunk_by_id``), not a model call --
   an id either came from retrieval or it didn't.
2. Unsupported claim: the chunk_id is real, but the NLI model scores the
   chunk as not entailing the sentence (probability below
   :data:`SUPPORTED_THRESHOLD`).

When a sentence cites more than one chunk, each cited chunk is scored against
the sentence individually and the *best* (highest-entailment) score wins,
rather than concatenating all cited chunks into one premise. This is a
simplicity trade-off: it can undercount sentences that only hold true when
two chunks' facts are combined, but it keeps every score attributable to one
concrete source chunk, which is what the UI highlights.

Whole chunks are not scored as a single premise, either -- see
:func:`_sentence_windows`. A ~180-word chunk almost always contains sentences
that have nothing to do with a given generated sentence, and empirically this
small NLI model is not robust to that: unrelated trailing content in the
premise can crash an otherwise-correct entailment score (observed directly --
a chunk whose first five sentences fully support a compound generated
sentence scored ~90% entailment; the exact same premise with two more,
unrelated, trailing sentences appended scored ~4%, even though the
supporting content was untouched). Scoring sliding windows of a few
consecutive sentences instead, and taking the best window, keeps the model
focused on whichever part of the chunk is actually relevant.

Model choice
------------
``cross-encoder/nli-deberta-v3-xsmall`` (~140MB) -- small enough for the
project's CPU-friendly, free-tier-hosting budget (well under the ~500MB
local-model limit) and purpose-trained for premise/hypothesis entailment,
unlike embedding cosine similarity, which conflates "similar topic" with
"actually implies."
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from sentence_transformers import CrossEncoder

from src.generation import GeneratedAnswer
from src.ingestion import Chunk
from src.utils import get_logger

logger = get_logger(__name__)

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"

# Sliding-window premise construction (see module docstring for why whole
# chunks aren't scored directly). Windows of 1..MAX_PREMISE_WINDOW_SENTENCES
# consecutive sentences are scored and the best wins; the full chunk text is
# always included too, as a safety net for facts that span a wider window.
MAX_PREMISE_WINDOW_SENTENCES = 4
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# P(entailment) thresholds. A sentence at or above SUPPORTED_THRESHOLD is
# considered grounded; below WEAK_THRESHOLD it's treated the same as an
# unsupported claim. Chosen empirically as a reasonable middle ground -- see
# tests/test_verification.py for the calibration cases these were checked
# against.
SUPPORTED_THRESHOLD = 0.5
WEAK_THRESHOLD = 0.2


class VerificationError(Exception):
    """Raised when the NLI model can't be loaded or scoring fails."""


class EntailmentLabel(str, Enum):
    """Verdict for one generated sentence against its cited source(s)."""

    SUPPORTED = "supported"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SentenceVerification:
    """Verification result for one generated sentence.

    Attributes:
        text: The generated sentence.
        chunk_ids: chunk_ids the sentence cited, as generated (unfiltered --
            includes any hallucinated ids).
        entailment_score: Highest P(entailment) across all cited chunks that
            actually exist in the index. ``None`` if no cited chunk_id
            resolved to a real chunk (e.g. all were hallucinated, or the
            sentence cited nothing).
        label: Overall verdict for this sentence.
        hallucinated_chunk_ids: Cited chunk_ids that don't exist in the index.
    """

    text: str
    chunk_ids: list[str]
    entailment_score: float | None
    label: EntailmentLabel
    hallucinated_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VerifiedAnswer:
    """A generated answer with per-sentence entailment verification applied.

    Attributes:
        abstained: Mirrors :class:`~src.generation.GeneratedAnswer.abstained`.
        abstain_reason: Mirrors the generation-time abstain reason.
        sentences: Per-sentence verification results. Empty when ``abstained``.
    """

    abstained: bool
    abstain_reason: str | None
    sentences: list[SentenceVerification] = field(default_factory=list)

    @property
    def all_supported(self) -> bool:
        """True if every sentence is SUPPORTED (vacuously True if abstained)."""
        return all(s.label == EntailmentLabel.SUPPORTED for s in self.sentences)

    @property
    def has_hallucinated_citation(self) -> bool:
        """True if any sentence cited a chunk_id that doesn't exist in the index."""
        return any(s.hallucinated_chunk_ids for s in self.sentences)


class Verifier:
    """Loads a local NLI cross-encoder once and scores (premise, hypothesis) pairs."""

    def __init__(self, model_name: str = NLI_MODEL_NAME) -> None:
        """Load the cross-encoder and resolve which output index is "entailment".

        The entailment class index is read from the model's own
        ``config.id2label`` rather than assumed, since different NLI
        checkpoints order their labels differently.

        Args:
            model_name: HuggingFace model id for a 3-class NLI cross-encoder.

        Raises:
            VerificationError: If the model fails to load, or its label set
                doesn't contain exactly one entailment-like class.
        """
        try:
            self._model = CrossEncoder(model_name)
        except (OSError, ValueError) as exc:
            raise VerificationError(f"Failed to load NLI model {model_name!r}: {exc}") from exc

        id2label = {i: label.lower() for i, label in self._model.model.config.id2label.items()}
        entailment_indices = [i for i, label in id2label.items() if "entail" in label]
        if len(entailment_indices) != 1:
            raise VerificationError(
                f"Could not uniquely identify the entailment class from model labels: {id2label}"
            )
        self._entailment_index = entailment_indices[0]
        logger.info(
            "Loaded NLI model %s (labels=%s, entailment index=%d)",
            model_name, id2label, self._entailment_index,
        )

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return P(entailment) for each ``(premise, hypothesis)`` pair, batched.

        Args:
            pairs: List of ``(premise, hypothesis)`` string tuples.

        Returns:
            P(entailment) for each pair, same order as ``pairs``. Empty list
            if ``pairs`` is empty (no model call is made).

        Raises:
            VerificationError: If scoring fails.
        """
        if not pairs:
            return []
        try:
            probs = self._model.predict(pairs, apply_softmax=True, convert_to_numpy=True)
        except (RuntimeError, ValueError) as exc:
            raise VerificationError(f"NLI scoring failed for {len(pairs)} pair(s): {exc}") from exc

        return [float(row[self._entailment_index]) for row in probs]


def _sentence_windows(text: str, max_window: int = MAX_PREMISE_WINDOW_SENTENCES) -> list[str]:
    """Split ``text`` into sentences and return every window of 1..``max_window``
    consecutive sentences, plus the full text, as candidate NLI premises.

    Args:
        text: Chunk text to window.
        max_window: Largest number of consecutive sentences per window.

    Returns:
        Deduplicated candidate premises, in no particular order. Always
        includes ``text`` itself even if sentence-splitting fails to find
        more than one sentence.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    windows: set[str] = {text}
    for size in range(1, min(max_window, len(sentences)) + 1):
        for start in range(0, len(sentences) - size + 1):
            windows.add(" ".join(sentences[start : start + size]))
    return list(windows)


def _label_for_score(score: float | None) -> EntailmentLabel:
    """Map a P(entailment) score (or ``None``) to a verdict."""
    if score is None:
        return EntailmentLabel.UNSUPPORTED
    if score >= SUPPORTED_THRESHOLD:
        return EntailmentLabel.SUPPORTED
    if score >= WEAK_THRESHOLD:
        return EntailmentLabel.WEAK
    return EntailmentLabel.UNSUPPORTED


def verify_answer(
    answer: GeneratedAnswer,
    get_chunk_by_id: Callable[[str], Chunk | None],
    verifier: Verifier | None = None,
) -> VerifiedAnswer:
    """Score every sentence of ``answer`` for entailment against its citations.

    Args:
        answer: The generated answer to verify.
        get_chunk_by_id: Looks up a chunk by id, e.g.
            :meth:`~src.retrieval.Retriever.get_chunk_by_id`. Returning
            ``None`` for an id marks it hallucinated.
        verifier: A :class:`Verifier` to score with. A new one is constructed
            (loading the NLI model) if not given -- pass one in explicitly to
            reuse a single loaded model across many calls.

    Returns:
        A :class:`VerifiedAnswer` mirroring ``answer`` with a verdict and
        score attached to each sentence. If ``answer.abstained`` is True, no
        model is loaded or called -- the abstention is passed through as-is.
    """
    if answer.abstained:
        return VerifiedAnswer(abstained=True, abstain_reason=answer.abstain_reason, sentences=[])

    verifier = verifier or Verifier()

    results: list[SentenceVerification] = []
    for sentence in answer.sentences:
        valid_chunks: list[Chunk] = []
        hallucinated: list[str] = []
        for cid in sentence.chunk_ids:
            chunk = get_chunk_by_id(cid)
            if chunk is None:
                hallucinated.append(cid)
            else:
                valid_chunks.append(chunk)

        if hallucinated:
            logger.warning(
                "Hallucinated citation id(s) %s in sentence %r", hallucinated, sentence.text
            )

        if not valid_chunks:
            results.append(
                SentenceVerification(
                    text=sentence.text,
                    chunk_ids=sentence.chunk_ids,
                    entailment_score=None,
                    label=EntailmentLabel.UNSUPPORTED,
                    hallucinated_chunk_ids=hallucinated,
                )
            )
            continue

        pairs = [
            (window, sentence.text)
            for chunk in valid_chunks
            for window in _sentence_windows(chunk.text)
        ]
        scores = verifier.score_many(pairs)
        best_score = max(scores)
        label = _label_for_score(best_score)
        if label != EntailmentLabel.SUPPORTED:
            logger.info("Sentence flagged %s (score=%.3f): %r", label.value, best_score, sentence.text)

        results.append(
            SentenceVerification(
                text=sentence.text,
                chunk_ids=sentence.chunk_ids,
                entailment_score=best_score,
                label=label,
                hallucinated_chunk_ids=hallucinated,
            )
        )

    return VerifiedAnswer(abstained=False, abstain_reason=None, sentences=results)
