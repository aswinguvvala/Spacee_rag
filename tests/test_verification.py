"""Tests for src/verification.py: entailment flagging and citation-hallucination detection."""
from __future__ import annotations

import pytest

from src.generation import GeneratedAnswer, SentenceCitation
from src.ingestion import Chunk
from src.verification import (
    SUPPORTED_THRESHOLD,
    WEAK_THRESHOLD,
    EntailmentLabel,
    SentenceVerification,
    Verifier,
    VerifiedAnswer,
    _label_for_score,
    _sentence_windows,
    verify_answer,
)

# ---- Fixtures / fakes ------------------------------------------------------

CHUNK_A = Chunk(chunk_id="doc::chunk0000", doc_id="doc", text="Jim Lovell commanded Apollo 13.")
CHUNK_B = Chunk(chunk_id="doc::chunk0001", doc_id="doc", text="Apollo 13 launched on April 11, 1970.")
CHUNKS_BY_ID = {CHUNK_A.chunk_id: CHUNK_A, CHUNK_B.chunk_id: CHUNK_B}


def fake_get_chunk_by_id(chunk_id: str) -> Chunk | None:
    return CHUNKS_BY_ID.get(chunk_id)


class FakeVerifier:
    """Deterministic stand-in for Verifier -- no model, fixed scores per premise text."""

    def __init__(self, score_by_premise: dict[str, float], default: float = 0.0) -> None:
        self._scores = score_by_premise
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.extend(pairs)
        return [self._scores.get(premise, self._default) for premise, _ in pairs]


def make_answer(sentences: list[tuple[str, list[str]]], abstained: bool = False) -> GeneratedAnswer:
    return GeneratedAnswer(
        abstained=abstained,
        abstain_reason="no sources" if abstained else None,
        sentences=[SentenceCitation(text=t, chunk_ids=cids) for t, cids in sentences],
    )


# ---- _label_for_score --------------------------------------------------

def test_label_for_score_none_is_unsupported():
    assert _label_for_score(None) == EntailmentLabel.UNSUPPORTED


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, EntailmentLabel.UNSUPPORTED),
        (0.19, EntailmentLabel.UNSUPPORTED),
        (0.2, EntailmentLabel.WEAK),
        (0.49, EntailmentLabel.WEAK),
        (0.5, EntailmentLabel.SUPPORTED),
        (0.99, EntailmentLabel.SUPPORTED),
    ],
)
def test_label_for_score_thresholds(score, expected):
    assert _label_for_score(score) == expected


# ---- _sentence_windows -----------------------------------------------

def test_sentence_windows_single_sentence_returns_just_itself():
    text = "Jim Lovell commanded Apollo 13."
    assert _sentence_windows(text) == [text]


def test_sentence_windows_includes_individual_and_combined_windows():
    text = "One. Two. Three."
    windows = _sentence_windows(text, max_window=2)
    assert "One." in windows
    assert "Two." in windows
    assert "Three." in windows
    assert "One. Two." in windows
    assert "Two. Three." in windows
    # Never a window spanning more than max_window consecutive sentences
    # (other than the full-text safety net itself).
    assert "One. Two. Three." in windows  # the full text is always included


def test_sentence_windows_caps_window_size():
    text = "One. Two. Three. Four. Five."
    windows = _sentence_windows(text, max_window=2)
    assert "One. Two. Three." not in windows  # a 3-sentence window, over the cap
    assert "One. Two." in windows


# ---- verify_answer: abstention short-circuit ----------------------------

def test_verify_answer_abstained_never_loads_a_model():
    answer = make_answer([], abstained=True)
    # Passing verifier=None on an abstained answer must not try to construct
    # a real Verifier (which would load a model) -- if it did, this call
    # would be slow/network-dependent instead of instant.
    result = verify_answer(answer, fake_get_chunk_by_id, verifier=None)
    assert isinstance(result, VerifiedAnswer)
    assert result.abstained is True
    assert result.abstain_reason == "no sources"
    assert result.sentences == []


# ---- verify_answer: hallucinated citations -------------------------------

def test_verify_answer_flags_hallucinated_citation():
    answer = make_answer([("Apollo 13 splashed down safely.", ["doc::chunk9999"])])
    fake = FakeVerifier({})
    result = verify_answer(answer, fake_get_chunk_by_id, verifier=fake)

    assert len(result.sentences) == 1
    sv = result.sentences[0]
    assert sv.hallucinated_chunk_ids == ["doc::chunk9999"]
    assert sv.entailment_score is None
    assert sv.label == EntailmentLabel.UNSUPPORTED
    assert result.has_hallucinated_citation is True
    # A hallucinated-only citation never reaches the model.
    assert fake.calls == []


def test_verify_answer_no_citations_is_unsupported_without_model_call():
    answer = make_answer([("A claim with no citation at all.", [])])
    fake = FakeVerifier({})
    result = verify_answer(answer, fake_get_chunk_by_id, verifier=fake)

    assert result.sentences[0].label == EntailmentLabel.UNSUPPORTED
    assert result.sentences[0].entailment_score is None
    assert fake.calls == []


# ---- verify_answer: supported / unsupported / multi-citation --------------

def test_verify_answer_supported_sentence():
    answer = make_answer([("Jim Lovell commanded Apollo 13.", [CHUNK_A.chunk_id])])
    fake = FakeVerifier({CHUNK_A.text: 0.95})
    result = verify_answer(answer, fake_get_chunk_by_id, verifier=fake)

    sv = result.sentences[0]
    assert sv.label == EntailmentLabel.SUPPORTED
    assert sv.entailment_score == 0.95
    assert sv.hallucinated_chunk_ids == []


def test_verify_answer_unsupported_sentence():
    answer = make_answer([("The Moon is made of cheese.", [CHUNK_A.chunk_id])])
    fake = FakeVerifier({CHUNK_A.text: 0.01})
    result = verify_answer(answer, fake_get_chunk_by_id, verifier=fake)

    sv = result.sentences[0]
    assert sv.label == EntailmentLabel.UNSUPPORTED
    assert sv.entailment_score == 0.01


def test_verify_answer_takes_best_score_across_multiple_citations():
    answer = make_answer(
        [
            (
                "Apollo 13 was commanded by Jim Lovell and launched in April 1970.",
                [CHUNK_A.chunk_id, CHUNK_B.chunk_id],
            ),
        ]
    )
    fake = FakeVerifier({CHUNK_A.text: 0.1, CHUNK_B.text: 0.9})
    result = verify_answer(answer, fake_get_chunk_by_id, verifier=fake)

    assert result.sentences[0].entailment_score == 0.9
    assert result.sentences[0].label == EntailmentLabel.SUPPORTED
    # Both cited chunks should have been scored, not just the first.
    assert len(fake.calls) == 2


def test_verify_answer_partial_hallucination_still_scores_valid_chunk():
    answer = make_answer([("Jim Lovell commanded Apollo 13.", [CHUNK_A.chunk_id, "doc::chunk9999"])])
    fake = FakeVerifier({CHUNK_A.text: 0.8})
    result = verify_answer(answer, fake_get_chunk_by_id, verifier=fake)

    sv = result.sentences[0]
    assert sv.hallucinated_chunk_ids == ["doc::chunk9999"]
    assert sv.entailment_score == 0.8
    assert sv.label == EntailmentLabel.SUPPORTED


# ---- VerifiedAnswer properties --------------------------------------------

def test_all_supported_property():
    supported = VerifiedAnswer(
        abstained=False,
        abstain_reason=None,
        sentences=[SentenceVerification("a", ["x"], 0.9, EntailmentLabel.SUPPORTED)],
    )
    assert supported.all_supported is True

    mixed = VerifiedAnswer(
        abstained=False,
        abstain_reason=None,
        sentences=[
            SentenceVerification("a", ["x"], 0.9, EntailmentLabel.SUPPORTED),
            SentenceVerification("b", ["y"], 0.1, EntailmentLabel.UNSUPPORTED),
        ],
    )
    assert mixed.all_supported is False


def test_all_supported_vacuously_true_when_abstained():
    abstained = VerifiedAnswer(abstained=True, abstain_reason="nothing found", sentences=[])
    assert abstained.all_supported is True


# ---- Verifier: real model integration (exercises the actual NLI call) -----


@pytest.fixture(scope="module")
def real_verifier() -> Verifier:
    return Verifier()


def test_real_verifier_scores_clear_entailment_high(real_verifier):
    score = real_verifier.score_many(
        [("Jim Lovell commanded the Apollo 13 mission.", "Apollo 13 was commanded by Jim Lovell.")]
    )[0]
    assert score >= SUPPORTED_THRESHOLD


def test_real_verifier_scores_unrelated_sentence_low(real_verifier):
    score = real_verifier.score_many(
        [
            (
                "Jim Lovell commanded the Apollo 13 mission.",
                "The Great Wall of China is visible from space.",
            )
        ]
    )[0]
    assert score < WEAK_THRESHOLD


def test_real_verifier_empty_pairs_returns_empty_list(real_verifier):
    assert real_verifier.score_many([]) == []


def test_real_verifier_end_to_end_through_verify_answer(real_verifier):
    answer = make_answer([("Jim Lovell commanded Apollo 13.", [CHUNK_A.chunk_id])])
    result = verify_answer(answer, fake_get_chunk_by_id, verifier=real_verifier)
    assert result.sentences[0].label == EntailmentLabel.SUPPORTED


def test_real_verifier_supported_fact_survives_unrelated_trailing_sentences(real_verifier):
    """Regression test for a real failure found via the Streamlit demo.

    A multi-sentence chunk (real ``chandrayaan_3::chunk0000`` text) fully
    supports this compound sentence in its middle three sentences. Scoring
    the *whole* chunk as one premise crashed to ~4% entailment because of two
    unrelated trailing sentences about the lander's post-mission fate --
    even though the supporting content was untouched. Windowed scoring
    (``_sentence_windows``) must find the supporting window and score this
    as SUPPORTED, not UNSUPPORTED.
    """
    chunk_text = (
        "Chandrayaan-3 is the third mission in the Chandrayaan programme, a series of "
        "lunar-exploration missions developed by ISRO. The mission consists of Vikram, a "
        "lunar lander, and Pragyan, a lunar rover, as replacements for the equivalents on "
        "Chandrayaan-2, which had crashed on landing in 2019. Chandrayaan-3 was launched on "
        "14 July 2023, at 14:35 IST from the Satish Dhawan Space Centre (SDSC) in "
        "Sriharikota, India. It entered lunar orbit on 5 August, and touched down near the "
        "lunar south pole, at 69 degrees S, on 23 August 2023 at 18:04 IST (12:33 UTC). With "
        "this landing, ISRO became the fourth national space agency to successfully land on "
        "the Moon, after the Soviet space program, NASA and CNSA, and the first in human "
        "history to achieve a soft landing near the lunar south pole. The lander was not "
        "built to withstand the cold temperatures of the lunar night, so it was shut down at "
        "sunset over the landing site, twelve days after landing. The orbiting propulsion "
        "module remained operational and was repurposed for scientific observations of Earth."
    )
    chunk = Chunk(chunk_id="chandrayaan_3::chunk0000", doc_id="chandrayaan_3", text=chunk_text)
    sentence = (
        "It was launched on 14 July 2023 from the Satish Dhawan Space Centre, entered lunar "
        "orbit on 5 August, and touched down near the lunar south pole on 23 August 2023, "
        "making India the fourth national space agency to achieve a soft landing on the Moon "
        "and the first to do so near the south pole."
    )
    answer = make_answer([(sentence, [chunk.chunk_id])])
    result = verify_answer(
        answer, lambda cid: chunk if cid == chunk.chunk_id else None, verifier=real_verifier
    )

    sv = result.sentences[0]
    assert sv.label == EntailmentLabel.SUPPORTED, (
        f"expected SUPPORTED, got {sv.label} (score={sv.entailment_score})"
    )
