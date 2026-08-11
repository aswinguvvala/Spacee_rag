"""Streamlit demo: ask a question, see a cited answer with sentence-level grounding.

Every sentence the model writes is shown with the source chunk(s) it cited and
the local NLI model's entailment score against that chunk -- green/amber/red
so a viewer can see at a glance which claims are actually backed by the
corpus, which are weakly supported, and which the system got wrong (or made
up a citation for) without having to read the source text themselves.

Must run standalone from a fresh clone with only an API key set: the FAISS
index is pre-built and committed under ``corpus/index/`` (see
``src/ingestion.py``), so no build step is required first.
"""
from __future__ import annotations

import html
import os

import streamlit as st
from dotenv import load_dotenv

from src.generation import GenerationError, generate_answer
from src.retrieval import RetrievalError, Retriever
from src.utils import get_logger
from src.verification import (
    EntailmentLabel,
    SentenceVerification,
    VerificationError,
    Verifier,
    verify_answer,
)

load_dotenv()
logger = get_logger(__name__)

EXAMPLE_QUESTIONS_ANSWERABLE = [
    "Who commanded Apollo 13?",
    "What is ISRO's Chandrayaan-3 mission?",
    "What happened during the Space Shuttle Challenger disaster?",
    "What is the James Webb Space Telescope designed to observe?",
    "Describe the Soviet Sputnik 1 satellite.",
]
EXAMPLE_QUESTIONS_UNANSWERABLE = [
    "How does the SETI Institute search for extraterrestrial signals?",
    "What are the current plans for asteroid mining companies?",
]

LABEL_STYLE: dict[EntailmentLabel, dict[str, str]] = {
    EntailmentLabel.SUPPORTED: {"bg": "#d9f2e3", "border": "#1e7e42", "text": "#0f3d21", "icon": "✅"},
    EntailmentLabel.WEAK: {"bg": "#fdf1cf", "border": "#a66b00", "text": "#4d3800", "icon": "⚠️"},
    EntailmentLabel.UNSUPPORTED: {"bg": "#fbdfe0", "border": "#a3222c", "text": "#4a0f13", "icon": "❌"},
}


def _api_key_configured() -> bool:
    """Check whether the configured LLM provider has a real (non-placeholder) key set."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    key_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENROUTER_API_KEY"
    key = os.environ.get(key_var, "")
    return bool(key) and key != "your-api-key-here"


@st.cache_resource(show_spinner="Loading retrieval index (first run only)...")
def load_retriever() -> Retriever:
    """Load the FAISS index + embedding model once per server process."""
    return Retriever()


@st.cache_resource(show_spinner="Loading entailment model (first run only)...")
def load_verifier() -> Verifier:
    """Load the local NLI cross-encoder once per server process."""
    return Verifier()


def render_sentence_html(sv: SentenceVerification) -> str:
    """Render one verified sentence as a color-coded HTML block.

    All dynamic text is HTML-escaped before embedding, since sentence text
    (LLM output) and chunk ids (from the corpus) are not trusted to be free
    of characters that would otherwise break the markup.
    """
    style = LABEL_STYLE[sv.label]
    score_txt = f"{sv.entailment_score:.0%}" if sv.entailment_score is not None else "n/a"
    cited = ", ".join(html.escape(c) for c in sv.chunk_ids) if sv.chunk_ids else "none"
    halluc_txt = ""
    if sv.hallucinated_chunk_ids:
        bad_ids = ", ".join(html.escape(c) for c in sv.hallucinated_chunk_ids)
        halluc_txt = f' &middot; <b>hallucinated citation:</b> {bad_ids}'
    text = html.escape(sv.text)

    return (
        f'<div style="background:{style["bg"]}; border-left:4px solid {style["border"]}; '
        f'color:{style["text"]}; padding:10px 14px; margin:8px 0; border-radius:6px;">'
        f'<div style="font-size:1rem;">{style["icon"]} {text}</div>'
        f'<div style="font-size:0.78rem; opacity:0.85; margin-top:4px;">'
        f'source: {cited} &middot; entailment: {score_txt}{halluc_txt}'
        f'</div></div>'
    )


def render_answer(question: str) -> None:
    """Run retrieve -> generate -> verify for ``question`` and render the result."""
    try:
        retriever = load_retriever()
    except RetrievalError as exc:
        st.error(f"Couldn't load the retrieval index: {exc}")
        return

    with st.spinner("Retrieving relevant source chunks..."):
        try:
            results = retriever.retrieve(question, k=5)
        except RetrievalError as exc:
            st.error(f"Retrieval failed: {exc}")
            return

    with st.spinner("Generating a cited answer..."):
        try:
            answer = generate_answer(question, results)
        except GenerationError as exc:
            st.error(f"Generation failed: {exc}")
            st.info(
                "Check that your API key is set correctly -- in `.env` locally, or in this "
                "app's **Secrets** if it's deployed."
            )
            return

    if answer.abstained:
        st.warning(f"**Abstained — the corpus doesn't support an answer.**\n\n{answer.abstain_reason}")
    else:
        try:
            verifier = load_verifier()
        except VerificationError as exc:
            st.error(f"Couldn't load the entailment model: {exc}")
            return

        with st.spinner("Scoring each sentence's grounding..."):
            try:
                verified = verify_answer(answer, retriever.get_chunk_by_id, verifier=verifier)
            except VerificationError as exc:
                st.error(f"Verification failed: {exc}")
                return

        for sv in verified.sentences:
            st.markdown(render_sentence_html(sv), unsafe_allow_html=True)

        n_supported = sum(1 for s in verified.sentences if s.label == EntailmentLabel.SUPPORTED)
        n_total = len(verified.sentences)
        if verified.has_hallucinated_citation:
            st.error("This answer cited at least one chunk id that doesn't exist in the index.")
        st.caption(f"{n_supported}/{n_total} sentence(s) fully grounded (≥ 50% entailment).")

    with st.expander(f"Retrieved source chunks ({len(results)})"):
        for r in results:
            preview = r.chunk.text[:400] + ("…" if len(r.chunk.text) > 400 else "")
            st.markdown(f"**`{r.chunk.chunk_id}`** &nbsp; score={r.score:.4f}")
            st.text(preview)


def main() -> None:
    st.set_page_config(page_title="Provenance-First RAG", page_icon="\U0001f6f0️", layout="wide")

    st.title("\U0001f6f0️ Provenance-First Verifiable RAG")
    st.caption(
        "Every sentence is traced to its source chunk and scored for entailment by a local "
        "NLI model — not just trusted from the LLM's own citation. The system abstains "
        "when the corpus doesn't support an answer instead of guessing."
    )

    with st.sidebar:
        st.header("About")
        st.markdown(
            "A retrieval-augmented QA system over an 84-article space-exploration corpus "
            "(NASA, ESA, Roscosmos, ISRO, CNSA, JAXA, and more — see `corpus/SOURCES.md`), "
            "all CC BY-SA 4.0 Wikipedia text."
        )
        st.markdown("**Legend**")
        for label in EntailmentLabel:
            style = LABEL_STYLE[label]
            st.markdown(
                f'<span style="background:{style["bg"]}; color:{style["text"]}; '
                f'border-left:3px solid {style["border"]}; padding:2px 8px; border-radius:4px;">'
                f'{style["icon"]} {label.value}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Try an answerable question**")
        for q in EXAMPLE_QUESTIONS_ANSWERABLE:
            if st.button(q, key=f"ex_a_{q}", use_container_width=True):
                st.session_state["question_input"] = q
                st.session_state["auto_submit"] = True

        st.markdown("**Try one it should abstain on**")
        for q in EXAMPLE_QUESTIONS_UNANSWERABLE:
            if st.button(q, key=f"ex_u_{q}", use_container_width=True):
                st.session_state["question_input"] = q
                st.session_state["auto_submit"] = True

    if not _api_key_configured():
        provider = os.environ.get("LLM_PROVIDER", "anthropic")
        st.error(
            f"No API key configured for `LLM_PROVIDER={provider}`. Set it in `.env` locally "
            "(see `.env.example`), or in this app's **Secrets** if deployed. A free OpenRouter "
            "key (`LLM_PROVIDER=openrouter`, no card required) works too."
        )
        st.stop()

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    with st.form("ask_form"):
        question = st.text_input(
            "Ask a question about space exploration",
            key="question_input",
            placeholder="e.g. Who commanded Apollo 13?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    auto_submit = st.session_state.pop("auto_submit", False)
    if (submitted or auto_submit) and question.strip():
        render_answer(question.strip())


if __name__ == "__main__":
    main()
