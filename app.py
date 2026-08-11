"""Streamlit demo: ask a question, see a cited answer with sentence-level grounding.

Every sentence the model writes is shown with the source chunk(s) it cited and
the local NLI model's entailment score against that chunk -- color-coded so a
viewer can see at a glance which claims are actually backed by the corpus,
which are weakly supported, and which the system got wrong (or made up a
citation for) without having to read the source text themselves.

Must run standalone from a fresh clone with only an API key set: the FAISS
index is pre-built and committed under ``corpus/index/`` (see
``src/ingestion.py``), so no build step is required first.

UI notes
--------
Streamlit's default theme is left mostly unstyled by design tools, so this
module injects its own CSS (targeting Streamlit's ``data-testid`` attributes,
which are documented as the stable styling hook across versions -- internal
``st-emotion-cache-*`` class names are not). The hero background is
``assets/earthrise.jpg`` -- see ``assets/ATTRIBUTION.md`` for sourcing/license.
"""
from __future__ import annotations

import base64
import html
import os
from pathlib import Path

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

ASSETS_DIR = Path(__file__).parent / "assets"
HERO_IMAGE_PATH = ASSETS_DIR / "earthrise.jpg"

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

LABEL_META: dict[EntailmentLabel, dict[str, str]] = {
    EntailmentLabel.SUPPORTED: {"word": "Supported", "css": "supported"},
    EntailmentLabel.WEAK: {"word": "Weak", "css": "weak"},
    EntailmentLabel.UNSUPPORTED: {"word": "Unsupported", "css": "unsupported"},
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


@st.cache_resource(show_spinner=False)
def load_hero_image_b64() -> str | None:
    """Read and base64-encode the hero image once per server process.

    Embedding as a data URI (rather than relying on Streamlit's static-file
    serving) keeps the background reliable across both local runs and
    Streamlit Cloud deploys.
    """
    try:
        data = HERO_IMAGE_PATH.read_bytes()
    except OSError as exc:
        logger.warning("Hero image not found at %s: %s", HERO_IMAGE_PATH, exc)
        return None
    return base64.b64encode(data).decode("ascii")


def inject_css() -> None:
    """Inject the site's custom theme, overriding Streamlit's default look.

    Uses ``st.markdown(..., unsafe_allow_html=True)`` -- the long-standing,
    well-proven way to inject a global ``<style>`` block in Streamlit.
    (``st.html`` was tried instead and, empirically, did not reliably land
    the block in the document at all -- reverted in favor of what's actually
    proven to work.) The hero background image is embedded here, as a CSS
    class rule with the base64 data URI baked in, rather than as a per-element
    inline ``style=`` attribute -- embedding that same data URI as an inline
    attribute on the hero ``<div>`` silently dropped the whole attribute,
    apparently from how Streamlit's Markdown-to-HTML pass handles a single
    very long attribute value. A rule inside a ``<style>`` block didn't have
    that problem.
    """
    hero_image_b64 = load_hero_image_b64()
    hero_bg = (
        f"url('data:image/jpeg;base64,{hero_image_b64}')" if hero_image_b64 else "none"
    )
    # A plain (non-f) string: the CSS below is full of literal `{`/`}` rule
    # blocks, which an f-string would try to interpret as expressions. The
    # one dynamic value (the hero background) is substituted afterward via a
    # placeholder token instead.
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #090b11;
    --bg-panel: #12151f;
    --bg-panel-raised: #171b28;
    --border: #262b3a;
    --text: #e7eaf2;
    --text-muted: #8992a8;
    --accent: #6fd3d9;
    --accent-bg: rgba(111, 211, 217, 0.10);
    --supported: #5cc98a;
    --supported-bg: rgba(92, 201, 138, 0.13);
    --weak: #e3b34f;
    --weak-bg: rgba(227, 179, 79, 0.13);
    --unsupported: #e3726c;
    --unsupported-bg: rgba(227, 114, 108, 0.13);
}

html, body, [data-testid="stApp"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Hide Streamlit's default chrome -- the multicolor top bar, hamburger menu,
   and toolbar are the biggest tells that this is a stock Streamlit app. */
[data-testid="stHeader"] { background: transparent; height: 2.25rem; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMainMenu"] { display: none; }
[data-testid="stToolbar"] { visibility: hidden; }
footer { visibility: hidden; height: 0; }

/* Main content column: narrower, centered, like an article -- not Streamlit's
   default edge-to-edge wide layout. */
[data-testid="stMainBlockContainer"] {
    max-width: 900px;
    padding-top: 0;
    padding-bottom: 3rem;
}

h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
    letter-spacing: -0.01em;
}

/* ---- Hero banner ----
   Contained within the normal content column (not full-bleed): Streamlit
   nests the block container inside several wrapper divs it controls, and
   the classic `left:50%; margin-left:-50vw` full-bleed trick depends on
   assumptions about the containing block that don't hold there -- it ended
   up mispositioned and painted behind a sibling. A normal in-flow block
   with rounded corners is simpler and robust regardless of Streamlit's
   internal DOM. */
.hero {
    width: 100%;
    height: 380px;
    margin-top: 1rem;
    margin-bottom: 2.25rem;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--border);
    /* Left-to-right scrim so the text sits on a readable dark panel while
       Earth stays visible, uncovered, on the right -- rather than a flat
       overlay dimming the whole photo (which was barely visible before). */
    background-image:
        linear-gradient(100deg, var(--bg) 0%, rgba(9,11,17,0.86) 32%, rgba(9,11,17,0.25) 62%, rgba(9,11,17,0) 85%),
        linear-gradient(180deg, transparent 75%, var(--bg) 100%),
        __HERO_BG__;
    background-size: cover, cover, cover;
    background-position: 0 0, 0 0, 78% 42%;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    text-align: left;
    padding: 0 2.75rem;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.9rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.5rem;
    line-height: 1.15;
    color: #ffffff;
    margin: 0;
    max-width: 540px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #cbd2e0;
    max-width: 420px;
    margin: 0.9rem 0 0 0;
    line-height: 1.6;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.55;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    font-weight: 600;
    margin-top: 1.6rem;
    margin-bottom: 0.6rem;
}
[data-testid="stSidebar"] code {
    background: var(--bg-panel-raised);
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82em;
}

/* ---- Buttons (example-question chips + Ask) ---- */
.stButton > button, .stFormSubmitButton > button {
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: border-color 0.15s ease, background 0.15s ease;
}
button[data-testid="stBaseButton-secondary"] {
    background: var(--bg-panel-raised);
    border: 1px solid var(--border);
    color: var(--text-muted);
    text-align: left;
    font-size: 0.85rem;
    padding: 0.5rem 0.8rem;
}
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--accent);
    color: var(--text);
}
button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: var(--accent);
    border: 1px solid var(--accent);
    color: #06232a;
    font-weight: 600;
}
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background: #85dce1;
}

/* ---- Text input ---- */
[data-testid="stTextInput"] input {
    background: var(--bg-panel-raised);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
}
[data-testid="stTextInput"] label p {
    font-size: 0.82rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stForm"] {
    border: 1px solid var(--border);
    background: var(--bg-panel);
    border-radius: 12px;
    padding: 1.4rem 1.4rem 1.1rem 1.4rem;
}

/* ---- Verdict cards (replaces raw colored <div> blocks) ---- */
.verdict-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 10px;
    padding: 0.95rem 1.15rem;
    margin: 0.55rem 0;
}
.verdict-card.supported { border-left-color: var(--supported); }
.verdict-card.weak { border-left-color: var(--weak); }
.verdict-card.unsupported { border-left-color: var(--unsupported); }
.verdict-text { font-size: 0.98rem; line-height: 1.55; color: var(--text); }
.verdict-meta {
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem;
    margin-top: 0.65rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
}
.verdict-badge {
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
    padding: 0.15rem 0.55rem; border-radius: 999px;
}
.verdict-badge.supported { background: var(--supported-bg); color: var(--supported); }
.verdict-badge.weak { background: var(--weak-bg); color: var(--weak); }
.verdict-badge.unsupported { background: var(--unsupported-bg); color: var(--unsupported); }
.verdict-source { color: var(--text-muted); }
.verdict-halluc { color: var(--unsupported); }

/* ---- Alerts (st.error / st.warning / st.info) ---- */
[data-testid="stAlert"] {
    background: var(--bg-panel-raised);
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* ---- Expander (retrieved source chunks) ---- */
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--bg-panel);
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-muted);
}

/* ---- Footer / caption ---- */
[data-testid="stCaptionContainer"] {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
}
.site-footer {
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.8rem;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
}
.site-footer a { color: var(--accent); text-decoration: none; }
.site-footer a:hover { text-decoration: underline; }
</style>
"""
    st.markdown(css.replace("__HERO_BG__", hero_bg), unsafe_allow_html=True)


def render_hero() -> None:
    """Render the full-bleed hero banner. Its background image is set via a CSS
    class rule in :func:`inject_css`, not an inline style here -- see that
    function's docstring for why.
    """
    st.markdown(
        """
<div class="hero">
    <div class="hero-eyebrow">Provenance-first &middot; retrieval-augmented QA</div>
    <h1 class="hero-title">Ask the space-exploration corpus.<br>See exactly what backs each claim.</h1>
    <p class="hero-subtitle">Every sentence is traced to its source chunk and scored for entailment by a
    local NLI model, not just trusted from the language model's own citation. When the corpus doesn't
    support an answer, it abstains instead of guessing.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sentence_html(sv: SentenceVerification) -> str:
    """Render one verified sentence as a color-coded card.

    All dynamic text is HTML-escaped before embedding, since sentence text
    (LLM output) and chunk ids (from the corpus) are not trusted to be free
    of characters that would otherwise break the markup.
    """
    meta = LABEL_META[sv.label]
    score_txt = f"{sv.entailment_score:.0%} entailment" if sv.entailment_score is not None else "n/a"
    cited = ", ".join(html.escape(c) for c in sv.chunk_ids) if sv.chunk_ids else "none"
    halluc_txt = ""
    if sv.hallucinated_chunk_ids:
        bad_ids = ", ".join(html.escape(c) for c in sv.hallucinated_chunk_ids)
        halluc_txt = f'<span class="verdict-halluc">hallucinated citation: {bad_ids}</span>'
    text = html.escape(sv.text)

    return (
        f'<div class="verdict-card {meta["css"]}">'
        f'<div class="verdict-text">{text}</div>'
        f'<div class="verdict-meta">'
        f'<span class="verdict-badge {meta["css"]}">{meta["word"]}</span>'
        f'<span class="verdict-source">{cited}</span>'
        f'<span class="verdict-source">{score_txt}</span>'
        f"{halluc_txt}"
        f"</div></div>"
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
        st.caption(f"{n_supported}/{n_total} sentence(s) fully grounded (≥ 50% entailment)")

    with st.expander(f"Retrieved source chunks ({len(results)})"):
        for r in results:
            preview = r.chunk.text[:400] + ("…" if len(r.chunk.text) > 400 else "")
            st.markdown(f"**`{r.chunk.chunk_id}`** &nbsp; score={r.score:.4f}")
            st.text(preview)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "A retrieval-augmented QA system over an 84-article space-exploration corpus "
            "(NASA, ESA, Roscosmos, ISRO, CNSA, JAXA, and more — see `corpus/SOURCES.md`), "
            "all CC BY-SA 4.0 Wikipedia text."
        )

        st.markdown("### Legend")
        st.markdown(
            '<div style="display:flex; gap:0.4rem; flex-wrap:wrap; margin-bottom:0.5rem;">'
            '<span class="verdict-badge supported">Supported</span>'
            '<span class="verdict-badge weak">Weak</span>'
            '<span class="verdict-badge unsupported">Unsupported</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Try an answerable question")
        for q in EXAMPLE_QUESTIONS_ANSWERABLE:
            if st.button(q, key=f"ex_a_{q}", use_container_width=True):
                st.session_state["question_input"] = q
                st.session_state["auto_submit"] = True

        st.markdown("### Try one it should abstain on")
        for q in EXAMPLE_QUESTIONS_UNANSWERABLE:
            if st.button(q, key=f"ex_u_{q}", use_container_width=True):
                st.session_state["question_input"] = q
                st.session_state["auto_submit"] = True


def render_footer() -> None:
    st.markdown(
        '<div class="site-footer">'
        "<span>Earthrise, Apollo 8 (NASA, public domain) — see assets/ATTRIBUTION.md</span>"
        '<a href="https://github.com/aswinguvvala/Spacee_rag" target="_blank">Source on GitHub →</a>'
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Provenance-First RAG", page_icon="\U0001f6f0️", layout="wide")
    inject_css()
    render_hero()
    render_sidebar()

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

    render_footer()


if __name__ == "__main__":
    main()
