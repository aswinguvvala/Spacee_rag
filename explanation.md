# Build log: what was done, and every real problem hit along the way

This documents the working session that took this project from "ingestion/retrieval/
generation modules exist, nothing else does" to a deployed, styled, working demo. It's
written as a challenge log on purpose — the interesting part of building this wasn't
writing the happy-path code, it was the five or six times something looked done and
turned out not to be.

## Starting point

When this session started, `src/ingestion.py`, `src/retrieval.py`, and `src/generation.py`
existed and worked. `src/verification.py`, `app.py`, and `tests/` did not exist. The corpus
was 8 Wikipedia articles about the Apollo program only. There was no `.env`, no API key, no
deployment.

---

## 1. Getting a free LLM working

**Goal:** run the generation pipeline without paying for API access.

**The problem:** `.env.example` already documented an OpenRouter option
(`LLM_PROVIDER=openrouter`) for free-tier models. The obvious default,
`openai/gpt-oss-20b:free`, failed immediately:

```
GenerationError: OpenRouter API error (400): "inference-enforced tool_choice
(required/named) is not supported for model \"gpt-oss-20b\""
```

This project's whole citation mechanism depends on *forced* tool-calling (the model must
call `provide_grounded_answer`, not just reply with text) — so a model that can't do that
is a non-starter, not a minor inconvenience.

**How it was found:** rather than guess at another model name, I queried OpenRouter's live
`/models` endpoint for the actual current list of `:free` models (most model names I'd have
guessed from training data were already retired or renamed), then wrote a small script that
tried forced tool-calling against each candidate directly and printed pass/fail.

**Fix:** `nvidia/nemotron-3-super-120b-a12b:free` — the largest free model that supported
forced tool-calling — became the default in `.env`.

**Lesson:** for anything API-shaped, check the provider's live catalog instead of trusting
memorized model names — free-tier offerings churn constantly, and the failure mode here
(400 error on a specific parameter) doesn't look like "model doesn't exist," it looks like a
config bug, which costs time to rule out.

---

## 2. Expanding the corpus (Apollo-only → 84 space-exploration articles)

**Goal:** broaden the corpus from just Apollo to international space agencies, historic
programs, spacecraft, and missions, per the user's request.

**The problem:** a `scripts/fetch_wikipedia_corpus.py` script fetching ~76 new articles from
Wikipedia's API hit a hard wall after 19 requests:

```
429 Client Error: Too Many Requests
```

**Fix:** added exponential backoff that respects the `Retry-After` header, and slowed the
base request rate. Re-running the script was idempotent by design (it skips any article
whose output file already exists), so the retry didn't need to restart from zero.

**Smaller issue caught in the same pass:** the original corpus's "unanswerable question"
eval set (`eval/qa_set.jsonl`) relied on specific topics (Artemis, the Shuttle, the Soviet
Luna program) being *absent* from the corpus so those questions would correctly get
abstentions. Expanding the corpus to include those exact topics quietly invalidates that
part of the eval set — flagged in `corpus/SOURCES.md`'s "Deliberately excluded" section and
in the README, but not yet fixed (still open).

---

## 3. The verification.py bug (the one that mattered most)

This is the one worth reading in full, because it wasn't caught by a test — it was caught by
actually looking at the live demo's output and not trusting a "looks plausible" result.

**Setup:** `src/verification.py` scores each generated sentence against its cited source
chunk using a local NLI model (`cross-encoder/nli-deberta-v3-xsmall`), producing a
P(entailment) score. The first working version scored the *entire* ~180-word chunk as one
premise against the sentence.

**What happened:** testing the live app with "What is ISRO's Chandrayaan-3 mission?", a
sentence that was **obviously correct** — restating facts that were, verbatim, in the cited
chunk — got flagged:

```
❌ It was launched on 14 July 2023 from the Satish Dhawan Space Centre, entered lunar
orbit on 5 August, and touched down near the lunar south pole on 23 August 2023...
entailment: 4%
```

A 4% score on a sentence that's almost a direct paraphrase of its source is a red flag for
the verification layer *itself* being broken — which matters more than a normal bug, since
"trustworthy verification" is this whole project's premise.

**Diagnosis, step by step:**
1. Manually re-ran the same premise/hypothesis pair outside the app: got **90%**, not 4%.
   That discrepancy — same inputs, different outputs — meant something about *how* the
   real pipeline called the model differed from my manual test, not that the model was
   simply bad at this example.
2. Reproduced it exactly through the real `Retriever.get_chunk_by_id()` object instead of
   a hand-retyped string, to rule out a copy-paste discrepancy. Score dropped back to 4%.
   So the *exact* chunk object mattered — meaning it wasn't a transcription error, it was
   something about the full chunk text specifically.
3. Bisected the premise sentence-by-sentence (scored the hypothesis against the chunk's
   first 1, 2, 3, 4, 5, 6, 7 sentences, in order):

   | Premise = first N sentences | Score |
   |---|---|
   | 1–4 sentences | ~0.0005 |
   | **5 sentences** | **0.898** |
   | 6 sentences | 0.008 |
   | 7 (all) sentences | 0.045 |

   Sentence 5 is where the last fact the hypothesis needed (ISRO becoming the "fourth
   national space agency") first appears — hence the jump to 90% at exactly 5 sentences.
   Sentences 6–7 are about the lander's post-mission fate, completely unrelated to the
   hypothesis — and *adding them back in crashed the score from 90% back down to 0.8–4.5%*,
   even though the actually-relevant sentences 3–5 were still sitting right there, unchanged.

**Root cause:** this specific small NLI model (trained mostly on single-sentence
premise/hypothesis pairs from SNLI/MNLI) is not robust to a premise that's a full paragraph
containing both the relevant fact and unrelated surrounding content. It's not that longer
premises are bad in general — it's that *irrelevant* content in the premise actively
confuses this checkpoint's entailment judgment, non-monotonically.

**Fix:** instead of scoring the whole chunk as one premise, score sliding windows of 1–4
consecutive sentences from the chunk (plus the full chunk as a safety net for facts spanning
a wider window), and take the best-scoring window. This let the model score against
"sentences 3–5 only" without the distracting sentences 6–7, without needing to know in
advance which window would be the right one.

**Verified the fix:** re-ran the same case — **99.2%**, correctly SUPPORTED. Locked in with
a regression test (`test_real_verifier_supported_fact_survives_unrelated_trailing_sentences`
in `tests/test_verification.py`) that uses the *real* chunk text and the *real* model, not a
mock, specifically so this can't silently regress.

**Lesson:** a demo that "looks like it's working" (renders, doesn't crash, produces
plausible-looking colors) is a different bar than "is actually correct." The bug here
produced a *confident, wrong* answer — a red badge on a true sentence, which for a project
literally about verification is the single worst failure mode it could have. It only
surfaced because I read the actual number instead of just checking that some badge, any
badge, appeared.

---

## 4. Deployment, round 1: `torch==2.5.1` had no wheels

**The problem:** first Streamlit Community Cloud deploy failed immediately:

```
ERROR: Could not find a version that satisfies the requirement torch==2.5.1
(from versions: 2.9.0, 2.9.1, 2.10.0, 2.11.0, 2.12.0, 2.12.1, 2.13.0)
```

`requirements.txt`'s pins (`torch==2.5.1`, `transformers==4.46.3`,
`sentence-transformers==3.3.1`, `faiss-cpu==1.9.0.post1`, `numpy==1.26.4`) were current when
this project was first scaffolded, but Streamlit Cloud's build environment simply doesn't
have wheels for versions that old anymore.

**Fix, and how it was validated (not just guessed):** rather than bump one version and hope,
I built a fresh local venv with an updated, mutually-compatible set of pins
(`sentence-transformers==5.7.0`, `transformers==5.15.0`, `torch==2.9.1`,
`faiss-cpu==1.15.0`, `numpy==2.5.2`), then actually ran the full test suite and a lint pass
against it locally before pushing — including specifically checking that the *already
committed* FAISS index (built under the old `faiss-cpu`) still loads correctly under the new
one, since index serialization formats can change across major versions and that would have
been a much nastier silent-corruption bug than an install failure.

---

## 5. Deployment, round 2: a *different* failure after the first fix

**The problem:** the ML stack fix worked — but the next deploy attempt failed at a
different, unrelated package:

```
error: the configured Python interpreter version (3.14) is newer than PyO3's
maximum supported version (3.13)
Failed building wheel for pydantic-core
```

**Diagnosis:** Streamlit Cloud's build environment turned out to be running **Python
3.14** — new enough that `pydantic==2.10.3` (which pins `pydantic-core==2.27.1` exactly)
had no prebuilt wheel for it, so pip fell back to compiling it from Rust source, which
failed outright because the Rust↔Python bindings library pinned by that version doesn't
support 3.14 yet.

**Fix:** researched actual current PyPI release metadata (not memorized version numbers,
since pydantic ships very frequently) to confirm `pydantic==2.13.4` pins
`pydantic-core==2.46.4`, which does ship `cp314` wheels. Bumped it, then re-validated the
*entire* `requirements.txt` end-to-end in a fresh venv again — full install, full test
suite, lint — before pushing.

**Lesson (both round 1 and round 2):** "the deploy failed" is not one bug, it's a queue of
them — fixing the first blocking error just reveals the next one. Re-validating the whole
dependency set locally after each fix, rather than pushing single-line guesses and waiting
on the next Cloud build to find out, turned what could have been four or five round-trips
into two.

---

## 6. UI redesign: three separate rendering bugs in one afternoon

The ask was to make the Streamlit app look like a designed website instead of a default
Streamlit app: real typography, a dark space theme, and a real NASA photo (the "Earthrise"
image, Apollo 8, public domain) as a hero background. The visual design itself came together
quickly; getting the image to actually *render* took three separate, unrelated fixes.

**Bug 1 — `st.html()` didn't reliably render.** First attempt used Streamlit's `st.html()`
API (added specifically for raw-HTML injection, seemingly the "correct" modern tool for
this). In practice, content injected this way didn't reliably land in the DOM at all in this
Streamlit version — confirmed by inspecting the live page's DOM directly via a browser
automation tool rather than guessing from the visual result. Reverted to the older, far more
battle-tested `st.markdown(html_string, unsafe_allow_html=True)` pattern everywhere, which
had already been working correctly for everything except the image.

**Bug 2 — a long inline `style=""` attribute got silently dropped.** The hero background
image was base64-encoded (~145KB → ~193KB of base64 text) and set via an inline
`style="--hero-image: url('data:image/jpeg;base64,...')"` attribute on the hero `<div>`.
Inspecting the rendered DOM showed the element existed but had **no `style` attribute at
all** — not malformed, just entirely absent. A *short* inline style on a different element
elsewhere on the same page rendered fine, which ruled out "inline styles are stripped" as
too broad an explanation and pointed at something specific to attribute *length*.

Fix: instead of setting the image per-element via an inline attribute, the base64 data was
baked directly into a `.hero { background-image: ...; }` rule inside the page's main
`<style>` block, which had already proven it could carry several KB of CSS correctly.
(Doing this required a small mechanical workaround: the CSS block has hundreds of literal
`{`/`}` characters, which collide with Python f-string syntax, so the image data was
substituted in afterward via a placeholder token + `str.replace()` rather than an f-string.)

**Bug 3 — the classic full-bleed CSS trick fought Streamlit's own DOM.** The hero was meant
to span the full browser width (bleeding past Streamlit's centered content column) using the
standard `position: relative; left: 50%; margin-left: -50vw; width: 100vw;` trick. Once the
image itself was fixed (bug 2), it *still* didn't show — DOM inspection this time showed the
hero `<div>` was sized and positioned correctly, but a *different* element was being painted
on top of it at the same screen coordinates, and the div wasn't actually flush with the true
viewport edge either. Streamlit wraps app content in several of its own nested layout divs
that this trick's assumptions (about the containing block being simply centered) don't hold
against. Rather than keep fighting Streamlit's internal DOM structure, the hero was changed
to a normal in-flow block within the existing content column, with rounded corners — simpler,
fully robust to Streamlit's internals, and arguably better-looking than an edge-to-edge
banner would have been anyway.

**Bug 4 (a design bug, not a rendering bug) — once the image did render, it was barely
visible.** The first working version used a flat, fairly dark gradient overlay across the
whole image, which — combined with the aspect-ratio mismatch of cropping a square photo into
a short wide banner — left Earth almost imperceptible behind the title text. Fixed by
switching to a left-to-right scrim (opaque where the text sits, clear on the right) instead
of a flat overlay, and repositioning the crop so Earth sits clearly in the uncovered area.

**How each of these was actually diagnosed:** in every case, by driving the real running app
in a real browser and inspecting the live DOM/computed styles directly (element presence,
attribute values, computed `background-image` length, what element was actually painted at a
given screen coordinate) — not by reading the CSS and reasoning about what *should* happen.
Two of these four bugs (2 and 3) look, from the CSS source alone, like they should have
worked; the DOM said otherwise both times.

---

## Summary: what actually made this hard

None of these were hard *coding* problems — every fix, in isolation, is a few lines. What
made the session take real effort was that almost every "this should just work" step had a
gap between what the code said and what actually happened at runtime:

- A model that accepts the request shape but silently rejects the one parameter this project
  needs.
- A retrieval/verification pipeline that runs cleanly and produces a *wrong, confident*
  answer instead of an error.
- A `requirements.txt` that installed perfectly on my machine and failed twice, for two
  independent reasons, on the actual target environment.
- CSS that reads correctly but three different framework-specific behaviors quietly
  stripped or repositioned what it was supposed to do.

The common thread in how each one got resolved: don't trust that something worked because it
didn't crash — check the actual number, the actual DOM, the actual installed package list,
against what it should be.
