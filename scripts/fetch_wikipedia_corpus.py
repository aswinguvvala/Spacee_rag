"""One-off (re-runnable) tool that fetches corpus source documents from Wikipedia.

This is how every ``corpus/*.txt`` file is produced -- kept as a script rather
than folded into ``src/ingestion.py`` because it's a data-acquisition step,
not part of the retrieval pipeline (``src/ingestion.py`` only ever reads
already-saved ``.txt`` files; it never talks to the network). Keeping it
separate also keeps provenance reproducible: anyone can re-run this file
against ``ARTICLES`` below to regenerate the corpus from scratch.

Uses the MediaWiki API (``action=query&prop=extracts&explaintext=1``) to pull
plain-text article bodies. All Wikipedia text is CC BY-SA 4.0 -- see
``corpus/SOURCES.md`` for the attribution this license requires.

Usage:
    python -m scripts.fetch_wikipedia_corpus
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from src.utils import CORPUS_DIR, get_logger

logger = get_logger(__name__)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
# Wikimedia's API etiquette requires a descriptive User-Agent identifying the
# project and a contact method: https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "provenance-rag-portfolio-project/1.0 (educational; contact via github)"
REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 10.0

# (article title as it appears on Wikipedia, output filename slug)
# Grouped by category purely for human readability of this list.
ARTICLES: list[tuple[str, str]] = [
    # --- Space agencies (world) ---
    ("NASA", "nasa"),
    ("European Space Agency", "esa"),
    ("Roscosmos", "roscosmos"),
    ("Indian Space Research Organisation", "isro"),
    ("China National Space Administration", "cnsa"),
    ("Japan Aerospace Exploration Agency", "jaxa"),
    ("Canadian Space Agency", "canadian_space_agency"),
    ("Soviet space program", "soviet_space_program"),

    # --- US crewed spaceflight programs (beyond the existing Apollo set) ---
    ("Project Mercury", "project_mercury"),
    ("Project Gemini", "project_gemini"),
    ("Skylab", "skylab"),
    ("Space Shuttle", "space_shuttle"),
    ("Space Shuttle Challenger disaster", "space_shuttle_challenger_disaster"),
    ("Space Shuttle Columbia disaster", "space_shuttle_columbia_disaster"),
    ("Artemis program", "artemis_program"),
    ("Artemis 1", "artemis_1"),
    ("International Space Station", "international_space_station"),
    ("Apollo 1", "apollo_1"),
    ("Apollo 8", "apollo_8"),
    ("Apollo 9", "apollo_9"),
    ("Apollo 10", "apollo_10"),
    ("Apollo 14", "apollo_14"),
    ("Apollo 15", "apollo_15"),
    ("Apollo 16", "apollo_16"),
    ("Apollo–Soyuz Test Project", "apollo_soyuz_test_project"),

    # --- Spacecraft & hardware ---
    ("Space Shuttle orbiter", "space_shuttle_orbiter"),
    ("Falcon 9", "falcon_9"),
    ("SpaceX Dragon 2", "spacex_dragon_2"),
    ("SpaceX Starship", "spacex_starship"),
    ("Soyuz (spacecraft)", "soyuz_spacecraft"),
    ("Voyager program", "voyager_program"),
    ("Voyager 1", "voyager_1"),
    ("Voyager 2", "voyager_2"),
    ("New Horizons", "new_horizons"),
    ("Cassini–Huygens", "cassini_huygens"),
    ("Galileo (spacecraft)", "galileo_spacecraft"),
    ("Juno (spacecraft)", "juno_spacecraft"),

    # --- Soviet / Russian programs ---
    ("Sputnik 1", "sputnik_1"),
    ("Vostok programme", "vostok_programme"),
    ("Voskhod programme", "voskhod_programme"),
    ("Soyuz programme", "soyuz_programme"),
    ("Mir", "mir"),
    ("Luna programme", "luna_programme"),
    ("Venera", "venera"),
    ("Salyut programme", "salyut_programme"),
    ("Buran (spacecraft)", "buran_spacecraft"),

    # --- International missions ---
    ("Chandrayaan programme", "chandrayaan_programme"),
    ("Chandrayaan-3", "chandrayaan_3"),
    ("Mars Orbiter Mission", "mars_orbiter_mission"),
    ("Gaganyaan", "gaganyaan"),
    ("Tianwen-1", "tianwen_1"),
    ("Chang'e program", "change_program"),
    ("Tiangong space station", "tiangong_space_station"),
    ("Hayabusa2", "hayabusa2"),
    ("Akatsuki (spacecraft)", "akatsuki_spacecraft"),

    # --- Mars exploration ---
    ("Mars rover", "mars_rover"),
    ("Curiosity (rover)", "curiosity_rover"),
    ("Perseverance (rover)", "perseverance_rover"),
    ("Viking program", "viking_program"),
    ("Mars Pathfinder", "mars_pathfinder"),
    ("InSight", "insight"),

    # --- Telescopes / astronomy missions ---
    ("Hubble Space Telescope", "hubble_space_telescope"),
    ("James Webb Space Telescope", "james_webb_space_telescope"),
    ("Kepler space telescope", "kepler_space_telescope"),
    ("Spitzer Space Telescope", "spitzer_space_telescope"),

    # --- Companies ---
    ("SpaceX", "spacex"),
    ("Blue Origin", "blue_origin"),

    # --- People ---
    ("Neil Armstrong", "neil_armstrong"),
    ("Buzz Aldrin", "buzz_aldrin"),
    ("Yuri Gagarin", "yuri_gagarin"),
    ("Valentina Tereshkova", "valentina_tereshkova"),
    ("Sally Ride", "sally_ride"),
    ("Kalpana Chawla", "kalpana_chawla"),

    # --- General / cross-cutting ---
    ("Space exploration", "space_exploration"),
    ("History of spaceflight", "history_of_spaceflight"),
    ("Space Race", "space_race"),
]


class FetchError(Exception):
    """Raised when an article can't be fetched or has no usable extract."""


def fetch_article(title: str) -> tuple[str, str]:
    """Fetch one article's plain-text body from the Wikipedia API.

    Args:
        title: Exact or approximate Wikipedia article title. Redirects are
            followed automatically.

    Returns:
        Tuple of ``(canonical_title, plain_text)`` -- the canonical title may
        differ from ``title`` if a redirect was followed.

    Raises:
        FetchError: If the request fails, the article doesn't exist, or the
            extract is empty.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }
    last_reason = "unknown error"
    data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                WIKIPEDIA_API_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", RETRY_BACKOFF_SECONDS * attempt))
                logger.warning(
                    "Rate limited fetching %r (attempt %d/%d); waiting %.0fs",
                    title, attempt, MAX_RETRIES, wait,
                )
                last_reason = "rate limited (429)"
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.RequestException as exc:
            last_reason = str(exc)
            time.sleep(RETRY_BACKOFF_SECONDS)

    if data is None:
        raise FetchError(f"Request failed for {title!r} after {MAX_RETRIES} attempts: {last_reason}")

    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise FetchError(f"Article not found: {title!r}")

    page = pages[0]
    text = page.get("extract", "")
    if not text.strip():
        raise FetchError(f"Empty extract for {title!r}")

    return page.get("title", title), text


def main() -> None:
    """Fetch every article in ``ARTICLES`` and write it to ``corpus/<slug>.txt``."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    fetched: list[tuple[str, str, str]] = []  # (title, slug, canonical_title)
    failures: list[tuple[str, str]] = []  # (title, reason)

    for title, slug in ARTICLES:
        out_path: Path = CORPUS_DIR / f"{slug}.txt"
        if out_path.exists():
            logger.info("Skipping %s: %s already exists", title, out_path.name)
            fetched.append((title, slug, title))
            continue
        try:
            canonical_title, text = fetch_article(title)
        except FetchError as exc:
            logger.warning("Failed to fetch %r: %s", title, exc)
            failures.append((title, str(exc)))
            continue

        out_path.write_text(text, encoding="utf-8")
        logger.info("Saved %s (%d chars) -> %s", canonical_title, len(text), out_path.name)
        fetched.append((title, slug, canonical_title))
        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info("Done: %d fetched/present, %d failed", len(fetched), len(failures))
    if failures:
        logger.warning("Failed titles: %s", [t for t, _ in failures])

    # Machine-readable summary for the caller (e.g. to regenerate SOURCES.md).
    import json

    summary_path = CORPUS_DIR / "_fetch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "fetched": [{"title": t, "slug": s, "canonical_title": c} for t, s, c in fetched],
                "failed": [{"title": t, "reason": r} for t, r in failures],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote fetch summary to %s", summary_path)


if __name__ == "__main__":
    main()
