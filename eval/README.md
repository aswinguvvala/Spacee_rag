# `qa_set.jsonl` schema and curation method

55 hand-written questions: 40 answerable from the corpus, 15 deliberately not.

```json
{"id": "a001", "question": "...", "answerable": true, "gold_chunk_ids": ["apollo_11::chunk0005"]}
{"id": "u001", "question": "...", "answerable": false, "gold_chunk_ids": [], "category": "adjacent_program"}
```

- `gold_chunk_ids` (answerable only): every question was written against a specific verbatim
  sentence from a source article, then the actual persisted `corpus/index/chunks.json` was
  searched for that exact substring (case/dash-insensitive) to find which chunk(s) really
  contain it. Labels are therefore checked against the real chunk boundaries, not guessed from
  the raw Wikipedia text before chunking. A handful of questions have 2-3 gold chunks where the
  underlying fact is genuinely repeated in the source article (e.g. "Wernher von Braun" is
  named several times in `saturn_v.txt`).
- `category` (unanswerable only): why the question can't be answered from this corpus —
  - `adjacent_program`: a real, plausible question about a related-but-uncovered program
    (Artemis, the Space Shuttle) that this corpus doesn't include.
  - `missing_detail`: a fact adjacent to something the corpus *does* cover, but the specific
    detail asked for isn't stated (e.g. a wristwatch brand, a per-mission sample mass).
  - `false_premise`: the question assumes something that didn't happen (NASA landing
    astronauts on Mars) or is chronologically impossible (radioing the ISS during Apollo 13,
    which flew decades before the ISS existed) — a stress test for whether the system corrects
    or refuses instead of confabulating.

Every unanswerable question was checked against the full corpus text (targeted keyword
searches across all 8 documents) to confirm the specific fact it asks for is genuinely absent,
rather than merely assumed absent. Two early draft questions were caught and discarded this
way after the check turned up a real, if brief, mention (Harrison Schmitt's later U.S. Senate
career, and the Apollo program's total cost in 2024 dollars) — both were promoted to
*answerable* questions instead.
