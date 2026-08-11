# Corpus Sources

Topic: **space exploration**, broadly — space agencies (national and commercial), crewed
and uncrewed missions, spacecraft/hardware, and key figures, spanning the US, Soviet/Russian,
European, Indian, Chinese, and Japanese programs. Originally scoped to just the Apollo program;
broadened because a single-country, single-program corpus was narrower than the project needed.
The domain is still a single coherent one (space exploration, not "everything"), which is what
keeps entailment scoring meaningful and "unanswerable" questions constructible in a principled
way (real, adjacent space-related facts that are simply not in this corpus, rather than
arbitrary out-of-domain nonsense).

All text is extracted verbatim (plain text, no markup) from English Wikipedia via the
MediaWiki API (`action=query&prop=extracts&explaintext=1`), and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Attribution below satisfies
the license's attribution requirement; no changes were made to the extracted text other than
plain-text extraction itself. The original 8 articles were fetched on 2026-08-04; the
remaining 76 were fetched on 2026-08-12 via `scripts/fetch_wikipedia_corpus.py` (re-runnable —
see that file's docstring for how to regenerate or extend the corpus).

| File | Article | Source |
|---|---|---|
| `akatsuki_spacecraft.txt` | Akatsuki (spacecraft) | https://en.wikipedia.org/wiki/Akatsuki_(spacecraft) |
| `apollo_1.txt` | Apollo 1 | https://en.wikipedia.org/wiki/Apollo_1 |
| `apollo_10.txt` | Apollo 10 | https://en.wikipedia.org/wiki/Apollo_10 |
| `apollo_11.txt` | Apollo 11 | https://en.wikipedia.org/wiki/Apollo_11 |
| `apollo_12.txt` | Apollo 12 | https://en.wikipedia.org/wiki/Apollo_12 |
| `apollo_13.txt` | Apollo 13 | https://en.wikipedia.org/wiki/Apollo_13 |
| `apollo_14.txt` | Apollo 14 | https://en.wikipedia.org/wiki/Apollo_14 |
| `apollo_15.txt` | Apollo 15 | https://en.wikipedia.org/wiki/Apollo_15 |
| `apollo_16.txt` | Apollo 16 | https://en.wikipedia.org/wiki/Apollo_16 |
| `apollo_17.txt` | Apollo 17 | https://en.wikipedia.org/wiki/Apollo_17 |
| `apollo_8.txt` | Apollo 8 | https://en.wikipedia.org/wiki/Apollo_8 |
| `apollo_9.txt` | Apollo 9 | https://en.wikipedia.org/wiki/Apollo_9 |
| `apollo_csm.txt` | Apollo command and service module | https://en.wikipedia.org/wiki/Apollo_command_and_service_module |
| `apollo_lunar_module.txt` | Apollo Lunar Module | https://en.wikipedia.org/wiki/Apollo_Lunar_Module |
| `apollo_program.txt` | Apollo program | https://en.wikipedia.org/wiki/Apollo_program |
| `apollo_soyuz_test_project.txt` | Apollo–Soyuz | https://en.wikipedia.org/wiki/Apollo–Soyuz |
| `artemis_1.txt` | Artemis 1 | https://en.wikipedia.org/wiki/Artemis_1 |
| `artemis_program.txt` | Artemis program | https://en.wikipedia.org/wiki/Artemis_program |
| `blue_origin.txt` | Blue Origin | https://en.wikipedia.org/wiki/Blue_Origin |
| `buran_spacecraft.txt` | Buran (spacecraft) | https://en.wikipedia.org/wiki/Buran_(spacecraft) |
| `buzz_aldrin.txt` | Buzz Aldrin | https://en.wikipedia.org/wiki/Buzz_Aldrin |
| `canadian_space_agency.txt` | Canadian Space Agency | https://en.wikipedia.org/wiki/Canadian_Space_Agency |
| `cassini_huygens.txt` | Cassini–Huygens | https://en.wikipedia.org/wiki/Cassini–Huygens |
| `chandrayaan_3.txt` | Chandrayaan-3 | https://en.wikipedia.org/wiki/Chandrayaan-3 |
| `chandrayaan_programme.txt` | Chandrayaan programme | https://en.wikipedia.org/wiki/Chandrayaan_programme |
| `change_program.txt` | Chinese Lunar Exploration Program | https://en.wikipedia.org/wiki/Chinese_Lunar_Exploration_Program |
| `cnsa.txt` | China National Space Administration | https://en.wikipedia.org/wiki/China_National_Space_Administration |
| `curiosity_rover.txt` | Curiosity (rover) | https://en.wikipedia.org/wiki/Curiosity_(rover) |
| `esa.txt` | European Space Agency | https://en.wikipedia.org/wiki/European_Space_Agency |
| `falcon_9.txt` | Falcon 9 | https://en.wikipedia.org/wiki/Falcon_9 |
| `gaganyaan.txt` | Gaganyaan | https://en.wikipedia.org/wiki/Gaganyaan |
| `galileo_spacecraft.txt` | Galileo (spacecraft) | https://en.wikipedia.org/wiki/Galileo_(spacecraft) |
| `hayabusa2.txt` | Hayabusa2 | https://en.wikipedia.org/wiki/Hayabusa2 |
| `history_of_spaceflight.txt` | History of spaceflight | https://en.wikipedia.org/wiki/History_of_spaceflight |
| `hubble_space_telescope.txt` | Hubble Space Telescope | https://en.wikipedia.org/wiki/Hubble_Space_Telescope |
| `insight.txt` | InSight | https://en.wikipedia.org/wiki/InSight |
| `international_space_station.txt` | International Space Station | https://en.wikipedia.org/wiki/International_Space_Station |
| `isro.txt` | Indian Space Research Organisation | https://en.wikipedia.org/wiki/Indian_Space_Research_Organisation |
| `james_webb_space_telescope.txt` | James Webb Space Telescope | https://en.wikipedia.org/wiki/James_Webb_Space_Telescope |
| `jaxa.txt` | Japan Aerospace Exploration Agency | https://en.wikipedia.org/wiki/Japan_Aerospace_Exploration_Agency |
| `juno_spacecraft.txt` | Juno (spacecraft) | https://en.wikipedia.org/wiki/Juno_(spacecraft) |
| `kalpana_chawla.txt` | Kalpana Chawla | https://en.wikipedia.org/wiki/Kalpana_Chawla |
| `kepler_space_telescope.txt` | Kepler space telescope | https://en.wikipedia.org/wiki/Kepler_space_telescope |
| `luna_programme.txt` | Luna programme | https://en.wikipedia.org/wiki/Luna_programme |
| `mars_orbiter_mission.txt` | Mars Orbiter Mission | https://en.wikipedia.org/wiki/Mars_Orbiter_Mission |
| `mars_pathfinder.txt` | Mars Pathfinder | https://en.wikipedia.org/wiki/Mars_Pathfinder |
| `mars_rover.txt` | Mars rover | https://en.wikipedia.org/wiki/Mars_rover |
| `mir.txt` | Mir | https://en.wikipedia.org/wiki/Mir |
| `nasa.txt` | NASA | https://en.wikipedia.org/wiki/NASA |
| `neil_armstrong.txt` | Neil Armstrong | https://en.wikipedia.org/wiki/Neil_Armstrong |
| `new_horizons.txt` | New Horizons | https://en.wikipedia.org/wiki/New_Horizons |
| `perseverance_rover.txt` | Perseverance (rover) | https://en.wikipedia.org/wiki/Perseverance_(rover) |
| `project_gemini.txt` | Project Gemini | https://en.wikipedia.org/wiki/Project_Gemini |
| `project_mercury.txt` | Project Mercury | https://en.wikipedia.org/wiki/Project_Mercury |
| `roscosmos.txt` | Roscosmos | https://en.wikipedia.org/wiki/Roscosmos |
| `sally_ride.txt` | Sally Ride | https://en.wikipedia.org/wiki/Sally_Ride |
| `salyut_programme.txt` | Salyut programme | https://en.wikipedia.org/wiki/Salyut_programme |
| `saturn_v.txt` | Saturn V | https://en.wikipedia.org/wiki/Saturn_V |
| `skylab.txt` | Skylab | https://en.wikipedia.org/wiki/Skylab |
| `soviet_space_program.txt` | Soviet space program | https://en.wikipedia.org/wiki/Soviet_space_program |
| `soyuz_programme.txt` | Soyuz programme | https://en.wikipedia.org/wiki/Soyuz_programme |
| `soyuz_spacecraft.txt` | Soyuz (spacecraft) | https://en.wikipedia.org/wiki/Soyuz_(spacecraft) |
| `space_exploration.txt` | Space exploration | https://en.wikipedia.org/wiki/Space_exploration |
| `space_race.txt` | Space Race | https://en.wikipedia.org/wiki/Space_Race |
| `space_shuttle.txt` | Space Shuttle | https://en.wikipedia.org/wiki/Space_Shuttle |
| `space_shuttle_challenger_disaster.txt` | Space Shuttle Challenger disaster | https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster |
| `space_shuttle_columbia_disaster.txt` | Space Shuttle Columbia disaster | https://en.wikipedia.org/wiki/Space_Shuttle_Columbia_disaster |
| `space_shuttle_orbiter.txt` | Space Shuttle orbiter | https://en.wikipedia.org/wiki/Space_Shuttle_orbiter |
| `spacex.txt` | SpaceX | https://en.wikipedia.org/wiki/SpaceX |
| `spacex_dragon_2.txt` | SpaceX Dragon 2 | https://en.wikipedia.org/wiki/SpaceX_Dragon_2 |
| `spacex_starship.txt` | SpaceX Starship | https://en.wikipedia.org/wiki/SpaceX_Starship |
| `spitzer_space_telescope.txt` | Spitzer Space Telescope | https://en.wikipedia.org/wiki/Spitzer_Space_Telescope |
| `sputnik_1.txt` | Sputnik 1 | https://en.wikipedia.org/wiki/Sputnik_1 |
| `tiangong_space_station.txt` | Tiangong space station | https://en.wikipedia.org/wiki/Tiangong_space_station |
| `tianwen_1.txt` | Tianwen-1 | https://en.wikipedia.org/wiki/Tianwen-1 |
| `valentina_tereshkova.txt` | Valentina Tereshkova | https://en.wikipedia.org/wiki/Valentina_Tereshkova |
| `venera.txt` | Venera program | https://en.wikipedia.org/wiki/Venera_program |
| `viking_program.txt` | Viking program | https://en.wikipedia.org/wiki/Viking_program |
| `voskhod_programme.txt` | Voskhod programme | https://en.wikipedia.org/wiki/Voskhod_programme |
| `vostok_programme.txt` | Vostok programme | https://en.wikipedia.org/wiki/Vostok_programme |
| `voyager_1.txt` | Voyager 1 | https://en.wikipedia.org/wiki/Voyager_1 |
| `voyager_2.txt` | Voyager 2 | https://en.wikipedia.org/wiki/Voyager_2 |
| `voyager_program.txt` | Voyager program | https://en.wikipedia.org/wiki/Voyager_program |
| `yuri_gagarin.txt` | Yuri Gagarin | https://en.wikipedia.org/wiki/Yuri_Gagarin |

## Deliberately excluded (for the abstention benchmark)

`eval/qa_set.jsonl`'s unanswerable-question subset needs real topics that are plausible things
a visitor might ask but aren't backed by any chunk in this index. With the old Apollo-only
corpus that was Apollo 18–20, the Shuttle, the Soviet Luna program, and Artemis — all of which
are now *in* the corpus and answerable. **The existing unanswerable questions in
`eval/qa_set.jsonl` need to be revisited before the next eval run**, or they'll silently start
passing as "answered" instead of correctly abstained.

Going forward, plausible-but-excluded topics for new unanswerable questions include: space
tourism / Virgin Galactic, Starlink and satellite mega-constellations, asteroid mining, SETI,
Mars colonization proposals, private space stations (Axiom, Orbital Reef), and any mission or
event after this corpus's fetch date (2026-08-12).
