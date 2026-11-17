# Shared topic dictionary: source and coverage

`topic_dictionary.json` is the shared vocabulary for labeling **MIND** and **ReCOVery** articles.
Its keys are stable topic IDs, and each value has a `name`, a `definition`, and optionally `aliases` and `parent`.
Article assignments are stored separately from `topic_dictionary.json`. Each article ID maps to zero to five topic IDs.

## Why IPTC Media Topics

Before building this, we looked for a team-maintained vocabulary. We checked:

- `main` and every remote branch (only `main` exists);
- stashes;
- untracked files;
- `topic_data/`, `scripts/data_preparation/`, `scripts/matching/`;
- the history (`git log -S taxonomy`).

None was found. The only label artifacts were free-form, LLM-generated label caches, such as `article_interests_P07.json` in `0b85707`. MIND's own `category`/`subcategory` fields were used only as reference, as alias crosswalk terms. IPTC Media Topics is therefore the starting point.

## Source

| | |
|---|---|
| Vocabulary | IPTC Media Topics (NewsCodes), scheme `medtop` |
| URL | https://cv.iptc.org/newscodes/mediatopic/?format=json&lang=en-GB |
| Version | `dateReleased` 2026-09-30T12:00:00+00:00 (the IPTC tree view shows it as "as of 2026-10-16") |
| Retrieved | 2026-11-17 |
| Raw file sha256 | `fdeca17fe2a31cf22dc0a2a66ebe4b1c940012ff08d05b02611192721d92738d` (not committed) |
| License | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) |
| Attribution | Media Topics © IPTC – International Press Telecommunications Council (https://iptc.org), licensed under CC BY 4.0 |

`topic_dictionary_source.json` holds the same provenance in machine-readable form, plus the dictionary content hash. `labeling.py` reads that file and refuses to run if the dictionary no longer matches it.

## What the build does (`scripts/data_preparation/build_topic_dictionary.py`)

- Keeps the IPTC qcodes **verbatim** as topic IDs, for example `medtop:20000449`.
- Drops **retired** concepts. That leaves 1,394 concepts → **1,059 active topics**, at these levels:

  | Level | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | Topics | 17 | 120 | 504 | 364 | 49 | 5 |

- `name` is the en-GB `prefLabel` and `definition` is the en-GB `definition`. No names or definitions are edited.
- `parent` is the IPTC `broader` concept. Every concept has at most one parent, and no active topic has a retired parent.
- `aliases` come only from `topic_aliases.json`. These are local, hand-written search terms:
  - US spellings and synonyms, such as `pediatric` → paediatrics and `airline` → air transport;
  - league names, such as `NFL` and `NHL`;
  - MIND subcategory codes, such as `football_nfl`, `autosnews` and `tv-celebrity`.

  Aliases never add or rename topics. The build fails if an alias targets an unknown or retired ID. Ambiguous standalone words such as *official*, *testing*, *evacuation*, *oil*, *tour* and *store* were deliberately left out. More specific phrases, such as *diagnostic test*, *brush fire* and *crude oil*, are used instead.
- Rebuilding from the same raw file and aliases is byte-identical (verified):

```sh
curl -o mediatopic-en-GB.json "https://cv.iptc.org/newscodes/mediatopic/?format=json&lang=en-GB"
python scripts/data_preparation/build_topic_dictionary.py mediatopic-en-GB.json --retrieved 2026-11-17
```

The dictionary content sha256 is `8cf72119231ad1071b22e10b0a65d5cc3cb30d7de369fdc8341dda140804f94c`. Any change to a topic or alias changes this hash. The hash is stored in every label cache, so assignments made under an older dictionary are refused rather than reused.

## How the vocabulary reaches the model

The full vocabulary is about 1M characters, which is too large for every prompt. Each labeling prompt therefore offers two things:

- the **backbone**: all 137 level-1 and level-2 topics, which are the same in every prompt;
- the **top 40 deeper topics** for that article, from a local BM25 search over name, aliases and definition. Headline terms are weighted ×2, and the IPTC ancestors of each hit are added too.

The model may return only IDs it was offered. This shortlist is the same for MIND and ReCOVery.

## Coverage check (shortlist only, no LLM calls)

**What it measures:** whether the expected topic was *offered* to the model, so that the model *could* choose it. It does not measure whether the model *did* choose it. That requires running the labeler.

**Protocol:**

1. Draw seeded samples: 3 MIND articles from each of the 14 categories with at least 500 articles (42 in total), plus 40 random ReCOVery articles.
2. For each article, pick the single **expected most-specific IPTC ID** by reading the article and the full dictionary tree, *before* running the shortlister.
3. Record whether that ID was offered, and whether it came from the backbone or from BM25.
4. Tune using round 1 only. Then draw a **fresh, disjoint round 2**, judge it the same way, and score it once with the tuned shortlist. No further tuning.

| Sample | Dataset | Expected ID offered | Backbone | BM25 | BM25 ancestor | Missed |
|---|---|---|---|---|---|---|
| Round 1, untuned | MIND | 25/42 (59.5%) | 6 | 17 | 2 | 17 |
| | ReCOVery | 28/40 (70.0%) | 2 | 25 | 1 | 12 |
| Round 1, tuned (in-sample) | MIND | 34/42 (81.0%) | 6 | 27 | 1 | 8 |
| | ReCOVery | 36/40 (90.0%) | 2 | 34 | 0 | 4 |
| **Round 2, tuned (held out)** | **MIND** | **34/42 (81.0%)** | 7 | 27 | 0 | 8 |
| | **ReCOVery** | **33/40 (82.5%)** | 2 | 29 | 2 | 7 |

Tuning after round 1 made three changes:

- plural folding, so *companies*/*company* and *movies*/*movie* share a token;
- the aliases above;
- double weight for headline terms.

On the held-out sample, the tuned shortlist offered the expected topic for 34/42 MIND articles (81.0%) and 33/40 ReCOVery articles (82.5%). Round 2 was not scored with the untuned shortlist, so these numbers do not by themselves show a gain on round 2.

**Example misses in round 2:**

- MIND N6140 "Houston Police … allow practicing Sikhs" → `medtop:20000684` Sikhism. Word form: *Sikhs* does not fold to *Sikhism*.
- MIND N6156 "Couple Sues Taco Bell Over Chalupa Prices" → `medtop:20000114` litigation. The topic is implied ("sues"), not named.
- MIND N51226 "stay at Jim Beam's bourbon distillery … Airbnb" → `medtop:20000333` hotel and accommodation.
- ReCOVery 500 "Tesla turns a profit …" → `medtop:20000178` corporate earnings. Implied, not named.
- ReCOVery 1903 "BBC blasted after political editor …" → `medtop:20000046` news media.
- ReCOVery 1159 "Obama criticizes coronavirus response" → `medtop:20001312` political leadership. It ranked 244th: the long lede dilutes the score.

When the expected topic is missed, its level-1 and level-2 ancestors are still offered, because they are in the backbone. In the Sikhism case, for example, "belief systems" and "religion" are offered.

If real assignments later show that these misses matter, there are two options: add aliases from reviewed misses, or move to an embedding-based shortlist.
