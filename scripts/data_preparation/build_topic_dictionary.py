"""Build the shared topic dictionary from an IPTC Media Topics en-GB JSON download.

  curl -o mediatopic-en-GB.json "https://cv.iptc.org/newscodes/mediatopic/?format=json&lang=en-GB"
  python scripts/data_preparation/build_topic_dictionary.py mediatopic-en-GB.json --retrieved 2026-11-17

Writes topic_data/topic_dictionary.json ({topic_id: {name, definition, parent?, aliases?}}, IPTC qcodes kept
verbatim, retired concepts dropped) and topic_data/topic_dictionary_source.json (provenance + content hash).
Local aliases come from topic_data/topic_aliases.json and may only point at existing, active topics.
"""

from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOPIC_DIR = ROOT / "topic_data"
DICTIONARY_PATH = TOPIC_DIR / "topic_dictionary.json"
SOURCE_PATH = TOPIC_DIR / "topic_dictionary_source.json"
ALIASES_PATH = TOPIC_DIR / "topic_aliases.json"
URL = "https://cv.iptc.org/newscodes/mediatopic/?format=json&lang=en-GB"
LANG = "en-GB"


def content_sha256(dictionary: dict) -> str:
    """Hash of the canonical JSON form, so formatting changes do not count as vocabulary changes."""
    canonical = json.dumps(dictionary, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build(raw: dict, aliases: dict[str, list[str]]) -> dict[str, dict]:
    concepts = raw["conceptSet"]
    qcode_by_uri = {c["uri"]: c["qcode"] for c in concepts}
    retired = {c["qcode"] for c in concepts if "retired" in c}
    out: dict[str, dict] = {}
    for c in concepts:
        if c["qcode"] in retired:
            continue
        name, definition = c["prefLabel"].get(LANG, "").strip(), c.get("definition", {}).get(LANG, "").strip()
        if not name or not definition:
            raise ValueError(f"{c['qcode']} lacks an {LANG} name or definition")
        entry = {"name": name, "definition": definition}
        broader = [qcode_by_uri[u] for u in c.get("broader", [])]
        if len(broader) > 1:
            raise ValueError(f"{c['qcode']} has several parents: {broader}")
        if broader:
            if broader[0] in retired:
                raise ValueError(f"{c['qcode']} has a retired parent {broader[0]}")
            entry["parent"] = broader[0]
        out[c["qcode"]] = entry
    for topic_id, extra in aliases.items():
        if topic_id not in out:
            raise ValueError(f"alias target {topic_id} is unknown or retired")
        cleaned = sorted({a.strip() for a in extra if a.strip()} - {out[topic_id]["name"]}, key=str.casefold)
        if cleaned:
            out[topic_id]["aliases"] = cleaned
    return dict(sorted(out.items()))


def levels(dictionary: dict) -> dict[int, int]:
    counts: dict[int, int] = {}
    for topic_id in dictionary:
        depth, node = 1, dictionary[topic_id]
        while "parent" in node:
            depth, node = depth + 1, dictionary[node["parent"]]
        counts[depth] = counts.get(depth, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", type=Path, help="IPTC Media Topics JSON downloaded from the URL above")
    parser.add_argument("--retrieved", default=dt.date.today().isoformat(), help="download date (YYYY-MM-DD)")
    args = parser.parse_args()

    raw_bytes = args.raw.read_bytes()
    raw = json.loads(raw_bytes)
    if raw.get("prefSchemeAlias") != "medtop":
        raise SystemExit(f"{args.raw} is not an IPTC Media Topics scheme")
    dictionary = build(raw, json.loads(ALIASES_PATH.read_text(encoding="utf-8")))
    DICTIONARY_PATH.write_text(json.dumps(dictionary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    source = {
        "source": "IPTC Media Topics",
        "url": URL,
        "language": LANG,
        "version": raw["dateReleased"],
        "retrieved": args.retrieved,
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "license": raw["licenceLink"],
        "copyright": raw["copyrightHolder"],
        "attribution": "Media Topics (c) IPTC - International Press Telecommunications Council, licensed under CC BY 4.0",
        "filters": "retired concepts removed; en-GB prefLabel/definition; broader -> parent; local aliases from topic_aliases.json",
        "topics": len(dictionary),
        "topics_per_level": levels(dictionary),
        "aliases_sha256": hashlib.sha256(ALIASES_PATH.read_bytes()).hexdigest(),
        "dictionary_sha256": content_sha256(dictionary),
    }
    SOURCE_PATH.write_text(json.dumps(source, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(dictionary)} topics to {DICTIONARY_PATH.name} (sha256 {source['dictionary_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
