"""Pair ten qualifying MIND users one-to-one with the ten sampled ANES respondents."""

from __future__ import annotations
import operator
import sys
from pathlib import Path
import pandas as pd # type: ignore

RANDOM_STATE = 42
NUMBER_OF_PERSONAS = 10
PERSONA_IDS = [f"P{number:02d}" for number in range(1, NUMBER_OF_PERSONAS + 1)]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MIND_DIR = REPOSITORY_ROOT / "user_data" / "MIND_clean"
BEHAVIORS_PATH = MIND_DIR / "behaviors.csv"
NEWS_PATH = MIND_DIR / "news.csv"
ANES_PATH = (
    REPOSITORY_ROOT
    / "user_data"
    / "anes_timeseries_2024_csv"
    / "anes_2024_timeseries_data.csv"
)
OUTPUT_DIR = REPOSITORY_ROOT / "user_data" / "merged_dataset"
MERGED_PATH = OUTPUT_DIR / "merged.csv"

CRITERIA = [
    ("unique_history_articles", operator.ge, 20, ">= 20 unique history articles"),
    ("found_fraction", operator.ge, 0.90, ">= 90% of history found in news.csv"),
    ("distinct_categories", operator.ge, 3, ">= 3 distinct categories"),
    ("behaviour_rows", operator.eq, 1, "appears in exactly 1 behaviours row"),
]

EXPECTED_QUALIFYING_USERS = 11_661
SELECTED_USER_IDS = [
    "U81131", "U28674", "U91253", "U88301", "U58352",
    "U28088", "U39845", "U38784", "U43303", "U56409",
]

ANES_COLUMNS = [
    "Case ID",
    "Respondent Age on Election Day",
    "Respondent Gender",
    "Respondent Occupation Status",
    "Respondent Education Level (5 Category)",
    "Total Household Income (6 Category)",
    "Sample State (Postal Abbreviation)",
    "Census Region",
    "Sample State (FIPS Code)",
    "Party Identification (Summary)",
    "Liberal-Conservative Self-Placement (7pt)",
]

BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
CARRIED_COLUMNS = ["user_id", "time", "history", "impressions"]
# user_id sits directly right of Case ID so the two identifiers read together.
PERSONA_COLUMNS = (
    ["persona_id", "Case ID", "user_id"]
    + ANES_COLUMNS[1:]
    + ["time", "history", "impressions"]
)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Integrity check failed: {message}")

def read_table(path: Path, name: str, required: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, na_filter=False, encoding="utf-8")
    missing = [column for column in required if column not in frame.columns]
    require(not missing, f"{name} is missing required columns: {missing}")
    return frame

def summarize_histories(
    behaviors: pd.DataFrame, categories: dict[str, str]
) -> pd.DataFrame:
    """Reduce behaviours to one row per user carrying every criterion's input."""
    require(
        behaviors.groupby("user_id")["history"].nunique().max() <= 1,
        "a user_id has more than one distinct history string; dedupe is unsafe",
    )

    summary = behaviors.drop_duplicates("user_id").set_index("user_id")[["history"]]
    articles = summary.pop("history").map(lambda h: list(dict.fromkeys(h.split())))

    summary["unique_history_articles"] = articles.map(len)
    summary["found_articles"] = articles.map(
        lambda ids: sum(article in categories for article in ids)
    )
    summary["found_fraction"] = (
        summary["found_articles"] / summary["unique_history_articles"]
    ).fillna(0.0)
    summary["distinct_categories"] = articles.map(
        lambda ids: len({categories[a] for a in ids if a in categories})
    )
    summary["behaviour_rows"] = behaviors.groupby("user_id").size()
    return summary

def qualify(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Apply the criteria cumulatively, returning the pool and the funnel counts."""
    mask = pd.Series(True, index=summary.index)
    funnel = [("all users", len(summary))]
    for column, compare, threshold, label in CRITERIA:
        mask &= compare(summary[column], threshold)
        funnel.append((label, int(mask.sum())))
    return summary[mask], funnel

def unmet_criteria(row: pd.Series) -> list[str]:
    # `:g` because a mixed-dtype row slice upcasts the integer counts to float.
    return [
        f"{label} (has {row[column]:g})"
        for column, compare, threshold, label in CRITERIA
        if not compare(row[column], threshold)
    ]

def derive_cohort(pool: pd.DataFrame) -> list[str]:
    """The seeded draw that produced SELECTED_USER_IDS.

    Retained as the method of record and re-run only to advise the operator when
    the frozen cohort stops validating. `sorted` is lexicographic on the raw id
    and is load-bearing, because `.sample` selects by position.
    """
    return (
        pd.Series(sorted(pool.index))
        .sample(n=NUMBER_OF_PERSONAS, random_state=RANDOM_STATE)
        .tolist()
    )

def validate_frozen_cohort(summary: pd.DataFrame, pool: pd.DataFrame) -> None:
    """Confirm the frozen ids are present, unique and still fully eligible."""
    require(
        len(SELECTED_USER_IDS) == NUMBER_OF_PERSONAS,
        f"SELECTED_USER_IDS holds {len(SELECTED_USER_IDS)} ids, "
        f"expected {NUMBER_OF_PERSONAS}",
    )
    duplicates = sorted({u for u in SELECTED_USER_IDS if SELECTED_USER_IDS.count(u) > 1})
    require(not duplicates, f"SELECTED_USER_IDS contains duplicate ids: {duplicates}")

    absent = [u for u in SELECTED_USER_IDS if u not in summary.index]
    require(not absent, f"frozen user ids are absent from behaviors.csv: {absent}")

    ineligible = [u for u in SELECTED_USER_IDS if u not in pool.index]
    if ineligible:
        detail = "; ".join(
            f"{u} fails {', '.join(unmet_criteria(summary.loc[u]))}" for u in ineligible
        )
        require(
            False,
            f"frozen user ids are no longer eligible: {detail}. A fresh seeded draw "
            f"would yield {derive_cohort(pool)}; update SELECTED_USER_IDS deliberately.",
        )

def build_personas(anes: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    personas = anes.copy()
    personas.insert(0, "persona_id", PERSONA_IDS)
    for column in CARRIED_COLUMNS:
        personas[column] = selected[column].to_numpy()
    return personas[PERSONA_COLUMNS]

def main() -> int:
    missing = [p for p in (BEHAVIORS_PATH, NEWS_PATH, ANES_PATH) if not p.is_file()]
    if missing:
        print(
            "Missing input files:\n  " + "\n  ".join(str(p) for p in missing),
            file=sys.stderr,
        )
        return 1

    behaviors = read_table(BEHAVIORS_PATH, "behaviors.csv", BEHAVIOR_COLUMNS)
    news = read_table(NEWS_PATH, "news.csv", ["news_id", "category"])
    anes = read_table(ANES_PATH, "processed ANES CSV", ANES_COLUMNS)

    require(
        len(anes) == NUMBER_OF_PERSONAS,
        f"ANES CSV must contain exactly {NUMBER_OF_PERSONAS} rows, found {len(anes)}",
    )
    require(anes["Case ID"].is_unique, "ANES Case ID values must be unique")
    require(news["news_id"].is_unique, "news.csv contains duplicate news_id values")

    summary = summarize_histories(behaviors, dict(zip(news["news_id"], news["category"])))
    pool, funnel = qualify(summary)

    print("Qualification funnel (MIND behaviors.csv):")
    for (label, count), (_, previous) in zip(funnel, [(None, None)] + funnel):
        delta = "" if previous is None else f"   (-{previous - count:,})"
        print(f"  {label:<42} {count:>7,}{delta}")

    require(
        len(pool) >= NUMBER_OF_PERSONAS,
        f"only {len(pool)} users qualify, need at least {NUMBER_OF_PERSONAS}",
    )
    if len(pool) != EXPECTED_QUALIFYING_USERS:
        print(
            f"WARNING: qualifying pool is {len(pool):,} users, "
            f"expected {EXPECTED_QUALIFYING_USERS:,}.",
            file=sys.stderr,
        )
    validate_frozen_cohort(summary, pool)
    print(
        f"\nFrozen cohort of {NUMBER_OF_PERSONAS} validated against "
        f"{len(pool):,} qualifying users (drawn with seed {RANDOM_STATE}).\n"
    )

    selected = behaviors.set_index("user_id").loc[SELECTED_USER_IDS].reset_index()
    personas = build_personas(anes, selected)

    require(len(personas) == NUMBER_OF_PERSONAS, "personas must contain exactly 10 rows")
    require(list(personas.columns) == PERSONA_COLUMNS, "persona column order is incorrect")
    for column in ("persona_id", "Case ID", "user_id"):
        require(personas[column].is_unique, f"personas {column} values must be unique")
    require(
        (personas["history"].str.strip() != "").all(),
        "a selected persona has an empty history",
    )
    require(
        personas[ANES_COLUMNS].equals(anes[ANES_COLUMNS]),
        "ANES columns were altered or reordered during the merge",
    )

    print("Persona pairing:")
    identifiers = personas[["persona_id", "Case ID", "user_id"]]
    for persona_id, case_id, user_id in identifiers.itertuples(index=False, name=None):
        stats = summary.loc[user_id]
        print(
            f"  {persona_id} -> ANES {case_id} -> MIND {user_id}"
            f" ({stats.unique_history_articles:g} articles,"
            f" {stats.distinct_categories:g} categories)"
        )
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    personas.to_csv(MERGED_PATH, index=False)
    print(
        f"{MERGED_PATH.relative_to(REPOSITORY_ROOT)}: "
        f"{len(personas):,} rows x {personas.shape[1]} columns"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())