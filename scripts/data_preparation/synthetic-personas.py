"""Create deterministic synthetic composite personas from ANES and EB-NeRD data."""

from pathlib import Path
import pandas as pd

RANDOM_STATE = 42
NUMBER_OF_PERSONAS = 10

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANES_PATH = (
    REPOSITORY_ROOT
    / "user-data"
    / "anes_timeseries_2024_csv"
    / "anes_2024_timeseries_data.csv"
)
EBNERD_PATH = (
    REPOSITORY_ROOT / "user-data" / "ebnerd-fixed" / "ebnerd_fixed.csv"
)

MAPPING_PATH = (
    REPOSITORY_ROOT
    / "user-data"
    / "merged_dataset"
    / "persona_mapping.csv"
)
PERSONAS_PATH = (
    REPOSITORY_ROOT / "user-data" / "merged_dataset" / "synthetic_personas.csv"
)
HISTORY_PATH = (
    REPOSITORY_ROOT
    / "user-data"
    / "merged_dataset"
    / "synthetic_persona_history.csv"
)

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
    "Frequency of Attention to Politics and Elections",
    "Interest in Following Campaigns",
    "Registered to Vote",
    "Talked to Anyone About Voting for/Against Party or Candidate",
    "Attended Online Political Meetings/Rallies/Speeches/Fundraisers",
    "Attended In-Person Political Meetings/Rallies/Speeches/Dinners",
    "Wore Campaign Button or Displayed Sign/Sticker",
    "Did Other Work for Party or Candidate",
    "Contributed Money to Individual Candidate",
    "Contributed Money to Political Party",
    "Voted in 2024 Election",
]

EBNERD_COLUMNS = [
    "ebnerd_user_id",
    "history_time",
    "article_id",
    "read_time",
    "scroll_percentage",
    "title",
    "subtitle",
    "body",
    "topics",
    "category",
    "subcategory",
    "category_str",
    "published_time",
    "url",
]

PERSONA_COLUMNS = [
    "persona_id",
    "anes_case_id",
    "ebnerd_user_id",
    *ANES_COLUMNS[1:],
]

HISTORY_COLUMNS = ["persona_id", *EBNERD_COLUMNS]

def require(condition: bool, message: str) -> None:
    """Stop immediately when an input or output integrity check fails."""
    if not condition:
        raise ValueError(f"Integrity check failed: {message}")

def validate_columns(
    dataframe: pd.DataFrame, required_columns: list[str], name: str
) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    require(not missing, f"{name} is missing required columns: {missing}")

def create_mapping(anes: pd.DataFrame, ebnerd: pd.DataFrame) -> pd.DataFrame:
    """Pair ordered ANES rows with a fixed-seed shuffle of sorted EB-NeRD IDs."""
    sorted_user_ids = sorted(ebnerd["ebnerd_user_id"].unique())
    shuffled_user_ids = (
        pd.Series(sorted_user_ids)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    return pd.DataFrame(
        {
            "persona_id": [
                f"P{number:02d}" for number in range(1, NUMBER_OF_PERSONAS + 1)
            ],
            "anes_case_id": anes["Case ID"].reset_index(drop=True),
            "ebnerd_user_id": shuffled_user_ids,
            "pairing_method": "fixed_seed_random",
            "pairing_seed": RANDOM_STATE,
        }
    )

def main() -> None:
    for path in (ANES_PATH, EBNERD_PATH):
        require(path.is_file(), f"input file not found: {path}")

    anes = pd.read_csv(ANES_PATH)
    ebnerd = pd.read_csv(EBNERD_PATH)
    history_rows_before = len(ebnerd)

    validate_columns(anes, ANES_COLUMNS, "processed ANES CSV")
    validate_columns(ebnerd, EBNERD_COLUMNS, "processed EB-NeRD CSV")
    require(len(anes) == NUMBER_OF_PERSONAS, "ANES must contain exactly 10 rows")
    require(anes["Case ID"].is_unique, "ANES Case ID values must be unique")
    require(
        ebnerd["ebnerd_user_id"].notna().all(),
        "EB-NeRD ebnerd_user_id contains missing values",
    )
    require(
        ebnerd["ebnerd_user_id"].nunique() == NUMBER_OF_PERSONAS,
        "EB-NeRD must contain exactly 10 unique ebnerd_user_id values",
    )

    mapping = create_mapping(anes, ebnerd)

    personas = anes.rename(columns={"Case ID": "anes_case_id"}).copy()
    personas.insert(0, "persona_id", mapping["persona_id"])
    personas.insert(2, "ebnerd_user_id", mapping["ebnerd_user_id"])
    personas = personas[PERSONA_COLUMNS]

    history = ebnerd.copy()
    history["_source_row"] = range(history_rows_before)
    history = history.merge(
        mapping[["persona_id", "ebnerd_user_id"]],
        how="left",
        on="ebnerd_user_id",
        validate="many_to_one",
        sort=False,
    )
    expected_source_rows = pd.Series(range(history_rows_before), name="_source_row")
    require(
        history["_source_row"].reset_index(drop=True).equals(expected_source_rows),
        "historical article rows were lost, duplicated, or reordered by the merge",
    )
    history = history.drop(columns="_source_row")[HISTORY_COLUMNS]

    checks = {
        "persona_mapping.csv has exactly 10 rows": len(mapping)
        == NUMBER_OF_PERSONAS,
        "persona_id is unique in persona_mapping.csv": mapping[
            "persona_id"
        ].is_unique,
        "anes_case_id is unique in persona_mapping.csv": mapping[
            "anes_case_id"
        ].is_unique,
        "ebnerd_user_id is unique in persona_mapping.csv": mapping[
            "ebnerd_user_id"
        ].is_unique,
        "synthetic_personas.csv has exactly 10 rows": len(personas)
        == NUMBER_OF_PERSONAS,
        "synthetic_personas.csv has exactly 10 unique persona IDs": personas[
            "persona_id"
        ].nunique()
        == NUMBER_OF_PERSONAS,
        "history row count is unchanged": len(history) == history_rows_before,
        "every history row has a persona_id": history["persona_id"].notna().all(),
        "history contains exactly 10 persona IDs": history["persona_id"].nunique()
        == NUMBER_OF_PERSONAS,
        "no historical rows were lost or duplicated": len(history)
        == history_rows_before,
    }
    for description, passed in checks.items():
        require(bool(passed), description)

    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERSONAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(MAPPING_PATH, index=False)
    personas.to_csv(PERSONAS_PATH, index=False)
    history.to_csv(HISTORY_PATH, index=False)

    print("Synthetic persona mapping created:")
    for row in mapping.itertuples(index=False):
        print(
            f"{row.persona_id} -> ANES {row.anes_case_id} "
            f"-> EB-NeRD {row.ebnerd_user_id}"
        )

if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
