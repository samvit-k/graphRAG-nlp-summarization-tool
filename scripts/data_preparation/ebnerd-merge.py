"""Build a long-format EB-NeRD history dataset for 10 sampled users."""

from pathlib import Path
import pandas as pd

RANDOM_STATE = 42
MIN_HISTORY_COUNT = 50
NUMBER_OF_USERS = 10

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = REPOSITORY_ROOT / "user-data/ebnerd_demo/train/history.parquet"
ARTICLES_PATH = REPOSITORY_ROOT / "user-data/ebnerd_demo/articles.parquet"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "user-data/ebnerd-fixed"
USER_HISTORY_PATH = OUTPUT_DIRECTORY / "ebnerd_fixed.csv"

HISTORY_COLUMNS = [
    "user_id",
    "impression_time_fixed",
    "scroll_percentage_fixed",
    "article_id_fixed",
    "read_time_fixed",
]

ALIGNED_COLUMNS = [
    "impression_time_fixed",
    "scroll_percentage_fixed",
    "article_id_fixed",
    "read_time_fixed",
]

ARTICLE_COLUMNS = [
    "article_id",
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

FINAL_COLUMNS = [
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

def fail(message: str) -> None:
    raise ValueError(f"Integrity check failed: {message}")

def validate_required_columns(
    dataframe: pd.DataFrame, required_columns: list[str], table_name: str
) -> None:
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        fail(f"{table_name} is missing required columns: {missing_columns}")

def main() -> None:
    history = pd.read_parquet(HISTORY_PATH, engine="pyarrow")
    articles = pd.read_parquet(ARTICLES_PATH, engine="pyarrow")

    validate_required_columns(history, HISTORY_COLUMNS, "history.parquet")
    validate_required_columns(articles, ARTICLE_COLUMNS, "articles.parquet")

    aligned_lengths = history[ALIGNED_COLUMNS].map(len)
    unequal_lengths = aligned_lengths.nunique(axis=1).ne(1)
    if unequal_lengths.any():
        bad_rows = history.index[unequal_lengths].tolist()[:5]
        fail(f"aligned history arrays have unequal lengths in rows: {bad_rows}")

    history = history[HISTORY_COLUMNS].copy()
    history["history_count"] = history["article_id_fixed"].map(len)

    eligible_users = history.loc[
        history["history_count"] >= MIN_HISTORY_COUNT,
        ["user_id", "history_count"],
    ]
    if len(eligible_users) < NUMBER_OF_USERS:
        fail(
            f"only {len(eligible_users)} users have at least "
            f"{MIN_HISTORY_COUNT} historical interactions"
        )

    selected_users = (
        eligible_users.sample(n=NUMBER_OF_USERS, random_state=RANDOM_STATE)
        .rename(columns={"user_id": "ebnerd_user_id"})
        .sort_values("ebnerd_user_id")
        .reset_index(drop=True)
    )
    if selected_users["ebnerd_user_id"].nunique() != NUMBER_OF_USERS:
        fail("the reproducible sample does not contain exactly 10 unique users")

    selected_history = history[
        history["user_id"].isin(selected_users["ebnerd_user_id"])
    ].copy()

    flattened_history = selected_history.explode(
        ALIGNED_COLUMNS, ignore_index=True
    ).rename(
        columns={
            "user_id": "ebnerd_user_id",
            "impression_time_fixed": "history_time",
            "article_id_fixed": "article_id",
            "read_time_fixed": "read_time",
            "scroll_percentage_fixed": "scroll_percentage",
        }
    )
    flattened_history = flattened_history[
        [
            "ebnerd_user_id",
            "history_time",
            "article_id",
            "read_time",
            "scroll_percentage",
        ]
    ]

    if flattened_history["ebnerd_user_id"].nunique() != NUMBER_OF_USERS:
        fail("flattened history does not contain exactly 10 unique users")
    if flattened_history["article_id"].isna().any():
        fail("article_id contains null values after flattening")

    articles = articles[ARTICLE_COLUMNS].copy()
    if not articles["article_id"].is_unique:
        duplicate_ids = (
            articles.loc[
                articles["article_id"].duplicated(keep=False), "article_id"
            ]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        fail(f"article_id is not unique in articles.parquet; sample: {duplicate_ids}")

    enriched_history = flattened_history.merge(
        articles,
        how="left",
        on="article_id",
        validate="many_to_one",
        indicator=True,
    )

    unique_article_count = flattened_history["article_id"].nunique()
    matched_article_count = enriched_history.loc[
        enriched_history["_merge"] == "both", "article_id"
    ].nunique()
    unmatched_ids = enriched_history.loc[
        enriched_history["_merge"] == "left_only", "article_id"
    ].drop_duplicates()
    unmatched_article_count = len(unmatched_ids)
    join_coverage = (
        100.0 * matched_article_count / unique_article_count
        if unique_article_count
        else 0.0
    )

    enriched_history = enriched_history.drop(columns="_merge")[FINAL_COLUMNS]

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    enriched_history.to_csv(USER_HISTORY_PATH, index=False)

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
