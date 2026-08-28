"""Create deterministic synthetic composite personas from ANES and EB-NeRD data.

Interests are extracted for one persona at a time. Each of that persona's unique
articles is read exactly once by a local multilingual model, stored in a JSON
file keyed by article_id, and only then written into the history CSV.
"""

from pathlib import Path
import json
import urllib.error
import urllib.request
import pandas as pd

RANDOM_STATE = 42
NUMBER_OF_PERSONAS = 10

TARGET_PERSONA = "P07"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANES_PATH = (
    REPOSITORY_ROOT
    / "user_data"
    / "anes_timeseries_2024_csv"
    / "anes_2024_timeseries_data.csv"
)
EBNERD_PATH = (
    REPOSITORY_ROOT / "user_data" / "ebnerd_fixed" / "ebnerd_fixed.csv"
)

MAPPING_PATH = (
    REPOSITORY_ROOT
    / "user_data"
    / "merged_dataset"
    / "persona_mapping.csv"
)
PERSONAS_PATH = (
    REPOSITORY_ROOT / "user_data" / "merged_dataset" / "synthetic_personas.csv"
)
HISTORY_PATH = (
    REPOSITORY_ROOT
    / "user_data"
    / "merged_dataset"
    / "synthetic_persona_history.csv"
)
INTERESTS_PATH = (
    REPOSITORY_ROOT
    / "user_data"
    / "merged_dataset"
    / f"article_interests_{TARGET_PERSONA}.json"
)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_TIMEOUT_SECONDS = 300

NUM_CTX = 8192
RESPONSE_TOKENS = 128
SAFETY_TOKENS = 256
CHARS_PER_TOKEN = 2.5  # conservative for Danish under the Gemma tokenizer
PROMPT_CHAR_BUDGET = int(
    (NUM_CTX - RESPONSE_TOKENS - SAFETY_TOKENS) * CHARS_PER_TOKEN
)

MIN_INTERESTS = 1
MAX_INTERESTS = 5
EXTRACTION_ATTEMPTS = 3
CHECKPOINT_EVERY = 10

INTERESTS_SCHEMA = {
    "type": "object",
    "properties": {
        "interests": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": MIN_INTERESTS,
            "maxItems": MAX_INTERESTS,
        }
    },
    "required": ["interests"],
}

EXTRACTION_PROMPT = """You are extracting user-interest topics from news articles.

Return a concise array of interests that a reader of this article may reasonably be interested in. Use the article body as the primary source of evidence. The title and subtitle may be used as supporting context.

Rules:
Extract only interests directly supported by the article.
Do not infer personality, political ideology, demographics, beliefs, or preferences beyond the article topic itself.
Use meaningful, reusable topic labels.
Make labels specific enough to describe the actual subject of the article.
Avoid vague labels such as "society", "news", "current events", "lifestyle", "business", or "entertainment" when a more specific topic is available.
Avoid overly specific one-off event descriptions when a broader recurring topic is more appropriate.
Store between 1 and 5 distinct interests per article.
Return all interests in English.
Do not summarize the article."""

OUTPUT_INSTRUCTION = """The article above is written in Danish. Write every
interest label in English, in Title Case. Translate Danish common nouns and
Danish topic labels rather than copying them. Preserve proper nouns - people,
places, organisations, teams, brands, and competitions - using the recognised
English name where one exists (Copenhagen, not Kobenhavn) and the original
otherwise (Folketing, Superliga, Ekstra Bladet).

Each label must name the most specific recurring topic the article actually
supports: prefer "Cycling" over "Sport", "Mortgage Rates" over "Finance",
"Cancer Treatment" over "Health". Do not restate what happened in this article -
"Minister Resigns Over Expenses" is a summary, "Political Scandals" is an
interest."""

VAGUE_LABELS = frozenset(
    label.casefold()
    for label in (
        "Society",
        "News",
        "Current Events",
        "Lifestyle",
        "Business",
        "Entertainment",
        "Culture",
        "Health",
        "Politics",
        "Crime",
        "Sport",
        "Sports",
        "Technology",
        "Media",
        "People",
        "Events",
        "Public Figures",
        "Allegations",
        "Financial Transactions",
        "General",
        "General Interest",
        "World News",
        "Local News",
        "Human Interest",
        "Public Interest",
        "Daily Life",
        "Miscellaneous",
        "Other",
    )
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

INTERESTS_COLUMN = "interests"
_BODY_INDEX = EBNERD_COLUMNS.index("body")
HISTORY_COLUMNS = [
    "persona_id",
    *EBNERD_COLUMNS[: _BODY_INDEX + 1],
    INTERESTS_COLUMN,
    *EBNERD_COLUMNS[_BODY_INDEX + 1 :],
]

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

def is_missing(value: object) -> bool:
    """Treat NaN, None, and blank strings as absent evidence."""
    return pd.isna(value) or not str(value).strip()


def truncate_body(body: str, available_chars: int) -> str:
    """Shorten a body only when it cannot fit, cutting on a word boundary.

    Deterministic: the same body and budget always yield the same text.
    """
    if available_chars <= 0 or len(body) <= available_chars:
        return body
    clipped = body[:available_chars]
    boundary = clipped.rfind(" ")
    if boundary > 0:
        clipped = clipped[:boundary]
    return clipped.rstrip() + " [...]"


def build_extraction_prompt(article: pd.Series) -> str:
    """Build the prompt from the article text only.

    The EB-NeRD editorial taxonomy (`topics`, `category_str`) is deliberately
    withheld: it is Danish, it is the publisher's own labelling, and feeding it
    in leads the model to echo it back instead of reading the article.
    """
    evidence: list[str] = []
    for label, column in (("Title", "title"), ("Subtitle", "subtitle")):
        value = article.get(column)
        if not is_missing(value):
            evidence.append(f"{label}: {str(value).strip()}")

    scaffold = (
        f"{EXTRACTION_PROMPT}\n\nArticle:\n"
        + "\n".join(evidence)
        + "\nBody: \n\n"
        + OUTPUT_INSTRUCTION
    )

    body = article.get("body")
    if is_missing(body):
        return (
            f"{EXTRACTION_PROMPT}\n\nArticle:\n"
            + "\n".join(evidence)
            + f"\n\n{OUTPUT_INSTRUCTION}"
        )

    body_text = truncate_body(
        str(body).strip(), PROMPT_CHAR_BUDGET - len(scaffold)
    )
    evidence.append(f"Body: {body_text}")

    return (
        f"{EXTRACTION_PROMPT}\n\nArticle:\n"
        + "\n".join(evidence)
        + f"\n\n{OUTPUT_INSTRUCTION}"
    )


def call_model(prompt: str, attempt: int = 1) -> dict:
    """Ask the local model for a schema-constrained interests object.

    The first attempt is greedy so a clean run is reproducible. Later attempts
    sample, because re-sending an identical greedy request would return the same
    rejected labels and the retry could never recover.
    """
    request_body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": INTERESTS_SCHEMA,
            "options": {
                "temperature": 0 if attempt == 1 else 0.4,
                "seed": RANDOM_STATE + attempt - 1,
                "num_predict": RESPONSE_TOKENS,
                "num_ctx": NUM_CTX,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
        completion = json.loads(response.read())

    return json.loads(completion["response"])


def parse_interests(payload: object) -> list[str]:
    """Normalise a response into 1-5 specific labels, or [] when unusable."""
    if not isinstance(payload, dict):
        return []

    raw = payload.get("interests")
    if not isinstance(raw, list):
        return []

    specific: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        label = " ".join(item.split())
        if not label or label.casefold() in seen:
            continue
        seen.add(label.casefold())
        if label.casefold() in VAGUE_LABELS:
            continue
        specific.append(label)

    return specific[:MAX_INTERESTS]


def load_existing_interests() -> dict[int, list[str]]:
    """Resume a partial run rather than paying for the same articles twice."""
    if not INTERESTS_PATH.is_file():
        return {}
    try:
        stored = json.loads(INTERESTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}

    # Store the CLEANED array, not the raw one: a cached entry carrying vague or
    # duplicate labels would otherwise survive a resume untouched.
    cleaned_cache: dict[int, list[str]] = {}
    for article_id, interests in stored.items():
        cleaned = parse_interests({"interests": interests})
        if cleaned:
            cleaned_cache[int(article_id)] = cleaned
    return cleaned_cache


def save_interests(interests_by_article: dict[int, list[str]]) -> None:
    """Write the JSON keyed by article_id, checkpointing as the run proceeds."""
    INTERESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = {
        str(article_id): interests_by_article[article_id]
        for article_id in sorted(interests_by_article)
    }
    INTERESTS_PATH.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_persona_interests(persona_rows: pd.DataFrame) -> dict[int, list[str]]:
    """Read each of the persona's unique articles exactly once."""
    articles = persona_rows.drop_duplicates(subset="article_id").sort_values(
        "article_id"
    )
    interests_by_article = load_existing_interests()
    if interests_by_article:
        print(f"resuming: {len(interests_by_article)} articles already extracted")
    extracted_since_checkpoint = 0

    for position, article in enumerate(articles.itertuples(index=False), start=1):
        article_id = int(article.article_id)
        if article_id in interests_by_article:
            continue

        prompt = build_extraction_prompt(pd.Series(article._asdict()))
        interests: list[str] = []
        last_error: Exception | None = None
        for attempt in range(1, EXTRACTION_ATTEMPTS + 1):
            try:
                interests = parse_interests(call_model(prompt, attempt))
            except (OSError, urllib.error.URLError, ValueError, KeyError) as error:
                last_error = error
                interests = []
            if interests:
                break
            print(
                f"  retrying {article_id} (attempt {attempt}/{EXTRACTION_ATTEMPTS})"
            )

        require(
            len(interests) >= MIN_INTERESTS,
            f"no usable interests for article_id {article_id}: {last_error}",
        )

        interests_by_article[article_id] = interests
        extracted_since_checkpoint += 1
        print(f"  [{position}/{len(articles)}] {article_id}: {interests}")

        if extracted_since_checkpoint >= CHECKPOINT_EVERY:
            save_interests(interests_by_article)
            extracted_since_checkpoint = 0

    save_interests(interests_by_article)
    return interests_by_article


def validate_interests_json(expected_article_ids: set[int]) -> dict[str, list[str]]:
    """Re-read the JSON from disk and prove it is complete before any CSV write."""
    require(INTERESTS_PATH.is_file(), f"interests JSON not written: {INTERESTS_PATH}")
    reloaded = json.loads(INTERESTS_PATH.read_text(encoding="utf-8"))

    checks = {
        f"interests JSON holds exactly {len(expected_article_ids)} articles": len(
            reloaded
        )
        == len(expected_article_ids),
        f"interests JSON keys match {TARGET_PERSONA}'s article_id set": {
            int(key) for key in reloaded
        }
        == expected_article_ids,
        "every interests value is a list of 1 to 5 non-empty strings": all(
            isinstance(value, list)
            and MIN_INTERESTS <= len(value) <= MAX_INTERESTS
            and all(isinstance(label, str) and label.strip() for label in value)
            for value in reloaded.values()
        ),
        "no vague catch-all labels remain in the interests JSON": not [
            label
            for value in reloaded.values()
            for label in value
            if label.casefold() in VAGUE_LABELS
        ],
        "no duplicate labels within a single interests array": all(
            len({label.casefold() for label in value}) == len(value)
            for value in reloaded.values()
        ),
    }
    for description, passed in checks.items():
        require(bool(passed), description)

    return reloaded


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
    history = history.drop(columns="_source_row")

    require(
        TARGET_PERSONA in set(history["persona_id"]),
        f"{TARGET_PERSONA} is not present in the merged history",
    )
    persona_rows = history[history["persona_id"] == TARGET_PERSONA]
    persona_article_ids = set(persona_rows["article_id"].astype(int))
    print(
        f"{TARGET_PERSONA} (ebnerd_user_id "
        f"{persona_rows['ebnerd_user_id'].iloc[0]}): {len(persona_rows)} history "
        f"rows, {len(persona_article_ids)} unique articles"
    )
    print(f"Extracting with {OLLAMA_MODEL} -> {INTERESTS_PATH.name}")

    extract_persona_interests(persona_rows)
    interests_by_article = {
        int(key): value
        for key, value in validate_interests_json(persona_article_ids).items()
    }

    is_target = history["persona_id"] == TARGET_PERSONA
    history[INTERESTS_COLUMN] = ""
    history.loc[is_target, INTERESTS_COLUMN] = (
        history.loc[is_target, "article_id"]
        .map(interests_by_article)
        .apply(lambda interests: json.dumps(interests, ensure_ascii=False))
    )
    history = history[HISTORY_COLUMNS]

    populated = history[INTERESTS_COLUMN] != ""
    target_history = history[history["persona_id"] == TARGET_PERSONA]

    checks = {
        "history columns are the source columns plus interests": set(
            history.columns
        )
        == {"persona_id", *EBNERD_COLUMNS, INTERESTS_COLUMN},
        "interests column sits immediately after body": list(
            history.columns
        ).index(INTERESTS_COLUMN)
        == list(history.columns).index("body") + 1,
        f"only {TARGET_PERSONA} rows have interests": set(
            history.loc[populated, "persona_id"]
        )
        == {TARGET_PERSONA},
        f"every {TARGET_PERSONA} row has interests": bool(
            (target_history[INTERESTS_COLUMN] != "").all()
        ),
        f"populated row count equals {TARGET_PERSONA} row count": int(
            populated.sum()
        )
        == len(persona_rows),
        "every populated value is a list of 1 to 5 strings": bool(
            history.loc[populated, INTERESTS_COLUMN]
            .apply(json.loads)
            .apply(
                lambda interests: isinstance(interests, list)
                and MIN_INTERESTS <= len(interests) <= MAX_INTERESTS
                and all(isinstance(label, str) and label for label in interests)
            )
            .all()
        ),
        f"{TARGET_PERSONA} rows sharing an article_id have identical interests": bool(
            target_history.groupby("article_id")[INTERESTS_COLUMN]
            .nunique()
            .eq(1)
            .all()
        ),
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
