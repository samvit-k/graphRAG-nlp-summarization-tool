"""Subset the ANES 2024 Time Series CSV to the variables needed downstream, draw a party-stratified random sample of respondents, then decode it against the codebook."""

from __future__ import annotations
import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
import anthropic
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "user-data" / "anes_timeseries_2024_csv"
DEFAULT_INPUT = DATA_DIR / "anes_timeseries_2024_csv_20260519.csv"
DEFAULT_OUTPUT = DATA_DIR / "anes_2024_selected_clean.csv"
DEFAULT_DECODED = DATA_DIR / "anes_2024_selected_decoded.csv"
CODEBOOK_PDF = DATA_DIR / "anes_timeseries_2024_userguidecodebook_20260519.pdf"

ANES_MISSING_CODES = {
    "-1": "Inapplicable",
    "-2": "Other missing / DK-RF in a source item of a summary variable",
    "-3": "Restricted access",
    "-4": "Error",
    "-5": "Sufficient partial, breakoff",
    "-6": "No post-election interview",
    "-7": "Insufficient partial, interview deleted",
    "-8": "Don't know",
    "-9": "Refused",
}

ID_COL = "V240001"

PROFILE_REQUIRED = {
    "V241458x": "PRE: SUMMARY: Respondent age on election day (80 = 80 or older)",
    "V241551": "PRE: What is R's gender?",
    "V241488x": "PRE: SUMMARY: R occupation status, 1 digit",
    "V241465x": "PRE: SUMMARY: Respondent education, 5 category",
    "V241567x": "PRE: SUMMARY: Total (household) income",
    "V243001": "SAMPLE: Sample location state postal abbreviation",
    "V243007": "SAMPLE: Census region",
}

PROFILE_OPTIONAL = {
    "V243002": "SAMPLE: Sample location FIPS state code (redundant with V243001)",
}

POLITICAL_REQUIRED = {
    "V241227x": "PRE: SUMMARY: Party ID (7 point)",
    "V241177": "PRE: 7pt scale liberal-conservative self-placement (99 = haven't thought much)",
    "V241004": "PRE: How often does R pay attention to politics and elections",
    "V241005": "PRE: How interested in following campaigns",
    "V241012": "PRE: Is R registered to vote",
    "V242010": "POST: R talk to anyone about voting for or against a party/candidate",
    "V242011": "POST: R attend online political meetings, rallies, speeches",
    "V242012": "POST: R go to any political meetings, rallies, speeches",
    "V242013": "POST: R wear campaign button or post sign or bumper sticker",
    "V242014": "POST: R do any (other) work for party or candidate",
    "V242015": "POST: Did R contribute money to individual candidate",
    "V242020": "POST: R contribute money to political party",
    "V242065": "POST: Did R vote in the 2024 election",
}

COLUMNS = {ID_COL: "Case ID", **PROFILE_REQUIRED, **PROFILE_OPTIONAL, **POLITICAL_REQUIRED}

REQUIRED = {**PROFILE_REQUIRED, **POLITICAL_REQUIRED}

SAMPLE_SIZE = 10
RANDOM_SEED = 2024

MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000
DECODED_SCHEMA = {
    "type": "object",
    "properties": {
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    },
    "required": ["columns", "rows"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = """You are decoding a small subset of the ANES 2024 Time Series dataset using the official ANES 2024 codebook provided with the request.

Your task is strictly limited to translating ANES variable IDs and coded values into their documented human-readable meanings.

For each column in the provided CSV:

1. Find the exact variable in the ANES 2024 codebook.
2. Replace the cryptic ANES variable ID with a concise, readable column name based on the codebook definition.
3. Replace each coded categorical value with its exact documented response label from the codebook.
4. Leave inherently numeric values, such as respondent age, as numeric values when the codebook indicates that the number itself is the substantive value.
5. Preserve meaningful special responses exactly as documented. For example, do not reinterpret values such as 'Haven't thought much about this' as neutral or missing.
6. Do not infer, guess, summarize, normalize, or invent any respondent information.
7. Do not change the selected respondents or add/remove rows.
8. Do not use general political knowledge to interpret a code. Use only the supplied ANES codebook.
9. If a variable or value cannot be confidently matched to the codebook, preserve the original ANES variable name or coded value rather than guessing.

Return only the fully decoded CSV with the same 10 rows and the same number of columns as the input. Do not include explanations, markdown, commentary, or additional text."""

def load_api_key() -> str:
    load_dotenv(ENV_PATH)
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()

def build_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=load_api_key())

def codebook_excerpt(pdf_path: Path, variables: list[str]) -> bytes:
    reader = PdfReader(str(pdf_path))
    wanted: set[int] = set()
    for i, page in enumerate(reader.pages):
        if any(v in (page.extract_text() or "") for v in variables):
            wanted.update((i, i + 1))
    writer = PdfWriter()
    for i in sorted(wanted):
        if i < len(reader.pages):
            writer.add_page(reader.pages[i])
    if not len(writer.pages):
        raise SystemExit(f"no codebook pages matched the selected variables in {pdf_path.name}")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()

def decode_sample(client: anthropic.Anthropic, csv_text: str, pdf_bytes: bytes) -> pd.DataFrame:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": DECODED_SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "document", "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(pdf_bytes).decode(),
            }},
            {"type": "text", "text":
                "Decode the CSV below. Return the decoded table as JSON matching the given "
                "schema: `columns` holds the readable column names in the original order, "
                "and `rows` holds one array of decoded values per respondent.\n\n" + csv_text},
        ]}],
    )
    if response.stop_reason == "max_tokens":
        raise SystemExit(f"decoded table was truncated at MAX_TOKENS={MAX_TOKENS}; raise it and rerun")
    payload = json.loads("".join(b.text for b in response.content if b.type == "text"))
    return pd.DataFrame(payload["rows"], columns=payload["columns"], dtype=str)

def validate_decoded(decoded: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    if decoded.shape != sample.shape:
        raise SystemExit(f"decoded table is {decoded.shape}, expected {sample.shape}")
    ids = decoded.iloc[:, 0].tolist()
    if ids != sample[ID_COL].tolist():
        raise SystemExit(f"decoded case IDs do not match the selected respondents: {ids}")
    return decoded

def is_missing(series: pd.Series) -> pd.Series:
    stripped = series.str.strip()
    return stripped.eq("") | stripped.isin(ANES_MISSING_CODES)

def audit(df: pd.DataFrame, title: str) -> None:
    print(title)
    for col in df.columns:
        stripped = df[col].str.strip()
        flags = is_missing(df[col])
        counts = stripped[flags].value_counts()
        detail = ", ".join(f"{code or '(blank)'}={n}" for code, n in counts.items()) or "-"
        print(f"  {col:<10} {flags.sum():>8,} {flags.mean():>6.1%}  {detail}")

def party_group(df: pd.DataFrame) -> pd.Series:
    pid = pd.to_numeric(df["V241227x"])
    return pid.map(lambda v: "dem" if v <= 3 else "ind" if v == 4 else "rep")

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    groups = party_group(df)
    strata = ["dem", "ind", "rep"]
    quota = {g: n // len(strata) + (i < n % len(strata)) for i, g in enumerate(strata)}
    picks = [df.loc[groups == g].sample(quota[g], random_state=seed) for g in strata]
    return pd.concat(picks).sort_index()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decoded-output", type=Path, default=DEFAULT_DECODED)
    parser.add_argument("--codebook", type=Path, default=CODEBOOK_PDF)
    args = parser.parse_args()

    for path in (args.input, args.codebook):
        if not path.exists():
            parser.error(f"input file not found: {path}")

    if not load_api_key():
        parser.error(f"ANTHROPIC_API_KEY is not set (checked env and {ENV_PATH})")

    header = pd.read_csv(args.input, nrows=0, encoding="utf-8-sig").columns
    absent = [c for c in COLUMNS if c not in header]
    if absent:
        parser.error(f"columns absent from {args.input.name}: {', '.join(absent)}")

    df = pd.read_csv(
        args.input,
        usecols=list(COLUMNS),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )[list(COLUMNS)]

    if df[ID_COL].duplicated().any():
        raise SystemExit(f"{ID_COL} is not unique; refusing to continue")

    audit(df, "Missingness before filtering")

    incomplete = pd.DataFrame({c: is_missing(df[c]) for c in REQUIRED}).any(axis=1)
    clean = df.loc[~incomplete].reset_index(drop=True)

    audit(clean, "Missingness after filtering")

    sample = stratified_sample(clean, SAMPLE_SIZE, RANDOM_SEED)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    print(f"Wrote coded sample: {args.output}")

    pdf_bytes = codebook_excerpt(args.codebook, list(COLUMNS))
    decoded = validate_decoded(
        decode_sample(build_client(), args.output.read_text(encoding="utf-8"), pdf_bytes),
        sample,
    )

    args.decoded_output.parent.mkdir(parents=True, exist_ok=True)
    decoded.to_csv(args.decoded_output, index=False)
    print(f"Wrote decoded sample: {args.decoded_output}")

    return 0

if __name__ == "__main__":
    sys.exit(main())