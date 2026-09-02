"""Subset the ANES 2024 Time Series CSV to the variables needed downstream, draw a party-stratified random sample of respondents, then decode it against the codebook."""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import pandas as pd # type: ignore
from pypdf import PdfReader # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "user_data" / "anes_timeseries_2024_csv"
DEFAULT_INPUT = DATA_DIR / "anes_timeseries_2024_csv_20260519.csv"
DEFAULT_OUTPUT = DATA_DIR / "anes_2024_selected_clean.csv"
DEFAULT_DECODED = DATA_DIR / "anes_2024_selected_decoded.csv"
CODEBOOK_PDF = DATA_DIR / "anes_timeseries_2024_userguidecodebook_20260519.pdf"
DEFAULT_CODEBOOK_JSON = DATA_DIR / "anes_2024_codebook.json"

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

COLUMN_NAMES = {
    ID_COL: "Case ID",
    "V241458x": "Respondent Age on Election Day",
    "V241551": "Respondent Gender",
    "V241488x": "Respondent Occupation Status",
    "V241465x": "Respondent Education Level (5 Category)",
    "V241567x": "Total Household Income (6 Category)",
    "V243001": "Sample State (Postal Abbreviation)",
    "V243007": "Census Region",
    "V243002": "Sample State (FIPS Code)",
    "V241227x": "Party Identification (Summary)",
    "V241177": "Liberal-Conservative Self-Placement (7pt)",
    "V241004": "Frequency of Attention to Politics and Elections",
    "V241005": "Interest in Following Campaigns",
    "V241012": "Registered to Vote",
    "V242010": "Talked to Anyone About Voting for/Against Party or Candidate",
    "V242011": "Attended Online Political Meetings/Rallies/Speeches/Fundraisers",
    "V242012": "Attended In-Person Political Meetings/Rallies/Speeches/Dinners",
    "V242013": "Wore Campaign Button or Displayed Sign/Sticker",
    "V242014": "Did Other Work for Party or Candidate",
    "V242015": "Contributed Money to Individual Candidate",
    "V242020": "Contributed Money to Political Party",
    "V242065": "Voted in 2024 Election",
}

PASSTHROUGH = {ID_COL, "V241458x", "V243001"}

# Read and used for completeness screening, but dropped before writing. These stay
# in REQUIRED deliberately: removing them widens the complete-case pool from 4,130
# to 4,753 respondents and changes all ten sampled Case IDs.
SCREENING_ONLY = [
    "V241004",
    "V241005",
    "V241012",
    "V242010",
    "V242011",
    "V242012",
    "V242013",
    "V242014",
    "V242015",
    "V242020",
    "V242065",
]

SAMPLE_SIZE = 10
RANDOM_SEED = 2024

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st"}
PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "‐": "-", "‑": "-", "–": "-", "—": "-", " ": " "})
VARIABLE_HEADER = re.compile(r"^(V\d{6}[a-z]?)(?=[A-Z])(.*)$")
SECTION_HEADER = re.compile(
    r"^(Value Labels|Survey Question\(s\)|Question|Universe|Randomization|Response Order|Variable Note|Note:|Source)\s*(.*)$"
)
VALUE_LABEL = re.compile(r"(?:^|(?<=\s{2}))(-?\d+)\.\s+(?=\S)")
BANNERS = ("VARIABLES", "USER GUIDE", "CODEBOOK", "APPENDIX")
INTEGER = re.compile(r"^[+-]?\d+$")

def normalize(text: str) -> str:
    for ligature, plain in LIGATURES.items():
        text = text.replace(ligature, plain)
    return text.translate(PUNCTUATION)

def is_page_furniture(line: str) -> bool:
    banner = re.sub(r"^\d+\s+|\s+\d+$", "", line).strip()
    return bool(banner) and banner == banner.upper() and banner.endswith(BANNERS)

def codebook_lines(pdf_path: Path):
    for page in PdfReader(str(pdf_path)).pages:
        for line in normalize(page.extract_text() or "").splitlines():
            if line.strip():
                yield line.strip()

def add_value_labels(entry: dict, text: str, continuation: bool = False) -> None:
    hits = list(VALUE_LABEL.finditer(text))
    if not hits:
        if continuation and entry["value_labels"]:
            last = next(reversed(entry["value_labels"]))
            entry["value_labels"][last] = f"{entry['value_labels'][last]} {text}".strip()
        return
    for i, match in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        entry["value_labels"][match.group(1)] = text[match.end():end].strip()

def parse_codebook(pdf_path: Path, wanted: set[str]) -> dict[str, dict]:
    variables: dict[str, dict] = {}
    entry: dict | None = None
    section: str | None = None
    for line in codebook_lines(pdf_path):
        header = VARIABLE_HEADER.match(line)
        if header:
            if header.group(1) not in wanted:
                entry = None
                continue
            entry = variables.setdefault(header.group(1), {"label": "", "value_labels": {}})
            entry["label"] = header.group(2).strip()
            section = "Title"
            continue
        if entry is None or is_page_furniture(line):
            continue
        heading = SECTION_HEADER.match(line)
        if heading:
            section, body = heading.group(1), heading.group(2)
            if section == "Value Labels" and body:
                add_value_labels(entry, body)
        elif section == "Value Labels":
            add_value_labels(entry, line, continuation=True)
        elif section == "Title":
            joiner = "" if entry["label"].endswith("-") else " "
            entry["label"] = f"{entry['label']}{joiner}{line}".strip()
    return variables

def build_codebook(pdf_path: Path) -> dict:
    parsed = parse_codebook(pdf_path, set(COLUMNS))
    undocumented = [v for v in COLUMNS if v not in PASSTHROUGH and not parsed.get(v, {}).get("value_labels")]
    if undocumented:
        raise SystemExit(f"no value labels found in {pdf_path.name} for: {', '.join(undocumented)}")
    return {
        "source": pdf_path.name,
        "variables": {
            variable: {
                "column": COLUMN_NAMES[variable],
                "label": parsed.get(variable, {}).get("label", ""),
                "description": COLUMNS[variable],
                "passthrough": variable in PASSTHROUGH,
                "value_labels": parsed.get(variable, {}).get("value_labels", {}),
            }
            for variable in COLUMNS
        },
    }

def decode_sample(sample: pd.DataFrame, codebook: dict) -> pd.DataFrame:
    decoded: dict[str, list[str]] = {}
    unmapped: set[str] = set()
    for variable in sample.columns:
        entry = codebook["variables"][variable]
        labels = entry["value_labels"]
        column = []
        for case_id, raw in zip(sample[ID_COL], sample[variable]):
            value = raw.strip()
            key = str(int(value)) if INTEGER.match(value) else value
            if not value:
                column.append("")
            elif key in labels:
                column.append(labels[key])
            elif entry["passthrough"]:
                column.append(value)
            else:
                unmapped.add(f"{variable}={value} (case {case_id})")
                column.append(value)
        decoded[entry["column"]] = column
    if unmapped:
        raise SystemExit("values absent from the codebook mappings:\n  " + "\n  ".join(sorted(unmapped)))
    return pd.DataFrame(decoded, dtype=str)

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
    parser.add_argument("--codebook-json", type=Path, default=DEFAULT_CODEBOOK_JSON)
    args = parser.parse_args()

    for path in (args.input, args.codebook):
        if not path.exists():
            parser.error(f"input file not found: {path}")

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
    emitted = sample.drop(columns=SCREENING_ONLY)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    emitted.to_csv(args.output, index=False)

    args.codebook_json.parent.mkdir(parents=True, exist_ok=True)
    args.codebook_json.write_text(
        json.dumps(build_codebook(args.codebook), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    mappings = json.loads(args.codebook_json.read_text(encoding="utf-8"))
    decoded = validate_decoded(decode_sample(emitted, mappings), emitted)

    args.decoded_output.parent.mkdir(parents=True, exist_ok=True)
    decoded.to_csv(args.decoded_output, index=False)

    return 0

if __name__ == "__main__":
    sys.exit(main())