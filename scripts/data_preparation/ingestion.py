"""Convert the ReCOVery news CSV into one .txt document per article for GraphRAG ingestion."""

import ast
import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "topic_data" / "recovery" / "recovery_news_data_cleaned.csv"
OUTPUT_DIR = ROOT / "graphRAG" / "input"

TEXT_COLUMN = "body_text"
METADATA_COLUMNS = [
    ("document_id", "Document ID"),
    ("news_id", "News ID"),  
    ("title", "Title"),
    ("author", "Author"),
    ("publisher", "Publisher"),
    ("publish_date", "Publish Date"),
    ("url", "URL"),
    ("country", "Country"),
    ("political_bias", "Political Bias"),
    ("reliability", "Reliability"),
]

def clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()

def format_authors(value: str) -> str:
    """Render list-like author cells (e.g. "['A', 'B']") as a plain comma-separated string."""
    value = clean(value)
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        if isinstance(parsed, (list, tuple)):
            return ", ".join(clean(str(item)) for item in parsed if clean(str(item)))
    return value

def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", clean(text).lower()).strip("-")
    return slug[:max_length].strip("-")

def build_filename(row: dict[str, str], body: str) -> str:
    """Stable name derived from the row's identifier (or a body hash) plus a title slug."""
    identifier = clean(row.get("document_id", "")) or clean(row.get("news_id", ""))
    if not identifier:
        identifier = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    slug = slugify(row.get("title", ""))
    return f"{identifier}-{slug}.txt" if slug else f"{identifier}.txt"

def build_document(row: dict[str, str], body: str) -> str:
    lines = []
    for column, label in METADATA_COLUMNS:
        value = format_authors(row[column]) if column == "author" else clean(row.get(column, ""))
        if value:
            lines.append(f"{label}: {value}")
    lines.extend(["", body])
    return "\n".join(lines) + "\n"

def main() -> None:
    if not CSV_PATH.is_file():
        sys.exit(f"CSV not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)

    processed = created = skipped = 0
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or TEXT_COLUMN not in reader.fieldnames:
            sys.exit(f"Required column {TEXT_COLUMN!r} missing from {CSV_PATH}")

        for row in reader:
            processed += 1
            body = clean(row.get(TEXT_COLUMN, ""))
            if not body:
                skipped += 1
                continue
            path = OUTPUT_DIR / build_filename(row, body)
            path.write_text(build_document(row, body), encoding="utf-8")
            created += 1

    print(f"Rows processed: {processed}")
    print(f"Files created:  {created}")
    print(f"Rows skipped:   {skipped}")
    print(f"Output dir:     {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
