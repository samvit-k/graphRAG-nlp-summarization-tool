"""Convert the MIND small dev TSV files (behaviors, news) to headed CSVs verbatim."""

from __future__ import annotations
import csv
import sys
from pathlib import Path
import pandas as pd # type: ignore

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "user_data" / "MIND_dataset" / "MIND_small_dev"
OUTPUT_DIR = REPOSITORY_ROOT / "user_data" / "MIND_clean"

BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

INPUTS = {
    "behaviors.csv": (DATA_DIR / "behaviors.tsv", BEHAVIORS_COLUMNS),
    "news.csv": (DATA_DIR / "news.tsv", NEWS_COLUMNS),
}

def convert(source: Path, destination: Path, columns: list[str]) -> None:
    frame = pd.read_csv(
        source,
        sep="\t",
        header=None,
        names=columns,
        dtype=str,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    frame.to_csv(destination, index=False)
    print(
        f"{source.name} -> {destination.relative_to(REPOSITORY_ROOT)}: "
        f"{len(frame):,} rows x {frame.shape[1]} columns"
    )

def main() -> int:
    missing = [str(path) for path, _ in INPUTS.values() if not path.exists()]
    if missing:
        print("Missing input files:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (source, columns) in INPUTS.items():
        convert(source, OUTPUT_DIR / name, columns)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())