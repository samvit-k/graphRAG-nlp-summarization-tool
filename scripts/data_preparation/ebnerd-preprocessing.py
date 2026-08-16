"""Convert the EB-NeRD demo Parquet files (articles, history, behaviors) to CSV verbatim."""

from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "user-data" / "ebnerd_demo"
OUTPUT_DIR = REPO_ROOT / "user-data" / "ebnerd_csv"

INPUTS = {
    "articles.csv": DATA_DIR / "articles.parquet",
    "history.csv": DATA_DIR / "train" / "history.parquet",
    "behaviors.csv": DATA_DIR / "train" / "behaviors.parquet",
}

def to_python(value):
    if isinstance(value, np.ndarray):
        return [to_python(v) for v in value.tolist()]
    if isinstance(value, list):
        return [to_python(v) for v in value]
    if isinstance(value, np.generic):
        return to_python(value.item())
    if isinstance(value, dt.datetime):
        return str(pd.Timestamp(value))
    return value

def convert(source: Path, destination: Path) -> None:
    df = pd.read_parquet(source, engine="pyarrow")
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(to_python)
    df.to_csv(destination, index=False)
    print(f"{source.name} -> {destination.relative_to(REPO_ROOT)}: {len(df):,} rows x {df.shape[1]} columns")

def main() -> int:
    missing = [str(p) for p in INPUTS.values() if not p.exists()]
    if missing:
        raise SystemExit("input file not found: " + ", ".join(missing))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in INPUTS.items():
        convert(source, OUTPUT_DIR / name)
    return 0

if __name__ == "__main__":
    sys.exit(main())
