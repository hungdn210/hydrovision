"""
Extract only horizon_1 column from every station_predictions CSV and write
to a parallel station_predictions_h1 folder.  Run once locally, then upload
the result to R2.

Usage:
    python scripts/extract_h1_predictions.py [--data-dir PATH]

The output folder mirrors the exact directory structure of station_predictions
so the app can load it with minimal code changes.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def extract(src_root: Path, dst_root: Path) -> None:
    csv_files = list(src_root.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files under {src_root}")

    for src in csv_files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        with src.open(newline="") as f_in:
            reader = csv.reader(f_in)
            rows = list(reader)

        if not rows:
            continue

        header = rows[0]
        # Find horizon_1 column index (first column, or column named 'horizon_1')
        h1_idx = 0
        if "horizon_1" in header:
            h1_idx = header.index("horizon_1")

        with dst.open("w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow([header[h1_idx]])       # header: just 'horizon_1'
            for row in rows[1:]:
                if row:
                    writer.writerow([row[h1_idx]])  # one value per window

    print(f"Done. h1-only files written to {dst_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract horizon_1 from station_predictions CSVs")
    parser.add_argument("--data-dir", default="data", help="Root data directory (default: data/)")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    for dataset in ["Mekong", "LamaH"]:
        src = data_root / dataset / "prediction_results" / "station_predictions"
        dst = data_root / dataset / "prediction_results" / "station_predictions_h1"
        if not src.exists():
            print(f"Skipping {src} (not found)")
            continue
        if dst.exists():
            print(f"Removing existing {dst}")
            shutil.rmtree(dst)
        extract(src, dst)


if __name__ == "__main__":
    main()
