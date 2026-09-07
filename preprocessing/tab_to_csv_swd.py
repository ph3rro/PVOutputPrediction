#!/usr/bin/env python3
"""Extract Date/Time and SWD columns from a PANGAEA .tab file into a CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def find_header_line(path: Path) -> tuple[int, list[str]]:
    """Return the 0-based index and column names of the data header row."""
    with path.open("r", encoding="utf-8", newline="") as f:
        for i, line in enumerate(f):
            if line.startswith("/*") or line.startswith("*/") or not line.strip():
                continue
            if "Date/Time" in line and "SWD" in line:
                cols = line.rstrip("\n").split("\t")
                return i, cols
    raise ValueError(f"Could not find data header with Date/Time and SWD in {path}")


def ensure_seconds(timestamp: str) -> str:
    """Return the timestamp with a seconds component.

    PANGAEA timestamps are formatted like ``2022-08-01T05:45`` (no seconds).
    Downstream parsing expects ``%Y-%m-%dT%H:%M:%S``, so append ``:00`` when
    the time component only has hours and minutes. Timestamps that already
    include seconds are returned unchanged.
    """
    timestamp = timestamp.strip()
    if "T" not in timestamp:
        return timestamp
    date_part, _, time_part = timestamp.partition("T")
    if time_part.count(":") == 1:
        time_part = f"{time_part}:00"
    return f"{date_part}T{time_part}"


def tab_to_csv(input_path: Path, output_path: Path) -> int:
    header_idx, cols = find_header_line(input_path)

    datetime_idx = next(i for i, c in enumerate(cols) if c.startswith("Date/Time"))
    swd_idx = next(i for i, c in enumerate(cols) if c.startswith("SWD"))

    n_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as infile, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Date/Time", "SWD"])

        for i, line in enumerate(infile):
            if i <= header_idx:
                continue
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            writer.writerow([ensure_seconds(parts[datetime_idx]), parts[swd_idx]])
            n_rows += 1

    return n_rows


def main() -> None:
    default_in = Path(
        __file__
    ).resolve().parent / "data/raw_data/pangaea/PVotSky/PVot_MET_20220801_20240731_irradiance_meteorology.tab"
    default_out = default_in.with_name(default_in.stem + "_SWD.csv")

    parser = argparse.ArgumentParser(
        description="Convert a PANGAEA .tab file to CSV with Date/Time and SWD columns."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=default_in,
        help=f"Input .tab file (default: {default_in})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output CSV path (default: <input_stem>_SWD.csv next to input)",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name(input_path.stem + "_SWD.csv")

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    n_rows = tab_to_csv(input_path, output_path)
    print(f"Wrote {n_rows:,} rows to {output_path}")


if __name__ == "__main__":
    main()
