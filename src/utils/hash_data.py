"""
Computes SHA256 hashes and row counts for the Vermont crash master CSV
and all yearly split files. Verifies that yearly row totals match the master.
Outputs results to data/raw/file_hashes.json.

Usage:
    python -m src.utils.hash_data
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
YEARLY_DIR = RAW_DIR / "yearly"
MASTER_FILE = RAW_DIR / "vt_crashes_all.csv"
OUTPUT_FILE = RAW_DIR / "file_hashes.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f) - 1


def main() -> None:
    if not MASTER_FILE.exists():
        print(f"ERROR: master file not found: {MASTER_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Hashing {MASTER_FILE.name} ...")
    master_hash = sha256(MASTER_FILE)
    master_rows = count_rows(MASTER_FILE)
    print(f"  sha256 : {master_hash}")
    print(f"  rows   : {master_rows:,}")

    yearly = {}
    if YEARLY_DIR.exists():
        yearly_files = sorted(YEARLY_DIR.glob("vt_crashes_*.csv"))
        print(f"\nHashing {len(yearly_files)} yearly files ...")
        for f in yearly_files:
            file_hash = sha256(f)
            rows = count_rows(f)
            yearly[f.name] = {"sha256": file_hash, "rows": rows}
            print(f"  {f.name}: {rows:,} rows")
    else:
        print(f"\nWARNING: yearly/ directory not found at {YEARLY_DIR}", file=sys.stderr)

    yearly_total = sum(v["rows"] for v in yearly.values())
    rows_match = yearly_total == master_rows

    print(f"\nRow count verification:")
    print(f"  master rows     : {master_rows:,}")
    print(f"  yearly rows sum : {yearly_total:,}")
    print(f"  result          : {'PASS' if rows_match else 'FAIL'}")

    output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "verification": {
            "master_rows": master_rows,
            "yearly_rows_sum": yearly_total,
            "rows_match": rows_match,
        },
        "master": {
            "file": MASTER_FILE.name,
            "sha256": master_hash,
            "rows": master_rows,
        },
        "yearly": yearly,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to: {OUTPUT_FILE}")

    if not rows_match:
        print(
            f"ERROR: row count mismatch ({yearly_total:,} yearly != {master_rows:,} master). "
            "Yearly files may be incomplete or overlapping.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
