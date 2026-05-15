"""
hash_splits.py — Compute SHA256 hashes for all split and window parquet files.

Registers Phase 2 data files before they enter the pipeline.
Per CLAUDE.md: unregistered data cannot enter the pipeline.
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
RAW_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "file_hashes_phase2.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 60)
    print("PHASE 2 DATA HASH REGISTRATION")
    print("=" * 60)

    results = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "phase": "2",
        "splits": {},
        "windows": {},
        "aadt": {},
    }

    # Hash split parquets
    print("\nSplit parquets:")
    for name in ["train", "val", "b1", "b2", "b4"]:
        path = SPLITS_DIR / f"{name}.parquet"
        if path.exists():
            h = sha256(path)
            size_mb = path.stat().st_size / (1024 * 1024)
            results["splits"][name] = {"sha256": h, "size_mb": round(size_mb, 2)}
            print(f"  {name:20s} {h[:16]}... ({size_mb:.1f} MB)")

    # Hash window parquets
    print("\nWindow parquets:")
    for name in ["train", "val", "b1", "b2", "b4"]:
        path = SPLITS_DIR / f"{name}_windows.parquet"
        if path.exists():
            h = sha256(path)
            size_mb = path.stat().st_size / (1024 * 1024)
            results["windows"][name] = {"sha256": h, "size_mb": round(size_mb, 2)}
            print(f"  {name:20s} {h[:16]}... ({size_mb:.1f} MB)")

    # Hash AADT files
    print("\nAADT files:")
    for name in ["aadt_limited.csv", "aadt_other.csv"]:
        path = RAW_DIR / name
        if path.exists():
            h = sha256(path)
            size_mb = path.stat().st_size / (1024 * 1024)
            results["aadt"][name] = {"sha256": h, "size_mb": round(size_mb, 2)}
            print(f"  {name:20s} {h[:16]}... ({size_mb:.1f} MB)")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nHashes saved to {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()