"""Write a checksum table for a folder of exported files (e.g. the Sentinel-2 GeoTIFF exports).

The large rasters are kept outside Git (see data/README.md). This table is what goes INTO Git
instead: it lets anyone confirm that the file they hold is byte-for-byte the file an experiment
used. If one byte differs, the SHA-256 differs.

Usage:
    python make_checksums.py <folder> --out checksums.csv
    python make_checksums.py D:/SDP/exports --out ../../data/checksums_exports.csv --pattern "*.tif"

Output CSV, one row per file, sorted by name so the table is the same on every run:
    filename     path relative to <folder>, with forward slashes
    size_bytes   file size in bytes
    sha256       SHA-256 of the file contents (64 hex characters)

Notes
  * Sub-folders are included. --pattern filters by file name (default: every file).
  * The output CSV itself is skipped if it sits inside <folder>.
  * Files are read in 1 MB pieces, so multi-gigabyte rasters do not need to fit in memory.
  * Nothing is modified, moved or deleted. Only standard-library modules are used.
  * To check one file by hand:  certutil -hashfile <file> SHA256   (Windows)
                                sha256sum <file>                   (Linux, macOS, Git Bash)
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import sys
from pathlib import Path

FIELDS = ["filename", "size_bytes", "sha256"]
CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def list_files(folder: Path, pattern: str = "*", skip: Path | None = None) -> list[Path]:
    skip = skip.resolve() if skip else None
    files = [p for p in folder.rglob("*")
             if p.is_file() and fnmatch.fnmatch(p.name, pattern) and p.resolve() != skip]
    return sorted(files, key=lambda p: p.relative_to(folder).as_posix())


def checksum_rows(folder: Path, pattern: str = "*", skip: Path | None = None, verbose: bool = False) -> list[dict]:
    rows = []
    for p in list_files(folder, pattern, skip):
        rows.append({"filename": p.relative_to(folder).as_posix(), "size_bytes": p.stat().st_size,
                     "sha256": sha256_of(p)})
        if verbose:
            print(f"  {rows[-1]['sha256'][:12]}...  {rows[-1]['size_bytes']:>14,}  {rows[-1]['filename']}", flush=True)
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder of exported files (sub-folders are included)")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--pattern", default="*", help='file-name filter, e.g. "*.tif" (default: every file)')
    ap.add_argument("--quiet", action="store_true", help="do not print one line per file")
    a = ap.parse_args(argv)

    folder, out = Path(a.folder), Path(a.out)
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")
    rows = checksum_rows(folder, a.pattern, skip=out, verbose=not a.quiet)
    if not rows:
        sys.exit(f"no files matching {a.pattern!r} in {folder}; nothing written")
    write_csv(rows, out)
    print(f"{len(rows)} files, {sum(r['size_bytes'] for r in rows):,} bytes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
