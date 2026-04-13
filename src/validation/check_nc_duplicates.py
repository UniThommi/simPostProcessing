"""check_nc_duplicates.py

Checks whether any NC identified by (muon_id, nc_id) appears in more than one
output_t*.hdf5 file within a Geant4/REMAGE run directory.

Usage
-----
    python check_nc_duplicates.py <run_dir> [<run_dir> ...]

Each positional argument is a run directory that contains output_t*.hdf5 files.
The script reports, per run:
  - how many files were scanned
  - how many NCs each file contains
  - which (muon_id, nc_id) pairs appear in more than one file
  - a cross-file matrix showing which file pairs share duplicates

Exit code is 0 if no duplicates were found in any run, 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import h5py
import numpy as np


# ── HDF5 helpers ──────────────────────────────────────────────────────

def _read_pages(group: h5py.Group, field: str) -> np.ndarray:
    return group[field]["pages"][:]


def _load_nc_ids(fpath: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (muon_id, nc_id) arrays from one output_t*.hdf5, or None if empty."""
    with h5py.File(fpath, "r") as f:
        grp = f["hit"]["MyNeutronCaptureOutput"]
        if int(grp["entries"][()]) == 0:
            return None
        muon_ids = _read_pages(grp, "evtid")
        nc_ids   = _read_pages(grp, "nC_track_id")
    return muon_ids, nc_ids


# ── Per-run analysis ──────────────────────────────────────────────────

def check_run(run_dir: str) -> int:
    """Check one run directory. Returns the number of duplicate NC pairs found."""
    files = sorted(glob.glob(os.path.join(run_dir, "output_t*.hdf5")))
    if not files:
        print(f"  [WARN] No output_t*.hdf5 found in {run_dir!r}")
        return 0

    print(f"  Scanning {len(files)} file(s) ...")

    # Map each (muon_id, nc_id) → list of file indices where it appears
    pair_to_files: dict[tuple[int, int], list[int]] = defaultdict(list)
    file_nc_counts: list[int] = []

    for file_idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        result = _load_nc_ids(fpath)
        if result is None:
            print(f"    [{file_idx}] {fname}: empty (0 NCs)")
            file_nc_counts.append(0)
            continue

        muon_ids, nc_ids = result
        n = len(muon_ids)
        file_nc_counts.append(n)
        print(f"    [{file_idx}] {fname}: {n:,} NCs")

        for m, c in zip(muon_ids.tolist(), nc_ids.tolist()):
            pair_to_files[(m, c)].append(file_idx)

    # Find pairs that appear in more than one file
    duplicates = {
        pair: idxs
        for pair, idxs in pair_to_files.items()
        if len(idxs) > 1
    }

    total_ncs = sum(file_nc_counts)
    unique_ncs = len(pair_to_files)
    n_dup_pairs = len(duplicates)
    n_dup_rows  = total_ncs - unique_ncs  # extra rows from duplicates

    print(f"\n  Total NC rows across all files : {total_ncs:,}")
    print(f"  Unique (muon_id, nc_id) pairs  : {unique_ncs:,}")
    print(f"  Duplicate pairs                : {n_dup_pairs:,}")
    print(f"  Redundant rows (to be dropped) : {n_dup_rows:,}")

    if n_dup_pairs == 0:
        print("  --> No duplicates found.")
        return 0

    # Cross-file matrix: how many duplicates are shared between each pair of files
    n_files = len(files)
    cross: np.ndarray = np.zeros((n_files, n_files), dtype=np.int64)
    for idxs in duplicates.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                cross[idxs[i], idxs[j]] += 1
                cross[idxs[j], idxs[i]] += 1

    print("\n  Cross-file duplicate counts (file_i × file_j):")
    # Header
    col_w = max(len(os.path.basename(f)) for f in files) + 2
    idx_w = len(str(n_files - 1)) + 1
    header = " " * (idx_w + 2) + "  ".join(
        f"[{i}]".rjust(6) for i in range(n_files)
    )
    print(f"  {header}")
    for i in range(n_files):
        row_cells = "  ".join(
            str(cross[i, j]).rjust(6) if j != i else "     -"
            for j in range(n_files)
        )
        print(f"  [{i}] {row_cells}   {os.path.basename(files[i])}")

    # Show a sample of duplicate pairs (up to 10)
    sample = list(duplicates.items())[:10]
    print(f"\n  Sample duplicate pairs (muon_id, nc_id) → file indices:")
    for (m, c), idxs in sample:
        file_names = ", ".join(os.path.basename(files[i]) for i in idxs)
        print(f"    ({m}, {c}) in files: {file_names}")
    if n_dup_pairs > 10:
        print(f"    ... and {n_dup_pairs - 10} more.")

    return n_dup_pairs


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check for cross-file NC duplicates in Geant4 run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        metavar="run_dir",
        help="One or more run directories containing output_t*.hdf5 files.",
    )
    args = parser.parse_args()

    any_duplicates = False
    for run_dir in args.run_dirs:
        run_dir = os.path.abspath(run_dir)
        print(f"\n{'='*60}")
        print(f"Run: {run_dir}")
        print(f"{'='*60}")
        if not os.path.isdir(run_dir):
            print(f"  [ERROR] Not a directory: {run_dir!r}")
            any_duplicates = True
            continue
        n_dups = check_run(run_dir)
        if n_dups > 0:
            any_duplicates = True

    print()
    if any_duplicates:
        print("RESULT: duplicates found in one or more runs.")
        sys.exit(1)
    else:
        print("RESULT: no duplicates found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
