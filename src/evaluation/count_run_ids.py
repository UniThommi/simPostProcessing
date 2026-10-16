#!/usr/bin/env python3
# Author: Thomas Buerger (University of Tübingen)
"""
Count the NC events per run_id in a post-processed HDF5 file.

Quick check of which simulation runs ended up in an output file and how many
NCs each contributes (e.g. after selecting runs with --sim3-runs or when runs
without optical data were dropped).

Works with the output of all post-processing scripts:
  - zylFullPostprocessingOptimized.py      (event_id_columns = ["run_id", "nc_id"])
  - musun2SimFullPostprocessing.py         (event_id_columns = ["run_id", "muon_id", "nc_id"])
  - musun2SimMultiOpticalPostprocessing.py (same as musun2Sim)

The position of the run_id column is taken from the 'event_id_columns'
metadata, so the script does not depend on the number of columns.

Usage:
    python src/evaluation/count_run_ids.py --data /path/to/ncscore_output_0.hdf5
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def count_run_ids(filepath, chunk_size: int = 1_000_000):
    """
    Read the run_id column of 'event_ids' in chunks.

    Returns:
        (Counter {run_id: number of events}, total number of events)
    """
    with h5py.File(filepath, "r") as f:
        if "event_ids" not in f:
            raise KeyError(
                f"'event_ids' dataset missing in {filepath}. "
                f"Is this really a post-processing output?"
            )

        # Column index of run_id (robust against the 2- vs. 3-column format)
        if "event_id_columns" in f:
            columns = [_decode(c) for c in f["event_id_columns"][:]]
            if "run_id" not in columns:
                raise KeyError(
                    f"'run_id' not found in event_id_columns {columns}."
                )
            run_id_col = columns.index("run_id")
        else:
            # Fallback: run_id is the first column in all scripts
            run_id_col = 0
            print(
                "  Warning: 'event_id_columns' missing, assuming run_id = column 0."
            )

        event_ids = f["event_ids"]
        n_events = event_ids.shape[0]

        run_id_counts = Counter()
        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            chunk = event_ids[start:end, run_id_col]
            unique, counts = np.unique(chunk, return_counts=True)
            for u, c in zip(unique, counts):
                run_id_counts[int(u)] += int(c)

    return run_id_counts, n_events


def main():
    parser = argparse.ArgumentParser(
        description="Count the NC events per run_id in a post-processed HDF5 file."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Post-processed HDF5 file (ncscore_output_*.hdf5).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Rows read at once (default: 1_000_000).",
    )
    args = parser.parse_args()

    filepath = Path(args.data)
    if not filepath.is_file():
        print(f"ERROR: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    print(f"File: {filepath}")
    run_id_counts, n_events = count_run_ids(filepath, chunk_size=args.chunk_size)

    n_unique = len(run_id_counts)
    print(f"\nNumber of run_ids: {n_unique}")
    print(f"Number of events:  {n_events:,}")

    print("\nrun_id    Events")
    print("------    ------")
    for run_id in sorted(run_id_counts):
        print(f"{run_id:>6}    {run_id_counts[run_id]:>6,}")


if __name__ == "__main__":
    main()
