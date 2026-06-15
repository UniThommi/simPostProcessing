#!/usr/bin/env python3
"""
Zählt die verwendeten run_ids in einer post-prozessierten HDF5-Datei.

Akzeptiert die Ausgabe beider Post-Processing-Skripte:
  - zylFullPostprocessingOptimized.py     (event_id_columns = ["run_id", "nc_id"])
  - musun2SimFullPostprocessing.py        (event_id_columns = ["run_id", "muon_id", "nc_id"])
  - musun2SimMultiOpticalPostprocessing.py (wie musun2Sim)

In jedem Format ist run_id eine benannte Spalte im 'event_ids'-Dataset. Die
Spaltenposition wird über das 'event_id_columns'-Metadaten-Dataset bestimmt,
sodass dieses Skript unabhängig von der Spaltenanzahl funktioniert.

Verwendung:
    python src/evaluation/count_run_ids.py --data /path/to/resum_output_*.hdf5
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
    Liest die run_id-Spalte aus 'event_ids' und gibt einen Counter
    {run_id: Anzahl Events} zurück.
    """
    with h5py.File(filepath, "r") as f:
        if "event_ids" not in f:
            raise KeyError(
                f"'event_ids' Dataset fehlt in {filepath}. "
                f"Ist dies wirklich eine Post-Processing-Ausgabe?"
            )

        # Spaltenindex von run_id bestimmen (robust gegenüber 2- vs. 3-Spalten-Format)
        if "event_id_columns" in f:
            columns = [_decode(c) for c in f["event_id_columns"][:]]
            if "run_id" not in columns:
                raise KeyError(
                    f"'run_id' nicht in event_id_columns {columns} gefunden."
                )
            run_id_col = columns.index("run_id")
        else:
            # Fallback: run_id ist in allen Skripten die erste Spalte
            run_id_col = 0
            print(
                "  Warnung: 'event_id_columns' fehlt, nehme run_id = Spalte 0 an."
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
        description="Zählt die verwendeten run_ids in einer post-prozessierten HDF5-Datei."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Pfad zur post-prozessierten HDF5-Datei (resum_output_*.hdf5).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Chunk-Größe für speichereffizientes Lesen (default: 1_000_000).",
    )
    args = parser.parse_args()

    filepath = Path(args.data)
    if not filepath.is_file():
        print(f"FEHLER: Datei nicht gefunden: {filepath}", file=sys.stderr)
        sys.exit(1)

    print(f"Datei: {filepath}")
    run_id_counts, n_events = count_run_ids(filepath, chunk_size=args.chunk_size)

    n_unique = len(run_id_counts)
    print(f"\nAnzahl verwendeter run_ids: {n_unique}")
    print(f"Anzahl Events gesamt:       {n_events:,}")

    print("\nrun_id    Events")
    print("------    ------")
    for run_id in sorted(run_id_counts):
        print(f"{run_id:>6}    {run_id_counts[run_id]:>6,}")


if __name__ == "__main__":
    main()
