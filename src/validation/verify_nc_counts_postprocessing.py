#!/usr/bin/env python3
# Author: Thomas Buerger (University of Tübingen)
"""
Neutron capture count verification for the single-simulation workflow.

Compares the number of neutron captures (NCs) between
  1. the raw simulation files (unique (evtid, nC_track_id) pairs in
     /hit/MyNeutronCaptureOutput, counted per file and summed), and
  2. the post-processed training + validation files (rows of phi_matrix),
and reports the difference and the train/validation ratio. Raw files that
cannot be read are listed and saved to corrupt_simulation_files.txt in the
output directory.

The check only compares counts, not NC identities; for the MUSUN workflow,
validate_neutron_capture_consistency.py compares the actual NC keys.

Note: the script looks for post-processed files named
resum_output_*_{train,validation}.hdf5, the naming of an earlier version of
zylFullPostprocessingOptimized.py. Current outputs are named
ncscore_output_0_{train,validation}.hdf5 and are not found by this script.
"""

import os
import sys
import h5py
import argparse
from pathlib import Path
from typing import Set, Tuple, Dict, List
import glob
from collections import defaultdict


def find_simulation_files(base_path: Path, nested: bool = True) -> list[Path]:
    """
    Find all simulation files (output_*.hdf5).

    Args:
        base_path: base directory of the simulation data
        nested: if True, search the subdirectories; otherwise only base_path

    Returns:
        Sorted list of HDF5 file paths
    """
    files = []

    if nested:
        # Search the subdirectories (each subdirectory = one simulation)
        print(f"Searching subdirectories of: {base_path}")
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        print(f"Subdirectories found: {len(subdirs)}")

        for subdir in sorted(subdirs):
            subdir_files = sorted(subdir.glob('output_*.hdf5'))
            if subdir_files:
                print(f"  {subdir.name}: {len(subdir_files)} files")
                files.extend(subdir_files)
    else:
        # Search base_path directly
        files = sorted(base_path.glob('output_*.hdf5'))
        print(f"Files found in {base_path}: {len(files)}")

    return files


def count_neutron_captures_in_simulation(
    file_path: Path,
    verbose: bool = False
) -> int:
    """
    Count the unique neutron captures in one simulation file.

    A neutron capture is identified by the pair (evtid, nC_track_id).

    Args:
        file_path: path of the HDF5 simulation file
        verbose: print the count of every file

    Returns:
        Number of unique (evtid, nC_track_id) pairs (0 if the file has no NC group)

    Raises:
        OSError: if the file cannot be read (corrupt)
        KeyError: if required datasets are missing
    """
    unique_pairs_count = 0
    total_entries = 0

    try:
        with h5py.File(file_path, 'r') as f:
            if 'hit/MyNeutronCaptureOutput' not in f:
                print(f"  ⚠ Warning: {file_path.name} has no MyNeutronCaptureOutput group")
                return 0

            evtid = f['hit/MyNeutronCaptureOutput/evtid/pages'][:]
            nC_track_id = f['hit/MyNeutronCaptureOutput/nC_track_id/pages'][:]

            assert len(evtid) == len(nC_track_id), "Mismatch between evtid and nC_track_id!"

            total_entries += len(evtid)
            unique_pairs_in_run = set(zip(evtid, nC_track_id))
            unique_pairs_count = len(unique_pairs_in_run)

            if verbose:
                print(
                f"  {file_path.name}: "
                f"{unique_pairs_count} unique NCs "
                f"({total_entries} entries)")

    except (OSError, KeyError, ValueError) as e:
        # Corrupt or malformed file
        print(f"  ✗ CORRUPT FILE: {file_path.name}")
        print(f"     Error: {type(e).__name__}: {e}")
        print(f"     Path: {file_path.absolute()}")
        raise  # handled in count_all_simulation_ncs()

    return unique_pairs_count


def count_all_simulation_ncs(
    sim_files: list[Path],
    verbose: bool = False,
    progress_interval: int = 50
) -> Tuple[int, list[Path]]:
    """
    Count the unique neutron captures over all simulation files.

    Pairs are unique per file; the counts of all files are summed.
    Unreadable files are skipped and returned in the list of corrupt files.

    Args:
        sim_files: list of simulation files
        verbose: print the count of every file
        progress_interval: print the progress every N files

    Returns:
        Tuple of (total number of unique NCs, list of corrupt files)
    """
    print(f"\n{'='*70}")
    print(f"SIMULATION DATA ANALYSIS")
    print(f"{'='*70}")
    print(f"Files to process: {len(sim_files)}\n")

    total_nc_count = 0
    files_processed = 0
    files_failed = 0
    corrupt_files = []

    for idx, file_path in enumerate(sim_files, 1):
        try:
            nc_count = count_neutron_captures_in_simulation(file_path, verbose)
            total_nc_count += nc_count
            files_processed += 1

            if idx % progress_interval == 0:
                print(f"  Progress: {idx}/{len(sim_files)} files processed "
                      f"({files_processed} successful, {files_failed} failed)")
                print(f"  Accumulated NCs (all runs): {total_nc_count}")

        except (OSError, KeyError, ValueError) as e:
            files_failed += 1
            corrupt_files.append(file_path)
            print(f"  → File skipped and recorded\n")
            continue

    print(f"\n{'='*70}")
    print(f"SIMULATION SUMMARY:")
    print(f"  Successfully processed: {files_processed}/{len(sim_files)}")
    print(f"  Failed (corrupt): {files_failed}")
    print(f"  Neutron captures in total (unique per run): {total_nc_count}")

    if corrupt_files:
        print(f"\n  ⚠ CORRUPT FILES ({len(corrupt_files)}):")
        for cf in corrupt_files:
            print(f"     - {cf.absolute()}")

    print(f"{'='*70}\n")

    return total_nc_count, corrupt_files

def count_postprocessed_entries(output_path: Path) -> Dict[str, int]:
    """
    Count the entries (rows of phi_matrix) of the post-processed training and validation files.

    Args:
        output_path: directory with resum_output_*_train.hdf5 and
                     resum_output_*_validation.hdf5

    Returns:
        Dictionary with 'train', 'validation' and 'total' counts

    Raises:
        FileNotFoundError: if no training or validation file is found
    """
    print(f"\n{'='*70}")
    print(f"POST-PROCESSED DATA ANALYSIS")
    print(f"{'='*70}")

    train_files = sorted(output_path.glob('resum_output_*_train.hdf5'))
    val_files = sorted(output_path.glob('resum_output_*_validation.hdf5'))

    if not train_files:
        raise FileNotFoundError(f"No training file found in {output_path}")
    if not val_files:
        raise FileNotFoundError(f"No validation file found in {output_path}")

    print(f"Training files: {[f.name for f in train_files]}")
    print(f"Validation files: {[f.name for f in val_files]}\n")

    counts = {'train': 0, 'validation': 0}

    for train_file in train_files:
        try:
            with h5py.File(train_file, 'r') as f:
                # One row = one neutron capture
                if 'phi_matrix' in f:
                    n_entries = f['phi_matrix'].shape[0]
                    counts['train'] += n_entries
                    print(f"  {train_file.name}: {n_entries} entries")
                else:
                    print(f"  ⚠ Warning: {train_file.name} has an unexpected structure")
        except (OSError, KeyError, ValueError) as e:
            print(f"  ✗ CORRUPT FILE: {train_file}")
            print(f"     Error: {e}")
            print(f"     Path: {train_file.absolute()}")
            raise RuntimeError(f"Training file is corrupt: {train_file}") from e

    for val_file in val_files:
        try:
            with h5py.File(val_file, 'r') as f:
                if 'phi_matrix' in f:
                    n_entries = f['phi_matrix'].shape[0]
                    counts['validation'] += n_entries
                    print(f"  {val_file.name}: {n_entries} entries")
                else:
                    print(f"  ⚠ Warning: {val_file.name} has an unexpected structure")
        except (OSError, KeyError, ValueError) as e:
            print(f"  ✗ CORRUPT FILE: {val_file}")
            print(f"     Error: {e}")
            print(f"     Path: {val_file.absolute()}")
            raise RuntimeError(f"Validation file is corrupt: {val_file}") from e

    counts['total'] = counts['train'] + counts['validation']

    print(f"\n{'='*70}")
    print(f"POST-PROCESSED SUMMARY:")
    print(f"  Training entries: {counts['train']}")
    print(f"  Validation entries: {counts['validation']}")
    print(f"  Total entries: {counts['total']}")
    print(f"  Validation fraction: {counts['validation']/counts['total']*100:.2f}%")
    print(f"{'='*70}\n")

    return counts


def compare_counts(sim_count: int, postproc_count: int) -> None:
    """
    Print the comparison of the neutron capture counts.

    Only prints the result; the exit code does not depend on it.

    Args:
        sim_count: number of NCs in the simulation data
        postproc_count: number of NCs in the post-processed data
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON: SIMULATION vs POST-PROCESSING")
    print(f"{'='*70}")

    difference = sim_count - postproc_count
    if sim_count > 0:
        recovery_rate = (postproc_count / sim_count) * 100
    else:
        recovery_rate = 0.0

    print(f"Simulation (input):      {sim_count:>12,} neutron captures")
    print(f"Post-processed (output): {postproc_count:>12,} neutron captures")
    print(f"{'-'*70}")
    print(f"Difference:              {difference:>12,} NCs")
    print(f"Recovery rate:           {recovery_rate:>11.2f}%")

    if difference == 0:
        print(f"\n✓ PERFECT: all neutron captures were processed!")
    elif difference > 0:
        loss_rate = (difference / sim_count) * 100
        print(f"\n⚠ LOSS: {loss_rate:.2f}% of the NCs are missing in the output")
        print(f"  Possible causes:")
        print(f"  - Filtering of photons outside the detector volume")
        print(f"  - NC events without detected optical photons")
        print(f"  - Errors during the post-processing")
    else:
        print(f"\n✗ ERROR: more entries in the output than in the simulation!")
        print(f"  This should not happen - please check the data")

    print(f"{'='*70}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Verify neutron capture counts: simulation vs post-processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Nested structure (one subdirectory per run)
  python %(prog)s \\
    --sim-path /pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawHomogeneousNCsSSD300PMTs \\
    --output-path /pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/MLFormatHomogeneousNCsZylSSD300PMTs/ \\
    --nested

  # Files directly in the simulation directory
  python %(prog)s \\
    --sim-path /path/to/simulation_files \\
    --output-path /path/to/output_files

  # Verbose output
  python %(prog)s -s /path/sim -o /path/out -n -v
        """
    )

    parser.add_argument(
        '-s', '--sim-path',
        type=Path,
        required=True,
        help='Path of the simulation data (with or without subdirectories)'
    )

    parser.add_argument(
        '-o', '--output-path',
        type=Path,
        required=True,
        help='Path of the post-processed data (contains the train/val HDF5 files)'
    )

    parser.add_argument(
        '-n', '--nested',
        action='store_true',
        help='The simulation data is stored in subdirectories'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print the count of every file'
    )

    parser.add_argument(
        '--progress-interval',
        type=int,
        default=100,
        help='Print the progress every N files (default: 100)'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.sim_path.exists():
        print(f"✗ Error: simulation path does not exist: {args.sim_path}")
        sys.exit(1)

    if not args.output_path.exists():
        print(f"✗ Error: output path does not exist: {args.output_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"NEUTRON CAPTURE COUNT VERIFICATION")
    print(f"{'='*70}")
    print(f"Simulation data:  {args.sim_path}")
    print(f"Post-processed:   {args.output_path}")
    print(f"Nested structure: {'yes' if args.nested else 'no'}")
    print(f"Verbose:          {'yes' if args.verbose else 'no'}")
    print(f"{'='*70}")

    try:
        # 1. Find all simulation files
        sim_files = find_simulation_files(args.sim_path, args.nested)

        if not sim_files:
            print(f"✗ Error: no output_*.hdf5 files found!")
            sys.exit(1)

        # 2. Count the neutron captures in the simulation
        sim_nc_count, corrupt_sim_files = count_all_simulation_ncs(
            sim_files,
            verbose=args.verbose,
            progress_interval=args.progress_interval
        )

        # 3. Count the entries of the post-processed files
        postproc_counts = count_postprocessed_entries(args.output_path)

        # 4. Compare
        compare_counts(sim_nc_count, postproc_counts['total'])

        if postproc_counts['total'] > 0:
            print(f"TRAIN/VAL SPLIT:")
            print(f"  Train:      {postproc_counts['train']:>12,} "
                  f"({postproc_counts['train']/postproc_counts['total']*100:.2f}%)")
            print(f"  Validation: {postproc_counts['validation']:>12,} "
                  f"({postproc_counts['validation']/postproc_counts['total']*100:.2f}%)")

        if corrupt_sim_files:
            print(f"\n{'='*70}")
            print(f"CORRUPT SIMULATION FILES ({len(corrupt_sim_files)}):")
            print(f"{'='*70}")
            for corrupt_file in corrupt_sim_files:
                print(f"  {corrupt_file}")
            print(f"{'='*70}")

            # Also save the list to a log file
            corrupt_log = args.output_path / "corrupt_simulation_files.txt"
            with open(corrupt_log, 'w') as f:
                f.write(f"# Corrupt simulation files\n")
                f.write(f"# Found on: {__import__('datetime').datetime.now()}\n")
                f.write(f"# Count: {len(corrupt_sim_files)}\n\n")
                for corrupt_file in corrupt_sim_files:
                    f.write(f"{corrupt_file}\n")
            print(f"\nCorrupt files saved to: {corrupt_log}")

        print(f"\n✓ Analysis finished successfully\n")

    except Exception as e:
        print(f"\n✗ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
