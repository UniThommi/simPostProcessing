#!/usr/bin/env python3
"""
Post-processing for the MUSUN workflow with repeated optical simulations.

Same processing and output format as musun2SimFullPostprocessing.py, but the
photon hits of several optical simulations of the same neutron captures (NCs)
are summed:
  -i1  sim1: NC truth of the MUSUN muon simulation    run_NNN/
  -i2  sim2: first optical simulation of the NCs      run_NNN/
  -i3  further optical simulations of the same NCs    run_NNN/sim_NNN/

Why several simulations: an SSD voxel combines the optical properties of a PMT
with those of the reflective foil that lines the rest of the wall. It therefore
reflects much more light than a PMT, and a single optical simulation yields a
lower detection efficiency per voxel than a PMT would have. Summing the hits of
several simulations of the same NCs raises the voxel efficiency to that of a PMT.

All optical files of a run (sim2 plus every sim_NNN of -i3) are scanned in turn
and their hits are accumulated in the same per-NC counters. --sim3-runs
restricts the processing to selected runs; all processed runs must contain the
same set of sim_NNN directories.

Matching, time window (0-200 ns, 1 ns tolerance for rounding), resuming and the
output files are the same as in musun2SimFullPostprocessing.py; see its module
docstring for details.

Author: Thomas Bürger, Eberhard Karls Universität Tübingen
"""

import os
import glob
import json
import h5py
import sys
import re
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial import KDTree
import multiprocessing as mp
import time
from typing import Optional, List, Dict
import psutil
import gc
import argparse
from pathlib import Path
import shutil


# ============================================================================
# Utility Functions
# ============================================================================

def get_memory_usage():
    """Return the memory usage of the current process (RSS/VMS in MB, percent of total)."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }


def print_memory_usage(label=""):
    mem = get_memory_usage()
    print(f"Memory {label}: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, {mem['percent']:.1f}%")


# ============================================================================
# ProgressTracker
# ============================================================================

class ProgressTracker:
    """Persists the processing state in the output directory so that a run can be resumed.

    processing_progress.json stores the run settings ('config'), whether the
    optical scan has finished ('sim2_scan_complete') and the number of
    completely written output rows ('rows_committed'); scan_checkpoint.npz
    stores the scan result. Both are removed after a successful run.
    """

    def __init__(self, output_dir, output_file):
        self.output_dir = Path(output_dir)
        self.output_file = Path(output_file)
        self.progress_file = self.output_dir / "processing_progress.json"
        self.lock_file = self.output_dir / "processing.lock"
        self.split_file = self.output_dir / "train_val_split.json"
        self.scan_checkpoint_file = self.output_dir / "scan_checkpoint.npz"
        self.progress_data = self.load_progress()

    def load_progress(self):
        """Load the saved progress, or return a fresh state if there is none."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                print(f"Existing progress found: {len(data.get('completed_files', []))} files already processed")
                return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: could not load progress ({e}), starting from scratch")

        return {
            'completed_files': [],
            'failed_files': [],
            'start_time': time.time(),
            'last_update': time.time(),
            'total_entries_written': 0,
            'total_unassigned': 0,
            'output_file': str(self.output_file),
            'sim2_scan_complete': False,
            'rows_committed': 0,
            'config': None,
        }


    def cleanup(self):
        """Remove the progress, lock and scan checkpoint files (after success or --reset)."""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.lock_file.exists():
                self.lock_file.unlink()
            if self.scan_checkpoint_file.exists():
                self.scan_checkpoint_file.unlink()
        except OSError:
            pass

    def save_progress(self):
        self.progress_data['last_update'] = time.time()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except IOError as e:
            print(f"Warning: could not save progress: {e}")

    def verify_hdf5_integrity(self, output_file):
        """Return True if the output file can be opened and contains the main datasets."""
        try:
            with h5py.File(output_file, 'r') as f:
                if 'phi_matrix' in f and 'target_matrix' in f and 'event_ids' in f:
                    _ = f['phi_matrix'].shape
                    return True
        except Exception as e:
            print(f"  WARNING: HDF5 integrity check failed: {e}")
            return False
        return False

    def mark_sim2_scan_complete(self):
        """Record that Phase 2 has finished and its checkpoint has been saved."""
        self.progress_data['sim2_scan_complete'] = True
        self.save_progress()

    def is_sim2_scan_complete(self):
        return self.progress_data.get('sim2_scan_complete', False)


# ============================================================================
# Resume support
# ============================================================================

# Row-aligned output datasets; they always have the same number of rows.
_ROW_DATASETS = ('phi_matrix', 'target_matrix', 'region_matrix', 'event_ids', 'weights')


def save_scan_checkpoint(path, nc_voxel_counters, total_unassigned, total_orphaned, scanned_runs):
    """Save the aggregated Phase-2 result so that a restart can skip the optical scan.

    The per-NC voxel counters are flattened into CSR-like arrays: the hits of the
    i-th key are voxel_ids[offsets[i]:offsets[i+1]] with the matching counts.
    The file is written under a temporary name and then renamed, so an
    interrupted save never leaves a truncated checkpoint behind.
    """
    keys = list(nc_voxel_counters.keys())
    offsets = np.zeros(len(keys) + 1, dtype=np.int64)
    voxel_ids, counts = [], []
    for i, key in enumerate(keys):
        counter = nc_voxel_counters[key]
        voxel_ids.extend(counter.keys())
        counts.extend(counter.values())
        offsets[i + 1] = len(counts)

    path = Path(path)
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp_path,
        keys=np.array(keys, dtype=np.int64).reshape(-1, 3),
        offsets=offsets,
        voxel_ids=np.array(voxel_ids, dtype=str),
        counts=np.array(counts, dtype=np.int64),
        totals=np.array([total_unassigned, total_orphaned], dtype=np.int64),
        scanned_runs=np.array(sorted(scanned_runs), dtype=np.int64),
    )
    os.replace(tmp_path, path)


def load_scan_checkpoint(path):
    """Inverse of save_scan_checkpoint(); returns the same tuple as scan_optical_and_aggregate()."""
    with np.load(path, allow_pickle=False) as data:
        keys = data['keys'].tolist()
        offsets = data['offsets']
        voxel_ids = data['voxel_ids'].tolist()
        counts = data['counts'].tolist()
        total_unassigned, total_orphaned = (int(x) for x in data['totals'])
        scanned_runs = {int(r) for r in data['scanned_runs']}

    nc_voxel_counters = {}
    for i, key in enumerate(keys):
        start, end = offsets[i], offsets[i + 1]
        nc_voxel_counters[tuple(key)] = Counter(dict(zip(voxel_ids[start:end], counts[start:end])))
    return nc_voxel_counters, total_unassigned, total_orphaned, scanned_runs


def prepare_output_for_resume(output_file, rows_committed):
    """Drop rows of an interrupted batch and return the keys that are already written.

    Rows beyond ``rows_committed`` (the row count recorded after the last fully
    written batch) may belong to a partially written batch and are discarded.
    If the file holds fewer rows than recorded (e.g. it had to be recreated),
    only the rows that are actually present are kept.

    Returns:
        (n_rows, written_keys): number of valid rows and the set of their
        (run_id, muon_id, nc_id) keys.
    """
    with h5py.File(output_file, 'a') as out:
        n_rows = min([rows_committed] + [out[name].shape[0] for name in _ROW_DATASETS])
        for name in _ROW_DATASETS:
            dset = out[name]
            if dset.shape[0] != n_rows:
                dset.resize((n_rows,) + dset.shape[1:])
        written_keys = {tuple(row) for row in out['event_ids'][:n_rows].tolist()}
    return n_rows, written_keys


def mark_output_complete(output_file):
    """Flag the output file as fully written (checked on the next start)."""
    with h5py.File(output_file, 'a') as out:
        out.attrs['processing_complete'] = True


def is_output_complete(output_file):
    try:
        with h5py.File(output_file, 'r') as f:
            return bool(f.attrs.get('processing_complete', False))
    except OSError:
        return False


# ============================================================================
# Mapping Functions
# ============================================================================
# Each simulation file numbers its materials and physical volumes in its own
# way. The functions below translate these local IDs into the global IDs stored
# in src/mappings/*.json, so that matID/volID mean the same in every row.

def loadMapping(filePath: str) -> dict:
    with open(filePath, 'r') as f:
        mapping = json.load(f)
    return mapping

def remapMaterialIDsToGlobal(
    glob_mapping_json_path: str,
    local_mat_map: dict,
    local_material_ids
) -> tuple[np.ndarray, dict]:
    """Translate file-local material IDs into global IDs.

    Read-only: the global mapping must already contain every material (see
    collect_all_materials_first() / create_global_material_mapping()).

    Args:
        glob_mapping_json_path: path of the global {material name: ID} mapping
        local_mat_map: {local ID: material name} of the current file
        local_material_ids: local material IDs to translate

    Returns:
        (global IDs as array, global mapping)
    """
    if not os.path.exists(glob_mapping_json_path):
        raise RuntimeError(
            f"Global material mapping is missing: {glob_mapping_json_path}\n"
            f"Was collect_all_materials_first() run?"
        )
    globMaterialMapping = loadMapping(glob_mapping_json_path)
    local_material_ids = np.array(local_material_ids)
    unique_local_ids = np.unique(local_material_ids)
    mapping_dict = {}

    # Only map, never extend the global mapping here.
    for local_id in unique_local_ids:
        local_name = local_mat_map[local_id]
        if local_name not in globMaterialMapping:
            raise RuntimeError(
                f"Material '{local_name}' not found in the global mapping!\n"
                f"Available materials: {list(globMaterialMapping.keys())}"
            )
        mapping_dict[local_id] = globMaterialMapping[local_name]

    vectorized_mapper = np.vectorize(mapping_dict.get)
    global_material_ids = vectorized_mapper(local_material_ids)
    return global_material_ids, globMaterialMapping


def remapVolumeIDsToGlobal(
    glob_mapping_json_path: str,
    local_vol_map: dict,
    local_volume_ids
) -> tuple[np.ndarray, dict]:
    """Translate file-local physical volume IDs into global IDs.

    Same as remapMaterialIDsToGlobal(); the empty volume name is stored as "noVolume".
    """
    if not os.path.exists(glob_mapping_json_path):
        raise RuntimeError(
            f"Global volume mapping is missing: {glob_mapping_json_path}\n"
            f"Was collect_all_volumes_first() run?"
        )
    globVolumeMapping = loadMapping(glob_mapping_json_path)
    local_volume_ids = np.array(local_volume_ids)
    unique_local_ids = np.unique(local_volume_ids)
    mapping_dict = {}

    for local_id in unique_local_ids:
        local_name = local_vol_map[local_id]
        local_name_clean = "noVolume" if local_name == "" else local_name
        if local_name_clean not in globVolumeMapping:
            raise RuntimeError(
                f"Volume '{local_name_clean}' not found in the global mapping!\n"
                f"Available volumes: {list(globVolumeMapping.keys())}"
            )
        mapping_dict[local_id] = globVolumeMapping[local_name_clean]

    vectorized_mapper = np.vectorize(mapping_dict.get)
    global_volume_ids = vectorized_mapper(local_volume_ids)
    return global_volume_ids, globVolumeMapping


def create_global_material_mapping(all_materials, existing_mapping_path):
    """Add unknown materials to the global mapping and save it (existing IDs never change)."""
    if os.path.exists(existing_mapping_path):
        with open(existing_mapping_path, 'r') as f:
            global_mapping = json.load(f)
    else:
        global_mapping = {}
    max_id = max(global_mapping.values()) if global_mapping else -1
    for material in all_materials:
        if material not in global_mapping:
            max_id += 1
            global_mapping[material] = max_id
    with open(existing_mapping_path, 'w') as f:
        json.dump(global_mapping, f, indent=2)
    return global_mapping


def create_global_volume_mapping(all_volumes, existing_mapping_path):
    """Add unknown volumes to the global mapping and save it (existing IDs never change)."""
    if os.path.exists(existing_mapping_path):
        with open(existing_mapping_path, 'r') as f:
            global_mapping = json.load(f)
    else:
        global_mapping = {}
    max_id = max(global_mapping.values()) if global_mapping else -1
    for volume in all_volumes:
        if volume not in global_mapping:
            max_id += 1
            global_mapping[volume] = max_id
    with open(existing_mapping_path, 'w') as f:
        json.dump(global_mapping, f, indent=2)
    return global_mapping


def collect_all_materials_first(files):
    """Collect the material names of all sim1 files before processing, so the global mapping is complete."""
    print("Collecting materials from all sim1 files...")
    all_materials = set()
    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                mapping_names = [x.decode() for x in f["hit/materials/materialNames"]["pages"][:]]
                all_materials.update(mapping_names)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    print(f"Found: {len(all_materials)} unique materials")
    return all_materials


def collect_all_volumes_first(files):
    """Collect the physical volume names of all sim1 files before processing (empty name -> "noVolume")."""
    print("Collecting physical volumes from all sim1 files...")
    all_volumes = set()
    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                volume_names = [x.decode() for x in f["hit/physVolumes/physVolumeNames"]["pages"][:]]
                volume_names_clean = [
                    "noVolume" if name == "" else name
                    for name in volume_names
                ]
                all_volumes.update(volume_names_clean)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    print(f"Found: {len(all_volumes)} unique physical volumes")
    if "" in all_volumes:
        print(f"  ⚠ WARNING: Empty string still present after sanitization!")
        all_volumes.remove("")
        all_volumes.add("noVolume")
    return all_volumes


# ============================================================================
# Run Directory Validation
# ============================================================================

_RUN_DIR_PATTERN = re.compile(r'^run_(\d+)$')
_SIM_DIR_PATTERN = re.compile(r'^sim_(\d+)$')


def _parse_run_id_from_name(name: str) -> int:
    """Parse integer run_id from a directory name like 'run_007' → 7 (-1 if it does not match)."""
    m = _RUN_DIR_PATTERN.match(name)
    if m:
        return int(m.group(1))
    return -1


def _extract_run_id_from_path(file_path: str) -> int:
    """Returns the run_id embedded in a path like .../run_003/output_t0.hdf5 → 3 (0 if there is none)."""
    for part in Path(file_path).parts:
        if part.startswith("run_"):
            try:
                return int(part.split("_")[1])
            except (IndexError, ValueError):
                pass
    return 0


def validate_run_structure(path: str, nested: bool, label: str):
    """
    Validates the input directory structure.

    Nested mode: subdirectories with output_*.hdf5 files that do not match the
    'run_NNN' pattern are ignored (with a message); at least one run_NNN
    directory is required. Raises RuntimeError otherwise.

    Non-nested mode: checks that output_*.hdf5 files are present directly
    in path.
    """
    p = Path(path)
    if nested:
        dirs_with_files = []
        for subdir in sorted(p.iterdir()):
            if not subdir.is_dir():
                continue
            if glob.glob(os.path.join(subdir, 'output_*.hdf5')):
                dirs_with_files.append(subdir)

        if not dirs_with_files:
            raise RuntimeError(
                f"[{label}] --nested specified but no subdirectories with "
                f"output_*.hdf5 files found in '{path}'"
            )

        ignored = [s for s in dirs_with_files if not _RUN_DIR_PATTERN.match(s.name)]
        dirs_with_files = [s for s in dirs_with_files if _RUN_DIR_PATTERN.match(s.name)]

        if ignored:
            print(f"[{label}] Ignoring {len(ignored)} directory(ies) without a run_NNN name: "
                  f"{[s.name for s in ignored]}")

        if not dirs_with_files:
            raise RuntimeError(
                f"[{label}] No subdirectories matching 'run_NNN' with output_*.hdf5 files "
                f"found in '{path}'"
            )

        print(f"[{label}] Run structure OK: {len(dirs_with_files)} run directories")
        for subdir in dirs_with_files:
            n = len(glob.glob(os.path.join(subdir, 'output_*.hdf5')))
            print(f"  {subdir.name}: {n} HDF5 file(s)")
    else:
        hdf5_files = sorted(glob.glob(os.path.join(path, 'output_*.hdf5')))
        if not hdf5_files:
            raise RuntimeError(
                f"[{label}] No output_*.hdf5 files found directly in '{path}'. "
                f"If files are in subdirectories, use --nested."
            )
        print(f"[{label}] Found {len(hdf5_files)} HDF5 file(s) directly in '{path}'")


def validate_sim3_structure(sim3_path: str) -> dict[int, list[str]]:
    """
    Validates and reports the run_NNN/sim_NNN/ structure of the -i3 input.

    Returns a summary: {run_id: [sim_dir_names...]}
    Raises RuntimeError for structural violations.
    """
    p = Path(sim3_path)
    summary: dict[int, list[str]] = {}

    run_dirs = sorted([d for d in p.iterdir() if d.is_dir()])
    if not run_dirs:
        raise RuntimeError(f"[Sim3] No subdirectories found in '{sim3_path}'")

    for run_dir in run_dirs:
        run_id = _parse_run_id_from_name(run_dir.name)
        if run_id == -1:
            # Non-run_NNN directories at the top level are silently skipped
            continue

        sim_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and _SIM_DIR_PATTERN.match(d.name)])
        if not sim_dirs:
            print(f"  [Sim3] WARNING: run_{run_id:03d} has no sim_NNN subdirectories - skipped")
            continue

        sim_names_with_files = []
        for sim_dir in sim_dirs:
            hdf5_files = (
                sorted(glob.glob(os.path.join(sim_dir, 'output_t*.hdf5'))) or
                sorted(glob.glob(os.path.join(sim_dir, 'output_*.hdf5')))
            )
            if hdf5_files:
                sim_names_with_files.append(sim_dir.name)
            else:
                print(f"  [Sim3] WARNING: {run_dir.name}/{sim_dir.name} contains no HDF5 files - skipped")

        if sim_names_with_files:
            summary[run_id] = sim_names_with_files

    if not summary:
        raise RuntimeError(
            f"[Sim3] No valid run_NNN/sim_NNN/ directories with HDF5 files found in '{sim3_path}'"
        )

    print(f"[Sim3] Structure OK: {len(summary)} run directories")
    for run_id, sims in sorted(summary.items()):
        print(f"  run_{run_id:03d}: {len(sims)} sim directories ({sims[0]} … {sims[-1]})")

    return summary


def validate_sim1_sim2_consistency(sim1_path: str, sim2_path: str):
    """
    Cross-checks that sim1 and sim2 have matching run directories.
    Warns about runs present in one but not the other.
    """
    def _get_run_names(path):
        return {
            d.name for d in Path(path).iterdir()
            if d.is_dir() and _RUN_DIR_PATTERN.match(d.name)
        }

    sim1_runs = _get_run_names(sim1_path)
    sim2_runs = _get_run_names(sim2_path)

    only_in_sim1 = sim1_runs - sim2_runs
    only_in_sim2 = sim2_runs - sim1_runs

    if only_in_sim1:
        print(f"  WARNING: Runs in Sim1 without matching Sim2 directory "
              f"(NC events will be dropped): {sorted(only_in_sim1)}")
    if only_in_sim2:
        print(f"  WARNING: Runs in Sim2 without matching Sim1 directory "
              f"(photon data will be ignored): {sorted(only_in_sim2)}")
    if not only_in_sim1 and not only_in_sim2:
        print(f"  Sim1/Sim2 run directories match: {sorted(sim1_runs)}")


def validate_sim1_sim3_consistency(sim1_path: str, sim3_path: str,
                                    active_run_ids: set[int]) -> None:
    """
    Validates the active runs (selected via --sim3-runs) against Sim1 and Sim3.

    Aborts (raises RuntimeError) if:
      - An active run_id is missing from Sim1.
      - An active run_id is missing from Sim3 or has no sim_NNN subdirectories.
      - The sim_NNN sets differ across active runs (unequal optical simulation counts).

    Prints a clear summary of which runs are active, and reports any inactive
    runs in Sim1/Sim3 as informational messages only.
    """
    sim1_run_names = {
        d.name for d in Path(sim1_path).iterdir()
        if d.is_dir() and _RUN_DIR_PATTERN.match(d.name)
    }
    sim1_run_ids = {_parse_run_id_from_name(n) for n in sim1_run_names}

    # Collect {run_id: {sim_name, ...}} for all runs present in Sim3
    sim3_run_to_sims: dict[int, set[str]] = {}
    for run_dir in sorted(Path(sim3_path).iterdir()):
        if not run_dir.is_dir() or not _RUN_DIR_PATTERN.match(run_dir.name):
            continue
        run_id = _parse_run_id_from_name(run_dir.name)
        sim_names = {
            d.name for d in run_dir.iterdir()
            if d.is_dir() and _SIM_DIR_PATTERN.match(d.name)
        }
        if sim_names:
            sim3_run_to_sims[run_id] = sim_names

    # --- Hard checks for each active run ---
    errors = []
    for run_id in sorted(active_run_ids):
        run_name = f"run_{run_id:03d}"
        if run_id not in sim1_run_ids:
            errors.append(f"  ERROR: {run_name} is active but missing in Sim1 ({sim1_path})")
        if run_id not in sim3_run_to_sims:
            errors.append(f"  ERROR: {run_name} is active but missing in Sim3 ({sim3_path})")
        elif not sim3_run_to_sims[run_id]:
            errors.append(f"  ERROR: {run_name}/sim_*** in Sim3 contains no sim_NNN directories")

    if errors:
        raise RuntimeError(
            "Consistency check Sim1 ↔ Sim3 failed:\n" + "\n".join(errors)
        )

    # --- Hard check: all active runs must have the same sim_NNN set ---
    active_sim_sets = {run_id: sim3_run_to_sims[run_id] for run_id in active_run_ids}
    reference_id = sorted(active_run_ids)[0]
    reference_sims = active_sim_sets[reference_id]
    inconsistent = []
    for run_id in sorted(active_run_ids):
        sims = active_sim_sets[run_id]
        missing = reference_sims - sims
        extra = sims - reference_sims
        if missing or extra:
            inconsistent.append((run_id, missing, extra))

    if inconsistent:
        lines = [
            f"  ERROR: the sim_NNN sets of the active runs differ "
            f"(reference: run_{reference_id:03d} with {sorted(reference_sims)}):"
        ]
        for run_id, missing, extra in inconsistent:
            if missing:
                lines.append(f"    run_{run_id:03d}: missing sim dirs: {sorted(missing)}")
            if extra:
                lines.append(f"    run_{run_id:03d}: extra sim dirs:   {sorted(extra)}")
        raise RuntimeError("Consistency check Sim1 ↔ Sim3 failed:\n" + "\n".join(lines))

    # --- Informational: inactive runs in Sim1 ---
    inactive_in_sim1 = sim1_run_ids - active_run_ids
    if inactive_in_sim1:
        print(f"  INFO: the following Sim1 runs are NOT active and are skipped: "
              f"{sorted(inactive_in_sim1)}")

    # --- Informational: runs in Sim3 not selected ---
    inactive_in_sim3 = set(sim3_run_to_sims.keys()) - active_run_ids
    if inactive_in_sim3:
        print(f"  INFO: the following Sim3 runs exist but are not active: "
              f"{sorted(inactive_in_sim3)}")

    n_sims = len(reference_sims)
    sim_range = f"{sorted(reference_sims)[0]} … {sorted(reference_sims)[-1]}"
    print(f"  OK: {len(active_run_ids)} active run(s), each with {n_sims} sim_*** directories ({sim_range})")


# ============================================================================
# File Discovery
# ============================================================================

def find_all_hdf5_files(input_path: str, nested: bool) -> list[str]:
    """Return all output_*.hdf5 files directly in input_path, or in its subdirectories if nested."""
    files = []
    if nested:
        print(f"Searching for simulation data in subdirectories of {input_path}...")
        subdirs = [d for d in Path(input_path).iterdir() if d.is_dir()]
        for subdir in subdirs:
            subdir_files = sorted(glob.glob(os.path.join(subdir, 'output_*.hdf5')))
            if subdir_files:
                print(f"  Found: {len(subdir_files)} files in {subdir.name}")
                files.extend(subdir_files)
    else:
        files = sorted(glob.glob(os.path.join(input_path, 'output_*.hdf5')))
    return files


def find_sim2_files(sim2_path: str, nested: bool = False) -> dict[int, list[str]]:
    """
    Find the sim2 HDF5 files grouped by run_id (output_t*.hdf5, else output_*.hdf5).

    Returns:
        {run_id: [file_paths...]}  if nested=True
        {0: [file_paths...]}       if nested=False
    """
    files_by_run: dict[int, list[str]] = {}

    if nested:
        subdirs = sorted(Path(sim2_path).iterdir())
        for subdir in subdirs:
            if not subdir.is_dir():
                continue
            run_files = sorted(glob.glob(os.path.join(subdir, 'output_t*.hdf5')))
            if not run_files:
                run_files = sorted(glob.glob(os.path.join(subdir, 'output_*.hdf5')))
            if not run_files:
                continue
            run_id = _parse_run_id_from_name(subdir.name)
            if run_id == -1:
                raise RuntimeError(
                    f"[Sim2] Directory '{subdir.name}' in '{sim2_path}' contains "
                    f"HDF5 files but does not match the expected 'run_NNN' pattern."
                )
            files_by_run[run_id] = run_files
            print(f"  Sim2 run_{run_id:03d}: {len(run_files)} files")
    else:
        files = sorted(glob.glob(os.path.join(sim2_path, 'output_t*.hdf5')))
        if not files:
            files = sorted(glob.glob(os.path.join(sim2_path, 'output_*.hdf5')))
        if files:
            files_by_run[0] = files

    return files_by_run


def find_sim3_files(sim3_path: str) -> dict[int, list[str]]:
    """
    Discovers all optical HDF5 files from the multi-sim structure -i3.

    Expected layout:
        sim3_path/run_NNN/sim_NNN/output_t*.hdf5

    All files from all sim_NNN subdirectories within a run are pooled into
    a single flat list per run_id.  This allows scan_optical_and_aggregate() to
    process them transparently alongside Sim2 files.

    Returns:
        {run_id: [file_path, ...]}  (all sim_*** files combined per run)
    """
    files_by_run: dict[int, list[str]] = {}

    p = Path(sim3_path)
    for run_dir in sorted(p.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = _parse_run_id_from_name(run_dir.name)
        if run_id == -1:
            continue

        run_files: list[str] = []
        sim_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and _SIM_DIR_PATTERN.match(d.name)])

        for sim_dir in sim_dirs:
            sim_files = (
                sorted(glob.glob(os.path.join(sim_dir, 'output_t*.hdf5'))) or
                sorted(glob.glob(os.path.join(sim_dir, 'output_*.hdf5')))
            )
            run_files.extend(sim_files)

        if run_files:
            files_by_run[run_id] = run_files
            print(f"  Sim3 run_{run_id:03d}: {len(sim_dirs)} sim directories → {len(run_files)} files in total")

    return files_by_run


def combine_optical_files(
    sim2_files_by_run: dict[int, list[str]],
    sim3_files_by_run: dict[int, list[str]],
) -> dict[int, list[str]]:
    """
    Merges Sim2 and Sim3 file lists per run_id into a single combined dict.

    Files from both sources are concatenated — scan_optical_and_aggregate()
    will process them in sequence and accumulate hits into shared counters,
    naturally summing photon statistics across all simulations.
    """
    combined: dict[int, list[str]] = {}

    all_run_ids = set(sim2_files_by_run.keys()) | set(sim3_files_by_run.keys())
    for run_id in sorted(all_run_ids):
        files: list[str] = []
        files.extend(sim2_files_by_run.get(run_id, []))
        files.extend(sim3_files_by_run.get(run_id, []))
        if files:
            combined[run_id] = files

    return combined


def check_hdf5_integrity(combined_files_by_run: dict, output_dir: str) -> bool:
    """
    Opens and immediately closes every optical HDF5 file to detect truncated or
    corrupt files before the expensive processing phase begins.

    Files are grouped by their parent directory — one directory = one simulation.
    If any file in a directory fails to open, that entire simulation is flagged.
    The check always runs to completion so the report is exhaustive.

    Returns True if all files are intact, False if any simulation is faulty.
    Writes a text file listing faulty simulations to output_dir.
    """
    print(f"\n=== HDF5 integrity check (optical files) ===")

    # Group files by parent directory (= one simulation unit)
    files_by_sim: dict = defaultdict(list)
    for files in combined_files_by_run.values():
        for fpath in files:
            sim_dir = str(Path(fpath).parent)
            files_by_sim[sim_dir].append(fpath)

    total_sims = len(files_by_sim)
    total_files = sum(len(v) for v in files_by_sim.values())
    print(f"  Checking {total_files} files in {total_sims} simulation directories...")

    faulty: list = []  # list of (sim_dir, first_faulty_file, error_msg)

    for sim_idx, sim_dir in enumerate(sorted(files_by_sim.keys()), 1):
        files = files_by_sim[sim_dir]

        first_bad_file = None
        first_error = None
        for fpath in files:
            try:
                with h5py.File(fpath, 'r'):
                    pass
            except Exception as e:
                first_bad_file = fpath
                first_error = str(e)
                break  # one corrupt file marks the whole simulation as faulty

        if first_bad_file is not None:
            faulty.append((sim_dir, first_bad_file, first_error))
            print(f"  [{sim_idx}/{total_sims}] FAULTY: {sim_dir}")
            print(f"    File: {os.path.basename(first_bad_file)}")
            print(f"    Error: {first_error}")
        elif sim_idx % 10 == 0 or sim_idx == total_sims:
            print(f"  [{sim_idx}/{total_sims}] OK so far...")

    if not faulty:
        print(f"  All {total_sims} simulations OK ({total_files} files checked).")
        return True

    # Write full report to text file
    report_path = os.path.join(output_dir, "faulty_simulations.txt")
    with open(report_path, 'w') as rf:
        rf.write(f"Faulty simulations: {len(faulty)} of {total_sims}\n")
        rf.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"Files checked: {total_files}\n\n")
        for sim_dir, bad_file, err_msg in faulty:
            rf.write(f"Directory : {sim_dir}\n")
            rf.write(f"  File    : {bad_file}\n")
            rf.write(f"  Error   : {err_msg}\n\n")

    print(f"\n  {len(faulty)} faulty simulation(s) found:")
    for sim_dir, bad_file, _ in faulty:
        print(f"    - {sim_dir}  (first faulty file: {os.path.basename(bad_file)})")
    print(f"\n  Full list of errors: {report_path}")
    return False


# ============================================================================
# Geometry & Voxel Functions
# ============================================================================

def defineZylinder(
    geometry_name: str,
    valid_detectors: Optional[List[int]] = None
):
    """Return the geometry constants of the cylindrical SSD (all lengths in mm).

    Only r_zylinder, h_zylinder and valid_detectors are used by the
    post-processing; see zylFullPostprocessingOptimized.py for the meaning of
    the other values.
    """
    valid_geometry_names = ["currentDist"]
    if geometry_name not in valid_geometry_names:
        print("Invalid geometry name. It must be one of the following:")
        for name in valid_geometry_names:
            print(name)
        sys.exit()

    # Detector UIDs of the SSD surfaces
    if valid_detectors is None:
        valid_detectors = [1965, 1966, 1967, 1968]

    t_zylinder = 1
    l_voxel = 195
    t_voxel = 1
    r_pit = 3800
    dz_pit = 1
    r_zyl_bot = 3950
    r_zyl_top = 1200
    z_offset = -5000
    h = 8900

    if geometry_name == "currentDist":
        r_zylinder = 4300  # inner radius of the Tyvek wall
        z_origin = 20
        h_zylinder = h + 20 - z_origin
        print("Set currentDist as geometry.")

    r_ref = 6000
    h_ref = 8080
    z_ref = 20

    return (t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top,
            z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref, valid_detectors)


def build_voxel_tree(voxels: List[Dict]) -> Optional[KDTree]:
    """Build a KD-tree on the voxel centers; nearest-center lookup assigns photons to voxels."""
    if not voxels:
        return None
    centers = [voxel['center'] for voxel in voxels]
    return KDTree(centers)


def checkRadialMomentumVectorized(x, y, z, px, py, pz):
    """Return True for photons whose momentum points radially outwards (towards the wall)."""
    r = np.sqrt(x**2 + y**2)
    valid_r = r > 0
    radial_momentum = np.zeros_like(r)
    radial_momentum[valid_r] = (x[valid_r]*px[valid_r] + y[valid_r]*py[valid_r]) / r[valid_r]
    return radial_momentum >= 0


# ============================================================================
# Event weight
# ============================================================================

def calculate_weight_from_files(files: list[str], sample_size: Optional[int] = None) -> float:
    """Compute the per-event weight 1 / (number of NCs) stored in the 'weights' dataset.

    If sample_size is given, the NC count is extrapolated from the first
    sample_size sim1 files. The weight is constant and not used by the
    downstream tools.
    """
    print("Computing weight...")
    unique_triplets = set()
    files_to_sample = files[:sample_size] if sample_size else files

    for file_idx, file in enumerate(files_to_sample):
        try:
            with h5py.File(file, 'r') as f:
                evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
                nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
                triplets = [(file_idx, int(evt), int(nc)) for evt, nc in zip(evtid, nC_id)]
                unique_triplets.update(triplets)
        except Exception as e:
            print(f"Warning: could not read {file} for the weight calculation: {e}")
            continue

    if sample_size and sample_size < len(files):
        scale_factor = len(files) / sample_size
        estimated = len(unique_triplets) * scale_factor
        weight = 1 / estimated if estimated > 0 else 1.0
        print(f"Estimated weight based on {sample_size} files: {weight:.2e}")
    else:
        weight = 1 / len(unique_triplets) if unique_triplets else 1.0
        print(f"Weight based on all {len(files)} files: {weight:.2e}")
    return weight


# ============================================================================
# Phase 1: Load the NC data from sim1
# ============================================================================

def load_all_nc_data_from_sim1(
    sim1_files: list[str],
    material_mapping_path: str,
    volume_mapping_path: str,
    nested: bool = False
) -> dict[tuple[int, int, int], dict]:
    """
    Load all NCs from all sim1 files, with material and volume IDs translated to global IDs.

    Returns:
        nc_data_dict: {(run_id, muon_id, nC_id): {nc_info...}}

    The keys include run_id so that NCs are unique across runs. If a key occurs
    twice, only the first occurrence is kept and a warning is printed.
    """
    print(f"\n=== Phase 1: loading NC data from {len(sim1_files)} sim1 files ===")
    phase1_start = time.time()
    nc_data_dict = {}
    duplicate_count = 0

    def _get_run_id(file_path: str) -> int:
        """Extract the run_id from a path like .../run_003/output_t0.hdf5."""
        parts = Path(file_path).parts
        for part in parts:
            if part.startswith("run_"):
                try:
                    return int(part.split("_")[1])
                except (IndexError, ValueError):
                    pass
        return 0

    for file_idx, file_path in enumerate(sim1_files):
        run_id = _get_run_id(file_path) if nested else 0
        print(f"  Loading sim1 file {file_idx + 1}/{len(sim1_files)}: {os.path.basename(file_path)} (run_id={run_id})")

        try:
            with h5py.File(file_path, 'r') as f:
                nc_out = f['hit']['MyNeutronCaptureOutput']

                nc_evtid = nc_out['evtid']['pages'][:]
                nc_nC_id = nc_out['nC_track_id']['pages'][:]
                nC_x = nc_out['nC_x_position_in_m']['pages'][:] * 1000
                nC_y = nc_out['nC_y_position_in_m']['pages'][:] * 1000
                nC_z = nc_out['nC_z_position_in_m']['pages'][:] * 1000
                gamma_amount = nc_out['nC_gamma_amount']['pages'][:]
                gamma_tot_energy = nc_out['nC_gamma_total_energy_in_keV']['pages'][:]
                nc_material_ids = nc_out['nC_material_id']['pages'][:]
                nc_volume_ids = nc_out['nC_phys_vol_id']['pages'][:]
                nc_time = nc_out['nC_time_in_ns']['pages'][:]
                nc_flag_ge77 = nc_out['nC_flag_Ge77']['pages'][:]

                # Capture gammas 1-4 (energy and momentum direction)
                gamma_data_nc = {}
                for i in range(1, 5):
                    gamma_data_nc[f'gamma{i}_px'] = nc_out[f'gamma{i}_px']['pages'][:]
                    gamma_data_nc[f'gamma{i}_py'] = nc_out[f'gamma{i}_py']['pages'][:]
                    gamma_data_nc[f'gamma{i}_pz'] = nc_out[f'gamma{i}_pz']['pages'][:]
                    gamma_data_nc[f'gamma{i}_E_in_keV'] = nc_out[f'gamma{i}_E_in_keV']['pages'][:]

                mapping_names = [x.decode() for x in f["hit/materials/materialNames"]["pages"][:]]
                mapping_ids = f["hit/materials/materialsID"]["pages"][:]
                local_mat_map = dict(zip(mapping_ids, mapping_names))
                globalMaterialIDs, _ = remapMaterialIDsToGlobal(
                    material_mapping_path, local_mat_map, nc_material_ids
                )

                volume_names = [x.decode() for x in f["hit/physVolumes/physVolumeNames"]["pages"][:]]
                volume_ids = f["hit/physVolumes/physVolumesID"]["pages"][:]
                local_vol_map = dict(zip(volume_ids, volume_names))
                globalVolumeIDs, _ = remapVolumeIDsToGlobal(
                    volume_mapping_path, local_vol_map, nc_volume_ids
                )

                for idx in range(len(nc_evtid)):
                    muon_id = int(nc_evtid[idx])
                    nc_id = int(nc_nC_id[idx])

                    key = (run_id, muon_id, nc_id)

                    if key in nc_data_dict:
                        duplicate_count += 1
                        continue

                    nc_data_dict[key] = {
                        'file_idx': file_idx,
                        'run_id': run_id,
                        'nC_x': nC_x[idx],
                        'nC_y': nC_y[idx],
                        'nC_z': nC_z[idx],
                        'nC_time': nc_time[idx],
                        'nC_flag_Ge77': int(nc_flag_ge77[idx]),
                        'gamma_amount': gamma_amount[idx],
                        'gamma_tot_energy': gamma_tot_energy[idx],
                        'material_id': globalMaterialIDs[idx],
                        'volume_id': globalVolumeIDs[idx],
                        'gamma_energies': [gamma_data_nc[f'gamma{i}_E_in_keV'][idx] for i in range(1, 5)],
                        'gamma_px': [gamma_data_nc[f'gamma{i}_px'][idx] for i in range(1, 5)],
                        'gamma_py': [gamma_data_nc[f'gamma{i}_py'][idx] for i in range(1, 5)],
                        'gamma_pz': [gamma_data_nc[f'gamma{i}_pz'][idx] for i in range(1, 5)],
                    }

        except Exception as e:
            print(f"  ERROR loading {file_path}: {e}")
            raise

    if duplicate_count > 0:
        print(f"  ⚠ WARNING: {duplicate_count} duplicate (run_id, muon_id, nC_id) triplets!")

    phase1_elapsed = time.time() - phase1_start
    print(f"  ✓ {len(nc_data_dict)} unique NC events loaded in {phase1_elapsed/60:.1f} min")
    print_memory_usage("after Phase 1")
    return nc_data_dict

# ============================================================================
# Phase 2: Scan all optical files (sim2 + all sim3) and aggregate the voxel hits
# ============================================================================

def get_chunk_size() -> int:
    """Number of photons read per chunk, chosen from the available memory."""
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    if available_mem_gb > 400:
        return 50000
    elif available_mem_gb > 200:
        return 30000
    elif available_mem_gb > 50:
        return 20000
    elif available_mem_gb > 30:
        return 15000
    elif available_mem_gb > 20:
        return 10000
    else:
        return 5000


def scan_optical_and_aggregate(
    combined_files_by_run: dict[int, list[str]],
    nc_data_dict: dict[tuple[int, int, int], dict],
    voxel_tree: Optional[KDTree],
    voxel_data: list[dict],
    geometry_params: dict,
) -> tuple[dict[tuple[int, int, int], Counter], int, int, set[int]]:
    """
    Scan all optical files (sim2 + every sim3 sim_***) run by run and count the
    selected photons per NC and voxel.

    The hits of sim2 and all sim3 simulations are accumulated in the same
    counters, so the result is the sum over all optical simulations. Photon
    selection and NC matching are the same as in
    musun2SimFullPostprocessing.scan_sim2_and_aggregate().

    Args:
        combined_files_by_run: {run_id: [all optical HDF5 files of this run]}
                               (sim2 files + all sim3 sim_*** files)
        nc_data_dict: {(run_id, muon_id, nC_id): nc_info}
        voxel_tree: KD-tree on the voxel centers
        voxel_data: voxel definitions (same order as the tree)
        geometry_params: r_zylinder, h_zylinder and valid_detectors

    Returns:
        nc_voxel_counters: {(run_id, muon_id, nC_id): Counter} (summed)
        total_unassigned, total_orphaned, successfully_scanned_runs
    """
    total_optical_files = sum(len(v) for v in combined_files_by_run.values())
    print(f"\n=== Phase 2: scanning {total_optical_files} optical files in {len(combined_files_by_run)} runs ===")
    print(f"  (hits of sim2 and all sim3 sim_*** directories are summed)")

    h_zylinder = geometry_params['h_zylinder']
    r_zylinder = geometry_params['r_zylinder']
    valid_detectors = geometry_params.get('valid_detectors', [1965, 1966, 1967, 1968])
    valid_detectors_array = np.array(valid_detectors, dtype=np.int32)

    # z-coordinates (mm) that separate bottom, barrel and top of the SSD
    z_cut_bot = -4979
    z_cut_top = z_cut_bot + h_zylinder - 2

    nc_voxel_counters: dict[tuple[int, int, int], Counter] = defaultdict(Counter)
    # Every selected photon is assigned to its nearest voxel, so total_unassigned
    # stays 0; it is kept for the log output.
    total_unassigned = 0
    total_orphaned = 0

    CHUNK_SIZE = get_chunk_size()
    print(f"  Chunk size: {CHUNK_SIZE}")

    file_counter = 0
    phase2_start = time.time()

    for run_id in sorted(combined_files_by_run.keys()):
        optical_files = combined_files_by_run[run_id]
        print(f"\n  --- Run {run_id}: {len(optical_files)} optical files (sim2 + sim3 summed) ---")

        # NC time lookup for this run only (muon/NC IDs are unique only within a run)
        nc_times_for_run: dict[tuple[int, int], float] = {
            (muon_id, nc_id): info['nC_time']
            for (rid, muon_id, nc_id), info in nc_data_dict.items()
            if rid == run_id
        }
        nc_keys_for_run = {
            (muon_id, nc_id)
            for (rid, muon_id, nc_id) in nc_data_dict.keys()
            if rid == run_id
        }

        for opt_idx, opt_file in enumerate(optical_files):
            file_counter += 1
            print(f"    Optical file {opt_idx + 1}/{len(optical_files)} (global {file_counter}/{total_optical_files}): {os.path.basename(opt_file)}")
            file_unassigned = 0
            file_orphaned = 0
            file_matched = 0

            try:
                with h5py.File(opt_file, 'r') as f:
                    total_photons = len(f['hit']['optical']['x_position_in_m']['pages'])

                    if total_photons == 0:
                        print(f"      No optical photons, skipping.")
                        continue

                    num_chunks = (total_photons - 1) // CHUNK_SIZE + 1

                    for chunk_idx in range(num_chunks):
                        chunk_start = chunk_idx * CHUNK_SIZE
                        chunk_end = min(chunk_start + CHUNK_SIZE, total_photons)

                        # Positions converted from m to mm
                        x = np.array(f['hit']['optical']['x_position_in_m']['pages'][chunk_start:chunk_end], dtype=np.float32) * 1000
                        y = np.array(f['hit']['optical']['y_position_in_m']['pages'][chunk_start:chunk_end], dtype=np.float32) * 1000
                        z = np.array(f['hit']['optical']['z_position_in_m']['pages'][chunk_start:chunk_end], dtype=np.float32) * 1000
                        px = np.array(f['hit']['optical']['x_momentum_direction']['pages'][chunk_start:chunk_end], dtype=np.float32)
                        py = np.array(f['hit']['optical']['y_momentum_direction']['pages'][chunk_start:chunk_end], dtype=np.float32)
                        pz = np.array(f['hit']['optical']['z_momentum_direction']['pages'][chunk_start:chunk_end], dtype=np.float32)

                        muon_ids = f['hit']['optical']['muon_track_id']['pages'][chunk_start:chunk_end]
                        nc_ids = f['hit']['optical']['nC_track_id']['pages'][chunk_start:chunk_end]
                        det_uids = f['hit']['optical']['det_uid']['pages'][chunk_start:chunk_end]
                        photon_times = f['hit']['optical']['time_in_ns']['pages'][chunk_start:chunk_end]

                        # Filter 1 (detector): keep only photons on the SSD surfaces
                        det_mask = np.isin(det_uids, valid_detectors_array)
                        x, y, z = x[det_mask], y[det_mask], z[det_mask]
                        px, py, pz = px[det_mask], py[det_mask], pz[det_mask]
                        muon_ids = muon_ids[det_mask]
                        nc_ids = nc_ids[det_mask]
                        photon_times = photon_times[det_mask]

                        if len(x) == 0:
                            continue

                        # Filter 2 (time window): capture time of the NC that produced
                        # each photon (inf = no matching NC, "orphaned" photon)
                        nc_times_arr = np.full(len(photon_times), np.inf, dtype=np.float64)
                        for idx in range(len(muon_ids)):
                            local_key = (int(muon_ids[idx]), int(nc_ids[idx]))
                            if local_key in nc_times_for_run:
                                nc_times_arr[idx] = nc_times_for_run[local_key]

                        has_nc = nc_times_arr != np.inf
                        file_orphaned += int(np.sum(~has_nc))

                        relative_times = np.where(has_nc, photon_times - nc_times_arr, np.inf)

                        # More than 1 ns before the capture cannot be a rounding artifact: abort.
                        negative_mask = has_nc & (relative_times < -1.0)
                        if np.any(negative_mask):
                            neg_indices = np.where(negative_mask)[0]
                            first_bad = neg_indices[0]
                            raise RuntimeError(
                                f"FATAL: negative relative photon time detected!\n"
                                f"  Optical file: {opt_file}, run_id={run_id}\n"
                                f"  muon_id={int(muon_ids[first_bad])}, nC_id={int(nc_ids[first_bad])}\n"
                                f"  photon_time={float(photon_times[first_bad]):.4f} ns, "
                                f"nC_time={float(nc_times_arr[first_bad]):.4f} ns\n"
                                f"  relative_time={float(relative_times[first_bad]):.4f} ns\n"
                                f"  {int(np.sum(negative_mask))} photons with negative relative time."
                            )

                        # Clamp rounding artifacts in [-1 ns, 0) to 0
                        rounding_mask = has_nc & (relative_times < 0.0)
                        if np.any(rounding_mask):
                            relative_times[rounding_mask] = 0.0

                        time_mask = has_nc & (relative_times >= 0) & (relative_times <= 200.0)
                        x, y, z = x[time_mask], y[time_mask], z[time_mask]
                        px, py, pz = px[time_mask], py[time_mask], pz[time_mask]
                        muon_ids = muon_ids[time_mask]
                        nc_ids = nc_ids[time_mask]

                        if len(x) == 0:
                            continue

                        # Filter 3 (direction): keep photons that hit the surface from
                        # inside the tank: moving down at the bottom, up at the top,
                        # outwards at the wall
                        mask_bot = z <= z_cut_bot
                        mask_top = z >= z_cut_top
                        mask_barrel = ~mask_bot & ~mask_top

                        final_mask = np.zeros(len(z), dtype=bool)
                        final_mask[mask_bot] = pz[mask_bot] <= 0
                        final_mask[mask_top] = pz[mask_top] >= 0
                        if np.any(mask_barrel):
                            final_mask[mask_barrel] = checkRadialMomentumVectorized(
                                x[mask_barrel], y[mask_barrel], z[mask_barrel],
                                px[mask_barrel], py[mask_barrel], pz[mask_barrel]
                            )

                        x, y, z = x[final_mask], y[final_mask], z[final_mask]
                        muon_ids = muon_ids[final_mask]
                        nc_ids = nc_ids[final_mask]

                        if len(x) == 0:
                            continue

                        file_matched += len(x)

                        # Voxel assignment: one batched KD-tree query for the whole chunk
                        points = np.column_stack((x, y, z))
                        _, voxel_idx_array = voxel_tree.query(points)
                        voxel_index_strings = np.array(
                            [str(voxel_data[vi]['index']) for vi in voxel_idx_array]
                        )

                        # Group the photons by NC and add them to the NC's hit counter
                        muon_ids_int = muon_ids.astype(np.int64)
                        nc_ids_int = nc_ids.astype(np.int64)
                        paired = np.stack([muon_ids_int, nc_ids_int], axis=1)
                        unique_pairs, inverse = np.unique(paired, axis=0, return_inverse=True)

                        for pair_idx, (m_id, n_id) in enumerate(unique_pairs):
                            full_key = (run_id, int(m_id), int(n_id))
                            mask = inverse == pair_idx

                            if full_key not in nc_data_dict:
                                file_orphaned += int(np.sum(mask))
                                continue

                            voxel_strs = voxel_index_strings[mask]
                            for vs in voxel_strs:
                                nc_voxel_counters[full_key][vs] += 1

                        del x, y, z, px, py, pz, muon_ids, nc_ids, det_uids, photon_times
                        del points, voxel_idx_array, voxel_index_strings, paired
                        gc.collect()

            except RuntimeError:
                raise
            except Exception as e:
                print(f"      ERROR processing {opt_file}: {e}")
                raise

            total_unassigned += file_unassigned
            total_orphaned += file_orphaned
            print(f"      Matched: {file_matched}, orphaned: {file_orphaned}, unassigned: {file_unassigned}")

        run_elapsed = time.time() - phase2_start
        print(f"  Run {run_id} finished after {run_elapsed/60:.1f} min (cumulative)")
        if file_counter % 50 == 0:
            print_memory_usage(f"after {file_counter} optical files")

    ncs_with_photons = sum(1 for c in nc_voxel_counters.values() if c)
    ncs_without_photons = len(nc_data_dict) - ncs_with_photons
    total_hits = sum(sum(c.values()) for c in nc_voxel_counters.values())

    phase2_elapsed = time.time() - phase2_start
    print(f"\n  ✓ Optical scan finished in {phase2_elapsed/60:.1f} min:")
    print(f"    NC events with photons: {ncs_with_photons}")
    print(f"    NC events without photons: {ncs_without_photons}")
    print(f"    Total voxel hits (summed over all simulations): {total_hits}")
    print(f"    Unassigned photons: {total_unassigned}")
    print(f"    Orphaned photons (no NC): {total_orphaned}")
    print_memory_usage("after Phase 2")

    successfully_scanned_runs = set(combined_files_by_run.keys())

    return dict(nc_voxel_counters), total_unassigned, total_orphaned, successfully_scanned_runs


# ============================================================================
# Phase 3: Merge NC data and hits, write the output
# ============================================================================

def write_output_in_batches(
    nc_data_dict: dict[tuple[int, int, int], dict],
    nc_voxel_counters: dict[tuple[int, int, int], Counter],
    voxel_data: list[dict],
    voxel_indices: list[int],
    geometry_params: dict,
    output_file: str,
    weight: float,
    batch_size: int = 10000,
    progress_tracker: Optional["ProgressTracker"] = None,
    skip_keys: frozenset = frozenset(),
) -> dict:
    """
    Build the output rows batch by batch and append them directly to the HDF5 file.

    Writing in batches keeps the memory usage bounded for large datasets.

    Args:
        nc_data_dict: NC data {(run_id, muon_id, nC_id): nc_info}
        nc_voxel_counters: aggregated hits {(run_id, muon_id, nC_id): Counter}
        voxel_data: voxel definitions
        voxel_indices: voxel indices in column order of target_matrix
        geometry_params: geometry parameters
        output_file: path of the HDF5 file
        weight: per-event weight written to 'weights'
        batch_size: number of NC events per batch
        progress_tracker: if given, the committed row count is saved after each batch
        skip_keys: NC keys already present in the output (resume); they are not written again

    Returns:
        dict with nc_with_photons, nc_without_photons, total_written (rows written in this call)
    """
    print(f"\n=== Phase 3: writing output in batches for {len(nc_data_dict)} NC events ===")
    phase3_start = time.time()
    print(f"  Batch size: {batch_size}")

    r_zylinder = geometry_params['r_zylinder']
    h_zylinder = geometry_params['h_zylinder']
    z_cut_bot = -4979
    z_cut_top = z_cut_bot + h_zylinder - 2

    nc_without_photons = 0
    nc_with_photons = 0
    total_written = 0

    # On resume, NCs that are already in the output file are skipped.
    all_keys = [key for key in nc_data_dict.keys() if key not in skip_keys]
    if skip_keys:
        print(f"  Resume: {len(nc_data_dict) - len(all_keys)} NC events already written, "
              f"{len(all_keys)} remaining")
    num_batches = (len(all_keys) - 1) // batch_size + 1

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(all_keys))
        batch_keys = all_keys[batch_start:batch_end]

        phi_data_batch = []
        target_data_batch = []
        target_regions_batch = []
        event_ids_batch = []

        for key in batch_keys:
            run_id, muon_id, nc_id = key
            nc_info = nc_data_dict[key]
            voxel_counter = nc_voxel_counters.get(key, Counter())

            if not voxel_counter:
                nc_without_photons += 1
            else:
                nc_with_photons += 1

            x = nc_info['nC_x']
            y = nc_info['nC_y']
            z = nc_info['nC_z']
            mat_id = nc_info['material_id']
            vol_id = nc_info['volume_id']
            n_gamma = max(0, nc_info['gamma_amount'])
            e_tot = max(0, nc_info['gamma_tot_energy'])

            # Up to 4 capture gammas; negative energies mark missing gammas and are zeroed
            gamma_row = []
            for i in range(4):
                e = nc_info['gamma_energies'][i]
                px = nc_info['gamma_px'][i]
                py = nc_info['gamma_py'][i]
                pz = nc_info['gamma_pz'][i]
                if e < 0:
                    e, px, py, pz = 0.0, 0.0, 0.0, 0.0
                gamma_row.extend([e, px, py, pz])

            # Cylindrical coordinates and signed distances to the SSD (positive inside)
            r_NC = np.sqrt(x**2 + y**2)
            phi_NC = np.arctan2(y, x)
            dist_to_wall = r_zylinder - r_NC
            dist_to_bot = z - z_cut_bot
            dist_to_top = z_cut_top - z

            # Mean gamma momentum direction in cylindrical components (gammas with E > 0 only)
            p_r_values = []
            p_z_values = []
            for i in range(4):
                e = nc_info['gamma_energies'][i]
                if e > 0:
                    px = nc_info['gamma_px'][i]
                    py = nc_info['gamma_py'][i]
                    pz = nc_info['gamma_pz'][i]
                    p_r_values.append(np.sqrt(px**2 + py**2))
                    p_z_values.append(pz)

            p_mean_r = np.mean(p_r_values) if p_r_values else 0.0
            p_mean_z = np.mean(p_z_values) if p_z_values else 0.0

            nC_time = nc_info['nC_time']
            flag_ge77 = nc_info['nC_flag_Ge77']

            # Column order must match phi_columns in create_or_open_output_file()
            phi_row = [x, y, z, mat_id, vol_id, n_gamma, e_tot,
                       r_NC, phi_NC, dist_to_wall, dist_to_bot, dist_to_top,
                       p_mean_r, p_mean_z,
                       nC_time, flag_ge77] + gamma_row

            target_row = [voxel_counter.get(str(voxel_idx), 0) for voxel_idx in voxel_indices]

            # Hits per detector region (pit, bot, wall, top)
            region_hits = {'pit': 0, 'bot': 0, 'wall': 0, 'top': 0}
            for voxel_idx_str, count in voxel_counter.items():
                voxel_info = next((v for v in voxel_data if str(v['index']) == voxel_idx_str), None)
                if voxel_info and 'layer' in voxel_info:
                    layer = voxel_info['layer'].lower()
                    if layer in region_hits:
                        region_hits[layer] += count

            target_regions_row = [region_hits['pit'], region_hits['bot'],
                                  region_hits['wall'], region_hits['top']]

            phi_data_batch.append(phi_row)
            target_data_batch.append(target_row)
            target_regions_batch.append(target_regions_row)
            event_ids_batch.append([run_id, muon_id, nc_id])

        batch_result = {
            'phi_data': np.array(phi_data_batch, dtype=np.float32),
            'target_data': np.array(target_data_batch, dtype=np.int32),
            'target_regions': np.array(target_regions_batch, dtype=np.int32),
            'event_ids': np.array(event_ids_batch, dtype=np.int64),
        }

        entries = append_results_to_hdf5(output_file, batch_result, voxel_indices, weight)
        total_written += entries

        # Record the batch only after it is completely written, so that a
        # restart can discard rows of a batch that was interrupted mid-write.
        if progress_tracker is not None:
            progress_tracker.progress_data['rows_committed'] += entries
            progress_tracker.save_progress()

        del phi_data_batch, target_data_batch, target_regions_batch, event_ids_batch, batch_result
        gc.collect()

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"  Batch {batch_idx + 1}/{num_batches}: {total_written} entries written")

    phase3_elapsed = time.time() - phase3_start
    print(f"\n  ✓ Writing finished in {phase3_elapsed/60:.1f} min:")
    print(f"    NC events with photons: {nc_with_photons}")
    print(f"    NC events without photons: {nc_without_photons}")
    print(f"    Total written: {total_written}")
    print_memory_usage("after writing in batches")

    return {
        'nc_with_photons': nc_with_photons,
        'nc_without_photons': nc_without_photons,
        'total_written': total_written,
    }

# ============================================================================
# HDF5 Output
# ============================================================================

def create_or_open_output_file(output_path, file_index, voxel_data, mat_map, vol_map, radius, n_primaries=0, suffix=""):
    """Create the output HDF5 file, or return the path of an existing one.

    Layout (one row per NC in all row-aligned datasets):
      phi_matrix      (n_events, 32)       float32  NC features, names in phi_columns
      target_matrix   (n_events, n_voxels) int32    photon hits per voxel, voxel indices in target_columns
      region_matrix   (n_events, 4)        int32    photon hits per region, names in region_columns
      event_ids       (n_events, 3)        int64    (run_id, muon_id, nc_id), names in event_id_columns
      weights         (n_events,)          float32  constant 1/N_NC (not used downstream)
    Static metadata: voxel geometry (voxels/, voxel_metadata), the global
    material/volume mappings (mat_map/, vol_map/), the SSD radius
    (theta/inner_radius_in_mm) and the number of simulated muons (primaries),
    which is needed downstream to normalize Ge77 rates to all muons, because
    the simulation output only contains muons that interacted.
    """
    output_file = os.path.join(output_path, f"ncscore_output_{file_index}{suffix}.hdf5")

    if os.path.exists(output_file):
        try:
            with h5py.File(output_file, 'r') as f:
                if ('phi_matrix' in f and 'target_matrix' in f and
                        'voxels' in f and 'voxel_metadata' in f and 'event_ids' in f):
                    print(f"Existing output file found: {output_file}")
                    return output_file
        except:
            print(f"Existing file is corrupt, creating a new one: {output_file}")
            os.remove(output_file)

    print(f"Creating new output file: {output_file}")

    voxel_indices = [voxel['index'] for voxel in voxel_data]
    n_voxels = len(voxel_indices)

    phi_columns = ["xNC_mm", "yNC_mm", "zNC_mm", "matID", "volID", "#gamma", "E_gamma_tot_keV",
                   "r_NC_mm", "phi_NC_rad", "dist_to_wall_mm", "dist_to_bot_mm", "dist_to_top_mm",
                   "p_mean_r", "p_mean_z",
                   "nC_time_in_ns", "nC_flag_Ge77",
                   "gammaE1_keV", "gammapx1", "gammapy1", "gammapz1",
                   "gammaE2_keV", "gammapx2", "gammapy2", "gammapz2",
                   "gammaE3_keV", "gammapx3", "gammapy3", "gammapz3",
                   "gammaE4_keV", "gammapx4", "gammapy4", "gammapz4"]
    region_columns = ['pit', 'bot', 'wall', 'top']
    event_id_columns = ["run_id", "muon_id", "nc_id"]

    # Force 64-bit offsets and lengths in the superblock (for files > 2 GB).
    # See: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t11.html
    # (superblock fields "Size of Offsets" and "Size of Lengths" set to 8 bytes)
    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fcpl.set_sizes(8, 8)  # 8 bytes = 64 bit for offsets and lengths
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fid = h5py.h5f.create(output_file.encode(), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
    with h5py.File(fid) as out:
        # === Resizable 2D datasets (one row per NC) ===
        out.create_dataset(
            "target_matrix",
            shape=(0, n_voxels),
            maxshape=(None, n_voxels),
            dtype=np.int32,
            chunks=(min(1000, max(1, n_voxels)), n_voxels),
            compression='lzf'
        )
        out.create_dataset(
            "phi_matrix",
            shape=(0, len(phi_columns)),
            maxshape=(None, len(phi_columns)),
            dtype=np.float32,
            chunks=(1000, len(phi_columns)),
            compression='lzf'
        )
        out.create_dataset(
            "region_matrix",
            shape=(0, len(region_columns)),
            maxshape=(None, len(region_columns)),
            dtype=np.int32,
            chunks=(1000, len(region_columns)),
            compression='lzf'
        )

        # === Event IDs (run_id, muon_id, nc_id) ===
        out.create_dataset(
            "event_ids",
            shape=(0, len(event_id_columns)),
            maxshape=(None, len(event_id_columns)),
            dtype=np.int64,
            chunks=(1000, len(event_id_columns)),
            compression='lzf'
        )

        # === Column names (column index -> name) ===
        dt = h5py.string_dtype(encoding='utf-8')
        out.create_dataset("target_columns", data=[str(idx) for idx in voxel_indices], dtype=dt)
        out.create_dataset("phi_columns", data=phi_columns, dtype=dt)
        out.create_dataset("region_columns", data=region_columns, dtype=dt)
        out.create_dataset("event_id_columns", data=event_id_columns, dtype=dt)

        # === Static data ===
        theta_grp = out.create_group("theta")
        theta_grp.create_dataset("inner_radius_in_mm", data=radius)
        out.create_dataset("primaries", data=n_primaries)

        mat_map_grp = out.create_group("mat_map")
        for key, value in mat_map.items():
            if key == "":
                key = "noMaterial"
            mat_map_grp.create_dataset(str(key), data=int(value))

        vol_map_grp = out.create_group("vol_map")
        for key, value in vol_map.items():
            if key == "":
                key = "noVolume"
            vol_map_grp.create_dataset(str(key), data=int(value))

        out.create_dataset(
            "weights",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=True,
            compression='lzf'
        )

        # === Voxel geometry: center, layer and corners of every voxel ===
        voxels_grp = out.create_group("voxels")
        print(f"  Initializing metadata of {len(voxel_data)} voxels...")
        for voxel in voxel_data:
            if isinstance(voxel, dict):
                voxel_idx = voxel['index']
                voxel_grp = voxels_grp.create_group(str(voxel_idx))
                voxel_grp.create_dataset("center", data=np.array(voxel['center'], dtype='f'))
                voxel_grp.create_dataset("layer", data=voxel['layer'], dtype=dt)

                corners = np.array(voxel['corners'])
                corners_grp = voxel_grp.create_group("corners")
                corners_grp.create_dataset("x", data=corners[:, 0])
                corners_grp.create_dataset("y", data=corners[:, 1])
                corners_grp.create_dataset("z", data=corners[:, 2])

        print(f"  ✓ Metadata of all {len(voxel_data)} voxels created")

        # voxel_metadata: one row per voxel, in the column order of target_matrix:
        # [is_pit, is_bot, is_wall, is_top, r, phi, z] of the voxel center
        voxel_metadata = []
        for voxel in voxel_data:
            center = np.array(voxel['center'], dtype=np.float32)
            layer = voxel.get('layer', '').lower()
            r_voxel = np.sqrt(center[0]**2 + center[1]**2)
            phi_voxel = np.arctan2(center[1], center[0])
            z_voxel = center[2]
            is_pit = 1.0 if layer == 'pit' else 0.0
            is_bot = 1.0 if layer == 'bot' else 0.0
            is_wall = 1.0 if layer == 'wall' else 0.0
            is_top = 1.0 if layer == 'top' else 0.0
            voxel_metadata.append([is_pit, is_bot, is_wall, is_top, r_voxel, phi_voxel, z_voxel])

        voxel_metadata = np.array(voxel_metadata, dtype=np.float32)
        out.create_dataset("voxel_metadata", data=voxel_metadata, dtype=np.float32, compression='lzf')
        print(f"  ✓ Voxel metadata created: shape {voxel_metadata.shape}")

    return output_file


def append_results_to_hdf5(output_file, result_data, voxel_indices, weight):
    """Append rows to the resizable output datasets; returns the number of rows written."""
    phi_data = result_data['phi_data']
    target_data = result_data['target_data']
    target_regions_data = result_data.get('target_regions')
    event_ids_data = result_data.get('event_ids')
    num_entries = len(phi_data)

    if num_entries == 0:
        return 0

    weights = np.full(num_entries, weight, dtype=np.float32)

    with h5py.File(output_file, 'a') as out:
        dset = out['phi_matrix']
        old_size = dset.shape[0]
        new_size = old_size + num_entries
        dset.resize((new_size, dset.shape[1]))
        dset[old_size:new_size, :] = phi_data

        dset = out['target_matrix']
        dset.resize((new_size, dset.shape[1]))
        dset[old_size:new_size, :] = target_data

        if target_regions_data is not None and len(target_regions_data) > 0:
            dset = out['region_matrix']
            dset.resize((new_size, dset.shape[1]))
            dset[old_size:new_size, :] = target_regions_data

        if event_ids_data is not None and len(event_ids_data) > 0:
            dset = out['event_ids']
            dset.resize((new_size, dset.shape[1]))
            dset[old_size:new_size, :] = event_ids_data

        dset = out['weights']
        dset.resize((new_size,))
        dset[old_size:new_size] = weights

        out.flush()

    return num_entries


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Post-processing: neutron captures + several optical simulations (summed)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Three-source workflow:
  -i1: NC data (sim1, run_NNN/ subdirectories)
  -i2: first optical simulation (run_NNN/ subdirectories, one simulation per run)
  -i3: further optical simulations (run_NNN/sim_NNN/ structure)

The optical hits of -i2 and of ALL sim_NNN in -i3 are summed for every NC event.

Example:
  python %(prog)s \\
    -i1 /path/to/sim1 \\
    -i2 /path/to/sim2 \\
    -i3 /path/to/sim3_multi \\
    -o  /path/to/output \\
    --nested --muons-per-run 1000
        """
    )

    parser.add_argument('-i1', '--input-sim1', required=True,
                        help='Input directory of simulation 1 (NC data)')
    parser.add_argument('-i2', '--input-sim2', required=True,
                        help='Input directory of simulation 2 (first optical simulation, run_NNN/)')
    parser.add_argument('-i3', '--input-sim3', required=True,
                        help='Input directory of the further optical simulations (run_NNN/sim_NNN/)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory')

    # Mapping and voxel files. The defaults point to the author's cluster
    # installation; pass the files in src/mappings and src/voxels explicitly elsewhere.
    parser.add_argument('-m', '--material-mapping',
                        default='/global/cfs/projectdirs/legend/users/tbuerger/postprocessing/src/mappings/globalMaterialMappings.json',
                        help='Global material mapping (JSON, extended in place with new materials)')
    parser.add_argument('-v', '--volume-mapping',
                        default='/global/cfs/projectdirs/legend/users/tbuerger/postprocessing/src/mappings/globalPhysVolMappings.json',
                        help='Global physical volume mapping (JSON, extended in place with new volumes)')
    parser.add_argument('-x', '--voxel-file',
                        default='/global/cfs/projectdirs/legend/users/tbuerger/postprocessing/src/voxels/currentDistZylVoxelsPMTSize.json',
                        help='Voxel definition of the SSD (JSON)')

    parser.add_argument('--reset', action='store_true',
                        help='Start from scratch (deletes existing output, progress and scan checkpoint)')
    parser.add_argument('--sample-weight', type=int, default=50,
                        help='Number of sim1 files used to estimate the event weight (0 = all)')
    parser.add_argument('--nested', action='store_true',
                        help='Search for HDF5 files in run_NNN subdirectories (applies to sim1 and sim2)')

    parser.add_argument('-g', '--geometry', default='currentDist',
                        choices=['currentDist'],
                        help='Geometry name (default: currentDist)')
    parser.add_argument('--valid-detectors', type=int, nargs='+',
                        default=[1965, 1966, 1967, 1968],
                        help='Detector UIDs of the SSD surfaces')
    parser.add_argument('--verbose', '-V', action='store_true',
                        help='No effect (kept for compatibility)')
    parser.add_argument('--muons-per-run', type=int, required=True,
                        help='Number of muons each sim1 run (run_NNN, all thread files together) '
                             'was started with; stored as primaries = runs x muons-per-run')
    parser.add_argument('--sim3-runs', type=int, nargs='+', default=None,
                        metavar='RUN_ID',
                        help='Run IDs of sim3 to process (e.g. --sim3-runs 1 '
                             'or --sim3-runs 1 2 3). Only these runs are written to the '
                             'output; all others are skipped. Without this option, '
                             'all runs present in sim3 are used. All selected '
                             'runs must have the same set of sim_*** directories, '
                             'otherwise the script aborts.')

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_arguments()
    print_memory_usage("main start")

    for label, path in [('Sim1', args.input_sim1), ('Sim2', args.input_sim2),
                        ('Sim3', args.input_sim3)]:
        if not os.path.exists(path):
            print(f"Error: {label} path does not exist: {path}")
            sys.exit(1)
    if not os.path.exists(args.voxel_file):
        print(f"Error: voxel file does not exist: {args.voxel_file}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # === Validate the input structures ===
    print("\n=== Validating input structures ===")
    if args.nested:
        validate_run_structure(args.input_sim1, nested=True, label="Sim1")
        validate_run_structure(args.input_sim2, nested=True, label="Sim2")
        print("=== Checking consistency Sim1 ↔ Sim2 ===")
        validate_sim1_sim2_consistency(args.input_sim1, args.input_sim2)
    else:
        validate_run_structure(args.input_sim1, nested=False, label="Sim1")
        validate_run_structure(args.input_sim2, nested=False, label="Sim2")

    print("\n=== Validating Sim3 structure (run_NNN/sim_NNN/) ===")
    sim3_summary = validate_sim3_structure(args.input_sim3)

    # Resolve which runs to process
    all_sim3_run_ids = set(sim3_summary.keys())
    if args.sim3_runs is not None:
        active_run_ids = set(args.sim3_runs)
        print(f"\n--sim3-runs given: processing only run(s) {sorted(active_run_ids)}")
    else:
        active_run_ids = all_sim3_run_ids
        print(f"\nNo --sim3-runs given: processing all {len(active_run_ids)} Sim3 run(s)")

    print("=== Checking consistency Sim1 ↔ Sim3 (active runs only) ===")
    try:
        validate_sim1_sim3_consistency(args.input_sim1, args.input_sim3, active_run_ids)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    # === Geometry ===
    geometry_result = defineZylinder(args.geometry, args.valid_detectors)
    (t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top,
     z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref, valid_detectors) = geometry_result

    geometry_params = {
        'h_zylinder': h_zylinder,
        'r_zylinder': r_zylinder,
        'valid_detectors': valid_detectors,
    }

    # === Voxel data ===
    print("Loading voxel data...")
    with open(args.voxel_file, 'r') as f:
        voxel_data = json.load(f)
    voxel_tree = build_voxel_tree(voxel_data)
    voxel_indices = [voxel['index'] for voxel in voxel_data]
    print(f"Voxel tree with {len(voxel_data)} voxels built")

    # === Find the input files ===
    sim1_files_all = find_all_hdf5_files(args.input_sim1, args.nested)
    if not sim1_files_all:
        print("Error: no output_*.hdf5 files found in the sim1 path!")
        sys.exit(1)

    sim2_files_by_run_all = find_sim2_files(args.input_sim2, args.nested)
    if not sim2_files_by_run_all:
        print("Error: no output_t*.hdf5 files found in the sim2 path!")
        sys.exit(1)

    print("\nSearching for Sim3 files...")
    sim3_files_by_run_all = find_sim3_files(args.input_sim3)
    if not sim3_files_by_run_all:
        print("Error: no HDF5 files found in the sim3 path!")
        sys.exit(1)

    # === Restrict to the active runs ===
    if args.nested:
        sim1_files = [f for f in sim1_files_all
                      if _extract_run_id_from_path(f) in active_run_ids]
        skipped_sim1 = [f for f in sim1_files_all
                        if _extract_run_id_from_path(f) not in active_run_ids]
    else:
        sim1_files = sim1_files_all
        skipped_sim1 = []

    sim2_files_by_run = {rid: files for rid, files in sim2_files_by_run_all.items()
                         if rid in active_run_ids}
    sim3_files_by_run = {rid: files for rid, files in sim3_files_by_run_all.items()
                         if rid in active_run_ids}

    # === Summary of the run selection ===
    all_known_run_ids = (
        {_extract_run_id_from_path(f) for f in sim1_files_all}
        | set(sim2_files_by_run_all.keys())
        | set(sim3_files_by_run_all.keys())
    ) if args.nested else active_run_ids

    skipped_run_ids = all_known_run_ids - active_run_ids

    print(f"\n{'='*60}")
    print(f"RUN SELECTION")
    print(f"{'='*60}")
    print(f"  Active runs : {sorted(active_run_ids)}")
    if skipped_run_ids:
        print(f"  Skipped     : {sorted(skipped_run_ids)}  ← not in --sim3-runs, NOT written to the output")
    else:
        print(f"  Skipped     : none")
    print(f"{'='*60}")

    total_sim3_sims = sum(len(sims) for run_id, sims in sim3_summary.items()
                          if run_id in active_run_ids)
    total_sim2_files = sum(len(v) for v in sim2_files_by_run.values())
    total_sim3_files = sum(len(v) for v in sim3_files_by_run.values())
    print(f"Sim1: {len(sim1_files)} files (active runs)")
    print(f"Sim2: {total_sim2_files} files in {len(sim2_files_by_run)} active runs")
    print(f"Sim3: {total_sim3_files} files in {total_sim3_sims} sim_*** directories of {len(sim3_files_by_run)} active runs")

    # === Combine the sim2 and sim3 files ===
    combined_files_by_run = combine_optical_files(sim2_files_by_run, sim3_files_by_run)
    total_combined = sum(len(v) for v in combined_files_by_run.values())
    print(f"\nCombined optical files: {total_combined} ({total_sim2_files} Sim2 + {total_sim3_files} Sim3)")
    for run_id in sorted(combined_files_by_run.keys()):
        n2 = len(sim2_files_by_run.get(run_id, []))
        n3 = len(sim3_files_by_run.get(run_id, []))
        print(f"  run_{run_id:03d}: {n2} Sim2 + {n3} Sim3 = {n2+n3} optical files")

    # === Integrity check of all optical input files ===
    if not check_hdf5_integrity(combined_files_by_run, args.output):
        print("\nAborting: faulty HDF5 files found. See faulty_simulations.txt for the list.")
        sys.exit(1)

    # === Output file ===
    output_file = os.path.join(args.output, "ncscore_output_0.hdf5")
    progress_tracker = ProgressTracker(args.output, output_file)

    if args.reset:
        print("Reset requested - deleting existing results...")
        if os.path.exists(output_file):
            os.remove(output_file)
        progress_tracker.cleanup()
        progress_tracker = ProgressTracker(args.output, output_file)

    # === Resume logic ===
    # Settings that must be identical between an interrupted run and its resume.
    run_config = {
        'input_sim1': os.path.abspath(args.input_sim1),
        'input_sim2': os.path.abspath(args.input_sim2),
        'input_sim3': os.path.abspath(args.input_sim3),
        'active_runs': sorted(active_run_ids),
        'voxel_file': os.path.abspath(args.voxel_file),
        'nested': args.nested,
        'valid_detectors': list(valid_detectors),
        'muons_per_run': args.muons_per_run,
    }
    resuming = os.path.exists(output_file)
    if resuming:
        if is_output_complete(output_file):
            print(f"Output is already complete: {output_file}")
            print("Use --reset to recompute it.")
            return
        if not progress_tracker.progress_file.exists():
            print(f"Error: {output_file} exists but is incomplete and has no progress file, "
                  f"so it cannot be resumed safely. Use --reset to recompute it.")
            sys.exit(1)
        saved_config = progress_tracker.progress_data.get('config')
        if saved_config != run_config:
            print("Error: the settings differ from those of the interrupted run.")
            print(f"  saved  : {saved_config}")
            print(f"  current: {run_config}")
            print("Resume with the original settings or start over with --reset.")
            sys.exit(1)
        print(f"Resuming interrupted run "
              f"({progress_tracker.progress_data.get('rows_committed', 0)} rows already written)")
    else:
        # Fresh start: discard stale progress/checkpoint files of an earlier attempt.
        progress_tracker.cleanup()
        progress_tracker = ProgressTracker(args.output, output_file)
        progress_tracker.progress_data['config'] = run_config
        progress_tracker.save_progress()

    # === Global material & volume mappings (completed only for a new run) ===
    if not os.path.exists(output_file):
        all_materials = collect_all_materials_first(sim1_files)
        all_volumes = collect_all_volumes_first(sim1_files)
        global_material_mapping = create_global_material_mapping(all_materials, args.material_mapping)
        print(f"Global material mapping: {len(global_material_mapping)} materials")
        global_volume_mapping = create_global_volume_mapping(all_volumes, args.volume_mapping)
        print(f"Global volume mapping: {len(global_volume_mapping)} volumes")
    else:
        global_material_mapping = loadMapping(args.material_mapping)
        print(f"Existing material mapping loaded: {len(global_material_mapping)} materials")
        global_volume_mapping = loadMapping(args.volume_mapping)
        print(f"Existing volume mapping loaded: {len(global_volume_mapping)} volumes")

    # === Event weight (based on sim1 only, as in musun2SimFullPostprocessing.py) ===
    sample_size = args.sample_weight if args.sample_weight > 0 else None
    weight = calculate_weight_from_files(sim1_files, sample_size)

    # === Output file ===
    # --muons-per-run refers to one run_NNN simulation (all thread files together).
    # Only runs that also have optical data end up in the output.
    sim1_run_ids = {
        _extract_run_id_from_path(f) if args.nested else 0
        for f in sim1_files
    }
    n_runs = len(sim1_run_ids & set(combined_files_by_run))
    n_primaries = n_runs * args.muons_per_run
    output_file = create_or_open_output_file(
        args.output, 0, voxel_data, global_material_mapping,
        global_volume_mapping, r_zylinder, n_primaries=n_primaries
    )

    written_keys = set()
    if resuming:
        rows_committed, written_keys = prepare_output_for_resume(
            output_file, progress_tracker.progress_data.get('rows_committed', 0)
        )
        progress_tracker.progress_data['rows_committed'] = rows_committed
        progress_tracker.save_progress()

    # =======================================================================
    # Main processing
    # =======================================================================
    start_time = time.time()

    try:
        # Phase 1: load the NC data from sim1
        nc_data_dict = load_all_nc_data_from_sim1(
            sim1_files, args.material_mapping, args.volume_mapping,
            nested=args.nested
        )

        # Phase 2: scan all optical files (sim2 + sim3) and sum the hits
        # (skipped if a previous run already saved the scan result)
        checkpoint = progress_tracker.scan_checkpoint_file
        if progress_tracker.is_sim2_scan_complete() and checkpoint.exists():
            print(f"\n=== Phase 2: loading scan checkpoint {checkpoint} (scan skipped) ===")
            nc_voxel_counters, total_unassigned, total_orphaned, scanned_runs = load_scan_checkpoint(checkpoint)
        else:
            nc_voxel_counters, total_unassigned, total_orphaned, scanned_runs = scan_optical_and_aggregate(
                combined_files_by_run, nc_data_dict, voxel_tree, voxel_data, geometry_params
            )
            save_scan_checkpoint(checkpoint, nc_voxel_counters, total_unassigned,
                                 total_orphaned, scanned_runs)
            progress_tracker.mark_sim2_scan_complete()

        # Drop NCs of runs without optical data
        nc_keys_before = len(nc_data_dict)
        nc_data_dict = {
            key: val for key, val in nc_data_dict.items()
            if key[0] in scanned_runs
        }
        nc_dropped = nc_keys_before - len(nc_data_dict)
        if nc_dropped > 0:
            print(f"  ⚠ {nc_dropped} NC events of runs without optical data removed")

        # Phase 3: write the output in batches
        output_stats = write_output_in_batches(
            nc_data_dict, nc_voxel_counters, voxel_data, voxel_indices,
            geometry_params, output_file, weight, batch_size=10000,
            progress_tracker=progress_tracker, skip_keys=written_keys
        )

        total_entries = progress_tracker.progress_data['rows_committed']

        del nc_data_dict, nc_voxel_counters
        gc.collect()

        if progress_tracker.verify_hdf5_integrity(output_file):
            mark_output_complete(output_file)

    except KeyboardInterrupt:
        print(f"\nProcessing interrupted by user.")
        print("Progress has been saved. Run the same command again to resume.")
        return
    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback
        traceback.print_exc()
        return

    # =======================================================================
    # Final statistics
    # =======================================================================
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"PROCESSING FINISHED")
    print(f"{'='*60}")
    print(f"Sim1 files (NC data): {len(sim1_files)}")
    print(f"Sim2 files (first optical simulation): {total_sim2_files}")
    print(f"Sim3 files (further optical simulations, summed): {total_sim3_files} in {total_sim3_sims} sim_*** dirs")
    print(f"Optical files in total (Sim2 + Sim3): {total_combined}")
    print(f"NC events total: {total_entries}")
    print(f"NC with photons: {output_stats['nc_with_photons']}")
    print(f"NC without photons: {output_stats['nc_without_photons']}")
    print(f"Unassigned photons: {total_unassigned}")
    print(f"Orphaned photons: {total_orphaned}")
    print(f"Total runtime: {total_time / 60:.1f} min")
    print(f"Output file: {output_file}")

    progress_tracker.cleanup()
    print_memory_usage("program end")


if __name__ == "__main__":
    main()
