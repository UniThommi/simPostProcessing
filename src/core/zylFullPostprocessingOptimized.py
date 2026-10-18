"""
Post-processing of homogeneously distributed neutron captures into ML training data.

This script creates the training and validation data for the diffusion model in
diffusionSSD (https://github.com/UniThommi/diffusionSSD). Its input is a
Geant4/remage simulation in which only neutron captures (NCs) are simulated:
every event starts with the capture gamma cascade at t = 0, so the NC truth
(/hit/MyNeutronCaptureOutput) and the resulting optical photons (/hit/optical)
are stored in the same file.

For every input file the script
  1. maps the file-local material and volume IDs to the global IDs in
     src/mappings/ so that all files share one ID scheme,
  2. keeps only optical photons that hit the sensitive surface detector (SSD)
     within 0-200 ns after the capture and that arrive from inside the tank
     (photons hitting the surface from outside could not be detected in reality),
  3. assigns each photon to the nearest voxel of the SSD (KD-tree on the voxel
     centers) and counts the hits per NC, and
  4. writes one row per NC: NC features (phi_matrix), photon hits per voxel
     (target_matrix) and per detector region (region_matrix).

NCs are split into a training and a validation file at event level (all NCs of
one event end up in the same file). The split is stored in train_val_split.json
so that it is reproducible.

The run can be resumed: files are processed in batches, and after each batch the
completed files are recorded in processing_progress.json. Restarting the same
command skips these files.

Outputs (in --output):
  ncscore_output_0_train.hdf5        training data
  ncscore_output_0_validation.hdf5   validation data
  train_val_split.json               the event-level split
  processing_progress.json           progress for resuming (removed after success)

Example:
  python zylFullPostprocessingOptimized.py -i /path/to/simulation -o /path/to/output --nested \\
      -m src/mappings/globalMaterialMappings.json \\
      -v src/mappings/globalPhysVolMappings.json \\
      -x src/voxels/currentDistZylVoxelsPMTSize.json

Author: Thomas Bürger, Eberhard Karls Universität Tübingen
"""

import os
import glob
import json
import h5py
import sys
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.spatial import KDTree
from multiprocessing import Pool, Manager, Lock
import multiprocessing as mp
from functools import partial
import time
import psutil
import gc
import argparse
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).resolve().parents[1]))  # src/, for local_paths
from local_paths import local_path


def get_memory_usage():
    """Return the memory usage of the current process (RSS/VMS in MB, percent of total)."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent()
    }

def print_memory_usage(label=""):
    mem = get_memory_usage()
    print(f"Memory {label}: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, {mem['percent']:.1f}%")

class ProgressTracker:
    """Persists the processing state in the output directory so that a run can be resumed.

    processing_progress.json  completed and failed input files, entry counts
    train_val_split.json      the train/validation split (kept after success)
    """

    def __init__(self, output_dir, output_file_train, output_file_val):
        self.output_dir = Path(output_dir)
        self.output_file_train = Path(output_file_train)
        self.output_file_val = Path(output_file_val)
        self.progress_file = self.output_dir / "processing_progress.json"
        self.lock_file = self.output_dir / "processing.lock"
        self.split_file = self.output_dir / "train_val_split.json"
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
            'total_entries_written_train': 0,
            'total_entries_written_val': 0,
            'total_unassigned': 0,
            'output_file_train': str(self.output_file_train),
            'output_file_val': str(self.output_file_val)
        }

    def save_train_val_split(self, train_triplets, val_triplets, random_seed, val_fraction):
        """
        Save the train/validation split for reproducibility.

        Args:
            train_triplets: set of (file_idx, evtid, nC_id) tuples for training
            val_triplets: set of (file_idx, evtid, nC_id) tuples for validation
            random_seed: seed used to create the split
            val_fraction: requested validation fraction
        """
        split_data = {
            'random_seed': random_seed,
            'val_fraction': val_fraction,
            'train_triplets': [[int(t[0]), int(t[1]), int(t[2])] for t in train_triplets],
            'val_triplets': [[int(t[0]), int(t[1]), int(t[2])] for t in val_triplets],
            'created_at': time.time()
        }

        try:
            with open(self.split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"Train/val split saved to {self.split_file}")
        except IOError as e:
            print(f"Warning: could not save split: {e}")

    def load_train_val_split(self):
        """
        Load a saved train/validation split.

        Returns:
            tuple: (train_triplets, val_triplets, random_seed, val_fraction), or None
        """
        if not self.split_file.exists():
            return None

        try:
            with open(self.split_file, 'r') as f:
                split_data = json.load(f)

            train_triplets = set(tuple(t) for t in split_data['train_triplets'])
            val_triplets = set(tuple(t) for t in split_data['val_triplets'])

            print(f"Train/val split loaded:")
            print(f"  Training: {len(train_triplets)} NC events")
            print(f"  Validation: {len(val_triplets)} NC events")
            print(f"  Random seed: {split_data['random_seed']}")
            print(f"  Val fraction: {split_data['val_fraction']}")

            return (train_triplets, val_triplets,
                split_data['random_seed'],
                split_data['val_fraction'])

        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Warning: could not load split ({e}), creating a new one")
            return None

    def cleanup(self):
        """Remove the progress and lock files after a successful run."""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.lock_file.exists():
                self.lock_file.unlink()
            # The split file is kept on purpose: it is needed to reproduce the split.
        except OSError:
            pass

    def save_progress(self):
        self.progress_data['last_update'] = time.time()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except IOError as e:
            print(f"Warning: could not save progress: {e}")

    def mark_file_completed(self, file_path, entries_count_train, entries_count_val, unassigned_count):
        """Record a file as successfully processed, together with its entry counts."""
        file_str = str(file_path)
        if file_str not in self.progress_data['completed_files']:
            self.progress_data['completed_files'].append(file_str)
            self.progress_data['total_entries_written_train'] += entries_count_train
            self.progress_data['total_entries_written_val'] += entries_count_val
            self.progress_data['total_unassigned'] += unassigned_count
            self.save_progress()

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

    def mark_file_failed(self, file_path, error_msg):
        self.progress_data['failed_files'].append({
            'file': str(file_path),
            'error': str(error_msg),
            'timestamp': time.time()
        })
        self.save_progress()

    def get_remaining_files(self, all_files):
        """Return the input files that have not been completed yet (failed files are retried)."""
        completed = set(self.progress_data['completed_files'])
        return [f for f in all_files if str(f) not in completed]

    def get_statistics(self):
        return {
            'completed': len(self.progress_data['completed_files']),
            'failed': len(self.progress_data['failed_files']),
            'total_entries_train': self.progress_data['total_entries_written_train'],
            'total_entries_val': self.progress_data['total_entries_written_val'],
            'total_unassigned': self.progress_data['total_unassigned'],
            'elapsed_time': time.time() - self.progress_data['start_time']
        }

# ============================================================================
# Mapping Functions
# ============================================================================
# Each simulation file numbers its materials and physical volumes in its own
# way. The functions below translate these local IDs into the global IDs stored
# in src/mappings/*.json, so that matID/volID mean the same in every row.

def loadMapping(filePath):
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

    # The global mapping must already be complete at this point.
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
                f"This should not happen after collect_all_materials_first().\n"
                f"Available materials: {list(globMaterialMapping.keys())}"
            )

        mapping_dict[local_id] = globMaterialMapping[local_name]

    # Vectorize the mapping process
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
                f"This should not happen after collect_all_volumes_first().\n"
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

    # New entries get IDs after the highest existing one
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

    # New entries get IDs after the highest existing one
    max_id = max(global_mapping.values()) if global_mapping else -1

    for volume in all_volumes:
        if volume not in global_mapping:
            max_id += 1
            global_mapping[volume] = max_id

    with open(existing_mapping_path, 'w') as f:
        json.dump(global_mapping, f, indent=2)

    return global_mapping

def collect_all_materials_first(files):
    """Collect the material names of all files before processing, so the global mapping is complete."""
    print("Collecting materials from all files...")
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
    """
    Collect the physical volume names of all files before processing (empty name -> "noVolume").
    """
    print("Collecting physical volumes from all files...")
    all_volumes = set()

    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                volume_names = [x.decode() for x in f["hit/physVolumes/physVolumeNames"]["pages"][:]]

                # Sanitize empty strings immediately
                volume_names_clean = [
                    "noVolume" if name == "" else name
                    for name in volume_names
                ]

                all_volumes.update(volume_names_clean)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"Found: {len(all_volumes)} unique physical volumes")

    # Verify no empty strings remain
    if "" in all_volumes:
        print(f"  ⚠ WARNING: Empty string still present after sanitization!")
        all_volumes.remove("")
        all_volumes.add("noVolume")

    return all_volumes

# ============================================================================
# Run Directory Validation
# ============================================================================

_RUN_DIR_PATTERN = re.compile(r'^run_(\d+)$')


def validate_run_structure(path: str, nested: bool, label: str):
    """
    Validates the input directory structure.

    Nested mode: every subdirectory that contains output_*.hdf5 files must
    match the 'run_NNN' naming pattern. Raises RuntimeError for violations.

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

        for subdir in dirs_with_files:
            if not _RUN_DIR_PATTERN.match(subdir.name):
                raise RuntimeError(
                    f"[{label}] Directory '{subdir.name}' in '{path}' contains "
                    f"HDF5 files but does not match the expected 'run_NNN' pattern. "
                    f"All run directories must follow this naming convention."
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


def build_file_run_id_map(files: list, nested: bool) -> dict:
    """
    Returns a mapping {file_path: run_id} for every file in 'files'.

    Non-nested: all files get run_id = -1.
    Nested: run_id is parsed from the parent directory name (run_NNN → N).
    """
    if not nested:
        return {f: -1 for f in files}

    result = {}
    for file_path in files:
        run_id = -1
        for part in Path(file_path).parts:
            m = _RUN_DIR_PATTERN.match(part)
            if m:
                run_id = int(m.group(1))
                break
        result[file_path] = run_id
    return result


# ============================================================================
# File Discovery
# ============================================================================

def find_all_hdf5_files(input_path, nested):
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


def check_hdf5_integrity(files_by_run: dict, output_dir: str) -> bool:
    """
    Opens and immediately closes every HDF5 file to detect truncated or corrupt
    files before the expensive processing phase begins.

    Files are grouped by their parent directory — one directory = one simulation.
    If any file in a directory fails to open, that entire simulation is flagged.
    The check always runs to completion so the report is exhaustive.

    Returns True if all files are intact, False if any simulation is faulty.
    Writes a text file listing faulty simulations to output_dir.
    """
    print(f"\n=== HDF5 integrity check ===")

    files_by_sim: dict = defaultdict(list)
    for file_list in files_by_run.values():
        for fpath in file_list:
            sim_dir = str(Path(fpath).parent)
            files_by_sim[sim_dir].append(fpath)

    total_sims = len(files_by_sim)
    total_files = sum(len(v) for v in files_by_sim.values())
    print(f"  Checking {total_files} files in {total_sims} simulation directories...")

    faulty: list = []  # list of (sim_dir, first_faulty_file, error_msg)

    for sim_idx, sim_dir in enumerate(sorted(files_by_sim.keys()), 1):
        dir_files = files_by_sim[sim_dir]

        first_bad_file = None
        first_error = None
        for fpath in dir_files:
            try:
                with h5py.File(fpath, 'r'):
                    pass
            except Exception as e:
                first_bad_file = fpath
                first_error = str(e)
                break

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

def checkRadialMomentumVectorized(x, y, z, px, py, pz):
    """Return True for photons whose momentum points radially outwards (towards the wall)."""
    r = np.sqrt(x**2 + y**2)
    valid_r = r > 0
    radial_momentum = np.zeros_like(r)

    # Avoid division by zero for radial components
    radial_momentum[valid_r] = (x[valid_r]*px[valid_r] + y[valid_r]*py[valid_r]) / r[valid_r]
    return radial_momentum >= 0


def build_voxel_tree(voxels):
    """Build a KD-tree on the voxel centers; nearest-center lookup assigns photons to voxels."""
    if not voxels:
        return None
    centers = [voxel['center'] for voxel in voxels]
    return KDTree(centers)

def defineZylinder(geometry_name, valid_detectors=None):
    """
    Return the geometry constants of the cylindrical SSD (all lengths in mm).

    Only r_zylinder, h_zylinder and valid_detectors are used by the
    post-processing; the other values are returned for completeness.

    Args:
        geometry_name: geometry name; only 'currentDist' is supported
        valid_detectors: detector UIDs of the SSD surfaces (default: [1965, 1966, 1967, 1968])

    Returns:
        tuple: geometry parameters including valid_detectors
    """

    valid_geometry_names = ["currentDist"]
    if not geometry_name in valid_geometry_names:
        print("Invalid geometry name. It must be one of the following:")
        for name in valid_geometry_names:
            print(name)
        sys.exit()

    # Default detector UIDs of the SSD surfaces (from the remage detector registration)
    if valid_detectors is None:
        valid_detectors = [1965, 1966, 1967, 1968]

    t_zylinder = 1         # mm
    l_voxel = 195      # mm, square with the area of a circle of radius 110 mm: sqrt((110 mm)² * pi) = 194.97 mm
    t_voxel = 1         # mm
    r_pit = 3800        # mm
    dz_pit = 1        # mm
    r_zyl_bot = 3950    # mm, inner radius
    r_zyl_top = 1200    # mm
    z_offset = -5000     # mm
    h = 8900

    # if geometry_name == "maxDist":
    #     r_zylinder = 5858  # inner radius, 5858 in the simulation
    #     z_origin = 820
    #     h_zylinder = h + 20 - z_origin
    #     print("Set maxDist as geometry.")
    if geometry_name == "currentDist":
        r_zylinder = 4300 # inner radius of the Tyvek wall
        z_origin = 20
        h_zylinder = h + 20 - z_origin
        print("Set currentDist as geometry.")

    # Reference cylinder dimensions (not used by the post-processing)
    r_ref = 6000  # apothem (inner radius); 5858 in the simulation
    h_ref = 8080 # should actually be 20 mm higher
    z_ref = 20

    return (t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top, z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref, valid_detectors)

def process_single_file(args):
    """Turn one simulation file into one output row per neutron capture.

    Args:
        args: tuple (file_path, file_idx, voxel_tree, voxel_data, voxel_indices,
              materialMappingPath, volumeMappingPath, geometry_params,
              val_triplets, run_id). file_idx is the index of the file in the
              list of all input files; (file_idx, evtid, nC_id) in val_triplets
              sends an NC to the validation set.

    Returns:
        dict with phi/target/region/event_id arrays for the training and the
        validation set, plus counts of unassigned and orphaned photons.
    """
    (file_path, file_idx, voxel_tree, voxel_data, voxel_indices,
     materialMappingPath, volumeMappingPath, geometry_params, val_triplets, run_id) = args

    h_zylinder = geometry_params['h_zylinder']
    r_zylinder = geometry_params['r_zylinder']
    valid_detectors = geometry_params.get('valid_detectors', [1965, 1966, 1967, 1968])

    print(f"Worker {mp.current_process().pid}: processing {os.path.basename(file_path)}")
    file_start_time = time.time()
    # print_memory_usage("worker start")

    phi_data_train = []
    target_data_train = []
    target_regions_train = []
    event_ids_train = []
    phi_data_val = []
    target_data_val = []
    target_regions_val = []
    event_ids_val = []
    unassigned_count = 0

    # Photons are read in chunks whose size depends on the available memory
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    if available_mem_gb > 400:
        CHUNK_SIZE = 50000
    elif available_mem_gb > 200:
        CHUNK_SIZE = 30000
    elif available_mem_gb > 50:
        CHUNK_SIZE = 20000
    elif available_mem_gb > 30:
        CHUNK_SIZE = 15000
    elif available_mem_gb > 20:
        CHUNK_SIZE = 10000
    else:
        CHUNK_SIZE = 5000

    with h5py.File(file_path, 'r') as f:
        # Read the neutron capture truth
        ##############################################
        nc_evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
        nc_nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
        nC_x = f['hit']['MyNeutronCaptureOutput']['nC_x_position_in_m']['pages'][:] * 1000  # mm
        nC_y = f['hit']['MyNeutronCaptureOutput']['nC_y_position_in_m']['pages'][:] * 1000
        nC_z = f['hit']['MyNeutronCaptureOutput']['nC_z_position_in_m']['pages'][:] * 1000
        gamma_amount = f['hit']['MyNeutronCaptureOutput']['nC_gamma_amount']['pages'][:]
        gamma_tot_energy = f['hit']['MyNeutronCaptureOutput']['nC_gamma_total_energy_in_keV']['pages'][:]
        nc_material_ids = f['hit']['MyNeutronCaptureOutput']['nC_material_id']['pages'][:]
        nc_volume_ids = f['hit']['MyNeutronCaptureOutput']['nC_phys_vol_id']['pages'][:]
        nc_time = f['hit']['MyNeutronCaptureOutput']['nC_time_in_ns']['pages'][:]
        nc_flag_ge77 = f['hit']['MyNeutronCaptureOutput']['nC_flag_Ge77']['pages'][:]

        # Capture gammas 1-4 (energy and momentum direction)
        gamma_data_nc = {}
        for i in range(1, 5):
            gamma_data_nc[f'gamma{i}_px'] = f['hit']['MyNeutronCaptureOutput'][f'gamma{i}_px']['pages'][:]
            gamma_data_nc[f'gamma{i}_py'] = f['hit']['MyNeutronCaptureOutput'][f'gamma{i}_py']['pages'][:]
            gamma_data_nc[f'gamma{i}_pz'] = f['hit']['MyNeutronCaptureOutput'][f'gamma{i}_pz']['pages'][:]
            gamma_data_nc[f'gamma{i}_E_in_keV'] = f['hit']['MyNeutronCaptureOutput'][f'gamma{i}_E_in_keV']['pages'][:]

        # Translate local material IDs into global IDs
        ##############################################
        mapping_names = [x.decode() for x in f["hit/materials/materialNames"]["pages"][:]]
        mapping_ids = f["hit/materials/materialsID"]["pages"][:]
        local_mat_map = dict(zip(mapping_ids, mapping_names))
        globalMaterialIDs, _ = remapMaterialIDsToGlobal(materialMappingPath, local_mat_map, nc_material_ids)

        # Translate local volume IDs into global IDs
        ##############################################
        volume_names = [x.decode() for x in f["hit/physVolumes/physVolumeNames"]["pages"][:]]
        volume_ids = f["hit/physVolumes/physVolumesID"]["pages"][:]
        local_vol_map = dict(zip(volume_ids, volume_names))
        globalVolumeIDs, _ = remapVolumeIDsToGlobal(volumeMappingPath, local_vol_map, nc_volume_ids)

        # NC lookup table keyed by (evtid, nC_id)
        nc_data_dict = {}
        for idx in range(len(nc_evtid)):
            key = (nc_evtid[idx], nc_nC_id[idx])
            nc_data_dict[key] = {
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
                'gamma_pz': [gamma_data_nc[f'gamma{i}_pz'][idx] for i in range(1, 5)]
            }

        # Free the raw NC arrays (only nc_data_dict is needed from here on)
        del nc_evtid, nc_nC_id, nC_x, nC_y, nC_z, gamma_amount, gamma_tot_energy, nc_material_ids, nc_volume_ids
        del gamma_data_nc, globalMaterialIDs, globalVolumeIDs
        gc.collect()

        # print_memory_usage("after building the NC dict")

        total_optical_events = len(f['hit']['optical']['x_position_in_m']['pages'])
        # print(f"Processing {total_optical_events} optical events in chunks of {CHUNK_SIZE}")

        # z-coordinates (mm) that separate bottom, barrel and top of the SSD
        z_cut_pit = -4979
        z_cut_bot = -4979  # pit/bottom surface (voxel centers at z = -4979.5 mm)
        z_cut_top = z_cut_bot + h_zylinder - 2 # the top surface is 1 mm thick and starts at h_zyl - 1

        num_chunks = (total_optical_events - 1) // CHUNK_SIZE + 1

        # One hit counter {voxel index: photons} per NC
        nc_voxel_counters = {key: Counter() for key in nc_data_dict.keys()}
        orphaned_photons = 0

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, total_optical_events)

            # print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            # print_memory_usage(f"Chunk {chunk_idx + 1} Start")

            # Load only the current chunk (positions converted from m to mm)
            x_chunk = np.array(f['hit']['optical']['x_position_in_m']['pages'][chunk_start:chunk_end], dtype=np.float32) * 1000
            y_chunk = np.array(f['hit']['optical']['y_position_in_m']['pages'][chunk_start:chunk_end], dtype=np.float32) * 1000
            z_chunk = np.array(f['hit']['optical']['z_position_in_m']['pages'][chunk_start:chunk_end], dtype=np.float32) * 1000
            px_chunk = np.array(f['hit']['optical']['x_momentum_direction']['pages'][chunk_start:chunk_end], dtype=np.float32)
            py_chunk = np.array(f['hit']['optical']['y_momentum_direction']['pages'][chunk_start:chunk_end], dtype=np.float32)
            pz_chunk = np.array(f['hit']['optical']['z_momentum_direction']['pages'][chunk_start:chunk_end], dtype=np.float32)

            photon_evtid_chunk = f['hit']['optical']['evtid']['pages'][chunk_start:chunk_end]
            photon_nC_id_chunk = f['hit']['optical']['nC_track_id']['pages'][chunk_start:chunk_end]
            photon_det_uid_chunk = f['hit']['optical']['det_uid']['pages'][chunk_start:chunk_end]
            photon_time_chunk = f['hit']['optical']['time_in_ns']['pages'][chunk_start:chunk_end]

            # print_memory_usage(f"chunk {chunk_idx + 1} after loading")

            # Filter 1 (detector): keep only photons on the SSD surfaces
            valid_detectors_array = np.array(valid_detectors, dtype=np.int32)
            detector_mask = np.isin(photon_det_uid_chunk, valid_detectors_array)

            x_chunk = x_chunk[detector_mask]
            y_chunk = y_chunk[detector_mask]
            z_chunk = z_chunk[detector_mask]
            px_chunk = px_chunk[detector_mask]
            py_chunk = py_chunk[detector_mask]
            pz_chunk = pz_chunk[detector_mask]
            photon_evtid_chunk = photon_evtid_chunk[detector_mask]
            photon_nC_id_chunk = photon_nC_id_chunk[detector_mask]
            photon_time_chunk = photon_time_chunk[detector_mask]

            # Capture time of the NC that produced each photon (inf = no matching NC).
            # float32 is sufficient because the capture happens at t = 0 in this workflow.
            nc_times = np.full(len(photon_time_chunk), np.inf, dtype=np.float32)

            for idx in range(len(photon_evtid_chunk)):
                key = (photon_evtid_chunk[idx], photon_nC_id_chunk[idx])
                if key in nc_data_dict:
                    nc_times[idx] = nc_data_dict[key]['nC_time']

            # Filter 2 (time window): 0 <= t_photon - t_NC <= 200 ns.
            # Unlike the MUSUN scripts, no tolerance for slightly negative times is
            # needed: photon and NC times come from the same simulation, which
            # starts with the capture at t = 0.
            time_mask = (
                (nc_times != np.inf) &  # a matching NC exists
                (photon_time_chunk >= nc_times) &  # photon after the capture
                (photon_time_chunk <= nc_times + 200.0)  # within 200 ns
            )

            # Photons without a matching NC ("orphaned") are only counted
            orphaned_in_chunk = np.sum(nc_times == np.inf)
            if orphaned_in_chunk > 0:
                orphaned_photons += orphaned_in_chunk

            x_chunk = x_chunk[time_mask]
            y_chunk = y_chunk[time_mask]
            z_chunk = z_chunk[time_mask]
            px_chunk = px_chunk[time_mask]
            py_chunk = py_chunk[time_mask]
            pz_chunk = pz_chunk[time_mask]
            photon_evtid_chunk = photon_evtid_chunk[time_mask]
            photon_nC_id_chunk = photon_nC_id_chunk[time_mask]
            photon_time_chunk = photon_time_chunk[time_mask]

            # Filter 3 (direction): keep photons that hit the surface from inside
            # the tank: moving down at the bottom, up at the top, outwards at the wall
            mask_bot = (z_chunk <= z_cut_bot)
            mask_top = (z_chunk >= z_cut_top)
            mask_barrel = ~mask_bot & ~mask_top

            mask_bot_valid = (pz_chunk[mask_bot] <= 0)
            mask_top_valid = (pz_chunk[mask_top] >= 0)

            # Radial momentum is only needed for the barrel (wall) region
            mask_barrel_valid = np.zeros(np.sum(mask_barrel), dtype=bool)
            if np.any(mask_barrel):
                mask_barrel_valid = checkRadialMomentumVectorized(
                    x_chunk[mask_barrel], y_chunk[mask_barrel], z_chunk[mask_barrel],
                    px_chunk[mask_barrel], py_chunk[mask_barrel], pz_chunk[mask_barrel]
                )

            # Combine masks
            final_mask = np.zeros_like(z_chunk, dtype=bool)
            final_mask[mask_bot] = mask_bot_valid
            final_mask[mask_top] = mask_top_valid
            final_mask[mask_barrel] = mask_barrel_valid

            x_filtered = x_chunk[final_mask]
            y_filtered = y_chunk[final_mask]
            z_filtered = z_chunk[final_mask]
            photon_evtid_filtered = photon_evtid_chunk[final_mask]
            photon_nC_id_filtered = photon_nC_id_chunk[final_mask]

            # Free arrays that are no longer needed
            del x_chunk, y_chunk, z_chunk, px_chunk, py_chunk, pz_chunk
            del photon_evtid_chunk, photon_nC_id_chunk
            del detector_mask, time_mask, photon_time_chunk
            del mask_bot, mask_top, mask_barrel, mask_bot_valid, mask_top_valid, mask_barrel_valid, final_mask
            gc.collect()

            # print_memory_usage(f"chunk {chunk_idx + 1} after filtering")

            # Voxel assignment: one batched KD-tree query for the whole chunk
            if len(x_filtered) > 0:
                points = np.column_stack((x_filtered, y_filtered, z_filtered))
                # Index of the nearest voxel center for every photon
                _, voxel_idx_array = voxel_tree.query(points)
                voxel_index_strings = np.array(
                    [str(voxel_data[vi]['index']) for vi in voxel_idx_array]
                )

                # Group the photons by NC and add them to the NC's hit counter
                evtid_int = photon_evtid_filtered.astype(np.int64)
                nC_id_int = photon_nC_id_filtered.astype(np.int64)
                paired = np.stack([evtid_int, nC_id_int], axis=1)
                unique_pairs, inverse = np.unique(paired, axis=0, return_inverse=True)

                for pair_idx, (e_id, nc_id) in enumerate(unique_pairs):
                    key = (int(e_id), int(nc_id))
                    if key not in nc_data_dict:
                        continue

                    mask = inverse == pair_idx
                    voxel_strs = voxel_index_strings[mask]
                    for vs in voxel_strs:
                        nc_voxel_counters[key][vs] += 1

                del points, voxel_idx_array, voxel_index_strings, paired

            del x_filtered, y_filtered, z_filtered
            del photon_evtid_filtered, photon_nC_id_filtered

            # print_memory_usage(f"chunk {chunk_idx + 1} end")

    nc_without_photons = 0
    nc_with_photons = 0

    # Build one output row per NC (NCs without detected photons are kept)
    for (e_id, nc_id), nc_info in nc_data_dict.items():
        voxel_counter = nc_voxel_counters[(e_id, nc_id)]

        if not voxel_counter:  # no photon detected
            nc_without_photons += 1
        else:
            nc_with_photons += 1

        # NC features (phi)
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

        # Cylindrical coordinates of the NC position
        r_NC = np.sqrt(x**2 + y**2)
        phi_NC = np.arctan2(y, x)

        # Signed distances in mm (positive inside the SSD cylinder, negative outside)
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

        # Column order must match phi_columns in create_or_open_output_file()
        phi_row = [x, y, z, mat_id, vol_id, n_gamma, e_tot,
                   r_NC, phi_NC, dist_to_wall, dist_to_bot, dist_to_top,
                   p_mean_r, p_mean_z,
                   nc_info['nC_time'], nc_info['nC_flag_Ge77']] + gamma_row

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

        event_id_row = [run_id, int(nc_id)]
        if (file_idx, e_id, nc_id) in val_triplets:
            phi_data_val.append(phi_row)
            target_data_val.append(target_row)
            target_regions_val.append(target_regions_row)
            event_ids_val.append(event_id_row)
        else:
            phi_data_train.append(phi_row)
            target_data_train.append(target_row)
            target_regions_train.append(target_regions_row)
            event_ids_train.append(event_id_row)

    # print_memory_usage("worker end")

    file_elapsed = time.time() - file_start_time
    total_nc_events = len(nc_data_dict)
    processed_events = len(phi_data_train) + len(phi_data_val)
    print(f"  NC events: {total_nc_events} total, {processed_events} processed in {file_elapsed:.1f}s")
    if orphaned_photons > 0:
        print(f"  ℹ INFO: {orphaned_photons} photons without a matching NC event discarded. Expected for photons created by processes before an NC event.")

    assert total_nc_events == processed_events, (
        f"Not all NC events were processed! "
        f"Expected: {total_nc_events}, processed: {processed_events}"
    )

    print(f"  NC events without photons: {nc_without_photons}")
    print(f"  NC events with photons: {nc_with_photons}")

    del nc_voxel_counters, nc_data_dict
    gc.collect()

    _empty2 = np.empty((0, 2), dtype=np.int64)
    return {
        'phi_data_train': np.array(phi_data_train, dtype=np.float32) if phi_data_train else np.array([]),
        'target_data_train': np.array(target_data_train, dtype=np.int32) if target_data_train else np.array([]),
        'target_regions_train': np.array(target_regions_train, dtype=np.int32) if target_regions_train else np.array([]),
        'event_ids_train': np.array(event_ids_train, dtype=np.int64) if event_ids_train else _empty2,
        'phi_data_val': np.array(phi_data_val, dtype=np.float32) if phi_data_val else np.array([]),
        'target_data_val': np.array(target_data_val, dtype=np.int32) if target_data_val else np.array([]),
        'target_regions_val': np.array(target_regions_val, dtype=np.int32) if target_regions_val else np.array([]),
        'event_ids_val': np.array(event_ids_val, dtype=np.int64) if event_ids_val else _empty2,
        'unassigned_count': unassigned_count,
        'orphaned_photons': orphaned_photons,
        'file_processed': os.path.basename(file_path)
    }


def append_results_to_hdf5(output_file, result_data, voxel_indices, weight, dataset_type='train'):
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

def collect_all_nc_triplets(files):
    """
    Collect the (file_idx, evtid, nC_id) triplets of all NCs in all files.

    file_idx makes the triplets unique across files, because evtid and nC_id
    are only unique within one simulation file.
    """
    print("Collecting all neutron capture events for the train/val split...")
    all_nc_triplets = set()

    for file_idx, file_path in enumerate(files):
        try:
            with h5py.File(file_path, 'r') as f:
                evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
                nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
                triplets = [(file_idx, int(evt), int(nc)) for evt, nc in zip(evtid, nC_id)]
                all_nc_triplets.update(triplets)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"Found: {len(all_nc_triplets)} unique NC events in {len(files)} files")
    return all_nc_triplets

def calculate_weight_from_files(files, sample_size=None):
    """
    Compute the per-event weight 1 / (number of NCs) stored in the 'weights' dataset.

    If sample_size is given, the NC count is extrapolated from the first
    sample_size files. The weight is constant and not used by the downstream tools.
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

    # Extrapolate if only a sample was used
    if sample_size and sample_size < len(files):
        scale_factor = len(files) / sample_size
        estimated_unique_triplets = len(unique_triplets) * scale_factor
        weight = 1 / estimated_unique_triplets if estimated_unique_triplets > 0 else 1.0
        print(f"Estimated weight based on {sample_size} files: {weight:.2e}")
    else:
        weight = 1 / len(unique_triplets) if unique_triplets else 1.0
        print(f"Weight based on all {len(files)} files: {weight:.2e}")

    return weight

def create_or_load_train_val_split(progress_tracker, all_nc_triplets,
                                   val_fraction, random_seed):
    """
    Create the train/validation split, or reuse the one saved by an earlier run.

    The split is done at event level: all NCs of one (file_idx, evtid) event go
    to the same set, so that correlated NCs of one event cannot leak between
    training and validation. A saved split is reused even if seed or fraction
    differ (with a warning), and recreated if it references NCs that no longer
    exist.

    Returns:
        set of (file_idx, evtid, nC_id) triplets that belong to the validation set
    """
    loaded_split = progress_tracker.load_train_val_split()

    if loaded_split is not None:
        train_triplets, val_triplets, saved_seed, saved_fraction = loaded_split

        if saved_seed != random_seed:
            print(f"⚠ WARNING: saved random seed ({saved_seed}) != current seed ({random_seed})")
            print(f"  Using the saved split for consistency!")

        if abs(saved_fraction - val_fraction) > 0.001:
            print(f"⚠ WARNING: saved val fraction ({saved_fraction}) != current ({val_fraction})")
            print(f"  Using the saved split for consistency!")

        # The saved split must only reference NCs that still exist
        all_triplets_set = set(all_nc_triplets)
        missing_train = train_triplets - all_triplets_set
        missing_val = val_triplets - all_triplets_set

        if missing_train or missing_val:
            print(f"⚠ WARNING: {len(missing_train) + len(missing_val)} NC events of the split are missing in the data!")
            print(f"  Creating a new split...")
            loaded_split = None
        else:
            print(f"✓ Using saved split: {len(val_triplets)} validation events")
            return val_triplets  # only the validation set is needed by the caller

    if loaded_split is None:
        print(f"Creating new train/val split (event level) with seed {random_seed}...")

        # Group NCs by (file_idx, evtid) to avoid data leakage
        evtid_groups = defaultdict(list)
        for file_idx, evt, nc in all_nc_triplets:
            evtid_groups[(file_idx, evt)].append((file_idx, evt, nc))

        # Shuffle the events (not the NCs) and split them
        event_keys = sorted(evtid_groups.keys())
        np.random.seed(random_seed)
        np.random.shuffle(event_keys)

        split_idx = int(len(event_keys) * (1 - val_fraction))
        train_event_keys = set(event_keys[:split_idx])

        # All NCs of one (file_idx, evtid) event go to the same set
        train_triplets = set()
        val_triplets = set()
        for evt_key, triplets in evtid_groups.items():
            if evt_key in train_event_keys:
                train_triplets.update(triplets)
            else:
                val_triplets.update(triplets)

        print(f"Train/val split created (event level):")
        print(f"  Events total: {len(event_keys)}")
        print(f"  Training: {len(train_event_keys)} events → {len(train_triplets)} NC events")
        print(f"  Validation: {len(event_keys) - len(train_event_keys)} events → {len(val_triplets)} NC events")
        print(f"  Actual val fraction: {len(val_triplets)/(len(train_triplets)+len(val_triplets))*100:.1f}%")

        progress_tracker.save_train_val_split(train_triplets, val_triplets,
                                            random_seed, val_fraction)

        return val_triplets

def process_files_in_batches(files, voxel_tree, voxel_data, voxel_indices,
                             material_mapping_path, volume_mapping_path,
                             geometry_params, output_file_train, output_file_val,
                             weight, progress_tracker, val_triplets, file_run_id_map,
                             batch_size=10):
    """Process all remaining files in batches and append each batch to the output files.

    Writing once per batch instead of once per file reduces the number of HDF5
    resize operations. After a batch is written, its files are marked as
    completed in the progress file, so a restart continues with the next batch.
    Files that raise an error are recorded as failed and retried on the next run.
    """

    remaining_files = progress_tracker.get_remaining_files(files)

    if not remaining_files:
        print("All files already processed!")
        return

    print(f"Processing {len(remaining_files)} files in batches of {batch_size}")

    for batch_start in range(0, len(remaining_files), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_files))
        batch_files = remaining_files[batch_start:batch_end]

        print(f"\n=== Batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end} ===")
        batch_start_time = time.time()

        # Accumulators for the batch
        batch_phi_train = []
        batch_target_train = []
        batch_target_regions_train = []
        batch_event_ids_train = []
        batch_phi_val = []
        batch_target_val = []
        batch_target_regions_val = []
        batch_event_ids_val = []
        file_stats = {}

        for file_path in batch_files:
            try:
                # file_idx must refer to the full file list (it is part of the split keys)
                file_idx = files.index(file_path)
                run_id = file_run_id_map.get(file_path, -1)

                args = (file_path, file_idx, voxel_tree, voxel_data, voxel_indices,
                        material_mapping_path, volume_mapping_path,
                        geometry_params, val_triplets, run_id)

                result = process_single_file(args)

                train_count = len(result['phi_data_train'])
                val_count = len(result['phi_data_val'])

                file_stats[file_path] = {
                    'train': train_count,
                    'val': val_count,
                    'unassigned': result['unassigned_count']
                }

                if train_count > 0:
                    batch_phi_train.append(result['phi_data_train'])
                    batch_target_train.append(result['target_data_train'])
                    batch_target_regions_train.append(result['target_regions_train'])
                    batch_event_ids_train.append(result['event_ids_train'])

                if val_count > 0:
                    batch_phi_val.append(result['phi_data_val'])
                    batch_target_val.append(result['target_data_val'])
                    batch_target_regions_val.append(result['target_regions_val'])
                    batch_event_ids_val.append(result['event_ids_val'])

                print(f"  ✓ {os.path.basename(file_path)} (train: {train_count}, val: {val_count}, run_id: {run_id})")

            except Exception as e:
                print(f"  ✗ {os.path.basename(file_path)}: {e}")
                progress_tracker.mark_file_failed(file_path, str(e))
                continue

        # Write the whole batch at once
        if batch_phi_train:
            combined_phi_train = np.vstack(batch_phi_train)
            combined_target_train = np.vstack(batch_target_train)
            combined_target_regions_train = np.vstack(batch_target_regions_train)
            combined_event_ids_train = np.vstack(batch_event_ids_train)
            entries_train = append_results_to_hdf5(
                output_file_train,
                {'phi_data': combined_phi_train, 'target_data': combined_target_train,
                 'target_regions': combined_target_regions_train,
                 'event_ids': combined_event_ids_train},
                voxel_indices, weight, 'train'
            )
            print(f"  Batch train written: {entries_train} entries")

        if batch_phi_val:
            combined_phi_val = np.vstack(batch_phi_val)
            combined_target_val = np.vstack(batch_target_val)
            combined_target_regions_val = np.vstack(batch_target_regions_val)
            combined_event_ids_val = np.vstack(batch_event_ids_val)
            entries_val = append_results_to_hdf5(
                output_file_val,
                {'phi_data': combined_phi_val, 'target_data': combined_target_val,
                 'target_regions': combined_target_regions_val,
                 'event_ids': combined_event_ids_val},
                voxel_indices, weight, 'val'
            )
            print(f"  Batch val written: {entries_val} entries")

        # Mark all files of the batch that did not fail as completed
        for file_path in batch_files:
            if str(file_path) not in [f['file'] for f in progress_tracker.progress_data['failed_files']]:
                stats = file_stats.get(file_path, {'train': 0, 'val': 0, 'unassigned': 0})
                progress_tracker.mark_file_completed(
                    file_path,
                    stats['train'],
                    stats['val'],
                    stats['unassigned']
                )

        batch_elapsed = time.time() - batch_start_time
        print(f"  Batch finished in {batch_elapsed:.1f}s")
        progress_tracker.verify_hdf5_integrity(output_file_val)

        # Free memory
        del batch_phi_train, batch_target_train, batch_target_regions_train, batch_event_ids_train
        del batch_phi_val, batch_target_val, batch_target_regions_val, batch_event_ids_val
        del result
        gc.collect()


# ----------------------------------------------------------------------------
# Output file
# ----------------------------------------------------------------------------
def create_or_open_output_file(output_path, file_index, voxel_data, mat_map, vol_map, radius, suffix=""):
    """Create the output HDF5 file, or return the path of an existing complete one.

    Layout (one row per NC in all row-aligned datasets):
      phi_matrix      (n_events, 32)       float32  NC features, names in phi_columns
      target_matrix   (n_events, n_voxels) int32    photon hits per voxel, voxel indices in target_columns
      region_matrix   (n_events, 4)        int32    photon hits per region, names in region_columns
      event_ids       (n_events, 2)        int64    (run_id, nc_id), names in event_id_columns
      weights         (n_events,)          float32  constant 1/N_NC (not used downstream)
    Static metadata: voxel geometry (voxels/, voxel_metadata), the global
    material/volume mappings (mat_map/, vol_map/), the SSD radius
    (theta/inner_radius_in_mm) and the number of NCs (primaries, set at the end).
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

    phi_columns = [
        "xNC_mm", "yNC_mm", "zNC_mm", "matID", "volID", "#gamma", "E_gamma_tot_keV",
        "r_NC_mm", "phi_NC_rad", "dist_to_wall_mm", "dist_to_bot_mm", "dist_to_top_mm",
        "p_mean_r", "p_mean_z",
        "nC_time_in_ns", "nC_flag_Ge77",
        "gammaE1_keV", "gammapx1", "gammapy1", "gammapz1",
        "gammaE2_keV", "gammapx2", "gammapy2", "gammapz2",
        "gammaE3_keV", "gammapx3", "gammapy3", "gammapz3",
        "gammaE4_keV", "gammapx4", "gammapy4", "gammapz4"
    ]
    region_columns = ['pit', 'bot', 'wall', 'top']
    event_id_columns = ["run_id", "nc_id"]

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
        # lzf compression for fast reading during training
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

        # === Event IDs (run_id, nc_id) ===
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
        out.create_dataset("primaries", data=0)

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

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Post-processing of homogeneous neutron-capture simulations into '
                    'ML training/validation data (photon hits per SSD voxel).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The script creates two output files:
- ncscore_output_0_train.hdf5 (by default 80% of the data)
- ncscore_output_0_validation.hdf5 (by default 20% of the data)

Examples:
  # Data directly in the input directory
  python %(prog)s -i /path/to/simulation_data -o /path/to/output

  # Data in run_NNN subdirectories
  python %(prog)s -i /path/to/simulations -o /path/to/output --nested
        """
    )

    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory with the simulation HDF5 files (required)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory (required)')

    # Mapping and voxel files. Default: the author's cluster copy if local_paths.json
    # is configured (see src/local_paths.py), otherwise the files in this repository.
    src_dir = Path(__file__).resolve().parents[1]
    parser.add_argument('-m', '--material-mapping',
                        default=(local_path('cfs', 'postprocessing/src/mappings/globalMaterialMappings.json')
                                 or str(src_dir / 'mappings' / 'globalMaterialMappings.json')),
                        help='Global material mapping (JSON, extended in place with new materials)')
    parser.add_argument('-v', '--volume-mapping',
                        default=(local_path('cfs', 'postprocessing/src/mappings/globalPhysVolMappings.json')
                                 or str(src_dir / 'mappings' / 'globalPhysVolMappings.json')),
                        help='Global physical volume mapping (JSON, extended in place with new volumes)')
    parser.add_argument('-x', '--voxel-file',
                        default=(local_path('cfs', 'postprocessing/src/voxels/currentDistZylVoxelsPMTSize.json')
                                 or str(src_dir / 'voxels' / 'currentDistZylVoxelsPMTSize.json')),
                        help='Voxel definition of the SSD (JSON)')

    # Resume and processing options
    parser.add_argument('--resume', action='store_true',
                        help='No effect: resuming is always active (already processed files are skipped)')
    parser.add_argument('--reset', action='store_true',
                        help='Start from scratch (deletes existing output, progress and split)')
    parser.add_argument('--sample-weight', type=int, default=50,
                        help='Number of files used to estimate the event weight (0 = all files)')
    parser.add_argument('--nested', action='store_true',
                        help='Search for HDF5 files in run_NNN subdirectories (run_001, run_002, ...)')

    # Train/val split
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='Fraction of events used for validation (default: 0.2 = 20%%)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for a reproducible train/val split (default: 42)')
    parser.add_argument('--no-split', action='store_true',
                        help='No train/val split: all data is written to the training file')

    # Further options
    parser.add_argument('-g', '--geometry', default='currentDist',
                        choices=['currentDist'],
                        help='Geometry name (default: currentDist)')
    parser.add_argument('--valid-detectors', type=int, nargs='+',
                    default=[1965, 1966, 1967, 1968],
                    help='Detector UIDs of the SSD surfaces (default: 1965 1966 1967 1968)')
    parser.add_argument('--verbose', '-V', action='store_true',
                        help='No effect (kept for compatibility)')

    return parser.parse_args()

def main():
    args = parse_arguments()
    print_memory_usage("main start")

    if not os.path.exists(args.input):
        print(f"Error: input path does not exist: {args.input}")
        sys.exit(1)

    # === Run structure validation ===
    print("\n=== Validating input structure ===")
    validate_run_structure(args.input, nested=args.nested, label="Sim")

    if not os.path.exists(args.voxel_file):
        print(f"Error: voxel file does not exist: {args.voxel_file}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Configuration
    geometry_name = args.geometry
    geometry_result = defineZylinder(geometry_name, args.valid_detectors)
    (t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top,
     z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref, valid_detectors) = geometry_result

    print("Loading voxel data...")
    with open(args.voxel_file, 'r') as f:
        voxel_data = json.load(f)

    voxel_tree = build_voxel_tree(voxel_data)
    voxel_indices = [voxel['index'] for voxel in voxel_data]
    print(f"Voxel tree with {len(voxel_data)} voxels built")

    files = find_all_hdf5_files(args.input, args.nested)

    if not files:
        print(f"Error: no output_*.hdf5 files found!")
        sys.exit(1)
    print(f"Found: {len(files)} files in total")

    # === Integrity check of all input files ===
    if not check_hdf5_integrity({0: files}, args.output):
        print("\nAborting: faulty HDF5 files found. See faulty_simulations.txt for the list.")
        sys.exit(1)

    # file_path -> run_id (written to event_ids)
    file_run_id_map = build_file_run_id_map(files, args.nested)

    output_file_train = os.path.join(args.output, "ncscore_output_0_train.hdf5")
    output_file_val = os.path.join(args.output, "ncscore_output_0_validation.hdf5")

    progress_tracker = ProgressTracker(args.output, output_file_train, output_file_val)

    if args.reset:
        print("Reset requested - deleting existing results...")
        if os.path.exists(output_file_train):
            os.remove(output_file_train)
        if os.path.exists(output_file_val):
            os.remove(output_file_val)
        progress_tracker.cleanup()
        if progress_tracker.split_file.exists():
            progress_tracker.split_file.unlink()
        progress_tracker = ProgressTracker(args.output, output_file_train, output_file_val)

    # Train/val split (reused from train_val_split.json on resume)
    if args.no_split:
        print("No train/val split - all data is written to the training file.")
        val_triplets = set()
    else:
        all_nc_triplets = collect_all_nc_triplets(files)
        val_triplets = create_or_load_train_val_split(
            progress_tracker, all_nc_triplets, args.val_fraction, args.random_seed
        )

    # Collect all materials/volumes and complete the global mappings only for a new
    # run (after --reset or if no output file exists yet); on resume they are loaded.
    if not os.path.exists(output_file_train):
        all_materials = collect_all_materials_first(files)
        all_volumes = collect_all_volumes_first(files)

        global_material_mapping = create_global_material_mapping(all_materials, args.material_mapping)
        print(f"Global material mapping created with {len(global_material_mapping)} materials")

        global_volume_mapping = create_global_volume_mapping(all_volumes, args.volume_mapping)
        print(f"Global volume mapping created with {len(global_volume_mapping)} volumes")
    else:
        global_material_mapping = loadMapping(args.material_mapping)
        print(f"Existing material mapping loaded: {len(global_material_mapping)} materials")

        global_volume_mapping = loadMapping(args.volume_mapping)
        print(f"Existing volume mapping loaded: {len(global_volume_mapping)} volumes")

    sample_size = args.sample_weight if args.sample_weight > 0 else None
    weight = calculate_weight_from_files(files, sample_size)

    output_file_train = create_or_open_output_file(args.output, 0, voxel_data, global_material_mapping, global_volume_mapping, r_zylinder, "_train")
    output_file_val = create_or_open_output_file(args.output, 0, voxel_data, global_material_mapping, global_volume_mapping, r_zylinder, "_validation")

    geometry_params = {
        'h_zylinder': h_zylinder,
        'r_zylinder': r_zylinder,
        'valid_detectors': valid_detectors
    }

    stats = progress_tracker.get_statistics()
    if stats['completed'] > 0:
        print(f"\nExisting progress:")
        print(f"  Already processed: {stats['completed']}/{len(files)} files")
        print(f"  Failed: {stats['failed']} files")
        print(f"  Entries train: {stats['total_entries_train']}")
        print(f"  Entries val: {stats['total_entries_val']}")
        print(f"  Runtime so far: {stats['elapsed_time']/60:.1f} min")

    print(f"\nStarting sequential processing...")
    start_time = time.time()

    try:
        process_files_in_batches(
            files, voxel_tree, voxel_data, voxel_indices,
            args.material_mapping, args.volume_mapping,
            geometry_params, output_file_train, output_file_val,
            weight, progress_tracker, val_triplets, file_run_id_map, batch_size=20
        )
    except KeyboardInterrupt:
        print(f"\nProcessing interrupted by user.")
        print(f"Progress has been saved. Run the script again to continue.")
        return
    except Exception as e:
        print(f"Critical error: {e}")
        return

    final_stats = progress_tracker.get_statistics()
    total_time = time.time() - start_time

    print(f"\n" + "="*60)
    print(f"PROCESSING FINISHED AFTER: ", total_time)
    print(f"="*60)
    print(f"Successfully processed files: {final_stats['completed']}/{len(files)}")
    print(f"Failed files: {final_stats['failed']}")
    print(f"Training entries: {final_stats['total_entries_train']}")
    print(f"Validation entries: {final_stats['total_entries_val']}")
    total_entries = final_stats['total_entries_train'] + final_stats['total_entries_val']
    val_pct = (final_stats['total_entries_val'] / total_entries * 100) if total_entries > 0 else 0.0
    print(f"Validation fraction: {val_pct:.1f}%")
    print(f"Unassigned points: {final_stats['total_unassigned']}")
    print(f"Total runtime: {final_stats['elapsed_time']/60:.1f} min")
    print(f"Average per file: {final_stats['elapsed_time']/max(1,final_stats['completed']):.2f}s")
    print(f"Output files:")
    print(f"  Training: {output_file_train}")
    print(f"  Validation: {output_file_val}")

    # primaries = number of processed NC events (training + validation)
    for out_path in [output_file_train, output_file_val]:
        if os.path.exists(out_path):
            with h5py.File(out_path, 'a') as f:
                f['primaries'][()] = total_entries
    print(f"primaries set: {total_entries} NC events")

    if final_stats['completed'] == len(files) and final_stats['failed'] == 0:
        print(f"\nAll files processed successfully - removing temporary files...")
        progress_tracker.cleanup()
    else:
        print(f"\nProgress has been saved so the run can be resumed.")

    print_memory_usage("program end")

if __name__ == "__main__":
    main()
