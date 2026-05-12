#!/usr/bin/env python3
"""
Post-Processing für Drei-Quellen-Workflow (Multi-Optik):
  Simulation 1: Muon → Neutron Capture (NC) Daten (MyNeutronCaptureOutput), KEINE optischen Photonen
  Simulation 2: Gamma-Vertices aus NCs → Optische Photonen (eine Simulation pro Run, -i2)
  Simulation 3: Mehrfache optische Simulationen derselben NC-Events (-i3),
                Struktur: run_NNN/sim_NNN/output_t*.hdf5

Workflow:
  1. NC-Daten aus allen Sim1-Files laden (Position, Gamma-Info, Material, Volume)
  2. Globale Material/Volume-Mappings erstellen oder laden
  3. Alle optischen Files (Sim2 + alle Sim3 sim_***-Unterordner) sequenziell scannen:
     Photonen filtern und per (run_id, muon_id, nC_id) aggregieren.
     Optische Hits werden über ALLE Simulationen (Sim2 + jede Sim3 sim_***) aufsummiert.
  4. NC-Daten mit aggregierten Voxel-Hits zusammenführen
  5. HDF5-Output schreiben (identisches Format wie musun2SimFullPostprocessing.py)

Zuordnung Sim1 ↔ Sim2/Sim3:
  - optical/muon_track_id  ==  MyNeutronCaptureOutput/evtid
  - optical/nC_track_id    ==  MyNeutronCaptureOutput/nC_track_id

Zeitfilter:
  relative_time = photon_time - nC_time (aus Sim1)
  Muss >= 0 sein; Fenster: 0 <= relative_time <= 200 ns
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
    def __init__(self, output_dir, output_file):
        self.output_dir = Path(output_dir)
        self.output_file = Path(output_file)
        self.progress_file = self.output_dir / "processing_progress.json"
        self.lock_file = self.output_dir / "processing.lock"
        self.split_file = self.output_dir / "train_val_split.json"
        self.progress_data = self.load_progress()

    def load_progress(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                print(f"Bestehender Fortschritt gefunden: {len(data.get('completed_files', []))} Files bereits verarbeitet")
                return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warnung: Konnte Fortschritt nicht laden ({e}), starte neu")

        return {
            'completed_files': [],
            'failed_files': [],
            'start_time': time.time(),
            'last_update': time.time(),
            'total_entries_written': 0,
            'total_unassigned': 0,
            'output_file': str(self.output_file),
            'sim2_scan_complete': False,
        }

    def save_train_val_split(self, train_triplets, val_triplets, random_seed, val_fraction):
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
            print(f"Train/Val Split gespeichert in {self.split_file}")
        except IOError as e:
            print(f"Warnung: Konnte Split nicht speichern: {e}")

    def load_train_val_split(self):
        if not self.split_file.exists():
            return None
        try:
            with open(self.split_file, 'r') as f:
                split_data = json.load(f)
            train_triplets = set(tuple(t) for t in split_data['train_triplets'])
            val_triplets = set(tuple(t) for t in split_data['val_triplets'])
            print(f"Train/Val Split geladen:")
            print(f"  Training: {len(train_triplets)} NC-Events")
            print(f"  Validation: {len(val_triplets)} NC-Events")
            print(f"  Random Seed: {split_data['random_seed']}")
            print(f"  Val Fraction: {split_data['val_fraction']}")
            return (train_triplets, val_triplets,
                    split_data['random_seed'],
                    split_data['val_fraction'])
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Warnung: Konnte Split nicht laden ({e}), erstelle neuen")
            return None

    def cleanup(self):
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.lock_file.exists():
                self.lock_file.unlink()
        except OSError:
            pass

    def save_progress(self):
        self.progress_data['last_update'] = time.time()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except IOError as e:
            print(f"Warnung: Konnte Fortschritt nicht speichern: {e}")

    def log_processing_params(self, chunk_size, available_mem_gb):
        if 'processing_params' not in self.progress_data:
            self.progress_data['processing_params'] = []
        self.progress_data['processing_params'].append({
            'timestamp': time.time(),
            'chunk_size': chunk_size,
            'available_mem_gb': round(available_mem_gb, 2)
        })
        self.save_progress()

    def is_file_completed(self, file_path):
        return str(file_path) in self.progress_data['completed_files']

    def mark_file_completed(self, file_path, entries_count, unassigned_count):
        file_str = str(file_path)
        if file_str not in self.progress_data['completed_files']:
            self.progress_data['completed_files'].append(file_str)
            self.progress_data['total_entries_written'] += entries_count
            self.progress_data['total_unassigned'] += unassigned_count
            self.save_progress()

    def create_checkpoint(self, output_file):
        checkpoint_file = str(output_file).replace('.hdf5', '_checkpoint.hdf5')
        checkpoint_old = str(output_file).replace('.hdf5', '_checkpoint_old.hdf5')
        try:
            if os.path.exists(checkpoint_file):
                shutil.move(checkpoint_file, checkpoint_old)
            shutil.copy2(output_file, checkpoint_file)
            print(f"  Checkpoint erstellt: {os.path.basename(checkpoint_file)}")
        except Exception as e:
            print(f"  Warnung: Konnte Checkpoint nicht erstellen: {e}")

    def verify_hdf5_integrity(self, output_file):
        try:
            with h5py.File(output_file, 'r') as f:
                if 'phi_matrix' in f and 'target_matrix' in f and 'event_ids' in f:
                    _ = f['phi_matrix'].shape
                    return True
        except Exception as e:
            print(f"  WARNUNG: HDF5-Integritätsprüfung fehlgeschlagen: {e}")
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
        completed = set(self.progress_data['completed_files'])
        return [f for f in all_files if str(f) not in completed]

    def get_statistics(self):
        return {
            'completed': len(self.progress_data['completed_files']),
            'failed': len(self.progress_data['failed_files']),
            'total_entries': self.progress_data['total_entries_written'],
            'total_unassigned': self.progress_data['total_unassigned'],
            'elapsed_time': time.time() - self.progress_data['start_time']
        }

    def mark_sim2_scan_complete(self):
        self.progress_data['sim2_scan_complete'] = True
        self.save_progress()

    def is_sim2_scan_complete(self):
        return self.progress_data.get('sim2_scan_complete', False)


# ============================================================================
# Mapping Functions
# ============================================================================

def loadMapping(filePath: str) -> dict:
    with open(filePath, 'r') as f:
        mapping = json.load(f)
    return mapping


def remapMaterialIDsToGlobal(
    glob_mapping_json_path: str,
    local_mat_map: dict,
    local_material_ids
) -> tuple[np.ndarray, dict]:
    if not os.path.exists(glob_mapping_json_path):
        raise RuntimeError(
            f"Globales Material-Mapping fehlt: {glob_mapping_json_path}\n"
            f"Wurde collect_all_materials_first() ausgeführt?"
        )
    globMaterialMapping = loadMapping(glob_mapping_json_path)
    local_material_ids = np.array(local_material_ids)
    unique_local_ids = np.unique(local_material_ids)
    mapping_dict = {}

    for local_id in unique_local_ids:
        local_name = local_mat_map[local_id]
        if local_name not in globMaterialMapping:
            raise RuntimeError(
                f"Material '{local_name}' nicht im globalen Mapping gefunden!\n"
                f"Verfügbare Materialien: {list(globMaterialMapping.keys())}"
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
    if not os.path.exists(glob_mapping_json_path):
        raise RuntimeError(
            f"Globales Volume-Mapping fehlt: {glob_mapping_json_path}\n"
            f"Wurde collect_all_volumes_first() ausgeführt?"
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
                f"Volume '{local_name_clean}' nicht im globalen Mapping gefunden!\n"
                f"Verfügbare Volumes: {list(globVolumeMapping.keys())}"
            )
        mapping_dict[local_id] = globVolumeMapping[local_name_clean]

    vectorized_mapper = np.vectorize(mapping_dict.get)
    global_volume_ids = vectorized_mapper(local_volume_ids)
    return global_volume_ids, globVolumeMapping


def create_global_material_mapping(all_materials, existing_mapping_path):
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
    print("Sammle alle Materialien aus allen Sim1-Files...")
    all_materials = set()
    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                mapping_names = [x.decode() for x in f["hit/materials/materialNames"]["pages"][:]]
                all_materials.update(mapping_names)
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path}: {e}")
            continue
    print(f"Gefunden: {len(all_materials)} einzigartige Materialien")
    return all_materials


def collect_all_volumes_first(files):
    print("Sammle alle physikalischen Volumes aus allen Sim1-Files...")
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
            print(f"Fehler beim Lesen von {file_path}: {e}")
            continue
    print(f"Gefunden: {len(all_volumes)} einzigartige physikalische Volumes")
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
    m = _RUN_DIR_PATTERN.match(name)
    if m:
        return int(m.group(1))
    return -1


def _extract_run_id_from_path(file_path: str) -> int:
    """Returns the run_id embedded in a path like .../run_003/output_t0.hdf5 → 3."""
    for part in Path(file_path).parts:
        if part.startswith("run_"):
            try:
                return int(part.split("_")[1])
            except (IndexError, ValueError):
                pass
    return 0


def validate_run_structure(path: str, nested: bool, label: str):
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
            print(f"  [Sim3] WARNING: run_{run_id:03d} has no sim_NNN subdirectories — skipped")
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
                print(f"  [Sim3] WARNING: {run_dir.name}/{sim_dir.name} contains no HDF5 files — skipped")

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
            errors.append(f"  FEHLER: {run_name} ist aktiv, aber fehlt in Sim1 ({sim1_path})")
        if run_id not in sim3_run_to_sims:
            errors.append(f"  FEHLER: {run_name} ist aktiv, aber fehlt in Sim3 ({sim3_path})")
        elif not sim3_run_to_sims[run_id]:
            errors.append(f"  FEHLER: {run_name}/sim_*** in Sim3 enthält keine sim_NNN-Verzeichnisse")

    if errors:
        raise RuntimeError(
            "Konsistenzprüfung Sim1 ↔ Sim3 fehlgeschlagen:\n" + "\n".join(errors)
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
            f"  FEHLER: sim_NNN-Sets der aktiven Runs stimmen nicht überein "
            f"(Referenz: run_{reference_id:03d} mit {sorted(reference_sims)}):"
        ]
        for run_id, missing, extra in inconsistent:
            if missing:
                lines.append(f"    run_{run_id:03d}: fehlende sim-Dirs: {sorted(missing)}")
            if extra:
                lines.append(f"    run_{run_id:03d}: extra sim-Dirs:    {sorted(extra)}")
        raise RuntimeError("Konsistenzprüfung Sim1 ↔ Sim3 fehlgeschlagen:\n" + "\n".join(lines))

    # --- Informational: inactive runs in Sim1 ---
    inactive_in_sim1 = sim1_run_ids - active_run_ids
    if inactive_in_sim1:
        print(f"  INFO: Folgende Sim1-Runs sind NICHT aktiv und werden übersprungen: "
              f"{sorted(inactive_in_sim1)}")

    # --- Informational: runs in Sim3 not selected ---
    inactive_in_sim3 = set(sim3_run_to_sims.keys()) - active_run_ids
    if inactive_in_sim3:
        print(f"  INFO: Folgende Sim3-Runs sind vorhanden aber nicht aktiv: "
              f"{sorted(inactive_in_sim3)}")

    n_sims = len(reference_sims)
    sim_range = f"{sorted(reference_sims)[0]} … {sorted(reference_sims)[-1]}"
    print(f"  OK: {len(active_run_ids)} aktive Run(s), je {n_sims} sim_***-Verzeichnisse ({sim_range})")


# ============================================================================
# File Discovery
# ============================================================================

def find_all_hdf5_files(input_path: str, nested: bool) -> list[str]:
    files = []
    if nested:
        print(f"Suche nach Simulationsdaten in Unterordnern von {input_path}...")
        subdirs = [d for d in Path(input_path).iterdir() if d.is_dir()]
        for subdir in subdirs:
            subdir_files = sorted(glob.glob(os.path.join(subdir, 'output_*.hdf5')))
            if subdir_files:
                print(f"  Gefunden: {len(subdir_files)} Files in {subdir.name}")
                files.extend(subdir_files)
    else:
        files = sorted(glob.glob(os.path.join(input_path, 'output_*.hdf5')))
    return files


def find_sim2_files(sim2_path: str, nested: bool = False) -> dict[int, list[str]]:
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
            print(f"  Sim2 run_{run_id:03d}: {len(run_files)} Files")
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
    a single flat list per run_id.  This allows scan_sim2_and_aggregate() to
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
            print(f"  Sim3 run_{run_id:03d}: {len(sim_dirs)} sim directories → {len(run_files)} total Files")

    return files_by_run


def combine_optical_files(
    sim2_files_by_run: dict[int, list[str]],
    sim3_files_by_run: dict[int, list[str]],
) -> dict[int, list[str]]:
    """
    Merges Sim2 and Sim3 file lists per run_id into a single combined dict.

    Files from both sources are concatenated — scan_sim2_and_aggregate()
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


# ============================================================================
# Geometry & Voxel Functions
# ============================================================================

def defineZylinder(
    geometry_name: str,
    valid_detectors: Optional[List[int]] = None
):
    valid_geometry_names = ["currentDist"]
    if geometry_name not in valid_geometry_names:
        print("Invalid geometry name. It must be one of the following:")
        for name in valid_geometry_names:
            print(name)
        sys.exit()

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
        r_zylinder = 4300
        z_origin = 20
        h_zylinder = h + 20 - z_origin
        print("Set currentDist as geometry.")

    r_ref = 6000
    h_ref = 8080
    z_ref = 20

    return (t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top,
            z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref, valid_detectors)


def build_voxel_tree(voxels: List[Dict]) -> Optional[KDTree]:
    if not voxels:
        return None
    centers = [voxel['center'] for voxel in voxels]
    return KDTree(centers)


def assignToNearestVoxel(tree: Optional[KDTree], voxels: list[dict], point) -> str:
    if tree is None or len(voxels) == 0:
        return "-1"
    dist, idx = tree.query(point)
    voxel = voxels[idx]
    return str(voxel['index'])


def checkRadialMomentumVectorized(x, y, z, px, py, pz):
    r = np.sqrt(x**2 + y**2)
    valid_r = r > 0
    radial_momentum = np.zeros_like(r)
    radial_momentum[valid_r] = (x[valid_r]*px[valid_r] + y[valid_r]*py[valid_r]) / r[valid_r]
    return radial_momentum >= 0


# ============================================================================
# Train/Val Split
# ============================================================================

def collect_all_nc_triplets(files: list[str]) -> set[tuple]:
    print("Sammle alle Neutron Capture Events für Train/Val Split...")
    all_nc_triplets = set()
    for file_idx, file_path in enumerate(files):
        try:
            with h5py.File(file_path, 'r') as f:
                evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
                nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
                triplets = [(file_idx, int(evt), int(nc)) for evt, nc in zip(evtid, nC_id)]
                all_nc_triplets.update(triplets)
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path}: {e}")
            continue
    print(f"Gefunden: {len(all_nc_triplets)} einzigartige NC-Events über {len(files)} Files")
    return all_nc_triplets


def create_or_load_train_val_split(progress_tracker, all_nc_triplets,
                                   val_fraction, random_seed):
    loaded_split = progress_tracker.load_train_val_split()

    if loaded_split is not None:
        train_triplets, val_triplets, saved_seed, saved_fraction = loaded_split
        if saved_seed != random_seed:
            print(f"⚠ WARNUNG: Gespeicherter Random Seed ({saved_seed}) != aktueller Seed ({random_seed})")
            print(f"  Verwende gespeicherten Split für Konsistenz!")
        if abs(saved_fraction - val_fraction) > 0.001:
            print(f"⚠ WARNUNG: Gespeicherte Val-Fraction ({saved_fraction}) != aktuelle ({val_fraction})")
            print(f"  Verwende gespeicherten Split für Konsistenz!")

        all_triplets_set = set(all_nc_triplets)
        missing_train = train_triplets - all_triplets_set
        missing_val = val_triplets - all_triplets_set
        if missing_train or missing_val:
            print(f"⚠ WARNUNG: {len(missing_train) + len(missing_val)} NC-Events aus Split fehlen in Daten!")
            print(f"  Erstelle neuen Split...")
            loaded_split = None
        else:
            print(f"✓ Verwende gespeicherten Split: {len(val_triplets)} Validation-Events")
            return val_triplets

    if loaded_split is None:
        print(f"Erstelle neuen Train/Val Split (Event-Level) mit Seed {random_seed}...")
        evtid_groups = defaultdict(list)
        for file_idx, evt, nc in all_nc_triplets:
            evtid_groups[(file_idx, evt)].append((file_idx, evt, nc))

        event_keys = sorted(evtid_groups.keys())
        np.random.seed(random_seed)
        np.random.shuffle(event_keys)

        split_idx = int(len(event_keys) * (1 - val_fraction))
        train_event_keys = set(event_keys[:split_idx])

        train_triplets = set()
        val_triplets = set()
        for evt_key, triplets in evtid_groups.items():
            if evt_key in train_event_keys:
                train_triplets.update(triplets)
            else:
                val_triplets.update(triplets)

        print(f"Train/Val Split erstellt (Event-Level):")
        print(f"  Events gesamt: {len(event_keys)}")
        print(f"  Training: {len(train_event_keys)} Events → {len(train_triplets)} NC-Events")
        print(f"  Validation: {len(event_keys) - len(train_event_keys)} Events → {len(val_triplets)} NC-Events")
        actual_val = len(val_triplets) / (len(train_triplets) + len(val_triplets)) * 100
        print(f"  Tatsächlicher Val-Anteil: {actual_val:.1f}%")

        progress_tracker.save_train_val_split(train_triplets, val_triplets,
                                              random_seed, val_fraction)
        return val_triplets


def calculate_weight_from_files(files: list[str], sample_size: Optional[int] = None) -> float:
    print("Berechne Gewichtung...")
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
            print(f"Warnung: Konnte {file} für Gewichtung nicht lesen: {e}")
            continue

    if sample_size and sample_size < len(files):
        scale_factor = len(files) / sample_size
        estimated = len(unique_triplets) * scale_factor
        weight = 1 / estimated if estimated > 0 else 1.0
        print(f"Geschätztes Gewicht basierend auf {sample_size} Files: {weight:.2e}")
    else:
        weight = 1 / len(unique_triplets) if unique_triplets else 1.0
        print(f"Gewicht basierend auf allen {len(files)} Files: {weight:.2e}")
    return weight


# ============================================================================
# Phase 1: NC-Daten aus Sim1 laden
# ============================================================================

def load_all_nc_data_from_sim1(
    sim1_files: list[str],
    material_mapping_path: str,
    volume_mapping_path: str,
    nested: bool = False
) -> dict[tuple[int, int, int], dict]:
    """
    Lädt alle NC-Daten aus allen Sim1-Files.

    Returns:
        nc_data_dict: {(run_id, muon_id, nC_id): {nc_info...}}
    """
    print(f"\n=== Phase 1: Lade NC-Daten aus {len(sim1_files)} Sim1-Files ===")
    phase1_start = time.time()
    nc_data_dict = {}
    duplicate_count = 0

    def _get_run_id(file_path: str) -> int:
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
        print(f"  Lade Sim1-File {file_idx + 1}/{len(sim1_files)}: {os.path.basename(file_path)} (run_id={run_id})")

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
            print(f"  FEHLER beim Laden von {file_path}: {e}")
            raise

    if duplicate_count > 0:
        print(f"  ⚠ WARNUNG: {duplicate_count} Duplikat-(run_id, muon_id, nC_id) Triplets!")

    phase1_elapsed = time.time() - phase1_start
    print(f"  ✓ {len(nc_data_dict)} einzigartige NC-Events geladen in {phase1_elapsed/60:.1f} min")
    print_memory_usage("Nach Phase 1")
    return nc_data_dict


# ============================================================================
# Phase 2: Optische Dateien (Sim2 + alle Sim3) sequenziell scannen
# ============================================================================

def get_chunk_size() -> int:
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
    Scannt alle optischen Files (Sim2 + alle Sim3 sim_***) run-weise und
    aggregiert Voxel-Hits pro NC-Event.

    Hits aus Sim2 und allen Sim3-Simulationen werden in dieselben Counter
    akkumuliert — das Ergebnis ist die Summe aller optischen Simulationen.

    Args:
        combined_files_by_run: {run_id: [alle optischen HDF5-Dateien für diesen Run]}
                               (Sim2-Dateien + alle Sim3 sim_***-Dateien zusammengeführt)
        nc_data_dict: {(run_id, muon_id, nC_id): nc_info}
        voxel_tree: KDTree der Voxel-Mittelpunkte
        voxel_data: Voxel-Definitionen
        geometry_params: Geometrie-Parameter

    Returns:
        nc_voxel_counters: {(run_id, muon_id, nC_id): Counter} (aufsummiert)
        total_unassigned, total_orphaned, successfully_scanned_runs
    """
    total_optical_files = sum(len(v) for v in combined_files_by_run.values())
    print(f"\n=== Phase 2: Scanne {total_optical_files} optische Files über {len(combined_files_by_run)} Runs ===")
    print(f"  (Hits aus Sim2 und allen Sim3 sim_***-Verzeichnissen werden summiert)")

    h_zylinder = geometry_params['h_zylinder']
    r_zylinder = geometry_params['r_zylinder']
    valid_detectors = geometry_params.get('valid_detectors', [1965, 1966, 1967, 1968])
    valid_detectors_array = np.array(valid_detectors, dtype=np.int32)

    z_cut_bot = -4979
    z_cut_top = z_cut_bot + h_zylinder - 2

    nc_voxel_counters: dict[tuple[int, int, int], Counter] = defaultdict(Counter)
    total_unassigned = 0
    total_orphaned = 0

    CHUNK_SIZE = get_chunk_size()
    print(f"  Chunk-Size: {CHUNK_SIZE}")

    file_counter = 0
    phase2_start = time.time()

    for run_id in sorted(combined_files_by_run.keys()):
        optical_files = combined_files_by_run[run_id]
        print(f"\n  --- Run {run_id}: {len(optical_files)} optische Files (Sim2 + Sim3 summiert) ---")

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
            print(f"    Opt-File {opt_idx + 1}/{len(optical_files)} (global {file_counter}/{total_optical_files}): {os.path.basename(opt_file)}")
            file_unassigned = 0
            file_orphaned = 0
            file_matched = 0

            try:
                with h5py.File(opt_file, 'r') as f:
                    total_photons = len(f['hit']['optical']['x_position_in_m']['pages'])

                    if total_photons == 0:
                        print(f"      Keine optischen Photonen, überspringe.")
                        continue

                    num_chunks = (total_photons - 1) // CHUNK_SIZE + 1

                    for chunk_idx in range(num_chunks):
                        chunk_start = chunk_idx * CHUNK_SIZE
                        chunk_end = min(chunk_start + CHUNK_SIZE, total_photons)

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

                        # Filter 1: Detektor
                        det_mask = np.isin(det_uids, valid_detectors_array)
                        x, y, z = x[det_mask], y[det_mask], z[det_mask]
                        px, py, pz = px[det_mask], py[det_mask], pz[det_mask]
                        muon_ids = muon_ids[det_mask]
                        nc_ids = nc_ids[det_mask]
                        photon_times = photon_times[det_mask]

                        if len(x) == 0:
                            continue

                        # Filter 2: Zeitfenster
                        nc_times_arr = np.full(len(photon_times), np.inf, dtype=np.float64)
                        for idx in range(len(muon_ids)):
                            local_key = (int(muon_ids[idx]), int(nc_ids[idx]))
                            if local_key in nc_times_for_run:
                                nc_times_arr[idx] = nc_times_for_run[local_key]

                        has_nc = nc_times_arr != np.inf
                        file_orphaned += int(np.sum(~has_nc))

                        relative_times = np.where(has_nc, photon_times - nc_times_arr, np.inf)

                        negative_mask = has_nc & (relative_times < 0)
                        if np.any(negative_mask):
                            neg_indices = np.where(negative_mask)[0]
                            first_bad = neg_indices[0]
                            raise RuntimeError(
                                f"FATAL: Negative relative Photon-Zeit detektiert!\n"
                                f"  Opt-File: {opt_file}, run_id={run_id}\n"
                                f"  muon_id={int(muon_ids[first_bad])}, nC_id={int(nc_ids[first_bad])}\n"
                                f"  photon_time={float(photon_times[first_bad]):.4f} ns, "
                                f"nC_time={float(nc_times_arr[first_bad]):.4f} ns\n"
                                f"  relative_time={float(relative_times[first_bad]):.4f} ns\n"
                                f"  {int(np.sum(negative_mask))} Photonen mit negativer relativer Zeit."
                            )

                        time_mask = has_nc & (relative_times >= 0) & (relative_times <= 200.0)
                        x, y, z = x[time_mask], y[time_mask], z[time_mask]
                        px, py, pz = px[time_mask], py[time_mask], pz[time_mask]
                        muon_ids = muon_ids[time_mask]
                        nc_ids = nc_ids[time_mask]

                        if len(x) == 0:
                            continue

                        # Filter 3: Momentum
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

                        # Voxel-Zuordnung (vektorisiert — Batch-KDTree)
                        points = np.column_stack((x, y, z))
                        _, voxel_idx_array = voxel_tree.query(points)
                        voxel_index_strings = np.array(
                            [str(voxel_data[vi]['index']) for vi in voxel_idx_array]
                        )

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
                print(f"      FEHLER beim Verarbeiten von {opt_file}: {e}")
                raise

            total_unassigned += file_unassigned
            total_orphaned += file_orphaned
            print(f"      Matched: {file_matched}, Orphaned: {file_orphaned}, Unassigned: {file_unassigned}")

        run_elapsed = time.time() - phase2_start
        print(f"  Run {run_id} abgeschlossen in {run_elapsed/60:.1f} min (kumulativ)")
        if file_counter % 50 == 0:
            print_memory_usage(f"Nach {file_counter} optischen Files")

    ncs_with_photons = sum(1 for c in nc_voxel_counters.values() if c)
    ncs_without_photons = len(nc_data_dict) - ncs_with_photons
    total_hits = sum(sum(c.values()) for c in nc_voxel_counters.values())

    phase2_elapsed = time.time() - phase2_start
    print(f"\n  ✓ Optischer Scan abgeschlossen in {phase2_elapsed/60:.1f} min:")
    print(f"    NC-Events mit Photonen: {ncs_with_photons}")
    print(f"    NC-Events ohne Photonen: {ncs_without_photons}")
    print(f"    Gesamte Voxel-Hits (summiert über alle Sim): {total_hits}")
    print(f"    Nicht zugeordnete Photonen: {total_unassigned}")
    print(f"    Orphaned Photonen (kein NC): {total_orphaned}")
    print_memory_usage("Nach Phase 2")

    successfully_scanned_runs = set(combined_files_by_run.keys())

    return dict(nc_voxel_counters), total_unassigned, total_orphaned, successfully_scanned_runs


# ============================================================================
# Phase 3: Zusammenführung und HDF5-Output
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
) -> dict:
    """
    Erstellt Output-Daten batchweise und schreibt direkt in HDF5.
    Identisches Output-Format wie musun2SimFullPostprocessing.py.
    """
    print(f"\n=== Phase 3: Schreibe Output batchweise für {len(nc_data_dict)} NC-Events ===")
    phase3_start = time.time()
    print(f"  Batch-Size: {batch_size}")

    r_zylinder = geometry_params['r_zylinder']
    h_zylinder = geometry_params['h_zylinder']
    z_cut_bot = -4979
    z_cut_top = z_cut_bot + h_zylinder - 2

    nc_without_photons = 0
    nc_with_photons = 0
    total_written = 0

    all_keys = list(nc_data_dict.keys())
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

            gamma_row = []
            for i in range(4):
                e = nc_info['gamma_energies'][i]
                px = nc_info['gamma_px'][i]
                py = nc_info['gamma_py'][i]
                pz = nc_info['gamma_pz'][i]
                if e < 0:
                    e, px, py, pz = 0.0, 0.0, 0.0, 0.0
                gamma_row.extend([e, px, py, pz])

            r_NC = np.sqrt(x**2 + y**2)
            phi_NC = np.arctan2(y, x)
            dist_to_wall = r_zylinder - r_NC
            dist_to_bot = z - z_cut_bot
            dist_to_top = z_cut_top - z

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

            phi_row = [x, y, z, mat_id, vol_id, n_gamma, e_tot,
                       r_NC, phi_NC, dist_to_wall, dist_to_bot, dist_to_top,
                       p_mean_r, p_mean_z,
                       nC_time, flag_ge77] + gamma_row

            target_row = [voxel_counter.get(str(voxel_idx), 0) for voxel_idx in voxel_indices]

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

        del phi_data_batch, target_data_batch, target_regions_batch, event_ids_batch, batch_result
        gc.collect()

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"  Batch {batch_idx + 1}/{num_batches}: {total_written} Einträge geschrieben")

    phase3_elapsed = time.time() - phase3_start
    print(f"\n  ✓ Schreiben abgeschlossen in {phase3_elapsed/60:.1f} min:")
    print(f"    NC-Events mit Photonen: {nc_with_photons}")
    print(f"    NC-Events ohne Photonen: {nc_without_photons}")
    print(f"    Gesamt geschrieben: {total_written}")
    print_memory_usage("Nach batchweisem Schreiben")

    return {
        'nc_with_photons': nc_with_photons,
        'nc_without_photons': nc_without_photons,
        'total_written': total_written,
    }


# ============================================================================
# HDF5 Output
# ============================================================================

def create_or_open_output_file(output_path, file_index, voxel_data, mat_map, vol_map, radius, n_primaries=0, suffix=""):
    """
    Erstellt eine neue HDF5-Datei oder öffnet eine bestehende.
    Identisches Format wie musun2SimFullPostprocessing.py.
    """
    output_file = os.path.join(output_path, f"ncscore_output_{file_index}{suffix}.hdf5")

    if os.path.exists(output_file):
        try:
            with h5py.File(output_file, 'r') as f:
                if ('phi_matrix' in f and 'target_matrix' in f and
                        'voxels' in f and 'voxel_metadata' in f and 'event_ids' in f):
                    print(f"Bestehende Output-Datei gefunden: {output_file}")
                    return output_file
        except:
            print(f"Bestehende Datei ist korrupt, erstelle neue: {output_file}")
            os.remove(output_file)

    print(f"Erstelle neue Output-Datei: {output_file}")

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

    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fcpl.set_sizes(8, 8)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fid = h5py.h5f.create(output_file.encode(), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
    with h5py.File(fid) as out:
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
        out.create_dataset(
            "event_ids",
            shape=(0, len(event_id_columns)),
            maxshape=(None, len(event_id_columns)),
            dtype=np.int64,
            chunks=(1000, len(event_id_columns)),
            compression='lzf'
        )

        dt = h5py.string_dtype(encoding='utf-8')
        out.create_dataset("target_columns", data=[str(idx) for idx in voxel_indices], dtype=dt)
        out.create_dataset("phi_columns", data=phi_columns, dtype=dt)
        out.create_dataset("region_columns", data=region_columns, dtype=dt)
        out.create_dataset("event_id_columns", data=event_id_columns, dtype=dt)

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

        voxels_grp = out.create_group("voxels")
        print(f"  Initialisiere {len(voxel_data)} Voxel-Metadaten...")
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

        print(f"  ✓ Alle {len(voxel_data)} Voxel-Metadaten erstellt")

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
        print(f"  ✓ Voxel Metadata erstellt: Shape {voxel_metadata.shape}")

    return output_file


def append_results_to_hdf5(output_file, result_data, voxel_indices, weight):
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
        description='Post-Processing: Neutron Captures + mehrere optische Simulationen (summiert)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Drei-Quellen-Workflow:
  -i1: NC-Daten (Sim1, run_NNN/ Unterordner)
  -i2: Erste optische Simulation (run_NNN/ Unterordner, eine Sim pro Run)
  -i3: Mehrfache optische Simulationen (run_NNN/sim_NNN/ Struktur)

Die optischen Hits aus -i2 und ALLEN sim_NNN in -i3 werden für jeden NC-Event summiert.

Beispiel:
  python %(prog)s \\
    -i1 /path/to/sim1 \\
    -i2 /path/to/sim2 \\
    -i3 /path/to/sim3_multi \\
    -o  /path/to/output \\
    --nested --muons-per-run 1000
        """
    )

    parser.add_argument('-i1', '--input-sim1', required=True,
                        help='Eingabe-Pfad für Simulation 1 (NC-Daten)')
    parser.add_argument('-i2', '--input-sim2', required=True,
                        help='Eingabe-Pfad für Simulation 2 (erste optische Simulation, run_NNN/)')
    parser.add_argument('-i3', '--input-sim3', required=True,
                        help='Eingabe-Pfad für mehrfache optische Simulationen (run_NNN/sim_NNN/)')
    parser.add_argument('-o', '--output', required=True,
                        help='Ausgabe-Pfad für Ergebnisse')

    parser.add_argument('-m', '--material-mapping',
                        default='/global/cfs/projectdirs/legend/users/tbuerger/postprocessing/src/mappings/globalMaterialMappings.json',
                        help='Pfad zur Material-Mapping-Datei')
    parser.add_argument('-v', '--volume-mapping',
                        default='/global/cfs/projectdirs/legend/users/tbuerger/postprocessing/src/mappings/globalPhysVolMappings.json',
                        help='Pfad zur Volume-Mapping-Datei')
    parser.add_argument('-x', '--voxel-file',
                        default='/global/cfs/projectdirs/legend/users/tbuerger/postprocessing/src/voxels/currentDistZylVoxelsPMTSize.json',
                        help='Pfad zur Voxel-Datei')

    parser.add_argument('--reset', action='store_true',
                        help='Verarbeitung von vorne beginnen')
    parser.add_argument('--sample-weight', type=int, default=50,
                        help='Anzahl Sim1-Files für Gewichtungsberechnung (0 = alle)')
    parser.add_argument('--nested', action='store_true',
                        help='Suche HDF5-Dateien in run_*/Unterordnern (gilt für Sim1 und Sim2)')

    parser.add_argument('-g', '--geometry', default='currentDist',
                        choices=['currentDist'],
                        help='Geometrie-Name (default: currentDist)')
    parser.add_argument('--valid-detectors', type=int, nargs='+',
                        default=[1965, 1966, 1967, 1968],
                        help='Liste der gültigen Detector UIDs')
    parser.add_argument('--verbose', '-V', action='store_true',
                        help='Ausführliche Ausgabe')
    parser.add_argument('--muons-per-run', type=int, required=True,
                        help='Anzahl simulierter Myonen pro Run (für primaries-Dataset im Output)')
    parser.add_argument('--sim3-runs', type=int, nargs='+', default=None,
                        metavar='RUN_ID',
                        help='Welche Run-IDs aus Sim3 verarbeitet werden sollen (z.B. --sim3-runs 1 '
                             'oder --sim3-runs 1 2 3). Nur diese Runs werden in den Output '
                             'geschrieben; alle anderen werden übersprungen. Fehlt diese Option, '
                             'werden alle in Sim3 vorhandenen Runs verwendet. Alle angegebenen '
                             'Runs müssen dieselbe Anzahl an sim_***-Verzeichnissen besitzen, '
                             'sonst wird abgebrochen.')

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_arguments()
    print_memory_usage("Hauptprogramm Start")

    for label, path in [('Sim1', args.input_sim1), ('Sim2', args.input_sim2),
                        ('Sim3', args.input_sim3)]:
        if not os.path.exists(path):
            print(f"Fehler: {label}-Pfad existiert nicht: {path}")
            sys.exit(1)
    if not os.path.exists(args.voxel_file):
        print(f"Fehler: Voxel-Datei existiert nicht: {args.voxel_file}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # === Struktur validieren ===
    print("\n=== Validiere Eingabe-Strukturen ===")
    if args.nested:
        validate_run_structure(args.input_sim1, nested=True, label="Sim1")
        validate_run_structure(args.input_sim2, nested=True, label="Sim2")
        print("=== Prüfe Konsistenz Sim1 ↔ Sim2 ===")
        validate_sim1_sim2_consistency(args.input_sim1, args.input_sim2)
    else:
        validate_run_structure(args.input_sim1, nested=False, label="Sim1")
        validate_run_structure(args.input_sim2, nested=False, label="Sim2")

    print("\n=== Validiere Sim3-Struktur (run_NNN/sim_NNN/) ===")
    sim3_summary = validate_sim3_structure(args.input_sim3)

    # Resolve which runs to process
    all_sim3_run_ids = set(sim3_summary.keys())
    if args.sim3_runs is not None:
        active_run_ids = set(args.sim3_runs)
        print(f"\n--sim3-runs angegeben: verarbeite nur Run(s) {sorted(active_run_ids)}")
    else:
        active_run_ids = all_sim3_run_ids
        print(f"\nKein --sim3-runs angegeben: verarbeite alle {len(active_run_ids)} Sim3-Run(s)")

    print("=== Prüfe Konsistenz Sim1 ↔ Sim3 (nur aktive Runs) ===")
    try:
        validate_sim1_sim3_consistency(args.input_sim1, args.input_sim3, active_run_ids)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    # === Geometrie ===
    geometry_result = defineZylinder(args.geometry, args.valid_detectors)
    (t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top,
     z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref, valid_detectors) = geometry_result

    geometry_params = {
        'h_zylinder': h_zylinder,
        'r_zylinder': r_zylinder,
        'valid_detectors': valid_detectors,
    }

    # === Voxel-Daten laden ===
    print("Lade Voxel-Daten...")
    with open(args.voxel_file, 'r') as f:
        voxel_data = json.load(f)
    voxel_tree = build_voxel_tree(voxel_data)
    voxel_indices = [voxel['index'] for voxel in voxel_data]
    print(f"Voxel-Tree mit {len(voxel_data)} Voxeln erstellt")

    # === Files finden ===
    sim1_files_all = find_all_hdf5_files(args.input_sim1, args.nested)
    if not sim1_files_all:
        print("Fehler: Keine output_*.hdf5 Dateien in Sim1-Pfad gefunden!")
        sys.exit(1)

    sim2_files_by_run_all = find_sim2_files(args.input_sim2, args.nested)
    if not sim2_files_by_run_all:
        print("Fehler: Keine output_t*.hdf5 Dateien in Sim2-Pfad gefunden!")
        sys.exit(1)

    print("\nSuche Sim3-Files...")
    sim3_files_by_run_all = find_sim3_files(args.input_sim3)
    if not sim3_files_by_run_all:
        print("Fehler: Keine HDF5-Dateien in Sim3-Pfad gefunden!")
        sys.exit(1)

    # === Auf aktive Runs filtern ===
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

    # === Run-Selektion zusammenfassen ===
    all_known_run_ids = (
        {_extract_run_id_from_path(f) for f in sim1_files_all}
        | set(sim2_files_by_run_all.keys())
        | set(sim3_files_by_run_all.keys())
    ) if args.nested else active_run_ids

    skipped_run_ids = all_known_run_ids - active_run_ids

    print(f"\n{'='*60}")
    print(f"RUN-SELEKTION")
    print(f"{'='*60}")
    print(f"  Aktive Runs   : {sorted(active_run_ids)}")
    if skipped_run_ids:
        print(f"  Übersprungene : {sorted(skipped_run_ids)}  ← nicht in --sim3-runs, werden NICHT in Output geschrieben")
    else:
        print(f"  Übersprungene : keine")
    print(f"{'='*60}")

    total_sim3_sims = sum(len(sims) for run_id, sims in sim3_summary.items()
                          if run_id in active_run_ids)
    total_sim2_files = sum(len(v) for v in sim2_files_by_run.values())
    total_sim3_files = sum(len(v) for v in sim3_files_by_run.values())
    print(f"Sim1: {len(sim1_files)} Dateien (aktive Runs)")
    print(f"Sim2: {total_sim2_files} Dateien in {len(sim2_files_by_run)} aktiven Runs")
    print(f"Sim3: {total_sim3_files} Dateien in {total_sim3_sims} sim_***-Verzeichnissen über {len(sim3_files_by_run)} aktive Runs")

    # === Sim2 + Sim3 Files kombinieren ===
    combined_files_by_run = combine_optical_files(sim2_files_by_run, sim3_files_by_run)
    total_combined = sum(len(v) for v in combined_files_by_run.values())
    print(f"\nKombinierte optische Files: {total_combined} ({total_sim2_files} Sim2 + {total_sim3_files} Sim3)")
    for run_id in sorted(combined_files_by_run.keys()):
        n2 = len(sim2_files_by_run.get(run_id, []))
        n3 = len(sim3_files_by_run.get(run_id, []))
        print(f"  run_{run_id:03d}: {n2} Sim2 + {n3} Sim3 = {n2+n3} optische Files")

    # === Output-Datei ===
    output_file = os.path.join(args.output, "ncscore_output_0.hdf5")
    progress_tracker = ProgressTracker(args.output, output_file)

    if args.reset:
        print("Reset angefordert - lösche bestehende Ergebnisse...")
        if os.path.exists(output_file):
            os.remove(output_file)
        progress_tracker.cleanup()
        progress_tracker = ProgressTracker(args.output, output_file)

    # === Materialien & Volumes ===
    if not os.path.exists(output_file):
        all_materials = collect_all_materials_first(sim1_files)
        all_volumes = collect_all_volumes_first(sim1_files)
        global_material_mapping = create_global_material_mapping(all_materials, args.material_mapping)
        print(f"Globales Material-Mapping: {len(global_material_mapping)} Materialien")
        global_volume_mapping = create_global_volume_mapping(all_volumes, args.volume_mapping)
        print(f"Globales Volume-Mapping: {len(global_volume_mapping)} Volumen")
    else:
        global_material_mapping = loadMapping(args.material_mapping)
        print(f"Bestehendes Material-Mapping geladen: {len(global_material_mapping)} Materialien")
        global_volume_mapping = loadMapping(args.volume_mapping)
        print(f"Bestehendes Volume-Mapping geladen: {len(global_volume_mapping)} Volumen")

    # === Gewichtung (basiert nur auf Sim1, identisch zur Referenz-Implementierung) ===
    sample_size = args.sample_weight if args.sample_weight > 0 else None
    weight = calculate_weight_from_files(sim1_files, sample_size)

    # === Output-Datei erstellen ===
    n_primaries = len(sim1_files) * args.muons_per_run
    output_file = create_or_open_output_file(
        args.output, 0, voxel_data, global_material_mapping,
        global_volume_mapping, r_zylinder, n_primaries=n_primaries
    )

    # =======================================================================
    # Hauptverarbeitung
    # =======================================================================
    start_time = time.time()

    try:
        # Phase 1: NC-Daten aus Sim1 laden
        nc_data_dict = load_all_nc_data_from_sim1(
            sim1_files, args.material_mapping, args.volume_mapping,
            nested=args.nested
        )

        # Phase 2: Alle optischen Files (Sim2 + Sim3) scannen, Hits summieren
        nc_voxel_counters, total_unassigned, total_orphaned, scanned_runs = scan_optical_and_aggregate(
            combined_files_by_run, nc_data_dict, voxel_tree, voxel_data, geometry_params
        )

        # NCs aus Runs ohne optische Daten entfernen
        nc_keys_before = len(nc_data_dict)
        nc_data_dict = {
            key: val for key, val in nc_data_dict.items()
            if key[0] in scanned_runs
        }
        nc_dropped = nc_keys_before - len(nc_data_dict)
        if nc_dropped > 0:
            print(f"  ⚠ {nc_dropped} NC-Events aus Runs ohne optische Daten entfernt")

        # Phase 3: Output batchweise schreiben
        output_stats = write_output_in_batches(
            nc_data_dict, nc_voxel_counters, voxel_data, voxel_indices,
            geometry_params, output_file, weight, batch_size=10000
        )

        total_entries = output_stats['total_written']

        del nc_data_dict, nc_voxel_counters
        gc.collect()

        progress_tracker.verify_hdf5_integrity(output_file)

    except KeyboardInterrupt:
        print(f"\nVerarbeitung durch Benutzer unterbrochen.")
        print(f"Fortschritt wurde gespeichert.")
        return
    except Exception as e:
        print(f"\nKritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        return

    # =======================================================================
    # Finale Statistiken
    # =======================================================================
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"VERARBEITUNG ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"Sim1-Files (NC-Daten): {len(sim1_files)}")
    print(f"Sim2-Files (erste optische Sim): {total_sim2_files}")
    print(f"Sim3-Files (mehrfache optische Sim, summiert): {total_sim3_files} in {total_sim3_sims} sim_***-Dirs")
    print(f"Optische Files gesamt (Sim2 + Sim3): {total_combined}")
    print(f"NC-Events gesamt: {total_entries}")
    print(f"NC mit Photonen: {output_stats['nc_with_photons']}")
    print(f"NC ohne Photonen: {output_stats['nc_without_photons']}")
    print(f"Nicht zugeordnete Photonen: {total_unassigned}")
    print(f"Orphaned Photonen: {total_orphaned}")
    print(f"Gesamtlaufzeit: {total_time / 60:.1f} min")
    print(f"Output-Datei: {output_file}")

    progress_tracker.cleanup()
    print_memory_usage("Programm Ende")


if __name__ == "__main__":
    main()
