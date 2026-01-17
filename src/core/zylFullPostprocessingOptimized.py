import os
import glob
import json
import h5py
import sys
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

"""
Das Post-Processing beinhaltet mehrere Schritte:
1. Das lokale Material Mapping auf das globale ändern und das globale gegebenenfalls erweitern
2. Photonen filtern die von dem SSD von außen aufgefangen wurden. In der wechten Welt können nur Photonen die von innen kommen detektiert werden.
3. Die Photonen den Voxeln zuordnen
4. Alle gelevanten Daten in ein hdf5 File schreiben was dem RESuM Format entspricht.

Das Programm merkt sich welche Files bereits verarbeitet wurden
Progress-Tracking: Status wird in einer JSON-Datei gespeichert
"""

def get_memory_usage():
    """Gibt aktuelle Speichernutzung zurück"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent()
    }

def print_memory_usage(label=""):
    """Druckt Speichernutzung mit Label"""
    mem = get_memory_usage()
    print(f"Memory {label}: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, {mem['percent']:.1f}%")

class ProgressTracker:
    """Klasse zum Verwalten des Verarbeitungsfortschritts"""
    
    def __init__(self, output_dir, output_file_train, output_file_val):
        self.output_dir = Path(output_dir)
        self.output_file_train = Path(output_file_train)  # [NEU]
        self.output_file_val = Path(output_file_val)      # [NEU]
        self.progress_file = self.output_dir / "processing_progress.json"
        self.lock_file = self.output_dir / "processing.lock"
        self.split_file = self.output_dir / "train_val_split.json"
        self.progress_data = self.load_progress()
    
    def load_progress(self):
        """Lädt bestehenden Fortschritt oder erstellt neuen"""
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
            'total_entries_written_train': 0,   
            'total_entries_written_val': 0,
            'total_unassigned': 0,
            'output_file_train': str(self.output_file_train), 
            'output_file_val': str(self.output_file_val)      
        }
    
    def save_train_val_split(self, train_triplets, val_triplets, random_seed, val_fraction):
        """
        Speichert Train/Val Split für Reproduzierbarkeit.
        
        Args:
            train_triplets: Set von (file_idx, evtid, nC_id) Tupeln für Training
            val_triplets: Set von (file_idx, evtid, nC_id) Tupeln für Validation
            random_seed: Verwendeter Random Seed
            val_fraction: Validation-Anteil
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
            print(f"Train/Val Split gespeichert in {self.split_file}")
        except IOError as e:
            print(f"Warnung: Konnte Split nicht speichern: {e}")
    
    def load_train_val_split(self):
        """
        Lädt gespeicherten Train/Val Split.
        
        Returns:
            tuple: (train_triplets, val_triplets, random_seed, val_fraction) oder None
        """
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
        """Räumt temporäre Dateien auf (nach erfolgreichem Abschluss)"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.lock_file.exists():
                self.lock_file.unlink()
            # Split-Datei NICHT löschen - könnte für Reproduzierbarkeit benötigt werden
        except OSError:
            pass

    def save_progress(self):
        """Speichert aktuellen Fortschritt"""
        self.progress_data['last_update'] = time.time()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except IOError as e:
            print(f"Warnung: Konnte Fortschritt nicht speichern: {e}")

    def log_processing_params(self, chunk_size, available_mem_gb):
        """Loggt Verarbeitungsparameter ins Progress-File"""
        if 'processing_params' not in self.progress_data:
            self.progress_data['processing_params'] = []
        
        self.progress_data['processing_params'].append({
            'timestamp': time.time(),
            'chunk_size': chunk_size,
            'available_mem_gb': round(available_mem_gb, 2)
        })
        self.save_progress()

    def is_file_completed(self, file_path):
        """Prüft ob eine Datei bereits verarbeitet wurde"""
        return str(file_path) in self.progress_data['completed_files']

    def mark_file_completed(self, file_path, entries_count_train, entries_count_val, unassigned_count):
        """Markiert eine Datei als erfolgreich verarbeitet"""
        file_str = str(file_path)
        if file_str not in self.progress_data['completed_files']:
            self.progress_data['completed_files'].append(file_str)
            self.progress_data['total_entries_written_train'] += entries_count_train
            self.progress_data['total_entries_written_val'] += entries_count_val
            self.progress_data['total_unassigned'] += unassigned_count
            self.save_progress()

    def create_checkpoint(self, output_file):
        """Erstellt einen Checkpoint der aktuellen HDF5-Datei mit Rotation"""
        checkpoint_file = str(output_file).replace('.hdf5', '_checkpoint.hdf5')
        checkpoint_old = str(output_file).replace('.hdf5', '_checkpoint_old.hdf5')

        try:
            # Rotate checkpoints
            if os.path.exists(checkpoint_file):
                shutil.move(checkpoint_file, checkpoint_old)
            
            shutil.copy2(output_file, checkpoint_file)
            print(f"  Checkpoint erstellt: {os.path.basename(checkpoint_file)}")
        except Exception as e:
            print(f"  Warnung: Konnte Checkpoint nicht erstellen: {e}")

    def verify_hdf5_integrity(self, output_file):
        """Prüft ob HDF5-Datei lesbar ist"""
        try:
            with h5py.File(output_file, 'r') as f:
                if 'phi' in f and 'xNC_mm' in f['phi']:
                    _ = f['phi']['xNC_mm'].shape
                    return True
        except Exception as e:
            print(f"  WARNUNG: HDF5-Integritätsprüfung fehlgeschlagen: {e}")
            return False
        return False

    def mark_file_failed(self, file_path, error_msg):
        """Markiert eine Datei als fehlgeschlagen"""
        self.progress_data['failed_files'].append({
            'file': str(file_path),
            'error': str(error_msg),
            'timestamp': time.time()
        })
        self.save_progress()

    def get_remaining_files(self, all_files):
        """Gibt Liste der noch zu verarbeitenden Dateien zurück"""
        completed = set(self.progress_data['completed_files'])
        return [f for f in all_files if str(f) not in completed]

    def get_statistics(self):
        """Gibt Statistiken zurück"""
        return {
            'completed': len(self.progress_data['completed_files']),
            'failed': len(self.progress_data['failed_files']),
            'total_entries_train': self.progress_data['total_entries_written_train'],
            'total_entries_val': self.progress_data['total_entries_written_val'],
            'total_unassigned': self.progress_data['total_unassigned'],
            'elapsed_time': time.time() - self.progress_data['start_time']
        }

def loadMapping(filePath):
    with open(filePath, 'r') as f:
        mapping = json.load(f)
    return mapping

def remapMaterialIDsToGlobal(
    glob_mapping_json_path: str,
    local_mat_map: dict,
    local_material_ids
) -> tuple[np.ndarray, dict]:
    """Vectorized material ID remapping - READ-ONLY version"""
    
    # Lade globales Mapping (sollte bereits vollständig sein!)
    if not os.path.exists(glob_mapping_json_path):
        raise RuntimeError(
            f"Globales Material-Mapping fehlt: {glob_mapping_json_path}\n"
            f"Wurde collect_all_materials_first() ausgeführt?"
        )
    
    globMaterialMapping = loadMapping(glob_mapping_json_path)
    
    local_material_ids = np.array(local_material_ids)
    unique_local_ids = np.unique(local_material_ids)
    mapping_dict = {}

    # CHANGED: Nur mappen, NICHT erweitern!
    for local_id in unique_local_ids:
        local_name = local_mat_map[local_id]
        
        if local_name not in globMaterialMapping:
            raise RuntimeError(
                f"Material '{local_name}' nicht im globalen Mapping gefunden!\n"
                f"Dies sollte nicht passieren nach collect_all_materials_first().\n"
                f"Verfügbare Materialien: {list(globMaterialMapping.keys())}"
            )
        
        mapping_dict[local_id] = globMaterialMapping[local_name]

    # Vectorize the mapping process
    vectorized_mapper = np.vectorize(mapping_dict.get)
    global_material_ids = vectorized_mapper(local_material_ids)

    return global_material_ids, globMaterialMapping  # Mapping unverändert zurückgeben

def remapVolumeIDsToGlobal(
    glob_mapping_json_path: str,
    local_vol_map: dict,
    local_volume_ids
) -> tuple[np.ndarray, dict]:
    """Vectorized volume ID remapping - READ-ONLY version"""
    
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
                f"Dies sollte nicht passieren nach collect_all_volumes_first().\n"
                f"Verfügbare Volumes: {list(globVolumeMapping.keys())}"
            )
        
        mapping_dict[local_id] = globVolumeMapping[local_name_clean]

    vectorized_mapper = np.vectorize(mapping_dict.get)
    global_volume_ids = vectorized_mapper(local_volume_ids)

    return global_volume_ids, globVolumeMapping

def create_global_material_mapping(all_materials, existing_mapping_path):
    """Erstellt vollständiges globales Material-Mapping"""
    if os.path.exists(existing_mapping_path):
        with open(existing_mapping_path, 'r') as f:
            global_mapping = json.load(f)
    else:
        global_mapping = {}
    
    # Höchste existierende ID finden
    max_id = max(global_mapping.values()) if global_mapping else -1
    
    # Neue Materialien hinzufügen
    for material in all_materials:
        if material not in global_mapping:
            max_id += 1
            global_mapping[material] = max_id
    
    # Mapping speichern
    with open(existing_mapping_path, 'w') as f:
        json.dump(global_mapping, f, indent=2)
    
    return global_mapping

def create_global_volume_mapping(all_volumes, existing_mapping_path):
    """Erstellt vollständiges globales Volume-Mapping"""
    if os.path.exists(existing_mapping_path):
        with open(existing_mapping_path, 'r') as f:
            global_mapping = json.load(f)
    else:
        global_mapping = {}
    
    # Höchste existierende ID finden
    max_id = max(global_mapping.values()) if global_mapping else -1
    
    # Neue Volumes hinzufügen
    for volume in all_volumes:
        if volume not in global_mapping:
            max_id += 1
            global_mapping[volume] = max_id
    
    # Mapping speichern
    with open(existing_mapping_path, 'w') as f:
        json.dump(global_mapping, f, indent=2)
    
    return global_mapping

def collect_all_materials_first(files):
    """Sammelt alle Materialien aus allen Files VOR der parallelen Verarbeitung"""
    print("Sammle alle Materialien aus allen Files...")
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
    """
    Sammelt alle physikalischen Volumes aus allen Files mit Sanitization.
    """
    print("Sammle alle physikalischen Volumes aus allen Files...")
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
            print(f"Fehler beim Lesen von {file_path}: {e}")
            continue
    
    print(f"Gefunden: {len(all_volumes)} einzigartige physikalische Volumes")
    
    # Verify no empty strings remain
    if "" in all_volumes:
        print(f"  ⚠ WARNING: Empty string still present after sanitization!")
        all_volumes.remove("")
        all_volumes.add("noVolume")
    
    return all_volumes

def find_all_hdf5_files(input_path, nested):
    """Findet alle HDF5-Dateien entweder direkt oder in Unterordnern"""
    files = []
    
    if nested:
        # Suche in Unterordnern
        print(f"Suche nach Simulationsdaten in Unterordnern von {input_path}...")
        subdirs = [d for d in Path(input_path).iterdir() if d.is_dir()]
        for subdir in subdirs:
            subdir_files = sorted(glob.glob(os.path.join(subdir, 'output_*.hdf5')))
            if subdir_files:
                print(f"  Gefunden: {len(subdir_files)} Files in {subdir.name}")
                files.extend(subdir_files)
    else:
        # Suche direkt im angegebenen Ordner
        files = sorted(glob.glob(os.path.join(input_path, 'output_*.hdf5')))
    
    return files

def checkRadialMomentumVectorized(x, y, z, px, py, pz):
    """Vectorized radial momentum calculation"""
    r = np.sqrt(x**2 + y**2)
    valid_r = r > 0
    radial_momentum = np.zeros_like(r)
    
    # Avoid division by zero for radial components
    radial_momentum[valid_r] = (x[valid_r]*px[valid_r] + y[valid_r]*py[valid_r]) / r[valid_r]
    return radial_momentum >= 0


# Neue Zuordnungsfunktion mit KDTree
def assignToNearestVoxel(tree, voxels, point):
    if tree is None or len(voxels) == 0:
        return "-1"
    
    # Find nearest voxel via KD-tree (based on voxel center)
    dist, idx = tree.query(point)
    voxel = voxels[idx]
    return str(voxel['index'])


# Baue KD-Trees für alle Voxel-Listen (vor der Schleife über Dateien)
def build_voxel_tree(voxels):
    if not voxels:
        return None
    centers = [voxel['center'] for voxel in voxels]
    return KDTree(centers)
    
def defineZylinder(geometry_name, valid_detectors=None):
    """
    Definiert Zylinder-Geometrie und optionale Detector-UIDs.
    
    Args:
        geometry_name: Name der Geometrie ('currentDist')
        valid_detectors: Liste der gültigen Detector UIDs (default: [1965, 1966, 1967, 1968])
    
    Returns:
        tuple: Geometrie-Parameter inkl. valid_detectors
    """

    valid_geometry_names = ["currentDist"]
    if not geometry_name in valid_geometry_names:
        print("Invalid geometry name. It must be one of the following:")
        for name in valid_geometry_names:
            print(name)
        sys.exit()

    # Default Detector UIDs
    if valid_detectors is None:
        valid_detectors = [1965, 1966, 1967, 1968]

    t_zylinder = 1         # mm          
    l_voxel = 195      # mm  sqrt((110mm)² * pi) = 194,97
    t_voxel = 1         # mm
    r_pit = 3800        # mm
    dz_pit = 1        # mm 
    r_zyl_bot = 3950    # mm innerer Radius
    r_zyl_top = 1200    # mm
    z_offset = -5000     # mm
    h = 8900

    # if geometry_name == "maxDist":                                        
    #     r_zylinder = 5858  # Innerer Radius # 5858 in Simulation
    #     z_origin = 820 
    #     h_zylinder = h + 20 - z_origin
    #     print("Set maxDist as geometry.")
    if geometry_name == "currentDist":
        r_zylinder = 4300 # Innerer Radius tyvek wand
        z_origin = 20
        h_zylinder = h + 20 - z_origin
        print("Set currentDist as geometry.")

    # Max zylindera Werte für gleichen Index in zylinderen
    r_ref = 6000  # Apothem (innerer Radius) # 5858 in Simulation
    h_ref = 8080 # Müsste eigentlich 20 mm höher sein
    z_ref = 20

    return (t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top, z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref, valid_detectors)

def process_single_file(args):
    (file_path, file_idx, voxel_tree, voxel_data, voxel_indices, 
     materialMappingPath, volumeMappingPath, geometry_params, val_triplets) = args
    
    # Geometrie-Parameter entpacken
    h_zylinder = geometry_params['h_zylinder']
    valid_detectors = geometry_params.get('valid_detectors', [1965, 1966, 1967, 1968])
    
    print(f"Worker {mp.current_process().pid}: Verarbeite {os.path.basename(file_path)}")
    # print_memory_usage("Start Worker")
    
    phi_data_train = []     
    target_data_train = []  
    phi_data_val = []      
    target_data_val = []   
    unassigned_count = 0

    # Anpassbare Chunk-Größe basierend auf verfügbarem Speicher
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    if available_mem_gb > 400:
        CHUNK_SIZE = 50000  # Neu für 500GB Systeme
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
        # Neutron Capture Output Daten lesen
        ##############################################
        nc_evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
        nc_nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
        nC_x = f['hit']['MyNeutronCaptureOutput']['nC_x_position_in_m']['pages'][:]
        nC_y = f['hit']['MyNeutronCaptureOutput']['nC_y_position_in_m']['pages'][:]
        nC_z = f['hit']['MyNeutronCaptureOutput']['nC_z_position_in_m']['pages'][:]
        gamma_amount = f['hit']['MyNeutronCaptureOutput']['nC_gamma_amount']['pages'][:]
        gamma_tot_energy = f['hit']['MyNeutronCaptureOutput']['nC_gamma_total_energy_in_keV']['pages'][:]
        nc_material_ids = f['hit']['MyNeutronCaptureOutput']['nC_material_id']['pages'][:]
        nc_volume_ids = f['hit']['MyNeutronCaptureOutput']['nC_phys_vol_id']['pages'][:]
        nc_time = f['hit']['MyNeutronCaptureOutput']['nC_time_in_ns']['pages'][:]
        
        # Gamma-Daten lesen (1-4)
        gamma_data_nc = {}
        for i in range(1, 5):
            gamma_data_nc[f'gamma{i}_px'] = f['hit']['MyNeutronCaptureOutput'][f'gamma{i}_px']['pages'][:]
            gamma_data_nc[f'gamma{i}_py'] = f['hit']['MyNeutronCaptureOutput'][f'gamma{i}_py']['pages'][:]
            gamma_data_nc[f'gamma{i}_pz'] = f['hit']['MyNeutronCaptureOutput'][f'gamma{i}_pz']['pages'][:]
            gamma_data_nc[f'gamma{i}_E'] = f['hit']['MyNeutronCaptureOutput'][f'gamma{i}_E']['pages'][:]
      
        # Umwandlung von lokalen zu globalen Mappings
        ##############################################
        # Decode material names and build local mapping
        mapping_names = [x.decode() for x in f["hit/materials/materialNames"]["pages"][:]]
        mapping_ids = f["hit/materials/materialsID"]["pages"][:]
        local_mat_map = dict(zip(mapping_ids, mapping_names))
        globalMaterialIDs, _ = remapMaterialIDsToGlobal(materialMappingPath, local_mat_map, nc_material_ids)

        # Umwandlung von lokalen zu globalen Volume-Mappings
        ##############################################
        # Decode volume names and build local mapping
        volume_names = [x.decode() for x in f["hit/physVolumes/physVolumeNames"]["pages"][:]]
        volume_ids = f["hit/physVolumes/physVolumesID"]["pages"][:]
        local_vol_map = dict(zip(volume_ids, volume_names))
        globalVolumeIDs, _ = remapVolumeIDsToGlobal(volumeMappingPath, local_vol_map, nc_volume_ids)

        # Erstelle Dictionary für NC-Daten basierend auf (evtid, nC_id)
        nc_data_dict = {}
        for idx in range(len(nc_evtid)):
            key = (nc_evtid[idx], nc_nC_id[idx])
            nc_data_dict[key] = {
                'nC_x': nC_x[idx],
                'nC_y': nC_y[idx], 
                'nC_z': nC_z[idx],
                'nC_time': nc_time[idx],
                'gamma_amount': gamma_amount[idx],
                'gamma_tot_energy': gamma_tot_energy[idx],
                'material_id': globalMaterialIDs[idx],
                'volume_id': globalVolumeIDs[idx],
                'gamma_energies': [gamma_data_nc[f'gamma{i}_E'][idx] for i in range(1, 5)],
                'gamma_px': [gamma_data_nc[f'gamma{i}_px'][idx] for i in range(1, 5)],
                'gamma_py': [gamma_data_nc[f'gamma{i}_py'][idx] for i in range(1, 5)],
                'gamma_pz': [gamma_data_nc[f'gamma{i}_pz'][idx] for i in range(1, 5)]
            }

        # Speicher für NC-Daten freigeben (außer nc_data_dict)
        del nc_evtid, nc_nC_id, nC_x, nC_y, nC_z, gamma_amount, gamma_tot_energy, nc_material_ids, nc_volume_ids
        del gamma_data_nc, globalMaterialIDs, globalVolumeIDs
        gc.collect()
        
        # print_memory_usage("Nach NC-Dict erstellen")
                
        # Gesamtanzahl optischer Photonen
        total_optical_events = len(f['hit']['optical']['x_position_in_m']['pages'])
        # print(f"Verarbeite {total_optical_events} optische Events in Chunks von {CHUNK_SIZE}")

        # Cut-Parameter
        z_cut_pit = -4979
        z_cut_bot = -4979 # Eigentlich 4179
        z_cut_top = z_cut_bot + h_zylinder - 2 # Fläche ist 1mm dick und startet bei h_zyl - 1

        # Chunk-basierte Verarbeitung der optischen Daten
        num_chunks = (total_optical_events - 1) // CHUNK_SIZE + 1

        # NEW: Initialisiere Voxel-Counter für alle NC-Events
        nc_voxel_counters = {key: Counter() for key in nc_data_dict.keys()}
        orphaned_photons = 0

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, total_optical_events)
            
            # print(f"Verarbeite Chunk {chunk_idx + 1}/{num_chunks}")
            # print_memory_usage(f"Chunk {chunk_idx + 1} Start")
            
            # Nur aktuellen Chunk laden mit optimierten Datentypen
            x_chunk = np.array(f['hit']['optical']['x_position_in_m']['pages'][chunk_start:chunk_end], dtype=np.float32) * 1000
            y_chunk = np.array(f['hit']['optical']['y_position_in_m']['pages'][chunk_start:chunk_end], dtype=np.float32) * 1000
            z_chunk = np.array(f['hit']['optical']['z_position_in_m']['pages'][chunk_start:chunk_end], dtype=np.float32) * 1000
            px_chunk = np.array(f['hit']['optical']['x_momentum_direction']['pages'][chunk_start:chunk_end], dtype=np.float32)
            py_chunk = np.array(f['hit']['optical']['y_momentum_direction']['pages'][chunk_start:chunk_end], dtype=np.float32)
            pz_chunk = np.array(f['hit']['optical']['z_momentum_direction']['pages'][chunk_start:chunk_end], dtype=np.float32)
            
            photon_evtid_chunk = f['hit']['optical']['evtid']['pages'][chunk_start:chunk_end]
            photon_nC_id_chunk = f['hit']['optical']['nC_track_id']['pages'][chunk_start:chunk_end]
            photon_gamma_energies_chunk = f['hit']['optical']['photon_gamma_kinetic_energy_in_keV']['pages'][chunk_start:chunk_end]
            photon_det_uid_chunk = f['hit']['optical']['det_uid']['pages'][chunk_start:chunk_end]
            photon_time_chunk = f['hit']['optical']['time_in_ns']['pages'][chunk_start:chunk_end]

            # print_memory_usage(f"Chunk {chunk_idx + 1} Nach Datenladen")

            # DETECTOR FILTER: Nur Photonen auf den relevanten Detektoren
            valid_detectors_array = np.array(valid_detectors, dtype=np.int32)
            detector_mask = np.isin(photon_det_uid_chunk, valid_detectors_array)

            # Wende Detector-Filter auf alle Arrays an
            x_chunk = x_chunk[detector_mask]
            y_chunk = y_chunk[detector_mask]
            z_chunk = z_chunk[detector_mask]
            px_chunk = px_chunk[detector_mask]
            py_chunk = py_chunk[detector_mask]
            pz_chunk = pz_chunk[detector_mask]
            photon_evtid_chunk = photon_evtid_chunk[detector_mask]
            photon_nC_id_chunk = photon_nC_id_chunk[detector_mask]
            photon_gamma_energies_chunk = photon_gamma_energies_chunk[detector_mask]
            photon_time_chunk = photon_time_chunk[detector_mask]

            # Erstelle Lookup-Array für NC-Zeiten (vektorisiert)
            nc_times = np.full(len(photon_time_chunk), np.inf, dtype=np.float32)  # Default: inf (ungültig)

            # Baue Event-ID zu Index Mapping für schnellen Zugriff
            for idx in range(len(photon_evtid_chunk)):
                key = (photon_evtid_chunk[idx], photon_nC_id_chunk[idx])
                if key in nc_data_dict:
                    nc_times[idx] = nc_data_dict[key]['nC_time']

            # Vektorisierte Zeitfenster-Prüfung
            time_mask = (
                (nc_times != np.inf) &  # NC-Event existiert
                (photon_time_chunk >= nc_times) &  # Photon nach NC
                (photon_time_chunk <= nc_times + 200.0)  # Innerhalb 200ns
            )

            # Logging für orphaned photons
            orphaned_in_chunk = np.sum(nc_times == np.inf)
            if orphaned_in_chunk > 0:
                orphaned_photons += orphaned_in_chunk

            # Wende Zeitfilter auf alle Arrays an
            x_chunk = x_chunk[time_mask]
            y_chunk = y_chunk[time_mask]
            z_chunk = z_chunk[time_mask]
            px_chunk = px_chunk[time_mask]
            py_chunk = py_chunk[time_mask]
            pz_chunk = pz_chunk[time_mask]
            photon_evtid_chunk = photon_evtid_chunk[time_mask]
            photon_nC_id_chunk = photon_nC_id_chunk[time_mask]
            photon_gamma_energies_chunk = photon_gamma_energies_chunk[time_mask]
            photon_time_chunk = photon_time_chunk[time_mask]  # Optional, falls später noch benötigt
        
            # Momentum filtering für diesen Chunk
            mask_bot = (z_chunk <= z_cut_bot)
            mask_top = (z_chunk >= z_cut_top)
            mask_barrel = ~mask_bot & ~mask_top
            
            mask_bot_valid = (pz_chunk[mask_bot] <= 0)    
            mask_top_valid = (pz_chunk[mask_top] >= 0)    
            
            # Nur für barrel region berechnen
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
            
            # Gefilterte Daten
            x_filtered = x_chunk[final_mask]
            y_filtered = y_chunk[final_mask]
            z_filtered = z_chunk[final_mask]
            photon_evtid_filtered = photon_evtid_chunk[final_mask]
            photon_nC_id_filtered = photon_nC_id_chunk[final_mask]
            photon_gamma_energies_filtered = photon_gamma_energies_chunk[final_mask]
            
            # Nicht mehr benötigte Arrays löschen
            del x_chunk, y_chunk, z_chunk, px_chunk, py_chunk, pz_chunk
            del photon_evtid_chunk, photon_nC_id_chunk, photon_gamma_energies_chunk
            del detector_mask, time_mask, photon_time_chunk
            del mask_bot, mask_top, mask_barrel, mask_bot_valid, mask_top_valid, mask_barrel_valid, final_mask
            gc.collect()
            
            # print_memory_usage(f"Chunk {chunk_idx + 1} Nach Filterung")
            
            # Gruppierung der Photonen nach (evtid, nC_id)
            photon_groups = defaultdict(list)
            for idx in range(len(photon_evtid_filtered)):
                photon_groups[(photon_evtid_filtered[idx], photon_nC_id_filtered[idx])].append(idx)
            
            # Akkumuliere nur Voxel-Hits für NC-Events in diesem Chunk
            for (e_id, nc_id), photon_indices in photon_groups.items():
                if (e_id, nc_id) not in nc_data_dict:
                    continue
                
                nc_info = nc_data_dict[(e_id, nc_id)]
                
                for i in photon_indices:
                    x, y, z = x_filtered[i], y_filtered[i], z_filtered[i]
                    voxel_index = assignToNearestVoxel(voxel_tree, voxel_data, (x, y, z))
                    if voxel_index == "-1":
                        unassigned_count += 1
                    else:
                        nc_voxel_counters[(e_id, nc_id)][voxel_index] += 1
            
            # Chunk-Daten explizit löschen
            del x_filtered, y_filtered, z_filtered
            del photon_evtid_filtered, photon_nC_id_filtered, photon_gamma_energies_filtered
            del photon_groups
            
            # print_memory_usage(f"Chunk {chunk_idx + 1} Ende")

    nc_without_photons = 0
    nc_with_photons = 0
    
    for (e_id, nc_id), nc_info in nc_data_dict.items():
        voxel_counter = nc_voxel_counters[(e_id, nc_id)]

        if not voxel_counter:  # Kein Photon detektiert
            nc_without_photons += 1
        else:
            nc_with_photons += 1
        
        # Phi Data erstellen
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
        
        phi_row = [x, y, z, mat_id, vol_id, n_gamma, e_tot] + gamma_row
        target_row = [voxel_counter.get(str(voxel_idx), 0) for voxel_idx in voxel_indices]
        
        if (file_idx, e_id, nc_id) in val_triplets:
            phi_data_val.append(phi_row)
            target_data_val.append(target_row)
        else:
            phi_data_train.append(phi_row)
            target_data_train.append(target_row)
    
    # print_memory_usage("Worker Ende")

    total_nc_events = len(nc_data_dict)
    processed_events = len(phi_data_train) + len(phi_data_val)
    print(f"  NC-Events: {total_nc_events} total, {processed_events} verarbeitet")
    if orphaned_photons > 0:
        print(f"  ℹ INFO: {orphaned_photons} Photonen ohne zugehöriges NC-Event verworfen. Erwartet für Photonen die durch Prozesse vor einem NC Event entstehen.")

    assert total_nc_events == processed_events, (
        f"Nicht alle NC-Events wurden verarbeitet! "
        f"Erwartet: {total_nc_events}, Verarbeitet: {processed_events}"
    )

    print(f"  NC-Events ohne Photonen: {nc_without_photons}")
    print(f"  NC-Events mit Photonen: {nc_with_photons}")

    del nc_voxel_counters, nc_data_dict
    gc.collect()
    
    return {
        'phi_data_train': np.array(phi_data_train, dtype=np.float32) if phi_data_train else np.array([]),
        'target_data_train': np.array(target_data_train, dtype=np.int32) if target_data_train else np.array([]),
        'phi_data_val': np.array(phi_data_val, dtype=np.float32) if phi_data_val else np.array([]),      
        'target_data_val': np.array(target_data_val, dtype=np.int32) if target_data_val else np.array([]),
        'unassigned_count': unassigned_count,
        'orphaned_photons': orphaned_photons,
        'file_processed': os.path.basename(file_path)
    }
    

def append_results_to_hdf5(output_file, result_data, voxel_indices, weight, dataset_type='train'):
    """Fügt Ergebnisse eines einzelnen Files zur HDF5-Datei hinzu"""
    if len(result_data) == 0:
        return 0
    
    phi_columns = ["xNC_mm", "yNC_mm", "zNC_mm", "matID", "volID", "#gamma", "E_gamma_tot_keV", 
                   "gammaE1_keV", "gammapx1", "gammapy1", "gammapz1",
                   "gammaE2_keV", "gammapx2", "gammapy2", "gammapz2",
                   "gammaE3_keV", "gammapx3", "gammapy3", "gammapz3",
                   "gammaE4_keV", "gammapx4", "gammapy4", "gammapz4"]

    phi_data = result_data['phi_data']      
    target_data = result_data['target_data']
    num_entries = len(phi_data)
    
    if num_entries == 0:
        return 0
    
    weights = np.full(num_entries, weight, dtype=np.float32)

    with h5py.File(output_file, 'a') as out:
        phi_grp = out['phi']
        target_grp = out['target']
        
        # Für phi data: entweder erweitern oder neu erstellen
        for i, col_name in enumerate(phi_columns):
            if col_name in phi_grp:
                # Dataset existiert, erweitern
                dset = phi_grp[col_name]
                old_size = dset.shape[0]
                new_size = old_size + num_entries
                
                # Resize und neue Daten schreiben
                dset.resize((new_size,))
                dset[old_size:new_size] = phi_data[:, i]
            else:
                raise RuntimeError(
                    f"Unreachable code reached: Dataset '{col_name}' should already exist "
                    f"or be handled in a previous branch."
                )
        
        # Für target data: FIXED - Use resize for efficiency
        for i, voxel_idx in enumerate(voxel_indices):
            voxel_str = str(voxel_idx)
            
            # Validierung: Dataset MUSS existieren (nach Fix in create_or_open_output_file)
            if voxel_str not in target_grp:
                raise RuntimeError(
                    f"CRITICAL ERROR: Target dataset '{voxel_str}' does not exist!\n"
                    f"This should never happen after create_or_open_output_file() fix.\n"
                    f"File may be corrupted: {output_file}"
                )
            
            # Erweitere existierendes Dataset effizient mit resize()
            dset = target_grp[voxel_str]
            old_size = dset.shape[0]
            new_size = old_size + num_entries
            
            # Resize and write new data
            dset.resize((new_size,))
            dset[old_size:new_size] = target_data[:, i]
        
        # Weights erweitern oder neu erstellen
        if "weights" in out:
            dset = out["weights"]
            old_size = dset.shape[0]
            new_size = old_size + num_entries
            dset.resize((new_size,))
            dset[old_size:new_size] = weights
        else:
            out.create_dataset(
                "weights", 
                data=weights, 
                maxshape=(None,),
                compression='gzip', 
                compression_opts=1,
                chunks=True
            )
        out.flush()
    
    return num_entries

def collect_all_nc_triplets(files): 
    """
    Sammelt alle (file_idx, evtid, nC_id) Triplets aus allen Files.
    Verwendet file_idx um Uniqueness über mehrere Simulationen hinweg zu gewährleisten.
    """
    print("Sammle alle Neutron Capture Events für Train/Val Split...")
    all_nc_triplets = set()
    
    for file_idx, file_path in enumerate(files):
        try:
            with h5py.File(file_path, 'r') as f:
                evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
                nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
                # Triplets mit file_idx für Uniqueness
                triplets = [(file_idx, int(evt), int(nc)) for evt, nc in zip(evtid, nC_id)]
                all_nc_triplets.update(triplets)
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path}: {e}")
            continue
    
    print(f"Gefunden: {len(all_nc_triplets)} einzigartige NC-Events über {len(files)} Files")
    return all_nc_triplets

def calculate_weight_from_files(files, sample_size=None):
    """
    Berechnet Gewichtung basierend auf einzigartigen (file_idx, evtid, nC_id) Triplets.
    Verwendet file_idx um Kollisionen über Files hinweg zu vermeiden.
    """
    print("Berechne Gewichtung...")
    unique_triplets = set()
    
    # Wenn sample_size angegeben, nur Teilmenge verwenden
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
    
    # Hochrechnung wenn nur Sample verwendet wurde
    if sample_size and sample_size < len(files):
        scale_factor = len(files) / sample_size
        estimated_unique_triplets = len(unique_triplets) * scale_factor
        weight = 1 / estimated_unique_triplets if estimated_unique_triplets > 0 else 1.0
        print(f"Geschätztes Gewicht basierend auf {sample_size} Files: {weight:.2e}")
    else:
        weight = 1 / len(unique_triplets) if unique_triplets else 1.0
        print(f"Gewicht basierend auf allen {len(files)} Files: {weight:.2e}")
    
    return weight

def create_or_load_train_val_split(progress_tracker, all_nc_triplets, 
                                   val_fraction, random_seed):
    """
    Erstellt oder lädt Train/Val Split mit Reproduzierbarkeit.
    Verwendet (file_idx, evtid, nC_id) Triplets für Uniqueness über Files hinweg.
    """
    # Versuche bestehenden Split zu laden
    loaded_split = progress_tracker.load_train_val_split()
    
    if loaded_split is not None:
        train_triplets, val_triplets, saved_seed, saved_fraction = loaded_split
        
        # Warne wenn Parameter nicht übereinstimmen
        if saved_seed != random_seed:
            print(f"⚠ WARNUNG: Gespeicherter Random Seed ({saved_seed}) != aktueller Seed ({random_seed})")
            print(f"  Verwende gespeicherten Split für Konsistenz!")
        
        if abs(saved_fraction - val_fraction) > 0.001:
            print(f"⚠ WARNUNG: Gespeicherte Val-Fraction ({saved_fraction}) != aktuelle ({val_fraction})")
            print(f"  Verwende gespeicherten Split für Konsistenz!")
        
        # Validierung: Prüfe ob alle Triplets noch existieren
        all_triplets_set = set(all_nc_triplets)
        missing_train = train_triplets - all_triplets_set
        missing_val = val_triplets - all_triplets_set
        
        if missing_train or missing_val:
            print(f"⚠ WARNUNG: {len(missing_train) + len(missing_val)} NC-Events aus Split fehlen in Daten!")
            print(f"  Erstelle neuen Split...")
            loaded_split = None
        else:
            print(f"✓ Verwende gespeicherten Split: {len(val_triplets)} Validation-Events")
            return val_triplets  # Nur val_triplets wird im Code verwendet
    
    # Erstelle neuen Split
    if loaded_split is None:
        print(f"Erstelle neuen Train/Val Split (Event-Level) mit Seed {random_seed}...")
        
        # Gruppiere nach (file_idx, evtid) um Data Leakage zu vermeiden
        evtid_groups = defaultdict(list)
        for file_idx, evt, nc in all_nc_triplets:
            evtid_groups[(file_idx, evt)].append((file_idx, evt, nc))

        # Split auf Event-Level (innerhalb jedes Files)
        event_keys = sorted(evtid_groups.keys())
        np.random.seed(random_seed)
        np.random.shuffle(event_keys)

        split_idx = int(len(event_keys) * (1 - val_fraction))
        train_event_keys = set(event_keys[:split_idx])

        # Alle NC-Events eines (file_idx, evtid) Paares im gleichen Split
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
        print(f"  Tatsächlicher Val-Anteil: {len(val_triplets)/(len(train_triplets)+len(val_triplets))*100:.1f}%")
        
        # Speichere Split
        progress_tracker.save_train_val_split(train_triplets, val_triplets, 
                                            random_seed, val_fraction)
        
        return val_triplets

def process_files_sequentially(files, voxel_tree, voxel_data, voxel_indices, 
                             material_mapping_path, volume_mapping_path, geometry_params, 
                             output_file_train, output_file_val, weight,
                             progress_tracker, val_triplets):     
    """Verarbeitet Files sequenziell und schreibt nach jedem File"""
    
    remaining_files = progress_tracker.get_remaining_files(files)
    
    if not remaining_files:
        print("Alle Files bereits verarbeitet!")
        return
    
    print(f"Verarbeite {len(remaining_files)} verbleibende Files von insgesamt {len(files)}")
    
    for i, file_path in enumerate(remaining_files):
        print(f"\nVerarbeite File {i+1}/{len(remaining_files)}: {os.path.basename(file_path)}")
        start_time = time.time()

        file_idx = files.index(file_path)        
        try:
            # File verarbeiten
            args = (file_path, file_idx, voxel_tree, voxel_data, voxel_indices, 
                   material_mapping_path, volume_mapping_path, geometry_params, val_triplets) 
            result = process_single_file(args)

            # Log processing params
            available_mem_gb = psutil.virtual_memory().available / (1024**3)
            chunk_size_used = 20000 if available_mem_gb > 50 else (15000 if available_mem_gb > 30 else (10000 if available_mem_gb > 20 else 5000))
            progress_tracker.log_processing_params(chunk_size_used, available_mem_gb)
            
            # [GEÄNDERT] Ergebnisse in beide HDF5-Dateien schreiben
            entries_train = append_results_to_hdf5(
                output_file_train, 
                {'phi_data': result['phi_data_train'], 'target_data': result['target_data_train']},
                voxel_indices, weight, 'train'
            )
            entries_val = append_results_to_hdf5(
                output_file_val,
                {'phi_data': result['phi_data_val'], 'target_data': result['target_data_val']},
                voxel_indices, weight, 'val'
            )
            
            # Fortschritt aktualisieren
            progress_tracker.mark_file_completed(file_path, entries_train, entries_val, result['unassigned_count'])  # [GEÄNDERT] - 2 counts
            
            processing_time = time.time() - start_time
            print(f"✓ File verarbeitet in {processing_time:.2f}s")
            print(f"  Train: {entries_train} Einträge, Val: {entries_val} Einträge")

            if (i + 1) % 100 == 0:
                print(f"\n=== Checkpoint bei File {i+1} ===")
                if not progress_tracker.verify_hdf5_integrity(output_file_val):
                    raise RuntimeError(f"Validation-File korrupt erkannt bei File {i+1}! Stoppe Verarbeitung.")
                progress_tracker.create_checkpoint(output_file_val)
                progress_tracker.create_checkpoint(output_file_train)
            
        except Exception as e:
            error_msg = f"Fehler bei der Verarbeitung: {str(e)}"
            print(f"✗ {error_msg}")
            progress_tracker.mark_file_failed(file_path, error_msg)
            continue
        
        # Zwischenstatistiken
        if (i + 1) % 10 == 0:
            stats = progress_tracker.get_statistics()
            print(f"\nZwischenstand: {stats['completed']}/{len(files)} Files verarbeitet")
            print(f"Laufzeit: {stats['elapsed_time']/60:.1f} min")
            print(f"Einträge Train: {stats['total_entries_train']}, Val: {stats['total_entries_val']}")  # [GEÄNDERT]
            
def process_files_in_batches(files, voxel_tree, voxel_data, voxel_indices, 
                             material_mapping_path, volume_mapping_path,
                             geometry_params, output_file_train, output_file_val, 
                             weight, progress_tracker, val_triplets, batch_size=10):
    """Verarbeitet Files in Batches und schreibt akkumuliert"""
    
    remaining_files = progress_tracker.get_remaining_files(files)
    
    if not remaining_files:
        print("Alle Files bereits verarbeitet!")
        return
    
    print(f"Verarbeite {len(remaining_files)} Files in Batches von {batch_size}")
    
    for batch_start in range(0, len(remaining_files), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_files))
        batch_files = remaining_files[batch_start:batch_end]
        
        print(f"\n=== Batch {batch_start//batch_size + 1}: Files {batch_start+1}-{batch_end} ===")
        
        # Akkumulatoren für Batch
        batch_phi_train = []
        batch_target_train = []
        batch_phi_val = []
        batch_target_val = []
        file_stats = {}
        
        for file_path in batch_files:
            try:
                file_idx = files.index(file_path)  # ← NEU
    
                args = (file_path, file_idx, voxel_tree, voxel_data, voxel_indices,
                    material_mapping_path, volume_mapping_path,
                    geometry_params, val_triplets)
                
                result = process_single_file(args)
                
                # Tracke individuelle Counts
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
                
                if val_count > 0:
                    batch_phi_val.append(result['phi_data_val'])
                    batch_target_val.append(result['target_data_val'])
                
                print(f"  ✓ {os.path.basename(file_path)} (Train: {train_count}, Val: {val_count})")
                                
            except Exception as e:
                print(f"  ✗ {os.path.basename(file_path)}: {e}")
                progress_tracker.mark_file_failed(file_path, str(e))
                continue
        
        # Batch-Write
        if batch_phi_train:
            combined_phi_train = np.vstack(batch_phi_train)
            combined_target_train = np.vstack(batch_target_train)
            entries_train = append_results_to_hdf5(
                output_file_train,
                {'phi_data': combined_phi_train, 'target_data': combined_target_train},
                voxel_indices, weight, 'train'
            )
            print(f"  Batch Train geschrieben: {entries_train} Einträge")
        
        if batch_phi_val:
            combined_phi_val = np.vstack(batch_phi_val)
            combined_target_val = np.vstack(batch_target_val)
            entries_val = append_results_to_hdf5(
                output_file_val,
                {'phi_data': combined_phi_val, 'target_data': combined_target_val},
                voxel_indices, weight, 'val'
            )
            print(f"  Batch Val geschrieben: {entries_val} Einträge")
        
        # Markiere alle Files im Batch als completed
        for file_path in batch_files:
            if str(file_path) not in [f['file'] for f in progress_tracker.progress_data['failed_files']]:
                stats = file_stats.get(file_path, {'train': 0, 'val': 0, 'unassigned': 0})
                progress_tracker.mark_file_completed(
                    file_path,
                    stats['train'],
                    stats['val'],
                    stats['unassigned']
                )
        
        # Checkpoint nach jedem Batch
        progress_tracker.verify_hdf5_integrity(output_file_val)
        
        # Speicher freigeben
        del batch_phi_train, batch_target_train, batch_phi_val, batch_target_val
        gc.collect()


# ----------------------------------------------------------------------------
# Allgemeine Funktionen
# ----------------------------------------------------------------------------
def create_or_open_output_file(output_path, file_index, voxel_data, mat_map, vol_map, radius, suffix=""):
    """Erstellt eine neue HDF5-Datei oder öffnet eine bestehende"""
    output_file = os.path.join(output_path, f"resum_output_{file_index}{suffix}.hdf5") 
    
    # Wenn Datei bereits existiert, prüfen ob sie gültig ist
    if os.path.exists(output_file):
        try:
            with h5py.File(output_file, 'r') as f:
                # Prüfen ob Grundstruktur vorhanden
                if 'phi' in f and 'target' in f and 'voxels' in f:
                    print(f"Bestehende Output-Datei gefunden: {output_file}")
                    return output_file
        except:
            print(f"Bestehende Datei ist korrupt, erstelle neue: {output_file}")
            os.remove(output_file)
    
    # Neue Datei erstellen
    print(f"Erstelle neue Output-Datei: {output_file}")
    with h5py.File(output_file, 'w') as out:
        # Gruppen erstellen
        phi_grp = out.create_group("phi")
        target_grp = out.create_group("target")
        theta_grp = out.create_group("theta")
        mat_map_grp = out.create_group("mat_map")
        vol_map_grp = out.create_group("vol_map")
        voxels_grp = out.create_group("voxels")
        
        # Statische Daten
        theta_grp.create_dataset("inner_radius_in_mm", data=radius)
        out.create_dataset("primaries", data=0)
        
        # Material mapping
        for key, value in mat_map.items():
            if key == "":
                key = "no_material"
            mat_map_grp.create_dataset(str(key), data=int(value))

        # Volume mapping
        for key, value in vol_map.items():
            if key == "":
                key_clean = "noVolume"
            else:
                key_clean = key
            vol_map_grp.create_dataset(str(key_clean), data=int(value))

        # Phi datasets pre-initialisieren (FEHLT AKTUELL!)
        phi_columns = ["xNC_mm", "yNC_mm", "zNC_mm", "matID", "volID", "#gamma", 
                    "E_gamma_tot_keV", "gammaE1_keV", "gammapx1", "gammapy1", "gammapz1",
                    "gammaE2_keV", "gammapx2", "gammapy2", "gammapz2",
                    "gammaE3_keV", "gammapx3", "gammapy3", "gammapz3",
                    "gammaE4_keV", "gammapx4", "gammapy4", "gammapz4"]

        for col_name in phi_columns:
            phi_grp.create_dataset(
                col_name,
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                chunks=True,
                compression='gzip',
                compression_opts=1
            )

        # Weights auch pre-initialisieren
        out.create_dataset(
            "weights",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=True,
            compression='gzip',
            compression_opts=1
        )
        
        # Voxel data UND Target datasets gleichzeitig initialisieren
        print(f"  Initialisiere {len(voxel_data)} Voxel mit Target-Datasets...")        
        for voxel in voxel_data:
            if isinstance(voxel, dict):
                voxel_idx = voxel['index']
                
                # Voxel metadata in /voxels/<voxel_id>/
                voxel_grp = voxels_grp.create_group(str(voxel_idx))
                voxel_grp.create_dataset("center", data=np.array(voxel['center'], dtype='f'))
                dt = h5py.string_dtype(encoding='utf-8')
                voxel_grp.create_dataset("layer", data=voxel['layer'], dtype=dt)
                
                corners = np.array(voxel['corners'])
                corners_grp = voxel_grp.create_group("corners")
                corners_grp.create_dataset("x", data=corners[:, 0])
                corners_grp.create_dataset("y", data=corners[:, 1])
                corners_grp.create_dataset("z", data=corners[:, 2])
                
                # CRITICAL FIX: Create EMPTY resizable target dataset
                target_grp.create_dataset(
                    str(voxel_idx),
                    shape=(0,),              # Start with 0 entries
                    maxshape=(None,),        # Allow unlimited growth
                    dtype=np.int32,
                    chunks=True,             # Required for resize
                    compression='gzip',
                    compression_opts=1
                )
        
        print(f"  ✓ Alle {len(voxel_data)} Voxel-Datasets erstellt")
    
    return output_file

def parse_arguments():
    """Parst Kommandozeilen-Argumente"""
    parser = argparse.ArgumentParser(
        description='Post-Processing Script für optische Photonen mit Train/Val Split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Das Script erstellt automatisch zwei Output-Files:                            
- resum_output_0_train.hdf5 (80% der Daten)                                   
- resum_output_0_validation.hdf5 (20% der Daten)                              

Beispiele:
  # Daten direkt im Ordner                                                    
  python %(prog)s -i /path/to/simulation_data -o /path/to/output
  
  # Daten in mehreren Unterordnern                                            
  python %(prog)s -i /path/to/simulations -o /path/to/output --nested         
        """
    )
    
    # Verpflichtende Argumente
    parser.add_argument('-i', '--input', required=True,
                        help='Eingabe-Pfad mit HDF5-Dateien (verpflichtend)')
    parser.add_argument('-o', '--output', required=True,
                        help='Ausgabe-Pfad für Ergebnisse (verpflichtend)')
    
    # Optionale Argumente mit Defaults
    parser.add_argument('-m', '--material-mapping',
                        default='/global/cfs/projectdirs/legend/users/tbuerger/postprocessing/src/mappings/globalMaterialMappings.json',
                        help='Pfad zur Material-Mapping-Datei')
    parser.add_argument('-v', '--volume-mapping',
                        default='/global/cfs/projectdirs/legend/users/tbuerger/postprocessing/src/mappings/globalPhysVolMappings.json',
                        help='Pfad zur Volume-Mapping-Datei')
    parser.add_argument('-x', '--voxel-file',
                        default='/global/cfs/projectdirs/legend/users/tbuerger/postprocessing/src/voxels/currentDistZylVoxelsPMTSize.json',
                        help='Pfad zur Voxel-Datei')
    
    # Resume und Processing-Optionen
    parser.add_argument('--resume', action='store_true',
                        help='Bereits verarbeitete Files überspringen (automatisch aktiviert)')
    parser.add_argument('--reset', action='store_true',
                        help='Verarbeitung von vorne beginnen (überschreibt bestehende Ergebnisse)')
    parser.add_argument('--sample-weight', type=int, default=50,
                        help='Anzahl Files für Gewichtungsberechnung (0 = alle Files verwenden)')
    parser.add_argument('--nested', action='store_true',
                        help='Suche nach HDF5-Dateien in Unterordnern (z.B. run_001, run_002, ...)')
    
    # Train/Val Split Parameter
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='Anteil der Daten für Validation (default: 0.2 = 20%%)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random Seed für reproduzierbaren Train/Val Split (default: 42)')
    
    # Zusätzliche Optionen
    parser.add_argument('-g', '--geometry', default='currentDist',
                        choices=['currentDist'],
                        help='Geometrie-Name (default: currentDist)')
    parser.add_argument('--valid-detectors', type=int, nargs='+', 
                    default=[1965, 1966, 1967, 1968],
                    help='Liste der gültigen Detector UIDs (default: 1965 1966 1967 1968)')
    parser.add_argument('--verbose', '-V', action='store_true',
                        help='Ausführliche Ausgabe')
    
    return parser.parse_args()

def main():
    # Argumente parsen
    args = parse_arguments()
    print_memory_usage("Hauptprogramm Start")

    # Pfade validieren
    if not os.path.exists(args.input):
        print(f"Fehler: Eingabe-Pfad existiert nicht: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.voxel_file):
        print(f"Fehler: Voxel-Datei existiert nicht: {args.voxel_file}")
        sys.exit(1)
    
    # Output-Verzeichnis erstellen
    os.makedirs(args.output, exist_ok=True)
    
    # Konfiguration
    geometry_name = args.geometry
    geometry_result = defineZylinder(geometry_name, args.valid_detectors)
    (t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top, 
     z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref, valid_detectors) = geometry_result

    # Voxel-Daten laden
    print("Lade Voxel-Daten...")
    with open(args.voxel_file, 'r') as f:
        voxel_data = json.load(f)
    
    voxel_tree = build_voxel_tree(voxel_data)
    voxel_indices = [voxel['index'] for voxel in voxel_data]
    print(f"Voxel-Tree mit {len(voxel_data)} Voxeln erstellt")
    
    # Dateien finden
    files = find_all_hdf5_files(args.input, args.nested)
    
    if not files:
        print(f"Fehler: Keine output_*.hdf5 Dateien gefunden!")
        sys.exit(1)
    print(f"Gefunden: {len(files)} Dateien insgesamt")

    # Output-Datei definieren
    output_file_train = os.path.join(args.output, "resum_output_0_train.hdf5")
    output_file_val = os.path.join(args.output, "resum_output_0_validation.hdf5")
    
    # Progress Tracker initialisieren
    progress_tracker = ProgressTracker(args.output, output_file_train, output_file_val)
    
    # Reset-Option prüfen
    if args.reset:
        print("Reset angefordert - lösche bestehende Ergebnisse...")
        if os.path.exists(output_file_train): 
            os.remove(output_file_train)
        if os.path.exists(output_file_val):    
            os.remove(output_file_val)
        progress_tracker.cleanup()
        if progress_tracker.split_file.exists():
            progress_tracker.split_file.unlink()        
        progress_tracker = ProgressTracker(args.output, output_file_train, output_file_val)

    # Train/Val Split erstellen
    all_nc_triplets = collect_all_nc_triplets(files)
    val_triplets = create_or_load_train_val_split(
        progress_tracker, all_nc_triplets, args.val_fraction, args.random_seed
    )
    
    # Alle Materialien sammeln (nur bei Reset oder wenn noch keine Output-Datei existiert)
    if not os.path.exists(output_file_train):
        all_materials = collect_all_materials_first(files)
        all_volumes = collect_all_volumes_first(files)
        
        # Vollständiges globales Mapping erstellen
        global_material_mapping = create_global_material_mapping(all_materials, args.material_mapping)
        print(f"Globales Material-Mapping erstellt mit {len(global_material_mapping)} Materialien")

        global_volume_mapping = create_global_volume_mapping(all_volumes, args.volume_mapping)
        print(f"Globales Volume-Mapping erstellt mit {len(global_volume_mapping)} Volumen")
    else:
        # Bestehendes Mapping laden
        global_material_mapping = loadMapping(args.material_mapping)
        print(f"Bestehendes Material-Mapping geladen: {len(global_material_mapping)} Materialien")

        global_volume_mapping = loadMapping(args.volume_mapping)
        print(f"Bestehendes Volume-Mapping geladen: {len(global_volume_mapping)} Volumen")
    
    # Gewichtung berechnen
    sample_size = args.sample_weight if args.sample_weight > 0 else None
    weight = calculate_weight_from_files(files, sample_size)
    
    # Output-Datei erstellen oder öffnen
    output_file_train = create_or_open_output_file(args.output, 0, voxel_data, global_material_mapping, global_volume_mapping, r_zylinder, "_train")
    output_file_val = create_or_open_output_file(args.output, 0, voxel_data, global_material_mapping, global_volume_mapping, r_zylinder, "_validation")
    
    # Geometrie-Parameter für Worker
    geometry_params = {
        'h_zylinder': h_zylinder,
        'valid_detectors': valid_detectors
    }
    
    # Aktuelle Statistiken anzeigen
    stats = progress_tracker.get_statistics()
    if stats['completed'] > 0:
        print(f"\nBestehender Fortschritt:")
        print(f"  Bereits verarbeitet: {stats['completed']}/{len(files)} Files")
        print(f"  Fehlgeschlagen: {stats['failed']} Files")
        print(f"  Einträge Train: {stats['total_entries_train']}")   
        print(f"  Einträge Val: {stats['total_entries_val']}") 
        print(f"  Bisherige Laufzeit: {stats['elapsed_time']/60:.1f} min")
    
    # Sequenzielle Verarbeitung
    print(f"\nStarte sequenzielle Verarbeitung...")
    start_time = time.time()
    
    try:
        process_files_in_batches(
            files, voxel_tree, voxel_data, voxel_indices,
            args.material_mapping, args.volume_mapping,
            geometry_params, output_file_train, output_file_val, 
            weight, progress_tracker, val_triplets, batch_size=20
        )
    except KeyboardInterrupt:
        print(f"\nVerarbeitung durch Benutzer unterbrochen.")
        print(f"Fortschritt wurde gespeichert. Führen Sie das Script erneut aus, um fortzufahren.")
        return
    except Exception as e:
        print(f"Kritischer Fehler: {e}")
        return
    
    # Finale Statistiken
    final_stats = progress_tracker.get_statistics()
    total_time = time.time() - start_time
    
    print(f"\n" + "="*60)
    print(f"VERARBEITUNG ABGESCHLOSSEN NACH: ", total_time)
    print(f"="*60)
    print(f"Erfolgreich verarbeitete Files: {final_stats['completed']}/{len(files)}")
    print(f"Fehlgeschlagene Files: {final_stats['failed']}")
    print(f"Training Einträge: {final_stats['total_entries_train']}")    
    print(f"Validation Einträge: {final_stats['total_entries_val']}")    
    print(f"Validation Anteil: {final_stats['total_entries_val']/(final_stats['total_entries_train']+final_stats['total_entries_val'])*100:.1f}%") 
    print(f"Nicht zugeordnete Punkte: {final_stats['total_unassigned']}")
    print(f"Gesamtlaufzeit: {final_stats['elapsed_time']/60:.1f} min")
    print(f"Durchschnitt pro File: {final_stats['elapsed_time']/max(1,final_stats['completed']):.2f}s")
    print(f"Output-Dateien:")                                            
    print(f"  Training: {output_file_train}")                            
    print(f"  Validation: {output_file_val}")
    
    # Aufräumen nach erfolgreichem Abschluss
    if final_stats['completed'] == len(files) and final_stats['failed'] == 0:
        print(f"\nAlle Files erfolgreich verarbeitet - räume temporäre Dateien auf...")
        progress_tracker.cleanup()
    else:
        print(f"\nFortschritt wurde gespeichert für eventuelle Wiederaufnahme.")
    
    print_memory_usage("Programm Ende")

if __name__ == "__main__":
    main()