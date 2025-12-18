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
            'output_file': str(self.output_file)
        }
    
    def save_progress(self):
        """Speichert aktuellen Fortschritt"""
        self.progress_data['last_update'] = time.time()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except IOError as e:
            print(f"Warnung: Konnte Fortschritt nicht speichern: {e}")
    
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
    
    def cleanup(self):
        """Räumt temporäre Dateien auf (nach erfolgreichem Abschluss)"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.lock_file.exists():
                self.lock_file.unlink()
        except OSError:
            pass

def loadMapping(filePath):
    with open(filePath, 'r') as f:
        mapping = json.load(f)
    return mapping

def remapMaterialIDsToGlobal(
    glob_mapping_json_path: str,
    local_mat_map: dict,
    local_material_ids
) -> tuple[np.ndarray, dict]:
    """Vectorized material ID remapping using dictionary lookups"""
    if os.path.exists(glob_mapping_json_path):
        globMaterialMapping = loadMapping(glob_mapping_json_path)
    else:
        globMaterialMapping = {}

    local_material_ids = np.array(local_material_ids)
    unique_local_ids = np.unique(local_material_ids)
    mapping_dict = {}
    max_global_id = max(globMaterialMapping.values()) if globMaterialMapping else -1

    # Build mapping dictionary for unique IDs
    for local_id in unique_local_ids:
        local_name = local_mat_map[local_id]
        if local_name in globMaterialMapping:
            mapping_dict[local_id] = globMaterialMapping[local_name]
        else:
            max_global_id += 1
            globMaterialMapping[local_name] = max_global_id
            mapping_dict[local_id] = max_global_id

    # Vectorize the mapping process
    vectorized_mapper = np.vectorize(mapping_dict.get)
    global_material_ids = vectorized_mapper(local_material_ids)

    return global_material_ids, globMaterialMapping

def remapVolumeIDsToGlobal(
    glob_mapping_json_path: str,
    local_vol_map: dict,
    local_volume_ids
) -> tuple[np.ndarray, dict]:
    """Vectorized volume ID remapping using dictionary lookups"""
    if os.path.exists(glob_mapping_json_path):
        globVolumeMapping = loadMapping(glob_mapping_json_path)
    else:
        globVolumeMapping = {}

    local_volume_ids = np.array(local_volume_ids)
    unique_local_ids = np.unique(local_volume_ids)
    mapping_dict = {}
    max_global_id = max(globVolumeMapping.values()) if globVolumeMapping else -1

    # Build mapping dictionary for unique IDs
    for local_id in unique_local_ids:
        local_name = local_vol_map[local_id]
        if local_name in globVolumeMapping:
            mapping_dict[local_id] = globVolumeMapping[local_name]
        else:
            max_global_id += 1
            globVolumeMapping[local_name] = max_global_id
            mapping_dict[local_id] = max_global_id

    # Vectorize the mapping process
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
    """Sammelt alle physikalischen Volumes aus allen Files VOR der parallelen Verarbeitung"""
    print("Sammle alle physikalischen Volumes aus allen Files...")
    all_volumes = set()
    
    for file_path in files:            
        try:
            with h5py.File(file_path, 'r') as f:
                volume_names = [x.decode() for x in f["hit/physVolumes/physVolumeNames"]["pages"][:]]
                all_volumes.update(volume_names)
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path}: {e}")
            continue
    
    print(f"Gefunden: {len(all_volumes)} einzigartige physikalische Volumes")
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
    
def defineZylinder(geometry_name):
    valid_geometry_names = ["currentDist"]
    if not geometry_name in valid_geometry_names:
        print("Invalid geometry name. It must be one of the following:")
        for name in valid_geometry_names:
            print(name)
        sys.exit()

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

    return t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top, z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref

def process_single_file(args):
    """Verarbeitet eine einzelne HDF5-Datei"""
    (file_path, voxel_tree, voxel_data, voxel_indices, 
     materialMappingPath, volumeMappingPath, geometry_params, validation_set) = args
    
    # Geometrie-Parameter entpacken
    h_zylinder = geometry_params['h_zylinder']
    
    print(f"Worker {mp.current_process().pid}: Verarbeite {os.path.basename(file_path)}")
    # print_memory_usage("Start Worker")
    
    phi_data_train = []     
    target_data_train = []  
    phi_data_val = []      
    target_data_val = []   
    unassigned_count = 0

    # Anpassbare Chunk-Größe basierend auf verfügbarem Speicher
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    if available_mem_gb > 50:
        CHUNK_SIZE = 20000
    elif available_mem_gb > 30:
        CHUNK_SIZE = 15000
    elif available_mem_gb > 20:
        CHUNK_SIZE = 10000
    else:
        CHUNK_SIZE = 5000
    
    # print(f"Verwende CHUNK_SIZE: {CHUNK_SIZE} (verfügbar: {available_mem_gb:.1f}GB)")

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
            
            # print_memory_usage(f"Chunk {chunk_idx + 1} Nach Datenladen")
        
            # Momentum filtering für diesen Chunk
            mask_bot = (z_chunk <= z_cut_bot)
            mask_top = (z_chunk >= z_cut_top)
            mask_barrel = ~mask_bot & ~mask_top
            
            mask_bot_valid = (pz_chunk <= 0)
            mask_top_valid = (pz_chunk >= 0)
            
            # Nur für barrel region berechnen
            mask_barrel_valid = np.zeros(np.sum(mask_barrel), dtype=bool)
            if np.any(mask_barrel):
                mask_barrel_valid = checkRadialMomentumVectorized(
                    x_chunk[mask_barrel], y_chunk[mask_barrel], z_chunk[mask_barrel],
                    px_chunk[mask_barrel], py_chunk[mask_barrel], pz_chunk[mask_barrel]
                )
            
            # Combine masks
            final_mask = np.zeros_like(z_chunk, dtype=bool)
            final_mask[mask_bot] = mask_bot_valid[mask_bot]
            final_mask[mask_top] = mask_top_valid[mask_top]
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
            del mask_bot, mask_top, mask_barrel, mask_bot_valid, mask_top_valid, mask_barrel_valid, final_mask
            gc.collect()
            
            # print_memory_usage(f"Chunk {chunk_idx + 1} Nach Filterung")
            
            # Gruppierung der Photonen nach (evtid, nC_id)
            photon_groups = defaultdict(list)
            for idx in range(len(photon_evtid_filtered)):
                photon_groups[(photon_evtid_filtered[idx], photon_nC_id_filtered[idx])].append(idx)
            
            # Verarbeitung jeder Gruppe
            for (e_id, nc_id), photon_indices in photon_groups.items():
                if (e_id, nc_id) not in nc_data_dict:
                    continue
                    
                nc_info = nc_data_dict[(e_id, nc_id)]
                
                voxel_counter = Counter()
                gamma_number_list = []
                
                for i in photon_indices:
                    x, y, z = x_filtered[i], y_filtered[i], z_filtered[i]
                    voxel_index = assignToNearestVoxel(voxel_tree, voxel_data, (x, y, z))
                    if voxel_index == "-1":
                        unassigned_count += 1
                    else:
                        voxel_counter[voxel_index] += 1
                    
                    photon_gamma_energy = photon_gamma_energies_filtered[i]
                    gamma_number = 0
                    
                    for gamma_idx, gamma_energy in enumerate(nc_info['gamma_energies']):
                        if abs(photon_gamma_energy - gamma_energy) < 0.1:
                            gamma_number = gamma_idx + 1
                            break
                    
                    gamma_number_list.append(gamma_number)
                
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

                if (e_id, nc_id) in validation_set:     
                    phi_data_val.append(phi_row)        
                    target_data_val.append(target_row)  
                else:                                   
                    phi_data_train.append(phi_row)      
                    target_data_train.append(target_row) 
            
            # Chunk-Daten explizit löschen
            del x_filtered, y_filtered, z_filtered
            del photon_evtid_filtered, photon_nC_id_filtered, photon_gamma_energies_filtered
            del photon_groups
            gc.collect()
            
            # print_memory_usage(f"Chunk {chunk_idx + 1} Ende")
    
    # print_memory_usage("Worker Ende")
    
    return {
        'phi_data_train': np.array(phi_data_train, dtype=np.float32) if phi_data_train else np.array([]),
        'target_data_train': np.array(target_data_train, dtype=np.int32) if target_data_train else np.array([]),
        'phi_data_val': np.array(phi_data_val, dtype=np.float32) if phi_data_val else np.array([]),      
        'target_data_val': np.array(target_data_val, dtype=np.int32) if target_data_val else np.array([]),
        'unassigned_count': unassigned_count,
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
                existing_data = phi_grp[col_name][:]
                new_data = np.concatenate([existing_data, phi_data[:, i]])
                del phi_grp[col_name]
                phi_grp.create_dataset(col_name, data=new_data)
            else:
                # Neues Dataset erstellen
                phi_grp.create_dataset(col_name, data=phi_data[:, i])
        
        # Für target data: entweder erweitern oder neu erstellen
        for i, voxel_idx in enumerate(voxel_indices):
            voxel_str = str(voxel_idx)
            if voxel_str in target_grp:
                # Dataset existiert, erweitern
                existing_data = target_grp[voxel_str][:]
                new_data = np.concatenate([existing_data, target_data[:, i]])
                del target_grp[voxel_str]
                target_grp.create_dataset(voxel_str, data=new_data)
            else:
                # Neues Dataset erstellen
                target_grp.create_dataset(voxel_str, data=target_data[:, i])
        
        # Weights erweitern oder neu erstellen
        if "weights" in out:
            existing_weights = out["weights"][:]
            new_weights = np.concatenate([existing_weights, weights])
            del out["weights"]
            out.create_dataset("weights", data=new_weights)
        else:
            out.create_dataset("weights", data=weights)
    
    return num_entries

def collect_all_nc_pairs(files): 
    """Sammelt alle (evtid, nC_id) Paare aus allen Files"""
    print("Sammle alle Neutron Capture Events für Train/Val Split...")
    all_nc_pairs = set()
    
    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
                nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
                pairs = zip(evtid, nC_id)
                all_nc_pairs.update(pairs)
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path}: {e}")
            continue
    
    print(f"Gefunden: {len(all_nc_pairs)} einzigartige NC-Events")
    return all_nc_pairs

def calculate_weight_from_files(files, sample_size=None):
    """Berechnet Gewichtung basierend auf einzigartigen (evtid, nC_id) Paaren"""
    print("Berechne Gewichtung...")
    unique_pairs = set()
    
    # Wenn sample_size angegeben, nur Teilmenge verwenden
    files_to_sample = files[:sample_size] if sample_size else files
    
    for file in files_to_sample:
        try:
            with h5py.File(file, 'r') as f:
                evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
                nC_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
                pairs = zip(evtid, nC_id)
                unique_pairs.update(pairs)
        except Exception as e:
            print(f"Warnung: Konnte {file} für Gewichtung nicht lesen: {e}")
            continue
    
    # Hochrechnung wenn nur Sample verwendet wurde
    if sample_size and sample_size < len(files):
        scale_factor = len(files) / sample_size
        estimated_unique_pairs = len(unique_pairs) * scale_factor
        weight = 1 / estimated_unique_pairs if estimated_unique_pairs > 0 else 1.0
        print(f"Geschätztes Gewicht basierend auf {sample_size} Files: {weight}")
    else:
        weight = 1 / len(unique_pairs) if unique_pairs else 1.0
        print(f"Gewicht basierend auf allen Files: {weight}")
    
    return weight

def create_train_val_split(all_nc_pairs, val_fraction=0.2, random_seed=42): 
    """Erstellt zufälliges Train/Val Split der NC-Events"""
    np.random.seed(random_seed)
    all_pairs_list = list(all_nc_pairs)
    np.random.shuffle(all_pairs_list)
    
    split_idx = int(len(all_pairs_list) * (1 - val_fraction))
    train_pairs = set(all_pairs_list[:split_idx])
    val_pairs = set(all_pairs_list[split_idx:])
    
    print(f"Train/Val Split erstellt:")
    print(f"  Training: {len(train_pairs)} NC-Events ({(1-val_fraction)*100:.0f}%)")
    print(f"  Validation: {len(val_pairs)} NC-Events ({val_fraction*100:.0f}%)")
    
    return train_pairs, val_pairs

def process_files_sequentially(files, voxel_tree, voxel_data, voxel_indices, 
                             material_mapping_path, volume_mapping_path, geometry_params, 
                             output_file_train, output_file_val, weight,
                             progress_tracker, validation_set):     
    """Verarbeitet Files sequenziell und schreibt nach jedem File"""
    
    remaining_files = progress_tracker.get_remaining_files(files)
    
    if not remaining_files:
        print("Alle Files bereits verarbeitet!")
        return
    
    print(f"Verarbeite {len(remaining_files)} verbleibende Files von insgesamt {len(files)}")
    
    for i, file_path in enumerate(remaining_files):
        print(f"\nVerarbeite File {i+1}/{len(remaining_files)}: {os.path.basename(file_path)}")
        start_time = time.time()
        
        try:
            # File verarbeiten
            args = (file_path, voxel_tree, voxel_data, voxel_indices, 
                   material_mapping_path, volume_mapping_path, geometry_params, validation_set)  # [GEÄNDERT] - validation_set hinzugefügt
            result = process_single_file(args)
            
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
            print(f"  Train: {entries_train} Einträge, Val: {entries_val} Einträge")  # [GEÄNDERT] - beide anzeigen
            
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
            
class ProgressTracker:
    """Klasse zum Verwalten des Verarbeitungsfortschritts"""
    
    def __init__(self, output_dir, output_file_train, output_file_val):  # [GEÄNDERT] - 2 output files statt 1
        self.output_dir = Path(output_dir)
        self.output_file_train = Path(output_file_train)  # [NEU]
        self.output_file_val = Path(output_file_val)      # [NEU]
        self.progress_file = self.output_dir / "processing_progress.json"
        self.lock_file = self.output_dir / "processing.lock"
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
            'total_entries_written_train': 0,   # [GEÄNDERT] - war total_entries_written
            'total_entries_written_val': 0,     # [NEU]
            'total_unassigned': 0,
            'output_file_train': str(self.output_file_train),  # [GEÄNDERT] - war output_file
            'output_file_val': str(self.output_file_val)       # [NEU]
        }
    
    def save_progress(self):
        """Speichert aktuellen Fortschritt"""
        self.progress_data['last_update'] = time.time()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except IOError as e:
            print(f"Warnung: Konnte Fortschritt nicht speichern: {e}")
    
    def is_file_completed(self, file_path):
        """Prüft ob eine Datei bereits verarbeitet wurde"""
        return str(file_path) in self.progress_data['completed_files']
    
    def mark_file_completed(self, file_path, entries_count_train, entries_count_val, unassigned_count):  # [GEÄNDERT] - 2 counts statt 1
        """Markiert eine Datei als erfolgreich verarbeitet"""
        file_str = str(file_path)
        if file_str not in self.progress_data['completed_files']:
            self.progress_data['completed_files'].append(file_str)
            self.progress_data['total_entries_written_train'] += entries_count_train  # [GEÄNDERT]
            self.progress_data['total_entries_written_val'] += entries_count_val      # [NEU]
            self.progress_data['total_unassigned'] += unassigned_count
            self.save_progress()
    
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
            'total_entries_train': self.progress_data['total_entries_written_train'],  # [GEÄNDERT]
            'total_entries_val': self.progress_data['total_entries_written_val'],      # [NEU]
            'total_unassigned': self.progress_data['total_unassigned'],
            'elapsed_time': time.time() - self.progress_data['start_time']
        }
    
    def cleanup(self):
        """Räumt temporäre Dateien auf (nach erfolgreichem Abschluss)"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.lock_file.exists():
                self.lock_file.unlink()
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Allgemeine Funktionen
# ----------------------------------------------------------------------------
def create_or_open_output_file(output_path, file_index, voxel_indices, mat_map, vol_map, radius, suffix=""):
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
                key = "no_volume"
            vol_map_grp.create_dataset(str(key), data=int(value))
        
        # Voxel data schreiben
        for voxel in voxel_indices:  # voxel_indices ist hier voxel_data
            if isinstance(voxel, dict):
                voxel_grp = voxels_grp.create_group(str(voxel['index']))
                voxel_grp.create_dataset("center", data=np.array(voxel['center'], dtype='f'))
                dt = h5py.string_dtype(encoding='utf-8')
                voxel_grp.create_dataset("layer", data=voxel['layer'], dtype=dt)
                
                corners = np.array(voxel['corners'])
                corners_grp = voxel_grp.create_group("corners")
                corners_grp.create_dataset("x", data=corners[:, 0])
                corners_grp.create_dataset("y", data=corners[:, 1])
                corners_grp.create_dataset("z", data=corners[:, 2])
    
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
                        default='/global/cfs/projectdirs/legend/users/tbuerger/createSSD/src/voxels/currentDistZylVoxelsPMTSize.json',
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
    t_zylinder, l_voxel, t_voxel, r_pit, dz_pit, r_zyl_bot, r_zyl_top, z_offset, r_zylinder, h_zylinder, z_origin, r_ref, h_ref, z_ref = defineZylinder(geometry_name)

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
        if os.path.exists(output_file_train):  # [GEÄNDERT]
            os.remove(output_file_train)
        if os.path.exists(output_file_val):    # [NEU]
            os.remove(output_file_val)
        progress_tracker.cleanup()
        progress_tracker = ProgressTracker(args.output, output_file_train, output_file_val)

    # Train/Val Split erstellen
    all_nc_pairs = collect_all_nc_pairs(files)
    train_pairs, val_pairs = create_train_val_split(all_nc_pairs, args.val_fraction, args.random_seed)
    
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
    geometry_params = {'h_zylinder': h_zylinder}
    
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
        process_files_sequentially(
            files, voxel_tree, voxel_data, voxel_indices,
            args.material_mapping, args.volume_mapping, geometry_params,
            output_file_train, output_file_val, weight, 
            progress_tracker, val_pairs  
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
    print(f"VERARBEITUNG ABGESCHLOSSEN!")
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