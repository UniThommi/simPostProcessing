import h5py
import numpy as np
import json
from typing import Dict, Any, Union
from pathlib import Path

def analyze_hdf5_structure(
    filepath: Union[str, Path],
    chunk_size: int = 100_000,
    output_json: Union[str, Path] = "analysis_results.json"
) -> Dict[str, Any]:
    """
    Analysiert HDF5-Datei mit Min/Max-Werten für phi-Features, 
    Target-Regionen und individuelle Targets.
    
    Parameters
    ----------
    filepath : str | Path
        Pfad zur HDF5-Datei
    chunk_size : int, default=100_000
        Chunk-Größe für Memory-effizientes Streaming
    output_json : str | Path, default="analysis_results.json"
        Ausgabepfad für JSON-Ergebnisse
        
    Returns
    -------
    Dict[str, Any]
        Strukturiertes Dictionary mit allen Analysen
    """
    
    results = {
        "phi_features": {},
        "target_regions": {},
        "target_regions_sum": {},
        "individual_targets": {}
    }
    
    with h5py.File(filepath, 'r') as f:
        n_events = f['phi']['#gamma'].shape[0]
        print(f"Analysiere {n_events:,} Events...")
        
        # === 1. PHI FEATURES (außer matID, volID) ===
        print("\n[1/3] Analysiere phi-Features...")
        phi_group = f['phi']
        excluded = {'matID', 'volID'}
        phi_keys = [k for k in phi_group.keys() if k not in excluded]
        
        # Initialisiere mit +/- inf
        phi_stats = {key: {'min': np.inf, 'max': -np.inf} for key in phi_keys}
        
        for start_idx in range(0, n_events, chunk_size):
            end_idx = min(start_idx + chunk_size, n_events)
            
            for key in phi_keys:
                chunk = phi_group[key][start_idx:end_idx]
                phi_stats[key]['min'] = min(phi_stats[key]['min'], float(np.min(chunk)))
                phi_stats[key]['max'] = max(phi_stats[key]['max'], float(np.max(chunk)))
            
            if (start_idx // chunk_size + 1) % 5 == 0:
                print(f"  Chunk {start_idx//chunk_size + 1}/{(n_events-1)//chunk_size + 1}")
        
        results['phi_features'] = phi_stats
        
        # === 2. TARGET REGIONS (bot, pit, top, wall) ===
        print("\n[2/3] Analysiere target_regions...")
        region_keys = ['bot', 'pit', 'top', 'wall']
        region_stats = {key: {'min': np.inf, 'max': -np.inf} for key in region_keys}
        sum_stats = {'min': np.inf, 'max': -np.inf}
        
        for start_idx in range(0, n_events, chunk_size):
            end_idx = min(start_idx + chunk_size, n_events)
            
            # Lade alle 4 Regionen für diesen Chunk
            region_chunks = {
                key: f['target_regions'][key][start_idx:end_idx] 
                for key in region_keys
            }
            
            # Individuelle Min/Max pro Region
            for key in region_keys:
                chunk = region_chunks[key]
                region_stats[key]['min'] = min(region_stats[key]['min'], int(np.min(chunk)))
                region_stats[key]['max'] = max(region_stats[key]['max'], int(np.max(chunk)))
            
            # Summe über alle 4 Regionen pro Event
            event_sums = sum(region_chunks.values())
            sum_stats['min'] = min(sum_stats['min'], int(np.min(event_sums)))
            sum_stats['max'] = max(sum_stats['max'], int(np.max(event_sums)))
            
            if (start_idx // chunk_size + 1) % 5 == 0:
                print(f"  Chunk {start_idx//chunk_size + 1}/{(n_events-1)//chunk_size + 1}")
        
        results['target_regions'] = region_stats
        results['target_regions_sum'] = sum_stats
        
        # === 3. INDIVIDUAL TARGETS (Voxel-weise) ===
        print("\n[3/3] Analysiere individual targets (Voxel)...")
        target_group = f['target']
        target_keys = list(target_group.keys())
        n_targets = len(target_keys)
        print(f"  Gefunden: {n_targets} Targets/Voxel")
        
        # Globale Min/Max über alle Voxel
        global_target_min = np.inf
        global_target_max = -np.inf
        
        for target_idx, target_key in enumerate(target_keys):
            target_dataset = target_group[target_key]
            
            # Chunked processing pro Target
            for start_idx in range(0, n_events, chunk_size):
                end_idx = min(start_idx + chunk_size, n_events)
                chunk = target_dataset[start_idx:end_idx]
                
                global_target_min = min(global_target_min, int(np.min(chunk)))
                global_target_max = max(global_target_max, int(np.max(chunk)))
            
            # Progress alle 500 Targets
            if (target_idx + 1) % 500 == 0:
                print(f"  Verarbeitet: {target_idx + 1}/{n_targets} Targets")
        
        results['individual_targets'] = {
            'min_over_all_voxels': int(global_target_min),
            'max_over_all_voxels': int(global_target_max),
            'n_voxels': n_targets
        }
    
    # === EXPORT TO JSON ===
    print(f"\n✓ Analyse abgeschlossen. Exportiere nach {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON gespeichert: {output_json}")
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """
    Druckt zusammengefasste Ergebnisse in die Konsole.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Ergebnis-Dictionary von analyze_hdf5_structure()
    """
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG DER ANALYSE")
    print("="*60)
    
    print("\n[PHI FEATURES]")
    for key, stats in results['phi_features'].items():
        print(f"  {key:25s}: [{stats['min']:12.4g}, {stats['max']:12.4g}]")
    
    print("\n[TARGET REGIONS]")
    for key, stats in results['target_regions'].items():
        print(f"  {key:25s}: [{stats['min']:6d}, {stats['max']:6d}]")
    
    print("\n[TARGET REGIONS SUM (bot+pit+top+wall)]")
    s = results['target_regions_sum']
    print(f"  Summe pro Event       : [{s['min']:6d}, {s['max']:6d}]")
    
    print("\n[INDIVIDUAL TARGETS (Voxel)]")
    t = results['individual_targets']
    print(f"  Min über alle Voxel   : {t['min_over_all_voxels']:6d}")
    print(f"  Max über alle Voxel   : {t['max_over_all_voxels']:6d}")
    print(f"  Anzahl Voxel          : {t['n_voxels']:6d}")
    print("="*60)


# === HAUPTAUSFÜHRUNG ===
if __name__ == "__main__":
    # Analysiere beide Dateien
    for dataset_name in ["train", "validation"]:
        hdf5_file = f"/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/MLFormatHomogeneousNCsZylSSD300PMTs/resum_output_0_{dataset_name}.hdf5"
        output_json = f"analysis_results_{dataset_name}.json"
        
        print(f"\n{'='*60}")
        print(f"ANALYSIERE: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        results = analyze_hdf5_structure(
            filepath=hdf5_file,
            chunk_size=100_000,
            output_json=output_json
        )
        
        print_summary(results)