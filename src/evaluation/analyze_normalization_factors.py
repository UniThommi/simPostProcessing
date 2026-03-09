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
        n_events = f['phi_matrix'].shape[0]
        phi_columns = [c.decode() if isinstance(c, bytes) else c for c in f['phi_columns'][:]]
        region_columns = [c.decode() if isinstance(c, bytes) else c for c in f['region_columns'][:]]
        target_columns = [c.decode() if isinstance(c, bytes) else c for c in f['target_columns'][:]]
        n_targets = len(target_columns)
        print(f"Analysiere {n_events:,} Events...")

        # === 1. PHI FEATURES (außer matID, volID) ===
        print("\n[1/3] Analysiere phi-Features...")
        excluded = {'matID', 'volID'}
        phi_indices = {name: i for i, name in enumerate(phi_columns) if name not in excluded}

        phi_stats = {name: {'min': np.inf, 'max': -np.inf} for name in phi_indices}

        for start_idx in range(0, n_events, chunk_size):
            end_idx = min(start_idx + chunk_size, n_events)
            chunk = f['phi_matrix'][start_idx:end_idx]

            for name, col_idx in phi_indices.items():
                col = chunk[:, col_idx]
                phi_stats[name]['min'] = min(phi_stats[name]['min'], float(np.min(col)))
                phi_stats[name]['max'] = max(phi_stats[name]['max'], float(np.max(col)))

            if (start_idx // chunk_size + 1) % 5 == 0:
                print(f"  Chunk {start_idx//chunk_size + 1}/{(n_events-1)//chunk_size + 1}")

        results['phi_features'] = phi_stats

        # === 2. TARGET REGIONS ===
        print("\n[2/3] Analysiere target_regions...")
        region_stats = {name: {'min': np.inf, 'max': -np.inf} for name in region_columns}
        sum_stats = {'min': np.inf, 'max': -np.inf}

        for start_idx in range(0, n_events, chunk_size):
            end_idx = min(start_idx + chunk_size, n_events)
            chunk = f['region_matrix'][start_idx:end_idx]

            for col_idx, name in enumerate(region_columns):
                col = chunk[:, col_idx]
                region_stats[name]['min'] = min(region_stats[name]['min'], int(np.min(col)))
                region_stats[name]['max'] = max(region_stats[name]['max'], int(np.max(col)))

            event_sums = chunk.sum(axis=1)
            sum_stats['min'] = min(sum_stats['min'], int(np.min(event_sums)))
            sum_stats['max'] = max(sum_stats['max'], int(np.max(event_sums)))

            if (start_idx // chunk_size + 1) % 5 == 0:
                print(f"  Chunk {start_idx//chunk_size + 1}/{(n_events-1)//chunk_size + 1}")

        results['target_regions'] = region_stats
        results['target_regions_sum'] = sum_stats

        # === 3. INDIVIDUAL TARGETS (Voxel-weise) ===
        print(f"\n[3/4] Analysiere individual targets ({n_targets} Voxel)...")
        global_target_min = np.inf
        global_target_max = -np.inf

        for start_idx in range(0, n_events, chunk_size):
            end_idx = min(start_idx + chunk_size, n_events)
            chunk = f['target_matrix'][start_idx:end_idx]

            global_target_min = min(global_target_min, int(np.min(chunk)))
            global_target_max = max(global_target_max, int(np.max(chunk)))

            if (start_idx // chunk_size + 1) % 5 == 0:
                print(f"  Chunk {start_idx//chunk_size + 1}/{(n_events-1)//chunk_size + 1}")

        results['individual_targets'] = {
            'min_over_all_voxels': int(global_target_min),
            'max_over_all_voxels': int(global_target_max),
            'n_voxels': n_targets
        }

        # === 4. VOXEL HIT MULTIPLICITY PRO EVENT ===
        print("\n[4/4] Analysiere Voxel-Hit-Multiplizität pro Event...")
        global_min_hits = np.inf
        global_max_hits = -np.inf
        total_hits = 0

        for start_idx in range(0, n_events, chunk_size):
            end_idx = min(start_idx + chunk_size, n_events)
            chunk = f['target_matrix'][start_idx:end_idx]

            hit_counts = np.count_nonzero(chunk, axis=1)
            global_min_hits = min(global_min_hits, int(hit_counts.min()))
            global_max_hits = max(global_max_hits, int(hit_counts.max()))
            total_hits += int(hit_counts.sum())

            if (start_idx // chunk_size + 1) % 5 == 0:
                print(f"  Chunk {start_idx//chunk_size + 1}/{(n_events-1)//chunk_size + 1}")

        results['voxel_hit_multiplicity'] = {
            'min_hits_per_event': int(global_min_hits),
            'max_hits_per_event': int(global_max_hits),
            'mean_hits_per_event': round(total_hits / n_events, 2),
            'n_voxels_total': n_targets
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

    print("\n[VOXEL HIT MULTIPLICITY (Voxel mit >0 Photonen pro Event)]")
    h = results['voxel_hit_multiplicity']
    print(f"  Min Hits/Event        : {h['min_hits_per_event']:6d}")
    print(f"  Max Hits/Event        : {h['max_hits_per_event']:6d}")
    print(f"  Mean Hits/Event       : {h['mean_hits_per_event']:10.2f}")
    print(f"  Voxel gesamt          : {h['n_voxels_total']:6d}")


# === HAUPTAUSFÜHRUNG ===
if __name__ == "__main__":
    # Analysiere beide Dateien
    for dataset_name in ["train", "validation"]:
        hdf5_file = f"/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/MLFormatHomogeneousNCsZylSSD300PMTs/ncscore_output_0_{dataset_name}.hdf5"
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