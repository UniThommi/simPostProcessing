# Author: Thomas Buerger (University of Tübingen)
"""
Minimum and maximum of the diffusion-model training data, used to find a suitable
normalization for the diffusion model.

Streams through the training and validation files written by
zylFullPostprocessingOptimized.py (the diffusion-model data set) and records:
  1. min/max of every NC feature in phi_matrix (except the categorical matID/volID),
  2. min/max of the photon hits per region (region_matrix) and of their sum per NC,
  3. min/max of the photon hits per voxel over the whole target_matrix,
  4. the number of hit voxels per NC (min, max, mean).

The results are printed and written to analysis_results_{train,validation}.json.
The input paths are set in the __main__ block.
"""
import h5py
import numpy as np
import json
from typing import Dict, Any, Union
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # src/, for local_paths
from local_paths import required_local_path

def analyze_hdf5_structure(
    filepath: Union[str, Path],
    chunk_size: int = 100_000,
    output_json: Union[str, Path] = "analysis_results.json"
) -> Dict[str, Any]:
    """
    Compute the value ranges of phi features, region hits and voxel hits of one file.

    Parameters
    ----------
    filepath : str | Path
        Post-processed HDF5 file
    chunk_size : int, default=100_000
        Number of rows read at once (keeps the memory usage bounded)
    output_json : str | Path, default="analysis_results.json"
        Path of the JSON file the results are written to

    Returns
    -------
    Dict[str, Any]
        Results with the keys phi_features, target_regions, target_regions_sum,
        individual_targets and voxel_hit_multiplicity
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
        print(f"Analyzing {n_events:,} events...")

        # === 1. PHI FEATURES (except the categorical matID and volID) ===
        print("\n[1/4] Analyzing phi features...")
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

        # === 2. TARGET REGIONS (per region and summed per event) ===
        print("\n[2/4] Analyzing target regions...")
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

        # === 3. INDIVIDUAL TARGETS (hits per voxel, global min/max) ===
        print(f"\n[3/4] Analyzing individual targets ({n_targets} voxels)...")
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

        # === 4. VOXEL HIT MULTIPLICITY (voxels with > 0 photons per event) ===
        print("\n[4/4] Analyzing voxel hit multiplicity per event...")
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
    print(f"\n✓ Analysis finished. Exporting to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=2, ensure_ascii=False)

    print(f"✓ JSON saved: {output_json}")

    return results


def print_summary(results: Dict[str, Any]) -> None:
    """
    Print the results of analyze_hdf5_structure() as a summary table.

    Parameters
    ----------
    results : Dict[str, Any]
        Result dictionary of analyze_hdf5_structure()
    """
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    print("\n[PHI FEATURES]")
    for key, stats in results['phi_features'].items():
        print(f"  {key:25s}: [{stats['min']:12.4g}, {stats['max']:12.4g}]")

    print("\n[TARGET REGIONS]")
    for key, stats in results['target_regions'].items():
        print(f"  {key:25s}: [{stats['min']:6d}, {stats['max']:6d}]")

    print("\n[TARGET REGIONS SUM (bot+pit+top+wall)]")
    s = results['target_regions_sum']
    print(f"  Sum per event         : [{s['min']:6d}, {s['max']:6d}]")

    print("\n[INDIVIDUAL TARGETS (voxels)]")
    t = results['individual_targets']
    print(f"  Min over all voxels   : {t['min_over_all_voxels']:6d}")
    print(f"  Max over all voxels   : {t['max_over_all_voxels']:6d}")
    print(f"  Number of voxels      : {t['n_voxels']:6d}")
    print("="*60)

    print("\n[VOXEL HIT MULTIPLICITY (voxels with >0 photons per event)]")
    h = results['voxel_hit_multiplicity']
    print(f"  Min hits/event        : {h['min_hits_per_event']:6d}")
    print(f"  Max hits/event        : {h['max_hits_per_event']:6d}")
    print(f"  Mean hits/event       : {h['mean_hits_per_event']:10.2f}")
    print(f"  Voxels in total       : {h['n_voxels_total']:6d}")


# === MAIN ===
if __name__ == "__main__":
    # Analyze the training and the validation file
    for dataset_name in ["train", "validation"]:
        # Root directory from local_paths.json (git-ignored)
        hdf5_file = required_local_path(
            "scratch",
            f"optPhotonSensitiveSurface/MLFormatHomogeneousNCsZylSSD300PMTs/ncscore_output_0_{dataset_name}.hdf5",
        )
        output_json = f"analysis_results_{dataset_name}.json"

        print(f"\n{'='*60}")
        print(f"ANALYZING: {dataset_name.upper()}")
        print(f"{'='*60}")

        results = analyze_hdf5_structure(
            filepath=hdf5_file,
            chunk_size=100_000,
            output_json=output_json
        )

        print_summary(results)
