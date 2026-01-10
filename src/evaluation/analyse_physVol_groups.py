#!/usr/bin/env python3
"""
Volume Hit Spectrum Analysis with Chunked Processing

Memory-efficient analysis for large HDF5 files using chunked reading.

Author: Scientific Data Pipeline
Date: 2026-01-08
"""

import re
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import argparse
import psutil
import traceback


# ============================================================================
# Volume Grouping Patterns (unverändert)
# ============================================================================

class VolumeGrouper:
    """
    Maps individual volume names to semantic groups using regex patterns.
    """
    
    def __init__(self):
        """Initialize regex patterns for volume grouping."""
        
        # Explicit single-name groups
        self.explicit_groups = {
            'foot', 'tank', 'skirt', 'water',
            'innercryostat', 'outercryostat', 
            'atmosphericlar', 'undergroundlar',
            'reentrancetube', 'neutronmoderator',
            'vacuumgap', 'noVolume'
        }
        
        # Regex-based patterns with group names
        self.regex_patterns = [
            # Germanium detectors
            ('V_detector', re.compile(r'^V\d{4}$')),
            
            # SiPM modules
            ('S_module_bottom', re.compile(r'^S\d{4}B$')),
            ('S_module_top', re.compile(r'^S\d{4}T$')),
            
            # PMTs
            ('PMT_main', re.compile(r'^PMT\d{6}$')),
            ('PMT_3inch', re.compile(r'^PMT300\d{4}$')),
            
            # Detector penetrations
            ('V_penetration', re.compile(r'^V\d{4}_pen$')),
            
            # SiPM wrappings
            ('S_wrap_bottom', re.compile(r'^S\d{4}B_wrap$')),
            ('S_wrap_top', re.compile(r'^S\d{4}T_wrap$')),
            
            # Fiber system
            ('fiber_sipm', re.compile(r'^fiber_S\d{4}_s$')),
            ('fiber_core', re.compile(r'^fiber_core_l[\d.]+$')),
            ('fiber_cl1', re.compile(r'^fiber_cl1_l[\d.]+$')),
            ('fiber_cl2_tpb', re.compile(r'^fiber_cl2_l[\d.]+_tpb\d+$')),
            
            # PMT components
            ('PMT_vacuum', re.compile(r'^PMT\d{6}_vacuum$')),
            ('PMT_window', re.compile(r'^PMT\d{6}_window$')),
            ('PMT_3inch_vacuum', re.compile(r'^PMT300\d{4}_vacuum$')),
            ('PMT_3inch_window', re.compile(r'^PMT300\d{4}_window$')),
            
            # Detector mounting - click positions
            ('V_click_top_0', re.compile(r'^V\d{4}_click_top_0$')),
            ('V_click_top_1', re.compile(r'^V\d{4}_click_top_1$')),
            ('V_click_top_2', re.compile(r'^V\d{4}_click_top_2$')),
            
            # HV cables per string (1-9)
            ('V_hv_cable_string_1', re.compile(r'^V\d{4}_hv_cable_string_1$')),
            ('V_hv_cable_string_2', re.compile(r'^V\d{4}_hv_cable_string_2$')),
            ('V_hv_cable_string_3', re.compile(r'^V\d{4}_hv_cable_string_3$')),
            ('V_hv_cable_string_4', re.compile(r'^V\d{4}_hv_cable_string_4$')),
            ('V_hv_cable_string_5', re.compile(r'^V\d{4}_hv_cable_string_5$')),
            ('V_hv_cable_string_6', re.compile(r'^V\d{4}_hv_cable_string_6$')),
            ('V_hv_cable_string_7', re.compile(r'^V\d{4}_hv_cable_string_7$')),
            ('V_hv_cable_string_8', re.compile(r'^V\d{4}_hv_cable_string_8$')),
            ('V_hv_cable_string_9', re.compile(r'^V\d{4}_hv_cable_string_9$')),
            
            # HV clamps per string
            ('V_hv_clamp_string_1', re.compile(r'^V\d{4}_hv_clamp_string_1$')),
            ('V_hv_clamp_string_2', re.compile(r'^V\d{4}_hv_clamp_string_2$')),
            ('V_hv_clamp_string_3', re.compile(r'^V\d{4}_hv_clamp_string_3$')),
            ('V_hv_clamp_string_4', re.compile(r'^V\d{4}_hv_clamp_string_4$')),
            ('V_hv_clamp_string_5', re.compile(r'^V\d{4}_hv_clamp_string_5$')),
            ('V_hv_clamp_string_6', re.compile(r'^V\d{4}_hv_clamp_string_6$')),
            ('V_hv_clamp_string_7', re.compile(r'^V\d{4}_hv_clamp_string_7$')),
            ('V_hv_clamp_string_8', re.compile(r'^V\d{4}_hv_clamp_string_8$')),
            ('V_hv_clamp_string_9', re.compile(r'^V\d{4}_hv_clamp_string_9$')),
            
            # Signal readout per string
            ('V_signal_asic_string_1', re.compile(r'^V\d{4}_signal_asic_string_1$')),
            ('V_signal_asic_string_2', re.compile(r'^V\d{4}_signal_asic_string_2$')),
            ('V_signal_asic_string_3', re.compile(r'^V\d{4}_signal_asic_string_3$')),
            ('V_signal_asic_string_4', re.compile(r'^V\d{4}_signal_asic_string_4$')),
            ('V_signal_asic_string_5', re.compile(r'^V\d{4}_signal_asic_string_5$')),
            ('V_signal_asic_string_6', re.compile(r'^V\d{4}_signal_asic_string_6$')),
            ('V_signal_asic_string_7', re.compile(r'^V\d{4}_signal_asic_string_7$')),
            ('V_signal_asic_string_8', re.compile(r'^V\d{4}_signal_asic_string_8$')),
            ('V_signal_asic_string_9', re.compile(r'^V\d{4}_signal_asic_string_9$')),
            
            ('V_signal_cable_string_1', re.compile(r'^V\d{4}_signal_cable_string_1$')),
            ('V_signal_cable_string_2', re.compile(r'^V\d{4}_signal_cable_string_2$')),
            ('V_signal_cable_string_3', re.compile(r'^V\d{4}_signal_cable_string_3$')),
            ('V_signal_cable_string_4', re.compile(r'^V\d{4}_signal_cable_string_4$')),
            ('V_signal_cable_string_5', re.compile(r'^V\d{4}_signal_cable_string_5$')),
            ('V_signal_cable_string_6', re.compile(r'^V\d{4}_signal_cable_string_6$')),
            ('V_signal_cable_string_7', re.compile(r'^V\d{4}_signal_cable_string_7$')),
            ('V_signal_cable_string_8', re.compile(r'^V\d{4}_signal_cable_string_8$')),
            ('V_signal_cable_string_9', re.compile(r'^V\d{4}_signal_cable_string_9$')),
            
            ('V_signal_clamp_string_1', re.compile(r'^V\d{4}_signal_clamp_string_1$')),
            ('V_signal_clamp_string_2', re.compile(r'^V\d{4}_signal_clamp_string_2$')),
            ('V_signal_clamp_string_3', re.compile(r'^V\d{4}_signal_clamp_string_3$')),
            ('V_signal_clamp_string_4', re.compile(r'^V\d{4}_signal_clamp_string_4$')),
            ('V_signal_clamp_string_5', re.compile(r'^V\d{4}_signal_clamp_string_5$')),
            ('V_signal_clamp_string_6', re.compile(r'^V\d{4}_signal_clamp_string_6$')),
            ('V_signal_clamp_string_7', re.compile(r'^V\d{4}_signal_clamp_string_7$')),
            ('V_signal_clamp_string_8', re.compile(r'^V\d{4}_signal_clamp_string_8$')),
            ('V_signal_clamp_string_9', re.compile(r'^V\d{4}_signal_clamp_string_9$')),
            
            # Insulator holders
            ('V_insulator_0', re.compile(r'^V\d{4}_insulator_du_holder_0$')),
            ('V_insulator_1', re.compile(r'^V\d{4}_insulator_du_holder_1$')),
            ('V_insulator_2', re.compile(r'^V\d{4}_insulator_du_holder_2$')),
            
            # Tyvek reflectors
            ('tyvek_bot', re.compile(r'^my_tyvek_bot_foil$')),
            ('tyvek_pit', re.compile(r'^my_tyvek_pit_foil$')),
            ('tyvek_top', re.compile(r'^my_tyvek_top_foil$')),
            ('tyvek_zyl', re.compile(r'^my_tyvek_zyl_foil$')),
            ('tyvek_bottom', re.compile(r'^tyvek_bottom_foil$')),
            ('tyvek_prism', re.compile(r'^tyvek_prism_foil$')),
            
            # Copper rods by position
            ('cu_rod_0', re.compile(r'^string_\d+_cu_rod_0$')),
            ('cu_rod_1', re.compile(r'^string_\d+_cu_rod_1$')),
            ('cu_rod_2', re.compile(r'^string_\d+_cu_rod_2$')),
            
            # Tristar mounting
            ('tristar_xlarge', re.compile(r'^tristar_xlarge_string_\d+$')),
            
            # String support
            ('string_support', re.compile(r'^string_support_structure_string_\d+$')),
        ]
    
    def assign_group(self, volume_name: str) -> str:
        """Assign volume name to group."""
        if volume_name in self.explicit_groups:
            return volume_name
        
        for group_name, pattern in self.regex_patterns:
            if pattern.match(volume_name):
                return group_name
        
        return "unmatched"
    
    def build_id_to_group_mapping(
        self, 
        volume_mapping: Dict[str, int]
    ) -> Tuple[Dict[int, str], Set[str]]:
        """Pre-compute volume ID → group name mapping."""
        id_to_group = {}
        unmatched_names = set()
        
        for vol_name, vol_id in volume_mapping.items():
            group = self.assign_group(vol_name)
            id_to_group[vol_id] = group
            
            if group == "unmatched":
                unmatched_names.add(vol_name)
        
        return id_to_group, unmatched_names


# ============================================================================
# Memory-Efficient Chunked Analysis
# ============================================================================

def get_optimal_chunk_size(n_events: int, n_voxels: int) -> int:
    """
    Calculate optimal chunk size based on available memory.
    
    Args:
        n_events: Total number of events in file
        n_voxels: Number of voxels
        
    Returns:
        Chunk size (number of events per chunk)
        
    Strategy:
        - Target: ~2 GB per chunk for target matrix
        - Conservative estimate: int32 = 4 bytes
        - Include safety margin (50%)
    """
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Target chunk memory: 2 GB or 25% of available (whichever is smaller)
    target_chunk_gb = min(2.0, available_memory_gb * 0.25)
    
    # Bytes per event row: n_voxels * 4 bytes (int32)
    bytes_per_event = n_voxels * 4
    
    # Calculate chunk size with 50% safety margin
    chunk_size = int((target_chunk_gb * 1024**3) / bytes_per_event * 0.5)
    
    # Ensure reasonable bounds
    chunk_size = max(1000, min(chunk_size, n_events))
    
    print(f"  Memory-aware chunking:")
    print(f"    Available RAM: {available_memory_gb:.1f} GB")
    print(f"    Target chunk size: {target_chunk_gb:.1f} GB")
    print(f"    Events per chunk: {chunk_size:,}")
    print(f"    Estimated chunk memory: {chunk_size * bytes_per_event / 1024**3:.2f} GB")
    
    return chunk_size


class VolumeHitAnalyzer:
    """
    Analyzes voxel hit distributions with chunked processing.
    """
    
    def __init__(self, grouper: VolumeGrouper):
        """Initialize analyzer."""
        self.grouper = grouper
        self.id_to_group = {}
        self.group_hits_summed = defaultdict(list)
        self.group_hits_unique = defaultdict(list)
        self.group_event_counts = Counter()
    
    def load_volume_mapping(self, mapping_path: Path) -> Dict[str, int]:
        """Load global volume mapping from JSON."""
        with mapping_path.open('r') as f:
            mapping = json.load(f)
        
        print(f"Loaded {len(mapping)} volume mappings from {mapping_path.name}")
        return mapping
    
    def preprocess_mappings(self, volume_mapping: Dict[str, int]) -> Set[str]:
        """Build ID→group mapping."""
        self.id_to_group, unmatched = self.grouper.build_id_to_group_mapping(
            volume_mapping
        )
        
        print(f"\nVolume grouping statistics:")
        print(f"  Total volumes in mapping: {len(volume_mapping)}")
        print(f"  Unique groups identified: {len(set(self.id_to_group.values()))}")
        print(f"  Unmatched volumes: {len(unmatched)}")
        
        return unmatched
    
    def analyze_hdf5_file_chunked(
        self, 
        hdf5_path: Path,
        chunk_size: Optional[int] = None
    ) -> None:
        """
        Analyze HDF5 file using chunked reading.
        
        Args:
            hdf5_path: Path to HDF5 file
            chunk_size: Events per chunk (auto-calculated if None)
            
        Key difference from original:
            - Does NOT load entire target matrix into memory
            - Processes chunks sequentially
            - Minimal memory footprint per chunk
        """
        print(f"\nProcessing: {hdf5_path.name}")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Get dataset dimensions
            n_events = len(f['phi']['volID'])
            
            # Get voxel IDs (small, can load fully)
            voxel_ids = sorted([int(k) for k in f['target'].keys()])
            n_voxels = len(voxel_ids)
            
            print(f"  Total events: {n_events:,}")
            print(f"  Total voxels: {n_voxels:,}")
            
            # Calculate optimal chunk size
            if chunk_size is None:
                chunk_size = get_optimal_chunk_size(n_events, n_voxels)
            
            n_chunks = int(np.ceil(n_events / chunk_size))
            print(f"  Processing in {n_chunks} chunk(s)\n")
            
            # Process chunks
            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, n_events)
                chunk_size_actual = chunk_end - chunk_start
                
                print(f"  Chunk {chunk_idx+1}/{n_chunks}: "
                      f"Events {chunk_start:,} - {chunk_end:,} "
                      f"({chunk_size_actual:,} events)")
                
                # Load ONLY this chunk's data
                vol_ids_chunk = f['phi']['volID'][chunk_start:chunk_end]
                
                # Load target data for this chunk
                # Strategy: Load each voxel column as needed
                target_chunk = np.zeros((chunk_size_actual, n_voxels), dtype=np.int32)
                
                for voxel_idx, voxel_id in enumerate(voxel_ids):
                    target_chunk[:, voxel_idx] = f['target'][str(voxel_id)][chunk_start:chunk_end]
                
                # Process events in this chunk
                for local_idx in range(chunk_size_actual):
                    vol_id = int(vol_ids_chunk[local_idx])
                    
                    # Get group
                    group = self.id_to_group.get(vol_id, "unmatched")
                    
                    # Extract hit counts for this event
                    hits_per_voxel = target_chunk[local_idx, :]
                    
                    # Metric A/B: Total hits
                    total_hits = int(np.sum(hits_per_voxel))
                    self.group_hits_summed[group].append(total_hits)
                    
                    # Metric C: Unique voxels
                    n_unique_voxels = int(np.count_nonzero(hits_per_voxel))
                    self.group_hits_unique[group].append(n_unique_voxels)
                    
                    # Count event
                    self.group_event_counts[group] += 1
                
                print(f"    ✓ Processed {chunk_size_actual:,} events")
            
            print(f"  ✓ File complete: {n_events:,} total events processed")
    
    def analyze_multiple_files(
        self, 
        hdf5_paths: List[Path],
        chunk_size: Optional[int] = None
    ) -> None:
        """Analyze multiple files sequentially."""
        print(f"\nAnalyzing {len(hdf5_paths)} HDF5 file(s)...")
        
        for path in hdf5_paths:
            try:
                self.analyze_hdf5_file_chunked(path, chunk_size)
            except Exception as e:
                print(f"  ✗ Error processing {path.name}: {e}")
                traceback.print_exc()
                continue
    
    def get_results(self) -> Dict:
        """Get accumulated results."""
        return {
            'groups': sorted(self.group_hits_summed.keys()),
            'summed_hits': dict(self.group_hits_summed),
            'unique_voxels': dict(self.group_hits_unique),
            'event_counts': dict(self.group_event_counts)
        }


# ============================================================================
# Visualization (unverändert, aber kompakter)
# ============================================================================

class HitSpectrumPlotter:
    """Create normalized histograms."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_group_spectra(
        self,
        results: Dict,
        metric: str,
        bins: int = 50
    ) -> None:
        """Create histogram plots for all groups."""
        groups = results['groups']
        
        if metric == 'summed':
            data_dict = results['summed_hits']
            title_suffix = "Total Voxel Hits per Event"
            xlabel = "Total Hits (Sum over all voxels)"
            filename = "volume_hit_spectra_summed_hits.png"
        elif metric == 'unique':
            data_dict = results['unique_voxels']
            title_suffix = "Unique Voxels Hit per Event"
            xlabel = "Number of Unique Voxels"
            filename = "volume_hit_spectra_unique_voxels.png"
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        event_counts = results['event_counts']
        
        # Layout
        n_groups = len(groups)
        n_cols = 4
        n_rows = int(np.ceil(n_groups / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_groups > 1 else [axes]
        
        for idx, group in enumerate(groups):
            ax = axes[idx]
            
            hits = np.array(data_dict[group])
            n_events = event_counts[group]
            
            if len(hits) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f"{group}\n(0 events)")
                continue
            
            # Histogram with linear bins
            max_val = np.max(hits)
            bin_edges = np.linspace(0, max_val, bins + 1)
            
            counts, edges = np.histogram(hits, bins=bin_edges)
            
            # Normalize by event count
            counts_normalized = counts / n_events
            
            # Plot
            bin_centers = (edges[:-1] + edges[1:]) / 2
            ax.bar(bin_centers, counts_normalized, width=np.diff(edges),
                   alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Statistics
            mean_val = np.mean(hits)
            median_val = np.median(hits)
            
            ax.set_title(f"{group}\n({n_events:,} events)", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Normalized Count\n(per event)", fontsize=8)
            ax.tick_params(labelsize=7)
            
            stats_text = f"μ={mean_val:.1f}\nmed={median_val:.1f}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=7, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot: {output_path}")
        plt.close()
    
    def create_all_plots(self, results: Dict, bins: int = 50) -> None:
        """Create both metric plots."""
        print("\nGenerating visualizations...")
        self.plot_group_spectra(results, metric='summed', bins=bins)
        self.plot_group_spectra(results, metric='unique', bins=bins)


# ============================================================================
# Validation and Reporting (unverändert)
# ============================================================================

def write_unmatched_report(unmatched_names: Set[str], output_path: Path) -> None:
    """Write unmatched volumes to file."""
    sorted_names = sorted(unmatched_names)
    
    with output_path.open('w') as f:
        f.write(f"# Unmatched Physical Volumes\n")
        f.write(f"# Total: {len(sorted_names)}\n\n")
        
        for name in sorted_names:
            f.write(f"{name}\n")
    
    print(f"\n✓ Unmatched volumes written to: {output_path}")


def print_group_summary(results: Dict) -> None:
    """Print summary statistics."""
    print("\n" + "="*70)
    print("GROUP SUMMARY STATISTICS")
    print("="*70)
    
    groups = results['groups']
    summed_hits = results['summed_hits']
    unique_voxels = results['unique_voxels']
    event_counts = results['event_counts']
    
    print(f"{'Group':<35} {'Events':>10} {'Mean Hits':>12} {'Mean Voxels':>13}")
    print("-"*70)
    
    for group in sorted(groups):
        n_events = event_counts[group]
        
        hits_array = np.array(summed_hits[group])
        voxels_array = np.array(unique_voxels[group])
        
        mean_hits = np.mean(hits_array) if len(hits_array) > 0 else 0.0
        mean_voxels = np.mean(voxels_array) if len(voxels_array) > 0 else 0.0
        
        print(f"{group:<35} {n_events:>10,} {mean_hits:>12.2f} {mean_voxels:>13.2f}")
    
    print("="*70)


# ============================================================================
# Main Pipeline
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Memory-efficient volume hit spectrum analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python %(prog)s \\
    --data /path/to/resum_output_0_train.hdf5 \\
    --mapping /path/to/globalPhysVolMappings.json \\
    --output ./analysis_results \\
    --chunk-size 10000
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=Path,
        nargs='+',
        required=True,
        help='Path(s) to post-processed HDF5 file(s)'
    )
    
    parser.add_argument(
        '--mapping', '-m',
        type=Path,
        required=True,
        help='Path to globalPhysVolMappings.json'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./volume_analysis_results'),
        help='Output directory (default: ./volume_analysis_results)'
    )
    
    parser.add_argument(
        '--bins',
        type=int,
        default=50,
        help='Number of histogram bins (default: 50)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Events per chunk (auto if not specified)'
    )
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_arguments()
    
    # Validate inputs
    for data_path in args.data:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not args.mapping.exists():
        raise FileNotFoundError(f"Mapping file not found: {args.mapping}")
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("VOLUME HIT SPECTRUM ANALYSIS (CHUNKED PROCESSING)")
    print("="*70)
    print(f"Input files: {len(args.data)}")
    for path in args.data:
        print(f"  - {path.name}")
    print(f"Mapping: {args.mapping}")
    print(f"Output: {args.output}")
    if args.chunk_size:
        print(f"Chunk size: {args.chunk_size:,} events (manual)")
    else:
        print(f"Chunk size: Auto-calculated based on available RAM")
    print("="*70)
    
    # Initialize
    grouper = VolumeGrouper()
    analyzer = VolumeHitAnalyzer(grouper)
    
    # Load mappings
    volume_mapping = analyzer.load_volume_mapping(args.mapping)
    unmatched_names = analyzer.preprocess_mappings(volume_mapping)
    
    # Write unmatched report
    if unmatched_names:
        unmatched_path = args.output / "unmatched_volumes.txt"
        write_unmatched_report(unmatched_names, unmatched_path)
    
    # Analyze files with chunking
    analyzer.analyze_multiple_files(args.data, args.chunk_size)
    
    # Get results
    results = analyzer.get_results()
    
    # Summary
    print_group_summary(results)
    
    # Plots
    plotter = HitSpectrumPlotter(args.output)
    plotter.create_all_plots(results, bins=args.bins)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()