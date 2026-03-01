#!/usr/bin/env python3
# zoneRatioAnalysis.py
"""
Zone-based Photon Detection Efficiency Ratio: SSD vs. PMT
Subdivides detector areas (pit, top, wall) into equal-area zones.
PMTs that are cut by zone boundaries contribute fractionally to both zones.

Author: Your analysis pipeline
Date: 2026-02-22
"""

import h5py
import numpy as np
import gc
import psutil
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

N_ZONES_PIT = 5    # radial zones
N_ZONES_WALL = 8   # z-zones
N_ZONES_TOP = 4    # radial zones
# Bot: no subdivision (1 zone)

# Paths
BASE_DIR = Path("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface")
SSD_DIR = BASE_DIR / "rawOpticalSSDFromMusunNCS"
PMT_DIR = BASE_DIR / "rawOpticalHomogeneousPMTsFromMusunNCs"
OUTPUT_DIR = BASE_DIR / "zone_analysis_results"
PMT_JSON_PATH = Path("/global/cfs/projectdirs/legend/users/tbuerger/sim/data/"
                     "optPhotonSensitiveSurface/homogeneous300PMTpositions_currentDist.json")

# PMT cathode radius (detection area)
PMT_RADIUS = 110.0  # mm

# Monte Carlo samples for fractional overlap calculation
MC_SAMPLES = 10000

# Zone scan configuration
ZONE_SCAN_AREAS = ['pit', 'top', 'wall']  # Areas to optimize. Empty list = no scan.
ZONE_SCAN_RANGE = range(2, 30)            # Scan from 2 to 10 zones per area
SNR_THRESHOLD = 3.0                       # Minimum SNR (3σ significance)
MIN_PMTS_PER_ZONE = 8                    # Minimum effective PMTs per zone

# Max allowed deviation of PMT fraction sums from 1.0
FRACTION_TOLERANCE = 0.01

# =============================================================================
# COMPARE MODE: Use reference zones instead of optimizing
# =============================================================================
COMPARE_MODE = True  # If True, load reference zones and skip optimization
REFERENCE_JSON = Path("/pscratch/sd/t/tbuerger/data/proofeDetectionEfficiencies/"
                      "experimentEfficiencyRatio/PMTs_div_SSD_1/zone_analysis/"
                      "zone_ratio_results.json")

# =============================================================================
# GEOMETRY
# =============================================================================

@dataclass
class GeometryConfig:
    """Detector geometry parameters."""
    geometry_name: str = "currentDist"
    h: float = 8900
    r_zylinder: float = 4300
    z_origin: float = 20
    r_pit: float = 3800
    r_zyl_bot: float = 3950
    r_zyl_top: float = 1200
    z_offset: float = -5000

    @property
    def h_zylinder(self) -> float:
        return self.h - 1  # 8899

    @property
    def z_cut_bot(self) -> float:
        return -4979

    @property
    def z_cut_top(self) -> float:
        return self.z_cut_bot + self.h_zylinder - 2  # 3918

    @property
    def wall_height(self) -> float:
        return self.z_cut_top - self.z_cut_bot  # 8897

    @property
    def area_pit(self) -> float:
        return np.pi * self.r_pit**2

    @property
    def area_bot(self) -> float:
        return np.pi * (self.r_zylinder**2 - self.r_zyl_bot**2)

    @property
    def area_top(self) -> float:
        return np.pi * (self.r_zylinder**2 - self.r_zyl_top**2)

    @property
    def area_wall(self) -> float:
        return 2 * np.pi * self.r_zylinder * self.wall_height


# =============================================================================
# PMT DATA
# =============================================================================

@dataclass
class PMTInfo:
    """Single PMT information."""
    index: str
    center: np.ndarray
    layer: str

    @property
    def r(self) -> float:
        return np.sqrt(self.center[0]**2 + self.center[1]**2)

    @property
    def z(self) -> float:
        return self.center[2]

    @property
    def phi(self) -> float:
        return np.arctan2(self.center[1], self.center[0])


def load_pmt_data(json_path: Path) -> List[PMTInfo]:
    """Load PMT positions from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    pmts = []
    for entry in data:
        pmts.append(PMTInfo(
            index=entry['index'],
            center=np.array(entry['center']),
            layer=entry['layer'].lower()
        ))
    return pmts


def get_pmts_by_layer(pmts: List[PMTInfo]) -> Dict[str, List[PMTInfo]]:
    """Group PMTs by layer."""
    grouped: Dict[str, List[PMTInfo]] = {'pit': [], 'bot': [], 'top': [], 'wall': []}
    for pmt in pmts:
        if pmt.layer in grouped:
            grouped[pmt.layer].append(pmt)
    return grouped


# =============================================================================
# UID <-> PMT INDEX MAPPING
# =============================================================================

def pmt_index_to_uid(index: str) -> int:
    """
    Convert PMT JSON index to detector UID.

    Normal:   '10' + index → 10XXXXXX (8 digits)
    Overflow: '1' + index  → 1XXXXXXX (8 digits, wall PMTs where '10'+index > 8 digits)
    
    Both produce 8-digit UIDs. Overflow is detected when '10'+index exceeds 8 digits.
    """
    uid_normal = '10' + index
    if len(uid_normal) == 8:
        return int(uid_normal)
    # Overflow: '1' + index
    uid_overflow = '1' + index
    if len(uid_overflow) == 8:
        return int(uid_overflow)
    raise ValueError(f"Cannot convert PMT index '{index}' to 8-digit UID")


def uid_to_pmt_index(det_uid: int) -> str:
    """
    Convert detector UID to PMT JSON index.
    
    Normal:   10XXXXXX → strip '10' → 6-digit index
    Overflow: 1XXXXXXX (2nd digit != '0') → strip '1' → 7-digit index
    """
    s = str(det_uid)
    if len(s) != 8:
        return ''
    if s[:2] == '10':
        return s[2:]
    elif s[0] == '1' and s[1] != '0':
        return s[1:]
    return ''


def build_uid_to_pmt_map(pmts: List[PMTInfo]) -> Dict[int, PMTInfo]:
    """Build mapping from detector UID to PMT info using forward mapping."""
    uid_map: Dict[int, PMTInfo] = {}
    for pmt in pmts:
        uid = pmt_index_to_uid(pmt.index)
        if uid in uid_map:
            print(f"  ⚠️  Duplicate UID {uid} for indices {uid_map[uid].index} and {pmt.index}")
        uid_map[uid] = pmt
    return uid_map


def crosscheck_uids(
    uid_to_pmt: Dict[int, PMTInfo],
    observed_uids: set,
    setup_name: str
) -> None:
    """Crosscheck: all observed PMT UIDs should map to a JSON entry and vice versa."""
    json_uids = set(uid_to_pmt.keys())
    in_data_not_json = observed_uids - json_uids
    in_json_not_data = json_uids - observed_uids

    print(f"\n  UID Crosscheck ({setup_name}):")
    print(f"    JSON PMTs: {len(json_uids)}, Observed UIDs: {len(observed_uids)}")

    if in_data_not_json:
        print(f"    ⚠️  {len(in_data_not_json)} UIDs in data but NOT in JSON: "
              f"{sorted(in_data_not_json)[:10]}...")
    if in_json_not_data:
        print(f"    ℹ️  {len(in_json_not_data)} PMTs in JSON but not observed in data "
              f"(normal if no photons hit them)")
    if not in_data_not_json:
        print(f"    ✅ All observed UIDs found in JSON.")


# =============================================================================
# ZONE DATA STRUCTURE
# =============================================================================

@dataclass
class Zone:
    """A single zone within an area."""
    zone_id: int
    area_name: str
    boundary_low: float    # r_min or z_min
    boundary_high: float   # r_max or z_max
    area_mm2: float
    pmt_fractions: Dict[str, float] = None  # pmt_index → fraction in this zone
    effective_n_pmts: float = 0.0

    @property
    def pmt_density(self) -> float:
        return self.effective_n_pmts / self.area_mm2 if self.area_mm2 > 0 else 0.0


# =============================================================================
# FRACTIONAL PMT OVERLAP (Monte Carlo)
# =============================================================================

def compute_radial_fractions(
    pmt: PMTInfo,
    boundaries: List[float],
    n_samples: int = MC_SAMPLES,
) -> Dict[int, float]:
    """
    Compute fraction of PMT cathode area in each radial zone via MC sampling.
    
    Samples uniform points in the PMT cathode circle (flat disk on pit/bot/top),
    computes radial distance from z-axis, assigns to zones.
    """
    rng = np.random.default_rng(seed=hash(pmt.index) % (2**31))
    angles = rng.uniform(0, 2 * np.pi, n_samples)
    radii = PMT_RADIUS * np.sqrt(rng.uniform(0, 1, n_samples))

    sample_x = pmt.center[0] + radii * np.cos(angles)
    sample_y = pmt.center[1] + radii * np.sin(angles)
    sample_r = np.sqrt(sample_x**2 + sample_y**2)

    fractions = {}
    n_zones = len(boundaries) - 1
    for zone_id in range(n_zones):
        r_low = boundaries[zone_id]
        r_high = boundaries[zone_id + 1]
        if zone_id == n_zones - 1:
            mask = (sample_r >= r_low) & (sample_r <= r_high)
        else:
            mask = (sample_r >= r_low) & (sample_r < r_high)
        frac = float(np.sum(mask)) / n_samples
        if frac > 0:
            fractions[zone_id] = frac

    return fractions


def compute_z_fractions(
    pmt: PMTInfo,
    boundaries: List[float],
    n_samples: int = MC_SAMPLES,
) -> Dict[int, float]:
    """
    Compute fraction of PMT cathode area in each z-zone via MC sampling.
    
    For wall PMTs the cathode is a circular disk on the cylinder surface.
    The disk normal points radially inward. Points on the disk have
    z = z_center + d_z, where d_z is the z-component of the random offset
    within the disk. The phi-component doesn't affect z-zone assignment.
    """
    rng = np.random.default_rng(seed=hash(pmt.index) % (2**31))
    angles = rng.uniform(0, 2 * np.pi, n_samples)
    radii = PMT_RADIUS * np.sqrt(rng.uniform(0, 1, n_samples))

    # On the cylinder surface, one axis of the disk is along z,
    # the other along the tangential direction (phi).
    # z-offset = radii * cos(angles)
    sample_z = pmt.z + radii * np.cos(angles)

    fractions = {}
    n_zones = len(boundaries) - 1
    for zone_id in range(n_zones):
        z_low = boundaries[zone_id]
        z_high = boundaries[zone_id + 1]
        if zone_id == n_zones - 1:
            mask = (sample_z >= z_low) & (sample_z <= z_high)
        else:
            mask = (sample_z >= z_low) & (sample_z < z_high)
        frac = float(np.sum(mask)) / n_samples
        if frac > 0:
            fractions[zone_id] = frac

    return fractions


# =============================================================================
# ZONE BUILDERS
# =============================================================================

def build_radial_zones(
    pmts: List[PMTInfo],
    n_zones: int,
    r_min: float,
    r_max: float,
    area_name: str,
    max_iter: int = 200,
    tol: float = 0.001,
) -> Tuple[List[Zone], List[float]]:
    """
    Build radial zones with optimized boundaries such that
    effective PMT density (eff_pmts / area) is uniform across zones.

    Uses iterative optimization: starts with equal-area boundaries,
    then shifts boundaries to equalize density.

    Args:
        pmts: PMTs in this area
        n_zones: Number of zones
        r_min, r_max: Radial range
        area_name: Name for logging
        max_iter: Max optimization iterations
        tol: Convergence tolerance (relative density deviation)
    """
    if n_zones == 1:
        total_area = np.pi * (r_max**2 - r_min**2)
        fracs_all = {}
        eff = 0.0
        bounds = [r_min, r_max]
        for pmt in pmts:
            f = compute_radial_fractions(pmt, bounds)
            if 0 in f:
                fracs_all[pmt.index] = f[0]
                eff += f[0]
        return [Zone(
            zone_id=0, area_name=area_name,
            boundary_low=r_min, boundary_high=r_max,
            area_mm2=total_area, pmt_fractions=fracs_all,
            effective_n_pmts=eff,
        )], bounds

    n_total = len(pmts)
    total_area = np.pi * (r_max**2 - r_min**2)
    target_density = n_total / total_area

    # Initialize with equal-area boundaries
    inner_bounds = []
    for i in range(1, n_zones):
        r = np.sqrt(r_min**2 + i / n_zones * (r_max**2 - r_min**2))
        inner_bounds.append(r)
    inner_bounds = np.array(inner_bounds, dtype=np.float64)

    def evaluate(inner_b):
        """Compute zones and return densities for given inner boundaries."""
        bounds = [r_min] + list(inner_b) + [r_max]
        zones = []
        for i in range(n_zones):
            area = np.pi * (bounds[i+1]**2 - bounds[i]**2)
            zones.append({'area': area, 'eff': 0.0, 'fracs': {}})

        for pmt in pmts:
            fracs = compute_radial_fractions(pmt, bounds)
            for zid, frac in fracs.items():
                if 0 <= zid < n_zones:
                    zones[zid]['eff'] += frac
                    zones[zid]['fracs'][pmt.index] = frac

        densities = [z['eff'] / z['area'] if z['area'] > 0 else 0.0 for z in zones]
        return zones, densities, bounds

    # Iterative optimization
    best_bounds = inner_bounds.copy()
    best_max_dev = float('inf')

    for iteration in range(max_iter):
        zones_data, densities, bounds = evaluate(inner_bounds)
        mean_d = np.mean(densities)
        if mean_d == 0:
            break
        devs = [abs(d - mean_d) / mean_d for d in densities]
        max_dev = max(devs)

        if max_dev < best_max_dev:
            best_max_dev = max_dev
            best_bounds = inner_bounds.copy()

        if max_dev < tol:
            break

        # Shift boundaries: if zone i has higher density than zone i+1,
        # move boundary i outward (increase r) to give zone i more area
        # and zone i+1 less area.
        new_bounds = inner_bounds.copy()
        for b_idx in range(len(inner_bounds)):
            d_left = densities[b_idx]
            d_right = densities[b_idx + 1]

            if mean_d == 0:
                continue

            # Proportional shift based on density imbalance
            imbalance = (d_left - d_right) / mean_d
            # Step size decreasing with iteration
            step = 0.3 * (1.0 - iteration / max_iter)
            # Shift in r²-space for area-proportional adjustment
            r_current = inner_bounds[b_idx]
            r2_range = r_max**2 - r_min**2
            delta_r2 = imbalance * step * r2_range / n_zones

            new_r2 = r_current**2 + delta_r2
            # Clamp to valid range
            r_low = r_min if b_idx == 0 else inner_bounds[b_idx - 1]
            r_high = r_max if b_idx == len(inner_bounds) - 1 else inner_bounds[b_idx + 1]
            # Keep minimum zone width of 50mm
            new_r2 = np.clip(new_r2, (r_low + 50)**2, (r_high - 50)**2)
            new_bounds[b_idx] = np.sqrt(new_r2)

        inner_bounds = new_bounds

    # Use best result
    zones_data, densities, bounds = evaluate(best_bounds)
    mean_d = np.mean(densities)

    zones = []
    for i in range(n_zones):
        zones.append(Zone(
            zone_id=i, area_name=area_name,
            boundary_low=bounds[i], boundary_high=bounds[i+1],
            area_mm2=zones_data[i]['area'],
            pmt_fractions=zones_data[i]['fracs'],
            effective_n_pmts=zones_data[i]['eff'],
        ))

    max_dev_pct = max(abs(d - mean_d) / mean_d * 100 for d in densities) if mean_d > 0 else 0
    print(f"  {area_name}: {n_zones} optimized zones, "
          f"target density = {target_density:.6e} PMTs/mm²")
    for z in zones:
        dev = abs(z.pmt_density - mean_d) / mean_d * 100 if mean_d > 0 else 0
        print(f"    Zone {z.zone_id}: r=[{z.boundary_low:.1f}, {z.boundary_high:.1f}]mm, "
              f"eff_PMTs={z.effective_n_pmts:.2f}, area={z.area_mm2:.0f}mm², "
              f"density={z.pmt_density:.6e} (dev={dev:.1f}%)")
    print(f"    Max density deviation: {max_dev_pct:.2f}% (after {min(iteration+1, max_iter)} iterations)")

    return zones, bounds


def build_z_zones(
    pmts: List[PMTInfo],
    n_zones: int,
    z_min: float,
    z_max: float,
    area_name: str,
    r_zylinder: float,
    max_iter: int = 200,
    tol: float = 0.001,
) -> Tuple[List[Zone], List[float]]:
    """
    Build z-zones for the wall with optimized boundaries such that
    effective PMT density is uniform across zones.

    Uses iterative optimization starting from equal-height boundaries.
    """
    circumference = 2 * np.pi * r_zylinder

    if n_zones == 1:
        total_area = circumference * (z_max - z_min)
        fracs_all = {}
        eff = 0.0
        bounds = [z_min, z_max]
        for pmt in pmts:
            f = compute_z_fractions(pmt, bounds)
            if 0 in f:
                fracs_all[pmt.index] = f[0]
                eff += f[0]
        return [Zone(
            zone_id=0, area_name=area_name,
            boundary_low=z_min, boundary_high=z_max,
            area_mm2=total_area, pmt_fractions=fracs_all,
            effective_n_pmts=eff,
        )], bounds

    n_total = len(pmts)
    total_area = circumference * (z_max - z_min)
    target_density = n_total / total_area

    # Initialize with equal-height boundaries
    dz = (z_max - z_min) / n_zones
    inner_bounds = np.array([z_min + i * dz for i in range(1, n_zones)], dtype=np.float64)

    def evaluate(inner_b):
        bounds = [z_min] + list(inner_b) + [z_max]
        zones = []
        for i in range(n_zones):
            height = bounds[i+1] - bounds[i]
            area = circumference * height
            zones.append({'area': area, 'eff': 0.0, 'fracs': {}})

        for pmt in pmts:
            fracs = compute_z_fractions(pmt, bounds)
            for zid, frac in fracs.items():
                if 0 <= zid < n_zones:
                    zones[zid]['eff'] += frac
                    zones[zid]['fracs'][pmt.index] = frac

        densities = [z['eff'] / z['area'] if z['area'] > 0 else 0.0 for z in zones]
        return zones, densities, bounds

    best_bounds = inner_bounds.copy()
    best_max_dev = float('inf')

    for iteration in range(max_iter):
        zones_data, densities, bounds = evaluate(inner_bounds)
        mean_d = np.mean(densities)
        if mean_d == 0:
            break
        devs = [abs(d - mean_d) / mean_d for d in densities]
        max_dev = max(devs)

        if max_dev < best_max_dev:
            best_max_dev = max_dev
            best_bounds = inner_bounds.copy()

        if max_dev < tol:
            break

        new_bounds = inner_bounds.copy()
        for b_idx in range(len(inner_bounds)):
            d_left = densities[b_idx]
            d_right = densities[b_idx + 1]

            if mean_d == 0:
                continue

            imbalance = (d_left - d_right) / mean_d
            step = 0.3 * (1.0 - iteration / max_iter)
            z_range = z_max - z_min
            delta_z = imbalance * step * z_range / n_zones

            new_z = inner_bounds[b_idx] + delta_z
            z_low = z_min if b_idx == 0 else inner_bounds[b_idx - 1]
            z_high = z_max if b_idx == len(inner_bounds) - 1 else inner_bounds[b_idx + 1]
            # Minimum zone height of 100mm
            new_z = np.clip(new_z, z_low + 100, z_high - 100)
            new_bounds[b_idx] = new_z

        inner_bounds = new_bounds

    # Use best result
    zones_data, densities, bounds = evaluate(best_bounds)
    mean_d = np.mean(densities)

    zones = []
    for i in range(n_zones):
        zones.append(Zone(
            zone_id=i, area_name=area_name,
            boundary_low=bounds[i], boundary_high=bounds[i+1],
            area_mm2=zones_data[i]['area'],
            pmt_fractions=zones_data[i]['fracs'],
            effective_n_pmts=zones_data[i]['eff'],
        ))

    max_dev_pct = max(abs(d - mean_d) / mean_d * 100 for d in densities) if mean_d > 0 else 0
    print(f"  {area_name}: {n_zones} optimized zones, "
          f"target density = {target_density:.6e} PMTs/mm²")
    for z in zones:
        dev = abs(z.pmt_density - mean_d) / mean_d * 100 if mean_d > 0 else 0
        print(f"    Zone {z.zone_id}: z=[{z.boundary_low:.1f}, {z.boundary_high:.1f}]mm, "
              f"eff_PMTs={z.effective_n_pmts:.2f}, area={z.area_mm2:.0f}mm², "
              f"density={z.pmt_density:.6e} (dev={dev:.1f}%)")
    print(f"    Max density deviation: {max_dev_pct:.2f}% (after {min(iteration+1, max_iter)} iterations)")

    return zones, bounds


# =============================================================================
# MEMORY & NC DATA
# =============================================================================

def get_chunk_size() -> int:
    """Determine optimal chunk size based on available memory."""
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


def load_nc_data_dict_from_csv(csv_path: Path) -> Dict[Tuple[int, int], Dict]:
    """Load neutron capture event data from merged_ncs.csv.
    
    Key: (muon_id, nc_id) → {'nC_time': float}
    Matches against hit/optical/muon_track_id and hit/optical/nC_track_id.
    """
    nc_data_dict = {}
    with open(csv_path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            muon_id = int(parts[0])
            nc_id = int(parts[1])
            nc_time = float(parts[5])
            key = (muon_id, nc_id)
            nc_data_dict[key] = {'nC_time': nc_time}
    return nc_data_dict


# =============================================================================
# MOMENTUM FILTER
# =============================================================================

def checkRadialMomentumVectorized(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    px: np.ndarray, py: np.ndarray, pz: np.ndarray
) -> np.ndarray:
    """Check if momentum vector points outward for barrel region."""
    return (x * px + y * py) >= 0


# =============================================================================
# SSD UID CONSTANTS
# =============================================================================

SSD_UID_PIT = 1966
SSD_UID_BOT = 1967
SSD_UID_TOP = 1968
SSD_UID_WALL = 1965


# =============================================================================
# ZONE ASSIGNMENT HELPERS
# =============================================================================

def assign_radial_zone(r: np.ndarray, boundaries: List[float]) -> np.ndarray:
    """Assign radial positions to zone IDs via searchsorted."""
    ids = np.searchsorted(boundaries, r, side='right') - 1
    n_zones = len(boundaries) - 1
    ids = np.clip(ids, 0, n_zones - 1)
    outside = (r < boundaries[0]) | (r > boundaries[-1])
    ids[outside] = -1
    return ids


def assign_z_zone(z: np.ndarray, boundaries: List[float]) -> np.ndarray:
    """Assign z positions to zone IDs via searchsorted."""
    ids = np.searchsorted(boundaries, z, side='right') - 1
    n_zones = len(boundaries) - 1
    ids = np.clip(ids, 0, n_zones - 1)
    outside = (z < boundaries[0]) | (z > boundaries[-1])
    ids[outside] = -1
    return ids


# =============================================================================
# FILE PROCESSING: SSD
# =============================================================================

def process_ssd_file(
    file_path: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    nc_data_dict: Dict[Tuple[int, int], Dict],
    pit_boundaries: List[float],
    top_boundaries: List[float],
    wall_boundaries: List[float],
    bot_boundaries: List[float],
    n_pit: int, n_top: int, n_wall: int, n_bot: int,
) -> Dict[str, int]:
    """Process SSD file, count photons per zone based on hit position."""
    counts: Dict[str, int] = {}
    for area, n in [('pit', n_pit), ('top', n_top), ('wall', n_wall), ('bot', n_bot)]:
        for i in range(n):
            counts[f"{area}_{i}"] = 0

    try:
        with h5py.File(file_path, 'r') as f:
            total = len(f['hit']['optical']['x_position_in_m']['pages'])
            num_chunks = (total - 1) // chunk_size + 1
            z_cut_bot = geometry.z_cut_bot
            z_cut_top = geometry.z_cut_top

            for chunk_idx in range(num_chunks):
                cs = chunk_idx * chunk_size
                ce = min(cs + chunk_size, total)

                x = np.array(f['hit']['optical']['x_position_in_m']['pages'][cs:ce], dtype=np.float32) * 1000
                y = np.array(f['hit']['optical']['y_position_in_m']['pages'][cs:ce], dtype=np.float32) * 1000
                z = np.array(f['hit']['optical']['z_position_in_m']['pages'][cs:ce], dtype=np.float32) * 1000
                px = np.array(f['hit']['optical']['x_momentum_direction']['pages'][cs:ce], dtype=np.float32)
                py = np.array(f['hit']['optical']['y_momentum_direction']['pages'][cs:ce], dtype=np.float32)
                pz = np.array(f['hit']['optical']['z_momentum_direction']['pages'][cs:ce], dtype=np.float32)
                muon_ids = f['hit']['optical']['muon_track_id']['pages'][cs:ce]
                nc_track_ids = f['hit']['optical']['nC_track_id']['pages'][cs:ce]
                time = f['hit']['optical']['time_in_ns']['pages'][cs:ce]
                det_uid = f['hit']['optical']['det_uid']['pages'][cs:ce]

                # Filter: NC time window
                nc_times = np.full(len(time), np.inf, dtype=np.float32)
                for idx in range(len(muon_ids)):
                    key = (int(muon_ids[idx]), int(nc_track_ids[idx]))
                    if key in nc_data_dict:
                        nc_times[idx] = nc_data_dict[key]['nC_time']
                time_mask = (nc_times != np.inf) & (time >= nc_times) & (time <= nc_times + 200.0)

                x = x[time_mask]; y = y[time_mask]; z = z[time_mask]
                px = px[time_mask]; py = py[time_mask]; pz = pz[time_mask]
                det_uid = det_uid[time_mask]

                # Filter: Momentum
                mask_bot = z <= z_cut_bot
                mask_top = z >= z_cut_top
                mask_barrel = ~mask_bot & ~mask_top

                final_mask = np.zeros_like(z, dtype=bool)
                final_mask[mask_bot] = pz[mask_bot] <= 0
                final_mask[mask_top] = pz[mask_top] >= 0
                if np.any(mask_barrel):
                    final_mask[mask_barrel] = checkRadialMomentumVectorized(
                        x[mask_barrel], y[mask_barrel], z[mask_barrel],
                        px[mask_barrel], py[mask_barrel], pz[mask_barrel]
                    )

                x_f = x[final_mask]; y_f = y[final_mask]; z_f = z[final_mask]
                uid_f = det_uid[final_mask]
                r_f = np.sqrt(x_f**2 + y_f**2)

                # Pit
                pit_mask = uid_f == SSD_UID_PIT
                if np.any(pit_mask):
                    zids = assign_radial_zone(r_f[pit_mask], pit_boundaries)
                    for zi in range(n_pit):
                        counts[f"pit_{zi}"] += int(np.sum(zids == zi))

                # Bot
                bot_mask = uid_f == SSD_UID_BOT
                if np.any(bot_mask):
                    zids = assign_radial_zone(r_f[bot_mask], bot_boundaries)
                    for zi in range(n_bot):
                        counts[f"bot_{zi}"] += int(np.sum(zids == zi))

                # Top
                top_mask = uid_f == SSD_UID_TOP
                if np.any(top_mask):
                    zids = assign_radial_zone(r_f[top_mask], top_boundaries)
                    for zi in range(n_top):
                        counts[f"top_{zi}"] += int(np.sum(zids == zi))

                # Wall
                wall_mask = uid_f == SSD_UID_WALL
                if np.any(wall_mask):
                    zids = assign_z_zone(z_f[wall_mask], wall_boundaries)
                    for zi in range(n_wall):
                        counts[f"wall_{zi}"] += int(np.sum(zids == zi))

                del x, y, z, px, py, pz, muon_ids, nc_track_ids, time, det_uid
                del nc_times, time_mask, mask_bot, mask_top, mask_barrel, final_mask
                del x_f, y_f, z_f, uid_f, r_f
                gc.collect()

    except Exception as e:
        print(f"Error processing SSD {file_path}: {e}")

    return counts


# =============================================================================
# FILE PROCESSING: PMT
# =============================================================================

def process_pmt_file(
    file_path: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    nc_data_dict: Dict[Tuple[int, int], Dict],
    uid_to_pmt: Dict[int, PMTInfo],
    zone_fractions: Dict[str, Dict[str, float]],
    all_zone_keys: List[str],
) -> Tuple[Dict[str, float], set]:
    """
    Process PMT file. Count photons per PMT UID, then distribute
    fractionally across zones based on precomputed overlap fractions.
    """
    pmt_photon_counts: Dict[int, int] = {}
    observed_uids: set = set()

    try:
        with h5py.File(file_path, 'r') as f:
            total = len(f['hit']['optical']['x_position_in_m']['pages'])
            num_chunks = (total - 1) // chunk_size + 1

            for chunk_idx in range(num_chunks):
                cs = chunk_idx * chunk_size
                ce = min(cs + chunk_size, total)

                time = f['hit']['optical']['time_in_ns']['pages'][cs:ce]
                det_uid = f['hit']['optical']['det_uid']['pages'][cs:ce]
                muon_ids = f['hit']['optical']['muon_track_id']['pages'][cs:ce]
                nc_track_ids = f['hit']['optical']['nC_track_id']['pages'][cs:ce]

                # Filter: NC time window
                nc_times = np.full(len(time), np.inf, dtype=np.float32)
                for idx in range(len(muon_ids)):
                    key = (int(muon_ids[idx]), int(nc_track_ids[idx]))
                    if key in nc_data_dict:
                        nc_times[idx] = nc_data_dict[key]['nC_time']
                time_mask = (nc_times != np.inf) & (time >= nc_times) & (time <= nc_times + 200.0)

                det_uid_filtered = det_uid[time_mask]

                # Filter: Valid PMT UIDs (8 digits starting with 1)
                pmt_mask = (det_uid_filtered >= 10000000) & (det_uid_filtered < 1000000000)
                det_uid_filtered = det_uid_filtered[pmt_mask]

                # Count per UID (vectorized)
                unique_uids, uid_counts = np.unique(det_uid_filtered, return_counts=True)
                for uid_val, cnt in zip(unique_uids, uid_counts):
                    uid_int = int(uid_val)
                    pmt_photon_counts[uid_int] = pmt_photon_counts.get(uid_int, 0) + int(cnt)
                    observed_uids.add(uid_int)

                del muon_ids, nc_track_ids, time, det_uid, nc_times, time_mask, det_uid_filtered
                gc.collect()

    except Exception as e:
        print(f"Error processing PMT {file_path}: {e}")
        return {k: 0.0 for k in all_zone_keys}, set()

    # Distribute photons fractionally across zones
    zone_counts: Dict[str, float] = {k: 0.0 for k in all_zone_keys}

    for uid_int, n_photons in pmt_photon_counts.items():
        if uid_int not in uid_to_pmt:
            continue
        pmt = uid_to_pmt[uid_int]
        for zone_key, pmt_fracs in zone_fractions.items():
            if pmt.index in pmt_fracs:
                zone_counts[zone_key] += n_photons * pmt_fracs[pmt.index]

    return zone_counts, observed_uids


# =============================================================================
# PROCESS ALL FILES
# =============================================================================

def process_all_files_ssd(
    base_path: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    pit_boundaries: List[float],
    top_boundaries: List[float],
    wall_boundaries: List[float],
    bot_boundaries: List[float],
    n_pit: int, n_top: int, n_wall: int, n_bot: int,
) -> Tuple[Dict[str, int], int, int]:
    """Process all SSD files."""
    all_keys = ([f"pit_{i}" for i in range(n_pit)] +
                [f"top_{i}" for i in range(n_top)] +
                [f"wall_{i}" for i in range(n_wall)] +
                [f"bot_{i}" for i in range(n_bot)])
    total_counts: Dict[str, int] = {k: 0 for k in all_keys}
    total_files = 0
    total_nc = 0

    run_dirs = sorted(base_path.glob("run_*"))
    print(f"Processing SSD setup: found {len(run_dirs)} runs")

    for run_idx, run_dir in enumerate(run_dirs, 1):
        print(f"  Run {run_idx}/{len(run_dirs)}: {run_dir.name}")
        hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))

        # Count NCs from merged_ncs.csv for this run
        nc_csv = run_dir / "merged_ncs.csv"
        nc_data_dict = load_nc_data_dict_from_csv(nc_csv)
        if nc_csv.exists():
            with open(nc_csv, 'r') as f_csv:
                total_nc += sum(1 for _ in f_csv) - 1  # minus header

        for file_idx, hdf5_file in enumerate(hdf5_files):
            if (file_idx + 1) % 50 == 0:
                print(f"    Processing file {file_idx + 1}/{len(hdf5_files)}")
            total_files += 1

            # NC count from CSV (counted once per run, not per file)

            file_counts = process_ssd_file(
                hdf5_file, geometry, chunk_size, nc_data_dict,
                pit_boundaries, top_boundaries, wall_boundaries, bot_boundaries,
                n_pit, n_top, n_wall, n_bot
            )
            for k, v in file_counts.items():
                total_counts[k] += v

        area_sums = {}
        for k, v in total_counts.items():
            area = k.split('_')[0]
            area_sums[area] = area_sums.get(area, 0) + v
        print(f"    Run {run_idx} complete: {area_sums}")

    return total_counts, total_files, total_nc


def process_all_files_pmt(
    base_path: Path,
    geometry: GeometryConfig,
    chunk_size: int,
    uid_to_pmt: Dict[int, PMTInfo],
    zone_fractions: Dict[str, Dict[str, float]],
    all_zone_keys: List[str],
) -> Tuple[Dict[str, float], int, int, set]:
    """Process all PMT files."""
    total_counts: Dict[str, float] = {k: 0.0 for k in all_zone_keys}
    total_files = 0
    total_nc = 0
    all_observed_uids: set = set()

    run_dirs = sorted(base_path.glob("run_*"))
    print(f"Processing PMT setup: found {len(run_dirs)} runs")

    for run_idx, run_dir in enumerate(run_dirs, 1):
        print(f"  Run {run_idx}/{len(run_dirs)}: {run_dir.name}")
        hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))

        # Count NCs from merged_ncs.csv for this run
        nc_csv = run_dir / "merged_ncs.csv"
        nc_data_dict = load_nc_data_dict_from_csv(nc_csv)
        if nc_csv.exists():
            with open(nc_csv, 'r') as f_csv:
                total_nc += sum(1 for _ in f_csv) - 1  # minus header

        for file_idx, hdf5_file in enumerate(hdf5_files):
            if (file_idx + 1) % 50 == 0:
                print(f"    Processing file {file_idx + 1}/{len(hdf5_files)}")
            total_files += 1

            file_counts, file_uids = process_pmt_file(
                hdf5_file, geometry, chunk_size, nc_data_dict,
                uid_to_pmt, zone_fractions, all_zone_keys
            )
            for k, v in file_counts.items():
                total_counts[k] += v
            all_observed_uids.update(file_uids)

        area_sums = {}
        for k, v in total_counts.items():
            area = k.split('_')[0]
            area_sums[area] = area_sums.get(area, 0) + v
        print(f"    Run {run_idx} complete: {area_sums}")

    return total_counts, total_files, total_nc, all_observed_uids


# =============================================================================
# PLOTTING
# =============================================================================

def plot_radial_zones(
    zones: List[Zone],
    pmts: List[PMTInfo],
    ratios: List[float],
    area_name: str,
    r_min: float,
    r_max: float,
    output_path: Path,
    global_norm: plt.Normalize = None,
):
    """Plot radial zones (pit, top, bot) as x-y view with PMTs and ratio heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    cmap = plt.cm.RdYlGn_r
    norm = global_norm if global_norm is not None else plt.Normalize(0, 1)

    for zone, ratio in zip(zones, ratios):
        width = zone.boundary_high - zone.boundary_low
        color = cmap(norm(ratio)) if not np.isnan(ratio) else 'grey'
        ring = mpatches.Annulus((0, 0), zone.boundary_high, width,
                                color=color, alpha=0.6)
        ax.add_patch(ring)

        r_mid = (zone.boundary_low + zone.boundary_high) / 2
        label = (f"Z{zone.zone_id}\n"
                 f"r=[{zone.boundary_low:.0f},{zone.boundary_high:.0f}]\n"
                 f"eff_PMTs={zone.effective_n_pmts:.1f}\n")
        label += f"ratio={ratio:.3f}" if not np.isnan(ratio) else "ratio=NaN"
        ax.text(0, r_mid, label, ha='center', va='center', fontsize=8,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    for pmt in pmts:
        circle = plt.Circle((pmt.center[0], pmt.center[1]), PMT_RADIUS,
                            fill=False, edgecolor='blue', linewidth=0.5, alpha=0.7)
        ax.add_patch(circle)
        ax.plot(pmt.center[0], pmt.center[1], 'b.', markersize=1)

    for zone in zones:
        for r_val in [zone.boundary_low, zone.boundary_high]:
            if r_val > 0:
                c = plt.Circle((0, 0), r_val, fill=False,
                              edgecolor='black', linewidth=1.5, linestyle='--')
                ax.add_patch(c)

    ax.set_xlim(-r_max * 1.15, r_max * 1.15)
    ax.set_ylim(-r_max * 1.15, r_max * 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'{area_name.upper()} - SSD/PMT Ratio per Zone')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, label='SSD/PMT Ratio')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


def plot_wall_zones(
    zones: List[Zone],
    pmts: List[PMTInfo],
    ratios: List[float],
    r_zylinder: float,
    output_path: Path,
    global_norm: plt.Normalize = None,
):
    """Plot wall zones as phi-z view with PMTs and ratio heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    cmap = plt.cm.RdYlGn_r
    norm = global_norm if global_norm is not None else plt.Normalize(0, 1)

    for zone, ratio in zip(zones, ratios):
        color = cmap(norm(ratio)) if not np.isnan(ratio) else 'grey'
        ax.axhspan(zone.boundary_low, zone.boundary_high, alpha=0.5, color=color)

        z_mid = (zone.boundary_low + zone.boundary_high) / 2
        label = (f"Z{zone.zone_id}: z=[{zone.boundary_low:.0f},{zone.boundary_high:.0f}]\n"
                 f"eff_PMTs={zone.effective_n_pmts:.1f}\n")
        label += f"ratio={ratio:.3f}" if not np.isnan(ratio) else "ratio=NaN"
        ax.text(0, z_mid, label, ha='center', va='center', fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    for zone in zones:
        ax.axhline(zone.boundary_low, color='black', linewidth=1.5, linestyle='--')
        ax.axhline(zone.boundary_high, color='black', linewidth=1.5, linestyle='--')

    for pmt in pmts:
        phi = np.arctan2(pmt.center[1], pmt.center[0])
        ax.plot(phi, pmt.z, 'bo', markersize=3, alpha=0.7)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel('φ [rad]')
    ax.set_ylabel('z [mm]')
    ax.set_title('WALL - SSD/PMT Ratio per Zone (φ-z view)')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, label='SSD/PMT Ratio')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")

def scan_optimal_zones(
    ssd_counts_by_n: Dict[int, Dict[str, int]],
    pmt_counts_by_n: Dict[int, Dict[str, float]],
    ssd_nc: int,
    pmt_nc: int,
    area_name: str,
    zone_configs: Dict[int, List[Zone]],
    snr_threshold: float = SNR_THRESHOLD,
) -> Tuple[int, Dict]:
    """
    Scan over different numbers of zones and find the optimal count.

    For each N_zones, computes:
    - corrected ratios per zone
    - σ_zone: std dev of corrected ratios (signal)
    - ⟨σ_R⟩: mean statistical uncertainty per zone (noise)
    - SNR = σ_zone / ⟨σ_R⟩

    Returns the largest N_zones where SNR > snr_threshold.

    Args:
        ssd_counts_by_n: {n_zones: {zone_key: ssd_photon_count}}
        pmt_counts_by_n: {n_zones: {zone_key: pmt_photon_count}}
        ssd_nc, pmt_nc: Total NC events
        area_name: Name of the area
        zone_configs: {n_zones: list of Zone objects}
        snr_threshold: Minimum SNR

    Returns:
        Optimal number of zones
    """
    print(f"\n  Zone scan for {area_name}:")
    print(f"  {'N_ZONES':>8} {'σ_zone':>10} {'⟨σ_R⟩':>10} {'SNR':>8} {'STATUS':>10}")
    print(f"  " + "-" * 50)

    best_n = 2  # minimum
    scan_data = {'n_zones': [], 'sigma_zone': [], 'mean_sigma_r': [], 'snr': []}
    consecutive_fails = 0  # abort after 2 consecutive failures

    for n_zones in sorted(zone_configs.keys()):
        zones = zone_configs[n_zones]
        ssd_c = ssd_counts_by_n[n_zones]
        pmt_c = pmt_counts_by_n[n_zones]

        # Check minimum PMTs per zone
        min_eff = min(z.effective_n_pmts for z in zones)
        if min_eff < MIN_PMTS_PER_ZONE:
            print(f"  {n_zones:>8} {'--':>10} {'--':>10} {'--':>8} "
                  f"{'<{0}PMTs'.format(MIN_PMTS_PER_ZONE):>10}")
            consecutive_fails += 1
            if consecutive_fails >= 2:
                print(f"  ⛔ Aborting: 2 consecutive failures at N={n_zones}")
                break
            continue

        # Compute mean density for this area
        total_eff = sum(z.effective_n_pmts for z in zones)
        total_area = sum(z.area_mm2 for z in zones)
        mean_density = total_eff / total_area if total_area > 0 else 0.0

        corr_ratios = []
        stat_uncertainties = []

        for zone in zones:
            key = f"{area_name}_{zone.zone_id}"
            n_ssd = ssd_c.get(key, 0)
            n_pmt = pmt_c.get(key, 0.0)

            if n_ssd <= 0 or n_pmt <= 0:
                continue

            ssd_per_nc = n_ssd / ssd_nc
            pmt_per_nc = n_pmt / pmt_nc
            raw_ratio = ssd_per_nc / pmt_per_nc

            # Density correction
            if mean_density > 0 and zone.pmt_density > 0:
                corr_ratio = raw_ratio * (zone.pmt_density / mean_density)
            else:
                continue

            # Statistical uncertainty: σ_R/R = sqrt(1/N_ssd + 1/N_pmt)
            rel_unc = np.sqrt(1.0 / n_ssd + 1.0 / n_pmt)
            abs_unc = corr_ratio * rel_unc

            corr_ratios.append(corr_ratio)
            stat_uncertainties.append(abs_unc)

        if len(corr_ratios) < 2:
            print(f"  {n_zones:>8} {'--':>10} {'--':>10} {'--':>8} {'skip':>10}")
            consecutive_fails += 1
            if consecutive_fails >= 2:
                print(f"  ⛔ Aborting: 2 consecutive failures at N={n_zones}")
                break
            continue

        sigma_zone = np.std(corr_ratios, ddof=1)
        mean_sigma_r = np.mean(stat_uncertainties)
        snr = sigma_zone / mean_sigma_r if mean_sigma_r > 0 else 0.0

        passed = snr >= snr_threshold
        status = "✅" if passed else "❌"
        print(f"  {n_zones:>8} {sigma_zone:>10.4f} {mean_sigma_r:>10.4f} {snr:>8.2f} {status:>10}")

        scan_data['n_zones'].append(n_zones)
        scan_data['sigma_zone'].append(sigma_zone)
        scan_data['mean_sigma_r'].append(mean_sigma_r)
        scan_data['snr'].append(snr)

        if passed:
            best_n = n_zones
            consecutive_fails = 0
        else:
            consecutive_fails += 1
            if consecutive_fails >= 2:
                print(f"  ⛔ Aborting: 2 consecutive failures at N={n_zones}")
                break

    print(f"  → Optimal zones for {area_name}: {best_n} (SNR ≥ {snr_threshold})")
    return best_n, scan_data

def plot_snr_scan(
    scan_results: Dict[str, Dict],
    snr_threshold: float,
    optimal_zones: Dict[str, int],
    output_path: Path,
):
    """
    Plot SNR vs N_zones for each area.
    Shows σ_zone, ⟨σ_R⟩, and SNR with threshold line.
    """
    areas = sorted(scan_results.keys())
    fig, axes = plt.subplots(len(areas), 2, figsize=(14, 4 * len(areas)),
                              squeeze=False)

    for row, area_name in enumerate(areas):
        data = scan_results[area_name]
        if not data['n_zones']:
            continue

        n_arr = np.array(data['n_zones'])
        sigma_zone = np.array(data['sigma_zone'])
        mean_sigma_r = np.array(data['mean_sigma_r'])
        snr = np.array(data['snr'])
        opt_n = optimal_zones.get(area_name, 2)

        # Left panel: σ_zone and ⟨σ_R⟩
        ax1 = axes[row, 0]
        ax1.plot(n_arr, sigma_zone, 'bo-', label='σ_zone (signal)', linewidth=2)
        ax1.plot(n_arr, mean_sigma_r, 'r^-', label='⟨σ_R⟩ (noise)', linewidth=2)
        ax1.axvline(opt_n, color='green', linestyle='--', linewidth=1.5,
                    label=f'optimal N={opt_n}')
        ax1.set_xlabel('N_zones')
        ax1.set_ylabel('Ratio units')
        ax1.set_title(f'{area_name.upper()} — Signal vs Noise')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(n_arr)

        # Right panel: SNR
        ax2 = axes[row, 1]
        colors = ['green' if s >= snr_threshold else 'red' for s in snr]
        ax2.bar(n_arr, snr, color=colors, alpha=0.7, width=0.6)
        ax2.axhline(snr_threshold, color='black', linestyle='--', linewidth=1.5,
                    label=f'threshold = {snr_threshold}')
        ax2.axvline(opt_n, color='green', linestyle='--', linewidth=1.5,
                    label=f'optimal N={opt_n}')
        ax2.set_xlabel('N_zones')
        ax2.set_ylabel('SNR')
        ax2.set_title(f'{area_name.upper()} — SNR = σ_zone / ⟨σ_R⟩')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(n_arr)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")

# =============================================================================
# REFERENCE LOADING & COMPARISON
# =============================================================================

def load_reference_json(json_path: Path) -> Dict:
    """Load reference zone results from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def build_zones_from_reference(
    ref_data: Dict,
    pmts: List[PMTInfo],
    pmt_by_layer: Dict[str, List[PMTInfo]],
    geometry: GeometryConfig,
) -> Tuple[
    List[Zone], List[Zone], List[Zone], List[Zone],
    List[float], List[float], List[float], List[float],
]:
    """
    Rebuild zones using boundaries from reference JSON.
    Recomputes PMT fractions for the current PMT positions.

    Returns:
        pit_zones, top_zones, wall_zones, bot_zones,
        pit_bounds, top_bounds, wall_bounds, bot_bounds
    """
    bounds = ref_data['boundaries']
    pit_bounds = bounds['pit']
    top_bounds = bounds['top']
    wall_bounds = bounds['wall']
    bot_bounds = bounds['bot']

    n_pit = len(pit_bounds) - 1
    n_top = len(top_bounds) - 1
    n_wall = len(wall_bounds) - 1

    # Build radial zones with fixed boundaries (no optimization)
    print("\n  Rebuilding zones from reference boundaries...")

    pit_zones = _build_fixed_radial_zones(
        pmt_by_layer['pit'], pit_bounds, "pit")
    top_zones = _build_fixed_radial_zones(
        pmt_by_layer['top'], top_bounds, "top")
    wall_zones = _build_fixed_z_zones(
        pmt_by_layer['wall'], wall_bounds, "wall", geometry.r_zylinder)

    bot_zones = [Zone(
        zone_id=0, area_name="bot",
        boundary_low=bot_bounds[0], boundary_high=bot_bounds[1],
        area_mm2=geometry.area_bot,
        pmt_fractions={pmt.index: 1.0 for pmt in pmt_by_layer['bot']},
        effective_n_pmts=float(len(pmt_by_layer['bot'])),
    )]
    print(f"    bot: 1 zone, {len(pmt_by_layer['bot'])} PMTs")

    return (pit_zones, top_zones, wall_zones, bot_zones,
            pit_bounds, top_bounds, wall_bounds, bot_bounds)


def _build_fixed_radial_zones(
    pmts: List[PMTInfo],
    boundaries: List[float],
    area_name: str,
) -> List[Zone]:
    """Build radial zones with fixed boundaries, computing PMT fractions."""
    n_zones = len(boundaries) - 1
    zones = []
    for i in range(n_zones):
        r_lo = boundaries[i]
        r_hi = boundaries[i + 1]
        area = np.pi * (r_hi**2 - r_lo**2)
        zones.append(Zone(
            zone_id=i, area_name=area_name,
            boundary_low=r_lo, boundary_high=r_hi,
            area_mm2=area, pmt_fractions={}, effective_n_pmts=0.0,
        ))

    for pmt in pmts:
        fracs = compute_radial_fractions(pmt, boundaries)
        for zid, frac in fracs.items():
            if 0 <= zid < n_zones:
                zones[zid].pmt_fractions[pmt.index] = frac
                zones[zid].effective_n_pmts += frac

    for z in zones:
        print(f"    {area_name} zone {z.zone_id}: "
              f"r=[{z.boundary_low:.1f}, {z.boundary_high:.1f}]mm, "
              f"eff_PMTs={z.effective_n_pmts:.2f}")
    return zones


def _build_fixed_z_zones(
    pmts: List[PMTInfo],
    boundaries: List[float],
    area_name: str,
    r_zylinder: float,
) -> List[Zone]:
    """Build z-zones with fixed boundaries, computing PMT fractions."""
    n_zones = len(boundaries) - 1
    circumference = 2 * np.pi * r_zylinder
    zones = []
    for i in range(n_zones):
        z_lo = boundaries[i]
        z_hi = boundaries[i + 1]
        area = circumference * (z_hi - z_lo)
        zones.append(Zone(
            zone_id=i, area_name=area_name,
            boundary_low=z_lo, boundary_high=z_hi,
            area_mm2=area, pmt_fractions={}, effective_n_pmts=0.0,
        ))

    for pmt in pmts:
        fracs = compute_z_fractions(pmt, boundaries)
        for zid, frac in fracs.items():
            if 0 <= zid < n_zones:
                zones[zid].pmt_fractions[pmt.index] = frac
                zones[zid].effective_n_pmts += frac

    for z in zones:
        print(f"    {area_name} zone {z.zone_id}: "
              f"z=[{z.boundary_low:.1f}, {z.boundary_high:.1f}]mm, "
              f"eff_PMTs={z.effective_n_pmts:.2f}")
    return zones


def plot_comparison(
    results_new: List[Dict],
    ref_data: Dict,
    output_dir: Path,
):
    """
    Plot comparison between new and reference corrected ratios.
    Two plots per area:
      1. Bar chart: new vs reference CORR_RATIO per zone
      2. Bar chart: absolute difference (new - ref) per zone
    """
    # Build reference lookup: (area, zone_id) → corr_ratio
    ref_lookup: Dict[Tuple[str, int], float] = {}
    for z in ref_data['zones']:
        cr = z['corr_ratio']
        ref_lookup[(z['area'], z['zone_id'])] = cr if cr is not None else float('nan')

    # Group new results by area
    areas_order = ['pit', 'bot', 'top', 'wall']
    results_by_area: Dict[str, List[Dict]] = {a: [] for a in areas_order}
    for r in results_new:
        area = r['zone'].area_name
        if area in results_by_area:
            results_by_area[area].append(r)

    for area_name in areas_order:
        area_results = sorted(results_by_area[area_name],
                              key=lambda r: r['zone'].zone_id)
        if not area_results:
            continue

        zone_ids = [r['zone'].zone_id for r in area_results]
        new_ratios = [r['corr_ratio'] for r in area_results]
        ref_ratios = [ref_lookup.get((area_name, zid), float('nan'))
                      for zid in zone_ids]
        diffs = [n - ref if not (np.isnan(n) or np.isnan(ref)) else float('nan')
                 for n, ref in zip(new_ratios, ref_ratios)]

        # Zone labels
        labels = []
        for r in area_results:
            z = r['zone']
            if area_name in ('pit', 'bot', 'top'):
                labels.append(f"Z{z.zone_id}\nr=[{z.boundary_low:.0f},\n{z.boundary_high:.0f}]")
            else:
                labels.append(f"Z{z.zone_id}\nz=[{z.boundary_low:.0f},\n{z.boundary_high:.0f}]")

        x = np.arange(len(zone_ids))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(zone_ids) * 1.5), 6))

        # Left: side-by-side bars
        bars_ref = ax1.bar(x - width/2, ref_ratios, width, label='Reference',
                           color='steelblue', alpha=0.8)
        bars_new = ax1.bar(x + width/2, new_ratios, width, label='Musun NCs',
                           color='coral', alpha=0.8)
        ax1.set_xlabel('Zone')
        ax1.set_ylabel('Corrected SSD/PMT Ratio')
        ax1.set_title(f'{area_name.upper()} — Corrected Ratio: Reference vs Musun NCs')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=7)
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)

        # Right: difference bars
        colors = ['green' if d >= 0 else 'red' for d in diffs]
        colors = ['grey' if np.isnan(d) else c for d, c in zip(diffs, colors)]
        diffs_plot = [d if not np.isnan(d) else 0 for d in diffs]
        ax2.bar(x, diffs_plot, width=0.5, color=colors, alpha=0.8)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Zone')
        ax2.set_ylabel('Δ Corrected Ratio (new − ref)')
        ax2.set_title(f'{area_name.upper()} — Absolute Difference per Zone')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=7)
        ax2.grid(True, axis='y', alpha=0.3)

        # Annotate differences
        for i, d in enumerate(diffs):
            if not np.isnan(d):
                ax2.text(i, diffs_plot[i], f"{d:+.4f}", ha='center',
                         va='bottom' if d >= 0 else 'top', fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{area_name}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Comparison plot saved: comparison_{area_name}.png")

    # Summary plot: all zones in one figure
    all_new = []
    all_ref = []
    all_labels = []
    for area_name in areas_order:
        area_results = sorted(results_by_area[area_name],
                              key=lambda r: r['zone'].zone_id)
        for r in area_results:
            z = r['zone']
            all_new.append(r['corr_ratio'])
            all_ref.append(ref_lookup.get((area_name, z.zone_id), float('nan')))
            all_labels.append(f"{area_name[0].upper()}{z.zone_id}")

    all_diffs = [n - ref if not (np.isnan(n) or np.isnan(ref)) else float('nan')
                 for n, ref in zip(all_new, all_ref)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(all_labels) * 0.6), 10))

    x = np.arange(len(all_labels))

    ax1.bar(x - 0.2, all_ref, 0.4, label='Reference', color='steelblue', alpha=0.8)
    ax1.bar(x + 0.2, all_new, 0.4, label='Musun NCs', color='coral', alpha=0.8)
    ax1.set_ylabel('Corrected SSD/PMT Ratio')
    ax1.set_title('All Zones — Corrected Ratio: Reference vs Musun NCs')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, fontsize=7, rotation=45)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    colors = ['green' if not np.isnan(d) and d >= 0
              else 'red' if not np.isnan(d)
              else 'grey' for d in all_diffs]
    diffs_plot = [d if not np.isnan(d) else 0 for d in all_diffs]
    ax2.bar(x, diffs_plot, 0.5, color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('Δ Corrected Ratio (new − ref)')
    ax2.set_title('All Zones — Absolute Difference')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, fontsize=7, rotation=45)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_all_zones.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary comparison plot saved: comparison_all_zones.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis workflow."""
    geometry = GeometryConfig()
    chunk_size = get_chunk_size()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "zone_ratio_results.txt"

    print("=" * 80)
    print("Zone-based SSD vs. PMT Photon Detection Efficiency Analysis")
    print("=" * 80)
    print(f"Geometry: {geometry.geometry_name}")
    print(f"Chunk size: {chunk_size}")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Zone config: Pit={N_ZONES_PIT}, Wall={N_ZONES_WALL}, Top={N_ZONES_TOP}, Bot=1")
    print(f"Zone scan areas: {ZONE_SCAN_AREAS if ZONE_SCAN_AREAS else 'none (manual zones)'}")
    print(f"Min PMTs per zone: {MIN_PMTS_PER_ZONE}")
    print(f"PMT cathode radius: {PMT_RADIUS} mm")
    print(f"MC samples for overlap: {MC_SAMPLES}")
    print("=" * 80)

    # Load PMT positions
    print("\nLoading PMT positions...")
    pmts = load_pmt_data(PMT_JSON_PATH)
    pmt_by_layer = get_pmts_by_layer(pmts)
    uid_to_pmt = build_uid_to_pmt_map(pmts)

    print(f"  Total PMTs: {len(pmts)}, UID mappings: {len(uid_to_pmt)}")
    for layer, layer_pmts in pmt_by_layer.items():
        print(f"  {layer}: {len(layer_pmts)} PMTs")

    # =========================================================================
    # COMPARE MODE: Load reference boundaries, skip optimization
    # =========================================================================
    if COMPARE_MODE:
        print(f"\n📊 COMPARE MODE: Loading reference from {REFERENCE_JSON}")
        ref_data = load_reference_json(REFERENCE_JSON)

        (pit_zones, top_zones, wall_zones, bot_zones,
         pit_bounds, top_bounds, wall_bounds, bot_bounds
        ) = build_zones_from_reference(ref_data, pmts, pmt_by_layer, geometry)

    else:
        # Build zones (original optimization logic)
        print("\nComputing zone boundaries with fractional PMT assignment...")
        pit_zones, pit_bounds = build_radial_zones(
            pmt_by_layer['pit'], N_ZONES_PIT, 0.0, geometry.r_pit, "pit")

        top_zones, top_bounds = build_radial_zones(
            pmt_by_layer['top'], N_ZONES_TOP, geometry.r_zyl_top, geometry.r_zylinder, "top")

        wall_zones, wall_bounds = build_z_zones(
            pmt_by_layer['wall'], N_ZONES_WALL,
            geometry.z_cut_bot, geometry.z_cut_top, "wall", geometry.r_zylinder)

        # Bot: single zone
        bot_bounds = [geometry.r_zyl_bot, geometry.r_zylinder]
        bot_zones = [Zone(
            zone_id=0, area_name="bot",
            boundary_low=geometry.r_zyl_bot, boundary_high=geometry.r_zylinder,
            area_mm2=geometry.area_bot,
            pmt_fractions={pmt.index: 1.0 for pmt in pmt_by_layer['bot']},
            effective_n_pmts=float(len(pmt_by_layer['bot'])),
        )]
        print(f"  bot: 1 zone, {len(pmt_by_layer['bot'])} PMTs, "
            f"area={geometry.area_bot:.0f}mm²")

    # Build zone_fractions lookup
    all_zones = pit_zones + top_zones + wall_zones + bot_zones
    zone_fractions: Dict[str, Dict[str, float]] = {}
    for z in all_zones:
        key = f"{z.area_name}_{z.zone_id}"
        zone_fractions[key] = z.pmt_fractions if z.pmt_fractions else {}
    all_zone_keys = [f"{z.area_name}_{z.zone_id}" for z in all_zones]

    # Validate: fraction sums per PMT ≈ 1.0
    print("\nValidating fractional PMT assignment...")
    pmt_frac_sums: Dict[str, float] = {}
    for zone_key, fracs in zone_fractions.items():
        for pmt_idx, frac in fracs.items():
            pmt_frac_sums[pmt_idx] = pmt_frac_sums.get(pmt_idx, 0.0) + frac

    # Check PMTs not assigned to any zone
    all_pmt_indices = {pmt.index for pmt in pmts}
    unassigned = all_pmt_indices - set(pmt_frac_sums.keys())
    if unassigned:
        print(f"  ⚠️  {len(unassigned)} PMTs not assigned to any zone: "
              f"{sorted(unassigned)[:5]}...")

    bad_fracs = {idx: s for idx, s in pmt_frac_sums.items() if abs(s - 1.0) > FRACTION_TOLERANCE}
    if bad_fracs:
        print(f"  ⚠️  {len(bad_fracs)} PMTs with fraction sum != 1.0 "
              f"(>{FRACTION_TOLERANCE*100:.0f}% off):")
        for idx, s in sorted(bad_fracs.items(), key=lambda x: abs(x[1]-1.0), reverse=True)[:5]:
            print(f"    PMT {idx}: sum = {s:.4f}")
    else:
        n_assigned = len(pmt_frac_sums)
        print(f"  ✅ All {n_assigned} assigned PMTs have fraction sums within "
              f"{FRACTION_TOLERANCE*100:.0f}% of 1.0")

    # =========================================================================
    # ZONE SCAN MODE: find optimal zone count per area
    # =========================================================================
    if ZONE_SCAN_AREAS and not COMPARE_MODE:
        print("\n" + "=" * 80)
        print(f"ZONE SCAN: Finding optimal number of zones for {ZONE_SCAN_AREAS}")
        print("=" * 80)

        all_scan_areas = {
            'pit': {'r_min': 0.0, 'r_max': geometry.r_pit, 'type': 'radial',
                    'pmts': pmt_by_layer['pit']},
            'top': {'r_min': geometry.r_zyl_top, 'r_max': geometry.r_zylinder,
                    'type': 'radial', 'pmts': pmt_by_layer['top']},
            'wall': {'z_min': geometry.z_cut_bot, 'z_max': geometry.z_cut_top,
                     'type': 'z', 'pmts': pmt_by_layer['wall']},
        }
        scan_areas = {k: v for k, v in all_scan_areas.items() if k in ZONE_SCAN_AREAS}

        optimal_zones = {}
        scan_results: Dict[str, Dict] = {}
        bot_ssd_counts_scan: Dict[str, int] = {"bot_0": 0}
        bot_pmt_counts_scan: Dict[str, float] = {"bot_0": 0.0}
        bot_counted = False

        for area_name, cfg in scan_areas.items():
            # Build zone configs for each N
            zone_configs: Dict[int, List[Zone]] = {}
            bounds_configs: Dict[int, List[float]] = {}
            frac_configs: Dict[int, Dict[str, Dict[str, float]]] = {}

            # Build zone configs lazily - check PMT constraint before expensive optimization
            for n in ZONE_SCAN_RANGE:
                # Quick pre-check: can n zones have ≥ MIN_PMTS_PER_ZONE each?
                n_pmts = len(cfg['pmts'])
                if n_pmts / n < MIN_PMTS_PER_ZONE:
                    print(f"    Skipping N={n}+ for {area_name}: "
                          f"{n_pmts} PMTs / {n} zones < {MIN_PMTS_PER_ZONE}")
                    break

                if cfg['type'] == 'radial':
                    zones_n, bounds_n = build_radial_zones(
                        cfg['pmts'], n, cfg['r_min'], cfg['r_max'], area_name)
                else:
                    zones_n, bounds_n = build_z_zones(
                        cfg['pmts'], n, cfg['z_min'], cfg['z_max'],
                        area_name, geometry.r_zylinder)
                zone_configs[n] = zones_n
                bounds_configs[n] = bounds_n
                frac_configs[n] = {f"{area_name}_{z.zone_id}": z.pmt_fractions
                                   for z in zones_n}

            # Process SSD files for this area with all zone configs
            print(f"\n  Scanning {area_name}: processing SSD files...")
            ssd_counts_by_n: Dict[int, Dict[str, int]] = {
                n: {f"{area_name}_{i}": 0 for i in range(n)}
                for n in zone_configs.keys()
            }

            ssd_scan_nc = 0
            run_dirs = sorted(SSD_DIR.glob("run_*"))
            for run_dir in run_dirs:
                nc_csv = run_dir / "merged_ncs.csv"
                if not nc_csv.exists():
                    print(f"    ⚠️ No merged_ncs.csv in {run_dir.name}, skipping")
                    continue
                with open(nc_csv, 'r') as f_csv:
                    ssd_scan_nc += sum(1 for _ in f_csv) - 1
                nc_data_dict = load_nc_data_dict_from_csv(nc_csv)

                hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))
                for hdf5_file in hdf5_files:
                    try:
                        with h5py.File(hdf5_file, 'r') as f:      
                            total = len(f['hit']['optical']['x_position_in_m']['pages'])
                            num_chunks = (total - 1) // chunk_size + 1

                            for chunk_idx in range(num_chunks):
                                cs = chunk_idx * chunk_size
                                ce = min(cs + chunk_size, total)

                                x = np.array(f['hit']['optical']['x_position_in_m']['pages'][cs:ce], dtype=np.float32) * 1000
                                y = np.array(f['hit']['optical']['y_position_in_m']['pages'][cs:ce], dtype=np.float32) * 1000
                                z = np.array(f['hit']['optical']['z_position_in_m']['pages'][cs:ce], dtype=np.float32) * 1000
                                px = np.array(f['hit']['optical']['x_momentum_direction']['pages'][cs:ce], dtype=np.float32)
                                py = np.array(f['hit']['optical']['y_momentum_direction']['pages'][cs:ce], dtype=np.float32)
                                pz_arr = np.array(f['hit']['optical']['z_momentum_direction']['pages'][cs:ce], dtype=np.float32)
                                muon_ids = f['hit']['optical']['muon_track_id']['pages'][cs:ce]
                                nc_track_ids = f['hit']['optical']['nC_track_id']['pages'][cs:ce]
                                time = f['hit']['optical']['time_in_ns']['pages'][cs:ce]
                                det_uid = f['hit']['optical']['det_uid']['pages'][cs:ce]

                                nc_times = np.full(len(time), np.inf, dtype=np.float32)
                                for idx in range(len(muon_ids)):
                                    key = (int(muon_ids[idx]), int(nc_track_ids[idx]))
                                    if key in nc_data_dict:
                                        nc_times[idx] = nc_data_dict[key]['nC_time']
                                time_mask = (nc_times != np.inf) & (time >= nc_times) & (time <= nc_times + 200.0)

                                x = x[time_mask]; y = y[time_mask]; z = z[time_mask]
                                px = px[time_mask]; py = py[time_mask]; pz_arr = pz_arr[time_mask]
                                det_uid = det_uid[time_mask]

                                z_cut_bot = geometry.z_cut_bot
                                z_cut_top = geometry.z_cut_top
                                mask_bot = z <= z_cut_bot
                                mask_top = z >= z_cut_top
                                mask_barrel = ~mask_bot & ~mask_top
                                final_mask = np.zeros_like(z, dtype=bool)
                                final_mask[mask_bot] = pz_arr[mask_bot] <= 0
                                final_mask[mask_top] = pz_arr[mask_top] >= 0
                                if np.any(mask_barrel):
                                    final_mask[mask_barrel] = checkRadialMomentumVectorized(
                                        x[mask_barrel], y[mask_barrel], z[mask_barrel],
                                        px[mask_barrel], py[mask_barrel], pz_arr[mask_barrel])

                                x_f = x[final_mask]; y_f = y[final_mask]; z_f = z[final_mask]
                                uid_f = det_uid[final_mask]

                                # Select only this area's SSD UID
                                if area_name == 'pit':
                                    area_mask = uid_f == SSD_UID_PIT
                                    coord = np.sqrt(x_f[area_mask]**2 + y_f[area_mask]**2)
                                elif area_name == 'top':
                                    area_mask = uid_f == SSD_UID_TOP
                                    coord = np.sqrt(x_f[area_mask]**2 + y_f[area_mask]**2)
                                elif area_name == 'wall':
                                    area_mask = uid_f == SSD_UID_WALL
                                    coord = z_f[area_mask]
                                else:
                                    continue

                                if not np.any(area_mask):
                                    continue

                                # Assign to all zone configs
                                for n in zone_configs.keys():
                                    if cfg['type'] == 'radial':
                                        zids = assign_radial_zone(coord, bounds_configs[n])
                                    else:
                                        zids = assign_z_zone(coord, bounds_configs[n])
                                    for zi in range(n):
                                        ssd_counts_by_n[n][f"{area_name}_{zi}"] += int(np.sum(zids == zi))

                                # Count bot photons (once, during first area scan)
                                if not bot_counted:
                                    bot_mask = uid_f == SSD_UID_BOT
                                    if np.any(bot_mask):
                                        r_bot = np.sqrt(x_f[bot_mask]**2 + y_f[bot_mask]**2)
                                        zids_bot = assign_radial_zone(r_bot, bot_bounds)
                                        bot_ssd_counts_scan["bot_0"] += int(np.sum(zids_bot == 0))

                                del x, y, z, px, py, pz_arr, muon_ids, nc_track_ids, time, det_uid
                                del nc_times, time_mask, final_mask, x_f, y_f, z_f, uid_f
                                gc.collect()
                    except Exception as e:
                        print(f"    Error: {e}")

            # Process PMT files for this area
            print(f"  Scanning {area_name}: processing PMT files...")
            pmt_counts_by_n: Dict[int, Dict[str, float]] = {
                n: {f"{area_name}_{i}": 0.0 for i in range(n)}
                for n in zone_configs.keys()
            }

            pmt_scan_nc = 0
            run_dirs = sorted(PMT_DIR.glob("run_*"))
            for run_dir in run_dirs:
                nc_csv = run_dir / "merged_ncs.csv"
                if not nc_csv.exists():
                    print(f"    ⚠️ No merged_ncs.csv in {run_dir.name}, skipping")
                    continue
                with open(nc_csv, 'r') as f_csv:
                    pmt_scan_nc += sum(1 for _ in f_csv) - 1
                nc_data_dict = load_nc_data_dict_from_csv(nc_csv)
                hdf5_files = sorted(run_dir.glob("output_t*.hdf5"))
                for hdf5_file in hdf5_files:

                    try:
                        # Count photons per PMT UID
                        pmt_photon_counts: Dict[int, int] = {}
                        with h5py.File(hdf5_file, 'r') as f:
                            total = len(f['hit']['optical']['x_position_in_m']['pages'])
                            num_chunks = (total - 1) // chunk_size + 1

                            for chunk_idx in range(num_chunks):
                                cs = chunk_idx * chunk_size
                                ce = min(cs + chunk_size, total)
                                muon_ids = f['hit']['optical']['muon_track_id']['pages'][cs:ce]
                                nc_track_ids = f['hit']['optical']['nC_track_id']['pages'][cs:ce]
                                time = f['hit']['optical']['time_in_ns']['pages'][cs:ce]
                                det_uid = f['hit']['optical']['det_uid']['pages'][cs:ce]

                                nc_times = np.full(len(time), np.inf, dtype=np.float32)
                                for idx in range(len(muon_ids)):
                                    key = (int(muon_ids[idx]), int(nc_track_ids[idx]))
                                    if key in nc_data_dict:
                                        nc_times[idx] = nc_data_dict[key]['nC_time']
                                time_mask = (nc_times != np.inf) & (time >= nc_times) & (time <= nc_times + 200.0)
                                det_uid_f = det_uid[time_mask]
                                pmt_mask = (det_uid_f >= 10000000) & (det_uid_f < 1000000000)
                                det_uid_f = det_uid_f[pmt_mask]

                                uids, cnts = np.unique(det_uid_f, return_counts=True)
                                for u, c in zip(uids, cnts):
                                    pmt_photon_counts[int(u)] = pmt_photon_counts.get(int(u), 0) + int(c)

                        # Distribute fractionally for each N
                        for n in zone_configs.keys():
                            for uid_int, n_ph in pmt_photon_counts.items():
                                if uid_int not in uid_to_pmt:
                                    continue
                                pmt = uid_to_pmt[uid_int]
                                if pmt.layer != area_name:
                                    continue
                                for zone_key, pmt_fracs in frac_configs[n].items():
                                    if pmt.index in pmt_fracs:
                                        pmt_counts_by_n[n][zone_key] += n_ph * pmt_fracs[pmt.index]

                        # Count bot PMT photons (once, during first area scan)
                        if not bot_counted:
                            bot_frac = zone_fractions.get("bot_0", {})
                            for uid_int, n_ph in pmt_photon_counts.items():
                                if uid_int not in uid_to_pmt:
                                    continue
                                pmt = uid_to_pmt[uid_int]
                                if pmt.layer != 'bot':
                                    continue
                                if pmt.index in bot_frac:
                                    bot_pmt_counts_scan["bot_0"] += n_ph * bot_frac[pmt.index]

                    except Exception as e:
                        print(f"    Error: {e}")

            # Find optimal N
            optimal_n, scan_data = scan_optimal_zones(
                ssd_counts_by_n, pmt_counts_by_n,
                ssd_scan_nc, pmt_scan_nc,
                area_name, zone_configs, SNR_THRESHOLD
            )
            optimal_zones[area_name] = optimal_n
            scan_results[area_name] = scan_data
            # Cache counts for optimal N to avoid reprocessing
            optimal_zones[f"{area_name}_ssd_counts"] = ssd_counts_by_n[optimal_n]
            optimal_zones[f"{area_name}_pmt_counts"] = pmt_counts_by_n[optimal_n]
            optimal_zones[f"{area_name}_ssd_nc"] = ssd_scan_nc
            optimal_zones[f"{area_name}_pmt_nc"] = pmt_scan_nc
            if not bot_counted:
                bot_counted = True

        N_ZONES_PIT_OPT = optimal_zones.get('pit', N_ZONES_PIT)
        N_ZONES_TOP_OPT = optimal_zones.get('top', N_ZONES_TOP)
        N_ZONES_WALL_OPT = optimal_zones.get('wall', N_ZONES_WALL)

        print("\n" + "=" * 80)
        print(f"OPTIMAL ZONE COUNTS: pit={N_ZONES_PIT_OPT}, "
              f"top={N_ZONES_TOP_OPT}, wall={N_ZONES_WALL_OPT}, bot=1")
        print("=" * 80)

        # Plot SNR scan results
        if scan_results:
            plot_snr_scan(scan_results, SNR_THRESHOLD, optimal_zones,
                         OUTPUT_DIR / "zone_snr_scan.png")

        # Rebuild only scanned areas
        print("\nRebuilding zones with optimal counts...")
        if 'pit' in ZONE_SCAN_AREAS:
            pit_zones, pit_bounds = build_radial_zones(
                pmt_by_layer['pit'], N_ZONES_PIT_OPT, 0.0, geometry.r_pit, "pit")
        if 'top' in ZONE_SCAN_AREAS:
            top_zones, top_bounds = build_radial_zones(
                pmt_by_layer['top'], N_ZONES_TOP_OPT, geometry.r_zyl_top, geometry.r_zylinder, "top")
        if 'wall' in ZONE_SCAN_AREAS:
            wall_zones, wall_bounds = build_z_zones(
                pmt_by_layer['wall'], N_ZONES_WALL_OPT,
                geometry.z_cut_bot, geometry.z_cut_top, "wall", geometry.r_zylinder)

        # Rebuild zone_fractions and all_zone_keys
        all_zones = pit_zones + top_zones + wall_zones + bot_zones
        zone_fractions = {}
        for z in all_zones:
            key = f"{z.area_name}_{z.zone_id}"
            zone_fractions[key] = z.pmt_fractions if z.pmt_fractions else {}
        all_zone_keys = [f"{z.area_name}_{z.zone_id}" for z in all_zones]

    # Check if all 3 areas were scanned (can reuse counts entirely)
    all_areas_scanned = (not COMPARE_MODE
                         and ZONE_SCAN_AREAS
                         and set(ZONE_SCAN_AREAS) >= {'pit', 'top', 'wall'})

    if all_areas_scanned:
        # All areas scanned: reuse counts entirely
        print("\nReusing photon counts from zone scan...")
        ssd_counts: Dict[str, int] = {}
        pmt_counts: Dict[str, float] = {}

        for area_name in ['pit', 'top', 'wall']:
            ssd_counts.update(optimal_zones[f"{area_name}_ssd_counts"])
            pmt_counts.update(optimal_zones[f"{area_name}_pmt_counts"])

        ssd_counts["bot_0"] = bot_ssd_counts_scan["bot_0"]
        pmt_counts["bot_0"] = bot_pmt_counts_scan["bot_0"]

        ssd_nc = optimal_zones["pit_ssd_nc"]
        pmt_nc = optimal_zones["pit_pmt_nc"]
        ssd_files = len(list(SSD_DIR.glob("run_*/output_t*.hdf5")))
        pmt_files = len(list(PMT_DIR.glob("run_*/output_t*.hdf5")))

        if ssd_nc != pmt_nc:
            nc_diff = abs(ssd_nc - pmt_nc)
            nc_diff_pct = 100.0 * nc_diff / max(ssd_nc, pmt_nc)
            print(f"\n⚠️  WARNING: NC event mismatch: SSD={ssd_nc:,}, PMT={pmt_nc:,} "
                  f"(diff={nc_diff:,}, {nc_diff_pct:.2f}%)")

    else:
        # Full processing (no scan, or only partial scan)
        print("\n[1/2] Processing SSD simulation data...")
        n_pit_final = len(pit_zones)
        n_top_final = len(top_zones)
        n_wall_final = len(wall_zones)

        ssd_counts, ssd_files, ssd_nc = process_all_files_ssd(
            SSD_DIR, geometry, chunk_size,
            pit_bounds, top_bounds, wall_bounds, bot_bounds,
            n_pit_final, n_top_final, n_wall_final, 1
        )

        # Process PMT data
        print("\n[2/2] Processing PMT simulation data...")
        pmt_counts, pmt_files, pmt_nc, observed_uids = process_all_files_pmt(
            PMT_DIR, geometry, chunk_size,
            uid_to_pmt, zone_fractions, all_zone_keys
        )

        # Cross-checks
        assert ssd_files == pmt_files, f"File count mismatch: SSD={ssd_files}, PMT={pmt_files}"
        crosscheck_uids(uid_to_pmt, observed_uids, "PMT")

        if ssd_nc != pmt_nc:
            nc_diff = abs(ssd_nc - pmt_nc)
            nc_diff_pct = 100.0 * nc_diff / max(ssd_nc, pmt_nc)
            print(f"\n⚠️  WARNING: NC event mismatch: SSD={ssd_nc:,}, PMT={pmt_nc:,} "
                  f"(diff={nc_diff:,}, {nc_diff_pct:.2f}%)")

    # Compute and print results
    print("\n" + "=" * 130)
    print("RESULTS - ZONE RATIOS")
    print("=" * 130)

    header = (f"{'AREA':<8} {'ZONE':<6} {'BOUNDARY':<30} {'SSD_PH':>12} "
              f"{'PMT_PH':>12} {'PH/NC_SSD':>12} {'PH/NC_PMT':>12} "
              f"{'RATIO':>10} {'EFF_PMTs':>10} {'AREA_mm2':>12} {'DENSITY':>14}")
    sep = "-" * len(header)

    # Compute mean density per area for normalization
    area_mean_density: Dict[str, float] = {}
    for area_name in ['pit', 'bot', 'top', 'wall']:
        area_zones = [z for z in all_zones if z.area_name == area_name]
        total_eff = sum(z.effective_n_pmts for z in area_zones)
        total_area = sum(z.area_mm2 for z in area_zones)
        area_mean_density[area_name] = total_eff / total_area if total_area > 0 else 0.0

    header_ext = (f"{'AREA':<8} {'ZONE':<6} {'BOUNDARY':<30} {'SSD_PH':>12} "
                  f"{'PMT_PH':>12} {'PH/NC_SSD':>12} {'PH/NC_PMT':>12} "
                  f"{'RAW_RATIO':>10} {'CORR_RATIO':>10} {'EFF_PMTs':>10} "
                  f"{'AREA_mm2':>12} {'DENSITY':>14} {'DENS_DEV':>9}")
    sep = "-" * len(header_ext)

    print(header_ext)
    print(sep)

    results = []
    for zone in all_zones:
        key = f"{zone.area_name}_{zone.zone_id}"
        ssd_ph = ssd_counts.get(key, 0)
        pmt_ph = pmt_counts.get(key, 0.0)

        ssd_per_nc = ssd_ph / ssd_nc if ssd_nc > 0 else 0.0
        pmt_per_nc = pmt_ph / pmt_nc if pmt_nc > 0 else 0.0
        raw_ratio = ssd_per_nc / pmt_per_nc if pmt_per_nc > 0 else float('nan')

        # Density-corrected ratio: normalize out PMT density variations
        mean_d = area_mean_density[zone.area_name]
        if mean_d > 0 and zone.pmt_density > 0 and not np.isnan(raw_ratio):
            density_factor = zone.pmt_density / mean_d
            corr_ratio = raw_ratio * density_factor
        else:
            corr_ratio = float('nan')

        dens_dev = (zone.pmt_density - mean_d) / mean_d * 100 if mean_d > 0 else 0.0

        if zone.area_name in ('pit', 'bot', 'top'):
            bstr = f"r=[{zone.boundary_low:.0f}, {zone.boundary_high:.0f}]mm"
        else:
            bstr = f"z=[{zone.boundary_low:.0f}, {zone.boundary_high:.0f}]mm"

        print(f"{zone.area_name:<8} {zone.zone_id:<6} {bstr:<30} "
              f"{ssd_ph:>12d} {pmt_ph:>12.1f} {ssd_per_nc:>12.6f} {pmt_per_nc:>12.6f} "
              f"{raw_ratio:>10.4f} {corr_ratio:>10.4f} {zone.effective_n_pmts:>10.2f} "
              f"{zone.area_mm2:>12.0f} {zone.pmt_density:>14.6e} {dens_dev:>+8.1f}%")

        results.append({
            'zone': zone, 'key': key,
            'ssd_photons': ssd_ph, 'pmt_photons': pmt_ph,
            'ssd_per_nc': ssd_per_nc, 'pmt_per_nc': pmt_per_nc,
            'ratio': raw_ratio, 'corr_ratio': corr_ratio,
            'dens_dev': dens_dev,
        })

    print(sep)

    # Aggregated per area
    print(f"\n{'AREA':<10} {'SSD_TOTAL':>12} {'PMT_TOTAL':>12} {'PH/NC_SSD':>12} "
          f"{'PH/NC_PMT':>12} {'RATIO':>10}")
    print("-" * 70)
    for area_name in ['pit', 'bot', 'top', 'wall']:
        ar = [r for r in results if r['zone'].area_name == area_name]
        ssd_t = sum(r['ssd_photons'] for r in ar)
        pmt_t = sum(r['pmt_photons'] for r in ar)
        ssd_pnc = ssd_t / ssd_nc if ssd_nc > 0 else 0.0
        pmt_pnc = pmt_t / pmt_nc if pmt_nc > 0 else 0.0
        a_ratio = ssd_pnc / pmt_pnc if pmt_pnc > 0 else float('nan')
        print(f"{area_name:<10} {ssd_t:>12d} {pmt_t:>12.1f} "
              f"{ssd_pnc:>12.6f} {pmt_pnc:>12.6f} {a_ratio:>10.4f}")

    g_ssd = sum(r['ssd_photons'] for r in results)
    g_pmt = sum(r['pmt_photons'] for r in results)
    g_ssd_pnc = g_ssd / ssd_nc if ssd_nc > 0 else 0.0
    g_pmt_pnc = g_pmt / pmt_nc if pmt_nc > 0 else 0.0
    g_ratio = g_ssd_pnc / g_pmt_pnc if g_pmt_pnc > 0 else float('nan')
    print("-" * 70)
    print(f"{'TOTAL':<10} {g_ssd:>12d} {g_pmt:>12.1f} "
          f"{g_ssd_pnc:>12.6f} {g_pmt_pnc:>12.6f} {g_ratio:>10.4f}")
    print("=" * 70)

    # Write results file
    with open(output_file, 'w') as fout:
        fout.write("=" * 130 + "\n")
        fout.write("Zone-based SSD vs. PMT Photon Detection Efficiency Ratio\n")
        fout.write("=" * 130 + "\n\n")
        fout.write(f"Zone config: Pit={len(pit_zones)}, Wall={len(wall_zones)}, "
                   f"Top={len(top_zones)}, Bot=1\n")
        if ZONE_SCAN_AREAS:
            fout.write(f"Zone scan areas: {ZONE_SCAN_AREAS}\n")
            fout.write(f"SNR threshold: {SNR_THRESHOLD}, Min PMTs/zone: {MIN_PMTS_PER_ZONE}\n")
        fout.write(f"PMT cathode radius: {PMT_RADIUS} mm\n")
        fout.write(f"MC samples: {MC_SAMPLES}\n")
        fout.write(f"Files: SSD={ssd_files}, PMT={pmt_files}\n")
        fout.write(f"NC events: SSD={ssd_nc:,}, PMT={pmt_nc:,}\n\n")
        fout.write(header_ext + "\n")
        fout.write(sep + "\n")
        for r in results:
            z = r['zone']
            if z.area_name in ('pit', 'bot', 'top'):
                bstr = f"r=[{z.boundary_low:.0f}, {z.boundary_high:.0f}]mm"
            else:
                bstr = f"z=[{z.boundary_low:.0f}, {z.boundary_high:.0f}]mm"
            fout.write(f"{z.area_name:<8} {z.zone_id:<6} {bstr:<30} "
                       f"{r['ssd_photons']:>12d} {r['pmt_photons']:>12.1f} "
                       f"{r['ssd_per_nc']:>12.6f} {r['pmt_per_nc']:>12.6f} "
                       f"{r['ratio']:>10.4f} {r['corr_ratio']:>10.4f} "
                       f"{z.effective_n_pmts:>10.2f} {z.area_mm2:>12.0f} "
                       f"{z.pmt_density:>14.6e} {r['dens_dev']:>+8.1f}%\n")
        fout.write(sep + "\n")
        fout.write(f"\nFiles: SSD={ssd_files}, PMT={pmt_files}\n")
        fout.write(f"NC events: SSD={ssd_nc:,}, PMT={pmt_nc:,}\n")
        fout.write("=" * 130 + "\n")

    print(f"\nResults written to: {output_file}")

    # Write machine-readable JSON for downstream comparison
    json_output = OUTPUT_DIR / "zone_ratio_results.json"
    json_results = {
        'metadata': {
            'zone_config': {
                'pit': len(pit_zones), 'top': len(top_zones),
                'wall': len(wall_zones), 'bot': 1,
            },
            'ssd_nc': ssd_nc, 'pmt_nc': pmt_nc,
            'ssd_files': ssd_files, 'pmt_files': pmt_files,
            'pmt_cathode_radius': PMT_RADIUS,
            'mc_samples': MC_SAMPLES,
            'geometry': geometry.geometry_name,
        },
        'boundaries': {
            'pit': [float(b) for b in pit_bounds],
            'top': [float(b) for b in top_bounds],
            'wall': [float(b) for b in wall_bounds],
            'bot': [float(b) for b in bot_bounds],
        },
        'zones': [],
    }
    for r in results:
        z = r['zone']
        json_results['zones'].append({
            'area': z.area_name,
            'zone_id': z.zone_id,
            'boundary_low': float(z.boundary_low),
            'boundary_high': float(z.boundary_high),
            'area_mm2': float(z.area_mm2),
            'effective_n_pmts': float(z.effective_n_pmts),
            'pmt_density': float(z.pmt_density),
            'ssd_photons': int(r['ssd_photons']),
            'pmt_photons': float(r['pmt_photons']),
            'ssd_per_nc': float(r['ssd_per_nc']),
            'pmt_per_nc': float(r['pmt_per_nc']),
            'raw_ratio': float(r['ratio']) if not np.isnan(r['ratio']) else None,
            'corr_ratio': float(r['corr_ratio']) if not np.isnan(r['corr_ratio']) else None,
            'dens_dev_pct': float(r['dens_dev']),
        })
    with open(json_output, 'w') as f_json:
        json.dump(json_results, f_json, indent=2)
    print(f"Machine-readable results written to: {json_output}")

    # Generate plots
    print("\nGenerating plots...")

    # Global color scale across all plots
    all_ratios = [r['corr_ratio'] for r in results if not np.isnan(r['corr_ratio'])]
    if all_ratios:
        global_norm = plt.Normalize(vmin=min(all_ratios) * 0.9, vmax=max(all_ratios) * 1.1)
    else:
        global_norm = plt.Normalize(0, 1)

    for area_name, zones_list, layer_pmts, r_min, r_max in [
        ("pit", pit_zones, pmt_by_layer['pit'], 0.0, geometry.r_pit),
        ("top", top_zones, pmt_by_layer['top'], geometry.r_zyl_top, geometry.r_zylinder),
        ("bot", bot_zones, pmt_by_layer['bot'], geometry.r_zyl_bot, geometry.r_zylinder),
    ]:
        ratios = [next((r['corr_ratio'] for r in results
                        if r['zone'].area_name == area_name and r['zone'].zone_id == z.zone_id),
                       float('nan'))
                  for z in zones_list]
        plot_radial_zones(zones_list, layer_pmts, ratios, area_name, r_min, r_max,
                         OUTPUT_DIR / f"zone_ratio_{area_name}.png", global_norm)

    wall_ratios = [next((r['corr_ratio'] for r in results
                         if r['zone'].area_name == 'wall' and r['zone'].zone_id == z.zone_id),
                        float('nan'))
                   for z in wall_zones]
    plot_wall_zones(wall_zones, pmt_by_layer['wall'], wall_ratios,
                   geometry.r_zylinder, OUTPUT_DIR / "zone_ratio_wall.png", global_norm)
    
    # Comparison plots (only in COMPARE_MODE)
    if COMPARE_MODE:
        print("\nGenerating comparison plots...")
        plot_comparison(results, ref_data, OUTPUT_DIR)

    # ==========================================================================
    # Diagnostic plot: Area-normalized photon flux per zone (all areas)
    # ==========================================================================
    for area_name in ['pit', 'top', 'wall']:
        area_results = [r for r in results if r['zone'].area_name == area_name]
        if not area_results:
            continue

        fig, ax1 = plt.subplots(figsize=(12, 6))

        if area_name == 'wall':
            mids = [(r['zone'].boundary_low + r['zone'].boundary_high) / 2
                    for r in area_results]
            xlabel = 'z [mm]'
        else:
            mids = [(r['zone'].boundary_low + r['zone'].boundary_high) / 2
                    for r in area_results]
            xlabel = 'r [mm]'

        ssd_flux = [r['ssd_per_nc'] / r['zone'].area_mm2 for r in area_results]
        pmt_flux = [r['pmt_per_nc'] / r['zone'].pmt_density
                    if r['zone'].pmt_density > 0 else 0.0
                    for r in area_results]

        color_ssd = 'tab:blue'
        color_pmt = 'tab:red'

        ax1.plot(mids, ssd_flux, 'o-', color=color_ssd,
                 linewidth=2, markersize=6, label='SSD: photons/NC/mm²')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('SSD photons / NC / mm²', color=color_ssd)
        ax1.tick_params(axis='y', labelcolor=color_ssd)

        ax2 = ax1.twinx()
        ax2.plot(mids, pmt_flux, 's-', color=color_pmt,
                 linewidth=2, markersize=6, label='PMT: photons/NC/(PMT/mm²)')
        ax2.set_ylabel('PMT photons / NC / (PMT/mm²)', color=color_pmt)
        ax2.tick_params(axis='y', labelcolor=color_pmt)

        ax1.set_title(f'{area_name.upper()}: Area-normalized photon flux (SSD vs PMT)')
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{area_name}_ssd_vs_pmt_flux.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {OUTPUT_DIR / f'{area_name}_ssd_vs_pmt_flux.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()