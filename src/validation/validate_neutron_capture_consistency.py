#!/usr/bin/env python3
"""
validate_neutron_capture_consistency.py
=======================================

Strict provenance validator for the neutron-capture (NC) ground truth across the
complete LEGEND simulation / post-processing chain.

PURPOSE
-------
The two-simulation MUSUN workflow seeds *several* downstream optical simulations
(SSD and PMT instrumentation) from one and the same NC-truth simulation ("sim1").
If, by mistake, a different NC-truth simulation (e.g. a re-run with a different
random seed) was used to seed one of the optical sims, the underlying set of
neutron captures would silently diverge between data products.

This tool answers a single, deliberately narrow question:

    Do all data formats describe the *exact same set* of neutron captures,
    identified by (run_id, muon_id, nc_id)?

It does NOT care about photon-detection efficiency, voxel hits, or geometry.
It compares NC *identity sets* only and fails loudly on any divergence.

------------------------------------------------------------------------------
STEP 1 FINDINGS — how the chain treats neutron captures
(documented here per task requirement, derived from reading:
   src/core/musun2SimFullPostprocessing.py
   ../voxelSelection/evaluation/ratio_derivation/ratio_analysis/raw_loading.py
   ../voxelSelection/evaluation/ratio_derivation/ratio_analysis/pmt_data.py)
------------------------------------------------------------------------------

1. How NCs are loaded from sim1
   musun2SimFullPostprocessing.load_all_nc_data_from_sim1():
     - Reads group  /hit/MyNeutronCaptureOutput  from every output_*.hdf5.
     - LGDO column layout: each field is <group>/<field>/pages (+ /entries scalar).
     - muon_id := evtid            (/hit/MyNeutronCaptureOutput/evtid/pages)
       nc_id   := nC_track_id      (/hit/MyNeutronCaptureOutput/nC_track_id/pages)
     - run_id is parsed from the enclosing  run_NNN  directory name (0 if flat).

2. Which identifiers uniquely define a neutron capture
   The composite key  (run_id, muon_id, nc_id)  is THE unique NC key.
   run_id is required because muon_id/nc_id are only locally unique per run.

3. How NCs are matched to SSD hits
   musun2SimFullPostprocessing.scan_sim2_and_aggregate():
     - sim2 optical hits carry muon_track_id (== sim1 evtid) and nC_track_id.
     - Hits are matched to NC truth within the same run_id by (muon_id, nc_id),
       filtered to det_uid in {1965,1966,1967,1968}, relative time 0..200 ns,
       and a radial/axial momentum cut, then KDTree-assigned to voxels.
     - (matching/filtering is irrelevant to *this* validator — it only affects
       photon counts, never the NC identity set written to the output.)

4. How the post-processed output is generated
   write_output_in_batches() / create_or_open_output_file():
     - ONE file:  ncscore_output_0.hdf5  (NOT a train/val split).
     - Dataset  event_ids  shape (N,3), columns event_id_columns =
       ["run_id","muon_id","nc_id"]  — exactly one row per NC.
     - NCs whose run has NO sim2 data are dropped; NCs with zero photons are
       STILL written. => In a correct setup the post-processed event_ids set
       equals the sim1 NC set restricted to the runs present in the raw sims.

5. Which fields are preserved / modified / discarded
   - Preserved as identity: (run_id, muon_id, nc_id)  -> event_ids.
   - Transformed (not relevant here): positions m->mm (x1000), material/volume
     IDs remapped to global mappings, gamma attributes copied into phi_matrix.
   - Discarded: per-photon information (collapsed into voxel/region counts).
   => Identity is preserved verbatim; this validator checks exactly that.

6. PMT side (raw_loading.py / pmt_data.py)
   - NC truth loaded identically from /hit/MyNeutronCaptureOutput.
   - Optical hits from /hit/optical, matched by (run_id, muon_track_id,
     nC_track_id). PMT det_uids are 8-digit (no {1965..1968} filter).
   - The user-confirmed, authoritative NC list for a raw optical sim is the
     /hit/DebugVertices group (one row per *gamma* of an NC, carrying muon_id
     and nc_id). Deduplicated, it enumerates ALL NCs of that simulation and is
     expected to be completely consistent with sim1.

------------------------------------------------------------------------------
NC-IDENTITY SOURCE PER DATA FORMAT
------------------------------------------------------------------------------
  sim1               : /hit/MyNeutronCaptureOutput  -> (evtid, nC_track_id)
  raw SSD (sim2)     : /hit/DebugVertices            -> (muon_id, nc_id)  [dedup]
  raw PMT (sim2)     : /hit/DebugVertices            -> (muon_id, nc_id)  [dedup]
  post-processed SSD : event_ids                     -> (run_id, muon_id, nc_id)
  run_id always derived from the run_NNN directory (0 for a flat layout).

If a raw sim has no /hit/DebugVertices, the script falls back to the NC ids
referenced by /hit/optical hits. That set is only a SUBSET (NCs that produced
detected photons), so for fallback datasets the comparison degrades from strict
equality to a traceability + overlap check (and emits a prominent warning).

EXIT STATUS
-----------
  0  -> every consistency check passed (all four ID sets identical).
  1  -> a RuntimeError was raised (inconsistency or I/O/integrity failure);
        a detailed, self-contained diagnostic is printed.

USAGE
-----
  python validate_neutron_capture_consistency.py \\
      --sim1              /path/to/sim1_nc_truth \\
      --raw-ssd          /path/to/raw_ssd_sim2 \\
      --postprocessed-ssd /path/to/ncscore_output_0.hdf5 \\
      --pmt              /path/to/raw_pmt_sim2
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Types & module-level configuration
# ---------------------------------------------------------------------------

#: Composite, globally-unique neutron-capture key: (run_id, muon_id, nc_id).
NCKey = Tuple[int, int, int]

logger = logging.getLogger("nc_consistency")

#: run directory naming convention, e.g. "run_007" -> 7
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")

#: sim sub-directory naming convention (multi-run layout), e.g. "sim_007" -> 7
_SIM_DIR_RE = re.compile(r"^sim_(\d+)$")

#: HDF5 file globs tried (in order) inside a run / flat directory.
_FILE_PATTERNS: Tuple[str, ...] = ("output_t*.hdf5", "output_*.hdf5")

#: Number of example mismatching keys included in error messages.
_MAX_EXAMPLES = 12


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class NCDataset:
    """Resolved neutron-capture identity set for one data format.

    Attributes
    ----------
    name:
        Human-readable dataset label (used in logs and error messages).
    ids:
        Set of unique ``(run_id, muon_id, nc_id)`` keys.
    is_full:
        ``True`` if this set is the *complete* NC list of the simulation
        (sim1 / DebugVertices / post-processed event_ids). ``False`` if it is
        only a subset (optical-hit fallback), which relaxes equality checks to
        traceability + overlap.
    source:
        Name of the underlying HDF5 group/dataset the ids came from.
    n_rows_read:
        Total number of raw rows read before de-duplication.
    n_duplicates:
        ``n_rows_read - len(ids)`` (expected and harmless for DebugVertices).
    per_run_counts:
        Unique NC count per ``run_id``.
    """

    name: str
    ids: Set[NCKey]
    is_full: bool
    source: str
    n_rows_read: int = 0
    n_duplicates: int = 0
    per_run_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def run_ids(self) -> Set[int]:
        return set(self.per_run_counts.keys())


# ---------------------------------------------------------------------------
# Low-level HDF5 helpers
# ---------------------------------------------------------------------------

def _safe_open(path: Path) -> h5py.File:
    """Open an HDF5 file read-only, converting corruption into a RuntimeError.

    Parameters
    ----------
    path:
        File to open.

    Returns
    -------
    h5py.File
        Open file handle (caller is responsible for closing / using ``with``).

    Raises
    ------
    RuntimeError
        If the file cannot be opened (truncated / corrupt / not HDF5).
    """
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        raise RuntimeError(
            f"[integrity] Failed to open HDF5 file (corrupt or truncated):\n"
            f"  File : {path}\n"
            f"  Error: {type(exc).__name__}: {exc}"
        ) from exc


def _read_pages(f: h5py.File, parts: Sequence[str], file_path: Path) -> np.ndarray:
    """Read an LGDO column (``.../<field>/pages``) and return it as an array.

    Parameters
    ----------
    f:
        Open HDF5 file.
    parts:
        Group/field path components *excluding* the trailing ``"pages"``,
        e.g. ``("hit", "MyNeutronCaptureOutput", "evtid")``.
    file_path:
        Source path, only used for error messages.

    Raises
    ------
    RuntimeError
        If any component or the ``pages`` dataset is missing.
    """
    node: object = f
    walked: List[str] = []
    for p in parts:
        walked.append(p)
        try:
            node = node[p]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"[schema] Missing group/field '{'/'.join(walked)}' in file:\n"
                f"  File : {file_path}\n"
                f"  Error: {exc}"
            ) from exc
    try:
        return node["pages"][:]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"[schema] Missing 'pages' dataset under '{'/'.join(parts)}' in file:\n"
            f"  File : {file_path}\n"
            f"  Error: {exc}"
        ) from exc


def _decode(value: object) -> str:
    """Decode an HDF5 string scalar/element to ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# ---------------------------------------------------------------------------
# File discovery (auto nested run_NNN vs flat)
# ---------------------------------------------------------------------------

def _glob_first(directory: Path, patterns: Sequence[str]) -> List[Path]:
    """Return sorted files matching the first pattern that yields any match."""
    for pat in patterns:
        matches = sorted(Path(p) for p in glob.glob(str(directory / pat)))
        if matches:
            return matches
    return []


def _pick_sim_subdir_files(run_dir: Path, label: str, run_id: int) -> List[Path]:
    """Pick one sim sub-directory inside a run dir and return its HDF5 files.

    In the multi-run layout, ``run_NNN`` contains several ``sim_MMM`` sub-dirs
    that all share the same underlying NC truth. The first sorted ``sim_MMM``
    that actually contains HDF5 files is chosen (the choice does not matter for
    NC identity, only that we read a complete, self-consistent NC set).

    Returns an empty list if no sim sub-directory with HDF5 files is found.
    """
    sim_dirs = sorted(
        (d for d in run_dir.iterdir() if d.is_dir() and _SIM_DIR_RE.match(d.name)),
        key=lambda d: int(_SIM_DIR_RE.match(d.name).group(1)),  # type: ignore[union-attr]
    )
    if not sim_dirs:
        # Be lenient: fall back to any sub-directory holding HDF5 files.
        sim_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir())
    for sd in sim_dirs:
        files = _glob_first(sd, _FILE_PATTERNS)
        if files:
            logger.info(
                "[%s] run_%03d: selected sim dir '%s' (%d file(s)) [multi-run].",
                label, run_id, sd.name, len(files),
            )
            return files
    return []


def discover_run_files(base: str, label: str, multi_run: bool = False) -> Dict[int, List[Path]]:
    """Discover simulation HDF5 files, auto-detecting nested vs flat layout.

    Nested layout:     ``base/run_NNN/output_*.hdf5``           -> keyed by run id.
    Flat layout:       ``base/output_*.hdf5``                   -> keyed by run id 0.
    Multi-run layout:  ``base/run_NNN/sim_MMM/output_*.hdf5``   -> one sim_MMM is
                       chosen per run (only when ``multi_run`` is True).

    Parameters
    ----------
    base:
        Directory to scan.
    label:
        Dataset label for log/error messages.
    multi_run:
        If True, each ``run_NNN`` is expected to contain ``sim_MMM`` sub-dirs;
        one is chosen per run.

    Returns
    -------
    dict[int, list[Path]]
        Mapping ``run_id -> [files...]`` (non-empty).

    Raises
    ------
    RuntimeError
        If the directory is missing or contains no simulation files.
    """
    base_path = Path(base)
    if not base_path.exists():
        raise RuntimeError(f"[{label}] Input path does not exist: {base_path}")
    if not base_path.is_dir():
        raise RuntimeError(f"[{label}] Input path is not a directory: {base_path}")

    run_dirs: List[Tuple[int, Path]] = []
    for child in sorted(base_path.iterdir()):
        if not child.is_dir():
            continue
        m = _RUN_DIR_RE.match(child.name)
        if m:
            run_dirs.append((int(m.group(1)), child))

    files_by_run: Dict[int, List[Path]] = {}

    if multi_run:
        if not run_dirs:
            raise RuntimeError(
                f"[{label}] --multi-run set but no run_NNN directories found in "
                f"{base_path}."
            )
        for run_id, run_dir in run_dirs:
            files = _pick_sim_subdir_files(run_dir, label, run_id)
            if files:
                files_by_run[run_id] = files
        if not files_by_run:
            raise RuntimeError(
                f"[{label}] --multi-run set but no run_NNN/sim_MMM directory in "
                f"{base_path} contained any of {list(_FILE_PATTERNS)}."
            )
        n_files = sum(len(v) for v in files_by_run.values())
        logger.info(
            "[%s] Multi-run layout: %d run(s), %d file(s) total (one sim dir/run).",
            label, len(files_by_run), n_files,
        )
    elif run_dirs:
        for run_id, run_dir in run_dirs:
            files = _glob_first(run_dir, _FILE_PATTERNS)
            if files:
                files_by_run[run_id] = files
        if not files_by_run:
            raise RuntimeError(
                f"[{label}] Found run_NNN directories in {base_path} but none "
                f"contained any of {list(_FILE_PATTERNS)}."
            )
        n_files = sum(len(v) for v in files_by_run.values())
        logger.info(
            "[%s] Nested layout: %d run(s), %d file(s) total.",
            label, len(files_by_run), n_files,
        )
    else:
        files = _glob_first(base_path, _FILE_PATTERNS)
        if not files:
            raise RuntimeError(
                f"[{label}] No run_NNN subdirectories and no "
                f"{list(_FILE_PATTERNS)} files directly in {base_path}."
            )
        files_by_run[0] = files
        logger.info("[%s] Flat layout (run_id=0): %d file(s).", label, len(files))

    return files_by_run


# ---------------------------------------------------------------------------
# Set / duplicate utilities
# ---------------------------------------------------------------------------

def _stack_keys(run_id: int, muon: np.ndarray, nc: np.ndarray, file_path: Path) -> np.ndarray:
    """Build an (N,3) int64 key array for one file's (muon, nc) columns."""
    if len(muon) != len(nc):
        raise RuntimeError(
            f"[schema] muon/nc column length mismatch in {file_path}: "
            f"{len(muon)} vs {len(nc)}."
        )
    n = len(muon)
    out = np.empty((n, 3), dtype=np.int64)
    out[:, 0] = run_id
    out[:, 1] = muon.astype(np.int64, copy=False)
    out[:, 2] = nc.astype(np.int64, copy=False)
    return out


def _array_to_keyset(arr: np.ndarray) -> Set[NCKey]:
    """Convert an (N,3) array to a set of ``(run_id, muon_id, nc_id)`` tuples."""
    if arr.size == 0:
        return set()
    return set(map(tuple, arr.tolist()))


def _find_duplicate_keys(arr: np.ndarray) -> np.ndarray:
    """Return the distinct rows of ``arr`` that occur more than once."""
    if arr.size == 0:
        return arr
    uniq, counts = np.unique(arr, axis=0, return_counts=True)
    return uniq[counts > 1]


def _format_examples(keys: Sequence[NCKey], limit: int = _MAX_EXAMPLES) -> str:
    """Render up to ``limit`` keys as ``(run=.., muon=.., nc=..)`` strings."""
    ordered = sorted(keys)
    shown = ordered[:limit]
    rendered = ", ".join(f"(run={r}, muon={m}, nc={n})" for r, m, n in shown)
    if len(ordered) > limit:
        rendered += f", ... (+{len(ordered) - limit} more)"
    return rendered or "<none>"


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------

def load_sim1(path: str, label: str = "sim1") -> NCDataset:
    """Load the reference NC truth from sim1 (/hit/MyNeutronCaptureOutput).

    Raises
    ------
    RuntimeError
        On I/O errors, schema errors, or duplicate NC rows (each NC must
        appear exactly once in the truth — duplicates indicate a sim bug).
    """
    files_by_run = discover_run_files(path, label)
    chunks: List[np.ndarray] = []
    per_run: Dict[int, int] = {}

    for run_id, files in sorted(files_by_run.items()):
        run_chunks: List[np.ndarray] = []
        for fp in files:
            with _safe_open(fp) as f:
                evt = _read_pages(f, ("hit", "MyNeutronCaptureOutput", "evtid"), fp)
                nc = _read_pages(f, ("hit", "MyNeutronCaptureOutput", "nC_track_id"), fp)
            run_chunks.append(_stack_keys(run_id, evt, nc, fp))
        run_arr = (
            np.concatenate(run_chunks, axis=0) if run_chunks else np.empty((0, 3), np.int64)
        )
        per_run[run_id] = len({tuple(r) for r in run_arr.tolist()})
        chunks.append(run_arr)

    all_arr = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), np.int64)

    duplicates = _find_duplicate_keys(all_arr)
    if duplicates.size:
        dup_keys = _array_to_keyset(duplicates)
        raise RuntimeError(
            "NC consistency check FAILED — unexpected event duplication\n"
            f"  Stage    : load {label}\n"
            f"  Dataset  : {label} (/hit/MyNeutronCaptureOutput)\n"
            f"  Detail   : {len(dup_keys)} NC key(s) appear more than once in the "
            "NC truth; each capture must be recorded exactly once.\n"
            f"  Examples : {_format_examples(dup_keys)}"
        )

    ids = _array_to_keyset(all_arr)
    ds = NCDataset(
        name=label,
        ids=ids,
        is_full=True,
        source="/hit/MyNeutronCaptureOutput",
        n_rows_read=len(all_arr),
        n_duplicates=len(all_arr) - len(ids),
        per_run_counts=per_run,
    )
    return ds


def load_raw_sim(
    path: str,
    label: str,
    allow_optical_fallback: bool = True,
    multi_run: bool = False,
) -> NCDataset:
    """Load NC ids from a raw optical sim (SSD or PMT).

    Prefers /hit/DebugVertices (one row per gamma -> deduped to all NCs). If
    absent and ``allow_optical_fallback`` is set, falls back to the NC ids
    referenced by /hit/optical (a SUBSET) and marks the dataset ``is_full=False``.

    When ``multi_run`` is True, the ``run_NNN/sim_MMM/output_*.hdf5`` layout is
    used and one ``sim_MMM`` is chosen per run.

    Raises
    ------
    RuntimeError
        On I/O / schema errors, or if a file mixes sources, or if neither
        DebugVertices nor optical is available.
    """
    files_by_run = discover_run_files(path, label, multi_run=multi_run)
    chunks: List[np.ndarray] = []
    per_run: Dict[int, int] = {}
    source: Optional[str] = None
    is_full = True

    for run_id, files in sorted(files_by_run.items()):
        run_chunks: List[np.ndarray] = []
        for fp in files:
            with _safe_open(fp) as f:
                hit = f.get("hit")
                if hit is None:
                    raise RuntimeError(f"[schema] Missing '/hit' group in {fp}")

                if "DebugVertices" in hit:
                    file_src = "/hit/DebugVertices"
                    muon = _read_pages(f, ("hit", "DebugVertices", "muon_id"), fp)
                    nc = _read_pages(f, ("hit", "DebugVertices", "nc_id"), fp)
                elif allow_optical_fallback and "optical" in hit:
                    file_src = "/hit/optical"
                    muon = _read_pages(f, ("hit", "optical", "muon_track_id"), fp)
                    nc = _read_pages(f, ("hit", "optical", "nC_track_id"), fp)
                else:
                    raise RuntimeError(
                        f"[schema] {label}: file has neither /hit/DebugVertices "
                        f"nor a usable /hit/optical group:\n  File: {fp}"
                    )

            if source is None:
                source = file_src
            elif source != file_src:
                raise RuntimeError(
                    f"[schema] {label}: inconsistent NC source across files "
                    f"('{source}' vs '{file_src}'). Refusing to mix DebugVertices "
                    f"and optical-fallback sets.\n  File: {fp}"
                )
            run_chunks.append(_stack_keys(run_id, muon, nc, fp))

        run_arr = (
            np.concatenate(run_chunks, axis=0) if run_chunks else np.empty((0, 3), np.int64)
        )
        per_run[run_id] = len({tuple(r) for r in run_arr.tolist()})
        chunks.append(run_arr)

    if source == "/hit/optical":
        is_full = False
        logger.warning(
            "[%s] /hit/DebugVertices not found — fell back to /hit/optical. "
            "This yields ONLY NCs with detected photons (a subset); strict "
            "equality is relaxed to traceability + overlap for this dataset.",
            label,
        )

    all_arr = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), np.int64)
    ids = _array_to_keyset(all_arr)
    ds = NCDataset(
        name=label,
        ids=ids,
        is_full=is_full,
        source=source or "<none>",
        n_rows_read=len(all_arr),
        n_duplicates=len(all_arr) - len(ids),
        per_run_counts=per_run,
    )
    return ds


def load_postprocessed(path: str, label: str = "postprocessed SSD") -> NCDataset:
    """Load NC ids from the post-processed output's ``event_ids`` dataset.

    ``path`` may be a single ncscore_output_*.hdf5 file or a directory
    containing one or more of them.

    Raises
    ------
    RuntimeError
        On I/O / schema errors, missing ``event_ids``, or duplicate rows
        (each NC must be written exactly once).
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(Path(x) for x in glob.glob(str(p / "ncscore_output_*.hdf5")))
        if not files:
            raise RuntimeError(
                f"[{label}] No ncscore_output_*.hdf5 files found in directory {p}"
            )
    elif p.is_file():
        files = [p]
    else:
        raise RuntimeError(f"[{label}] Path does not exist: {p}")

    chunks: List[np.ndarray] = []
    for fp in files:
        with _safe_open(fp) as f:
            if "event_ids" not in f:
                raise RuntimeError(
                    f"[schema] {label}: dataset 'event_ids' missing in {fp}. "
                    "Expected the consolidated 2D output format written by "
                    "musun2SimFullPostprocessing.py."
                )
            ev = np.asarray(f["event_ids"][:], dtype=np.int64)
            if ev.ndim != 2 or ev.shape[1] < 3:
                raise RuntimeError(
                    f"[schema] {label}: 'event_ids' has unexpected shape "
                    f"{ev.shape} in {fp} (expected (N,3))."
                )
            # Resolve column order from event_id_columns when present.
            col_idx = {"run_id": 0, "muon_id": 1, "nc_id": 2}
            if "event_id_columns" in f:
                cols = [_decode(c) for c in f["event_id_columns"][:]]
                try:
                    col_idx = {
                        "run_id": cols.index("run_id"),
                        "muon_id": cols.index("muon_id"),
                        "nc_id": cols.index("nc_id"),
                    }
                except ValueError as exc:
                    raise RuntimeError(
                        f"[schema] {label}: event_id_columns={cols} does not "
                        f"contain run_id/muon_id/nc_id in {fp}."
                    ) from exc
            ordered = np.column_stack(
                (ev[:, col_idx["run_id"]], ev[:, col_idx["muon_id"]], ev[:, col_idx["nc_id"]])
            ).astype(np.int64)
            chunks.append(ordered)

    all_arr = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), np.int64)

    duplicates = _find_duplicate_keys(all_arr)
    if duplicates.size:
        dup_keys = _array_to_keyset(duplicates)
        raise RuntimeError(
            "NC consistency check FAILED — unexpected event duplication\n"
            f"  Stage    : load {label}\n"
            f"  Dataset  : {label} (event_ids)\n"
            f"  Detail   : {len(dup_keys)} NC key(s) written more than once; "
            "each capture must appear exactly once in the output.\n"
            f"  Examples : {_format_examples(dup_keys)}"
        )

    ids = _array_to_keyset(all_arr)
    per_run: Dict[int, int] = {}
    for r, _m, _n in ids:
        per_run[r] = per_run.get(r, 0) + 1

    ds = NCDataset(
        name=label,
        ids=ids,
        is_full=True,
        source="event_ids",
        n_rows_read=len(all_arr),
        n_duplicates=len(all_arr) - len(ids),
        per_run_counts=per_run,
    )
    return ds


# ---------------------------------------------------------------------------
# Reporting & comparison
# ---------------------------------------------------------------------------

def log_dataset_summary(ds: NCDataset) -> None:
    """Log a one-block summary of a loaded dataset."""
    logger.info("-" * 70)
    logger.info("Dataset : %s", ds.name)
    logger.info("  source        : %s", ds.source)
    logger.info("  complete set  : %s", "yes" if ds.is_full else "NO (subset)")
    logger.info("  unique NCs    : %d", len(ds.ids))
    logger.info("  rows read     : %d  (duplicates collapsed: %d)",
                ds.n_rows_read, ds.n_duplicates)
    logger.info("  runs          : %d  %s",
                len(ds.run_ids), sorted(ds.run_ids)[:20])


def _log_overlap(a: NCDataset, b: NCDataset) -> Tuple[Set[NCKey], Set[NCKey]]:
    """Log overlap statistics and return ``(only_in_a, only_in_b)``."""
    only_a = a.ids - b.ids
    only_b = b.ids - a.ids
    inter = a.ids & b.ids
    union = len(a.ids | b.ids)
    jaccard = (len(inter) / union) if union else 1.0
    logger.info(
        "  overlap %-22s |%-22s : shared=%d  only_%s=%d  only_%s=%d  jaccard=%.6f",
        a.name, b.name, len(inter), a.name, len(only_a), b.name, len(only_b), jaccard,
    )
    return only_a, only_b


def _raise_mismatch(
    stage: str,
    a: NCDataset,
    b: NCDataset,
    only_a: Set[NCKey],
    only_b: Set[NCKey],
    extra_hint: str = "",
) -> None:
    """Raise a self-contained RuntimeError describing an ID-set mismatch."""
    lines = [
        "NC consistency check FAILED",
        f"  Stage      : {stage}",
        f"  Datasets   : '{a.name}'  vs  '{b.name}'",
        f"  Sources    : {a.source}  |  {b.source}",
        f"  Sizes      : {a.name}={len(a.ids)}  {b.name}={len(b.ids)}",
        f"  Mismatches : {len(only_a)} only in '{a.name}', "
        f"{len(only_b)} only in '{b.name}'",
    ]
    if only_a:
        lines.append(f"  In '{a.name}' but NOT in '{b.name}': {_format_examples(only_a)}")
    if only_b:
        lines.append(f"  In '{b.name}' but NOT in '{a.name}': {_format_examples(only_b)}")
    if only_a and only_b:
        lines.append(
            "  NOTE: presence on BOTH sides typically means a capture identifier "
            "was altered (e.g. wrong run_id/muon_id/nc_id), not merely missing."
        )
    if extra_hint:
        lines.append(f"  Hint       : {extra_hint}")
    raise RuntimeError("\n".join(lines))


def compare(stage: str, a: NCDataset, b: NCDataset) -> None:
    """Compare two datasets' NC identity sets and raise on inconsistency.

    Semantics
    ---------
    * Both complete  -> require EXACT set equality.
    * One a subset   -> require the subset to be fully traceable into the full
      set (no untraceable ids); report overlap fraction but do not require the
      reverse direction.

    Raises
    ------
    RuntimeError
        On any disallowed difference.
    """
    logger.info("=" * 70)
    logger.info("STAGE  %s", stage)
    only_a, only_b = _log_overlap(a, b)

    if a.is_full and b.is_full:
        if only_a or only_b:
            _raise_mismatch(stage, a, b, only_a, only_b)
        logger.info("  PASS  exact equality (%d NCs identical).", len(a.ids))
        return

    # Subset semantics (optical fallback present on one side).
    subset, full = (a, b) if not a.is_full else (b, a)
    untraceable = subset.ids - full.ids
    if untraceable:
        _raise_mismatch(
            stage, subset, full, untraceable, set(),
            extra_hint=(
                f"'{subset.name}' is an optical-hit subset; every one of its NCs "
                f"must still trace back to '{full.name}'. The listed ids cannot."
            ),
        )
    frac = len(subset.ids) / len(full.ids) if full.ids else 1.0
    missing_in_subset = len(full.ids) - len(subset.ids & full.ids)
    logger.warning(
        "  PASS (subset mode): all %d NCs of '%s' trace into '%s'. "
        "Overlap=%.4f; %d NC(s) of '%s' have no detected photons (expected).",
        len(subset.ids), subset.name, full.name, frac, missing_in_subset, full.name,
    )
    if frac < 0.01:
        raise RuntimeError(
            "NC consistency check FAILED — suspiciously low overlap\n"
            f"  Stage      : {stage}\n"
            f"  Datasets   : '{subset.name}' (subset) vs '{full.name}' (full)\n"
            f"  Overlap    : {frac:.6f} ({len(subset.ids)} / {len(full.ids)})\n"
            "  Detail     : near-zero overlap strongly suggests the two formats "
            "were seeded from DIFFERENT NC-truth simulations."
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_validation(
    sim1_path: str,
    raw_ssd_path: str,
    postprocessed_path: str,
    pmt_path: Optional[str],
    allow_optical_fallback: bool = True,
    multi_run_ssd: bool = False,
) -> None:
    """Load every data format and run all consistency stages.

    Raises
    ------
    RuntimeError
        On the first detected inconsistency or I/O failure.
    """
    logger.info("#" * 70)
    logger.info("# NEUTRON CAPTURE CONSISTENCY VALIDATION")
    logger.info("#" * 70)

    # --- Load all formats -------------------------------------------------
    logger.info(">>> Loading sim1 NC truth ...")
    sim1 = load_sim1(sim1_path)
    logger.info(">>> Loading raw SSD NC ids ...")
    raw_ssd = load_raw_sim(
        raw_ssd_path, "raw SSD", allow_optical_fallback, multi_run=multi_run_ssd
    )
    logger.info(">>> Loading post-processed SSD NC ids ...")
    post = load_postprocessed(postprocessed_path)
    pmt: Optional[NCDataset] = None
    if pmt_path:
        logger.info(">>> Loading raw PMT NC ids ...")
        pmt = load_raw_sim(pmt_path, "raw PMT", allow_optical_fallback)

    # --- Summaries --------------------------------------------------------
    logger.info("")
    logger.info("DATASET SUMMARIES")
    for ds in [sim1, raw_ssd, post] + ([pmt] if pmt else []):
        log_dataset_summary(ds)

    # --- Step 2A: sim1  <->  post-processed SSD ---------------------------
    compare("2A. sim1  <->  post-processed SSD", sim1, post)

    # --- Step 2 (cross): sim1  <->  raw SSD -------------------------------
    compare("2.   sim1  <->  raw SSD", sim1, raw_ssd)

    # --- Step 2B: raw SSD  <->  post-processed SSD ------------------------
    compare("2B. raw SSD  <->  post-processed SSD", raw_ssd, post)

    # --- Step 3: PMT consistency -----------------------------------------
    if pmt is not None:
        compare("3A. PMT  <->  sim1", pmt, sim1)
        compare("3B. PMT  <->  raw SSD", pmt, raw_ssd)
        compare("3C. PMT  <->  post-processed SSD", pmt, post)
    else:
        logger.warning("No --pmt path provided; skipping Step 3 (PMT checks).")

    logger.info("=" * 70)
    logger.info("ALL CONSISTENCY CHECKS PASSED — NC ground truth is identical "
                "across all provided data formats.")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="validate_neutron_capture_consistency.py",
        description=(
            "Strict validator: verify the neutron-capture ground truth "
            "(run_id, muon_id, nc_id) is identical across sim1, raw SSD, "
            "post-processed SSD, and raw PMT simulations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sim1", required=True,
                        help="sim1 NC-truth directory (/hit/MyNeutronCaptureOutput).")
    parser.add_argument("--raw-ssd", required=True,
                        help="Raw SSD sim2 directory (nested run_NNN; /hit/DebugVertices).")
    parser.add_argument("--postprocessed-ssd", required=True,
                        help="Post-processed SSD output file (ncscore_output_*.hdf5) "
                             "or directory containing it.")
    parser.add_argument("--pmt", default=None,
                        help="Raw PMT sim2 directory (nested run_NNN; /hit/DebugVertices). "
                             "Optional; omit to skip Step 3.")
    parser.add_argument("--multi-run", action="store_true",
                        help="Raw SSD uses the run_NNN/sim_MMM/output_*.hdf5 layout; "
                             "one sim_MMM is chosen per run. (Applies to --raw-ssd only.)")
    parser.add_argument("--no-optical-fallback", action="store_true",
                        help="Disallow falling back to /hit/optical when "
                             "/hit/DebugVertices is missing (fail instead).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns process exit code (0 ok, 1 on failure)."""
    args = parse_arguments(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_validation(
            sim1_path=args.sim1,
            raw_ssd_path=args.raw_ssd,
            postprocessed_path=args.postprocessed_ssd,
            pmt_path=args.pmt,
            allow_optical_fallback=not args.no_optical_fallback,
            multi_run_ssd=args.multi_run,
        )
    except RuntimeError as exc:
        logger.error("\n%s", exc)
        logger.error("VALIDATION FAILED.")
        return 1
    except Exception as exc:  # noqa: BLE001 — surface anything unexpected clearly
        logger.exception("Unexpected error during validation: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
