#!/usr/bin/env python3
"""
Check that all data products of the MUSUN workflow contain the same neutron captures.

Background
----------
In the two-simulation MUSUN workflow, one NC-truth simulation (sim1) seeds
several optical simulations (SSD, PMT, and optionally repeated SSD runs). If one
of them was accidentally seeded from a different sim1 (e.g. a re-run with
another random seed), nothing crashes, but the data products no longer describe
the same events, and SSD and PMT can no longer be compared fairly.

This script compares the *identity* of the NCs, not photon counts or voxel
hits. Every NC is identified by the key (run_id, muon_id, nc_id), and the key
sets of all data products must be identical.

Where the NC keys are read from
-------------------------------
  sim1 (NC truth)       /hit/MyNeutronCaptureOutput   evtid, nC_track_id
  raw SSD / raw PMT     /hit/DebugVertices            muon_id, nc_id (one row per gamma)
  post-processed SSD    event_ids                     run_id, muon_id, nc_id

For the raw simulations, run_id is taken from the run_NNN directory name
(0 for a flat directory without run_NNN sub-directories).

Rules
-----
* sim1 may contain runs without optical data. The post-processing drops these
  runs, so sim1 is restricted to the runs of the optical simulation it is
  compared with. All other comparisons are strict.
* With --multi-run (layout run_NNN/sim_MMM/), every sim_MMM of a run must
  contain the same NCs.
* If a raw file has no /hit/DebugVertices, the keys are read from /hit/optical.
  This only covers NCs with detected photons, so the check is relaxed: every
  key must exist in the other data product.

Exit code 0 if all checks pass, 1 otherwise (with a diagnostic message).

Usage
-----
  python validate_neutron_capture_consistency.py \\
      --sim1              /path/to/sim1 \\
      --raw-ssd           /path/to/raw_ssd \\
      --postprocessed-ssd /path/to/ncscore_output_0.hdf5 \\
      [--pmt /path/to/raw_pmt] [--multi-run]
"""

from __future__ import annotations

import argparse
import glob
import logging
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np

#: Unique neutron-capture key: (run_id, muon_id, nc_id).
NCKey = Tuple[int, int, int]

logger = logging.getLogger("nc_consistency")

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")
_SIM_DIR_RE = re.compile(r"^sim_(\d+)$")

#: HDF5 file patterns, tried in order; the first one with matches is used.
_FILE_PATTERNS: Tuple[str, ...] = ("output_t*.hdf5", "output_*.hdf5")

#: Maximum number of example keys shown in error messages.
_MAX_EXAMPLES = 12


@dataclass
class NCDataset:
    """NC keys of one data product.

    ``is_full`` is False if the keys come from the /hit/optical fallback and
    therefore only cover NCs with detected photons.
    """

    name: str
    ids: Set[NCKey]
    is_full: bool
    source: str  # HDF5 group/dataset the keys were read from
    n_rows_read: int = 0  # rows read before de-duplication
    per_run_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def run_ids(self) -> Set[int]:
        return set(self.per_run_counts)

    @property
    def n_duplicates(self) -> int:
        return self.n_rows_read - len(self.ids)


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _safe_open(path: Path) -> h5py.File:
    """Open an HDF5 file read-only; raise a RuntimeError if it is corrupt."""
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        raise RuntimeError(
            f"[integrity] Failed to open HDF5 file (corrupt or truncated):\n"
            f"  File : {path}\n"
            f"  Error: {type(exc).__name__}: {exc}"
        ) from exc


def _read_pages(f: h5py.File, parts: Sequence[str], file_path: Path) -> np.ndarray:
    """Read the LGDO column ``<parts>/pages``, e.g. parts=("hit", "optical", "nC_track_id")."""
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
    """Decode an HDF5 string element to ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _require_dir(base: str, label: str) -> Path:
    path = Path(base)
    if not path.is_dir():
        raise RuntimeError(f"[{label}] Input directory does not exist: {path}")
    return path


def _glob_first(directory: Path) -> List[Path]:
    """Return the sorted files of the first pattern in _FILE_PATTERNS that matches."""
    for pat in _FILE_PATTERNS:
        matches = sorted(Path(p) for p in glob.glob(str(directory / pat)))
        if matches:
            return matches
    return []


def _run_dirs(base: Path) -> List[Tuple[int, Path]]:
    """Return all run_NNN sub-directories as (run_id, path), sorted by run_id."""
    runs = []
    for child in base.iterdir():
        m = _RUN_DIR_RE.match(child.name)
        if child.is_dir() and m:
            runs.append((int(m.group(1)), child))
    return sorted(runs)


def discover_run_files(base: str, label: str) -> Dict[int, List[Path]]:
    """Map run_id -> HDF5 files for ``base/run_NNN/`` or a flat ``base/`` (run_id 0)."""
    base_path = _require_dir(base, label)
    run_dirs = _run_dirs(base_path)

    if not run_dirs:
        files = _glob_first(base_path)
        if not files:
            raise RuntimeError(
                f"[{label}] No run_NNN directories and no {list(_FILE_PATTERNS)} "
                f"files in {base_path}."
            )
        logger.info("[%s] Flat layout (run_id=0): %d file(s).", label, len(files))
        return {0: files}

    files_by_run = {run_id: files for run_id, d in run_dirs if (files := _glob_first(d))}
    if not files_by_run:
        raise RuntimeError(
            f"[{label}] None of the run_NNN directories in {base_path} contains "
            f"{list(_FILE_PATTERNS)} files."
        )
    logger.info(
        "[%s] Nested layout: %d run(s), %d file(s).",
        label, len(files_by_run), sum(map(len, files_by_run.values())),
    )
    return files_by_run


def discover_multi_run_files(base: str, label: str) -> Dict[int, Dict[str, List[Path]]]:
    """Map run_id -> {sim_MMM: files} for the layout ``base/run_NNN/sim_MMM/``.

    If a run has no sim_MMM directories, any sub-directory with HDF5 files is used.
    """
    base_path = _require_dir(base, label)
    run_dirs = _run_dirs(base_path)
    if not run_dirs:
        raise RuntimeError(f"[{label}] --multi-run set but no run_NNN directories in {base_path}.")

    sims_by_run: Dict[int, Dict[str, List[Path]]] = {}
    for run_id, run_dir in run_dirs:
        subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
        sim_dirs = sorted(
            (d for d in subdirs if _SIM_DIR_RE.match(d.name)),
            key=lambda d: int(_SIM_DIR_RE.match(d.name).group(1)),  # type: ignore[union-attr]
        ) or sorted(subdirs)
        sims = {d.name: files for d in sim_dirs if (files := _glob_first(d))}
        if sims:
            sims_by_run[run_id] = sims

    if not sims_by_run:
        raise RuntimeError(
            f"[{label}] --multi-run set but no run_NNN/sim_MMM directory in "
            f"{base_path} contains {list(_FILE_PATTERNS)} files."
        )
    logger.info(
        "[%s] Multi-run layout: %d run(s), %d sim directories.",
        label, len(sims_by_run), sum(map(len, sims_by_run.values())),
    )
    return sims_by_run


# ---------------------------------------------------------------------------
# Key utilities
# ---------------------------------------------------------------------------

def _stack_keys(run_id: int, muon: np.ndarray, nc: np.ndarray, file_path: Path) -> np.ndarray:
    """Build an (N, 3) int64 array of (run_id, muon_id, nc_id) keys."""
    if len(muon) != len(nc):
        raise RuntimeError(
            f"[schema] muon/nc column length mismatch in {file_path}: "
            f"{len(muon)} vs {len(nc)}."
        )
    out = np.empty((len(muon), 3), dtype=np.int64)
    out[:, 0] = run_id
    out[:, 1] = muon
    out[:, 2] = nc
    return out


def _array_to_keyset(arr: np.ndarray) -> Set[NCKey]:
    if arr.size == 0:
        return set()
    return set(map(tuple, arr.tolist()))


def _count_per_run(ids: Set[NCKey], runs: Sequence[int] = ()) -> Dict[int, int]:
    """Count NCs per run_id; ``runs`` are included even if they contain no NCs."""
    counts = dict.fromkeys(runs, 0)
    for run_id, _, _ in ids:
        counts[run_id] = counts.get(run_id, 0) + 1
    return counts


def _reject_duplicates(arr: np.ndarray, label: str, source: str, hint: str = "") -> None:
    """Raise if any key occurs more than once (each NC must be stored exactly once)."""
    if arr.size == 0:
        return
    uniq, counts = np.unique(arr, axis=0, return_counts=True)
    dups = _array_to_keyset(uniq[counts > 1])
    if dups:
        raise RuntimeError(
            "NC consistency check FAILED: duplicated NCs\n"
            f"  Dataset  : {label} ({source})\n"
            f"  Detail   : {len(dups)} NC key(s) occur more than once; "
            "each NC must be stored exactly once.\n"
            f"  Examples : {_format_examples(dups)}"
            + (f"\n  Hint     : {hint}" if hint else "")
        )


def _format_examples(keys: Set[NCKey], limit: int = _MAX_EXAMPLES) -> str:
    """Render up to ``limit`` keys as ``(run=.., muon=.., nc=..)``."""
    ordered = sorted(keys)
    rendered = ", ".join(f"(run={r}, muon={m}, nc={n})" for r, m, n in ordered[:limit])
    if len(ordered) > limit:
        rendered += f", ... (+{len(ordered) - limit} more)"
    return rendered or "<none>"


# ---------------------------------------------------------------------------
# Loaders (one per data product)
# ---------------------------------------------------------------------------

def load_sim1(path: str, label: str = "sim1") -> NCDataset:
    """Load the reference NC keys from sim1 (/hit/MyNeutronCaptureOutput)."""
    files_by_run = discover_run_files(path, label)
    chunks: List[np.ndarray] = []
    for run_id, files in sorted(files_by_run.items()):
        for fp in files:
            with _safe_open(fp) as f:
                evt = _read_pages(f, ("hit", "MyNeutronCaptureOutput", "evtid"), fp)
                nc = _read_pages(f, ("hit", "MyNeutronCaptureOutput", "nC_track_id"), fp)
            chunks.append(_stack_keys(run_id, evt, nc, fp))

    all_arr = np.concatenate(chunks) if chunks else np.empty((0, 3), np.int64)
    _reject_duplicates(
        all_arr, label, "/hit/MyNeutronCaptureOutput",
        hint="check_nc_duplicates.py shows which files share the duplicates.",
    )
    ids = _array_to_keyset(all_arr)
    return NCDataset(
        name=label,
        ids=ids,
        is_full=True,
        source="/hit/MyNeutronCaptureOutput",
        n_rows_read=len(all_arr),
        per_run_counts=_count_per_run(ids, list(files_by_run)),
    )


def _read_raw_keys(
    fp: Path, run_id: int, label: str, allow_optical_fallback: bool
) -> Tuple[Set[NCKey], int, str]:
    """Read the NC keys of one raw optical file. Returns (keys, n_rows, source)."""
    with _safe_open(fp) as f:
        hit = f.get("hit")
        if hit is None:
            raise RuntimeError(f"[schema] Missing '/hit' group in {fp}")
        if "DebugVertices" in hit:
            source = "/hit/DebugVertices"
            muon = _read_pages(f, ("hit", "DebugVertices", "muon_id"), fp)
            nc = _read_pages(f, ("hit", "DebugVertices", "nc_id"), fp)
        elif allow_optical_fallback and "optical" in hit:
            source = "/hit/optical"
            muon = _read_pages(f, ("hit", "optical", "muon_track_id"), fp)
            nc = _read_pages(f, ("hit", "optical", "nC_track_id"), fp)
        else:
            raise RuntimeError(
                f"[schema] {label}: file has neither /hit/DebugVertices "
                f"nor a usable /hit/optical group:\n  File: {fp}"
            )
    return _array_to_keyset(_stack_keys(run_id, muon, nc, fp)), len(muon), source


def _check_sims_agree(label: str, run_id: int, sim_sets: Dict[str, Set[NCKey]], source: str) -> None:
    """Raise if the sim_MMM directories of one run contain different NCs."""
    if len(sim_sets) < 2:
        return
    (ref_name, ref_ids), *others = sim_sets.items()
    for name, ids in others:
        if ids != ref_ids:
            run = f"run_{run_id:03d}"
            _raise_mismatch(
                f"{label}: sim directories of {run}",
                NCDataset(f"{run}/{ref_name}", ref_ids, True, source),
                NCDataset(f"{run}/{name}", ids, True, source),
                ref_ids - ids,
                ids - ref_ids,
            )
    logger.info(
        "[%s] run_%03d: all %d sim directories contain the same %d NCs.",
        label, run_id, len(sim_sets), len(ref_ids),
    )


def load_raw_sim(
    path: str,
    label: str,
    allow_optical_fallback: bool = True,
    multi_run: bool = False,
) -> NCDataset:
    """Load the NC keys of a raw optical simulation (SSD or PMT).

    Keys are read from /hit/DebugVertices (one row per gamma, de-duplicated
    here). Without DebugVertices, /hit/optical is used, which only contains NCs
    with detected photons (``is_full=False``). With ``multi_run``, every sim_MMM
    of a run is read and all of them must contain the same NCs.
    """
    if multi_run:
        sims_by_run = discover_multi_run_files(path, label)
    else:
        sims_by_run = {run_id: {"": files} for run_id, files in discover_run_files(path, label).items()}

    ids: Set[NCKey] = set()
    n_rows = 0
    source: Optional[str] = None

    for run_id, sims in sorted(sims_by_run.items()):
        sim_sets: Dict[str, Set[NCKey]] = {}
        for sim_name, files in sims.items():
            sim_ids: Set[NCKey] = set()
            for fp in files:
                keys, rows, file_source = _read_raw_keys(fp, run_id, label, allow_optical_fallback)
                if source is None:
                    source = file_source
                elif file_source != source:
                    raise RuntimeError(
                        f"[schema] {label}: inconsistent NC source across files "
                        f"('{source}' vs '{file_source}'). Refusing to mix DebugVertices "
                        f"and the optical fallback.\n  File: {fp}"
                    )
                sim_ids |= keys
                n_rows += rows
            sim_sets[sim_name] = sim_ids

        # Optical-fallback sets depend on which photons were detected, so
        # repeated simulations of the same NCs legitimately differ.
        if source == "/hit/DebugVertices":
            _check_sims_agree(label, run_id, sim_sets, source)
        for sim_ids in sim_sets.values():
            ids |= sim_ids

    is_full = source == "/hit/DebugVertices"
    if not is_full:
        logger.warning(
            "[%s] /hit/DebugVertices not found, using /hit/optical. This only "
            "contains NCs with detected photons, so the checks for this dataset "
            "are relaxed to: every NC must exist in the other dataset.",
            label,
        )

    return NCDataset(
        name=label,
        ids=ids,
        is_full=is_full,
        source=source or "<none>",
        n_rows_read=n_rows,
        per_run_counts=_count_per_run(ids, list(sims_by_run)),
    )


def load_postprocessed(path: str, label: str = "post-processed SSD") -> NCDataset:
    """Load the NC keys from ``event_ids`` of the post-processed output.

    ``path`` is an ncscore_output_*.hdf5 file or a directory containing them.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(Path(x) for x in glob.glob(str(p / "ncscore_output_*.hdf5")))
        if not files:
            raise RuntimeError(f"[{label}] No ncscore_output_*.hdf5 files in directory {p}")
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
                    "Expected the output format of musun2SimFullPostprocessing.py."
                )
            ev = np.asarray(f["event_ids"][:], dtype=np.int64)
            if ev.ndim != 2 or ev.shape[1] < 3:
                raise RuntimeError(
                    f"[schema] {label}: 'event_ids' has shape {ev.shape} in {fp}, "
                    "expected (N, 3)."
                )
            col_idx = {"run_id": 0, "muon_id": 1, "nc_id": 2}
            if "event_id_columns" in f:
                cols = [_decode(c) for c in f["event_id_columns"][:]]
                try:
                    col_idx = {name: cols.index(name) for name in col_idx}
                except ValueError as exc:
                    raise RuntimeError(
                        f"[schema] {label}: event_id_columns={cols} does not "
                        f"contain run_id/muon_id/nc_id in {fp}."
                    ) from exc
            chunks.append(ev[:, [col_idx["run_id"], col_idx["muon_id"], col_idx["nc_id"]]])

    all_arr = np.concatenate(chunks) if chunks else np.empty((0, 3), np.int64)
    _reject_duplicates(all_arr, label, "event_ids")
    ids = _array_to_keyset(all_arr)
    return NCDataset(
        name=label,
        ids=ids,
        is_full=True,
        source="event_ids",
        n_rows_read=len(all_arr),
        per_run_counts=_count_per_run(ids),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def log_dataset_summary(ds: NCDataset) -> None:
    logger.info("-" * 70)
    logger.info("Dataset : %s", ds.name)
    logger.info("  source        : %s", ds.source)
    logger.info("  complete set  : %s", "yes" if ds.is_full else "NO (subset)")
    logger.info("  unique NCs    : %d", len(ds.ids))
    logger.info("  rows read     : %d  (duplicates collapsed: %d)", ds.n_rows_read, ds.n_duplicates)
    logger.info("  runs          : %d  %s", len(ds.run_ids), sorted(ds.run_ids)[:20])


def restrict_sim1_to_runs_of(sim1: NCDataset, optical: NCDataset) -> NCDataset:
    """Return sim1 limited to the runs of an optical simulation.

    sim1 runs without optical data are dropped by the post-processing and are
    therefore fine. Optical runs that do not exist in sim1 are an error.
    """
    unknown = optical.run_ids - sim1.run_ids
    if unknown:
        raise RuntimeError(
            "NC consistency check FAILED: runs without NC truth\n"
            f"  Dataset : {optical.name}\n"
            f"  Detail  : run(s) {sorted(unknown)} do not exist in '{sim1.name}'."
        )
    ignored = sim1.run_ids - optical.run_ids
    if ignored:
        logger.info(
            "[%s] Ignoring %d run(s) without '%s' data: %s",
            sim1.name, len(ignored), optical.name, sorted(ignored)[:20],
        )
    ids = {key for key in sim1.ids if key[0] in optical.run_ids}
    return replace(
        sim1,
        ids=ids,
        n_rows_read=len(ids),
        per_run_counts={r: n for r, n in sim1.per_run_counts.items() if r in optical.run_ids},
    )


def _log_overlap(a: NCDataset, b: NCDataset) -> Tuple[Set[NCKey], Set[NCKey]]:
    """Log overlap statistics and return ``(only_in_a, only_in_b)``."""
    only_a = a.ids - b.ids
    only_b = b.ids - a.ids
    shared = len(a.ids & b.ids)
    union = len(a.ids | b.ids)
    jaccard = shared / union if union else 1.0
    logger.info(
        "  shared=%d  only in '%s'=%d  only in '%s'=%d  jaccard=%.6f",
        shared, a.name, len(only_a), b.name, len(only_b), jaccard,
    )
    return only_a, only_b


def _raise_mismatch(
    stage: str,
    a: NCDataset,
    b: NCDataset,
    only_a: Set[NCKey],
    only_b: Set[NCKey],
    hint: str = "",
) -> None:
    lines = [
        "NC consistency check FAILED",
        f"  Stage      : {stage}",
        f"  Datasets   : '{a.name}'  vs  '{b.name}'",
        f"  Sources    : {a.source}  |  {b.source}",
        f"  Sizes      : {a.name}={len(a.ids)}  {b.name}={len(b.ids)}",
        f"  Mismatches : {len(only_a)} only in '{a.name}', {len(only_b)} only in '{b.name}'",
    ]
    if only_a:
        lines.append(f"  In '{a.name}' but NOT in '{b.name}': {_format_examples(only_a)}")
    if only_b:
        lines.append(f"  In '{b.name}' but NOT in '{a.name}': {_format_examples(only_b)}")
    if only_a and only_b:
        lines.append(
            "  NOTE: differences on BOTH sides usually mean the datasets were "
            "seeded from different NC-truth simulations, not that NCs are missing."
        )
    if hint:
        lines.append(f"  Hint       : {hint}")
    raise RuntimeError("\n".join(lines))


def compare(stage: str, a: NCDataset, b: NCDataset) -> None:
    """Compare the NC keys of two datasets and raise on any difference.

    Two complete sets must be equal. If one set is an optical-fallback subset,
    all of its keys must exist in the other set, and the overlap must not be
    near zero.
    """
    logger.info("=" * 70)
    logger.info("STAGE  %s", stage)
    only_a, only_b = _log_overlap(a, b)

    if a.is_full and b.is_full:
        if only_a or only_b:
            _raise_mismatch(stage, a, b, only_a, only_b)
        logger.info("  PASS  identical (%d NCs).", len(a.ids))
        return

    subset, full = (a, b) if not a.is_full else (b, a)
    untraceable = subset.ids - full.ids
    if untraceable:
        _raise_mismatch(
            stage, subset, full, untraceable, set(),
            hint=f"every NC of the subset '{subset.name}' must exist in '{full.name}'.",
        )
    frac = len(subset.ids) / len(full.ids) if full.ids else 1.0
    logger.warning(
        "  PASS (subset): all %d NCs of '%s' exist in '%s' (overlap %.4f); "
        "%d NC(s) without detected photons.",
        len(subset.ids), subset.name, full.name, frac, len(full.ids) - len(subset.ids),
    )
    if frac < 0.01:
        raise RuntimeError(
            "NC consistency check FAILED: suspiciously low overlap\n"
            f"  Stage      : {stage}\n"
            f"  Datasets   : '{subset.name}' (subset) vs '{full.name}' (full)\n"
            f"  Overlap    : {frac:.6f} ({len(subset.ids)} / {len(full.ids)})\n"
            "  Detail     : a near-zero overlap suggests the two datasets were "
            "seeded from different NC-truth simulations."
        )


# ---------------------------------------------------------------------------
# Orchestration & CLI
# ---------------------------------------------------------------------------

def run_validation(
    sim1_path: str,
    raw_ssd_path: str,
    postprocessed_path: str,
    pmt_path: Optional[str],
    allow_optical_fallback: bool = True,
    multi_run_ssd: bool = False,
) -> None:
    """Load all data products and compare them; raise RuntimeError on the first problem."""
    logger.info("#" * 70)
    logger.info("# NEUTRON CAPTURE CONSISTENCY VALIDATION")
    logger.info("#" * 70)

    sim1 = load_sim1(sim1_path)
    raw_ssd = load_raw_sim(raw_ssd_path, "raw SSD", allow_optical_fallback, multi_run=multi_run_ssd)
    post = load_postprocessed(postprocessed_path)
    pmt = load_raw_sim(pmt_path, "raw PMT", allow_optical_fallback) if pmt_path else None

    logger.info("")
    logger.info("DATASET SUMMARIES")
    for ds in (sim1, raw_ssd, post, pmt):
        if ds is not None:
            log_dataset_summary(ds)

    # The post-processed SSD output inherits its runs from the raw SSD simulation.
    sim1_ssd = restrict_sim1_to_runs_of(sim1, raw_ssd)
    compare("sim1  vs  raw SSD", sim1_ssd, raw_ssd)
    compare("raw SSD  vs  post-processed SSD", raw_ssd, post)
    compare("sim1  vs  post-processed SSD", sim1_ssd, post)

    if pmt is not None:
        compare("sim1  vs  raw PMT", restrict_sim1_to_runs_of(sim1, pmt), pmt)
        compare("raw SSD  vs  raw PMT", raw_ssd, pmt)
        compare("post-processed SSD  vs  raw PMT", post, pmt)
    else:
        logger.warning("No --pmt path given; skipping the PMT checks.")

    logger.info("=" * 70)
    logger.info("ALL CHECKS PASSED: all data products contain the same NCs.")
    logger.info("=" * 70)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_neutron_capture_consistency.py",
        description=(
            "Check that sim1, raw SSD, post-processed SSD and raw PMT contain "
            "the same neutron captures (run_id, muon_id, nc_id)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sim1", required=True,
                        help="sim1 NC-truth directory (/hit/MyNeutronCaptureOutput).")
    parser.add_argument("--raw-ssd", required=True,
                        help="Raw SSD optical simulation directory.")
    parser.add_argument("--postprocessed-ssd", required=True,
                        help="Post-processed SSD file (ncscore_output_*.hdf5) or its directory.")
    parser.add_argument("--pmt", default=None,
                        help="Raw PMT optical simulation directory (optional).")
    parser.add_argument("--multi-run", action="store_true",
                        help="Raw SSD uses the layout run_NNN/sim_MMM/output_*.hdf5 "
                             "(repeated optical simulations); all sim_MMM are checked.")
    parser.add_argument("--no-optical-fallback", action="store_true",
                        help="Fail instead of using /hit/optical when /hit/DebugVertices is missing.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns the exit code (0 = passed, 1 = failed)."""
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
    except Exception as exc:  # noqa: BLE001  (report anything unexpected clearly)
        logger.exception("Unexpected error during validation: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
