#!/usr/bin/env python3
# Author: Thomas Buerger (University of Tübingen)
"""
3D view of the voxel hits of one neutron capture (NC).

Picks one NC (one row) from a post-processed HDF5 file and draws
  * every SSD voxel as a tile, coloured by the number of detected photons
    (the NC's row of target_matrix); voxels without hits are light grey,
  * the NC position (xNC_mm, yNC_mm, zNC_mm from phi_matrix) as a red point,
  * the outline of the SSD cylinder: wall, bottom (pit disk and bot ring) and
    top ring, with the dimensions used by the post-processing scripts.

The tiles are built from the voxel corners stored in the 'voxels' group of the
file. Works with the output of all post-processing scripts:
  - zylFullPostprocessingOptimized.py      (event_id_columns = ["run_id", "nc_id"])
  - musun2SimFullPostprocessing.py         (event_id_columns = ["run_id", "muon_id", "nc_id"])
  - musun2SimMultiOpticalPostprocessing.py (same as musun2Sim)

Choice of the NC: by default a random NC among those with at least --min-hits
detected photons (many NCs have no or only a few hits). The total number of
hits per NC is taken from region_matrix. --seed makes the choice reproducible,
--index selects a row directly. The chosen row is printed, so the same plot can
be made again with --index.

The tiles of the hit voxels are drawn over the translucent grey tiles, so hits
on the far side of the cylinder stay visible; the NC is drawn on top of both.

Usage:
    python src/evaluation/plot_nc_voxel_hits.py --data /path/to/ncscore_output_0.hdf5 \\
        --min-hits 50 --seed 1 --output nc_voxel_hits.png
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# SSD cylinder in mm, as used by the post-processing scripts:
# R_WALL, R_PIT, R_BOT_INNER, R_TOP_INNER from defineZylinder()
# (r_zylinder, r_pit, r_zyl_bot, r_zyl_top), Z_BOT/Z_TOP = z_cut_bot/z_cut_top.
R_WALL = 4300.0       # inner radius of the wall
R_PIT = 3800.0        # radius of the pit disk at the bottom
R_BOT_INNER = 3950.0  # inner radius of the bot ring (outer radius: R_WALL)
R_TOP_INNER = 1200.0  # inner radius of the top ring (outer radius: R_WALL)
Z_BOT = -4979.0       # height of the bottom (pit and bot)
Z_TOP = 3919.0        # height of the top (z_cut_bot + h_zylinder - 2, h_zylinder = 8900)

CHUNK_SIZE = 1_000_000  # rows of region_matrix read at once when selecting the NC


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def choose_nc(f, index=None, min_hits=1, seed=None):
    """Return the row index of the NC to plot.

    With index given, check and return it. Otherwise pick a random row among the
    NCs whose total number of hits (row sum of region_matrix) is >= min_hits.
    """
    n_rows = f["phi_matrix"].shape[0]
    if index is not None:
        if not 0 <= index < n_rows:
            raise ValueError(f"--index {index} is out of range (file has {n_rows:,} NCs)")
        return index

    region = f["region_matrix"]
    candidates = []
    for start in range(0, n_rows, CHUNK_SIZE):
        total = region[start:start + CHUNK_SIZE].sum(axis=1)
        candidates.append(np.flatnonzero(total >= min_hits) + start)
    candidates = np.concatenate(candidates)
    print(f"NCs with at least {min_hits} hit(s): {len(candidates):,} of {n_rows:,}")
    if len(candidates) == 0:
        raise ValueError(f"no NC has at least {min_hits} hit(s); lower --min-hits")
    return int(np.random.default_rng(seed).choice(candidates))


def _tile(corners, layer):
    """Return the 4 corners (in polygon order) of the face of a voxel box.

    The voxels are 1 mm thick boxes. Horizontal voxels (pit, bot, top) are drawn
    at their lower face, wall voxels at their inner face. The corners are not
    stored in polygon order, so they are sorted by angle around the face center.
    """
    if layer == "wall":
        r = np.hypot(corners[:, 0], corners[:, 1])
        face = corners[np.argsort(r)[:4]]
        center = face.mean(axis=0)
        # azimuth relative to the face center, wrapped so that tiles at phi = +-180 deg work
        dphi = np.arctan2(face[:, 1], face[:, 0]) - np.arctan2(center[1], center[0])
        u, v = np.angle(np.exp(1j * dphi)), face[:, 2] - center[2]
    else:
        face = corners[np.argsort(corners[:, 2])[:4]]
        u, v = face[:, 0] - face[:, 0].mean(), face[:, 1] - face[:, 1].mean()
    return face[np.argsort(np.arctan2(v, u))]


def load_voxel_tiles(f, voxel_ids):
    """Read the corners and layer of every voxel and return one tile per voxel.

    The tiles are returned in the order of voxel_ids (= target_columns), so
    tile i belongs to column i of target_matrix.
    """
    tiles = np.empty((len(voxel_ids), 4, 3))
    voxels = f["voxels"]
    for i, vid in enumerate(voxel_ids):
        grp = voxels[vid]
        corners = np.column_stack([grp["corners"][axis][()] for axis in ("x", "y", "z")])
        tiles[i] = _tile(corners, _decode(grp["layer"][()]).lower())
    return tiles


def draw_cylinder(ax, azim):
    """Draw the outline of the SSD: circles of wall, pit, bot and top, and the
    two vertical silhouette lines of the wall for the viewing angle azim."""
    phi = np.linspace(0, 2 * np.pi, 361)
    style = dict(color="0.2", lw=0.8, zorder=3)
    for r, z in ((R_WALL, Z_BOT), (R_PIT, Z_BOT), (R_BOT_INNER, Z_BOT),
                 (R_WALL, Z_TOP), (R_TOP_INNER, Z_TOP)):
        ax.plot(r * np.cos(phi), r * np.sin(phi), np.full_like(phi, z), **style)
    for side in (-90, 90):
        a = np.deg2rad(azim + side)
        ax.plot([R_WALL * np.cos(a)] * 2, [R_WALL * np.sin(a)] * 2, [Z_BOT, Z_TOP], **style)


def plot_nc(data_path, output, index=None, min_hits=1, seed=None,
            elev=25.0, azim=-60.0, cmap="viridis", dpi=150, show=False):
    """Plot the voxel hits of one NC and save the figure to output."""
    with h5py.File(data_path, "r") as f:
        row = choose_nc(f, index, min_hits, seed)
        phi_columns = [_decode(c) for c in f["phi_columns"][:]]
        region_columns = [_decode(c) for c in f["region_columns"][:]]
        event_id_columns = [_decode(c) for c in f["event_id_columns"][:]]
        voxel_ids = [_decode(c) for c in f["target_columns"][:]]

        phi = f["phi_matrix"][row]
        hits = f["target_matrix"][row]
        region_hits = f["region_matrix"][row]
        event_id = f["event_ids"][row]
        print(f"Loading the corners of {len(voxel_ids):,} voxels...")
        tiles = load_voxel_tiles(f, voxel_ids)

    nc_pos = [phi[phi_columns.index(c)] for c in ("xNC_mm", "yNC_mm", "zNC_mm")]
    event_str = ", ".join(f"{c}={int(v)}" for c, v in zip(event_id_columns, event_id))
    region_str = "  ".join(f"{c}: {int(v)}" for c, v in zip(region_columns, region_hits))
    hit = hits > 0
    print(f"Row {row} ({event_str}): {int(hits.sum())} photons on {int(hit.sum())} voxels")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False  # draw order = zorder: grey tiles, hit tiles, outline, NC

    ax.add_collection3d(Poly3DCollection(
        tiles[~hit], facecolors=(0.8, 0.8, 0.8, 0.2), edgecolors=(0.5, 0.5, 0.5, 0.25),
        linewidths=0.2, zorder=1))

    norm = mcolors.Normalize(vmin=1, vmax=max(2, int(hits.max())))
    cm = plt.get_cmap(cmap)
    ax.add_collection3d(Poly3DCollection(
        tiles[hit], facecolors=cm(norm(hits[hit])), edgecolors="k", linewidths=0.2, zorder=2))

    draw_cylinder(ax, azim)
    ax.scatter(*nc_pos, color="red", s=80, edgecolors="k", depthshade=False, zorder=4)

    lim = R_WALL * 1.05
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(Z_BOT, Z_TOP),
           xlabel="x [mm]", ylabel="y [mm]")
    ax.set_zlabel("z [mm]", labelpad=8)
    ax.set_box_aspect((2 * lim, 2 * lim, Z_TOP - Z_BOT))
    ax.view_init(elev=elev, azim=azim)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax,
                        shrink=0.55, pad=0.06)
    cbar.set_label("detected photons per voxel")
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()

    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", color="red", markeredgecolor="k", label="NC position"),
        Line2D([], [], color="0.2", lw=0.8, label="SSD outline (pit, bot, wall, top)"),
    ], loc="upper left")
    ax.set_title(
        f"Voxel hits of one neutron capture ({event_str})\n"
        f"{int(hits.sum())} photons on {int(hit.sum())} voxels   |   {region_str}\n"
        f"NC at x={nc_pos[0]:.0f} mm, y={nc_pos[1]:.0f} mm, z={nc_pos[2]:.0f} mm")

    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="3D view of the voxel hits of one NC from a post-processed HDF5 file."
    )
    parser.add_argument("--data", required=True,
                        help="Post-processed HDF5 file (ncscore_output_*.hdf5)")
    parser.add_argument("--output", default="nc_voxel_hits.png",
                        help="Output image (default: nc_voxel_hits.png)")
    parser.add_argument("--index", type=int, default=None,
                        help="Row of the NC to plot (default: random NC, see --min-hits)")
    parser.add_argument("--min-hits", type=int, default=1,
                        help="Random choice only among NCs with at least this many hits (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed of the random choice, for a reproducible plot (default: none)")
    parser.add_argument("--elev", type=float, default=25.0,
                        help="Elevation of the viewing angle in degrees (default: 25)")
    parser.add_argument("--azim", type=float, default=-60.0,
                        help="Azimuth of the viewing angle in degrees (default: -60)")
    parser.add_argument("--cmap", default="viridis",
                        help="Matplotlib colormap of the hits (default: viridis)")
    parser.add_argument("--dpi", type=int, default=150, help="Resolution of the image (default: 150)")
    parser.add_argument("--show", action="store_true", help="Also open the plot in a window")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"ERROR: file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    try:
        plot_nc(data_path, args.output, index=args.index, min_hits=args.min_hits,
                seed=args.seed, elev=args.elev, azim=args.azim, cmap=args.cmap,
                dpi=args.dpi, show=args.show)
    except (KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
