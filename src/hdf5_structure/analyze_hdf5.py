# Author: Thomas Buerger (University of Tübingen)
#
# Usage:
#   python src/hdf5_structure/analyze_hdf5.py /path/to/file.hdf5
#   python src/hdf5_structure/analyze_hdf5.py /path/to/file.hdf5 --output my_structure.txt
#
# Output is written to hdf5_structure.txt by default (or the path given via --output).

import argparse
import h5py
from pathlib import Path

DEFAULT_OUTPUT_FILE = "hdf5_structure.txt"

def analyze_hdf5_structure(file_path, output_file):
    """
    Analyze HDF5 structure and write it into a text file.
    Groups with >40 entries are truncated to first 5 with summary.
    """

    def write(line):
        with open(output_file, "a") as f:
            f.write(line + "\n")

    def process_group(name, obj, depth=0):
        indent = "  " * depth

        if isinstance(obj, h5py.Dataset):
            write(f"{indent}- Dataset: {name}  shape={obj.shape}  dtype={obj.dtype}")

            # Prüfe ob zugehöriges _columns Dataset existiert
            parent = obj.parent
            base = name.rsplit("/", 1)[-1]
            if base.endswith("_matrix"):
                col_key = base.replace("_matrix", "_columns")
                if col_key in parent:
                    cols = parent[col_key][:]
                    cols = [c.decode() if isinstance(c, bytes) else c for c in cols]
                    total_cols = len(cols)
                    if total_cols > 40:
                        display = cols[:5]
                        write(f"{indent}  Columns ({total_cols}): {display} ... and {total_cols - 5} more")
                    else:
                        write(f"{indent}  Columns ({total_cols}): {list(cols)}")
            return

        if isinstance(obj, h5py.Group):
            keys = list(obj.keys())
            total = len(keys)

            write(f"{indent}- Group: {name}  (#entries: {total})")

            # truncate long lists
            display_keys = keys
            truncated = False

            if total > 40:
                display_keys = keys[:5]
                truncated = True

            for key in display_keys:
                process_group(f"{name}/{key}", obj[key], depth + 1)

            if truncated:
                write(f"{indent}  ... and {total - 5} more entries ...")

    # clear old file
    Path(output_file).unlink(missing_ok=True)

    with h5py.File(file_path, "r") as f:
        write("=== HDF5 Structure (compressed) ===")
        process_group("/", f, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the structure of an HDF5 file.")
    parser.add_argument("hdf5_file", type=Path, help="Path to the HDF5 file to analyze.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_FILE,
        help=f"Output text file (default: {DEFAULT_OUTPUT_FILE})."
    )
    args = parser.parse_args()

    analyze_hdf5_structure(args.hdf5_file, args.output)
    print(f"Done. Structure written to {args.output}.")
