#!/usr/bin/env python3
# Author: Thomas Buerger (University of Tübingen)
"""
Extract the unique physical volume names from the GDML geometry of the simulation.

Searches for <physvol name="..."> entries and writes the unique names to
physvol_names.txt, sorted by length, then alphabetically. The list is a
reference for the volume names that appear in
src/mappings/globalPhysVolMappings.json and in the name patterns of
analyse_physVol_groups.py.

The GDML path is set in main().
"""

import re
import sys
from pathlib import Path
from typing import Set

sys.path.append(str(Path(__file__).resolve().parents[1]))  # src/, for local_paths
from local_paths import required_local_path


def extract_physvol_names(gdml_path: Path) -> Set[str]:
    """
    Extract all unique physical volume names from GDML file.
    
    Args:
        gdml_path: Path to GDML file
        
    Returns:
        Set of unique physical volume name strings
        
    Raises:
        FileNotFoundError: If GDML file doesn't exist
        PermissionError: If file cannot be read
    """
    # Pattern: optional whitespace, <physvol name="...", capture group for name
    # [^"] matches any character except quote (non-greedy capture)
    pattern = re.compile(r'^\s*<physvol\s+name="([^"]+)"')
    
    names: Set[str] = set()
    
    with gdml_path.open('r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                names.add(match.group(1))
    
    return names


def write_sorted_names(names: Set[str], output_path: Path) -> None:
    """
    Write volume names to file, sorted by length then alphabetically.
    
    Args:
        names: Set of volume name strings
        output_path: Path to output text file
    """
    # Sort: primary key = length, secondary key = alphabetical
    sorted_names = sorted(names, key=lambda x: (len(x), x))
    
    with output_path.open('w', encoding='utf-8') as f:
        for name in sorted_names:
            f.write(f"{name}\n")


def main() -> None:
    """Main execution function."""
    
    # Input/Output paths
    # Root directory from local_paths.json (git-ignored)
    gdml_path = Path(required_local_path(
        "cfs", "sim/RMGApplications/01-FullCosmogenics/gdml/currentDistZylPMT300PMTs.gdml"))
    output_path = Path("physvol_names.txt")
    
    # Validate input
    if not gdml_path.exists():
        raise FileNotFoundError(f"GDML file not found: {gdml_path}")
    
    # Extract names
    print(f"Reading GDML file: {gdml_path}")
    names = extract_physvol_names(gdml_path)
    
    # Check for empty result
    if not names:
        print("Warning: No <physvol name=\"...\"> entries found in file")
        return
    
    # Write output
    write_sorted_names(names, output_path)
    print(f"Extracted {len(names)} unique physical volume names")
    print(f"Output written to: {output_path.absolute()}")
    
    # Show first/last few entries as sanity check
    sorted_names = sorted(names, key=lambda x: (len(x), x))
    print("\nFirst 5 entries:")
    for name in sorted_names[:5]:
        print(f"  {name}")
    print("\nLast 5 entries:")
    for name in sorted_names[-5:]:
        print(f"  {name}")


if __name__ == "__main__":
    main()