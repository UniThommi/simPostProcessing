#!/usr/bin/env python3
"""
Neutron Capture Count Verification Script

Vergleicht die Anzahl der Neutron Captures zwischen:
1. Raw simulation data (input files)
2. Postprocessed ML format data (train + validation files)

Autorin: Scientific Data Pipeline Verification
Datum: 2026-01-02
"""

import os
import sys
import h5py
import argparse
from pathlib import Path
from typing import Set, Tuple, Dict, List
import glob
from collections import defaultdict


def find_simulation_files(base_path: Path, nested: bool = True) -> list[Path]:
    """
    Findet alle Simulationsdateien (output_*.hdf5).
    
    Args:
        base_path: Basisverzeichnis mit Simulationsdaten
        nested: Wenn True, durchsuche Unterordner; sonst nur Hauptordner
        
    Returns:
        Sortierte Liste von Pfaden zu HDF5-Dateien
    """
    files = []
    
    if nested:
        # Suche in Unterordnern (jeder Unterordner = eine Simulation)
        print(f"Durchsuche Unterordner in: {base_path}")
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        print(f"Gefundene Unterordner: {len(subdirs)}")
        
        for subdir in sorted(subdirs):
            subdir_files = sorted(subdir.glob('output_*.hdf5'))
            if subdir_files:
                print(f"  {subdir.name}: {len(subdir_files)} Files")
                files.extend(subdir_files)
    else:
        # Direkte Suche im Hauptordner
        files = sorted(base_path.glob('output_*.hdf5'))
        print(f"Gefundene Files in {base_path}: {len(files)}")
    
    return files


def count_neutron_captures_in_simulation(
    file_path: Path,
    verbose: bool = False
) -> int:
    """
    Zählt eindeutige Neutron Captures in einer Simulationsdatei.
    
    Ein Neutron Capture wird identifiziert durch das Paar (evtid, nC_track_id).
    
    Args:
        file_path: Pfad zur HDF5-Simulationsdatei
        verbose: Detaillierte Ausgabe
        
    Returns:
        Set von (evtid, nC_track_id) Tupeln
        
    Raises:
        OSError: Wenn Datei nicht gelesen werden kann (korrupt)
        KeyError: Wenn erforderliche Datenfelder fehlen
    """
    unique_pairs_count = 0
    total_entries = 0

    try:
        with h5py.File(file_path, 'r') as f:
            # Prüfe ob MyNeutronCaptureOutput existiert
            if 'hit/MyNeutronCaptureOutput' not in f:
                print(f"  ⚠ Warnung: {file_path.name} enthält keine MyNeutronCaptureOutput Gruppe")
                return 0
            
            # Lese evtid und nC_track_id
            evtid = f['hit/MyNeutronCaptureOutput/evtid/pages'][:]
            nC_track_id = f['hit/MyNeutronCaptureOutput/nC_track_id/pages'][:]
            
            assert len(evtid) == len(nC_track_id), "Mismatch between evtid and nC_track_id!"

            total_entries += len(evtid)
            # Erstelle Set von eindeutigen Paaren
            unique_pairs_in_run = set(zip(evtid, nC_track_id))
            unique_pairs_count = len(unique_pairs_in_run)
            
            if verbose:
                print(
                f"  {file_path.name}: "
                f"{unique_pairs_count} eindeutige NCs "
                f"({total_entries} Einträge)")

    except (OSError, KeyError, ValueError) as e:
        # Korrupte oder fehlerhafte Datei
        print(f"  ✗ KORRUPTE DATEI: {file_path.name}")
        print(f"     Fehler: {type(e).__name__}: {e}")
        print(f"     Pfad: {file_path.absolute()}")
        raise  # Re-raise um in count_all_simulation_ncs zu behandeln

    return unique_pairs_count


def count_all_simulation_ncs(
    sim_files: list[Path],
    verbose: bool = False,
    progress_interval: int = 50
) -> Tuple[int, list[Path]]:
    """
    Zählt alle eindeutigen Neutron Captures über alle Simulationsdateien.
    
    Args:
        sim_files: Liste von Simulationsdateien
        verbose: Detaillierte Ausgabe pro File
        progress_interval: Fortschrittsanzeige alle N Dateien
        
    Returns:
        Tuple von (Gesamtzahl eindeutiger NCs, Liste korrupter Dateien)
    """
    print(f"\n{'='*70}")
    print(f"SIMULATIONSDATEN ANALYSE")
    print(f"{'='*70}")
    print(f"Zu verarbeitende Dateien: {len(sim_files)}\n")
    
    total_nc_count = 0
    files_processed = 0
    files_failed = 0
    corrupt_files = []
    
    for idx, file_path in enumerate(sim_files, 1):
        try:
            nc_count = count_neutron_captures_in_simulation(file_path, verbose)
            total_nc_count += nc_count
            files_processed += 1
            
            # Fortschrittsanzeige
            if idx % progress_interval == 0:
                print(f"  Fortschritt: {idx}/{len(sim_files)} Files verarbeitet "
                      f"({files_processed} erfolgreich, {files_failed} fehlgeschlagen)")
                print(f"  Akkumulierte NCs (über alle Runs): {total_nc_count}")
                
        except (OSError, KeyError, ValueError) as e:
            files_failed += 1
            corrupt_files.append(file_path)
            print(f"  → Datei wird übersprungen und protokolliert\n")
            continue
    
    print(f"\n{'='*70}")
    print(f"SIMULATION ZUSAMMENFASSUNG:")
    print(f"  Erfolgreich verarbeitet: {files_processed}/{len(sim_files)}")
    print(f"  Fehlgeschlagen (korrupt): {files_failed}")
    print(f"  Neutron Captures gesamt (Run-lokal eindeutig): {total_nc_count}")
    
    if corrupt_files:
        print(f"\n  ⚠ KORRUPTE DATEIEN ({len(corrupt_files)}):")
        for cf in corrupt_files:
            print(f"     - {cf.absolute()}")
    
    print(f"{'='*70}\n")
    
    return total_nc_count, corrupt_files

def count_postprocessed_entries(output_path: Path) -> Dict[str, int]:
    """
    Zählt Einträge in postprocessed Train und Validation Dateien.
    
    Args:
        output_path: Pfad zum Ordner mit resum_output_*_train.hdf5 und 
                     resum_output_*_validation.hdf5
        
    Returns:
        Dictionary mit 'train', 'validation', 'total' counts
        
    Raises:
        FileNotFoundError: Wenn Train- oder Val-Datei nicht gefunden
    """
    print(f"\n{'='*70}")
    print(f"POSTPROCESSED DATEN ANALYSE")
    print(f"{'='*70}")
    
    # Suche nach Train/Val Dateien
    train_files = sorted(output_path.glob('resum_output_*_train.hdf5'))
    val_files = sorted(output_path.glob('resum_output_*_validation.hdf5'))
    
    if not train_files:
        raise FileNotFoundError(f"Keine Train-Datei gefunden in {output_path}")
    if not val_files:
        raise FileNotFoundError(f"Keine Validation-Datei gefunden in {output_path}")
    
    print(f"Train Dateien: {[f.name for f in train_files]}")
    print(f"Validation Dateien: {[f.name for f in val_files]}\n")
    
    counts = {'train': 0, 'validation': 0}
    
    # Zähle Train Einträge
    for train_file in train_files:
        try:
            with h5py.File(train_file, 'r') as f:
                # Jeder Eintrag = ein Neutron Capture
                # Wir können eine beliebige phi-Spalte nehmen
                if 'phi' in f and 'xNC_mm' in f['phi']:
                    n_entries = len(f['phi']['xNC_mm'][:])
                    counts['train'] += n_entries
                    print(f"  {train_file.name}: {n_entries} Einträge")
                else:
                    print(f"  ⚠ Warnung: {train_file.name} hat unerwartete Struktur")
        except (OSError, KeyError, ValueError) as e:
            print(f"  ✗ KORRUPTE DATEI: {train_file}")
            print(f"     Fehler: {e}")
            print(f"     Pfad: {train_file.absolute()}")
            raise RuntimeError(f"Train-Datei ist korrupt: {train_file}") from e
    
    # Zähle Validation Einträge
    for val_file in val_files:
        try:
            with h5py.File(val_file, 'r') as f:
                if 'phi' in f and 'xNC_mm' in f['phi']:
                    n_entries = len(f['phi']['xNC_mm'][:])
                    counts['validation'] += n_entries
                    print(f"  {val_file.name}: {n_entries} Einträge")
                else:
                    print(f"  ⚠ Warnung: {val_file.name} hat unerwartete Struktur")
        except (OSError, KeyError, ValueError) as e:
            print(f"  ✗ KORRUPTE DATEI: {val_file}")
            print(f"     Fehler: {e}")
            print(f"     Pfad: {val_file.absolute()}")
            raise RuntimeError(f"Validation-Datei ist korrupt: {val_file}") from e
    
    counts['total'] = counts['train'] + counts['validation']
    
    print(f"\n{'='*70}")
    print(f"POSTPROCESSED ZUSAMMENFASSUNG:")
    print(f"  Train Einträge: {counts['train']}")
    print(f"  Validation Einträge: {counts['validation']}")
    print(f"  Total Einträge: {counts['total']}")
    print(f"  Val Anteil: {counts['validation']/counts['total']*100:.2f}%")
    print(f"{'='*70}\n")
    
    return counts


def compare_counts(sim_count: int, postproc_count: int) -> None:
    """
    Vergleicht und visualisiert die Neutron Capture Counts.
    
    Args:
        sim_count: Anzahl NCs in Simulationsdaten
        postproc_count: Anzahl NCs in postprocessed Daten
    """
    print(f"\n{'='*70}")
    print(f"VERGLEICH: SIMULATION vs POSTPROCESSING")
    print(f"{'='*70}")
    
    difference = sim_count - postproc_count
    if sim_count > 0:
        recovery_rate = (postproc_count / sim_count) * 100
    else:
        recovery_rate = 0.0
    
    print(f"Simulation (Input):      {sim_count:>12,} Neutron Captures")
    print(f"Postprocessed (Output):  {postproc_count:>12,} Neutron Captures")
    print(f"{'-'*70}")
    print(f"Differenz:               {difference:>12,} NCs")
    print(f"Recovery Rate:           {recovery_rate:>11.2f}%")
    
    if difference == 0:
        print(f"\n✓ PERFEKT: Alle Neutron Captures wurden verarbeitet!")
    elif difference > 0:
        loss_rate = (difference / sim_count) * 100
        print(f"\n⚠ VERLUST: {loss_rate:.2f}% der NCs fehlen im Output")
        print(f"  Mögliche Ursachen:")
        print(f"  - Filterung von Photonen außerhalb des Detektorvolumens")
        print(f"  - NC Events ohne detektierte optische Photonen")
        print(f"  - Fehler während des Postprocessings")
    else:
        print(f"\n✗ FEHLER: Mehr Einträge im Output als in Simulation!")
        print(f"  Dies sollte nicht vorkommen - bitte Daten prüfen")
    
    print(f"{'='*70}\n")


def parse_arguments() -> argparse.Namespace:
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(
        description='Verify Neutron Capture counts: Simulation vs Postprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Mit nested Structure (Standard)
  python %(prog)s \\
    --sim-path /pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/rawHomogeneousNCsSSD300PMTs \\
    --output-path /pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/MLFormatHomogeneousNCsZylSSD300PMTs/ \\
    --nested
  
  # Ohne nested Structure
  python %(prog)s \\
    --sim-path /path/to/simulation_files \\
    --output-path /path/to/output_files
  
  # Mit verbose Output
  python %(prog)s -s /path/sim -o /path/out -n -v
        """
    )
    
    parser.add_argument(
        '-s', '--sim-path',
        type=Path,
        required=True,
        help='Pfad zu Simulationsdaten (mit oder ohne Unterordner)'
    )
    
    parser.add_argument(
        '-o', '--output-path',
        type=Path,
        required=True,
        help='Pfad zu postprocessed Daten (enthält train/val HDF5 Files)'
    )
    
    parser.add_argument(
        '-n', '--nested',
        action='store_true',
        help='Simulationsdaten liegen in Unterordnern'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Detaillierte Ausgabe pro File'
    )
    
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=100,
        help='Fortschrittsanzeige alle N Dateien (default: 100)'
    )
    
    return parser.parse_args()


def main():
    """Hauptfunktion des Verifikationsskripts."""
    args = parse_arguments()
    
    # Validiere Pfade
    if not args.sim_path.exists():
        print(f"✗ Fehler: Simulationspfad existiert nicht: {args.sim_path}")
        sys.exit(1)
    
    if not args.output_path.exists():
        print(f"✗ Fehler: Output-Pfad existiert nicht: {args.output_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"NEUTRON CAPTURE COUNT VERIFICATION")
    print(f"{'='*70}")
    print(f"Simulationsdaten: {args.sim_path}")
    print(f"Postprocessed:    {args.output_path}")
    print(f"Nested Structure: {'Ja' if args.nested else 'Nein'}")
    print(f"Verbose:          {'Ja' if args.verbose else 'Nein'}")
    print(f"{'='*70}")
    
    try:
        # 1. Finde alle Simulationsdateien
        sim_files = find_simulation_files(args.sim_path, args.nested)
        
        if not sim_files:
            print(f"✗ Fehler: Keine output_*.hdf5 Dateien gefunden!")
            sys.exit(1)
        
        # 2. Zähle Neutron Captures in Simulation
        sim_nc_count, corrupt_sim_files = count_all_simulation_ncs(
            sim_files,
            verbose=args.verbose,
            progress_interval=args.progress_interval
        )
        
        # 3. Zähle Einträge in postprocessed Files
        postproc_counts = count_postprocessed_entries(args.output_path)
        
        # 4. Vergleiche
        compare_counts(sim_nc_count, postproc_counts['total'])
        
        # Zusätzliche Statistik: Train/Val Split Validation
        if postproc_counts['total'] > 0:
            print(f"TRAIN/VAL SPLIT VALIDATION:")
            print(f"  Train:      {postproc_counts['train']:>12,} "
                  f"({postproc_counts['train']/postproc_counts['total']*100:.2f}%)")
            print(f"  Validation: {postproc_counts['validation']:>12,} "
                  f"({postproc_counts['validation']/postproc_counts['total']*100:.2f}%)")
        
        # Ausgabe korrupter Dateien
        if corrupt_sim_files:
            print(f"\n{'='*70}")
            print(f"KORRUPTE SIMULATIONSDATEIEN ({len(corrupt_sim_files)}):")
            print(f"{'='*70}")
            for corrupt_file in corrupt_sim_files:
                print(f"  {corrupt_file}")
            print(f"{'='*70}")
            
            # Optional: Speichere in Logfile
            corrupt_log = args.output_path / "corrupt_simulation_files.txt"
            with open(corrupt_log, 'w') as f:
                f.write(f"# Korrupte Simulationsdateien\n")
                f.write(f"# Gefunden am: {__import__('datetime').datetime.now()}\n")
                f.write(f"# Anzahl: {len(corrupt_sim_files)}\n\n")
                for corrupt_file in corrupt_sim_files:
                    f.write(f"{corrupt_file}\n")
            print(f"\nKorrupte Dateien wurden gespeichert in: {corrupt_log}")
        
        print(f"\n✓ Analyse erfolgreich abgeschlossen\n")
        
    except Exception as e:
        print(f"\n✗ Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()