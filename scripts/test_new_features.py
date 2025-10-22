#!/usr/bin/env python3
"""
Test script for new features:
1. Ligand parameterization with antechamber + GAFF
2. Modified RNA residue handling
"""

import subprocess
import sys
from pathlib import Path

def check_command(cmd):
    """Check if a command exists"""
    try:
        result = subprocess.run(
            ["which", cmd],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ {cmd} found: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ {cmd} not found")
            return False
    except Exception as e:
        print(f"✗ Error checking {cmd}: {e}")
        return False

def check_python_modules():
    """Check required Python modules"""
    required_modules = [
        'MDAnalysis',
        'Bio',
        'pandas',
        'parmed'
    ]

    all_good = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ Python module '{module}' found")
        except ImportError:
            print(f"✗ Python module '{module}' not found")
            all_good = False

    return all_good

def main():
    print("="*70)
    print("Testing New Feature Dependencies")
    print("="*70)

    # Check Amber tools needed for ligand parameterization
    print("\n1. Checking Amber tools for ligand parameterization:")
    antechamber_ok = check_command("antechamber")
    parmchk2_ok = check_command("parmchk2")
    tleap_ok = check_command("tleap")

    # Check other tools
    print("\n2. Checking other dependencies:")
    pdb4amber_ok = check_command("pdb4amber")

    # Check Python modules
    print("\n3. Checking Python modules:")
    modules_ok = check_python_modules()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if antechamber_ok and parmchk2_ok and tleap_ok:
        print("✓ Ligand parameterization: READY")
        print("  - antechamber, parmchk2, and tleap are available")
        print("  - Can parameterize ligands with GAFF force field")
    else:
        print("✗ Ligand parameterization: NOT READY")
        print("  - Missing: ", end="")
        missing = []
        if not antechamber_ok:
            missing.append("antechamber")
        if not parmchk2_ok:
            missing.append("parmchk2")
        if not tleap_ok:
            missing.append("tleap")
        print(", ".join(missing))

    print()

    if antechamber_ok and parmchk2_ok and tleap_ok:
        print("✓ Modified RNA handling: READY")
        print("  - Can parameterize modified RNA residues using GAFF")
    else:
        print("✗ Modified RNA handling: NOT READY")
        print("  - Requires same tools as ligand parameterization")

    print()

    if modules_ok:
        print("✓ Python dependencies: READY")
    else:
        print("✗ Python dependencies: INCOMPLETE")
        print("  - Install missing modules with: pip install <module>")

    print("\n" + "="*70)

    # Return exit code
    all_ready = (antechamber_ok and parmchk2_ok and tleap_ok and
                 pdb4amber_ok and modules_ok)

    if all_ready:
        print("✓ All systems ready!")
        return 0
    else:
        print("⚠ Some dependencies missing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
