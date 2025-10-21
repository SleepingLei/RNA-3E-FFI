#!/usr/bin/env python3
"""
Setup script to create the project directory structure.
"""
import os
from pathlib import Path


def setup_directories(project_root=None):
    """
    Create the project directory structure.

    Args:
        project_root: Root directory for the project. If None, uses current directory.
    """
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    # Define directory structure
    directories = [
        "data/raw/mmCIF",
        "data/processed/pockets",
        "data/processed/amber",
        "data/processed/graphs",
        "data/splits",
        "models",
        "scripts",
        "results",
    ]

    print(f"Setting up project directories in: {project_root}")

    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")

    # Note: hariboss directory with compounds.csv and Complexes.csv should already exist
    hariboss_dir = project_root / "hariboss"
    if hariboss_dir.exists():
        print(f"  Found existing: hariboss/")
        if (hariboss_dir / "compounds.csv").exists():
            print(f"    Found: hariboss/compounds.csv")
        else:
            print(f"    WARNING: hariboss/compounds.csv not found")
        if (hariboss_dir / "Complexes.csv").exists():
            print(f"    Found: hariboss/Complexes.csv")
        else:
            print(f"    WARNING: hariboss/Complexes.csv not found")
    else:
        print(f"  WARNING: hariboss/ directory not found")

    print("\nDirectory setup complete!")


if __name__ == "__main__":
    setup_directories()
