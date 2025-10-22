#!/usr/bin/env python3
"""
Test script to verify that model file naming is handled correctly across all scripts.
"""
import sys
from pathlib import Path
import glob

def test_file_naming_patterns():
    """
    Test that file naming patterns are consistent across the pipeline.
    """
    print("="*80)
    print("Testing File Naming Pattern Handling")
    print("="*80)

    # Test pattern matching logic from 03_build_dataset.py
    print("\n1. Testing pattern matching for graph files...")
    print("-" * 60)

    test_cases = [
        ("1aju_ARG", ["1aju_ARG_model0.pt", "1aju_ARG_model1.pt"]),
        ("1akx_ARG", ["1akx_ARG.pt"]),  # No model number
        ("7ych_GTP", ["7ych_GTP_model0.pt"]),
    ]

    for complex_base, expected_files in test_cases:
        print(f"\nComplex: {complex_base}")
        print(f"Expected files: {expected_files}")

        # Simulate pattern matching
        pattern = f"{complex_base}_model*.pt"
        print(f"Pattern: {pattern}")

        # Would match: any file like {complex_base}_model{N}.pt
        example_matches = [f for f in expected_files if "_model" in f and f.startswith(complex_base)]
        print(f"Would match: {example_matches}")

        # Fallback
        fallback = f"{complex_base}.pt"
        fallback_matches = [f for f in expected_files if f == fallback]
        print(f"Fallback would match: {fallback_matches}")

    # Test embedding key extraction from 04_train_model.py
    print("\n\n2. Testing embedding key extraction...")
    print("-" * 60)

    graph_ids = [
        "1aju_ARG_model0",
        "1aju_ARG_model1",
        "7ych_GTP_model0",
        "1akx_ARG",  # No model number
    ]

    for graph_id in graph_ids:
        if '_model' in graph_id:
            # Extract base ID without model number
            base_id = '_'.join(graph_id.split('_model')[0].split('_'))
        else:
            base_id = graph_id

        print(f"Graph ID: {graph_id:25s} -> Embedding key: {base_id}")

    # Test complex ID creation
    print("\n\n3. Testing complex ID creation...")
    print("-" * 60)

    # Simulate row data from HARIBOSS CSV
    test_rows = [
        {'id': '1aju', 'sm_ligand_ids': "['ARG_.:B/47:A']"},
        {'id': '7ych', 'sm_ligand_ids': "['GTP_.:A/1:N']"},
    ]

    import ast

    for row in test_rows:
        pdb_id = row['id'].lower()
        ligand_str = row['sm_ligand_ids']

        # Parse ligand
        try:
            ligands = ast.literal_eval(ligand_str)
            if not isinstance(ligands, list):
                ligands = [ligand_str]
        except:
            ligands = [ligand_str]

        if ligands and len(ligands) > 0:
            ligand_resname = ligands[0].split('_')[0].split(':')[0]
        else:
            ligand_resname = 'LIG'

        complex_base = f"{pdb_id}_{ligand_resname}"
        print(f"PDB: {pdb_id}, Ligand: {ligand_resname} -> Complex base: {complex_base}")

    print("\n" + "="*80)
    print("Summary of File Naming Convention")
    print("="*80)
    print("""
File naming patterns across the pipeline:

1. 01_process_data.py output:
   - Pocket: {pdb_id}_{ligand}_model{N}_pocket.pdb
   - RNA PDB: {pdb_id}_{ligand}_model{N}_rna.pdb
   - PRMTOP: {pdb_id}_{ligand}_model{N}_rna.prmtop
   - INPCRD: {pdb_id}_{ligand}_model{N}_rna.inpcrd

2. 02_embed_ligands.py output:
   - HDF5 keys: {pdb_id}_{ligand} (NO model number)
   - Reason: Same ligand across all models

3. 03_build_dataset.py output:
   - Graph files: {pdb_id}_{ligand}_model{N}.pt
   - Maps to embedding: {pdb_id}_{ligand}

4. 04_train_model.py:
   - Loads graphs: {pdb_id}_{ligand}_model{N}.pt
   - Maps to embeddings: {pdb_id}_{ligand}
   - All models of same complex use same embedding target

5. 05_run_inference.py:
   - Input: Any .pt graph file
   - Output: Predictions for that specific model
    """)


def test_actual_files(amber_dir="data/processed/amber", graph_dir="data/processed/graphs"):
    """
    Check actual files in the directories.
    """
    print("\n" + "="*80)
    print("Checking Actual Files")
    print("="*80)

    amber_dir = Path(amber_dir)
    graph_dir = Path(graph_dir)

    if amber_dir.exists():
        print(f"\n1. AMBER directory: {amber_dir}")
        print("-" * 60)

        # Count files with model numbers
        model_pdb = list(amber_dir.glob("*_model*_rna.pdb"))
        model_prmtop = list(amber_dir.glob("*_model*_rna.prmtop"))
        no_model_pdb = [f for f in amber_dir.glob("*_rna.pdb") if "_model" not in f.name]
        no_model_prmtop = [f for f in amber_dir.glob("*_rna.prmtop") if "_model" not in f.name]

        print(f"Files WITH model numbers:")
        print(f"  PDB files: {len(model_pdb)}")
        print(f"  PRMTOP files: {len(model_prmtop)}")
        print(f"\nFiles WITHOUT model numbers:")
        print(f"  PDB files: {len(no_model_pdb)}")
        print(f"  PRMTOP files: {len(no_model_prmtop)}")

        if model_pdb:
            print(f"\nSample files with model numbers:")
            for f in sorted(model_pdb)[:5]:
                print(f"  {f.name}")
    else:
        print(f"\n⚠️  AMBER directory not found: {amber_dir}")

    if graph_dir.exists():
        print(f"\n2. Graph directory: {graph_dir}")
        print("-" * 60)

        # Count graph files
        model_graphs = list(graph_dir.glob("*_model*.pt"))
        no_model_graphs = [f for f in graph_dir.glob("*.pt") if "_model" not in f.name]

        print(f"Graph files WITH model numbers: {len(model_graphs)}")
        print(f"Graph files WITHOUT model numbers: {len(no_model_graphs)}")

        if model_graphs:
            print(f"\nSample graph files with model numbers:")
            for f in sorted(model_graphs)[:5]:
                print(f"  {f.name}")
    else:
        print(f"\n⚠️  Graph directory not found: {graph_dir}")


if __name__ == "__main__":
    test_file_naming_patterns()
    test_actual_files()
