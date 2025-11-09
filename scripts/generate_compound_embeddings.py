#!/usr/bin/env python3
"""
Generate Ligand Embeddings for HARIBOSS Compounds with pH Adjustment
This script:
1. Reads hariboss/compounds.csv
2. Adjusts SMILES protonation state at pH 7.4 using OpenBabel
3. Generates Uni-Mol embeddings
4. Applies normalization using saved parameters
5. Saves in deduplicated format (ligand_name as key)
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import subprocess
import tempfile
from tqdm import tqdm
from unimol_tools import UniMolRepr
import warnings
warnings.filterwarnings('ignore')
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
def process_smiles_with_obabel(smiles, ph=7.4):
    """
    Process SMILES with OpenBabel to adjust protonation state at given pH.
    Args:
        smiles: Input SMILES string
        ph: Target pH value (default: 7.4)
    Returns:
        pH-adjusted SMILES string, or None if processing fails
    """
    try:
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f_in:
            f_in.write(smiles + '\n')
            temp_in = f_in.name
        with tempfile.NamedTemporaryFile(mode='r', suffix='.smi', delete=False) as f_out:
            temp_out = f_out.name
        # Run obabel with pH adjustment
        cmd = ['obabel', temp_in, '-osmi', '-O', temp_out, '-p', str(ph)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            # Clean up
            os.unlink(temp_in)
            os.unlink(temp_out)
            return None
        # Read the processed SMILES
        with open(temp_out, 'r') as f:
            line = f.read().strip()
            if line:
                processed_smiles = line.split()[0]  # First column is SMILES
            else:
                processed_smiles = None
        # Clean up temporary files
        os.unlink(temp_in)
        os.unlink(temp_out)
        return processed_smiles
    except Exception as e:
        return None
def load_normalization_params(norm_params_path):
    """Load normalization parameters."""
    print(f"Loading normalization parameters from {norm_params_path}...")
    params = np.load(norm_params_path)
    # Check for different possible key names
    if 'ligand_mean' in params:
        mean = params['ligand_mean']
        std = params['ligand_std']
    elif 'mean' in params:
        mean = params['mean']
        std = params['std']
    else:
        raise ValueError(f"Cannot find normalization parameters in {norm_params_path}. "
                        f"Available keys: {list(params.keys())}")
    print(f"✓ Loaded normalization params: mean shape={mean.shape}, std shape={std.shape}")
    return mean, std
def normalize_embedding(embedding, mean, std):
    """Apply z-score normalization."""
    return (embedding - mean) / (std + 1e-8)
def generate_ligand_embeddings(compounds_csv, output_h5, norm_params_path,
                               ph=7.4, batch_size=32):
    """
    Generate and save ligand embeddings.
    
    Args:
        compounds_csv: Path to hariboss/compounds.csv
        output_h5: Output HDF5 file path
        norm_params_path: Path to normalization parameters .npz
        ph: pH for protonation adjustment
        batch_size: Batch size for inference
    """
    print(f"\n{'='*70}")
    print("Generating Ligand Embeddings with pH Adjustment")
    print(f"{'='*70}\n")
    # Load compounds
    print(f"Loading compounds from {compounds_csv}...")
    df = pd.read_csv(compounds_csv)
    print(f"✓ Loaded {len(df)} compounds")
    # Check required columns
    if 'id' not in df.columns or 'canonical_smiles' not in df.columns:
        available_cols = df.columns.tolist()
        raise ValueError(f"Missing required columns 'id' or 'canonical_smiles'. "
                        f"Available columns: {available_cols}")
    # Load normalization parameters
    mean, std = load_normalization_params(norm_params_path)
    # Process SMILES with pH adjustment
    print(f"\nAdjusting SMILES protonation state at pH {ph}...")
    processed_data = []
    failed_ph_adjustment = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="pH adjustment"):
        ligand_name = row['id']
        smiles = row['canonical_smiles']
        if pd.isna(smiles) or str(smiles).strip() == '':
            failed_ph_adjustment.append({
                'ligand_name': ligand_name,
                'reason': 'empty_smiles'
            })
            continue
        # Adjust pH
        processed_smiles = process_smiles_with_obabel(str(smiles), ph=ph)
        if processed_smiles is None:
            # Fallback to original SMILES if obabel fails
            processed_smiles = str(smiles)
            failed_ph_adjustment.append({
                'ligand_name': ligand_name,
                'reason': 'obabel_failed_using_original'
            })
        processed_data.append({
            'ligand_name': ligand_name,
            'original_smiles': str(smiles),
            'processed_smiles': processed_smiles
        })
    print(f"✓ Processed {len(processed_data)} SMILES")
    if failed_ph_adjustment:
        print(f"⚠️  pH adjustment issues for {len(failed_ph_adjustment)} compounds")
    # Initialize Uni-Mol model
    print(f"\nInitializing Uni-Mol model...")
    try:
        clf = UniMolRepr(
            data_type='molecule',
            remove_hs=False,
            model_name='unimolv2',
            model_size='1.1B',
            compute_atomic_reprs=False
        )
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    # Generate embeddings in batches
    print(f"\nGenerating embeddings for {len(processed_data)} compounds...")
    embeddings_dict = {}
    failed_embeddings = []
    for i in tqdm(range(0, len(processed_data), batch_size), desc="Generating embeddings"):
        batch_data = processed_data[i:i+batch_size]
        batch_smiles = [item['processed_smiles'] for item in batch_data]
        batch_names = [item['ligand_name'] for item in batch_data]
        try:
            # Get representations
            reprs = clf.get_repr(batch_smiles, return_atomic_reprs=False)
            # Extract CLS representations
            if isinstance(reprs, dict) and 'cls_repr' in reprs:
                cls_reprs = reprs['cls_repr']
            else:
                cls_reprs = reprs
            # Store embeddings
            for ligand_name, embedding in zip(batch_names, cls_reprs):
                if isinstance(embedding, np.ndarray):
                    embeddings_dict[ligand_name] = embedding
                else:
                    embeddings_dict[ligand_name] = np.array(embedding)
        except Exception as e:
            print(f"\n⚠️  Error processing batch starting at index {i}: {e}")
            for ligand_name in batch_names:
                failed_embeddings.append({
                    'ligand_name': ligand_name,
                    'reason': f'embedding_error: {str(e)}'
                })
    print(f"\n✓ Generated {len(embeddings_dict)} embeddings")
    # Apply normalization
    print(f"\nApplying normalization...")
    embeddings_normalized = {}
    for ligand_name, embedding in embeddings_dict.items():
        embeddings_normalized[ligand_name] = normalize_embedding(embedding, mean, std)
    # Save to HDF5 in deduplicated format
    print(f"\nSaving {len(embeddings_normalized)} embeddings to {output_h5}...")
    output_path = Path(output_h5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        for ligand_name, embedding in tqdm(embeddings_normalized.items(), desc="Writing HDF5"):
            f.create_dataset(ligand_name, data=embedding, compression='gzip')
    print(f"✓ Saved {len(embeddings_normalized)} embeddings")
    # Report failures
    all_failures = failed_ph_adjustment + failed_embeddings
    if all_failures:
        print(f"\n⚠️  Total failures: {len(all_failures)}")
        # Count by reason
        failure_counts = {}
        for failure in all_failures:
            reason = failure['reason'].split(':')[0]
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
        for reason, count in failure_counts.items():
            print(f"  - {reason}: {count}")
        # Save failure log
        failure_log = output_path.parent / f"{output_path.stem}_failures.csv"
        pd.DataFrame(all_failures).to_csv(failure_log, index=False)
        print(f"  Full failure log: {failure_log}")
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total compounds:      {len(df)}")
    print(f"Successful:           {len(embeddings_normalized)}")
    print(f"Failed:               {len(all_failures)}")
    print(f"Success rate:         {len(embeddings_normalized)/len(df)*100:.2f}%")
    print(f"\nOutput file:          {output_path}")
    print(f"Format:               HDF5 with ligand names as keys (deduplicated)")
    print(f"{'='*70}\n")
def main():
    parser = argparse.ArgumentParser(
        description="Generate normalized Uni-Mol embeddings for HARIBOSS compounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/generate_compound_embeddings.py \\
    --compounds_csv hariboss/compounds.csv \\
    --output_h5 data/processed/compound_embeddings.h5 \\
    --norm_params data/processed/ligand_embedding_norm_params.npz \\
    --ph 7.4
        """
    )
    parser.add_argument("--compounds_csv", type=str, required=True,
                        help="Path to hariboss/compounds.csv")
    parser.add_argument("--output_h5", type=str, required=True,
                        help="Output HDF5 file path")
    parser.add_argument("--norm_params", type=str, required=True,
                        help="Path to normalization parameters .npz file")
    parser.add_argument("--ph", type=float, default=7.4,
                        help="pH for protonation adjustment (default: 7.4)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for inference (default: 32)")
    args = parser.parse_args()
    generate_ligand_embeddings(
        compounds_csv=args.compounds_csv,
        output_h5=args.output_h5,
        norm_params_path=args.norm_params,
        ph=args.ph,
        batch_size=args.batch_size
    )
if __name__ == "__main__":
    main()
