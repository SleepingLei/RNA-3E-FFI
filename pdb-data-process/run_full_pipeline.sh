#!/bin/bash
# Complete pipeline for processing receptors and ligands
# Run this script in the pdb-data-process directory

set -e  # Exit on error

# Configuration
RECEPTOR_DIR="effect_receptor"
LIGAND_DIR="processed_ligands_effect_1"
OUTPUT_DIR="processed_output"
WORKERS=16

echo "======================================"
echo "RNA-3E-FFI Data Processing Pipeline"
echo "======================================"
echo ""
echo "Configuration:"
echo "  Receptor dir: $RECEPTOR_DIR"
echo "  Ligand dir:   $LIGAND_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Workers:      $WORKERS"
echo "======================================"
echo ""

# Step 1: Convert ligands to SMILES
echo "=== Step 1: Converting ligands to SMILES ==="
python pdb_to_smiles.py \
    --input-dir "$LIGAND_DIR" \
    --output-csv ligands_smiles.csv \
    --ph 7.4 \
    --workers $WORKERS

if [ $? -ne 0 ]; then
    echo "Error: SMILES conversion failed"
    exit 1
fi
echo ""

# Step 2: Parameterize receptors with AMBER
echo "=== Step 2: Parameterizing receptors with AMBER ==="
python parameterize_receptors.py \
    --receptor-dir "$RECEPTOR_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --workers $WORKERS

if [ $? -ne 0 ]; then
    echo "Error: Receptor parameterization failed"
    exit 1
fi
echo ""

# Step 3: Build receptor graphs
echo "=== Step 3: Building receptor graphs ==="
python build_receptor_graphs.py \
    --amber-dir "$OUTPUT_DIR/amber" \
    --output-dir "$OUTPUT_DIR" \
    --distance-cutoff 5.0 \
    --workers $WORKERS

if [ $? -ne 0 ]; then
    echo "Error: Graph construction failed"
    exit 1
fi
echo ""

# Step 4: Generate ligand embeddings
echo "=== Step 4: Generating ligand embeddings ==="
python generate_ligand_embeddings.py \
    --csv-file ligands_smiles.csv \
    --output-h5 "$OUTPUT_DIR/ligand_embeddings.h5" \
    --batch-size 32

if [ $? -ne 0 ]; then
    echo "Error: Ligand embedding generation failed"
    exit 1
fi
echo ""

# Summary
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo ""
echo "Output files:"
echo "  Receptor graphs: $OUTPUT_DIR/graphs/*.pt"
echo "  Ligand embeddings: $OUTPUT_DIR/ligand_embeddings.h5"
echo "  SMILES: ligands_smiles.csv"
echo ""
echo "Statistics:"
ls -lh "$OUTPUT_DIR/graphs/" | wc -l | xargs echo "  Graph files:"
ls -lh "$OUTPUT_DIR/ligand_embeddings.h5" 2>/dev/null || echo "  Embeddings: (not found)"
echo ""
