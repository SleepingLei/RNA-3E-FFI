# RNA-3E-FFI: E(3) Equivariant GNN for RNA-Ligand Virtual Screening

E(3)-equivariant Graph Neural Network for RNA-ligand virtual screening. Learns pocket embeddings that align with ligand embeddings in a shared latent space for similarity-based ligand screening.

## Overview

**Goal**: Given an RNA binding pocket, find similar ligands from a library by comparing embeddings in a shared latent space.

**Approach**:
- RNA pocket → E(3) GNN → Pocket embedding
- Ligand → Uni-Mol → Ligand embedding
- Train with contrastive learning to align embeddings
- Virtual screening: Find ligands with similar embeddings to query pocket

## Features

- ✅ **Complete Residue-based Pocket Selection**: Ensures all residues are complete (100% atoms vs 0% with atom-based)
- ✅ **Separated Parameterization**: RNA, ligand, and protein components handled independently
- ✅ **Ligand Parameterization with GAFF**: Automatic antechamber + GAFF2 workflow for small molecules
- ✅ **Modified RNA Support**: Handles non-standard residues (PSU, 5MU, 7MG, etc.) via GAFF
- ✅ **Terminal Atom Cleaning**: Automatic handling of RNA fragment terminals for Amber force field
- ✅ **Robust Processing**: Pre-checks and fallbacks to avoid pdb4amber crashes
- ✅ **E(3) Equivariance**: Rotation and translation invariant pocket encoder
- ✅ **Embedding Alignment**: Learns shared latent space with ligand embeddings (Uni-Mol)

## Project Structure

```
RNA-3E-FFI/
├── scripts/
│   ├── 01_process_data.py       # Main data processing pipeline
│   ├── 02_embed_ligands.py      # Ligand embedding generation
│   ├── 03_build_dataset.py      # Graph construction
│   ├── 04_train_model.py        # Model training
│   └── 05_run_inference.py      # Inference script
├── data/
│   ├── raw/mmCIF/              # Input CIF files from PDB
│   └── processed/              # Processed outputs
│       ├── pockets/            # Pocket PDB files
│       ├── amber/              # Amber parameter files (.prmtop, .inpcrd)
│       ├── graphs/             # PyTorch Geometric graphs
│       └── processing_results.json
├── hariboss/
│   ├── Complexes.csv           # HARIBOSS dataset metadata
│   └── compounds.csv
├── archive/                    # Old versions (for reference only)
│   ├── old_scripts/
│   └── old_docs/
├── requirements.txt
└── README.md
```

## Installation

### 1. Install Dependencies

```bash
mamba create -n RNA pyhon=3.11
mamba activate RNA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install -r requirements.txt

```

### 2. Install AmberTools

AmberTools is required for processing RNA structures. Install via conda:

```bash
mamba install -c conda-forge ambertools
```

### 3. Setup Directories

```bash
python setup_directories.py
```

## Usage

### Step 1: Data Preprocessing

Process RNA-ligand complexes and extract binding pockets:

```bash
python scripts/01_process_data.py \
  --hariboss_csv hariboss/Complexes.csv \
  --output_dir data/processed \
  --pocket_cutoff 5.0 \
  --max_complexes 10  # Optional: limit for testing
```

**What it does**:
1. Loads RNA-ligand complex structures from mmCIF files (in `data/raw/mmCIF/`)
2. Classifies molecules: RNA, modified RNA, ligand, protein, water, ions
3. Defines binding pocket using **complete residues** within cutoff distance
4. Cleans RNA terminal atoms (removes 5' phosphate, 3' hydroxyl)
5. Parameterizes each component with appropriate force field:
   - Standard RNA → RNA.OL3
   - Modified RNA → GAFF2 (via antechamber)
   - Ligands → GAFF2 (via antechamber + AM1-BCC charges)
   - Proteins → ff14SB

**Key Innovations**:
- ✅ Residue-based (not atom-based) pocket selection
- ✅ 100% residue completeness (vs 0% with atom-based)
- ✅ No pdb4amber crashes from O5'/O3' mismatches
- ✅ Automatic ligand parameterization with GAFF
- ✅ Handles modified RNA residues (PSU, 5MU, 7MG, etc.)
- ✅ Biologically meaningful structural units

**Output**:
- `data/processed/pockets/*.pdb` - Pocket structures (RNA + ligand)
- `data/processed/amber/*_rna.prmtop/.inpcrd` - Standard RNA topology/coords
- `data/processed/amber/*_modified_rna.prmtop/.inpcrd` - Modified RNA topology/coords
- `data/processed/amber/*_ligand.prmtop/.inpcrd` - Ligand topology/coords
- `data/processed/amber/*_protein.prmtop/.inpcrd` - Protein topology/coords (if present)
- `data/processed/processing_results.json` - Detailed processing summary

### Step 2: Generate Ligand Embeddings

Extract ligands and compute embeddings using Uni-Mol:

```bash
python scripts/02_embed_ligands.py \
  --hariboss_csv hariboss/Complexes.csv \
  --mmcif_dir data/raw/mmCIF \
  --output_dir data/processed \
  --output_h5 data/processed/ligand_embeddings.h5 \
  --batch_size 32
```

This script will:
- Extract ligands from structures to SDF format
- Generate 3D-aware embeddings using Uni-Mol2
- Save embeddings to HDF5 file

### Step 3: Build Molecular Graphs

Construct PyTorch Geometric graphs from processed pockets:

```bash
python scripts/03_build_dataset.py \
  --hariboss_csv hariboss/Complexes.csv \
  --pocket_dir data/processed/pockets \
  --amber_dir data/processed/amber \
  --output_dir data/processed/graphs \
  --distance_cutoff 4.0
```

This script will:
- Load pocket PDB and AMBER topology files
- Extract node features (atom type, hybridization, charges, etc.)
- Build edges based on distance cutoff
- Save graphs as PyTorch files

### Step 4: Train the Model

Train the E(3) equivariant GNN:

```bash
python scripts/04_train_model.py \
  --hariboss_csv hariboss/Complexes.csv \
  --graph_dir data/processed/graphs \
  --embeddings_path data/processed/ligand_embeddings.h5 \
  --batch_size 16 \
  --num_epochs 100 \
  --lr 1e-4 \
  --num_layers 4 \
  --hidden_irreps "32x0e + 16x1o + 8x2e" \
  --output_dir models/checkpoints
```

Key arguments:
- `--hidden_irreps`: E(3) irreducible representations (scalars + vectors + tensors)
- `--num_layers`: Number of message passing layers
- `--patience`: Early stopping patience
- `--output_dim`: Dimension of pocket embeddings (should match Uni-Mol output)

### Step 5: Run Inference

Use the trained model to find similar ligands for a query pocket:

```bash
python scripts/05_run_inference.py \
  --checkpoint models/checkpoints/best_model.pt \
  --query_graph data/processed/graphs/1abc_LIG.pt \
  --ligand_library data/processed/ligand_embeddings.h5 \
  --top_k 10 \
  --metric euclidean \
  --output results/predictions.json
```

This will:
- Load the trained model
- Predict embedding for query pocket
- Find top-k most similar ligands
- Save results to JSON

## Model Architecture

The model implements an E(3) equivariant Graph Neural Network:

1. **Input Embedding**: Maps node features to E(3) irreps
2. **Message Passing Layers**:
   - Compute relative positions and spherical harmonics
   - Use radial MLPs to weight tensor products
   - Maintain E(3) equivariance through geometric operations
3. **Pooling**: Attention-based weighted pooling of scalar features
4. **Output**: Fixed-size invariant embedding vector

### Key Features

- **E(3) Equivariance**: Invariant to rotations and translations
- **Geometric Features**: Spherical harmonics for directional information
- **Learnable Radial Functions**: Distance-dependent message weighting
- **Attention Pooling**: Learns important binding site regions

## Data Format

### Input Features (per atom)
- Atomic number
- Hybridization (one-hot: SP, SP2, SP3, SP3D, SP3D2)
- Aromaticity
- Degree
- Formal charge
- Partial charge (from AMBER)
- Atom type (from AMBER)

### Graph Structure
- **Nodes**: RNA atoms in binding pocket
- **Edges**: Atom pairs within distance cutoff
- **Node positions**: 3D coordinates

### Target
- Uni-Mol ligand embedding (512-dimensional vector)

## Processing Results Format

`processing_results.json` contains detailed information for each complex:

```json
{
  "pdb_id": "1aju",
  "ligand": "ARG",
  "success": true,
  "components": {
    "rna": {
      "success": true,
      "atoms": 350,
      "residues": 11,
      "prmtop": "data/processed/amber/1aju_ARG_rna.prmtop",
      "inpcrd": "data/processed/amber/1aju_ARG_rna.inpcrd"
    },
    "ligand": {
      "success": true,
      "atoms": 26,
      "prmtop": "data/processed/amber/1aju_ARG_ligand.prmtop",
      "inpcrd": "data/processed/amber/1aju_ARG_ligand.inpcrd"
    },
    "modified_rna": {
      "success": true,
      "atoms": 78,
      "residues": 2,
      "prmtop": "data/processed/amber/1aju_ARG_modified_rna.prmtop",
      "inpcrd": "data/processed/amber/1aju_ARG_modified_rna.inpcrd"
    }
  },
  "errors": []
}
```

## New Features (Latest Update)

### 1. Ligand Parameterization with GAFF

The pipeline now automatically parameterizes small molecule ligands using the GAFF (General Amber Force Field) workflow:

```
Ligand PDB → antechamber → parmchk2 → tleap → prmtop/inpcrd
```

**Features**:
- Automatic atom type assignment (GAFF2)
- AM1-BCC charge calculation
- Generation of missing force field parameters
- Compatible with AMBER topology format

**Usage**: Automatic when processing complexes with ligands

**Testing**:
```bash
python scripts/test_new_features.py
python scripts/demo_new_features.py --summary
```

### 2. Modified RNA Residue Support

Now supports non-standard RNA nucleotides commonly found in biological structures:

**Supported modifications**:
- PSU (Pseudouridine)
- 5MU (5-Methyluridine)
- 5MC (5-Methylcytidine)
- 7MG (7-Methylguanosine)
- And 12+ more modifications

**Approach**:
- Each modified residue parameterized as a small molecule using GAFF
- Multiple modifications combined into single topology
- Automatic charge calculation

**Implementation**: See `scripts/01_process_data.py:491-646`

For detailed documentation, see [`docs/NEW_FEATURES.md`](docs/NEW_FEATURES.md)

## Troubleshooting

### Common Issues

1. **"CIF file not found"**
   - Ensure CIF files are in `data/raw/mmCIF/`
   - Files should be named like `1aju.cif`

2. **"pdb4amber failed"**
   - This is expected and handled automatically
   - Script uses fallback to original PDB

3. **"tleap warnings about gaps"**
   - Normal for pocket fragments (not complete RNA chains)
   - As long as no ERRORS, parameterization succeeded

4. **NumPy compatibility error**
   - Ensure NumPy < 2.0: `pip install 'numpy<2.0'`
   - pdb4amber's parmed library requires older NumPy

5. **"Atom does not have a type" errors**
   - Should not occur with terminal cleaning
   - If it does, check the cleaned PDB files

### Performance

Tested on 3 complexes:

| PDB ID | Ligand | RNA Atoms | RNA Residues | Processing Time |
|--------|--------|-----------|--------------|-----------------|
| 1aju   | ARG    | 350       | 11           | ~0.8s          |
| 1akx   | ARG    | 286       | 9            | ~0.4s          |
| 1am0   | AMP    | 360       | 11           | ~0.4s          |

**Total**: 3 complexes in ~2 seconds

## Citation

If you use this code, please cite the relevant papers:

- **e3nn**: Geiger & Smidt (2022)
- **Uni-Mol**: Zhou et al. (2023)
- **HARIBOSS**: Garc໚-Recio et al. (2022)

## License

This project is provided as-is for research purposes.

## Archive

Old versions and analysis documents are preserved in `archive/` for reference:
- `archive/old_scripts/` - Previous implementations and debug scripts  
- `archive/old_docs/` - Development documentation and crash analysis

These are kept for historical context but are **not needed** for running the pipeline.
