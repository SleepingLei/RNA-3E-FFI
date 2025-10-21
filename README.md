# RNA-3E-FFI: E(3) Equivariant GNN for RNA-Ligand Virtual Screening

This project implements an E(3) equivariant Graph Neural Network to predict RNA binding pocket embeddings that align with 3D-aware ligand embeddings from Uni-Mol.

## Project Structure

```
RNA-3E-FFI/
├── data/
│   ├── raw/mmCIF/              # Downloaded PDB structures
│   ├── processed/
│   │   ├── pockets/            # Extracted binding pockets
│   │   ├── amber/              # AMBER topology files
│   │   ├── graphs/             # PyTorch Geometric graphs
│   │   └── ligands/            # Ligand SDF files
│   └── splits/                 # Train/val/test splits
├── models/
│   ├── e3_gnn_encoder.py       # E(3) GNN model
│   └── checkpoints/            # Trained model checkpoints
├── scripts/
│   ├── 01_process_data.py      # Data preprocessing
│   ├── 02_embed_ligands.py     # Ligand embedding generation
│   ├── 03_build_dataset.py     # Graph construction
│   ├── 04_train_model.py       # Model training
│   └── 05_run_inference.py     # Inference script
├── hariboss/
│   ├── Complexes.csv           # HARIBOSS dataset
│   └── compounds.csv
├── requirements.txt
├── setup_directories.py
└── README.md
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install AmberTools

AmberTools is required for processing RNA structures. Install via conda:

```bash
conda install -c conda-forge ambertools
```

### 3. Setup Directories

```bash
python setup_directories.py
```

## Usage

### Step 1: Data Preprocessing

Download PDB structures and extract binding pockets:

```bash
python scripts/01_process_data.py \
  --hariboss_csv hariboss/Complexes.csv \
  --output_dir data \
  --pocket_cutoff 5.0 \
  --max_complexes 100  # Optional: limit for testing
```

This script will:
- Download mmCIF files from PDB
- Extract RNA binding pockets around ligands
- Generate AMBER topology files using pdb4amber and tleap

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

## Troubleshooting

### Common Issues

1. **AmberTools not found**: Ensure AmberTools is installed and `tleap` is in PATH
2. **CUDA out of memory**: Reduce batch size or hidden dimensions
3. **Missing ligand**: Check ligand residue name in CSV matches structure
4. **Graph construction fails**: Verify pocket PDB and prmtop files exist

### Tips

- Start with `--max_complexes 10` for testing
- Check failed complexes in `data/failed_*.csv` files
- Use smaller `--hidden_irreps` like "16x0e + 8x1o + 4x2e" for faster training
- Monitor GPU memory with `nvidia-smi`

## Citation

If you use this code, please cite the relevant papers:

- **e3nn**: Geiger & Smidt (2022)
- **Uni-Mol**: Zhou et al. (2023)
- **HARIBOSS**: Garc໚-Recio et al. (2022)

## License

This project is provided as-is for research purposes.
