#!/usr/bin/env python3
"""
Training Script for RNA Pocket Encoder v2.0

This script trains the E(3) equivariant GNN to predict ligand embeddings
from RNA binding pocket structures.

Version 2.0 changes:
- Uses RNAPocketEncoderV2 with embedding-based input
- Supports multi-hop message passing (1/2/3-hop)
- Supports non-bonded interactions
- Learnable combination weights
- 4-dimensional input features [atom_type_idx, charge, residue_idx, atomic_num]
"""
import os
import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import h5py
import warnings

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.amber_vocabulary import get_global_encoder


class LigandEmbeddingDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-computed graphs and ligand embeddings.

    v2.0 changes:
    - Validates that graphs use 4-dimensional input features
    - Checks for multi-hop indices (triple_index, quadra_index)
    - Checks for non-bonded edges
    """

    def __init__(self, complex_ids, graph_dir, ligand_embeddings_path, validate_format=True):
        """
        Args:
            complex_ids: List of complex IDs to include (may have _model{N} suffix)
            graph_dir: Directory containing graph .pt files
            ligand_embeddings_path: Path to HDF5 file with ligand embeddings
            validate_format: Whether to validate v2.0 data format
        """
        self.complex_ids = complex_ids
        self.graph_dir = Path(graph_dir)
        self.validate_format = validate_format

        # Load all ligand embeddings (keys don't have model numbers)
        self.ligand_embeddings = {}
        with h5py.File(ligand_embeddings_path, 'r') as f:
            for key in f.keys():
                self.ligand_embeddings[key] = torch.tensor(
                    f[key][:],
                    dtype=torch.float
                )

        # Create mapping from complex_id (with model) to embedding key (without model)
        # Format: {pdb_id}_{ligand}_model{N} -> {pdb_id}_{ligand}
        self.id_to_embedding_key = {}
        for complex_id in complex_ids:
            if '_model' in complex_id:
                # Extract base ID without model number
                base_id = '_'.join(complex_id.split('_model')[0].split('_'))
            else:
                base_id = complex_id

            if base_id in self.ligand_embeddings:
                self.id_to_embedding_key[complex_id] = base_id

        # Filter to only include complexes with both graph and embedding
        self.valid_ids = []
        format_warnings = []

        for complex_id in complex_ids:
            graph_path = self.graph_dir / f"{complex_id}.pt"
            if graph_path.exists() and complex_id in self.id_to_embedding_key:
                # Validate data format if requested
                if validate_format:
                    try:
                        data = torch.load(graph_path)

                        # Check input feature dimension (should be 4 for v2.0)
                        if data.x.shape[1] != 4:
                            format_warnings.append(
                                f"{complex_id}: Expected 4D features, got {data.x.shape[1]}D"
                            )
                            continue

                        # Check for required attributes
                        if not hasattr(data, 'edge_index'):
                            format_warnings.append(f"{complex_id}: Missing edge_index")
                            continue

                        # Optional: warn if multi-hop indices are missing
                        if not hasattr(data, 'triple_index'):
                            if len(format_warnings) < 3:  # Limit warnings
                                format_warnings.append(
                                    f"{complex_id}: Missing triple_index (2-hop angles)"
                                )

                    except Exception as e:
                        format_warnings.append(f"{complex_id}: Error loading - {str(e)}")
                        continue

                self.valid_ids.append(complex_id)

        print(f"Dataset initialized with {len(self.valid_ids)} valid complexes")

        if format_warnings:
            print(f"\n⚠️  Format validation warnings ({len(format_warnings)} total):")
            for warning in format_warnings[:5]:  # Show first 5
                print(f"  - {warning}")
            if len(format_warnings) > 5:
                print(f"  ... and {len(format_warnings) - 5} more")
            print("\n💡 Tip: Regenerate graphs with scripts/03_build_dataset.py for v2.0 format\n")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        complex_id = self.valid_ids[idx]

        # Load graph (use weights_only=False for backward compatibility)
        graph_path = self.graph_dir / f"{complex_id}.pt"
        data = torch.load(graph_path, weights_only=False)

        # Get ligand embedding using the mapping
        # complex_id may be "1aju_ARG_model0", but embedding key is "1aju_ARG"
        embedding_key = self.id_to_embedding_key[complex_id]
        ligand_embedding = self.ligand_embeddings[embedding_key]

        # Store target in data object (clone to avoid reference issues)
        data.y = torch.tensor(ligand_embedding, dtype=torch.float32)

        return data


def train_epoch(model, loader, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: The model to train
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Dictionary with training metrics including loss and learnable weights
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        batch = batch.to(device)

        # Forward pass
        pocket_embedding = model(batch)
        target_embedding = batch.y

        # Reshape target if needed (PyG DataLoader flattens batch.y)
        # Expected: [batch_size, embedding_dim]
        if target_embedding.dim() == 1:
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else pocket_embedding.size(0)
            target_embedding = target_embedding.view(batch_size, -1)

        # MSE loss
        loss = F.mse_loss(pocket_embedding, target_embedding)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optional: Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Record loss before clearing variables
        total_loss += loss.item()
        num_batches += 1

        # Explicitly delete to free memory
        del pocket_embedding, target_embedding, loss, batch

        # More aggressive cache clearing to prevent fragmentation
        if device.type == 'cuda' and (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            # Also sync to ensure operations complete
            if (batch_idx + 1) % 20 == 0:
                torch.cuda.synchronize()

    # Get learnable weights if available
    metrics = {'loss': total_loss / num_batches}

    if hasattr(model, 'angle_weight'):
        metrics['angle_weight'] = model.angle_weight.item()
    if hasattr(model, 'dihedral_weight'):
        metrics['dihedral_weight'] = model.dihedral_weight.item()
    if hasattr(model, 'nonbonded_weight'):
        metrics['nonbonded_weight'] = model.nonbonded_weight.item()

    return metrics


def evaluate(model, loader, device):
    """
    Evaluate model on validation/test set.

    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation data
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    total_l1_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            batch = batch.to(device)

            # Forward pass
            pocket_embedding = model(batch)
            target_embedding = batch.y

            # Reshape target if needed (PyG DataLoader flattens batch.y)
            # Expected: [batch_size, embedding_dim]
            if target_embedding.dim() == 1:
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else pocket_embedding.size(0)
                target_embedding = target_embedding.view(batch_size, -1)

            # MSE loss
            mse_loss = F.mse_loss(pocket_embedding, target_embedding)
            l1_loss = F.l1_loss(pocket_embedding, target_embedding)

            total_loss += mse_loss.item()
            total_l1_loss += l1_loss.item()
            num_batches += 1

            # Explicitly delete to free memory
            del pocket_embedding, target_embedding, mse_loss, l1_loss, batch

            # More aggressive cache clearing during validation
            if device.type == 'cuda' and (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

    return {
        'mse_loss': total_loss / num_batches,
        'l1_loss': total_l1_loss / num_batches
    }


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train RNA Pocket Encoder")

    # Data arguments
    parser.add_argument("--hariboss_csv", type=str, default="hariboss/Complexes.csv",
                        help="Path to HARIBOSS CSV")
    parser.add_argument("--graph_dir", type=str, default="data/processed/graphs",
                        help="Directory containing graph files")
    parser.add_argument("--embeddings_path", type=str, default="data/processed/ligand_embeddings.h5",
                        help="Path to ligand embeddings HDF5 file")
    parser.add_argument("--splits_file", type=str, default="data/splits/splits.json",
                        help="Path to save/load dataset splits")

    # Model arguments (v2.0)
    parser.add_argument("--atom_embed_dim", type=int, default=32,
                        help="Embedding dimension for atom types")
    parser.add_argument("--residue_embed_dim", type=int, default=16,
                        help="Embedding dimension for residues")
    parser.add_argument("--hidden_irreps", type=str, default="32x0e + 16x1o + 8x2e",
                        help="Hidden layer irreps")
    parser.add_argument("--output_dim", type=int, default=1536,
                        help="Output embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of message passing layers")
    parser.add_argument("--num_radial_basis", type=int, default=8,
                        help="Number of radial basis functions")

    # v2.0 specific arguments
    parser.add_argument("--use_multi_hop", action="store_true", default=True,
                        help="Enable multi-hop message passing (2-hop angles, 3-hop dihedrals)")
    parser.add_argument("--use_nonbonded", action="store_true", default=True,
                        help="Enable non-bonded interactions")
    parser.add_argument("--use_weight_constraints", action="store_true", default=False,
                        help="Use fixed version with weight constraints to prevent weights from going to zero")
    parser.add_argument("--use_gate", action="store_true", default=True,
                        help="Use gate activation (requires improved layers)")
    parser.add_argument("--use_layer_norm", action="store_true", default=False,
                        help="Use layer normalization (requires improved layers)")
    parser.add_argument("--pooling_type", type=str, default="attention",
                        choices=["attention", "mean", "sum", "max"],
                        help="Graph pooling type")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw"],
                        help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine"],
                        help="Learning rate scheduler")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable)")

    # Splitting arguments
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="models/checkpoints_v2_normalized",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=3,
                        help="Save checkpoint every N epochs")

    # Resume training arguments
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a checkpoint")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints_v2_normalized/best_model.pt",
                        help="Path to checkpoint file to resume from")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Load or create dataset splits
    splits_path = Path(args.splits_file)
    if splits_path.exists():
        print(f"Loading splits from {splits_path}")
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        train_ids = splits['train']
        val_ids = splits['val']
        test_ids = splits.get('test', [])
    else:
        print("Creating new dataset splits...")
        # Read all available complex IDs
        hariboss_df = pd.read_csv(args.hariboss_csv)

        # Find PDB ID and ligand columns
        pdb_id_column = None
        for col in ['id','pdb_id', 'PDB_ID', 'pdbid', 'PDBID', 'PDB']:
            if col in hariboss_df.columns:
                pdb_id_column = col
                break

        ligand_column = None
        for col in ['sm_ligand_ids','ligand', 'Ligand', 'ligand_resname', 'LIGAND', 'ligand_name']:
            if col in hariboss_df.columns:
                ligand_column = col
                break

        if pdb_id_column is None or ligand_column is None:
            print("Error: Could not find required columns in CSV")
            sys.exit(1)

        # Create complex IDs - need to handle models from 01_process_data.py
        # Format: {pdb_id}_{ligand}_model{N}.pt
        import glob
        import ast

        complex_ids = []
        for _, row in hariboss_df.iterrows():
            pdb_id = str(row[pdb_id_column]).lower()

            # Parse ligand name
            ligand_str = str(row[ligand_column])
            if ligand_column == 'sm_ligand_ids':
                # Parse format like "['ARG_.:B/1:N']" or "ARG_.:B/1:N"
                try:
                    ligands = ast.literal_eval(ligand_str)
                    if not isinstance(ligands, list):
                        ligands = [ligand_str]
                except:
                    ligands = [ligand_str]

                if ligands and len(ligands) > 0:
                    ligand_resname = ligands[0].split('_')[0].split(':')[0]
                else:
                    ligand_resname = ligand_str
            else:
                ligand_resname = ligand_str

            complex_base = f"{pdb_id}_{ligand_resname}"
            complex_ids.append(complex_base)

        # Filter to only include complexes with both graph and embedding
        # Need to find all model variants: {pdb_id}_{ligand}_model*.pt or {pdb_id}_{ligand}.pt
        graph_dir = Path(args.graph_dir)
        valid_ids = []

        with h5py.File(args.embeddings_path, 'r') as f:
            for complex_base in complex_ids:
                # Try to find model files
                pattern = str(graph_dir / f"{complex_base}_model*.pt")
                model_files = sorted(glob.glob(pattern))

                if model_files:
                    # Found model files, add all of them
                    # Note: Embeddings use complex_base (without model number)
                    # but graphs use model_id (with model number)
                    if complex_base in f:
                        for model_file in model_files:
                            model_id = Path(model_file).stem  # e.g., "1aju_ARG_model0"
                            valid_ids.append(model_id)
                else:
                    # Fallback: try without model number
                    graph_path = graph_dir / f"{complex_base}.pt"
                    if graph_path.exists() and complex_base in f:
                        valid_ids.append(complex_base)

        print(f"Found {len(valid_ids)} valid complexes")

        # Shuffle and split
        np.random.shuffle(valid_ids)
        n_train = int(len(valid_ids) * args.train_ratio)
        n_val = int(len(valid_ids) * args.val_ratio)

        train_ids = valid_ids[:n_train]
        val_ids = valid_ids[n_train:n_train+n_val]
        test_ids = valid_ids[n_train+n_val:]

        # Save splits
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)

    print(f"Dataset splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Create datasets
    train_dataset = LigandEmbeddingDataset(train_ids, args.graph_dir, args.embeddings_path)
    val_dataset = LigandEmbeddingDataset(val_ids, args.graph_dir, args.embeddings_path)

    # Create data loaders
    # Note: Reduced workers and disabled pin_memory to save GPU memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(args.num_workers, 2),  # Limit to 2 workers max
        pin_memory=False,  # Disable to save GPU memory
        persistent_workers=False  # Don't keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 2),  # Limit to 2 workers max
        pin_memory=False,  # Disable to save GPU memory
        persistent_workers=False
    )

    # Initialize model (v2.0)
    print("\nInitializing v2.0 model...")

    # Import appropriate model class based on weight constraints flag
    if args.use_weight_constraints:
        from models.e3_gnn_encoder_v2_fixed import RNAPocketEncoderV2Fixed as ModelClass
        print("Using RNAPocketEncoderV2Fixed (with weight constraints)")
    else:
        from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2 as ModelClass
        print("Using RNAPocketEncoderV2 (standard)")

    # Get vocabulary sizes
    encoder = get_global_encoder()
    print(f"Vocabulary sizes: {encoder.num_atom_types} atom types, {encoder.num_residues} residues")

    model = ModelClass(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        atom_embed_dim=args.atom_embed_dim,
        residue_embed_dim=args.residue_embed_dim,
        hidden_irreps=args.hidden_irreps,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_radial_basis=args.num_radial_basis,
        use_multi_hop=args.use_multi_hop,
        use_nonbonded=args.use_nonbonded,
        use_gate=args.use_gate,
        use_layer_norm=args.use_layer_norm,
        pooling_type=args.pooling_type,
        dropout=args.dropout
    )

    model = model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Print model configuration
    print("\nModel configuration:")
    print(f"  Multi-hop: {args.use_multi_hop}")
    print(f"  Non-bonded: {args.use_nonbonded}")
    print(f"  Pooling: {args.pooling_type}")
    if args.use_multi_hop:
        print(f"  Initial angle weight: {model.angle_weight.item():.3f}")
        print(f"  Initial dihedral weight: {model.dihedral_weight.item():.3f}")
    if args.use_nonbonded:
        print(f"  Initial nonbonded weight: {model.nonbonded_weight.item():.3f}")

    # Initialize optimizer
    if args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Initialize scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=args.lr * 0.01
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    # Load checkpoint if resuming
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []
    val_history = []

    if args.resume:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"\nLoading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model and optimizer states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Resume from next epoch
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))

            # Load training history if available
            history_path = output_dir / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    train_history = history.get('train_loss', [])
                    val_history = history.get('val_loss', [])

            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best validation loss so far: {best_val_loss:.6f}")
        else:
            print(f"\nWarning: Checkpoint file {checkpoint_path} not found!")
            print("Starting training from scratch...\n")
    else:
        print("\nStarting training from scratch...\n")

    # Additional tracking for v2.0
    weight_history = {
        'angle_weight': [],
        'dihedral_weight': [],
        'nonbonded_weight': []
    }

    print("\nStarting training...\n")
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        print("-" * 60)

        # Clear cache at start of epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        train_loss = train_metrics['loss']
        print(f"Train Loss: {train_loss:.6f}")

        # Print learnable weights if available
        if 'angle_weight' in train_metrics:
            print(f"  Angle weight: {train_metrics['angle_weight']:.4f}")
            weight_history['angle_weight'].append(train_metrics['angle_weight'])
        if 'dihedral_weight' in train_metrics:
            print(f"  Dihedral weight: {train_metrics['dihedral_weight']:.4f}")
            weight_history['dihedral_weight'].append(train_metrics['dihedral_weight'])
        if 'nonbonded_weight' in train_metrics:
            print(f"  Nonbonded weight: {train_metrics['nonbonded_weight']:.4f}")
            weight_history['nonbonded_weight'].append(train_metrics['nonbonded_weight'])

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics['mse_loss']
        print(f"Val Loss: {val_loss:.6f}, Val L1: {val_metrics['l1_loss']:.6f}")

        # Update learning rate
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")

        # Save history
        train_history.append(train_loss)
        val_history.append(val_loss)

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            # Clear cache after saving to free up temporary memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_model_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"New best model! Saved to {best_model_path}")
            # Clear cache after saving
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        # Aggressive memory cleanup at end of epoch to prevent fragmentation
        if device.type == 'cuda':
            # Force synchronization to complete all pending operations
            torch.cuda.synchronize()
            # Clear cache
            torch.cuda.empty_cache()
            # Reset peak memory stats for monitoring
            torch.cuda.reset_peak_memory_stats()

        print()

    # Save training history
    history = {
        'train_loss': train_history,
        'val_loss': val_history,
        'learnable_weights': weight_history,
        'config': vars(args)
    }
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to {output_dir / 'best_model.pt'}")

    # Print final learnable weights
    if args.use_multi_hop or args.use_nonbonded:
        print("\nFinal learnable weights:")
        if hasattr(model, 'angle_weight'):
            print(f"  Angle weight: {model.angle_weight.item():.4f} (initial: 0.333)")
        if hasattr(model, 'dihedral_weight'):
            print(f"  Dihedral weight: {model.dihedral_weight.item():.4f} (initial: 0.333)")
        if hasattr(model, 'nonbonded_weight'):
            print(f"  Nonbonded weight: {model.nonbonded_weight.item():.4f} (initial: 0.333)")

        # If using fixed version, show log-space parameters
        if args.use_weight_constraints and hasattr(model, 'get_weight_summary'):
            print("\nWeight constraints (log-space parameters):")
            summary = model.get_weight_summary()
            for key, value in summary.items():
                if 'log' in key:
                    print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
