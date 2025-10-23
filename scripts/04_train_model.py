#!/usr/bin/env python3
"""
Training Script for RNA Pocket Encoder

This script trains the E(3) equivariant GNN to predict ligand embeddings
from RNA binding pocket structures.
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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import h5py

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.e3_gnn_encoder import RNAPocketEncoder


class LigandEmbeddingDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-computed graphs and ligand embeddings.
    """

    def __init__(self, complex_ids, graph_dir, ligand_embeddings_path):
        """
        Args:
            complex_ids: List of complex IDs to include (may have _model{N} suffix)
            graph_dir: Directory containing graph .pt files
            ligand_embeddings_path: Path to HDF5 file with ligand embeddings
        """
        self.complex_ids = complex_ids
        self.graph_dir = Path(graph_dir)

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
        for complex_id in complex_ids:
            graph_path = self.graph_dir / f"{complex_id}.pt"
            if graph_path.exists() and complex_id in self.id_to_embedding_key:
                self.valid_ids.append(complex_id)

        print(f"Dataset initialized with {len(self.valid_ids)} valid complexes")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        complex_id = self.valid_ids[idx]

        # Load graph
        graph_path = self.graph_dir / f"{complex_id}.pt"
        data = torch.load(graph_path)

        # Get ligand embedding using the mapping
        # complex_id may be "1aju_ARG_model0", but embedding key is "1aju_ARG"
        embedding_key = self.id_to_embedding_key[complex_id]
        ligand_embedding = self.ligand_embeddings[embedding_key]

        # Store target in data object
        data.y = ligand_embedding

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
        Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Training"):
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
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Clear cache to prevent OOM
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return total_loss / num_batches


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
        for batch in tqdm(loader, desc="Evaluating"):
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

            # Clear cache to prevent OOM
            if device.type == 'cuda':
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

    # Model arguments
    parser.add_argument("--input_dim", type=int, default=11,
                        help="Input feature dimension")
    parser.add_argument("--hidden_irreps", type=str, default="32x0e + 16x1o + 8x2e",
                        help="Hidden layer irreps")
    parser.add_argument("--output_dim", type=int, default=1536,
                        help="Output embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of message passing layers")
    parser.add_argument("--num_radial_basis", type=int, default=8,
                        help="Number of radial basis functions")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")

    # Splitting arguments
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="models/checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Resume training arguments
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a checkpoint")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pt",
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize model
    print("\nInitializing model...")
    model = RNAPocketEncoder(
        input_dim=args.input_dim,
        hidden_irreps=args.hidden_irreps,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_radial_basis=args.num_radial_basis
    )

    model = model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Initialize optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

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

    print("\nStarting training...\n")
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        print("-" * 60)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics['mse_loss']
        print(f"Val Loss: {val_loss:.6f}, Val L1: {val_metrics['l1_loss']:.6f}")

        # Update learning rate
        scheduler.step(val_loss)
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
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        print()

    # Save training history
    history = {
        'train_loss': train_history,
        'val_loss': val_history
    }
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
