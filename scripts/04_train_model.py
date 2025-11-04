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


def compute_loss(pred, target, loss_fn='cosine', cosine_weight=0.7, mse_weight=0.3, temperature=0.07):
    """
    Compute loss based on specified loss function.

    Args:
        pred: Predicted embeddings [batch_size, embedding_dim]
        target: Target embeddings [batch_size, embedding_dim]
        loss_fn: Loss function type ('mse', 'cosine', 'cosine_mse', 'infonce')
        cosine_weight: Weight for cosine loss in cosine_mse mode
        mse_weight: Weight for MSE loss in cosine_mse mode
        temperature: Temperature for InfoNCE loss

    Returns:
        loss: Computed loss (scalar)
        metrics: Dictionary with detailed metrics for logging
    """
    metrics = {}

    if loss_fn == 'mse':
        # MSE Loss
        loss = F.mse_loss(pred, target)
        metrics['mse_loss'] = loss.item()

        # Also compute cosine for monitoring
        with torch.no_grad():
            cosine_sim = F.cosine_similarity(pred, target, dim=1).mean()
            metrics['cosine_similarity'] = cosine_sim.item()

    elif loss_fn == 'cosine':
        # Cosine Similarity Loss
        cosine_sim = F.cosine_similarity(pred, target, dim=1)
        loss = (1 - cosine_sim).mean()
        metrics['cosine_loss'] = loss.item()
        metrics['cosine_similarity'] = cosine_sim.mean().item()

        # Also compute MSE for monitoring
        with torch.no_grad():
            mse = F.mse_loss(pred, target)
            metrics['mse_loss'] = mse.item()

    elif loss_fn == 'cosine_mse':
        # Combined: Cosine + MSE
        cosine_sim = F.cosine_similarity(pred, target, dim=1)
        cosine_loss = (1 - cosine_sim).mean()
        mse_loss = F.mse_loss(pred, target)

        loss = cosine_weight * cosine_loss + mse_weight * mse_loss

        metrics['total_loss'] = loss.item()
        metrics['cosine_loss'] = cosine_loss.item()
        metrics['mse_loss'] = mse_loss.item()
        metrics['cosine_similarity'] = cosine_sim.mean().item()
        metrics['cosine_weight'] = cosine_weight
        metrics['mse_weight'] = mse_weight

    elif loss_fn == 'infonce':
        # InfoNCE Contrastive Loss
        batch_size = pred.shape[0]

        # Normalize embeddings
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)

        # Compute similarity matrix [batch_size, batch_size]
        logits = torch.matmul(pred_norm, target_norm.T) / temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=pred.device, dtype=torch.long)

        # Cross-entropy loss (symmetric)
        loss_p2t = F.cross_entropy(logits, labels)
        loss_t2p = F.cross_entropy(logits.T, labels)
        loss = (loss_p2t + loss_t2p) / 2

        metrics['infonce_loss'] = loss.item()
        metrics['temperature'] = temperature

        # Compute accuracy (how often correct pair ranks first)
        with torch.no_grad():
            pred_labels = logits.argmax(dim=1)
            accuracy = (pred_labels == labels).float().mean()
            metrics['infonce_accuracy'] = accuracy.item()

            # Also compute cosine similarity for monitoring
            cosine_sim = F.cosine_similarity(pred, target, dim=1).mean()
            metrics['cosine_similarity'] = cosine_sim.item()

            # And MSE
            mse = F.mse_loss(pred, target)
            metrics['mse_loss'] = mse.item()

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return loss, metrics


def train_epoch(model, loader, optimizer, device, loss_fn='cosine',
                cosine_weight=0.7, mse_weight=0.3, temperature=0.07,
                grad_clip=0.0, monitor_gradients=False):
    """
    Train for one epoch.

    Args:
        model: The model to train
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        loss_fn: Loss function type ('mse', 'cosine', 'cosine_mse', 'infonce')
        cosine_weight: Weight for cosine loss in cosine_mse mode
        mse_weight: Weight for MSE loss in cosine_mse mode
        temperature: Temperature for InfoNCE loss

    Returns:
        Dictionary with training metrics including loss and learnable weights
    """
    model.train()
    total_loss = 0
    num_batches = 0
    accumulated_metrics = {}

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

        # Compute loss using specified loss function
        loss, batch_metrics = compute_loss(
            pocket_embedding, target_embedding,
            loss_fn=loss_fn,
            cosine_weight=cosine_weight,
            mse_weight=mse_weight,
            temperature=temperature
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm before clipping (for monitoring)
        if monitor_gradients and batch_idx % 50 == 0:
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            print(f"  Batch {batch_idx}: Grad norm before clip = {total_grad_norm:.6f}")

        # Optional: Gradient clipping for stability
        # Note: For cosine loss, gradients are naturally smaller than MSE
        if grad_clip > 0:
            # Use user-specified clipping value
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        else:
            # Use automatic defaults based on loss function
            if loss_fn == 'infonce':
                # InfoNCE can have larger gradients due to softmax
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            elif loss_fn == 'cosine':
                # Cosine loss has smaller gradients, use higher threshold
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            else:
                # MSE and combined losses
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        num_batches += 1

        for key, value in batch_metrics.items():
            if key not in accumulated_metrics:
                accumulated_metrics[key] = 0
            accumulated_metrics[key] += value

        # Explicitly delete to free memory
        del pocket_embedding, target_embedding, loss, batch

        # More aggressive cache clearing to prevent fragmentation
        if device.type == 'cuda' and (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            # Also sync to ensure operations complete
            if (batch_idx + 1) % 20 == 0:
                torch.cuda.synchronize()

    # Average accumulated metrics
    metrics = {'loss': total_loss / num_batches}
    for key, value in accumulated_metrics.items():
        metrics[key] = value / num_batches

    # Get learnable weights if available
    if hasattr(model, 'angle_weight'):
        metrics['angle_weight'] = model.angle_weight.item()
    if hasattr(model, 'dihedral_weight'):
        metrics['dihedral_weight'] = model.dihedral_weight.item()
    if hasattr(model, 'nonbonded_weight'):
        metrics['nonbonded_weight'] = model.nonbonded_weight.item()

    return metrics


def evaluate(model, loader, device, loss_fn='cosine',
             cosine_weight=0.7, mse_weight=0.3, temperature=0.07):
    """
    Evaluate model on validation/test set.

    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation data
        device: Device to evaluate on
        loss_fn: Loss function type ('mse', 'cosine', 'cosine_mse', 'infonce')
        cosine_weight: Weight for cosine loss in cosine_mse mode
        mse_weight: Weight for MSE loss in cosine_mse mode
        temperature: Temperature for InfoNCE loss

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    accumulated_metrics = {}

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

            # Compute loss using specified loss function
            loss, batch_metrics = compute_loss(
                pocket_embedding, target_embedding,
                loss_fn=loss_fn,
                cosine_weight=cosine_weight,
                mse_weight=mse_weight,
                temperature=temperature
            )

            total_loss += loss.item()
            num_batches += 1

            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key not in accumulated_metrics:
                    accumulated_metrics[key] = 0
                accumulated_metrics[key] += value

            # Explicitly delete to free memory
            del pocket_embedding, target_embedding, loss, batch

            # More aggressive cache clearing during validation
            if device.type == 'cuda' and (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

    # Average accumulated metrics
    result = {'loss': total_loss / num_batches}
    for key, value in accumulated_metrics.items():
        result[key] = value / num_batches

    return result


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
    parser.add_argument("--use_multi_hop", action="store_true", default=False,
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
    parser.add_argument("--dropout", type=float, default=0.10,
                        help="Dropout rate")

    # Training arguments
    parser.add_argument("--loss_fn", type=str, default="cosine",
                        choices=["mse", "cosine", "cosine_mse", "infonce"],
                        help="Loss function: mse (MSE), cosine (Cosine Similarity), "
                             "cosine_mse (Cosine + MSE), infonce (InfoNCE contrastive)")
    parser.add_argument("--cosine_weight", type=float, default=0.7,
                        help="Weight for cosine loss in cosine_mse mode (default: 0.7)")
    parser.add_argument("--mse_weight", type=float, default=0.3,
                        help="Weight for MSE loss in cosine_mse mode (default: 0.3)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for InfoNCE loss (default: 0.07)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-6,
                        help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw"],
                        help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine"],
                        help="Learning rate scheduler")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of data loader workers")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to use automatic per-loss defaults)")
    parser.add_argument("--monitor_gradients", action="store_true", default=False,
                        help="Print gradient statistics during training for debugging")

    # Splitting arguments
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="models/checkpoints_v2_normalized_1351_4",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=3,
                        help="Save checkpoint every N epochs")

    # Resume training arguments
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a checkpoint")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints_v2_normalized_1351_4/best_model.pt",
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
    cosine_sim_history = []  # Track cosine similarity

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
                    cosine_sim_history = history.get('val_cosine_similarity', [])

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

    # Print loss function configuration
    print("\n" + "="*60)
    print("Loss Function Configuration")
    print("="*60)
    print(f"Loss Function: {args.loss_fn}")
    if args.loss_fn == 'cosine_mse':
        print(f"  Cosine weight: {args.cosine_weight}")
        print(f"  MSE weight: {args.mse_weight}")
    elif args.loss_fn == 'infonce':
        print(f"  Temperature: {args.temperature}")
        print(f"  ⚠️  Note: InfoNCE requires batch_size >= 16 for effective negative sampling")
        if args.batch_size < 16:
            print(f"  ⚠️  Warning: Current batch_size={args.batch_size} may be too small for InfoNCE")

    # Print gradient clipping configuration
    if args.grad_clip > 0:
        print(f"\nGradient Clipping: {args.grad_clip} (manual)")
    else:
        # Show automatic defaults
        if args.loss_fn == 'cosine':
            print(f"\nGradient Clipping: 10.0 (auto for cosine loss)")
        elif args.loss_fn == 'infonce':
            print(f"\nGradient Clipping: 5.0 (auto for InfoNCE)")
        else:
            print(f"\nGradient Clipping: 5.0 (auto for {args.loss_fn})")

    if args.monitor_gradients:
        print("Gradient Monitoring: ENABLED (will print every 50 batches)")

    # Batch size warning for cosine loss
    if args.loss_fn == 'cosine' and args.batch_size < 4:
        print(f"\n⚠️  Warning: batch_size={args.batch_size} is very small for cosine loss")
        print("   Consider using batch_size >= 8 for more stable training")

    print("="*60)

    print("\nStarting training...\n")
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        print("-" * 60)

        # Clear cache at start of epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            loss_fn=args.loss_fn,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            monitor_gradients=args.monitor_gradients
        )
        train_loss = train_metrics['loss']

        # Format training metrics output based on loss function
        train_str = f"Train Loss: {train_loss:.6f}"
        if 'cosine_similarity' in train_metrics:
            train_str += f", Cosine Sim: {train_metrics['cosine_similarity']:.4f}"
        if 'infonce_accuracy' in train_metrics:
            train_str += f", InfoNCE Acc: {train_metrics['infonce_accuracy']*100:.2f}%"
        print(train_str)

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
        val_metrics = evaluate(
            model, val_loader, device,
            loss_fn=args.loss_fn,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            temperature=args.temperature
        )
        val_loss = val_metrics['loss']

        # Format validation metrics output based on loss function
        val_str = f"Val Loss: {val_loss:.6f}"
        if 'cosine_similarity' in val_metrics:
            val_cosine_sim = val_metrics['cosine_similarity']
            val_str += f", Cosine Sim: {val_cosine_sim:.4f}"
        if 'mse_loss' in val_metrics:
            val_str += f", MSE: {val_metrics['mse_loss']:.4f}"
        if 'infonce_accuracy' in val_metrics:
            val_str += f", InfoNCE Acc: {val_metrics['infonce_accuracy']*100:.2f}%"
        print(val_str)

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
        if 'cosine_similarity' in val_metrics:
            cosine_sim_history.append(val_metrics['cosine_similarity'])

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
        'val_cosine_similarity': cosine_sim_history,
        'learnable_weights': weight_history,
        'config': vars(args)
    }
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete!")
    print(f"Best validation cosine loss: {best_val_loss:.6f}")
    if cosine_sim_history:
        print(f"Best validation cosine similarity: {max(cosine_sim_history):.4f}")
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
