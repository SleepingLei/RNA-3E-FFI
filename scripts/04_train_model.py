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

# For distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# For mixed precision training
try:
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# AMBER vocabulary removed - using pure physical features

# Import V3 improvements
try:
    from models.improved_components import PhysicsConstraintLoss
    from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3
    _has_v3_model = True
    _has_physics_loss = True
except ImportError:
    _has_v3_model = False
    _has_physics_loss = False
    warnings.warn("V3 improvements not available. Using V2 model only.")


def setup_ddp(rank, world_size):
    """
    Initialize the distributed environment.

    Args:
        rank: Unique identifier for each process (0 to world_size-1)
        world_size: Total number of processes (GPUs)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set the GPU for this process
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


class LigandEmbeddingDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-computed graphs and ligand embeddings.

    v2.0 (Pure Physical Features) changes:
    - Validates that graphs use 3-dimensional input features [charge, atomic_num, mass]
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

                        # Check input feature dimension (should be 3 for pure physical features)
                        if data.x.shape[1] != 3:
                            format_warnings.append(
                                f"{complex_id}: Expected 3D features [charge, atomic_num, mass], got {data.x.shape[1]}D"
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
                grad_clip=0.0, monitor_gradients=False, accumulation_steps=1,
                use_amp=False, scaler=None, physics_loss_fn=None, physics_weight=0.1,
                monitor_weights=True, monitor_features=False):
    """
    Train for one epoch with gradient accumulation and mixed precision support.

    Args:
        model: The model to train
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        loss_fn: Loss function type ('mse', 'cosine', 'cosine_mse', 'infonce')
        cosine_weight: Weight for cosine loss in cosine_mse mode
        mse_weight: Weight for MSE loss in cosine_mse mode
        temperature: Temperature for InfoNCE loss
        accumulation_steps: Number of steps to accumulate gradients before updating
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for mixed precision training
        physics_loss_fn: PhysicsConstraintLoss function (optional, V3 improvement)
        physics_weight: Weight for physics constraint loss
        monitor_weights: Whether to monitor learnable weights
        monitor_features: Whether to monitor feature statistics (expensive)

    Returns:
        Dictionary with training metrics including loss and learnable weights
    """
    model.train()
    total_loss = 0
    num_batches = 0
    accumulated_metrics = {}

    # 用于权重监控
    weight_history = []

    # Zero gradients at the start
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        batch = batch.to(device)

        # Forward pass with mixed precision
        with autocast(device_type='cuda', enabled=use_amp):
            pocket_embedding = model(batch)
            target_embedding = batch.y

            # Reshape target if needed (PyG DataLoader flattens batch.y)
            # Expected: [batch_size, embedding_dim]
            if target_embedding.dim() == 1:
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else pocket_embedding.size(0)
                target_embedding = target_embedding.view(batch_size, -1)

            # Compute main task loss using specified loss function
            loss, batch_metrics = compute_loss(
                pocket_embedding, target_embedding,
                loss_fn=loss_fn,
                cosine_weight=cosine_weight,
                mse_weight=mse_weight,
                temperature=temperature
            )

            # Add physics constraint loss if enabled (V3 improvement)
            if physics_loss_fn is not None:
                physics_loss, physics_dict = physics_loss_fn(batch)
                # Add physics loss with weight
                loss = loss + physics_weight * physics_loss
                # Track physics metrics
                batch_metrics['physics_loss'] = physics_loss.item()
                batch_metrics['physics_weight'] = physics_weight
                for key, val in physics_dict.items():
                    batch_metrics[key] = val

            # Scale loss by accumulation steps (to get the correct average)
            loss = loss / accumulation_steps

        # Backward pass (accumulate gradients) with mixed precision
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Only update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            # Compute gradient norm before clipping (for monitoring)
            if monitor_gradients:
                # 自适应监控频率: 前100个batch密集监控(每10次)，之后降低频率(每50次)
                monitor_interval = 10 if batch_idx < 100 else 50
                if batch_idx % monitor_interval == 0:
                    total_grad_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.norm().item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5

                    # 获取权重梯度统计
                    weight_grad_info = ""
                    if hasattr(model, 'get_weight_stats'):
                        try:
                            model_to_check = model.module if hasattr(model, 'module') else model
                            weight_stats = model_to_check.get_weight_stats()
                            if 'angle_weight_grad' in weight_stats:
                                weight_grad_info += f", angle_w_grad={weight_stats['angle_weight_grad']:.6f}"
                            if 'dihedral_weight_grad' in weight_stats:
                                weight_grad_info += f", dihedral_w_grad={weight_stats['dihedral_weight_grad']:.6f}"
                            if 'nonbonded_weight_grad' in weight_stats:
                                weight_grad_info += f", nonbonded_w_grad={weight_stats['nonbonded_weight_grad']:.6f}"
                        except:
                            pass

                    print(f"  Batch {batch_idx}: Grad norm = {total_grad_norm:.6f}{weight_grad_info}")

            # Optional: Gradient clipping for stability
            # Note: For cosine loss, gradients are naturally smaller than MSE
            if use_amp:
                # Unscale gradients for clipping with AMP
                scaler.unscale_(optimizer)

            if grad_clip > 0:
                # Use user-specified clipping value
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            else:
                # V3 model uses stricter gradient clipping (204-dim invariants + multi-hop MP)
                # Lower thresholds than V2 to handle increased model complexity
                # 注意：权重约束 (sigmoid) 已经添加，可以适当放宽阈值
                if loss_fn == 'infonce':
                    # InfoNCE: 2.0 (was 5.0 for V2)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                elif loss_fn == 'cosine':
                    # Cosine: 1.5 (was 10.0 for V2)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                elif loss_fn == 'mse':
                    # MSE: 1.0 (更严格，因为MSE梯度通常较大)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    # Combined: 1.5
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)

            # Update weights with AMP support
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Accumulate metrics (use unscaled loss for reporting)
        total_loss += loss.item() * accumulation_steps
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

    # Get learnable weights if available (支持DDP包装的模型)
    model_for_weights = model.module if hasattr(model, 'module') else model

    # 使用新的 get_weight_stats 方法获取完整的权重统计
    if hasattr(model_for_weights, 'get_weight_stats'):
        weight_stats = model_for_weights.get_weight_stats()
        metrics.update(weight_stats)
    else:
        # 后备方案：直接访问属性
        if hasattr(model_for_weights, 'angle_weight'):
            metrics['angle_weight'] = model_for_weights.angle_weight.item()
        if hasattr(model_for_weights, 'dihedral_weight'):
            metrics['dihedral_weight'] = model_for_weights.dihedral_weight.item()
        if hasattr(model_for_weights, 'nonbonded_weight'):
            metrics['nonbonded_weight'] = model_for_weights.nonbonded_weight.item()

    return metrics


def evaluate(model, loader, device, loss_fn='cosine',
             cosine_weight=0.7, mse_weight=0.3, temperature=0.07, use_amp=False):
    """
    Evaluate model on validation/test set with mixed precision support.

    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation data
        device: Device to evaluate on
        loss_fn: Loss function type ('mse', 'cosine', 'cosine_mse', 'infonce')
        cosine_weight: Weight for cosine loss in cosine_mse mode
        mse_weight: Weight for MSE loss in cosine_mse mode
        temperature: Temperature for InfoNCE loss
        use_amp: Whether to use automatic mixed precision

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

            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=use_amp):
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


def train_worker(rank, world_size, args):
    """
    Training worker function for each GPU process.

    Args:
        rank: Process rank (0 to world_size-1). When not using DDP, rank=0.
        world_size: Total number of processes. When not using DDP, world_size=1.
        args: Parsed command line arguments
    """
    # Setup DDP if using distributed training
    if args.use_ddp and world_size > 1:
        setup_ddp(rank, world_size)
        is_main_process = (rank == 0)
    else:
        is_main_process = True

    # Set random seed (different for each process to ensure different data shuffling)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Setup device
    if args.use_ddp and world_size > 1:
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_main_process:
        print(f"Using device: {device}")
        if args.use_ddp and world_size > 1:
            print(f"Distributed training with {world_size} GPUs")

    # Create output directory (only main process)
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = vars(args)
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    # Synchronize all processes
    if args.use_ddp and world_size > 1:
        dist.barrier()

    # Load or create dataset splits (only main process creates splits)
    splits_path = Path(args.splits_file)
    if splits_path.exists():
        if is_main_process:
            print(f"Loading splits from {splits_path}")
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        train_ids = splits['train']
        val_ids = splits['val']
        test_ids = splits.get('test', [])
    else:
        if is_main_process:
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
        else:
            # Non-main processes wait for main to create splits
            if args.use_ddp and world_size > 1:
                dist.barrier()
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            train_ids = splits['train']
            val_ids = splits['val']
            test_ids = splits.get('test', [])

    if is_main_process:
        print(f"Dataset splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Create datasets
    train_dataset = LigandEmbeddingDataset(train_ids, args.graph_dir, args.embeddings_path)
    val_dataset = LigandEmbeddingDataset(val_ids, args.graph_dir, args.embeddings_path)

    # Create data loaders with DistributedSampler if using DDP
    if args.use_ddp and world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=min(args.num_workers, 2),
            pin_memory=False,
            persistent_workers=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=min(args.num_workers, 2),
            pin_memory=False,
            persistent_workers=False
        )
    else:
        # Standard non-distributed data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(args.num_workers, 2),
            pin_memory=False,
            persistent_workers=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 2),
            pin_memory=False,
            persistent_workers=False
        )

    # Initialize model (v2.0 or v3.0)
    if is_main_process:
        if args.use_v3_model and _has_v3_model:
            print("\nInitializing v3.0 model (with V3 improvements)...")
        else:
            print("\nInitializing v2.0 model (Pure Physical Features)...")

    # Select model class based on version and constraints
    if args.use_v3_model and _has_v3_model:
        # Use V3 model with improvements
        ModelClass = RNAPocketEncoderV3
        if is_main_process:
            print("Using RNAPocketEncoderV3 (geometric MP + enhanced invariants + multi-head attention)")
            print(f"  Geometric MP: {args.use_geometric_mp}")
            print(f"  Enhanced invariants: {args.use_enhanced_invariants} (204-dim vs 56-dim)")
            print(f"  Improved layers: {args.use_improved_layers} (Bessel+Cutoff+ImprovedMP)")
            print(f"  Norm type: {args.norm_type}")
            print(f"  Attention heads: {args.num_attention_heads}")
            print(f"  Initial angle weight: {args.initial_angle_weight:.3f}")
            print(f"  Initial dihedral weight: {args.initial_dihedral_weight:.3f}")
            print(f"  Initial nonbonded weight: {args.initial_nonbonded_weight:.3f}")

        # V3-specific pooling type override
        v3_pooling_type = 'multihead_attention' if args.num_attention_heads > 1 else args.pooling_type

        model = ModelClass(
            input_dim=args.input_dim,
            feature_hidden_dim=args.feature_hidden_dim,
            hidden_irreps=args.hidden_irreps,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            num_radial_basis=args.num_radial_basis,
            use_multi_hop=args.use_multi_hop,
            use_nonbonded=args.use_nonbonded,
            use_gate=args.use_gate,
            use_layer_norm=args.use_layer_norm,
            pooling_type=v3_pooling_type,
            dropout=args.dropout,
            # V3-specific parameters
            use_geometric_mp=args.use_geometric_mp,
            use_enhanced_invariants=args.use_enhanced_invariants,
            num_attention_heads=args.num_attention_heads,
            # Learnable weight initial values
            initial_angle_weight=args.initial_angle_weight,
            initial_dihedral_weight=args.initial_dihedral_weight,
            initial_nonbonded_weight=args.initial_nonbonded_weight,
            # Improved layers parameters (V3 only)
            use_improved_layers=args.use_improved_layers,
            norm_type=args.norm_type
        )
    else:
        # Use V2 model (backward compatible)
        if args.use_weight_constraints:
            from models.e3_gnn_encoder_v2_fixed import RNAPocketEncoderV2Fixed as ModelClass
            if is_main_process:
                print("Using RNAPocketEncoderV2Fixed (with weight constraints)")
        else:
            from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2 as ModelClass
            if is_main_process:
                print("Using RNAPocketEncoderV2 (standard)")

        if is_main_process:
            print(f"Input features: 3D physical properties [charge, atomic_num, mass]")

        model = ModelClass(
            input_dim=args.input_dim,
            feature_hidden_dim=args.feature_hidden_dim,
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

    # Wrap model with DDP if using distributed training
    if args.use_ddp and world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        # Access the underlying model for parameter count
        model_for_params = model.module
    else:
        model_for_params = model

    if is_main_process:
        print(f"Model has {sum(p.numel() for p in model_for_params.parameters()):,} parameters")

        # Print model configuration
        print("\nModel configuration:")
        print(f"  Multi-hop: {args.use_multi_hop}")
        print(f"  Non-bonded: {args.use_nonbonded}")
        print(f"  Pooling: {args.pooling_type}")
        if args.use_multi_hop:
            print(f"  Initial angle weight: {model_for_params.angle_weight.item():.3f}")
            print(f"  Initial dihedral weight: {model_for_params.dihedral_weight.item():.3f}")
        if args.use_nonbonded:
            print(f"  Initial nonbonded weight: {model_for_params.nonbonded_weight.item():.3f}")

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
            verbose=is_main_process
        )

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if args.use_amp else None

    # Load checkpoint if resuming
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []
    val_history = []
    cosine_sim_history = []

    if args.resume:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            if is_main_process:
                print(f"\nLoading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model and optimizer states
            if args.use_ddp and world_size > 1:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
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

            if is_main_process:
                print(f"Resumed from epoch {checkpoint['epoch']}")
                print(f"Best validation loss so far: {best_val_loss:.6f}")
        else:
            if is_main_process:
                print(f"\nWarning: Checkpoint file {checkpoint_path} not found!")
                print("Starting training from scratch...\n")
    else:
        if is_main_process:
            print("\nStarting training from scratch...\n")

    # Additional tracking for v2.0
    weight_history = {
        'angle_weight': [],
        'dihedral_weight': [],
        'nonbonded_weight': []
    }

    # Print training configuration (only main process)
    if is_main_process:
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)

        # Batch size and gradient accumulation
        effective_batch_size = args.batch_size * args.accumulation_steps
        if args.use_ddp and world_size > 1:
            effective_batch_size *= world_size
            print(f"Batch Size per GPU: {args.batch_size}")
            print(f"Number of GPUs: {world_size}")
        else:
            print(f"Batch Size: {args.batch_size}")

        if args.accumulation_steps > 1:
            print(f"Gradient Accumulation Steps: {args.accumulation_steps}")
        print(f"Effective Batch Size: {effective_batch_size}")
        if args.accumulation_steps > 1:
            print(f"  💡 This simulates batch_size={effective_batch_size} without extra GPU memory")

        # Mixed precision training
        print(f"\nMixed Precision Training (AMP): {'Enabled' if args.use_amp else 'Disabled'}")
        if args.use_amp:
            print(f"  ⚡ Using float16 for forward/backward pass (~50% memory reduction)")
            print(f"  💡 This can significantly reduce GPU memory usage")

        # Loss function configuration
        print(f"\nLoss Function: {args.loss_fn}")
        if args.loss_fn == 'cosine_mse':
            print(f"  Cosine weight: {args.cosine_weight}")
            print(f"  MSE weight: {args.mse_weight}")
        elif args.loss_fn == 'infonce':
            print(f"  Temperature: {args.temperature}")
            print(f"  ⚠️  Note: InfoNCE requires batch_size >= 16 for effective negative sampling")
            if effective_batch_size < 16:
                print(f"  ⚠️  Warning: Current effective_batch_size={effective_batch_size} may be too small for InfoNCE")

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
        if args.loss_fn == 'cosine' and effective_batch_size < 4:
            print(f"\n⚠️  Warning: effective_batch_size={effective_batch_size} is very small for cosine loss")
            print("   Consider using effective_batch_size >= 8 for more stable training")
            if args.accumulation_steps == 1:
                print(f"   You can increase --accumulation_steps to achieve this")

        print("="*60)
        print("\nStarting training...\n")

    # Initialize physics constraint loss if enabled (V3 improvement)
    physics_loss_fn = None
    if args.use_physics_loss and _has_physics_loss:
        physics_loss_fn = PhysicsConstraintLoss(
            use_bond=args.physics_use_bond,
            use_angle=args.physics_use_angle,
            use_dihedral=args.physics_use_dihedral,
            use_nonbonded=args.physics_use_nonbonded
        ).to(device)
        if is_main_process:
            print("\nPhysics Constraint Loss (V3 Improvement): ENABLED")
            print(f"  Physics weight: {args.physics_weight}")
            print(f"  Bond energy: {'✓' if args.physics_use_bond else '✗'}")
            print(f"  Angle energy: {'✓' if args.physics_use_angle else '✗'}")
            print(f"  Dihedral energy: {'✓' if args.physics_use_dihedral else '✗'}")
            print(f"  Non-bonded energy: {'✓' if args.physics_use_nonbonded else '✗'}")
            print("="*60)

    # Training loop
    for epoch in range(start_epoch, args.num_epochs + 1):
        if is_main_process:
            print(f"Epoch {epoch}/{args.num_epochs}")
            print("-" * 60)

        # Set epoch for distributed sampler
        if args.use_ddp and world_size > 1:
            train_loader.sampler.set_epoch(epoch)

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
            monitor_gradients=args.monitor_gradients,
            accumulation_steps=args.accumulation_steps,
            use_amp=args.use_amp,
            scaler=scaler,
            physics_loss_fn=physics_loss_fn,
            physics_weight=args.physics_weight
        )
        train_loss = train_metrics['loss']

        # Format training metrics output based on loss function (only main process)
        if is_main_process:
            train_str = f"Train Loss: {train_loss:.6f}"
            if 'cosine_similarity' in train_metrics:
                train_str += f", Cosine Sim: {train_metrics['cosine_similarity']:.4f}"
            if 'infonce_accuracy' in train_metrics:
                train_str += f", InfoNCE Acc: {train_metrics['infonce_accuracy']*100:.2f}%"
            print(train_str)

            # Print physics constraint loss metrics if available (V3 improvement)
            if 'physics_loss' in train_metrics:
                print(f"  Physics Loss: {train_metrics['physics_loss']:.6f} (weight: {train_metrics['physics_weight']:.3f})")
                if 'bond_energy' in train_metrics:
                    print(f"    Bond: {train_metrics['bond_energy']:.4f}, Angle: {train_metrics['angle_energy']:.4f}, Dihedral: {train_metrics['dihedral_energy']:.4f}")
                if 'nonbonded_energy' in train_metrics:
                    print(f"    Non-bonded: {train_metrics['nonbonded_energy']:.4f}")

            # Print learnable weights if available (增强版，显示更多信息)
            print("\n  📊 Learnable Weights Monitoring:")
            if 'angle_weight' in train_metrics:
                log_val = train_metrics.get('log_angle_weight', 'N/A')
                grad_val = train_metrics.get('angle_weight_grad', 'N/A')
                print(f"    Angle:     weight={train_metrics['angle_weight']:.4f}, "
                      f"log_space={log_val if isinstance(log_val, str) else f'{log_val:.4f}'}, "
                      f"grad={grad_val if isinstance(grad_val, str) else f'{grad_val:.6f}'}")
                weight_history['angle_weight'].append(train_metrics['angle_weight'])

            if 'dihedral_weight' in train_metrics:
                log_val = train_metrics.get('log_dihedral_weight', 'N/A')
                grad_val = train_metrics.get('dihedral_weight_grad', 'N/A')
                print(f"    Dihedral:  weight={train_metrics['dihedral_weight']:.4f}, "
                      f"log_space={log_val if isinstance(log_val, str) else f'{log_val:.4f}'}, "
                      f"grad={grad_val if isinstance(grad_val, str) else f'{grad_val:.6f}'}")
                weight_history['dihedral_weight'].append(train_metrics['dihedral_weight'])

            if 'nonbonded_weight' in train_metrics:
                log_val = train_metrics.get('log_nonbonded_weight', 'N/A')
                grad_val = train_metrics.get('nonbonded_weight_grad', 'N/A')
                print(f"    Nonbonded: weight={train_metrics['nonbonded_weight']:.4f}, "
                      f"log_space={log_val if isinstance(log_val, str) else f'{log_val:.4f}'}, "
                      f"grad={grad_val if isinstance(grad_val, str) else f'{grad_val:.6f}'}")
                weight_history['nonbonded_weight'].append(train_metrics['nonbonded_weight'])

            # 检查权重是否异常增长
            if 'angle_weight' in train_metrics and train_metrics['angle_weight'] > 0.9:
                print(f"    ⚠️  WARNING: Angle weight is very high ({train_metrics['angle_weight']:.4f})!")
            if 'dihedral_weight' in train_metrics and train_metrics['dihedral_weight'] > 0.9:
                print(f"    ⚠️  WARNING: Dihedral weight is very high ({train_metrics['dihedral_weight']:.4f})!")
            if 'nonbonded_weight' in train_metrics and train_metrics['nonbonded_weight'] > 0.9:
                print(f"    ⚠️  WARNING: Nonbonded weight is very high ({train_metrics['nonbonded_weight']:.4f})!")

        # Validate
        val_metrics = evaluate(
            model, val_loader, device,
            loss_fn=args.loss_fn,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            temperature=args.temperature,
            use_amp=args.use_amp
        )
        val_loss = val_metrics['loss']

        # Format validation metrics output based on loss function (only main process)
        if is_main_process:
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

        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.2e}")

            # Save history
            train_history.append(train_loss)
            val_history.append(val_loss)
            if 'cosine_similarity' in val_metrics:
                cosine_sim_history.append(val_metrics['cosine_similarity'])

        # Save checkpoint (only main process)
        if is_main_process and epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            model_state = model.module.state_dict() if (args.use_ddp and world_size > 1) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            # Clear cache after saving to free up temporary memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Save best model (only main process)
        if is_main_process:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                best_model_path = output_dir / "best_model.pt"
                model_state = model.module.state_dict() if (args.use_ddp and world_size > 1) else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
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

        if is_main_process:
            print()

        # Synchronize all processes at end of epoch
        if args.use_ddp and world_size > 1:
            dist.barrier()

    # Save training history (only main process)
    if is_main_process:
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

        # Print final learnable weights (增强版)
        if args.use_multi_hop or args.use_nonbonded:
            print("\n" + "=" * 60)
            print("📊 Final Learnable Weights Summary")
            print("=" * 60)

            if hasattr(model_for_params, 'get_weight_stats'):
                final_stats = model_for_params.get_weight_stats()

                if 'angle_weight' in final_stats:
                    print(f"  Angle weight:")
                    print(f"    Final value:     {final_stats['angle_weight']:.4f} (initial: 0.500)")
                    print(f"    Log-space param: {final_stats['log_angle_weight']:.4f}")
                    print(f"    Change:          {final_stats['angle_weight'] - 0.5:+.4f}")

                if 'dihedral_weight' in final_stats:
                    print(f"  Dihedral weight:")
                    print(f"    Final value:     {final_stats['dihedral_weight']:.4f} (initial: 0.500)")
                    print(f"    Log-space param: {final_stats['log_dihedral_weight']:.4f}")
                    print(f"    Change:          {final_stats['dihedral_weight'] - 0.5:+.4f}")

                if 'nonbonded_weight' in final_stats:
                    print(f"  Nonbonded weight:")
                    print(f"    Final value:     {final_stats['nonbonded_weight']:.4f} (initial: 0.500)")
                    print(f"    Log-space param: {final_stats['log_nonbonded_weight']:.4f}")
                    print(f"    Change:          {final_stats['nonbonded_weight'] - 0.5:+.4f}")
            else:
                # 后备方案
                if hasattr(model_for_params, 'angle_weight'):
                    print(f"  Angle weight: {model_for_params.angle_weight.item():.4f} (initial: 0.500)")
                if hasattr(model_for_params, 'dihedral_weight'):
                    print(f"  Dihedral weight: {model_for_params.dihedral_weight.item():.4f} (initial: 0.500)")
                if hasattr(model_for_params, 'nonbonded_weight'):
                    print(f"  Nonbonded weight: {model_for_params.nonbonded_weight.item():.4f} (initial: 0.500)")

            print("=" * 60)

    # Cleanup DDP
    if args.use_ddp and world_size > 1:
        cleanup_ddp()


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
    parser.add_argument("--splits_file", type=str, default="data/splits/filtered_splits.json",
                        help="Path to save/load dataset splits")

    # Model arguments (v2.0 - Pure Physical Features)
    parser.add_argument("--input_dim", type=int, default=3,
                        help="Input feature dimension (3: charge, atomic_num, mass)")
    parser.add_argument("--feature_hidden_dim", type=int, default=64,
                        help="Hidden dimension for feature embedding MLP")
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

    # V3 model arguments (advanced improvements)
    parser.add_argument("--use_v3_model", action="store_true", default=False,
                        help="Use V3 model with geometric MP, enhanced invariants, and multi-head attention")
    parser.add_argument("--use_geometric_mp", action="store_true", default=True,
                        help="Use geometric angle/dihedral message passing (V3 only)")
    parser.add_argument("--use_enhanced_invariants", action="store_true", default=True,
                        help="Use enhanced invariant feature extraction 204-dim (V3 only)")
    parser.add_argument("--use_improved_layers", action="store_true", default=True,
                        help="Use improved layers from layers/ (Bessel+Cutoff+ImprovedMP, V3 only)")
    parser.add_argument("--norm_type", type=str, default='layer', choices=['layer', 'rms'],
                        help="Normalization type: 'layer' or 'rms' (V3 only, default: layer)")
    parser.add_argument("--num_attention_heads", type=int, default=4,
                        help="Number of attention heads for pooling (V3 only, default: 4)")
    parser.add_argument("--initial_angle_weight", type=float, default=0.5,
                        help="Initial weight for angle message passing (V3 only, range: 0~1, default: 0.5)")
    parser.add_argument("--initial_dihedral_weight", type=float, default=0.5,
                        help="Initial weight for dihedral message passing (V3 only, range: 0~1, default: 0.5)")
    parser.add_argument("--initial_nonbonded_weight", type=float, default=0.5,
                        help="Initial weight for non-bonded message passing (V3 only, range: 0~1, default: 0.5)")

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
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
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
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps (effective batch size = batch_size * accumulation_steps)")
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Use Automatic Mixed Precision (AMP) training for reduced memory usage (~50% less)")

    # Physics constraint arguments (V3 improvement) - NOT RECOMMENDED
    parser.add_argument("--use_physics_loss", action="store_true", default=False,
                        help="[NOT RECOMMENDED] Enable physics constraint loss (bond/angle/dihedral energies). "
                             "This adds extra complexity without clear benefits for representation learning.")
    parser.add_argument("--physics_weight", type=float, default=0.1,
                        help="Weight for physics constraint loss (default: 0.1, only used if --use_physics_loss is set)")
    parser.add_argument("--physics_use_bond", action="store_true", default=True,
                        help="Include bond stretching energy in physics loss")
    parser.add_argument("--physics_use_angle", action="store_true", default=True,
                        help="Include angle bending energy in physics loss")
    parser.add_argument("--physics_use_dihedral", action="store_true", default=True,
                        help="Include dihedral torsion energy in physics loss")
    parser.add_argument("--physics_use_nonbonded", action="store_true", default=False,
                        help="Include non-bonded (LJ) energy in physics loss (expensive, usually not needed)")

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

    # Distributed training arguments
    parser.add_argument("--use_ddp", action="store_true",
                        help="Use DistributedDataParallel for multi-GPU training")
    parser.add_argument("--world_size", type=int, default=None,
                        help="Number of GPUs to use (default: all available GPUs)")

    args = parser.parse_args()

    # Determine world_size
    if args.use_ddp:
        if args.world_size is None:
            world_size = torch.cuda.device_count()
        else:
            world_size = args.world_size

        if world_size < 1:
            print("Error: No CUDA devices available for distributed training")
            sys.exit(1)

        print(f"\nStarting distributed training with {world_size} GPUs...")
        print("Spawning processes...\n")

        # Use torch.multiprocessing.spawn to launch multiple processes
        mp.spawn(
            train_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU or CPU training
        world_size = 1
        train_worker(0, world_size, args)


if __name__ == "__main__":
    main()
