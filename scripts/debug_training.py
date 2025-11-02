#!/usr/bin/env python3
"""
Diagnostic script to check actual model training with real data.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import json
import h5py
from torch_geometric.loader import DataLoader

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.amber_vocabulary import get_global_encoder

print("="*60)
print("Diagnostic Script for Training Issues")
print("="*60)

# Load configuration
config_path = Path("models/checkpoints_v2_normalized_1351_4/config.json")
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("\nLoaded config from checkpoint")
else:
    print("\n⚠️  No config found, using defaults")
    config = {
        'graph_dir': 'data/processed/graphs',
        'embeddings_path': 'data/processed/ligand_embeddings.h5',
        'splits_file': 'data/splits/splits.json',
        'atom_embed_dim': 32,
        'residue_embed_dim': 16,
        'hidden_irreps': '32x0e + 16x1o + 8x2e',
        'output_dim': 1536,
        'num_layers': 4,
        'num_radial_basis': 8,
        'use_multi_hop': False,
        'use_nonbonded': True,
        'use_weight_constraints': False,
        'use_gate': True,
        'use_layer_norm': False,
        'pooling_type': 'attention',
        'dropout': 0.10,
        'batch_size': 2,
        'lr': 1e-3,
        'weight_decay': 5e-6
    }

# Define dataset class (copied from 04_train_model.py)
class LigandEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, complex_ids, graph_dir, ligand_embeddings_path, validate_format=True):
        self.complex_ids = complex_ids
        self.graph_dir = Path(graph_dir)
        self.validate_format = validate_format

        # Load all ligand embeddings
        self.ligand_embeddings = {}
        with h5py.File(ligand_embeddings_path, 'r') as f:
            for key in f.keys():
                self.ligand_embeddings[key] = torch.tensor(f[key][:], dtype=torch.float)

        # Create mapping from complex_id to embedding key
        self.id_to_embedding_key = {}
        for complex_id in complex_ids:
            if '_model' in complex_id:
                base_id = '_'.join(complex_id.split('_model')[0].split('_'))
            else:
                base_id = complex_id

            if base_id in self.ligand_embeddings:
                self.id_to_embedding_key[complex_id] = base_id

        # Filter to only include valid complexes
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
        graph_path = self.graph_dir / f"{complex_id}.pt"
        data = torch.load(graph_path, weights_only=False)
        embedding_key = self.id_to_embedding_key[complex_id]
        ligand_embedding = self.ligand_embeddings[embedding_key]
        data.y = torch.tensor(ligand_embedding, dtype=torch.float32)
        return data

# Load dataset
print("\n" + "-"*60)
print("Loading Dataset")
print("-"*60)

with open(config['splits_file'], 'r') as f:
    splits = json.load(f)
train_ids = splits['train'][:10]  # Use first 10 samples for quick test

print(f"First 10 train IDs: {train_ids}")

# Debug: check which files exist
graph_dir = Path(config['graph_dir'])
for tid in train_ids:
    graph_path = graph_dir / f"{tid}.pt"
    print(f"  {tid}: graph exists={graph_path.exists()}")

dataset = LigandEmbeddingDataset(
    train_ids,
    config['graph_dir'],
    config['embeddings_path'],
    validate_format=False
)

if len(dataset) == 0:
    print("\n❌ ERROR: Dataset has 0 samples!")
    print("This usually means graph files don't exist at the expected paths.")
    print("Trying alternative approach...")

    # Try to find any .pt files in the graph directory
    import glob
    all_graphs = list(Path(config['graph_dir']).glob("*.pt"))
    print(f"\nFound {len(all_graphs)} .pt files in {config['graph_dir']}")
    if all_graphs:
        print(f"Example files: {[f.name for f in all_graphs[:5]]}")

        # Try loading first few files that exist
        existing_ids = []
        for f in all_graphs[:10]:
            # Extract complex_id from filename
            complex_id = f.stem
            existing_ids.append(complex_id)

        print(f"\nRetrying with existing file IDs: {existing_ids[:5]}")
        dataset = LigandEmbeddingDataset(
            existing_ids,
            config['graph_dir'],
            config['embeddings_path'],
            validate_format=False
        )

loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

print(f"Loaded {len(dataset)} samples")

# Initialize model
print("\n" + "-"*60)
print("Initializing Model")
print("-"*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if config.get('use_weight_constraints', False):
    from models.e3_gnn_encoder_v2_fixed import RNAPocketEncoderV2Fixed as ModelClass
    print("Using RNAPocketEncoderV2Fixed")
else:
    from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2 as ModelClass
    print("Using RNAPocketEncoderV2")

encoder = get_global_encoder()
model = ModelClass(
    num_atom_types=encoder.num_atom_types,
    num_residues=encoder.num_residues,
    atom_embed_dim=config['atom_embed_dim'],
    residue_embed_dim=config['residue_embed_dim'],
    hidden_irreps=config['hidden_irreps'],
    output_dim=config['output_dim'],
    num_layers=config['num_layers'],
    num_radial_basis=config['num_radial_basis'],
    use_multi_hop=config.get('use_multi_hop', False),
    use_nonbonded=config.get('use_nonbonded', True),
    use_gate=config.get('use_gate', True),
    use_layer_norm=config.get('use_layer_norm', False),
    pooling_type=config.get('pooling_type', 'attention'),
    dropout=config.get('dropout', 0.10)
)
model = model.to(device)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Test forward pass
print("\n" + "-"*60)
print("Test 1: Forward Pass")
print("-"*60)

model.eval()
batch = next(iter(loader)).to(device)

with torch.no_grad():
    output = model(batch)
    target = batch.y
    if target.dim() == 1:
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else output.size(0)
        target = target.view(batch_size, -1)

print(f"Output shape: {output.shape}")
print(f"Target shape: {target.shape}")
print(f"\nOutput statistics:")
print(f"  Mean: {output.mean().item():.6f}")
print(f"  Std: {output.std().item():.6f}")
print(f"  Min: {output.min().item():.6f}")
print(f"  Max: {output.max().item():.6f}")
print(f"  Norm (per sample): {output.norm(dim=1).mean().item():.6f}")

print(f"\nTarget statistics:")
print(f"  Mean: {target.mean().item():.6f}")
print(f"  Std: {target.std().item():.6f}")
print(f"  Min: {target.min().item():.6f}")
print(f"  Max: {target.max().item():.6f}")
print(f"  Norm (per sample): {target.norm(dim=1).mean().item():.6f}")

# Check if output is constant
if output.std().item() < 1e-6:
    print("❌ ERROR: Model output is nearly constant!")
elif torch.isnan(output).any():
    print("❌ ERROR: Model output contains NaN!")
elif torch.isinf(output).any():
    print("❌ ERROR: Model output contains Inf!")
else:
    print("✓ Model output looks reasonable")

# Compute initial loss
cosine_sim = F.cosine_similarity(output, target, dim=1)
loss = (1 - cosine_sim).mean()
print(f"\nInitial loss: {loss.item():.6f}")
print(f"Initial cosine similarity: {cosine_sim.mean().item():.6f}")

# Test backward pass
print("\n" + "-"*60)
print("Test 2: Backward Pass (Single Step)")
print("-"*60)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

# Forward
output = model(batch)
target = batch.y
if target.dim() == 1:
    batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else output.size(0)
    target = target.view(batch_size, -1)

cosine_sim = F.cosine_similarity(output, target, dim=1)
loss = (1 - cosine_sim).mean()

print(f"Loss before backward: {loss.item():.6f}")

# Backward
optimizer.zero_grad()
loss.backward()

# Check gradients
total_grad_norm = 0
num_params_with_grad = 0
num_params_without_grad = 0
max_grad = 0
min_grad = float('inf')

for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        total_grad_norm += grad_norm ** 2
        num_params_with_grad += 1
        max_grad = max(max_grad, param.grad.abs().max().item())
        min_grad = min(min_grad, param.grad.abs().min().item())

        if grad_norm < 1e-10:
            print(f"⚠️  Very small gradient for {name}: {grad_norm:.2e}")
    else:
        num_params_without_grad += 1
        print(f"⚠️  No gradient for {name}")

total_grad_norm = total_grad_norm ** 0.5

print(f"\nGradient statistics:")
print(f"  Total gradient norm: {total_grad_norm:.6f}")
print(f"  Params with gradients: {num_params_with_grad}")
print(f"  Params without gradients: {num_params_without_grad}")
print(f"  Max gradient: {max_grad:.6f}")
print(f"  Min gradient: {min_grad:.6f}")

if total_grad_norm < 1e-8:
    print("❌ ERROR: Gradients are too small (vanishing gradient)")
elif total_grad_norm > 1e3:
    print("⚠️  WARNING: Gradients are very large")
else:
    print("✓ Gradient norm looks reasonable")

# Apply gradient clipping (as in training script)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
clipped_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
print(f"  Gradient norm after clipping: {clipped_grad_norm:.6f}")

if abs(clipped_grad_norm - total_grad_norm) > 0.01:
    print(f"⚠️  WARNING: Gradient was clipped (before: {total_grad_norm:.6f}, after: {clipped_grad_norm:.6f})")

# Optimizer step
optimizer.step()
print("\n✓ Optimizer step completed")

# Test training for multiple steps
print("\n" + "-"*60)
print("Test 3: Multi-Step Training (10 steps on same batch)")
print("-"*60)

model.train()
losses = []
cosine_sims = []

for step in range(10):
    # Forward
    output = model(batch)
    target = batch.y
    if target.dim() == 1:
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else output.size(0)
        target = target.view(batch_size, -1)

    cosine_sim = F.cosine_similarity(output, target, dim=1)
    loss = (1 - cosine_sim).mean()

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    losses.append(loss.item())
    cosine_sims.append(cosine_sim.mean().item())

    if step == 0 or step == 9:
        print(f"Step {step}: Loss={loss.item():.6f}, Cosine Sim={cosine_sim.mean().item():.6f}")

print(f"\nLoss change: {losses[0]:.6f} -> {losses[-1]:.6f}")
print(f"Cosine sim change: {cosine_sims[0]:.6f} -> {cosine_sims[-1]:.6f}")

if losses[-1] < losses[0]:
    reduction = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"✓ Loss decreased by {reduction:.2f}%")
else:
    print(f"❌ ERROR: Loss did not decrease!")
    print(f"   Loss increased by: {losses[-1] - losses[0]:.6f}")

# Additional diagnostics
print("\n" + "-"*60)
print("Test 4: Check for Common Issues")
print("-"*60)

# Check if model has learnable parameters
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {num_params:,}")
print(f"Trainable parameters: {num_trainable:,}")

if num_trainable == 0:
    print("❌ ERROR: No trainable parameters!")
else:
    print(f"✓ Model has {num_trainable:,} trainable parameters")

# Check learning rate
lr = optimizer.param_groups[0]['lr']
print(f"\nLearning rate: {lr:.2e}")
if lr < 1e-6:
    print("⚠️  WARNING: Learning rate is very small")
elif lr > 1e-1:
    print("⚠️  WARNING: Learning rate is very large")
else:
    print("✓ Learning rate looks reasonable")

# Check weight decay
wd = optimizer.param_groups[0]['weight_decay']
print(f"Weight decay: {wd:.2e}")

print("\n" + "="*60)
print("Diagnostic Complete")
print("="*60)
