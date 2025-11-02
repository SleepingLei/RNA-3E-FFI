#!/usr/bin/env python3
"""
精确定位NaN出现的位置
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import json
import h5py
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.amber_vocabulary import get_global_encoder

# 简单Dataset
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, ids, graph_dir, emb_path):
        self.ids = []
        self.embs = {}
        with h5py.File(emb_path, 'r') as f:
            for cid in ids:
                base_id = cid.split('_model')[0] if '_model' in cid else cid
                graph_path = Path(graph_dir) / f'{cid}.pt'
                if graph_path.exists() and base_id in f:
                    self.ids.append(cid)
                    self.embs[cid] = torch.tensor(f[base_id][:], dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]
        data = torch.load(Path('data/processed/graphs') / f'{cid}.pt', weights_only=False)
        data.y = self.embs[cid]
        return data

print("="*80)
print("精确定位NaN出现位置")
print("="*80)

# Load data
with open('data/splits/splits.json', 'r') as f:
    splits = json.load(f)
train_ids = splits['train'][:2]  # 只用2个样本

dataset = SimpleDataset(train_ids, 'data/processed/graphs', 'data/processed/ligand_embeddings.h5')
loader = DataLoader(dataset, batch_size=2, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = get_global_encoder()

print(f"\nDevice: {device}")
print(f"Dataset size: {len(dataset)}")

from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2

model = RNAPocketEncoderV2(
    num_atom_types=encoder.num_atom_types,
    num_residues=encoder.num_residues,
    atom_embed_dim=32,
    residue_embed_dim=16,
    hidden_irreps='32x0e + 16x1o + 8x2e',
    output_dim=1536,
    num_layers=6,
    num_radial_basis=8,
    use_multi_hop=True,
    use_nonbonded=True,
    use_gate=True,
    use_layer_norm=False,  # 不用LayerNorm
    pooling_type='attention',
    dropout=0.0
).to(device)

print("\n" + "="*80)
print("检查初始化")
print("="*80)

# 检查初始化的参数
print("\n1. 检查angle/dihedral/nonbonded权重初始化:")
if hasattr(model, 'angle_weight_raw'):
    print(f"  angle_weight_raw: {model.angle_weight_raw.item():.6f}")
    print(f"  angle_weight (computed): {model.angle_weight.item() if model.angle_weight is not None else 'None':.6f}")
    if torch.isnan(model.angle_weight_raw):
        print("  ❌ angle_weight_raw 初始化就是NaN！")
    if model.angle_weight is not None and torch.isnan(model.angle_weight):
        print("  ❌ angle_weight (property) 是NaN！")

if hasattr(model, 'dihedral_weight_raw'):
    print(f"  dihedral_weight_raw: {model.dihedral_weight_raw.item():.6f}")
    print(f"  dihedral_weight (computed): {model.dihedral_weight.item() if model.dihedral_weight is not None else 'None':.6f}")

if hasattr(model, 'nonbonded_weight_raw'):
    print(f"  nonbonded_weight_raw: {model.nonbonded_weight_raw.item():.6f}")
    print(f"  nonbonded_weight (computed): {model.nonbonded_weight.item() if model.nonbonded_weight is not None else 'None':.6f}")

print("\n" + "="*80)
print("第一次Forward Pass")
print("="*80)

batch = next(iter(loader)).to(device)

model.eval()
with torch.no_grad():
    print("\n2. 运行forward pass...")
    try:
        output = model(batch)

        print(f"  Output shape: {output.shape}")
        print(f"  Output stats:")
        print(f"    mean: {output.mean().item():.6f}")
        print(f"    std: {output.std().item():.6f}")
        print(f"    min: {output.min().item():.6f}")
        print(f"    max: {output.max().item():.6f}")

        if torch.isnan(output).any():
            print(f"  ❌ Output包含NaN！")
            print(f"    NaN数量: {torch.isnan(output).sum().item()}")
        else:
            print(f"  ✓ Output正常")

    except Exception as e:
        print(f"  ❌ Forward pass出错: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("测试Backward Pass")
print("="*80)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\n3. 运行backward pass...")
batch = next(iter(loader)).to(device)

try:
    # Forward
    output = model(batch)
    target = batch.y
    if target.dim() == 1:
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else output.size(0)
        target = target.view(batch_size, -1)

    print(f"  Output: mean={output.mean().item():.6f}, contains NaN: {torch.isnan(output).any()}")
    print(f"  Target: mean={target.mean().item():.6f}, contains NaN: {torch.isnan(target).any()}")

    # Compute loss
    cosine_sim = F.cosine_similarity(output, target, dim=1)
    print(f"  Cosine sim: {cosine_sim}")

    if torch.isnan(cosine_sim).any():
        print(f"  ❌ Cosine similarity包含NaN！")
        print(f"    Output norms: {output.norm(dim=1)}")
        print(f"    Target norms: {target.norm(dim=1)}")

    loss = (1 - cosine_sim).mean()
    print(f"  Loss: {loss.item():.6f}")

    if torch.isnan(loss):
        print(f"  ❌ Loss是NaN！")

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    print("\n4. 检查梯度:")
    nan_grads = []
    total_grad_norm = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2

            if torch.isnan(param.grad).any():
                nan_grads.append((name, param.grad))
                print(f"  ❌ {name}: grad包含NaN")
                print(f"    Grad stats: mean={param.grad.mean().item()}, std={param.grad.std().item()}")
                print(f"    Param stats: mean={param.data.mean().item()}, std={param.data.std().item()}")

    total_grad_norm = total_grad_norm ** 0.5
    print(f"\n  Total grad norm: {total_grad_norm}")

    if torch.isnan(torch.tensor(total_grad_norm)):
        print(f"  ❌ 总梯度范数是NaN！")
        print(f"\n  发现 {len(nan_grads)} 个参数的梯度是NaN")

        # 找出第一个NaN梯度的参数
        if nan_grads:
            print(f"\n  第一个NaN梯度的参数: {nan_grads[0][0]}")

except Exception as e:
    print(f"  ❌ Backward pass出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("检查angle_weight的梯度计算")
print("="*80)

# 专门测试angle_weight的property
if hasattr(model, 'angle_weight_raw'):
    print("\n5. 测试angle_weight property:")

    # 手动测试
    with torch.enable_grad():
        model.angle_weight_raw.requires_grad = True

        # 通过property获取
        aw = model.angle_weight
        print(f"  angle_weight value: {aw.item() if aw is not None else 'None':.6f}")

        if aw is not None:
            # 测试是否可以反向传播
            test_loss = aw.sum()
            test_loss.backward()

            if model.angle_weight_raw.grad is not None:
                print(f"  angle_weight_raw.grad: {model.angle_weight_raw.grad.item():.6f}")
                if torch.isnan(model.angle_weight_raw.grad):
                    print(f"  ❌ angle_weight_raw的梯度是NaN！")
            else:
                print(f"  ⚠️  angle_weight_raw没有梯度")

print("\n" + "="*80)
print("诊断完成")
print("="*80)
