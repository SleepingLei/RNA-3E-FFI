#!/usr/bin/env python3
"""
诊断脚本：找出没有LayerNorm时出现NaN的根本原因
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
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
print("诊断：为什么去掉LayerNorm会导致NaN")
print("="*80)

# Load data
with open('data/splits/splits.json', 'r') as f:
    splits = json.load(f)
train_ids = splits['train'][:5]  # 只用5个样本

dataset = SimpleDataset(train_ids, 'data/processed/graphs', 'data/processed/ligand_embeddings.h5')
loader = DataLoader(dataset, batch_size=2, shuffle=False)

if len(dataset) == 0:
    print("❌ No data available")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = get_global_encoder()

print(f"\nDevice: {device}")
print(f"Dataset size: {len(dataset)}")

# 测试配置
configs = [
    {"name": "6层 + LayerNorm", "layers": 6, "use_ln": True},
    {"name": "6层 无LayerNorm", "layers": 6, "use_ln": False},
    {"name": "3层 无LayerNorm", "layers": 3, "use_ln": False},
]

for config in configs:
    print("\n" + "="*80)
    print(f"测试配置: {config['name']}")
    print("="*80)

    from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2

    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        atom_embed_dim=32,
        residue_embed_dim=16,
        hidden_irreps='32x0e + 16x1o + 8x2e',
        output_dim=1536,
        num_layers=config['layers'],
        num_radial_basis=8,
        use_multi_hop=True,
        use_nonbonded=True,
        use_gate=True,
        use_layer_norm=config['use_ln'],
        pooling_type='attention',
        dropout=0.0  # 去掉dropout排除干扰
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练5步
    model.train()
    batch = next(iter(loader)).to(device)

    for step in range(5):
        # Forward
        output = model(batch)
        target = batch.y
        if target.dim() == 1:
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else output.size(0)
            target = target.view(batch_size, -1)

        # Compute loss (使用MSE避免cosine的除零问题)
        loss = torch.nn.functional.mse_loss(output, target)

        # 检查
        print(f"\nStep {step}:")
        print(f"  Output: mean={output.mean().item():.6f}, std={output.std().item():.6f}, "
              f"min={output.min().item():.6f}, max={output.max().item():.6f}")
        print(f"  Loss: {loss.item():.6f}")

        # 检查NaN
        if torch.isnan(output).any():
            print(f"  ❌ Output包含NaN！")
            break
        if torch.isnan(loss):
            print(f"  ❌ Loss是NaN！")
            break

        # 检查可学习权重
        if hasattr(model, 'angle_weight'):
            aw = model.angle_weight.item()
            dw = model.dihedral_weight.item()
            nw = model.nonbonded_weight.item()
            print(f"  Weights: angle={aw:.6f}, dihedral={dw:.6f}, nonbonded={nw:.6f}")

            if torch.isnan(model.angle_weight):
                print(f"  ❌ angle_weight是NaN！")
                break

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # 检查梯度
        total_grad_norm = 0
        max_grad = 0
        nan_grads = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                max_grad = max(max_grad, param.grad.abs().max().item())

                if torch.isnan(param.grad).any():
                    nan_grads.append(name)

        total_grad_norm = total_grad_norm ** 0.5

        print(f"  Grad: total_norm={total_grad_norm:.6f}, max={max_grad:.6f}")

        if nan_grads:
            print(f"  ❌ 发现NaN梯度在以下参数中：")
            for name in nan_grads[:5]:
                print(f"     - {name}")
            break

        if torch.isnan(torch.tensor(total_grad_norm)):
            print(f"  ❌ 梯度范数是NaN！")
            break

        if torch.isinf(torch.tensor(total_grad_norm)):
            print(f"  ❌ 梯度范数是Inf（梯度爆炸）！")
            break

        # Update
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 检查更新后的参数
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)

        if nan_params:
            print(f"  ❌ 更新后发现NaN参数：")
            for name in nan_params[:5]:
                print(f"     - {name}")
            break

    # 总结
    if step == 4:
        print(f"\n✓ {config['name']} - 训练5步没有出现NaN")
    else:
        print(f"\n❌ {config['name']} - 在第{step}步出现NaN")

    # 清理
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

print("\n" + "="*80)
print("诊断完成")
print("="*80)
