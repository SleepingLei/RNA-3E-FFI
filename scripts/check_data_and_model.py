#!/usr/bin/env python3
"""
全面检查输入数据和模型输出
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import json
import h5py
import numpy as np
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

def check_tensor_stats(tensor, name, check_nan=True, check_inf=True, check_range=True):
    """检查tensor的统计信息"""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Norm: {tensor.norm().item():.6f}")

    issues = []

    if check_nan and torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        print(f"  ❌ 包含 {nan_count} 个NaN值！")
        issues.append(f"NaN ({nan_count}个)")

    if check_inf and torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        print(f"  ❌ 包含 {inf_count} 个Inf值！")
        issues.append(f"Inf ({inf_count}个)")

    if check_range:
        if tensor.abs().max().item() > 1e6:
            print(f"  ⚠️  存在极大值 (>{1e6})")
            issues.append("极大值")
        if tensor.std().item() < 1e-6:
            print(f"  ⚠️  标准差极小 (<{1e-6})，几乎是常数")
            issues.append("近常数")

    if not issues:
        print(f"  ✓ 数值正常")

    return issues

print("="*80)
print("数据和模型全面检查")
print("="*80)

# Load data
with open('data/splits/splits.json', 'r') as f:
    splits = json.load(f)
train_ids = splits['train'][:10]  # 用10个样本

dataset = SimpleDataset(train_ids, 'data/processed/graphs', 'data/processed/ligand_embeddings.h5')
loader = DataLoader(dataset, batch_size=2, shuffle=False)

print(f"\nDataset size: {len(dataset)}")

if len(dataset) == 0:
    print("❌ 没有可用数据！")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
print("\n" + "="*80)
print("第一部分：检查输入数据")
print("="*80)

print("\n1. 检查Graph数据结构:")
batch = next(iter(loader))
print(f"  Batch size: {batch.num_graphs if hasattr(batch, 'num_graphs') else 'unknown'}")
print(f"  Total nodes: {batch.x.shape[0]}")
print(f"  Total edges: {batch.edge_index.shape[1]}")

# 检查节点特征
all_issues = []
issues = check_tensor_stats(batch.x, "  Node features (x)", check_nan=True, check_inf=True)
all_issues.extend(issues)

# 检查坐标
if hasattr(batch, 'pos'):
    issues = check_tensor_stats(batch.pos, "  Node positions (pos)", check_nan=True, check_inf=True)
    all_issues.extend(issues)
else:
    print("  ⚠️  缺少pos（节点坐标）")

# 检查edge
print(f"\n  Edge index:")
print(f"    Shape: {batch.edge_index.shape}")
print(f"    Max node idx: {batch.edge_index.max().item()}")
print(f"    Min node idx: {batch.edge_index.min().item()}")

if batch.edge_index.min().item() < 0:
    print(f"    ❌ 存在负数索引！")
    all_issues.append("负数edge索引")

if batch.edge_index.max().item() >= batch.x.shape[0]:
    print(f"    ❌ Edge索引超出节点数量！")
    all_issues.append("edge索引越界")

# 检查edge_attr
if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
    issues = check_tensor_stats(batch.edge_attr, "  Edge attributes", check_nan=True, check_inf=True)
    all_issues.extend(issues)

# 检查multi-hop
if hasattr(batch, 'triple_index'):
    print(f"\n  Triple index (angles): {batch.triple_index.shape}")
    if hasattr(batch, 'triple_attr'):
        issues = check_tensor_stats(batch.triple_attr, "  Triple attributes", check_nan=True, check_inf=True)
        all_issues.extend(issues)

if hasattr(batch, 'nonbonded_edge_index'):
    print(f"\n  Nonbonded edges: {batch.nonbonded_edge_index.shape}")
    if hasattr(batch, 'nonbonded_edge_attr'):
        issues = check_tensor_stats(batch.nonbonded_edge_attr, "  Nonbonded attributes", check_nan=True, check_inf=True)
        all_issues.extend(issues)

# 检查target embedding
print("\n2. 检查Target Embeddings:")
target = batch.y
if target.dim() == 1:
    batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 2
    target = target.view(batch_size, -1)

issues = check_tensor_stats(target, "  Target embeddings", check_nan=True, check_inf=True)
all_issues.extend(issues)

# 检查target的diversity
if target.shape[0] >= 2:
    from sklearn.metrics.pairwise import cosine_similarity
    target_np = target.cpu().numpy()
    cos_sim = cosine_similarity(target_np)
    mask = ~np.eye(cos_sim.shape[0], dtype=bool)

    print(f"\n  Target diversity (pairwise cosine similarity):")
    print(f"    Mean: {cos_sim[mask].mean():.6f}")
    print(f"    Std: {cos_sim[mask].std():.6f}")
    print(f"    Min: {cos_sim[mask].min():.6f}")
    print(f"    Max: {cos_sim[mask].max():.6f}")

    if cos_sim[mask].mean() > 0.8:
        print(f"    ⚠️  Target embeddings太相似（平均余弦相似度>0.8）")
        all_issues.append("Target缺乏diversity")

# ============================================================================
print("\n" + "="*80)
print("第二部分：检查模型和前向传播")
print("="*80)

encoder = get_global_encoder()
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
    use_layer_norm=False,
    pooling_type='attention',
    dropout=0.0
).to(device)

print("\n3. 检查模型初始化:")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 检查可学习权重
if hasattr(model, 'get_angle_weight'):
    aw = model.get_angle_weight().item()
    dw = model.get_dihedral_weight().item()
    nw = model.get_nonbonded_weight().item()
    print(f"\n  Learnable weights:")
    print(f"    angle_weight: {aw:.6f}")
    print(f"    dihedral_weight: {dw:.6f}")
    print(f"    nonbonded_weight: {nw:.6f}")

    if np.isnan(aw) or np.isnan(dw) or np.isnan(nw):
        print(f"    ❌ 权重初始化为NaN！")
        all_issues.append("权重初始化NaN")

# 检查模型参数是否有NaN
nan_params = []
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        nan_params.append(name)

if nan_params:
    print(f"\n  ❌ 发现 {len(nan_params)} 个参数包含NaN:")
    for name in nan_params[:5]:
        print(f"    - {name}")
    all_issues.append("参数包含NaN")
else:
    print(f"  ✓ 所有参数初始化正常")

# ============================================================================
print("\n4. 检查前向传播:")

batch = batch.to(device)
model.eval()

with torch.no_grad():
    try:
        output = model(batch)

        # 检查输出
        issues = check_tensor_stats(output, "  Model output", check_nan=True, check_inf=True)
        all_issues.extend(issues)

        # 检查输出形状
        expected_shape = (batch.num_graphs if hasattr(batch, 'num_graphs') else 2, 1536)
        if output.shape != expected_shape:
            print(f"  ⚠️  输出形状不匹配：期望{expected_shape}，实际{output.shape}")
            all_issues.append("输出形状错误")

        # 计算与target的相似度
        target = batch.y
        if target.dim() == 1:
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else output.size(0)
            target = target.view(batch_size, -1)

        cosine_sim = F.cosine_similarity(output, target, dim=1)
        print(f"\n  初始性能:")
        print(f"    Cosine similarity: {cosine_sim.mean().item():.6f} ± {cosine_sim.std().item():.6f}")
        print(f"    MSE: {F.mse_loss(output, target).item():.6f}")

        if cosine_sim.mean().item() > 0.5:
            print(f"    ⚠️  未训练模型的余弦相似度异常高 (>0.5)")
            all_issues.append("初始相似度异常")

    except Exception as e:
        print(f"  ❌ 前向传播出错: {e}")
        all_issues.append(f"前向传播错误: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================================================================
print("\n" + "="*80)
print("第三部分：检查梯度计算")
print("="*80)

print("\n5. 检查反向传播:")

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

try:
    # Forward
    output = model(batch)
    target = batch.y
    if target.dim() == 1:
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else output.size(0)
        target = target.view(batch_size, -1)

    # Compute loss
    cosine_sim = F.cosine_similarity(output, target, dim=1)
    loss = (1 - cosine_sim).mean()

    print(f"  Loss: {loss.item():.6f}")

    if torch.isnan(loss):
        print(f"  ❌ Loss是NaN！")
        all_issues.append("Loss是NaN")

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # 检查梯度
    nan_grads = []
    zero_grads = []
    total_grad_norm = 0
    max_grad = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            max_grad = max(max_grad, param.grad.abs().max().item())

            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            elif grad_norm < 1e-10:
                zero_grads.append(name)
        else:
            zero_grads.append(name + " (no grad)")

    total_grad_norm = total_grad_norm ** 0.5

    print(f"\n  梯度统计:")
    print(f"    Total gradient norm: {total_grad_norm:.6f}")
    print(f"    Max gradient: {max_grad:.6f}")
    print(f"    Parameters with gradients: {len([p for p in model.parameters() if p.grad is not None])}")
    print(f"    Parameters without gradients: {len([p for p in model.parameters() if p.grad is None])}")

    if nan_grads:
        print(f"\n  ❌ 发现 {len(nan_grads)} 个参数的梯度是NaN:")
        for name in nan_grads[:5]:
            print(f"    - {name}")
        all_issues.append(f"{len(nan_grads)}个NaN梯度")

    if torch.isnan(torch.tensor(total_grad_norm)):
        print(f"  ❌ 总梯度范数是NaN！")
        all_issues.append("总梯度NaN")
    elif total_grad_norm < 1e-6:
        print(f"  ⚠️  梯度范数极小 (<1e-6)，可能是梯度消失")
        all_issues.append("梯度消失")
    elif total_grad_norm > 1e6:
        print(f"  ⚠️  梯度范数极大 (>1e6)，可能是梯度爆炸")
        all_issues.append("梯度爆炸")
    else:
        print(f"  ✓ 梯度范数正常")

    if len(zero_grads) > 10:
        print(f"\n  ⚠️  有 {len(zero_grads)} 个参数的梯度为0或不存在")
        print(f"    前5个: {zero_grads[:5]}")

except Exception as e:
    print(f"  ❌ 反向传播出错: {e}")
    all_issues.append(f"反向传播错误: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
print("\n" + "="*80)
print("总结")
print("="*80)

if all_issues:
    print(f"\n❌ 发现 {len(all_issues)} 个问题:")
    for i, issue in enumerate(all_issues, 1):
        print(f"  {i}. {issue}")

    print(f"\n🔧 建议:")
    if "NaN" in str(all_issues):
        print("  - 检查数据预处理是否正确")
        print("  - 检查模型初始化")
        print("  - 运行 python scripts/debug_exact_nan_location.py 精确定位")
    if "diversity" in str(all_issues).lower():
        print("  - 检查target embeddings是否正确加载")
        print("  - 检查embeddings是否被过度归一化")
    if "梯度" in str(all_issues):
        print("  - 尝试减少网络层数 (--num_layers 3)")
        print("  - 尝试增大学习率 (--lr 1e-3)")
        print("  - 尝试使用LayerNorm (--use_layer_norm)")
else:
    print("\n✅ 所有检查通过！数据和模型看起来正常。")
    print("\n可以开始训练:")
    print("  bash test_nan_fix.sh")

print("\n" + "="*80)
