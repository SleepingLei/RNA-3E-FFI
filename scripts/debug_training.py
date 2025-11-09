#!/usr/bin/env python3
"""
训练梯度稳定性调试脚本 - 基于实际训练流程

完全模拟 scripts/04_train_model.py 的训练流程，用于诊断梯度不稳定问题。
可以在远程服务器上运行，生成详细的诊断报告。
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch_geometric.loader import DataLoader
from torch.cuda.amp import autocast, GradScaler
import h5py
import warnings
import numpy as np

# Import models and components
try:
    from models.improved_components import PhysicsConstraintLoss
    from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3
    _has_v3_model = True
    _has_physics_loss = True
except ImportError:
    _has_v3_model = False
    _has_physics_loss = False
    warnings.warn("V3 components not available")

try:
    from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
    _has_v2_model = True
except ImportError:
    _has_v2_model = False


# ============================================================================
# Copy Dataset class from 04_train_model.py
# ============================================================================

class LigandEmbeddingDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-computed graphs and ligand embeddings."""

    def __init__(self, complex_ids, graph_dir, ligand_embeddings_path, validate_format=True):
        self.complex_ids = complex_ids
        self.graph_dir = Path(graph_dir)
        self.validate_format = validate_format

        # Load all ligand embeddings
        self.ligand_embeddings = {}
        with h5py.File(ligand_embeddings_path, 'r') as f:
            for key in f.keys():
                self.ligand_embeddings[key] = torch.tensor(
                    f[key][:],
                    dtype=torch.float
                )

        # Create mapping from complex_id to embedding key
        self.id_to_embedding_key = {}
        for complex_id in complex_ids:
            if '_model' in complex_id:
                base_id = '_'.join(complex_id.split('_model')[0].split('_'))
            else:
                base_id = complex_id

            if base_id in self.ligand_embeddings:
                self.id_to_embedding_key[complex_id] = base_id

        # Filter valid IDs
        self.valid_ids = []
        for complex_id in complex_ids:
            graph_path = self.graph_dir / f"{complex_id}.pt"
            if graph_path.exists() and complex_id in self.id_to_embedding_key:
                if validate_format:
                    try:
                        data = torch.load(graph_path, weights_only=False)
                        if data.x.shape[1] != 3:
                            continue
                        if not hasattr(data, 'edge_index'):
                            continue
                    except Exception:
                        continue
                self.valid_ids.append(complex_id)

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


# ============================================================================
# Copy loss function from 04_train_model.py
# ============================================================================

def compute_loss(pred, target, loss_fn='cosine', cosine_weight=0.7, mse_weight=0.3, temperature=0.07):
    """Compute loss based on specified loss function."""
    metrics = {}

    if loss_fn == 'mse':
        loss = F.mse_loss(pred, target)
        metrics['mse_loss'] = loss.item()

    elif loss_fn == 'cosine':
        cosine_sim = F.cosine_similarity(pred, target, dim=-1).mean()
        loss = 1 - cosine_sim
        metrics['cosine_similarity'] = cosine_sim.item()

    elif loss_fn == 'cosine_mse':
        cosine_sim = F.cosine_similarity(pred, target, dim=-1).mean()
        cosine_loss = 1 - cosine_sim
        mse = F.mse_loss(pred, target)
        loss = cosine_weight * cosine_loss + mse_weight * mse
        metrics['cosine_similarity'] = cosine_sim.item()
        metrics['cosine_loss'] = cosine_loss.item()
        metrics['mse_loss'] = mse.item()

    elif loss_fn == 'infonce':
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        similarity_matrix = torch.mm(pred_norm, target_norm.t()) / temperature
        labels = torch.arange(pred.size(0), device=pred.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        with torch.no_grad():
            predictions = torch.argmax(similarity_matrix, dim=1)
            accuracy = (predictions == labels).float().mean()
            metrics['infonce_accuracy'] = accuracy.item()

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return loss, metrics


# ============================================================================
# Monitoring Classes
# ============================================================================

class GradientMonitor:
    """Monitor gradient statistics with detailed analysis."""

    def __init__(self):
        self.grad_stats = {}
        self.problematic_params = []

    def check_gradients(self, model):
        """Check all gradients and collect statistics."""
        self.grad_stats = {}
        self.problematic_params = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_norm = grad.norm().item()
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()

                self.grad_stats[name] = {
                    'norm': grad_norm,
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                }

                # Flag problematic gradients
                if has_nan or has_inf:
                    self.problematic_params.append({
                        'name': name,
                        'issue': 'NaN/Inf',
                        'norm': grad_norm,
                    })
                elif grad_norm > 100.0:
                    self.problematic_params.append({
                        'name': name,
                        'issue': 'Large gradient',
                        'norm': grad_norm,
                    })

    def get_summary(self):
        """Get summary statistics."""
        if not self.grad_stats:
            return None

        norms = [s['norm'] for s in self.grad_stats.values()]
        return {
            'num_params': len(self.grad_stats),
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'max_norm': np.max(norms),
            'min_norm': np.min(norms),
            'num_problematic': len(self.problematic_params),
        }


class ActivationMonitor:
    """Monitor activation statistics during forward pass."""

    def __init__(self):
        self.activation_stats = {}
        self.problematic_layers = []
        self.hooks = []

    def register_hooks(self, model):
        """Register hooks to monitor activations."""

        def hook_fn(module, input, output, name):
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                output_max = output.abs().max().item()

                self.activation_stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                }

                if has_nan or has_inf or output_max > 1000:
                    self.problematic_layers.append({
                        'name': name,
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'max_value': output_max,
                    })

        # Register hooks for key layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# Debug Functions
# ============================================================================

def check_data_statistics(batch, device):
    """Check data statistics and normalization."""
    print("\n" + "="*70)
    print("数据统计检查")
    print("="*70)

    batch = batch.to(device)
    stats = {}

    # Node features
    print(f"\n节点特征 (x): {batch.x.shape}")
    print(f"  Range: [{batch.x.min().item():.4f}, {batch.x.max().item():.4f}]")
    print(f"  Mean: {batch.x.mean().item():.4f}, Std: {batch.x.std().item():.4f}")
    stats['x_range'] = [batch.x.min().item(), batch.x.max().item()]

    # Positions
    print(f"\n位置 (pos): {batch.pos.shape}")
    print(f"  Range: [{batch.pos.min().item():.4f}, {batch.pos.max().item():.4f}]")
    stats['pos_range'] = [batch.pos.min().item(), batch.pos.max().item()]

    # Edge attributes
    if hasattr(batch, 'edge_attr') and batch.edge_attr.shape[0] > 0:
        print(f"\n键参数 (edge_attr): {batch.edge_attr.shape}")
        print(f"  Col 0 range: [{batch.edge_attr[:, 0].min().item():.4f}, {batch.edge_attr[:, 0].max().item():.4f}]")
        print(f"  Col 1 range: [{batch.edge_attr[:, 1].min().item():.4f}, {batch.edge_attr[:, 1].max().item():.4f}]")
        stats['edge_attr_ranges'] = [
            [batch.edge_attr[:, 0].min().item(), batch.edge_attr[:, 0].max().item()],
            [batch.edge_attr[:, 1].min().item(), batch.edge_attr[:, 1].max().item()],
        ]

    # Triple attributes
    if hasattr(batch, 'triple_attr') and batch.triple_attr.shape[0] > 0:
        print(f"\n角度参数 (triple_attr): {batch.triple_attr.shape}")
        print(f"  Col 0 (theta_eq/180): [{batch.triple_attr[:, 0].min().item():.4f}, {batch.triple_attr[:, 0].max().item():.4f}]")
        print(f"  Col 1 (k/200): [{batch.triple_attr[:, 1].min().item():.4f}, {batch.triple_attr[:, 1].max().item():.4f}]")

        # Check expected ranges
        if batch.triple_attr[:, 0].max().item() > 2.5:
            print(f"  ⚠️  警告: Col 0 超出预期范围 [0, ~2]，可能未归一化")
        if batch.triple_attr[:, 1].max().item() > 1.5:
            print(f"  ⚠️  警告: Col 1 超出预期范围 [0, ~1]，可能未归一化")

    # Quadra attributes
    if hasattr(batch, 'quadra_attr') and batch.quadra_attr.shape[0] > 0:
        print(f"\n二面角参数 (quadra_attr): {batch.quadra_attr.shape}")
        for i in range(min(3, batch.quadra_attr.shape[1])):
            print(f"  Col {i}: [{batch.quadra_attr[:, i].min().item():.4f}, {batch.quadra_attr[:, i].max().item():.4f}]")

    return stats


def debug_forward_backward(model, batch, target, optimizer, loss_fn, device,
                          use_amp=False, physics_loss_fn=None, physics_weight=0.1,
                          cosine_weight=0.7, mse_weight=0.3, temperature=0.07):
    """Debug one forward-backward pass."""

    print("\n" + "="*70)
    print("前向-反向传播调试")
    print("="*70)

    # Setup monitors
    act_monitor = ActivationMonitor()
    act_monitor.register_hooks(model)
    grad_monitor = GradientMonitor()

    batch = batch.to(device)
    target = target.to(device)

    try:
        # Forward pass
        print("\n[1/4] 前向传播...")
        with autocast(enabled=use_amp):
            output = model(batch)

            # Reshape target if needed
            if target.dim() == 1:
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else output.size(0)
                target = target.view(batch_size, -1)

            # Compute main loss
            loss, metrics = compute_loss(
                output, target,
                loss_fn=loss_fn,
                cosine_weight=cosine_weight,
                mse_weight=mse_weight,
                temperature=temperature
            )

            # Add physics loss if enabled
            if physics_loss_fn is not None:
                physics_loss, physics_dict = physics_loss_fn(batch)
                total_loss = loss + physics_weight * physics_loss
                metrics['physics_loss'] = physics_loss.item()
                for key, val in physics_dict.items():
                    metrics[key] = val
            else:
                total_loss = loss

        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  主损失: {loss.item():.6f}")
        if physics_loss_fn is not None:
            print(f"  物理损失: {physics_loss.item():.6f}")
            print(f"  总损失: {total_loss.item():.6f}")

        # Check activations
        if act_monitor.problematic_layers:
            print(f"\n  ⚠️  发现 {len(act_monitor.problematic_layers)} 个异常激活层:")
            for layer in act_monitor.problematic_layers[:3]:
                print(f"    - {layer['name']}: NaN={layer['has_nan']}, Inf={layer['has_inf']}, Max={layer['max_value']:.2e}")
        else:
            print("  ✓ 激活值正常")

        # Backward pass
        print("\n[2/4] 反向传播...")
        optimizer.zero_grad()
        total_loss.backward()

        # Check gradients
        print("\n[3/4] 梯度检查...")
        grad_monitor.check_gradients(model)
        grad_summary = grad_monitor.get_summary()

        if grad_summary:
            print(f"  参数数量: {grad_summary['num_params']}")
            print(f"  梯度范数: Mean={grad_summary['mean_norm']:.4f}, Max={grad_summary['max_norm']:.4f}")

            if grad_monitor.problematic_params:
                print(f"\n  ⚠️  发现 {len(grad_monitor.problematic_params)} 个异常梯度:")
                for param in grad_monitor.problematic_params[:5]:
                    print(f"    - {param['name']}: {param['issue']}, Norm={param['norm']:.4e}")
            else:
                print("  ✓ 梯度正常")

        # Gradient clipping simulation
        print("\n[4/4] 梯度裁剪模拟...")
        max_norm_before = max([s['norm'] for s in grad_monitor.grad_stats.values()])
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_monitor.check_gradients(model)
        max_norm_after = max([s['norm'] for s in grad_monitor.grad_stats.values()])
        print(f"  裁剪前最大梯度范数: {max_norm_before:.4f}")
        print(f"  裁剪后最大梯度范数: {max_norm_after:.4f}")

        act_monitor.remove_hooks()

        return {
            'success': True,
            'loss': loss.item(),
            'metrics': metrics,
            'activation_issues': len(act_monitor.problematic_layers),
            'gradient_issues': len(grad_monitor.problematic_params),
            'grad_summary': grad_summary,
            'problematic_layers': act_monitor.problematic_layers,
            'problematic_grads': grad_monitor.problematic_params,
        }

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        act_monitor.remove_hooks()
        return {'success': False, 'error': str(e)}


def save_report(report, output_dir="debug_logs"):
    """Save debug report to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_training_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    report = convert(report)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 调试报告已保存: {filepath}")
    return filepath


# ============================================================================
# Main Debug Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="训练流程梯度稳定性调试")

    # Data arguments
    parser.add_argument("--graph_dir", type=str, required=True,
                        help="图数据目录")
    parser.add_argument("--ligand_embeddings", type=str, required=True,
                        help="配体嵌入 HDF5 文件")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="调试样本数量")

    # Model arguments
    parser.add_argument("--model_type", type=str, default="v3",
                        choices=["v2", "v3"])
    parser.add_argument("--use_v3_features", action="store_true",
                        help="使用 V3 特征")
    parser.add_argument("--use_multi_hop", action="store_true", default=True,
                        help="使用多跳消息传递")
    parser.add_argument("--use_nonbonded", action="store_true", default=True,
                        help="使用非键相互作用")

    # Training arguments
    parser.add_argument("--loss_fn", type=str, default="cosine",
                        choices=["mse", "cosine", "cosine_mse", "infonce"])
    parser.add_argument("--cosine_weight", type=float, default=0.7)
    parser.add_argument("--mse_weight", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_amp", action="store_true",
                        help="使用混合精度训练")

    # Physics loss arguments
    parser.add_argument("--use_physics_loss", action="store_true")
    parser.add_argument("--physics_weight", type=float, default=0.1)

    # Other arguments
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_report", action="store_true")

    args = parser.parse_args()

    print("="*70)
    print("训练流程梯度稳定性调试")
    print("="*70)
    print(f"图目录: {args.graph_dir}")
    print(f"配体嵌入: {args.ligand_embeddings}")
    print(f"模型: {args.model_type}")
    print(f"损失函数: {args.loss_fn}")
    print(f"设备: {args.device}")

    device = torch.device(args.device)

    # Load data
    print("\n加载数据...")
    graph_dir = Path(args.graph_dir)
    available_graphs = [f.stem for f in graph_dir.glob("*.pt")]
    complex_ids = available_graphs[:args.num_samples]

    dataset = LigandEmbeddingDataset(
        complex_ids,
        args.graph_dir,
        args.ligand_embeddings,
        validate_format=True
    )

    if len(dataset) == 0:
        print("❌ 没有找到有效数据")
        return

    loader = DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False)
    batch = next(iter(loader))

    print(f"加载了 {len(dataset)} 个样本")

    # Check data statistics
    data_stats = check_data_statistics(batch, device)

    # Initialize model
    print("\n初始化模型...")
    if args.model_type == "v3" and _has_v3_model:
        model = RNAPocketEncoderV3(
            input_dim=3,
            feature_hidden_dim=64,
            hidden_irreps="32x0e + 16x1o + 8x2e",
            output_dim=512,
            num_layers=3,
            use_multi_hop=args.use_multi_hop,
            use_nonbonded=args.use_nonbonded,
            use_geometric_mp=args.use_v3_features,
            use_enhanced_invariants=args.use_v3_features,
            num_attention_heads=4,
            pooling_type='multihead_attention' if args.use_v3_features else 'attention',
        )
    elif _has_v2_model:
        model = RNAPocketEncoderV2(
            input_dim=3,
            feature_hidden_dim=64,
            hidden_irreps="32x0e + 16x1o + 8x2e",
            output_dim=512,
            num_layers=3,
            use_multi_hop=args.use_multi_hop,
            use_nonbonded=args.use_nonbonded,
        )
    else:
        print("❌ 没有可用的模型")
        return

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {num_params:,}")

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Initialize physics loss
    physics_loss_fn = None
    if args.use_physics_loss and _has_physics_loss:
        physics_loss_fn = PhysicsConstraintLoss(
            use_bond=True,
            use_angle=True,
            use_dihedral=True,
            use_nonbonded=False
        ).to(device)
        print("物理损失: 已启用")

    # Debug forward-backward pass
    result = debug_forward_backward(
        model, batch, batch.y, optimizer, args.loss_fn, device,
        use_amp=args.use_amp,
        physics_loss_fn=physics_loss_fn,
        physics_weight=args.physics_weight,
        cosine_weight=args.cosine_weight,
        mse_weight=args.mse_weight,
        temperature=args.temperature,
    )

    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'model': {
            'type': args.model_type,
            'num_params': num_params,
        },
        'data_stats': data_stats,
        'debug_result': result,
    }

    # Print summary
    print("\n" + "="*70)
    print("调试总结")
    print("="*70)

    if result['success']:
        print(f"✓ 前向-反向传播成功")
        print(f"  损失值: {result['loss']:.6f}")
        if result['activation_issues'] > 0:
            print(f"  ⚠️  激活异常: {result['activation_issues']} 层")
        if result['gradient_issues'] > 0:
            print(f"  ⚠️  梯度异常: {result['gradient_issues']} 个参数")

        if result['activation_issues'] == 0 and result['gradient_issues'] == 0:
            print("\n  🎉 所有检查通过！")
    else:
        print(f"❌ 调试失败: {result.get('error', 'Unknown error')}")

    # Save report
    if args.save_report:
        save_report(report)

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
