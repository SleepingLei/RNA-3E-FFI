#!/usr/bin/env python3
"""
梯度稳定性调试脚本

用于诊断 V3 模型训练中的数值不稳定问题：
- 检查数据加载和特征范围
- 监控前向传播中每层的输出
- 追踪梯度流和梯度范数
- 检测 NaN/Inf
- 生成详细的诊断报告
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import json
from datetime import datetime

# 导入模型
try:
    from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3
    from models.improved_components import PhysicsConstraintLoss
    _has_v3 = True
except ImportError:
    _has_v3 = False
    print("Warning: V3 components not available")


class GradientMonitor:
    """监控梯度统计"""

    def __init__(self):
        self.grad_stats = {}

    def register_hooks(self, model):
        """注册梯度钩子"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self._grad_hook(grad, name))

    def _grad_hook(self, grad, name):
        """梯度钩子回调"""
        if grad is not None:
            self.grad_stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'norm': grad.norm().item(),
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item(),
            }
        return grad

    def get_stats(self):
        """获取统计信息"""
        return self.grad_stats.copy()

    def clear(self):
        """清空统计"""
        self.grad_stats = {}


class ActivationMonitor:
    """监控激活值统计"""

    def __init__(self):
        self.activation_stats = {}
        self.hooks = []

    def register_hooks(self, model):
        """注册前向钩子"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: self._forward_hook(out, name)
                )
                self.hooks.append(hook)

    def _forward_hook(self, output, name):
        """前向钩子回调"""
        if isinstance(output, torch.Tensor):
            self.activation_stats[name] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'has_nan': torch.isnan(output).any().item(),
                'has_inf': torch.isinf(output).any().item(),
            }

    def get_stats(self):
        """获取统计信息"""
        return self.activation_stats.copy()

    def clear(self):
        """清空统计"""
        self.activation_stats = {}

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def check_data_normalization(data):
    """检查数据归一化"""
    print("\n" + "="*60)
    print("数据归一化检查")
    print("="*60)

    stats = {}

    # 节点特征
    if hasattr(data, 'x'):
        stats['node_features'] = {
            'shape': list(data.x.shape),
            'mean': data.x.mean().item(),
            'std': data.x.std().item(),
            'min': data.x.min().item(),
            'max': data.max().item(),
            'has_nan': torch.isnan(data.x).any().item(),
            'has_inf': torch.isinf(data.x).any().item(),
        }
        print(f"\n节点特征 (x):")
        print(f"  Shape: {data.x.shape}")
        print(f"  Range: [{data.x.min().item():.4f}, {data.x.max().item():.4f}]")
        print(f"  Mean: {data.x.mean().item():.4f}, Std: {data.x.std().item():.4f}")

    # 位置
    if hasattr(data, 'pos'):
        stats['positions'] = {
            'shape': list(data.pos.shape),
            'mean': data.pos.mean().item(),
            'std': data.pos.std().item(),
            'min': data.pos.min().item(),
            'max': data.pos.max().item(),
        }
        print(f"\n位置 (pos):")
        print(f"  Shape: {data.pos.shape}")
        print(f"  Range: [{data.pos.min().item():.4f}, {data.pos.max().item():.4f}]")
        print(f"  Mean: {data.pos.mean().item():.4f}, Std: {data.pos.std().item():.4f}")

    # 键参数
    if hasattr(data, 'edge_attr') and data.edge_attr.shape[0] > 0:
        stats['edge_attr'] = {
            'shape': list(data.edge_attr.shape),
            'col0_range': [data.edge_attr[:, 0].min().item(), data.edge_attr[:, 0].max().item()],
            'col1_range': [data.edge_attr[:, 1].min().item(), data.edge_attr[:, 1].max().item()],
        }
        print(f"\n键参数 (edge_attr):")
        print(f"  Shape: {data.edge_attr.shape}")
        print(f"  Col 0 (req_norm): [{data.edge_attr[:, 0].min().item():.4f}, {data.edge_attr[:, 0].max().item():.4f}]")
        print(f"  Col 1 (k_norm): [{data.edge_attr[:, 1].min().item():.4f}, {data.edge_attr[:, 1].max().item():.4f}]")

    # 角度参数
    if hasattr(data, 'triple_attr') and data.triple_attr.shape[0] > 0:
        stats['triple_attr'] = {
            'shape': list(data.triple_attr.shape),
            'col0_range': [data.triple_attr[:, 0].min().item(), data.triple_attr[:, 0].max().item()],
            'col1_range': [data.triple_attr[:, 1].min().item(), data.triple_attr[:, 1].max().item()],
        }
        print(f"\n角度参数 (triple_attr):")
        print(f"  Shape: {data.triple_attr.shape}")
        print(f"  Col 0 (theta_eq_norm): [{data.triple_attr[:, 0].min().item():.4f}, {data.triple_attr[:, 0].max().item():.4f}]")
        print(f"  Col 1 (k_norm): [{data.triple_attr[:, 1].min().item():.4f}, {data.triple_attr[:, 1].max().item():.4f}]")

    # 二面角参数
    if hasattr(data, 'quadra_attr') and data.quadra_attr.shape[0] > 0:
        stats['quadra_attr'] = {
            'shape': list(data.quadra_attr.shape),
            'col0_range': [data.quadra_attr[:, 0].min().item(), data.quadra_attr[:, 0].max().item()],
            'col1_range': [data.quadra_attr[:, 1].min().item(), data.quadra_attr[:, 1].max().item()],
            'col2_range': [data.quadra_attr[:, 2].min().item(), data.quadra_attr[:, 2].max().item()],
        }
        print(f"\n二面角参数 (quadra_attr):")
        print(f"  Shape: {data.quadra_attr.shape}")
        print(f"  Col 0 (phi_k_norm): [{data.quadra_attr[:, 0].min().item():.4f}, {data.quadra_attr[:, 0].max().item():.4f}]")
        print(f"  Col 1 (per_norm): [{data.quadra_attr[:, 1].min().item():.4f}, {data.quadra_attr[:, 1].max().item():.4f}]")
        print(f"  Col 2 (phase_norm): [{data.quadra_attr[:, 2].min().item():.4f}, {data.quadra_attr[:, 2].max().item():.4f}]")

    # 非键参数
    if hasattr(data, 'nonbonded_edge_attr') and data.nonbonded_edge_attr.shape[0] > 0:
        stats['nonbonded_edge_attr'] = {
            'shape': list(data.nonbonded_edge_attr.shape),
            'col0_range': [data.nonbonded_edge_attr[:, 0].min().item(), data.nonbonded_edge_attr[:, 0].max().item()],
            'col1_range': [data.nonbonded_edge_attr[:, 1].min().item(), data.nonbonded_edge_attr[:, 1].max().item()],
            'col2_range': [data.nonbonded_edge_attr[:, 2].min().item(), data.nonbonded_edge_attr[:, 2].max().item()],
        }
        print(f"\n非键参数 (nonbonded_edge_attr):")
        print(f"  Shape: {data.nonbonded_edge_attr.shape}")
        print(f"  Col 0 (lj_a_log): [{data.nonbonded_edge_attr[:, 0].min().item():.4f}, {data.nonbonded_edge_attr[:, 0].max().item():.4f}]")
        print(f"  Col 1 (lj_b_log): [{data.nonbonded_edge_attr[:, 1].min().item():.4f}, {data.nonbonded_edge_attr[:, 1].max().item():.4f}]")
        print(f"  Col 2 (dist): [{data.nonbonded_edge_attr[:, 2].min().item():.4f}, {data.nonbonded_edge_attr[:, 2].max().item():.4f}]")

    return stats


def debug_forward_pass(model, batch, device):
    """调试前向传播"""
    print("\n" + "="*60)
    print("前向传播调试")
    print("="*60)

    # 移动数据到设备
    batch = batch.to(device)

    # 激活监控
    act_monitor = ActivationMonitor()
    act_monitor.register_hooks(model)

    # 前向传播
    try:
        output = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr,
                      batch.triple_index, batch.triple_attr,
                      batch.quadra_index, batch.quadra_attr,
                      batch.nonbonded_edge_index, batch.nonbonded_edge_attr,
                      batch.batch)

        print(f"\n模型输出:")
        print(f"  Shape: {output.shape}")
        print(f"  Range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
        print(f"  Has NaN: {torch.isnan(output).any().item()}")
        print(f"  Has Inf: {torch.isinf(output).any().item()}")

        # 激活统计
        act_stats = act_monitor.get_stats()
        print(f"\n激活值统计 (共 {len(act_stats)} 层):")

        # 找出异常层
        problematic_layers = []
        for name, stats in act_stats.items():
            if stats['has_nan'] or stats['has_inf']:
                problematic_layers.append(name)
                print(f"\n⚠️  异常层: {name}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Has NaN: {stats['has_nan']}, Has Inf: {stats['has_inf']}")
            elif abs(stats['max']) > 100 or abs(stats['min']) > 100:
                print(f"\n⚠️  数值较大层: {name}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

        if not problematic_layers:
            print("  ✓ 所有层的激活值正常")

        act_monitor.remove_hooks()
        return output, act_stats, problematic_layers

    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        act_monitor.remove_hooks()
        return None, None, None


def debug_backward_pass(model, output, target, loss_fn='mse'):
    """调试反向传播"""
    print("\n" + "="*60)
    print("反向传播调试")
    print("="*60)

    # 梯度监控
    grad_monitor = GradientMonitor()
    grad_monitor.register_hooks(model)

    # 计算损失
    try:
        if loss_fn == 'mse':
            loss = torch.nn.functional.mse_loss(output, target)
        elif loss_fn == 'cosine':
            loss = 1 - torch.nn.functional.cosine_similarity(output, target, dim=-1).mean()
        else:
            loss = torch.nn.functional.mse_loss(output, target)

        print(f"\nLoss: {loss.item():.6f}")

        # 反向传播
        loss.backward()

        # 梯度统计
        grad_stats = grad_monitor.get_stats()
        print(f"\n梯度统计 (共 {len(grad_stats)} 个参数):")

        # 找出异常梯度
        problematic_grads = []
        large_grads = []

        for name, stats in grad_stats.items():
            if stats['has_nan'] or stats['has_inf']:
                problematic_grads.append(name)
                print(f"\n❌ 异常梯度: {name}")
                print(f"  Norm: {stats['norm']:.4f}")
                print(f"  Has NaN: {stats['has_nan']}, Has Inf: {stats['has_inf']}")
            elif stats['norm'] > 10.0:
                large_grads.append(name)
                print(f"\n⚠️  大梯度: {name}")
                print(f"  Norm: {stats['norm']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        if not problematic_grads and not large_grads:
            print("  ✓ 所有梯度正常")

        # 总体梯度统计
        all_norms = [s['norm'] for s in grad_stats.values()]
        print(f"\n总体梯度范数统计:")
        print(f"  Mean: {np.mean(all_norms):.4f}")
        print(f"  Std: {np.std(all_norms):.4f}")
        print(f"  Max: {np.max(all_norms):.4f}")
        print(f"  Min: {np.min(all_norms):.4f}")

        return loss, grad_stats, problematic_grads, large_grads

    except Exception as e:
        print(f"\n❌ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def debug_physics_loss(batch, device):
    """调试物理约束损失"""
    print("\n" + "="*60)
    print("物理约束损失调试")
    print("="*60)

    if not _has_v3:
        print("V3 组件不可用，跳过物理损失测试")
        return None

    batch = batch.to(device)

    try:
        physics_loss_fn = PhysicsConstraintLoss(
            use_bond=True,
            use_angle=True,
            use_dihedral=True,
            use_nonbonded=False
        ).to(device)

        physics_loss, physics_dict = physics_loss_fn(batch)

        print(f"\n物理损失:")
        print(f"  Total: {physics_loss.item():.6f}")
        print(f"  Has NaN: {torch.isnan(physics_loss).any().item()}")
        print(f"  Has Inf: {torch.isinf(physics_loss).any().item()}")

        print(f"\n各项能量:")
        for key, val in physics_dict.items():
            print(f"  {key}: {val:.6f}")

        return physics_loss, physics_dict

    except Exception as e:
        print(f"\n❌ 物理损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_debug_report(report, output_dir="debug_logs"):
    """保存调试报告"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_report_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n调试报告已保存到: {filepath}")
    return filepath


def main():
    import argparse
    parser = argparse.ArgumentParser(description="梯度稳定性调试脚本")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径 (HDF5 文件)")
    parser.add_argument("--model_type", type=str, default="v3",
                        choices=["v2", "v3"],
                        help="模型类型")
    parser.add_argument("--use_v3_features", action="store_true",
                        help="使用 V3 特征（几何 MP + 增强不变量）")
    parser.add_argument("--use_physics_loss", action="store_true",
                        help="测试物理约束损失")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")
    parser.add_argument("--save_report", action="store_true",
                        help="保存调试报告")

    args = parser.parse_args()

    print("="*60)
    print("梯度稳定性调试脚本")
    print("="*60)
    print(f"数据路径: {args.data_path}")
    print(f"模型类型: {args.model_type}")
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")

    device = torch.device(args.device)

    # 加载数据
    print("\n加载数据...")
    from torch_geometric.data import InMemoryDataset
    import h5py

    # 简单加载一个 batch
    class SimpleDataset(InMemoryDataset):
        def __init__(self, h5_path):
            super().__init__(None, None, None)
            with h5py.File(h5_path, 'r') as f:
                keys = list(f.keys())[:args.batch_size]  # 只加载前几个样本

                data_list = []
                for key in keys:
                    grp = f[key]
                    data = Data(
                        x=torch.tensor(grp['x'][:], dtype=torch.float),
                        pos=torch.tensor(grp['pos'][:], dtype=torch.float),
                        edge_index=torch.tensor(grp['edge_index'][:], dtype=torch.long),
                        edge_attr=torch.tensor(grp['edge_attr'][:], dtype=torch.float),
                        triple_index=torch.tensor(grp['triple_index'][:], dtype=torch.long),
                        triple_attr=torch.tensor(grp['triple_attr'][:], dtype=torch.float),
                        quadra_index=torch.tensor(grp['quadra_index'][:], dtype=torch.long),
                        quadra_attr=torch.tensor(grp['quadra_attr'][:], dtype=torch.float),
                        nonbonded_edge_index=torch.tensor(grp['nonbonded_edge_index'][:], dtype=torch.long),
                        nonbonded_edge_attr=torch.tensor(grp['nonbonded_edge_attr'][:], dtype=torch.float),
                        y=torch.tensor(grp['y'][:], dtype=torch.float),
                    )
                    data_list.append(data)

                self.data, self.slices = self.collate(data_list)

    dataset = SimpleDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))

    print(f"加载了 {len(dataset)} 个样本")

    # 检查数据归一化
    data_stats = check_data_normalization(batch)

    # 初始化模型
    print("\n初始化模型...")
    if args.model_type == "v3" and _has_v3:
        model = RNAPocketEncoderV3(
            input_dim=3,
            feature_hidden_dim=64,
            hidden_irreps="32x0e + 16x1o + 8x2e",
            output_dim=512,
            num_layers=3,
            num_radial_basis=8,
            use_multi_hop=True,
            use_nonbonded=True,
            use_gate=True,
            use_layer_norm=True,
            pooling_type='multihead_attention',
            dropout=0.1,
            use_geometric_mp=args.use_v3_features,
            use_enhanced_invariants=args.use_v3_features,
            num_attention_heads=4,
        )
    else:
        from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
        model = RNAPocketEncoderV2(
            input_dim=3,
            feature_hidden_dim=64,
            hidden_irreps="32x0e + 16x1o + 8x2e",
            output_dim=512,
            num_layers=3,
            num_radial_basis=8,
            use_multi_hop=True,
            use_nonbonded=True,
            use_gate=True,
            use_layer_norm=True,
            pooling_type='attention',
            dropout=0.1,
        )

    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 调试前向传播
    output, act_stats, problematic_layers = debug_forward_pass(model, batch, device)

    if output is None:
        print("\n❌ 前向传播失败，调试终止")
        return

    # 调试反向传播
    target = batch.y.to(device)
    loss, grad_stats, problematic_grads, large_grads = debug_backward_pass(
        model, output, target, loss_fn='mse'
    )

    # 调试物理损失（如果启用）
    physics_loss_result = None
    if args.use_physics_loss:
        physics_loss_result = debug_physics_loss(batch, device)

    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model_type': args.model_type,
            'use_v3_features': args.use_v3_features,
            'use_physics_loss': args.use_physics_loss,
            'batch_size': args.batch_size,
            'device': args.device,
        },
        'data_stats': data_stats,
        'forward_pass': {
            'success': output is not None,
            'problematic_layers': problematic_layers,
            'activation_stats': act_stats,
        },
        'backward_pass': {
            'success': loss is not None,
            'loss_value': loss.item() if loss is not None else None,
            'problematic_grads': problematic_grads,
            'large_grads': large_grads,
            'gradient_stats': grad_stats,
        },
    }

    if physics_loss_result is not None:
        physics_loss, physics_dict = physics_loss_result
        report['physics_loss'] = {
            'success': physics_loss is not None,
            'value': physics_loss.item() if physics_loss is not None else None,
            'components': physics_dict,
        }

    # 总结
    print("\n" + "="*60)
    print("调试总结")
    print("="*60)

    if problematic_layers:
        print(f"\n❌ 发现 {len(problematic_layers)} 个异常激活层:")
        for layer in problematic_layers[:5]:  # 只显示前5个
            print(f"  - {layer}")
    else:
        print("\n✓ 所有激活层正常")

    if problematic_grads:
        print(f"\n❌ 发现 {len(problematic_grads)} 个异常梯度:")
        for name in problematic_grads[:5]:
            print(f"  - {name}")
    elif large_grads:
        print(f"\n⚠️  发现 {len(large_grads)} 个大梯度 (norm > 10):")
        for name in large_grads[:5]:
            print(f"  - {name}")
    else:
        print("\n✓ 所有梯度正常")

    # 保存报告
    if args.save_report:
        save_debug_report(report)

    print("\n" + "="*60)
    print("调试完成")
    print("="*60)


if __name__ == "__main__":
    main()
