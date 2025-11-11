#!/usr/bin/env python3
"""
梯度爆炸诊断脚本

用于深度分析训练过程中的梯度爆炸问题，监控并保存：
1. 每层的梯度范数
2. 激活值统计
3. 权重变化
4. Loss变化
5. 数据统计
6. 学习率变化

使用方法:
    python scripts/diagnose_gradient_explosion.py \
        --model_version v3 \
        --data_dir data/processed_pockets \
        --ligand_embeddings data/ligand_embeddings.h5 \
        --split_dir data/splits \
        --epochs 10 \
        --batch_size 32 \
        --output_dir gradient_diagnosis
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import csv
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch_geometric.loader import DataLoader
from torch.cuda.amp import autocast, GradScaler
import h5py

# Import models
try:
    from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3
    _has_v3 = True
except ImportError:
    _has_v3 = False

try:
    from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
    _has_v2 = True
except ImportError:
    _has_v2 = False


# ============================================================================
# 数据集类（从 04_train_model.py 复制）
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

        # Create mapping
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
        target_embedding = self.ligand_embeddings[embedding_key]

        data.target_embedding = target_embedding
        data.complex_id = complex_id

        return data


# ============================================================================
# 监控类
# ============================================================================

class GradientMonitor:
    """监控每层梯度的详细统计"""

    def __init__(self):
        self.gradient_stats = defaultdict(list)
        self.gradient_history = defaultdict(list)

    def update(self, model, step):
        """记录当前step的梯度统计"""
        stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                grad_max = grad.abs().max().item()
                grad_min = grad.abs().min().item()

                # 检测异常
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()

                stat = {
                    'step': step,
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'max': grad_max,
                    'min': grad_min,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                }

                stats[name] = stat
                self.gradient_history[name].append(stat)

        self.gradient_stats[step] = stats
        return stats

    def get_suspicious_layers(self, threshold=10.0):
        """找出梯度异常的层"""
        suspicious = []

        if not self.gradient_stats:
            return suspicious

        latest_step = max(self.gradient_stats.keys())
        latest_stats = self.gradient_stats[latest_step]

        for name, stat in latest_stats.items():
            if stat['has_nan'] or stat['has_inf']:
                suspicious.append({
                    'layer': name,
                    'issue': 'NaN/Inf',
                    'norm': stat['norm']
                })
            elif stat['norm'] > threshold:
                suspicious.append({
                    'layer': name,
                    'issue': 'Large gradient',
                    'norm': stat['norm']
                })

        return suspicious


class ActivationMonitor:
    """监控每层激活值的统计"""

    def __init__(self):
        self.activation_stats = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model):
        """为模型的每一层注册hook"""

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._record_activation(name, output)
                elif isinstance(output, tuple):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            self._record_activation(f"{name}_out{i}", out)
            return hook

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

    def _record_activation(self, name, tensor):
        """记录激活值统计"""
        with torch.no_grad():
            stat = {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'max': tensor.max().item(),
                'min': tensor.min().item(),
                'has_nan': torch.isnan(tensor).any().item(),
                'has_inf': torch.isinf(tensor).any().item(),
            }
            self.activation_stats[name].append(stat)

    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_suspicious_layers(self):
        """找出激活值异常的层"""
        suspicious = []

        for name, stats in self.activation_stats.items():
            if not stats:
                continue

            latest = stats[-1]
            if latest['has_nan'] or latest['has_inf']:
                suspicious.append({
                    'layer': name,
                    'issue': 'NaN/Inf activation',
                    'mean': latest['mean'],
                    'std': latest['std']
                })
            elif abs(latest['mean']) > 100 or latest['std'] > 100:
                suspicious.append({
                    'layer': name,
                    'issue': 'Large activation',
                    'mean': latest['mean'],
                    'std': latest['std']
                })

        return suspicious


class WeightMonitor:
    """监控模型权重的变化"""

    def __init__(self):
        self.weight_history = defaultdict(list)
        self.initial_weights = {}

    def record_initial_weights(self, model):
        """记录初始权重"""
        for name, param in model.named_parameters():
            self.initial_weights[name] = {
                'norm': param.data.norm().item(),
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
            }

    def update(self, model, step):
        """记录当前step的权重统计"""
        for name, param in model.named_parameters():
            weight = param.data
            stat = {
                'step': step,
                'norm': weight.norm().item(),
                'mean': weight.mean().item(),
                'std': weight.std().item(),
                'max': weight.abs().max().item(),
                'min': weight.abs().min().item(),
            }

            # 计算与初始值的变化
            if name in self.initial_weights:
                initial = self.initial_weights[name]
                stat['norm_change'] = stat['norm'] - initial['norm']
                stat['norm_change_ratio'] = stat['norm'] / (initial['norm'] + 1e-8)

            self.weight_history[name].append(stat)

    def get_weight_changes(self):
        """获取权重变化最大的层"""
        changes = []

        for name, history in self.weight_history.items():
            if len(history) < 2:
                continue

            latest = history[-1]
            if 'norm_change_ratio' in latest:
                changes.append({
                    'layer': name,
                    'norm_change_ratio': latest['norm_change_ratio'],
                    'current_norm': latest['norm'],
                })

        changes.sort(key=lambda x: abs(x['norm_change_ratio'] - 1.0), reverse=True)
        return changes[:10]


class InputFeatureMonitor:
    """监控输入特征的统计和变化"""

    def __init__(self):
        self.feature_history = defaultdict(list)

        # 预期范围（用于检测异常）
        self.expected_ranges = {
            'x_col0': (-2, 2),       # charge (normalized)
            'x_col1': (0, 100),      # atomic_num
            'x_col2': (0, 300),      # mass
            'pos': (-100, 100),      # positions
            'edge_attr_col0': (0, 2),      # req/2.0
            'edge_attr_col1': (0, 1),      # k/500.0
            'triple_attr_col0': (0, 2),    # theta_eq/180.0
            'triple_attr_col1': (0, 1),    # k/200.0
            'quadra_attr_col0': (0, 1),    # phi_k/20.0
            'quadra_attr_col1': (0, 1),    # per/6.0
            'quadra_attr_col2': (0, 1),    # phase/(2π)
        }

    def update(self, batch, step):
        """记录当前batch的输入特征统计"""
        stats = {'step': step}

        with torch.no_grad():
            # 1. 节点特征 (x) - [charge, atomic_num, mass]
            if hasattr(batch, 'x') and batch.x is not None:
                x = batch.x
                for col in range(x.shape[1]):
                    col_data = x[:, col]
                    stats[f'x_col{col}_mean'] = col_data.mean().item()
                    stats[f'x_col{col}_std'] = col_data.std().item()
                    stats[f'x_col{col}_max'] = col_data.max().item()
                    stats[f'x_col{col}_min'] = col_data.min().item()
                    stats[f'x_col{col}_has_nan'] = torch.isnan(col_data).any().item()
                    stats[f'x_col{col}_has_inf'] = torch.isinf(col_data).any().item()

            # 2. 位置 (pos)
            if hasattr(batch, 'pos') and batch.pos is not None:
                pos = batch.pos
                stats['pos_mean'] = pos.mean().item()
                stats['pos_std'] = pos.std().item()
                stats['pos_max'] = pos.max().item()
                stats['pos_min'] = pos.min().item()
                stats['pos_has_nan'] = torch.isnan(pos).any().item()
                stats['pos_has_inf'] = torch.isinf(pos).any().item()

            # 3. 边特征 (edge_attr)
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                edge_attr = batch.edge_attr
                for col in range(edge_attr.shape[1]):
                    col_data = edge_attr[:, col]
                    stats[f'edge_attr_col{col}_mean'] = col_data.mean().item()
                    stats[f'edge_attr_col{col}_std'] = col_data.std().item()
                    stats[f'edge_attr_col{col}_max'] = col_data.max().item()
                    stats[f'edge_attr_col{col}_min'] = col_data.min().item()
                    stats[f'edge_attr_col{col}_has_nan'] = torch.isnan(col_data).any().item()

            # 4. 角度特征 (triple_attr)
            if hasattr(batch, 'triple_attr') and batch.triple_attr is not None:
                triple_attr = batch.triple_attr
                for col in range(min(triple_attr.shape[1], 2)):  # theta_eq, k
                    col_data = triple_attr[:, col]
                    stats[f'triple_attr_col{col}_mean'] = col_data.mean().item()
                    stats[f'triple_attr_col{col}_std'] = col_data.std().item()
                    stats[f'triple_attr_col{col}_max'] = col_data.max().item()
                    stats[f'triple_attr_col{col}_min'] = col_data.min().item()

            # 5. 二面角特征 (quadra_attr)
            if hasattr(batch, 'quadra_attr') and batch.quadra_attr is not None:
                quadra_attr = batch.quadra_attr
                for col in range(min(quadra_attr.shape[1], 3)):  # phi_k, per, phase
                    col_data = quadra_attr[:, col]
                    stats[f'quadra_attr_col{col}_mean'] = col_data.mean().item()
                    stats[f'quadra_attr_col{col}_std'] = col_data.std().item()
                    stats[f'quadra_attr_col{col}_max'] = col_data.max().item()
                    stats[f'quadra_attr_col{col}_min'] = col_data.min().item()

        self.feature_history['all'].append(stats)
        return stats

    def get_suspicious_features(self):
        """检测异常的输入特征"""
        suspicious = []

        if not self.feature_history['all']:
            return suspicious

        latest = self.feature_history['all'][-1]

        # 检查 NaN/Inf
        for key, value in latest.items():
            if '_has_nan' in key and value:
                feature_name = key.replace('_has_nan', '')
                suspicious.append({
                    'feature': feature_name,
                    'issue': 'NaN detected',
                    'step': latest['step']
                })
            elif '_has_inf' in key and value:
                feature_name = key.replace('_has_inf', '')
                suspicious.append({
                    'feature': feature_name,
                    'issue': 'Inf detected',
                    'step': latest['step']
                })

        # 检查超出预期范围
        for feature_base, (min_val, max_val) in self.expected_ranges.items():
            max_key = f'{feature_base}_max'
            min_key = f'{feature_base}_min'

            if max_key in latest and latest[max_key] > max_val:
                suspicious.append({
                    'feature': feature_base,
                    'issue': f'Max value {latest[max_key]:.4f} exceeds expected {max_val}',
                    'step': latest['step']
                })

            if min_key in latest and latest[min_key] < min_val:
                suspicious.append({
                    'feature': feature_base,
                    'issue': f'Min value {latest[min_key]:.4f} below expected {min_val}',
                    'step': latest['step']
                })

        return suspicious

    def detect_sudden_changes(self, window=10, threshold=2.0):
        """检测输入特征的突然变化"""
        if len(self.feature_history['all']) < window + 1:
            return []

        sudden_changes = []
        recent_history = self.feature_history['all'][-window-1:]

        # 对每个均值特征检测突变
        for key in recent_history[0].keys():
            if '_mean' in key and 'step' not in key:
                values = [h[key] for h in recent_history if key in h]

                if len(values) < window:
                    continue

                # 计算前window步的均值和标准差
                prev_mean = np.mean(values[:-1])
                prev_std = np.std(values[:-1]) + 1e-6
                latest_value = values[-1]

                # 检测是否突变
                z_score = abs(latest_value - prev_mean) / prev_std

                if z_score > threshold:
                    sudden_changes.append({
                        'feature': key,
                        'z_score': z_score,
                        'prev_mean': prev_mean,
                        'latest_value': latest_value,
                        'change': latest_value - prev_mean
                    })

        return sudden_changes


class TrainingMonitor:
    """综合监控器，整合所有监控功能"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_monitor = GradientMonitor()
        self.activation_monitor = ActivationMonitor()
        self.weight_monitor = WeightMonitor()
        self.input_feature_monitor = InputFeatureMonitor()

        self.loss_history = []
        self.lr_history = []
        self.data_stats_history = []

        self.step_counter = 0
        self.epoch_counter = 0

        # CSV文件
        self.csv_file = self.output_dir / 'gradient_stats.csv'
        self.features_csv_file = self.output_dir / 'input_features.csv'
        self.init_csv()

    def init_csv(self):
        """初始化CSV文件"""
        # 梯度统计CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'batch_idx', 'loss', 'lr',
                'total_grad_norm', 'max_grad_norm', 'min_grad_norm',
                'num_nan_grads', 'num_inf_grads',
                'suspicious_layers', 'data_mean', 'data_std'
            ])

        # 输入特征CSV
        with open(self.features_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'batch_idx',
                'x_col0_mean', 'x_col0_std', 'x_col0_max', 'x_col0_min',
                'x_col1_mean', 'x_col1_std', 'x_col1_max', 'x_col1_min',
                'x_col2_mean', 'x_col2_std', 'x_col2_max', 'x_col2_min',
                'pos_mean', 'pos_std', 'pos_max', 'pos_min',
                'edge_attr_col0_mean', 'edge_attr_col0_std',
                'edge_attr_col1_mean', 'edge_attr_col1_std',
                'triple_attr_col0_mean', 'triple_attr_col0_std',
                'triple_attr_col1_mean', 'triple_attr_col1_std',
                'suspicious_features'
            ])

    def record_step(self, model, loss, optimizer, batch, batch_idx):
        """记录一个训练step的所有信息"""
        self.step_counter += 1

        # 1. 梯度统计
        grad_stats = self.gradient_monitor.update(model, self.step_counter)

        # 计算总体梯度统计
        grad_norms = [s['norm'] for s in grad_stats.values()]
        total_grad_norm = sum(grad_norms)
        max_grad_norm = max(grad_norms) if grad_norms else 0
        min_grad_norm = min(grad_norms) if grad_norms else 0
        num_nan = sum(1 for s in grad_stats.values() if s['has_nan'])
        num_inf = sum(1 for s in grad_stats.values() if s['has_inf'])

        # 2. Loss
        self.loss_history.append({
            'step': self.step_counter,
            'epoch': self.epoch_counter,
            'batch_idx': batch_idx,
            'loss': loss.item()
        })

        # 3. 学习率
        lr = optimizer.param_groups[0]['lr']
        self.lr_history.append({
            'step': self.step_counter,
            'lr': lr
        })

        # 4. 数据统计
        with torch.no_grad():
            data_stat = {
                'step': self.step_counter,
                'x_mean': batch.x.mean().item(),
                'x_std': batch.x.std().item(),
                'pos_mean': batch.pos.mean().item(),
                'pos_std': batch.pos.std().item(),
            }
            self.data_stats_history.append(data_stat)

        # 4.5 输入特征统计
        feature_stats = self.input_feature_monitor.update(batch, self.step_counter)

        # 5. 权重统计（每10步记录一次）
        if self.step_counter % 10 == 0:
            self.weight_monitor.update(model, self.step_counter)

        # 6. 检测异常
        suspicious_grad = self.gradient_monitor.get_suspicious_layers(threshold=10.0)
        suspicious_act = self.activation_monitor.get_suspicious_layers()
        suspicious_feat = self.input_feature_monitor.get_suspicious_features()

        # 检测输入特征的突然变化（每10步检测一次）
        sudden_changes = []
        if self.step_counter % 10 == 0 and self.step_counter > 20:
            sudden_changes = self.input_feature_monitor.detect_sudden_changes()

        # 7. 写入CSV
        suspicious_str = ';'.join([f"{s['layer']}:{s['issue']}" for s in suspicious_grad[:3]])

        # 写入梯度统计CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.step_counter, self.epoch_counter, batch_idx, loss.item(), lr,
                total_grad_norm, max_grad_norm, min_grad_norm,
                num_nan, num_inf,
                suspicious_str, data_stat['x_mean'], data_stat['x_std']
            ])

        # 写入输入特征CSV
        suspicious_feat_str = ';'.join([f"{s['feature']}:{s['issue']}" for s in suspicious_feat[:3]])
        with open(self.features_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.step_counter, self.epoch_counter, batch_idx,
                feature_stats.get('x_col0_mean', 0), feature_stats.get('x_col0_std', 0),
                feature_stats.get('x_col0_max', 0), feature_stats.get('x_col0_min', 0),
                feature_stats.get('x_col1_mean', 0), feature_stats.get('x_col1_std', 0),
                feature_stats.get('x_col1_max', 0), feature_stats.get('x_col1_min', 0),
                feature_stats.get('x_col2_mean', 0), feature_stats.get('x_col2_std', 0),
                feature_stats.get('x_col2_max', 0), feature_stats.get('x_col2_min', 0),
                feature_stats.get('pos_mean', 0), feature_stats.get('pos_std', 0),
                feature_stats.get('pos_max', 0), feature_stats.get('pos_min', 0),
                feature_stats.get('edge_attr_col0_mean', 0), feature_stats.get('edge_attr_col0_std', 0),
                feature_stats.get('edge_attr_col1_mean', 0), feature_stats.get('edge_attr_col1_std', 0),
                feature_stats.get('triple_attr_col0_mean', 0), feature_stats.get('triple_attr_col0_std', 0),
                feature_stats.get('triple_attr_col1_mean', 0), feature_stats.get('triple_attr_col1_std', 0),
                suspicious_feat_str
            ])

        # 8. 打印警告
        if suspicious_grad or suspicious_act or suspicious_feat or sudden_changes:
            print(f"\n⚠️  Step {self.step_counter} - 检测到异常!")
            if suspicious_grad:
                print(f"  梯度异常层: {suspicious_grad[:3]}")
            if suspicious_act:
                print(f"  激活值异常层: {suspicious_act[:3]}")
            if suspicious_feat:
                print(f"  输入特征异常: {suspicious_feat[:3]}")
            if sudden_changes:
                print(f"  特征突变: {sudden_changes[:3]}")
            print(f"  Total grad norm: {total_grad_norm:.4f}")
            print(f"  Max grad norm: {max_grad_norm:.4f}")

        return {
            'total_grad_norm': total_grad_norm,
            'suspicious_grad': suspicious_grad,
            'suspicious_act': suspicious_act,
            'suspicious_feat': suspicious_feat,
            'sudden_changes': sudden_changes
        }

    def start_epoch(self, epoch):
        """开始新的epoch"""
        self.epoch_counter = epoch

    def save_report(self):
        """生成并保存诊断报告"""
        print("\n生成诊断报告...")

        # 1. JSON报告
        report = {
            'summary': {
                'total_steps': self.step_counter,
                'total_epochs': self.epoch_counter,
                'num_gradient_explosions': len([
                    s for s in self.loss_history
                    if s['loss'] > 1000 or np.isnan(s['loss']) or np.isinf(s['loss'])
                ]),
            },
            'gradient_stats': {
                name: history[-10:] if len(history) > 10 else history
                for name, history in list(self.gradient_monitor.gradient_history.items())[:10]
            },
            'weight_changes': self.weight_monitor.get_weight_changes(),
            'loss_history': self.loss_history[-100:],
            'input_feature_stats': {
                'latest': self.input_feature_monitor.feature_history['all'][-1] if self.input_feature_monitor.feature_history['all'] else None,
                'suspicious_features': self.input_feature_monitor.get_suspicious_features(),
                'sudden_changes': self.input_feature_monitor.detect_sudden_changes() if len(self.input_feature_monitor.feature_history['all']) > 20 else [],
            }
        }

        report_file = self.output_dir / 'diagnosis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"✓ JSON报告已保存: {report_file}")

        # 2. 可视化
        self.plot_diagnostics()

        print(f"\n所有诊断结果已保存到: {self.output_dir}")

    def plot_diagnostics(self):
        """生成诊断图表"""
        # 设置样式
        sns.set_style("whitegrid")

        # 1. 损失曲线
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        steps = [h['step'] for h in self.loss_history]
        losses = [h['loss'] for h in self.loss_history]
        axes[0, 0].plot(steps, losses)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss History')
        axes[0, 0].set_yscale('log')

        # Learning Rate
        lr_steps = [h['step'] for h in self.lr_history]
        lrs = [h['lr'] for h in self.lr_history]
        axes[0, 1].plot(lr_steps, lrs)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')

        # Gradient Norms (top 5 layers)
        top_layers = sorted(
            self.gradient_monitor.gradient_history.items(),
            key=lambda x: x[1][-1]['norm'] if x[1] else 0,
            reverse=True
        )[:5]

        for name, history in top_layers:
            steps_grad = [h['step'] for h in history]
            norms = [h['norm'] for h in history]
            axes[1, 0].plot(steps_grad, norms, label=name.split('.')[-1][:20], alpha=0.7)

        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Top 5 Layers Gradient Norms')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].set_yscale('log')

        # Data statistics
        data_steps = [h['step'] for h in self.data_stats_history]
        x_means = [h['x_mean'] for h in self.data_stats_history]
        x_stds = [h['x_std'] for h in self.data_stats_history]

        ax_twin = axes[1, 1].twinx()
        axes[1, 1].plot(data_steps, x_means, 'b-', label='Mean', alpha=0.7)
        ax_twin.plot(data_steps, x_stds, 'r-', label='Std', alpha=0.7)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Data Mean', color='b')
        ax_twin.set_ylabel('Data Std', color='r')
        axes[1, 1].set_title('Input Data Statistics')

        plt.tight_layout()
        plot_file = self.output_dir / 'diagnostics.png'
        plt.savefig(plot_file, dpi=150)
        plt.close()

        print(f"✓ 诊断图表已保存: {plot_file}")

        # 2. 梯度热图（最近100步）
        if len(self.gradient_monitor.gradient_stats) > 0:
            self.plot_gradient_heatmap()

        # 3. 输入特征变化图表
        if len(self.input_feature_monitor.feature_history['all']) > 0:
            self.plot_feature_changes()

    def plot_gradient_heatmap(self):
        """绘制梯度范数热图"""
        # 获取最近100步的数据
        recent_steps = sorted(self.gradient_monitor.gradient_stats.keys())[-100:]

        if len(recent_steps) < 2:
            return

        # 收集所有层的梯度范数
        all_layers = set()
        for step in recent_steps:
            all_layers.update(self.gradient_monitor.gradient_stats[step].keys())

        all_layers = sorted(list(all_layers))[:30]  # 最多显示30层

        # 构建矩阵
        matrix = np.zeros((len(all_layers), len(recent_steps)))

        for i, layer in enumerate(all_layers):
            for j, step in enumerate(recent_steps):
                stats = self.gradient_monitor.gradient_stats[step]
                if layer in stats:
                    matrix[i, j] = np.log10(stats[layer]['norm'] + 1e-10)

        # 绘制热图
        fig, ax = plt.subplots(figsize=(15, 10))

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')

        # 设置标签
        ax.set_yticks(range(len(all_layers)))
        ax.set_yticklabels([name.split('.')[-1][:30] for name in all_layers], fontsize=8)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Layer')
        ax.set_title('Gradient Norm Heatmap (log10 scale)')

        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log10(Gradient Norm)', rotation=270, labelpad=20)

        plt.tight_layout()
        heatmap_file = self.output_dir / 'gradient_heatmap.png'
        plt.savefig(heatmap_file, dpi=150)
        plt.close()

        print(f"✓ 梯度热图已保存: {heatmap_file}")

    def plot_feature_changes(self):
        """绘制输入特征变化图表"""
        feature_history = self.input_feature_monitor.feature_history['all']

        if len(feature_history) < 2:
            return

        # 创建3x3的子图
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        steps = [h['step'] for h in feature_history]

        # 定义要绘制的特征
        features_to_plot = [
            ('x_col0_mean', 'Node Feature: Charge (mean)'),
            ('x_col1_mean', 'Node Feature: Atomic Number (mean)'),
            ('x_col2_mean', 'Node Feature: Mass (mean)'),
            ('pos_mean', 'Position (mean)'),
            ('pos_std', 'Position (std)'),
            ('edge_attr_col0_mean', 'Edge: r_eq (mean)'),
            ('edge_attr_col1_mean', 'Edge: k (mean)'),
            ('triple_attr_col0_mean', 'Angle: theta_eq (mean)'),
            ('triple_attr_col1_mean', 'Angle: k (mean)'),
        ]

        for idx, (feat_key, title) in enumerate(features_to_plot):
            if idx >= len(axes):
                break

            # 提取数据
            values = [h.get(feat_key, np.nan) for h in feature_history]

            # 过滤NaN
            valid_mask = ~np.isnan(values)
            valid_steps = np.array(steps)[valid_mask]
            valid_values = np.array(values)[valid_mask]

            if len(valid_values) > 0:
                axes[idx].plot(valid_steps, valid_values, 'b-', alpha=0.7)
                axes[idx].set_xlabel('Step')
                axes[idx].set_ylabel('Value')
                axes[idx].set_title(title, fontsize=10)
                axes[idx].grid(True, alpha=0.3)

                # 标注异常值
                if len(valid_values) > 10:
                    mean = np.mean(valid_values)
                    std = np.std(valid_values)
                    threshold = mean + 3 * std
                    anomaly_mask = valid_values > threshold
                    if anomaly_mask.any():
                        axes[idx].scatter(valid_steps[anomaly_mask], valid_values[anomaly_mask],
                                        color='red', s=30, zorder=5, label='Anomaly')
                        axes[idx].legend(fontsize=8)

        plt.tight_layout()
        feature_plot_file = self.output_dir / 'feature_changes.png'
        plt.savefig(feature_plot_file, dpi=150)
        plt.close()

        print(f"✓ 特征变化图表已保存: {feature_plot_file}")


# ============================================================================
# 损失函数（从 04_train_model.py 复制）
# ============================================================================

def compute_loss(pred, target, loss_fn='cosine', cosine_weight=0.7, mse_weight=0.3, temperature=0.07):
    """计算损失"""
    if loss_fn == 'mse':
        return F.mse_loss(pred, target), {}
    elif loss_fn == 'cosine':
        cosine_sim = F.cosine_similarity(pred, target, dim=-1).mean()
        return 1 - cosine_sim, {'cosine_sim': cosine_sim.item()}
    elif loss_fn == 'cosine_mse':
        mse = F.mse_loss(pred, target)
        cosine_sim = F.cosine_similarity(pred, target, dim=-1).mean()
        cosine_loss = 1 - cosine_sim
        loss = cosine_weight * cosine_loss + mse_weight * mse
        return loss, {'cosine_sim': cosine_sim.item(), 'mse': mse.item()}
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


# ============================================================================
# 主训练函数
# ============================================================================

def train_with_diagnosis(
    model, train_loader, val_loader, optimizer, scheduler,
    monitor, device, args
):
    """带诊断的训练函数"""

    # 记录初始权重
    monitor.weight_monitor.record_initial_weights(model)

    # 注册激活值监控hooks
    monitor.activation_monitor.register_hooks(model)

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None

    best_val_loss = float('inf')

    try:
        for epoch in range(args.epochs):
            monitor.start_epoch(epoch)

            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"{'='*60}")

            # Training
            model.train()
            train_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                target = batch.target_embedding.to(device)

                optimizer.zero_grad()

                # Forward
                with autocast(enabled=args.use_amp):
                    pocket_emb = model(batch)
                    loss, metrics = compute_loss(
                        pocket_emb, target,
                        loss_fn=args.loss_fn,
                        cosine_weight=args.cosine_weight,
                        mse_weight=args.mse_weight,
                        temperature=args.temperature
                    )

                # Backward
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                # 记录诊断信息
                diag = monitor.record_step(model, loss, optimizer, batch, batch_idx)

                # 检测梯度爆炸
                if diag['total_grad_norm'] > 1000:
                    print(f"\n❌ 检测到梯度爆炸! Total grad norm: {diag['total_grad_norm']:.2f}")
                    print("  停止训练并生成报告...")
                    raise KeyboardInterrupt

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                # Optimizer step
                if args.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                train_loss += loss.item()

                # 打印进度
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}: "
                          f"Loss={loss.item():.4f}, "
                          f"GradNorm={diag['total_grad_norm']:.4f}")

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    target = batch.target_embedding.to(device)

                    pocket_emb = model(batch)
                    loss, _ = compute_loss(pocket_emb, target, loss_fn=args.loss_fn)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), monitor.output_dir / 'best_model.pt')
                print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

    except KeyboardInterrupt:
        print("\n训练被中断")

    finally:
        # 移除hooks
        monitor.activation_monitor.remove_hooks()

        # 生成报告
        monitor.save_report()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="梯度爆炸诊断工具")

    # Model arguments
    parser.add_argument("--model_version", type=str, default="v3", choices=["v2", "v3"])
    parser.add_argument("--output_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ligand_embeddings", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--split_idx", type=int, default=0)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.5)
    parser.add_argument("--loss_fn", type=str, default="cosine")
    parser.add_argument("--cosine_weight", type=float, default=0.7)
    parser.add_argument("--mse_weight", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--use_amp", action="store_true")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="gradient_diagnosis")

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load split
    split_file = Path(args.split_dir) / f"split_{args.split_idx}.json"
    with open(split_file, 'r') as f:
        split = json.load(f)

    # Create datasets
    train_dataset = LigandEmbeddingDataset(
        split['train'],
        args.data_dir,
        args.ligand_embeddings
    )
    val_dataset = LigandEmbeddingDataset(
        split['val'],
        args.data_dir,
        args.ligand_embeddings
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    if args.model_version == 'v3':
        model = RNAPocketEncoderV3(
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            use_geometric_mp=True,
            use_enhanced_invariants=True,
            pooling_type='multihead_attention',
            num_attention_heads=4,
            dropout=0.1
        )
    else:
        model = RNAPocketEncoderV2(
            output_dim=args.output_dim,
            num_layers=args.num_layers
        )

    model = model.to(device)
    print(f"Model: {args.model_version}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create monitor
    monitor = TrainingMonitor(args.output_dir)

    # Train with diagnosis
    train_with_diagnosis(
        model, train_loader, val_loader,
        optimizer, scheduler,
        monitor, device, args
    )


if __name__ == "__main__":
    main()
