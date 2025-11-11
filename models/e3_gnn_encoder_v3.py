#!/usr/bin/env python3
"""
E(3) Equivariant GNN Encoder - Version 3.0 (Improved)

改进版本，集成以下增强功能:
1. ✅ 几何信息融入的角度/二面角消息传递
2. ✅ 更丰富的不变特征提取 (56 → 204 维)
3. ✅ Multi-head attention pooling
4. ✅ 物理约束loss支持
5. ✅ Bessel basis + Polynomial cutoff (NEW!)
6. ✅ Improved message passing from layers/ (NEW!)
7. ✅ Affine LayerNorm with learnable parameters (NEW!)

使用方法:
    from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3

    model = RNAPocketEncoderV3(
        output_dim=512,
        num_layers=4,
        use_geometric_mp=True,
        use_enhanced_invariants=True,
        use_improved_layers=True,  # 使用 layers/ 改进组件
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, scatter
from e3nn import o3
from e3nn.nn import Gate
import warnings

# Setup path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Priority 1: Import improved layers (最优先)
try:
    from .layers import (
        ImprovedE3MessagePassingLayer,
        BesselBasis,
        PolynomialCutoff,
        EquivariantLayerNorm as LayersEquivariantLayerNorm,
        EquivariantRMSNorm
    )
    _has_improved_layers = True
except ImportError:
    try:
        from layers import (
            ImprovedE3MessagePassingLayer,
            BesselBasis,
            PolynomialCutoff,
            EquivariantLayerNorm as LayersEquivariantLayerNorm,
            EquivariantRMSNorm
        )
        _has_improved_layers = True
    except ImportError:
        _has_improved_layers = False
        ImprovedE3MessagePassingLayer = None
        warnings.warn("layers/ module not found. Using basic implementations.")

# Priority 2: Import V2 base components (备用)
try:
    from e3_gnn_encoder_v2 import PhysicalFeatureEmbedding
    _has_v2_components = True
except ImportError:
    _has_v2_components = False
    warnings.warn("Could not import from e3_gnn_encoder_v2.")

# Priority 3: Import improved components (几何MP等)
try:
    from improved_components import (
        GeometricAngleMessagePassing,
        GeometricDihedralMessagePassing,
        EnhancedInvariantExtractor,
        MultiHeadAttentionPooling,
        PhysicsConstraintLoss
    )
    _has_improved_components = True
except ImportError:
    _has_improved_components = False
    warnings.warn("Could not import improved_components.")


class EquivariantLayerNorm(nn.Module):
    """
    E(3)-equivariant LayerNorm - 改进版，归一化所有特征类型

    策略:
    - 标量特征 (l=0): 使用标准 LayerNorm with affine parameters
    - 向量特征 (l=1): 归一化每个向量的范数
    - 张量特征 (l=2): 归一化每个张量的范数

    这样既保持了E(3)等变性，又防止了特征幅值爆炸

    改进: 添加可学习的 affine 参数 (scale/shift)
    """
    def __init__(self, irreps, normalize_vectors=True, normalize_tensors=True,
                 affine=True, eps=1e-5):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.normalize_vectors = normalize_vectors
        self.normalize_tensors = normalize_tensors
        self.affine = affine
        self.eps = eps

        # 找到所有特征的位置
        self.scalar_indices = []
        self.vector_slices = []  # [(start, end), ...]
        self.tensor_slices = []  # [(start, end), ...]

        idx = 0
        for mul, ir in self.irreps:
            if ir.l == 0:
                # 标量特征
                for _ in range(mul):
                    self.scalar_indices.append(idx)
                    idx += ir.dim
            elif ir.l == 1:
                # 向量特征 (3D)
                for _ in range(mul):
                    self.vector_slices.append((idx, idx + ir.dim))
                    idx += ir.dim
            elif ir.l == 2:
                # 张量特征 (5D)
                for _ in range(mul):
                    self.tensor_slices.append((idx, idx + ir.dim))
                    idx += ir.dim
            else:
                idx += mul * ir.dim

        # 为标量特征创建 LayerNorm (affine=False，我们自己管理)
        if len(self.scalar_indices) > 0:
            self.layer_norm = nn.LayerNorm(len(self.scalar_indices), elementwise_affine=False, eps=eps)

            # 添加可学习的 affine 参数
            if affine:
                self.weight = nn.Parameter(torch.ones(len(self.scalar_indices)))
                self.bias = nn.Parameter(torch.zeros(len(self.scalar_indices)))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
        else:
            self.layer_norm = None
            self.weight = None
            self.bias = None

    def forward(self, x):
        """
        Args:
            x: [num_atoms, irreps_dim]

        Returns:
            x_norm: [num_atoms, irreps_dim] - 所有特征都归一化
        """
        x_norm = x.clone()

        # 1. 归一化标量特征
        if self.layer_norm is not None and len(self.scalar_indices) > 0:
            scalar_features = x[:, self.scalar_indices]  # [num_atoms, num_scalars]
            scalar_features_norm = self.layer_norm(scalar_features)

            # 应用 affine 变换 (如果启用)
            if self.affine and self.weight is not None:
                scalar_features_norm = scalar_features_norm * self.weight + self.bias

            x_norm[:, self.scalar_indices] = scalar_features_norm

        # 2. 归一化向量范数 (保持方向，缩放幅值)
        if self.normalize_vectors and len(self.vector_slices) > 0:
            for start, end in self.vector_slices:
                vec = x[:, start:end]  # [num_atoms, 3]
                norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp(min=1e-6)
                # 归一化到单位范数，然后乘以可学习的缩放因子
                # 这里使用均值范数作为目标
                mean_norm = norm.mean()
                vec_normalized = vec / norm * mean_norm
                x_norm[:, start:end] = vec_normalized

        # 3. 归一化张量范数 (保持方向，缩放幅值)
        if self.normalize_tensors and len(self.tensor_slices) > 0:
            for start, end in self.tensor_slices:
                tensor = x[:, start:end]  # [num_atoms, 5]
                norm = torch.linalg.norm(tensor, dim=-1, keepdim=True).clamp(min=1e-6)
                mean_norm = norm.mean()
                tensor_normalized = tensor / norm * mean_norm
                x_norm[:, start:end] = tensor_normalized

        return x_norm


class RNAPocketEncoderV3(nn.Module):
    """
    E(3) Equivariant GNN for RNA binding pockets - Version 3.0 (Improved)

    主要改进:
    1. 几何增强的角度/二面角消息传递
    2. 更丰富的不变特征提取 (204维 vs 56维)
    3. Multi-head attention pooling
    4. 物理约束loss集成

    相比 V2 的优势:
    - 更准确的几何建模
    - 更强的特征表达能力
    - 更好的图级别表示
    - 物理约束正则化
    """

    def __init__(
        self,
        input_dim=3,  # [charge, atomic_num, mass]
        feature_hidden_dim=64,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=4,
        num_radial_basis=8,
        radial_hidden_dim=64,
        pooling_hidden_dim=128,
        r_max=6.0,
        avg_num_neighbors=None,
        use_gate=True,
        use_layer_norm=False,
        use_multi_hop=True,
        use_nonbonded=True,
        pooling_type='multihead_attention',  # 'multihead_attention' or 'attention'
        num_attention_heads=4,  # For multihead attention
        dropout=0.0,
        # V3新增参数
        use_geometric_mp=True,  # 是否使用几何增强的MP
        use_enhanced_invariants=True,  # 是否使用增强的不变量提取
        use_improved_layers=True,  # 是否使用 layers/ 改进组件 (NEW!)
        norm_type='layer',  # 'layer' or 'rms' (NEW!)
        # 可学习权重的初始值（在实际权重空间，会被转换到log-space）
        initial_angle_weight=0.5,
        initial_dihedral_weight=0.5,
        initial_nonbonded_weight=0.5,
    ):
        """
        Args:
            use_geometric_mp: 是否在角度/二面角MP中使用几何信息
            use_enhanced_invariants: 是否使用增强的不变量提取(204维 vs 56维)
            use_improved_layers: 是否使用 layers/ 改进组件 (Bessel+Cutoff+ImprovedMP)
            norm_type: 归一化类型 ('layer' 或 'rms')
            pooling_type: 'multihead_attention' 或 'attention'
            num_attention_heads: 多头注意力的头数
            initial_angle_weight: 角度消息传递的初始权重 (0~1之间，默认0.5)
            initial_dihedral_weight: 二面角消息传递的初始权重 (0~1之间，默认0.5)
            initial_nonbonded_weight: 非键消息传递的初始权重 (0~1之间，默认0.5)
        """
        super().__init__()

        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.use_multi_hop = use_multi_hop
        self.use_nonbonded = use_nonbonded
        self.use_geometric_mp = use_geometric_mp
        self.use_enhanced_invariants = use_enhanced_invariants
        self.use_improved_layers = use_improved_layers and _has_improved_layers
        self.norm_type = norm_type

        # Learnable combining weights (使用 log-space 参数化防止无限增长)
        # 使用 sigmoid 约束到 [0, 1] 范围
        # 从权重空间转换到log-space: logit(w) = log(w / (1-w))
        if use_multi_hop:
            # 将初始权重从 [0, 1] 转换到 log-space
            # 裁剪到 [0.01, 0.99] 避免log(0)或log(∞)
            angle_w_clipped = max(0.01, min(0.99, initial_angle_weight))
            dihedral_w_clipped = max(0.01, min(0.99, initial_dihedral_weight))

            angle_logit = torch.log(torch.tensor(angle_w_clipped / (1 - angle_w_clipped)))
            dihedral_logit = torch.log(torch.tensor(dihedral_w_clipped / (1 - dihedral_w_clipped)))

            self.log_angle_weight = nn.Parameter(angle_logit)
            self.log_dihedral_weight = nn.Parameter(dihedral_logit)

        if use_nonbonded:
            nonbonded_w_clipped = max(0.01, min(0.99, initial_nonbonded_weight))
            nonbonded_logit = torch.log(torch.tensor(nonbonded_w_clipped / (1 - nonbonded_w_clipped)))
            self.log_nonbonded_weight = nn.Parameter(nonbonded_logit)

        # Input embedding (same as V2)
        self.input_embedding = PhysicalFeatureEmbedding(
            input_dim=input_dim,
            hidden_dim=feature_hidden_dim,
            output_irreps=hidden_irreps
        )

        # 1-hop bonded message passing (使用改进版本 if available)
        self.bonded_mp_layers = nn.ModuleList()
        for i in range(num_layers):
            if self.use_improved_layers:
                # 使用 ImprovedE3MessagePassingLayer from layers/
                layer = ImprovedE3MessagePassingLayer(
                    irreps_in=self.hidden_irreps,
                    irreps_out=self.hidden_irreps,
                    irreps_sh="0e + 1o + 2e",
                    r_max=r_max,
                    num_radial_basis=num_radial_basis,
                    radial_hidden_dim=radial_hidden_dim,
                    avg_num_neighbors=avg_num_neighbors,
                    use_gate=use_gate,
                    use_sc=True,
                    use_resnet=True,
                    use_layer_norm=use_layer_norm,
                    edge_attr_dim=2  # [req, k]
                )
            else:
                # 使用 V2 的基础实现
                from e3_gnn_encoder_v2 import E3GNNMessagePassingLayer
                layer = E3GNNMessagePassingLayer(
                    irreps_in=self.hidden_irreps,
                    irreps_out=self.hidden_irreps,
                    irreps_sh="0e + 1o + 2e",
                    num_radial_basis=num_radial_basis,
                    radial_hidden_dim=radial_hidden_dim,
                    edge_attr_dim=2,  # [req, k]
                    r_max=r_max,
                    avg_num_neighbors=avg_num_neighbors,
                    use_gate=use_gate,
                    use_sc=True,
                    use_resnet=True,
                    use_layer_norm=use_layer_norm
                )
            self.bonded_mp_layers.append(layer)

        # 2-hop angle message passing (IMPROVED with geometry!)
        if use_multi_hop:
            self.angle_mp_layers = nn.ModuleList()
            for i in range(num_layers):
                if use_geometric_mp:
                    # 使用几何增强版本（带 LayerNorm 稳定性改进）
                    layer = GeometricAngleMessagePassing(
                        irreps_in=self.hidden_irreps,
                        irreps_out=self.hidden_irreps,
                        angle_attr_dim=2,
                        hidden_dim=64,
                        use_geometry=True,
                        use_layer_norm=True  # 启用 LayerNorm 提高数值稳定性
                    )
                else:
                    # 使用原始版本（从v2导入）
                    from e3_gnn_encoder_v2 import AngleMessagePassing
                    layer = AngleMessagePassing(
                        irreps_in=self.hidden_irreps,
                        irreps_out=self.hidden_irreps,
                        angle_attr_dim=2,
                        hidden_dim=64
                    )
                self.angle_mp_layers.append(layer)

        # 3-hop dihedral message passing (IMPROVED with geometry!)
        if use_multi_hop:
            self.dihedral_mp_layers = nn.ModuleList()
            for i in range(num_layers):
                if use_geometric_mp:
                    # 使用几何增强版本（带 LayerNorm 稳定性改进）
                    layer = GeometricDihedralMessagePassing(
                        irreps_in=self.hidden_irreps,
                        irreps_out=self.hidden_irreps,
                        dihedral_attr_dim=3,
                        hidden_dim=64,
                        use_geometry=True,
                        use_layer_norm=True  # 启用 LayerNorm 提高数值稳定性
                    )
                else:
                    # 使用原始版本
                    from e3_gnn_encoder_v2 import DihedralMessagePassing
                    layer = DihedralMessagePassing(
                        irreps_in=self.hidden_irreps,
                        irreps_out=self.hidden_irreps,
                        dihedral_attr_dim=3,
                        hidden_dim=64
                    )
                self.dihedral_mp_layers.append(layer)

        # Non-bonded message passing (same as V2)
        if use_nonbonded:
            self.nonbonded_mp_layers = nn.ModuleList()
            for i in range(num_layers):
                layer = E3GNNMessagePassingLayer(
                    irreps_in=self.hidden_irreps,
                    irreps_out=self.hidden_irreps,
                    irreps_sh="0e + 1o + 2e",
                    num_radial_basis=num_radial_basis,
                    radial_hidden_dim=radial_hidden_dim,
                    edge_attr_dim=3,  # [LJ_A, LJ_B, distance]
                    r_max=r_max,
                    avg_num_neighbors=avg_num_neighbors,
                    use_gate=use_gate,
                    use_sc=False,
                    use_resnet=False,
                    use_layer_norm=use_layer_norm
                )
                self.nonbonded_mp_layers.append(layer)

        # LayerNorm for stabilizing multi-hop aggregation (防止特征幅值爆炸)
        # 支持 RMSNorm (更快) 或 LayerNorm (with affine)
        if use_multi_hop or use_nonbonded:
            self.aggregation_layer_norms = nn.ModuleList()
            for i in range(num_layers):
                if self.norm_type == 'rms' and self.use_improved_layers:
                    # 使用 RMSNorm from layers/ (更快)
                    self.aggregation_layer_norms.append(
                        EquivariantRMSNorm(self.hidden_irreps, affine=True)
                    )
                elif self.use_improved_layers:
                    # 使用 layers/ 的 LayerNorm (有 affine)
                    self.aggregation_layer_norms.append(
                        LayersEquivariantLayerNorm(self.hidden_irreps, affine=True)
                    )
                else:
                    # 使用本地的 EquivariantLayerNorm (现在也有 affine)
                    self.aggregation_layer_norms.append(
                        EquivariantLayerNorm(self.hidden_irreps, affine=True)
                    )

        # Invariant feature extraction (IMPROVED!)
        if use_enhanced_invariants:
            # 使用增强版本: 204维（带归一化稳定性改进）
            self.invariant_extractor = EnhancedInvariantExtractor(
                hidden_irreps,
                normalize_features=True  # 启用特征归一化提高数值稳定性
            )
            self.invariant_dim = self.invariant_extractor.invariant_dim  # 204
        else:
            # 使用原始版本: 56维
            self.invariant_extractor = None
            scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0])
            self.scalar_dim = scalar_irreps.dim
            self.num_l1_irreps = sum(mul for mul, ir in self.hidden_irreps if ir.l == 1)
            self.num_l2_irreps = sum(mul for mul, ir in self.hidden_irreps if ir.l == 2)
            self.invariant_dim = self.scalar_dim + self.num_l1_irreps + self.num_l2_irreps  # 56
            self._build_irreps_slices()

        # Pooling (IMPROVED with multi-head attention!)
        if pooling_type == 'multihead_attention':
            self.pooling = MultiHeadAttentionPooling(
                input_dim=self.invariant_dim,
                num_heads=num_attention_heads,
                hidden_dim=pooling_hidden_dim,
                dropout=dropout
            )
            self.pooling_mlp = None
        elif pooling_type == 'attention':
            # 原始的MLP attention
            self.pooling_mlp = nn.Sequential(
                nn.Linear(self.invariant_dim, pooling_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(pooling_hidden_dim, pooling_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(pooling_hidden_dim, 1)
            )
            self.pooling = None
        else:
            # No attention
            self.pooling_mlp = None
            self.pooling = None

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.invariant_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def _build_irreps_slices(self):
        """Build index slices (for original invariant extraction)"""
        self.irreps_slices = {'l0': [], 'l1': [], 'l2': []}

        idx = 0
        for mul, ir in self.hidden_irreps:
            dim = ir.dim
            for _ in range(mul):
                if ir.l == 0:
                    self.irreps_slices['l0'].append((idx, idx + dim))
                elif ir.l == 1:
                    self.irreps_slices['l1'].append((idx, idx + dim))
                elif ir.l == 2:
                    self.irreps_slices['l2'].append((idx, idx + dim))
                idx += dim

    @property
    def angle_weight(self):
        """返回约束后的角度权重 (范围: [0, 1])"""
        if hasattr(self, 'log_angle_weight'):
            return torch.sigmoid(self.log_angle_weight)
        return torch.tensor(0.0, device=self.log_angle_weight.device if hasattr(self, 'log_angle_weight') else 'cpu')

    @property
    def dihedral_weight(self):
        """返回约束后的二面角权重 (范围: [0, 1])"""
        if hasattr(self, 'log_dihedral_weight'):
            return torch.sigmoid(self.log_dihedral_weight)
        return torch.tensor(0.0, device=self.log_dihedral_weight.device if hasattr(self, 'log_dihedral_weight') else 'cpu')

    @property
    def nonbonded_weight(self):
        """返回约束后的非键权重 (范围: [0, 1])"""
        if hasattr(self, 'log_nonbonded_weight'):
            return torch.sigmoid(self.log_nonbonded_weight)
        return torch.tensor(0.0, device=self.log_nonbonded_weight.device if hasattr(self, 'log_nonbonded_weight') else 'cpu')

    def extract_invariant_features(self, h):
        """Extract E(3) invariant features (original version for V2 compatibility)"""
        if self.invariant_extractor is not None:
            # Use enhanced version
            return self.invariant_extractor(h)

        # Original version (from V2)
        invariant_features = []

        # Scalars
        for start, end in self.irreps_slices['l0']:
            invariant_features.append(h[:, start:end])

        # Vector norms
        for start, end in self.irreps_slices['l1']:
            vec = h[:, start:end]
            norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
            invariant_features.append(norm)

        # Tensor norms
        for start, end in self.irreps_slices['l2']:
            tensor = h[:, start:end]
            norm = torch.linalg.norm(tensor, dim=-1, keepdim=True)
            invariant_features.append(norm)

        t = torch.cat(invariant_features, dim=-1)
        return t

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Pocket embeddings [batch_size, output_dim]
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # Get batch
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial embedding
        h = self.input_embedding(x)

        # Message passing layers
        for i in range(self.num_layers):
            h_new = h

            # 1-hop bonded
            h_bonded = self.bonded_mp_layers[i](h, pos, edge_index, edge_attr)
            h_new = h_bonded

            # 2-hop angles
            if self.use_multi_hop and hasattr(data, 'triple_index'):
                if self.use_geometric_mp:
                    # 传入pos用于几何计算
                    h_angle = self.angle_mp_layers[i](h, pos, data.triple_index, data.triple_attr)
                else:
                    # 原始版本不需要pos
                    h_angle = self.angle_mp_layers[i](h, data.triple_index, data.triple_attr)
                h_new = h_new + self.angle_weight * h_angle

            # 3-hop dihedrals
            if self.use_multi_hop and hasattr(data, 'quadra_index'):
                if self.use_geometric_mp:
                    # 传入pos用于几何计算
                    h_dihedral = self.dihedral_mp_layers[i](h, pos, data.quadra_index, data.quadra_attr)
                else:
                    # 原始版本不需要pos
                    h_dihedral = self.dihedral_mp_layers[i](h, data.quadra_index, data.quadra_attr)
                h_new = h_new + self.dihedral_weight * h_dihedral

            # Non-bonded
            if self.use_nonbonded and hasattr(data, 'nonbonded_edge_index'):
                h_nonbonded = self.nonbonded_mp_layers[i](
                    h, pos, data.nonbonded_edge_index, data.nonbonded_edge_attr
                )
                h_new = h_new + self.nonbonded_weight * h_nonbonded

            # Apply LayerNorm to stabilize aggregated features (防止幅值爆炸)
            if (self.use_multi_hop or self.use_nonbonded) and hasattr(self, 'aggregation_layer_norms'):
                h = self.aggregation_layer_norms[i](h_new)
            else:
                h = h_new

            # Dropout on scalars only
            if self.dropout > 0 and self.training:
                scalar_mask = torch.zeros(h.size(-1), dtype=torch.bool, device=h.device)
                idx = 0
                for mul, ir in self.hidden_irreps:
                    if ir.l == 0:
                        scalar_mask[idx:idx + mul * ir.dim] = True
                    idx += mul * ir.dim

                h_dropped = h.clone()
                h_dropped[..., scalar_mask] = F.dropout(
                    h[..., scalar_mask],
                    p=self.dropout,
                    training=self.training
                )
                h = h_dropped

        # Extract invariant features
        t = self.extract_invariant_features(h)

        # Pooling
        if self.pooling_type == 'multihead_attention' and self.pooling is not None:
            # Multi-head attention pooling
            graph_embedding = self.pooling(t, batch)
        elif self.pooling_type == 'attention' and self.pooling_mlp is not None:
            # Original MLP attention pooling
            attention_logits = self.pooling_mlp(t)
            attention_weights = softmax(attention_logits, index=batch, dim=0)
            weighted_features = t * attention_weights
            graph_embedding = scatter(
                weighted_features,
                index=batch,
                dim=0,
                reduce='sum'
            )
        elif self.pooling_type == 'mean':
            graph_embedding = scatter(t, index=batch, dim=0, reduce='mean')
        elif self.pooling_type == 'sum':
            graph_embedding = scatter(t, index=batch, dim=0, reduce='sum')
        elif self.pooling_type == 'max':
            graph_embedding = scatter(t, index=batch, dim=0, reduce='max')
        else:
            # Default to mean
            graph_embedding = scatter(t, index=batch, dim=0, reduce='mean')

        # Project to output
        output = self.output_projection(graph_embedding)

        return output

    def get_node_embeddings(self, data):
        """
        Get per-node invariant embeddings (before pooling).

        Useful for visualization and analysis.
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # Initial embedding
        h = self.input_embedding(x)

        # Message passing
        for i in range(self.num_layers):
            h_bonded = self.bonded_mp_layers[i](h, pos, edge_index, edge_attr)
            h_new = h_bonded

            if self.use_multi_hop and hasattr(data, 'triple_index'):
                if self.use_geometric_mp:
                    h_angle = self.angle_mp_layers[i](h, pos, data.triple_index, data.triple_attr)
                else:
                    h_angle = self.angle_mp_layers[i](h, data.triple_index, data.triple_attr)
                h_new = h_new + self.angle_weight * h_angle

            if self.use_multi_hop and hasattr(data, 'quadra_index'):
                if self.use_geometric_mp:
                    h_dihedral = self.dihedral_mp_layers[i](h, pos, data.quadra_index, data.quadra_attr)
                else:
                    h_dihedral = self.dihedral_mp_layers[i](h, data.quadra_index, data.quadra_attr)
                h_new = h_new + self.dihedral_weight * h_dihedral

            if self.use_nonbonded and hasattr(data, 'nonbonded_edge_index'):
                h_nonbonded = self.nonbonded_mp_layers[i](
                    h, pos, data.nonbonded_edge_index, data.nonbonded_edge_attr
                )
                h_new = h_new + self.nonbonded_weight * h_nonbonded

            h = h_new

        # Extract invariant features
        t = self.extract_invariant_features(h)

        return t

    def get_weight_stats(self):
        """
        获取可学习权重的统计信息 (用于监控)

        Returns:
            dict: 包含权重值、梯度、log空间参数等信息
        """
        stats = {}

        if hasattr(self, 'log_angle_weight'):
            stats['angle_weight'] = self.angle_weight.item()
            stats['log_angle_weight'] = self.log_angle_weight.item()
            if self.log_angle_weight.grad is not None:
                stats['angle_weight_grad'] = self.log_angle_weight.grad.item()

        if hasattr(self, 'log_dihedral_weight'):
            stats['dihedral_weight'] = self.dihedral_weight.item()
            stats['log_dihedral_weight'] = self.log_dihedral_weight.item()
            if self.log_dihedral_weight.grad is not None:
                stats['dihedral_weight_grad'] = self.log_dihedral_weight.grad.item()

        if hasattr(self, 'log_nonbonded_weight'):
            stats['nonbonded_weight'] = self.nonbonded_weight.item()
            stats['log_nonbonded_weight'] = self.log_nonbonded_weight.item()
            if self.log_nonbonded_weight.grad is not None:
                stats['nonbonded_weight_grad'] = self.log_nonbonded_weight.grad.item()

        return stats

    def get_feature_stats(self, data):
        """
        获取每层特征的统计信息 (用于诊断)

        Returns:
            dict: 包含每层特征的范数、均值、标准差等
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        stats = {'layers': []}

        # Initial embedding
        h = self.input_embedding(x)
        stats['input_norm'] = torch.linalg.norm(h, dim=-1).mean().item()

        # Message passing
        for i in range(self.num_layers):
            layer_stats = {}

            # 1-hop
            h_bonded = self.bonded_mp_layers[i](h, pos, edge_index, edge_attr)
            layer_stats['bonded_norm'] = torch.linalg.norm(h_bonded, dim=-1).mean().item()
            h_new = h_bonded

            # 2-hop
            if self.use_multi_hop and hasattr(data, 'triple_index'):
                if self.use_geometric_mp:
                    h_angle = self.angle_mp_layers[i](h, pos, data.triple_index, data.triple_attr)
                else:
                    h_angle = self.angle_mp_layers[i](h, data.triple_index, data.triple_attr)
                layer_stats['angle_norm'] = torch.linalg.norm(h_angle, dim=-1).mean().item()
                h_new = h_new + self.angle_weight * h_angle

            # 3-hop
            if self.use_multi_hop and hasattr(data, 'quadra_index'):
                if self.use_geometric_mp:
                    h_dihedral = self.dihedral_mp_layers[i](h, pos, data.quadra_index, data.quadra_attr)
                else:
                    h_dihedral = self.dihedral_mp_layers[i](h, data.quadra_index, data.quadra_attr)
                layer_stats['dihedral_norm'] = torch.linalg.norm(h_dihedral, dim=-1).mean().item()
                h_new = h_new + self.dihedral_weight * h_dihedral

            # Non-bonded
            if self.use_nonbonded and hasattr(data, 'nonbonded_edge_index'):
                h_nonbonded = self.nonbonded_mp_layers[i](
                    h, pos, data.nonbonded_edge_index, data.nonbonded_edge_attr
                )
                layer_stats['nonbonded_norm'] = torch.linalg.norm(h_nonbonded, dim=-1).mean().item()
                h_new = h_new + self.nonbonded_weight * h_nonbonded

            # 聚合后的统计
            layer_stats['aggregated_norm'] = torch.linalg.norm(h_new, dim=-1).mean().item()

            # LayerNorm后
            if (self.use_multi_hop or self.use_nonbonded) and hasattr(self, 'aggregation_layer_norms'):
                h = self.aggregation_layer_norms[i](h_new)
                layer_stats['after_norm'] = torch.linalg.norm(h, dim=-1).mean().item()
            else:
                h = h_new

            stats['layers'].append(layer_stats)

        return stats


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    print("=" * 80)
    print("Testing E(3) GNN Encoder V3.0 (Improved)")
    print("=" * 80)

    # Create realistic test data
    num_nodes = 100
    num_edges = 300
    num_angles = 150
    num_dihedrals = 80
    num_nonbonded = 200

    x = torch.randn(num_nodes, 3)  # Pure physical features
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.rand(num_edges, 2) * 2  # [k, req]

    # Angles: [i, j, k]
    triple_index = torch.randint(0, num_nodes, (3, num_angles))
    triple_attr = torch.rand(num_angles, 2)  # [theta_eq, k]
    triple_attr[:, 0] = triple_attr[:, 0] * 3.14  # Convert to radians

    # Dihedrals: [i, j, k, l]
    quadra_index = torch.randint(0, num_nodes, (4, num_dihedrals))
    quadra_attr = torch.rand(num_dihedrals, 3)  # [phi_k, per, phase]
    quadra_attr[:, 1] = torch.randint(1, 4, (num_dihedrals,)).float()  # periodicity
    quadra_attr[:, 2] = quadra_attr[:, 2] * 6.28  # phase in radians

    # Non-bonded
    nonbonded_edge_index = torch.randint(0, num_nodes, (2, num_nonbonded))
    nonbonded_edge_attr = torch.rand(num_nonbonded, 3)  # [LJ_A, LJ_B, dist]

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        triple_index=triple_index,
        triple_attr=triple_attr,
        quadra_index=quadra_index,
        quadra_attr=quadra_attr,
        nonbonded_edge_index=nonbonded_edge_index,
        nonbonded_edge_attr=nonbonded_edge_attr
    )

    # Test V3 with all improvements
    print("\nTest 1: V3 with all improvements enabled")
    print("-" * 80)
    model_v3 = RNAPocketEncoderV3(
        output_dim=512,
        num_layers=4,
        use_geometric_mp=True,
        use_enhanced_invariants=True,
        pooling_type='multihead_attention',
        num_attention_heads=4
    )

    print(f"Model parameters: {sum(p.numel() for p in model_v3.parameters()):,}")

    # Forward pass
    output = model_v3(data)
    print(f"Input: {num_nodes} atoms")
    print(f"Output: {output.shape}")
    print(f"✓ Forward pass successful!")

    # Test node embeddings
    node_emb = model_v3.get_node_embeddings(data)
    print(f"Node embeddings: {node_emb.shape}")
    print(f"Invariant dim: {model_v3.invariant_dim}")

    # Test V3 without improvements (should be similar to V2)
    print("\nTest 2: V3 with improvements disabled (V2-like)")
    print("-" * 80)
    model_v2_like = RNAPocketEncoderV3(
        output_dim=512,
        num_layers=4,
        use_geometric_mp=False,
        use_enhanced_invariants=False,
        pooling_type='attention'
    )

    output2 = model_v2_like(data)
    print(f"Output: {output2.shape}")
    print(f"Invariant dim: {model_v2_like.invariant_dim}")
    print(f"✓ V2-compatible mode works!")

    # Test batched data
    print("\nTest 3: Batched data")
    print("-" * 80)
    batch_data = Batch.from_data_list([data, data, data])
    batch_output = model_v3(batch_data)
    print(f"Batch size: 3")
    print(f"Output: {batch_output.shape}")
    print(f"✓ Batch processing works!")

    # Test physics loss
    print("\nTest 4: Physics constraint loss")
    print("-" * 80)
    physics_loss_fn = PhysicsConstraintLoss(
        use_bond=True,
        use_angle=True,
        use_dihedral=True
    )

    loss, loss_dict = physics_loss_fn(data)
    print(f"Total physics loss: {loss.item():.4f}")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")
    print(f"✓ Physics loss works!")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

    # Print summary
    print("\n📊 Model Comparison Summary:")
    print("-" * 80)
    print(f"{'Feature':<40} {'V2':<15} {'V3':<15}")
    print("-" * 80)
    print(f"{'Geometric Angle/Dihedral MP':<40} {'❌ No':<15} {'✅ Yes':<15}")
    print(f"{'Invariant Features Dim':<40} {'56':<15} {'204':<15}")
    print(f"{'Multi-head Attention Pooling':<40} {'❌ No':<15} {'✅ Yes':<15}")
    print(f"{'Physics Constraint Loss':<40} {'❌ No':<15} {'✅ Yes':<15}")
    print(f"{'Backward Compatible':<40} {'N/A':<15} {'✅ Yes':<15}")
    print("-" * 80)
