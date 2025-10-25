# RNA-3E-FFI Models v2.0 - 修改总结

## 📋 概述

完成了模型架构的完整升级，以适配新的 v2.0 数据格式。新模型 `RNAPocketEncoderV2` 现在完全支持：

- ✅ 基于 Embedding 的离散特征处理
- ✅ 多跳消息传递（1-hop, 2-hop, 3-hop）
- ✅ 非键相互作用（LJ 参数）
- ✅ 物理参数完整整合
- ✅ E(3) 等变性保持

---

## 🎯 主要修改

### 1. **输入特征处理 - 完全重写**

**v1.0 输入格式（旧）：**
```python
x: [num_atoms, input_dim]  # 连续特征向量
```

**v2.0 输入格式（新）：**
```python
x: [num_atoms, 4]
   x[:, 0] = atom_type_idx    # 整数索引 (1-71)
   x[:, 1] = charge           # 浮点数
   x[:, 2] = residue_idx      # 整数索引 (1-43)
   x[:, 3] = atomic_num       # 整数 (原子序数)
```

**实现方案：**
创建了 `AMBERFeatureEmbedding` 模块：

```python
class AMBERFeatureEmbedding(nn.Module):
    def __init__(self, num_atom_types, num_residues, ...):
        # 1. Embedding 层处理离散特征
        self.atom_type_embedding = nn.Embedding(num_atom_types + 1, atom_embed_dim)
        self.residue_embedding = nn.Embedding(num_residues + 1, residue_embed_dim)

        # 2. 线性投影处理连续特征
        self.continuous_projection = nn.Sequential(
            nn.Linear(2, continuous_dim),  # charge + atomic_num
            nn.SiLU(),
            ...
        )

        # 3. 特征融合
        self.feature_fusion = nn.Sequential(...)

        # 4. 投影到 irreps
        self.irreps_projection = o3.Linear(...)
```

**关键特性：**
- 使用 `padding_idx=0` 处理缺失值
- 独立处理离散和连续特征
- 融合后统一投影到等变表示
- 保持 E(3) 等变性

---

### 2. **多跳消息传递 - 新增功能**

**v1.0（旧）：**
- 仅使用 1-hop bonded edges (`edge_index`)

**v2.0（新）：**
- **1-hop**: Bonded edges with bond parameters `[r_eq, k]`
- **2-hop**: Angle paths with angle parameters `[theta_eq, k]`
- **3-hop**: Dihedral paths with dihedral parameters `[phi_k, per, phase]`

#### 2.1 Angle Message Passing (2-hop)

```python
class AngleMessagePassing(nn.Module):
    """处理 i -> j -> k 角度路径"""

    def forward(self, x, triple_index, triple_attr):
        # triple_index: [3, num_angles] (i, j, k)
        # triple_attr: [num_angles, 2] (theta_eq, k)

        # 提取标量特征
        x_i = x_scalar[i]  # 起始原子
        x_k = x_scalar[k]  # 终止原子

        # 结合角度参数
        angle_input = torch.cat([x_i, x_k, triple_attr], dim=-1)

        # MLP 处理
        angle_messages = self.angle_mlp(angle_input)

        # 聚合到中心原子 j
        return scatter(angle_messages, j, ...)
```

#### 2.2 Dihedral Message Passing (3-hop)

```python
class DihedralMessagePassing(nn.Module):
    """处理 i -> j -> k -> l 二面角路径"""

    def forward(self, x, quadra_index, quadra_attr):
        # quadra_index: [4, num_dihedrals] (i, j, k, l)
        # quadra_attr: [num_dihedrals, 3] (phi_k, per, phase)

        x_i = x_scalar[i]
        x_l = x_scalar[l]

        dihedral_input = torch.cat([x_i, x_l, quadra_attr], dim=-1)
        dihedral_messages = self.dihedral_mlp(dihedral_input)

        # 聚合到中心原子 j 和 k
        dihedral_aggr = (scatter(messages, j, ...) + scatter(messages, k, ...)) / 2
        return dihedral_aggr
```

---

### 3. **非键相互作用 - 新增功能**

**v2.0 新增：**
- 处理 `nonbonded_edge_index` 和 `nonbonded_edge_attr`
- 使用真实的 Lennard-Jones 参数

```python
# 非键边属性：
nonbonded_edge_attr: [num_nb, 3]
    [:, 0] = LJ_A     # LJ A 系数
    [:, 1] = LJ_B     # LJ B 系数
    [:, 2] = distance # 空间距离
```

**实现：**
```python
# 独立的非键消息传递层
self.nonbonded_mp_layers = nn.ModuleList()
for i in range(num_layers):
    layer = E3GNNMessagePassingLayer(
        irreps_in=self.hidden_irreps,
        irreps_out=self.hidden_irreps,
        edge_attr_dim=3,  # LJ_A, LJ_B, distance
        use_sc=False,     # 非键交互不需要自连接
        use_resnet=False
    )
```

---

### 4. **物理参数整合 - 增强**

所有消息传递层现在都接受并使用物理参数：

```python
class E3GNNMessagePassingLayer(MessagePassing):
    def __init__(self, ..., edge_attr_dim=0):
        # 径向 MLP 输入：RBF + edge_attr
        radial_input_dim = num_radial_basis + edge_attr_dim
        self.radial_mlp = nn.Sequential(...)

    def message(self, x_j, pos_i, pos_j, edge_attr=None):
        # 计算径向基函数
        rbf = self.bessel_basis(distance)

        # 与边属性拼接
        if edge_attr is not None:
            radial_input = torch.cat([rbf, edge_attr], dim=-1)

        # 生成张量积权重
        tp_weights = self.radial_mlp(radial_input)

        # 应用张量积
        messages = self.tp(x_j, sh, tp_weights)
        return messages
```

**边属性使用：**
- **Bonded edges**: `[r_eq, k]` → 键长平衡位置和力常数
- **Angle paths**: `[theta_eq, k]` → 角度平衡值和力常数
- **Dihedral paths**: `[phi_k, per, phase]` → 二面角参数
- **Non-bonded**: `[LJ_A, LJ_B, dist]` → Lennard-Jones 参数

---

### 5. **主模型架构整合**

**Forward Pass 流程：**

```python
def forward(self, data):
    # 1. 输入嵌入（离散+连续特征）
    h = self.input_embedding(x)

    # 2. 多层消息传递
    for i in range(self.num_layers):
        # 2a. 1-hop bonded 消息传递
        h_bonded = self.bonded_mp_layers[i](h, pos, edge_index, edge_attr)

        # 2b. 2-hop angle 消息传递
        h_angle = self.angle_mp_layers[i](h, triple_index, triple_attr)

        # 2c. 3-hop dihedral 消息传递
        h_dihedral = self.dihedral_mp_layers[i](h, quadra_index, quadra_attr)

        # 2d. 非键消息传递
        h_nonbonded = self.nonbonded_mp_layers[i](h, pos, nonbonded_edge_index, nonbonded_edge_attr)

        # 2e. 加权组合
        h = h_bonded + 0.5*h_angle + 0.3*h_dihedral + 0.2*h_nonbonded

    # 3. 提取标量特征
    h_scalar = h[:, :self.scalar_dim]

    # 4. 池化
    graph_embedding = attention_pooling(h_scalar, batch)

    # 5. 输出投影
    output = self.output_projection(graph_embedding)

    return output
```

**加权系数设计（可学习）：**

模型使用**可学习的权重参数**来自适应地平衡不同类型的相互作用：

```python
# 初始化为经验值，训练过程中自动优化
self.angle_weight = nn.Parameter(torch.tensor(0.5))      # 角度贡献
self.dihedral_weight = nn.Parameter(torch.tensor(0.3))   # 二面角贡献
self.nonbonded_weight = nn.Parameter(torch.tensor(0.2))  # 非键贡献

# Forward pass 中的使用
h_new = h_bonded + self.angle_weight * h_angle + \
        self.dihedral_weight * h_dihedral + \
        self.nonbonded_weight * h_nonbonded
```

**优势**：
- ✅ 无需手动调优权重
- ✅ 模型自动学习最优组合
- ✅ 适应不同数据集特性
- ✅ 增加了 3 个可学习参数（angle_weight, dihedral_weight, nonbonded_weight）

**初始值**：
- Bonded: 1.0（固定，主要贡献）
- Angle: 0.5（可学习）
- Dihedral: 0.3（可学习）
- Non-bonded: 0.2（可学习）

训练过程中，这些权重会根据任务自动调整。

---

## 🔧 技术细节

### 兼容性处理

1. **PyTorch Geometric scatter**：
   - 使用 `from torch_geometric.utils import scatter`
   - 避免了 `torch_scatter` 的编译问题

2. **可选组件**：
   - Improved layers (BesselBasis, PolynomialCutoff) 可选
   - 降级到基础实现（Gaussian RBF）

3. **词汇表集成**：
   ```python
   from amber_vocabulary import get_global_encoder
   encoder = get_global_encoder()

   model = RNAPocketEncoderV2(
       num_atom_types=encoder.num_atom_types,  # 71
       num_residues=encoder.num_residues,      # 43
       ...
   )
   ```

---

## 📊 模型参数统计

测试配置：
```python
RNAPocketEncoderV2(
    num_atom_types=71,
    num_residues=43,
    hidden_irreps="32x0e + 16x1o + 8x2e",
    output_dim=512,
    num_layers=3,
    use_multi_hop=True,
    use_nonbonded=True
)
```

**参数量：** ~2,696,516
- 包含 3 个可学习组合权重（angle_weight, dihedral_weight, nonbonded_weight）

**输入/输出：**
- Input: `[num_atoms, 4]`
- Output: `[batch_size, 512]`

**性能：**
- ✅ 单图前向传播: 正常
- ✅ 批处理: 正常
- ✅ E(3) 等变性: 保持
- ✅ 可学习权重: 自适应组合

---

## 🚀 使用方法

### 基础使用

```python
from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
from amber_vocabulary import get_global_encoder

# 获取词汇表大小
encoder = get_global_encoder()

# 创建模型
model = RNAPocketEncoderV2(
    num_atom_types=encoder.num_atom_types,
    num_residues=encoder.num_residues,
    hidden_irreps="32x0e + 16x1o + 8x2e",
    output_dim=512,
    num_layers=4,
    use_multi_hop=True,      # 启用多跳路径
    use_nonbonded=True,      # 启用非键交互
    pooling_type='attention'
)

# 前向传播
output = model(data)  # data 来自 build_dataset
```

### 数据要求

确保 `data` 对象包含：
- **必需**:
  - `x`: `[num_atoms, 4]`
  - `pos`: `[num_atoms, 3]`
  - `edge_index`: `[2, num_bonds]`
  - `edge_attr`: `[num_bonds, 2]`

- **可选（推荐）**:
  - `triple_index`: `[3, num_angles]`
  - `triple_attr`: `[num_angles, 2]`
  - `quadra_index`: `[4, num_dihedrals]`
  - `quadra_attr`: `[num_dihedrals, 3]`
  - `nonbonded_edge_index`: `[2, num_nb]`
  - `nonbonded_edge_attr`: `[num_nb, 3]`

---

## 🔍 关键改进点

### 与 v1.0 的对比

| 特性 | v1.0 | v2.0 |
|------|------|------|
| 输入特征 | 连续向量 | Embedding + 连续 |
| 消息传递 | 1-hop only | 1/2/3-hop 多跳 |
| 非键交互 | ❌ | ✅ LJ 参数 |
| 物理参数 | 部分使用 | 完全整合 |
| 可扩展性 | 中等 | 高 |

### 优势

1. **更丰富的特征表示**：
   - Embedding 捕获离散特征的语义
   - 连续特征保留物理意义

2. **更完整的分子建模**：
   - 多跳路径捕获长程依赖
   - 非键交互建模空间效应

3. **更强的物理基础**：
   - 所有 AMBER 力场参数被整合
   - 更接近真实分子力学

4. **更好的泛化能力**：
   - 固定词汇表确保一致性
   - 模块化设计易于扩展

---

## 📝 下一步建议

### 模型优化

1. **超参数调优**：
   - 多跳加权系数（目前 0.5, 0.3, 0.2）
   - hidden_irreps 配置
   - num_layers 数量

2. **训练策略**：
   - 渐进式训练（先 1-hop，后多跳）
   - 注意力权重正则化
   - Dropout 调整

3. **架构扩展**：
   - 添加交叉注意力机制
   - 实现 FFiNet 风格的 attention
   - 引入图级特征

### 实验验证

1. **消融实验**：
   - 逐步关闭多跳/非键交互
   - 评估各组件贡献

2. **性能测试**：
   - 计算时间分析
   - 内存占用优化

3. **泛化测试**：
   - 跨数据集验证
   - 不同 RNA 类型测试

---

## 📁 文件清单

新增文件：
- `models/e3_gnn_encoder_v2.py` - 完整的 v2 模型实现
- `MODELS_V2_SUMMARY.md` - 本文档

修改文件：
- 无（保留了所有原有文件）

测试通过：
- ✅ 基础前向传播
- ✅ 批处理
- ✅ 词汇表集成
- ✅ 多跳路径处理
- ✅ 非键交互

---

## ⚠️ 注意事项

1. **依赖要求**：
   - `e3nn >= 0.5.0`
   - `torch-geometric >= 2.0.0`
   - 无需 `torch-scatter`（使用 PyG 内置）

2. **内存考虑**：
   - 多跳路径会增加内存占用
   - 建议根据 GPU 内存调整 batch size

3. **训练建议**：
   - 初期可关闭多跳/非键（`use_multi_hop=False, use_nonbonded=False`）
   - 确认基础模型收敛后再启用

---

**完成时间**: 2025-10-25
**版本**: v2.0
**状态**: ✅ 测试通过，可用于训练
