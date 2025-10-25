# 多跳消息传递实现详解

## 🎯 总体设计

### 实现方式
**混合架构**: RNA-3E-FFI 数据结构 + E(3)-GNN 消息传递 + 自定义多跳处理

```
数据层面: RNA-3E-FFI (triple_index, quadra_index)
         ↓
消息传递: E(3)-GNN (spherical harmonics, tensor products)
         ↓
多跳处理: 自定义 (AngleMessagePassing, DihedralMessagePassing)
         ↓
融合策略: 可学习权重
```

---

## 📐 架构对比

### 1. FFINet 原始架构
```
FFINet (Field-Induced Interaction Network):
├── Multi-hop paths as explicit attention paths
├── Attention over 1/2/3-hop neighbors
├── Graph Transformer-style architecture
└── Not E(3)-equivariant
```

**特点**:
- 使用 Transformer attention 机制
- 显式处理多跳路径的注意力
- 不保证 E(3) 等变性
- 参考: GemNet, DimeNet++

### 2. 我的实现（v2.0）
```
E(3)-GNN + Multi-hop:
├── 1-hop: E(3)-equivariant message passing (bonds)
├── 2-hop: Scalar-based angle aggregation
├── 3-hop: Scalar-based dihedral aggregation
├── Non-bonded: E(3)-equivariant spatial interaction
└── Learnable combination weights
```

**特点**:
- 保持 E(3) 等变性（1-hop, non-bonded）
- 多跳路径用标量处理（不破坏等变性）
- 更轻量级的实现
- 可学习的融合策略

---

## 🔧 详细实现

### 1-hop: 标准 E(3)-GNN 消息传递

**来源**: 完全基于 e3nn + E(3)-GNN 理论

```python
class E3GNNMessagePassingLayer(MessagePassing):
    """
    标准的 E(3) 等变消息传递

    核心思想：
    - 使用球谐函数 (spherical harmonics) 编码方向信息
    - 张量积 (tensor product) 保持等变性
    - 径向基函数 (radial basis) 编码距离信息
    """

    def message(self, x_j, pos_i, pos_j, edge_attr):
        # 1. 计算相对位置向量
        rel_pos = pos_i - pos_j  # [num_edges, 3]
        distance = ||rel_pos||

        # 2. 球谐函数编码方向 (E(3)-equivariant)
        sh = SphericalHarmonics(rel_pos / distance)  # Y^l_m(r̂)

        # 3. 径向基函数编码距离
        rbf = BesselBasis(distance)

        # 4. 径向 MLP 结合物理参数
        radial_input = [rbf, edge_attr]  # [RBF, r_eq, k]
        weights = MLP(radial_input)

        # 5. 张量积: h_j ⊗ Y^l_m → h_message
        message = TensorProduct(x_j, sh, weights)

        return message  # E(3)-equivariant!
```

**物理意义**:
- `edge_attr = [r_eq, k]`: 键长平衡位置和力常数
- 径向基函数模拟键的弹性势能
- 方向信息通过球谐函数保持等变性

---

### 2-hop: 角度路径处理

**来源**: **自定义设计**（受 FFINet 启发，但实现不同）

```python
class AngleMessagePassing(nn.Module):
    """
    2-hop 角度路径: i -> j -> k

    设计思想：
    - 不使用 E(3)-equivariant 操作（计算量大）
    - 仅使用标量特征（从 irreps 中提取）
    - 通过 MLP 聚合路径信息
    """

    def forward(self, x, triple_index, triple_attr):
        # triple_index: [3, num_angles] = [i, j, k]
        # triple_attr: [num_angles, 2] = [theta_eq, k]

        # 1. 提取标量特征（只用 l=0 部分）
        x_scalar = x[:, :scalar_dim]  # [num_atoms, scalar_dim]

        # 2. 获取路径端点特征
        i, j, k = triple_index[0], triple_index[1], triple_index[2]
        x_i = x_scalar[i]  # 起始原子
        x_k = x_scalar[k]  # 终止原子

        # 3. 结合角度物理参数
        angle_input = concat([x_i, x_k, triple_attr])
        # shape: [num_angles, 2*scalar_dim + 2]

        # 4. MLP 处理角度信息
        angle_messages = MLP(angle_input)
        # shape: [num_angles, scalar_dim]

        # 5. 聚合到中心原子 j
        angle_features = scatter(angle_messages, j, reduce='mean')
        # shape: [num_atoms, scalar_dim]

        # 6. 投影回 irreps 空间
        output = Linear(angle_features)  # scalar -> irreps

        return output
```

**为什么这样设计？**

1. **效率考虑**:
   - 完全等变的角度处理需要 Clebsch-Gordan 系数
   - 计算复杂度高：O(l³)
   - 对 2-hop 路径，标量处理已经足够

2. **物理意义**:
   - `triple_attr = [theta_eq, k]`: 角度平衡值和力常数
   - 模拟角度弯曲势能：V = k(θ - θ_eq)²
   - 端点原子特征反映角度的化学环境

3. **与 FFINet 的区别**:
   - FFINet: 使用 attention 机制，query = x_j, key/value = x_i, x_k
   - 我的实现: 直接 MLP 融合，更简单直接

---

### 3-hop: 二面角路径处理

**来源**: **自定义设计**（类似角度，但聚合到两个中心原子）

```python
class DihedralMessagePassing(nn.Module):
    """
    3-hop 二面角路径: i -> j -> k -> l

    设计思想：
    - 类似角度处理，但路径更长
    - 聚合到两个中心原子 j 和 k
    """

    def forward(self, x, quadra_index, quadra_attr):
        # quadra_index: [4, num_dihedrals] = [i, j, k, l]
        # quadra_attr: [num_dihedrals, 3] = [phi_k, per, phase]

        # 1. 提取标量特征
        x_scalar = x[:, :scalar_dim]

        # 2. 获取路径端点特征
        i, j, k, l = quadra_index[0], quadra_index[1], quadra_index[2], quadra_index[3]
        x_i = x_scalar[i]  # 起始原子
        x_l = x_scalar[l]  # 终止原子

        # 3. 结合二面角物理参数
        dihedral_input = concat([x_i, x_l, quadra_attr])
        # quadra_attr: [phi_k, periodicity, phase]

        # 4. MLP 处理二面角信息
        dihedral_messages = MLP(dihedral_input)

        # 5. 聚合到两个中心原子 j 和 k（平均）
        dihedral_j = scatter(dihedral_messages, j, reduce='mean')
        dihedral_k = scatter(dihedral_messages, k, reduce='mean')
        dihedral_features = (dihedral_j + dihedral_k) / 2

        # 6. 投影回 irreps 空间
        output = Linear(dihedral_features)

        return output
```

**物理意义**:
- `quadra_attr = [phi_k, per, phase]`: 二面角势能参数
- AMBER 二面角势能: V = phi_k * (1 + cos(per*φ - phase))
- per (periodicity): 旋转周期性（1, 2, 3, ...）
- phase: 相位偏移

**与 FFINet 的区别**:
- FFINet: 可能使用 multi-head attention over paths
- 我的实现: 简化为端点特征聚合

---

### 非键交互: E(3)-GNN 消息传递

**来源**: 完全基于 E(3)-GNN（与 1-hop 相同）

```python
# 使用相同的 E3GNNMessagePassingLayer
self.nonbonded_mp_layers = nn.ModuleList([
    E3GNNMessagePassingLayer(
        edge_attr_dim=3,  # [LJ_A, LJ_B, distance]
        use_sc=False,     # 非键不需要自连接
        ...
    )
    for _ in range(num_layers)
])
```

**物理意义**:
- LJ 势能: V = LJ_A/r¹² - LJ_B/r⁶
- 排斥项 (r⁻¹²) 和吸引项 (r⁻⁶)
- 真实参数从 AMBER prmtop 提取

---

## 🔄 融合策略

### FFINet 的方式
```python
# FFINet: 使用 attention 融合多跳信息
attention_weights = softmax(Q @ K^T / sqrt(d))
output = attention_weights @ V

# 不同 hop 的信息通过 attention mask 控制
```

### 我的方式（可学习权重）
```python
# 直接加权融合（更简单，更可解释）
h_new = h_bonded + \
        self.angle_weight * h_angle + \
        self.dihedral_weight * h_dihedral + \
        self.nonbonded_weight * h_nonbonded

# 权重是可学习参数
self.angle_weight = nn.Parameter(torch.tensor(0.5))
self.dihedral_weight = nn.Parameter(torch.tensor(0.3))
self.nonbonded_weight = nn.Parameter(torch.tensor(0.2))
```

**优势**:
- ✅ 更简单、更高效
- ✅ 可解释性强（可以查看学习到的权重）
- ✅ 参数少（仅 3 个参数）
- ✅ 训练稳定

**劣势**:
- ❌ 灵活性略低（相比 attention）
- ❌ 不能建模跨层交互

---

## 📊 数据流

### 完整的 Forward Pass

```python
def forward(self, data):
    # 0. 输入嵌入
    h = self.input_embedding(data.x)  # [num_atoms, 4] -> [num_atoms, irreps_dim]

    # 1-4 层消息传递
    for layer in range(num_layers):

        # === 1-hop: E(3)-equivariant ===
        h_bonded = self.bonded_mp_layers[layer](
            h, pos, edge_index, edge_attr=[r_eq, k]
        )  # E(3)-equivariant

        # === 2-hop: Scalar ===
        h_angle = self.angle_mp_layers[layer](
            h, triple_index, triple_attr=[theta_eq, k]
        )  # Scalar -> irreps

        # === 3-hop: Scalar ===
        h_dihedral = self.dihedral_mp_layers[layer](
            h, quadra_index, quadra_attr=[phi_k, per, phase]
        )  # Scalar -> irreps

        # === Non-bonded: E(3)-equivariant ===
        h_nonbonded = self.nonbonded_mp_layers[layer](
            h, pos, nonbonded_edge_index, nonbonded_attr=[LJ_A, LJ_B, dist]
        )  # E(3)-equivariant

        # === 融合 (可学习权重) ===
        h = h_bonded + \
            angle_weight * h_angle + \
            dihedral_weight * h_dihedral + \
            nonbonded_weight * h_nonbonded

    # 5. 提取标量 + 池化
    h_scalar = h[:, :scalar_dim]
    graph_embed = attention_pooling(h_scalar, batch)

    # 6. 输出投影
    output = MLP(graph_embed)  # [batch, output_dim]

    return output
```

---

## 🆚 与其他方法对比

### 1. 纯 E(3)-GNN (e3nn)
```
✅ 完全 E(3)-equivariant
❌ 只有 1-hop
❌ 计算量大
```

### 2. FFINet
```
✅ 多跳信息
✅ Attention 机制
❌ 不保证等变性
❌ 计算量大（attention 复杂度 O(n²)）
```

### 3. 我的实现
```
✅ 多跳信息 (1/2/3-hop + 非键)
✅ 1-hop 和非键保持等变性
✅ 计算高效（标量处理 2/3-hop）
✅ 可学习融合权重
⚠️ 2/3-hop 不完全等变（权衡）
```

---

## 🎯 设计哲学

### 核心思想
**"关键路径等变，辅助路径标量"**

1. **主要贡献（1-hop bonds）**: 完全 E(3)-equivariant
   - 最重要的化学信息
   - 值得付出计算代价

2. **次要贡献（2/3-hop）**: 标量处理
   - 角度、二面角的贡献相对较小
   - 标量处理已经足够，且高效

3. **长程贡献（非键）**: E(3)-equivariant
   - 空间信息重要
   - 保持等变性

### 权衡考虑

| 方面 | 完全等变 | 我的方案 | 理由 |
|-----|---------|---------|-----|
| 1-hop bonds | ✅ | ✅ | 最重要，必须等变 |
| 2-hop angles | ✅ | ❌ | 效率优先 |
| 3-hop dihedrals | ✅ | ❌ | 效率优先 |
| Non-bonded | ✅ | ✅ | 空间信息重要 |
| 计算复杂度 | High | Medium | 实用性 |

---

## 📈 性能分析

### 计算复杂度（单层）

**完全 E(3)-equivariant (all paths)**:
```
1-hop: O(E₁ * l³ * d²)
2-hop: O(E₂ * l³ * d²)  # E₂ >> E₁
3-hop: O(E₃ * l³ * d²)  # E₃ >> E₂
Total: O((E₁ + E₂ + E₃) * l³ * d²) ≈ O(E₃ * l³ * d²)
```
- E: 边数
- l: 最大角动量
- d: 特征维度

**我的实现**:
```
1-hop (E3): O(E₁ * l³ * d²)
2-hop (Scalar): O(E₂ * d²)
3-hop (Scalar): O(E₃ * d²)
Non-bonded (E3): O(E_nb * l³ * d²)
Total: O((E₁ + E_nb) * l³ * d² + (E₂ + E₃) * d²)
```

**加速比**: 约 **2-3x**（对于 l=2, E₃ ≈ 2E₁）

---

## 🔬 实验验证建议

### 消融实验
```python
# 1. Baseline (1-hop only)
model = RNAPocketEncoderV2(use_multi_hop=False, use_nonbonded=False)

# 2. + 2-hop angles
model = RNAPocketEncoderV2(use_multi_hop=True, use_nonbonded=False)

# 3. + 3-hop dihedrals
model = RNAPocketEncoderV2(use_multi_hop=True, use_nonbonded=False)

# 4. + Non-bonded
model = RNAPocketEncoderV2(use_multi_hop=True, use_nonbonded=True)
```

### 权重分析
```python
# 训练后检查学习到的权重
print(f"Angle weight: {model.angle_weight.item():.3f}")
print(f"Dihedral weight: {model.dihedral_weight.item():.3f}")
print(f"Non-bonded weight: {model.nonbonded_weight.item():.3f}")
```

---

## 📚 参考文献

1. **E(3)-GNN**:
   - Geiger & Smidt, "e3nn: Euclidean Neural Networks", 2021

2. **FFINet**:
   - Inspired by field-induced interaction networks

3. **AMBER Force Field**:
   - Cornell et al., "A second generation force field", 1995

4. **Multi-hop GNN**:
   - Klicpera et al., "Directional Message Passing", 2020

---

## 🎓 总结

### 创新点
1. ✅ 混合架构：E(3)-GNN + 多跳路径
2. ✅ 效率优化：标量处理 2/3-hop
3. ✅ 可学习融合：自适应权重
4. ✅ 物理完整：所有 AMBER 参数

### 适用场景
- ✅ RNA-蛋白质相互作用
- ✅ 需要 E(3) 等变性
- ✅ 计算资源有限
- ✅ 需要可解释性

### 不适用场景
- ❌ 需要完全等变性（所有路径）
- ❌ 数据量极大（attention 更灵活）
- ❌ 需要跨层交互建模

---

**实现哲学**: "Simple is better than complex, but complete is better than incomplete."

我的设计在 **完整性（多跳）、效率（标量处理）、等变性（关键路径）** 之间找到了平衡。
