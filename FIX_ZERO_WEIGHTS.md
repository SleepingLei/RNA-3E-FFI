# 修复可学习权重归零问题

## 🔍 问题诊断

你遇到的问题：
```
Train Loss: 199.524375       # 损失很高
  Angle weight: 0.0000       # 变成0了！
  Dihedral weight: 0.0000    # 变成0了！
  Nonbonded weight: 0.0771   # 只有这个有值
```

**根本原因**：使用了 **v1.0 旧格式数据**训练 **v2.0 新模型**

---

## ❌ 当前数据问题

运行诊断发现：
```bash
python diagnose_weights.py
```

输出：
```
节点特征: torch.Size([443, 11])  # ❌ 应该是 4 维！
2-hop angles: 缺失 ❌
3-hop dihedrals: 缺失 ❌
Non-bonded edges: 缺失 ❌
```

**v1.0 vs v2.0 数据格式**：

| 项目 | v1.0（旧） | v2.0（新） |
|-----|-----------|-----------|
| 特征维度 | 11 | 4 |
| triple_index | ❌ 无 | ✅ 必需 |
| quadra_index | ❌ 无 | ✅ 必需 |
| nonbonded_edge_index | ❌ 无 | ✅ 必需 |

---

## 💡 为什么权重变成0？

### 训练过程：

1. **前向传播**：
   ```python
   # 模型期待 triple_index
   if hasattr(data, 'triple_index'):
       h_angle = self.angle_mp_layers[i](h, triple_index, triple_attr)
   else:
       h_angle = 0  # 数据缺失 → 返回 0

   # 角度贡献为 0
   h_new = h_bonded + angle_weight * 0  # = h_bonded
   ```

2. **反向传播**：
   ```python
   # angle_weight 对损失没有影响
   # 梯度 ≈ 0 或微小负值
   angle_weight.grad ≈ -0.0001
   ```

3. **优化器更新**：
   ```python
   # Adam 优化器
   angle_weight -= lr * grad
   # 0.5 - 0.0001 = 0.4999
   # 0.4999 - 0.0001 = 0.4998
   # ...
   # 经过多轮 → 0.0000
   ```

**结果**：权重被优化到0，因为它对损失没有贡献！

---

## ✅ 解决方案

### **方案 1：重新生成数据（强烈推荐）** ⭐⭐⭐

#### 步骤 1：备份旧数据
```bash
mv data/processed/graphs data/processed/graphs_v1_backup
mkdir -p data/processed/graphs
```

#### 步骤 2：重新生成 v2.0 格式数据
```bash
python scripts/03_build_dataset.py \
    --hariboss_csv hariboss/Complexes.csv \
    --amber_dir data/processed/amber \
    --output_dir data/processed/graphs \
    --distance_cutoff 5.0 \
    --num_workers 8
```

**预计时间**：根据数据量，10-60 分钟

#### 步骤 3：验证数据格式
```bash
python -c "
import torch
from pathlib import Path

graph_dir = Path('data/processed/graphs')
graph_files = list(graph_dir.glob('*.pt'))

if not graph_files:
    print('❌ 未找到图文件')
else:
    data = torch.load(graph_files[0])
    print(f'✅ 文件: {graph_files[0].name}')
    print(f'   特征维度: {data.x.shape[1]} (应该是4)')
    print(f'   Angles: {data.triple_index.shape[1] if hasattr(data, \"triple_index\") else \"缺失\"}')
    print(f'   Dihedrals: {data.quadra_index.shape[1] if hasattr(data, \"quadra_index\") else \"缺失\"}')
    print(f'   Non-bonded: {data.nonbonded_edge_index.shape[1] if hasattr(data, \"nonbonded_edge_index\") else \"缺失\"}')

    if data.x.shape[1] == 4 and hasattr(data, 'triple_index'):
        print('\\n✅✅✅ 数据格式正确！可以开始训练')
    else:
        print('\\n❌ 数据格式仍然有问题')
"
```

#### 步骤 4：重新开始训练
```bash
python scripts/04_train_model.py \
    --graph_dir data/processed/graphs \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dir models/checkpoints_v2_correct \
    --batch_size 4 \
    --num_epochs 100 \
    --use_multi_hop \
    --use_nonbonded
```

**预期输出**：
```
Train Loss: 0.234567  # 损失正常
  Angle weight: 0.5123    # 有正常值
  Dihedral weight: 0.2987  # 有正常值
  Nonbonded weight: 0.2145  # 有正常值
```

---

### **方案 2：临时禁用多跳（不推荐）**

如果无法重新生成数据，可以临时禁用多跳功能：

```bash
python scripts/04_train_model.py \
    --output_dir models/baseline_old_data \
    --batch_size 4 \
    --num_epochs 100
    # 不添加 --use_multi_hop 和 --use_nonbonded
```

**缺点**：
- ❌ 失去了 v2.0 的核心功能
- ❌ 退化成 baseline 模型
- ❌ 性能会比完整的 v2.0 差

---

### **方案 3：修复权重约束（防止再次发生）**

为了防止将来再次出现权重归零的问题，可以修改模型代码，添加权重约束。

创建改进的模型版本：

```python
# models/e3_gnn_encoder_v2_fixed.py

class RNAPocketEncoderV2Fixed(RNAPocketEncoderV2):
    """
    添加权重约束的版本

    使用 sigmoid 或 softplus 确保权重非负
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 使用 log-space 参数化
        # weight = exp(log_weight) 确保 weight > 0
        if hasattr(self, 'angle_weight'):
            self.angle_log_weight = nn.Parameter(self.angle_weight.log())
            del self.angle_weight

        if hasattr(self, 'dihedral_weight'):
            self.dihedral_log_weight = nn.Parameter(self.dihedral_weight.log())
            del self.dihedral_weight

        if hasattr(self, 'nonbonded_weight'):
            self.nonbonded_log_weight = nn.Parameter(self.nonbonded_weight.log())
            del self.nonbonded_weight

    @property
    def angle_weight(self):
        """确保权重为正"""
        return self.angle_log_weight.exp()

    @property
    def dihedral_weight(self):
        """确保权重为正"""
        return self.dihedral_log_weight.exp()

    @property
    def nonbonded_weight(self):
        """确保权重为正"""
        return self.nonbonded_log_weight.exp()
```

**优点**：
- ✅ 权重永远 > 0
- ✅ 防止归零问题
- ✅ 数学上更稳定

---

## 🎯 推荐步骤（按顺序）

### 1️⃣ **立即执行**：重新生成数据

```bash
# 备份旧数据
mv data/processed/graphs data/processed/graphs_old

# 重新生成
python scripts/03_build_dataset.py
```

### 2️⃣ **验证数据**：

```bash
python diagnose_weights.py
```

应该看到：
```
✅ 数据格式正确，包含多跳路径
```

### 3️⃣ **重新训练**：

```bash
python scripts/04_train_model.py \
    --output_dir models/v2_correct \
    --use_multi_hop \
    --use_nonbonded
```

### 4️⃣ **监控训练**：

第一个 epoch 应该看到：
```
Train Loss: 0.5~2.0（正常范围）
  Angle weight: 0.4~0.6
  Dihedral weight: 0.2~0.4
  Nonbonded weight: 0.1~0.3
```

如果权重还是变成0 → 检查数据是否真的包含多跳路径

---

## 🔍 调试检查清单

- [ ] 数据特征维度是 4（不是 11）
- [ ] 数据包含 `triple_index`
- [ ] 数据包含 `quadra_index`
- [ ] 数据包含 `nonbonded_edge_index`
- [ ] `triple_index.shape[1] > 0`（不是空的）
- [ ] `quadra_index.shape[1] > 0`（不是空的）
- [ ] 训练时 `--use_multi_hop` 和 `--use_nonbonded` 都启用
- [ ] 第一个 epoch 后权重不为 0

---

## 📞 需要帮助？

如果重新生成数据后问题仍然存在：

1. **检查数据生成日志**：
   ```bash
   python scripts/03_build_dataset.py 2>&1 | tee build_log.txt
   ```
   查看是否有错误或警告

2. **检查单个样本**：
   ```bash
   python diagnose_weights.py
   ```

3. **运行快速测试**：
   ```bash
   python test_v2_model.py
   ```

---

**最后一次强调**：
- ⭐ **必须重新生成数据**
- ⭐ **验证数据格式正确**
- ⭐ **然后才能正常训练**

不要试图用 v1.0 数据训练 v2.0 模型！
