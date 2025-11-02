# 256维配体嵌入 - 快速参考

## 🎯 核心结论

```
✅ 配体嵌入降维完成：1536维 → 256维
✅ 信息保留：99.92%方差
✅ 参数减少：83.3%（输出层）
✅ 数据量充足：917个样本
✅ 模型无需瘦身：保持原架构
```

---

## ⚡ 快速开始

### 1. 使用降维后的嵌入训练

```bash
python scripts/04_train_model.py \
    --embeddings_path data/processed/ligand_embeddings_256d.h5 \
    --output_dim 256 \
    --hidden_irreps "32x0e + 16x1o + 8x2e" \
    --num_layers 4 \
    --batch_size 16 \
    --num_epochs 300
```

### 2. 对新配体降维（推理时）

```python
import pickle

# 加载PCA模型
with open('data/processed/pca_model_256d.pkl', 'rb') as f:
    pca = pickle.load(f)

# 降维
ligand_256d = pca.transform(ligand_1536d.reshape(1, -1))
```

---

## 📊 关键数据

### PCA分析结果

| 指标 | 值 |
|------|---|
| 原始维度 | 1536 |
| 降维后 | 256 |
| 方差保留 | 99.92% |
| 前10个PC | 90.84%方差 |
| 有效秩 | 37.4 |

### 维度需求

| 方差保留 | 所需维度 |
|----------|----------|
| 90% | 10 |
| 95% | 20 |
| 99% | 81 |
| 99.92% | **256** ✓ |

---

## 🔧 推荐配置

### 完整训练命令

```bash
python scripts/04_train_model.py \
    --embeddings_path data/processed/ligand_embeddings_256d.h5 \
    --output_dim 256 \
    --hidden_irreps "32x0e + 16x1o + 8x2e" \
    --num_layers 4 \
    --num_radial_basis 8 \
    --dropout 0.1 \
    --weight_decay 5e-6 \
    --use_gate \
    --use_layer_norm \
    --use_multi_hop \
    --use_nonbonded \
    --pooling_type attention \
    --batch_size 16 \
    --num_epochs 300 \
    --lr 1e-3 \
    --patience 30 \
    --output_dir models/checkpoints_256d
```

---

## 📁 生成的文件

```
data/processed/
├── ligand_embeddings_256d.h5      # 降维后的嵌入
├── pca_model_256d.pkl             # PCA模型（推理用）
└── pca_info_256d.txt              # PCA详细信息

analysis_results/
├── embedding_pca_analysis.png     # PCA可视化
└── pca_results.txt                # PCA详细结果
```

---

## ⚠️ 注意事项

1. **维度必须匹配**：
   - 训练时：`--embeddings_path` 和 `--output_dim` 必须一致
   - 推理时：必须使用相同的PCA模型

2. **不要混用**：
   ```bash
   # ❌ 错误
   --embeddings_path ligand_embeddings.h5  # 1536维
   --output_dim 256                         # 不匹配！

   # ✅ 正确
   --embeddings_path ligand_embeddings_256d.h5
   --output_dim 256
   ```

3. **检查点不兼容**：
   - 256维训练的模型不能加载1536维的检查点

---

## 📈 预期收益

| 指标 | 1536维 | 256维 | 改进 |
|------|--------|-------|------|
| 输出层参数 | 87.5K | 14.6K | -83.3% |
| 训练速度 | 基线 | +10% | ✓ |
| 内存占用 | 基线 | -28% | ✓ |
| 过拟合风险 | 高 | 低 | ✓ |
| 信息损失 | 0% | 0.08% | ≈0 |

---

## 🎯 下一步

1. ✅ 降维完成
2. → **训练256维模型**
3. → 对比性能（256维 vs 1536维）
4. → 更新推理脚本
5. → 部署使用

---

## 🔗 详细文档

- **完整指南**: `docs/embedding_reduction_guide.md`
- **归一化与PCA分析**: `docs/normalization_and_pca.md` ⭐ 新增
- **过拟合解决方案**: `docs/overfitting_solutions.md`
- **降维脚本**: `scripts/reduce_ligand_embeddings.py` (已更新：支持归一化)
- **验证脚本**: `scripts/verify_embedding_reduction.py` ⭐ 新增
- **分析脚本**: `scripts/analyze_embedding_dimensionality.py`
