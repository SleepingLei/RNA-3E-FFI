# 快速参考：多模型文件处理

## 文件命名速查

```
Pipeline 阶段                    文件名格式
─────────────────────────────────────────────────────────────
01_process_data.py      →  {pdb}_{lig}_model{N}_rna.pdb
                           {pdb}_{lig}_model{N}_rna.prmtop

02_embed_ligands.py     →  HDF5['{pdb}_{lig}']  ← 无 model 编号!

03_build_dataset.py     →  {pdb}_{lig}_model{N}.pt

04_train_model.py       →  读取: {pdb}_{lig}_model{N}.pt
                           映射到: HDF5['{pdb}_{lig}']

05_run_inference.py     →  任意 .pt 文件
```

## 常用命令

### 诊断问题
```bash
# 检查 prmtop 文件健康状况
python scripts/debug_prmtop_files.py --amber_dir data/processed/amber

# 分析参数化失败原因
python scripts/analyze_failed_parameterization.py

# 测试文件处理逻辑
python scripts/test_model_file_handling.py
```

### 运行 Pipeline
```bash
# 1. 图构建（多进程）
python scripts/03_build_dataset.py --num_workers 8

# 2. 训练
python scripts/04_train_model.py \
    --batch_size 16 \
    --num_epochs 100 \
    --num_workers 4

# 3. 推理
python scripts/05_run_inference.py \
    --checkpoint models/best_model.pt \
    --query_graph data/processed/graphs/1aju_ARG_model0.pt \
    --ligand_library data/processed/ligand_embeddings.h5
```

## 关键映射关系

```
Graph ID (文件名)           Embedding Key (HDF5)
─────────────────────────────────────────────────
1aju_ARG_model0        →    1aju_ARG
1aju_ARG_model1        →    1aju_ARG  ← 同一个 embedding
7ych_GTP_model0        →    7ych_GTP
1akx_ARG (无 model)    →    1akx_ARG
```

## 问题排查流程

```
遇到 "rna_pdb_not_found" 错误？
  ↓
1. 检查文件是否存在
   ls data/processed/amber/*{pdb_id}*_rna.pdb
  ↓
2. 文件存在但报错？
   → 检查是否有 _model{N} 后缀
   → 已修复：03/04/05 脚本都支持了
  ↓
3. 检查 prmtop 文件
   python scripts/debug_prmtop_files.py
  ↓
4. prmtop 是空文件？
   → 参数化失败，查看 TROUBLESHOOTING_PRMTOP.md
```

## 重要提示

⚠️ **Embedding 无 model 编号**
- 配体结构相同 → 所有 model 共享 embedding
- Graph ID 要映射到 base ID

✅ **向后兼容**
- 支持新格式: `{pdb}_{lig}_model{N}.pt`
- 支持旧格式: `{pdb}_{lig}.pt`

📈 **多模型优势**
- 增加训练数据
- 学习结构变化的不变性
- 提高泛化能力

## 文档索引

- `MODEL_FILE_NAMING.md` - 完整命名约定
- `TROUBLESHOOTING_PRMTOP.md` - prmtop 问题排查
- `SUMMARY_OF_FIXES.md` - 修复总结
