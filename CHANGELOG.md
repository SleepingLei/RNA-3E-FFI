# 项目更新日志 | Changelog

## 2025-01-22 - 项目定位更正

### ✅ 重大更正
- **项目目标**: 从"结合位点预测"更正为"RNA-配体虚拟筛选"
- **核心任务**: 给定RNA口袋，从配体库中筛选出潜在结合配体

### 📝 更新的文档
1. `README.md` - 更新标题和概述
2. `PROJECT_OVERVIEW.md` - 更正项目简介
3. `USAGE.md` - 添加完整虚拟筛选流程
4. `docs/QUICK_START.md` - 添加项目目标说明
5. `docs/VIRTUAL_SCREENING_WORKFLOW.md` - 新增详细流程图和说明

### 🎯 明确的工作流程

**训练阶段**:
```
RNA-配体复合物 → 提取口袋 → 构建图
                ↓
              生成配体嵌入(Uni-Mol)
                ↓
         对比学习训练(E(3) GNN)
                ↓
            训练好的模型
```

**筛选阶段**:
```
查询RNA口袋 → E(3) GNN编码 → 口袋嵌入
                             ↓
配体库嵌入 ← 相似度计算 ← 口袋嵌入
     ↓
Top-K候选配体
```

### 🔑 关键概念澄清

| 概念 | 说明 |
|------|------|
| 共享嵌入空间 | RNA口袋和配体被映射到同一512维空间 |
| 对比学习 | 拉近配对的口袋-配体，推远不配对的 |
| E(3)等变性 | 口袋编码器对旋转平移保持不变性 |
| 虚拟筛选 | 通过相似度计算快速筛选大型配体库 |

---

## 2025-01-22 - 代码和文档整理

### ✅ 代码整理
- 归档7个旧版本/调试脚本到 `archive/old_scripts/`
- 重命名V2脚本为主脚本 `01_process_data.py`
- 新增环境检查脚本 `00_check_environment.py`

### ✅ 文档整理
- 归档4个开发文档到 `archive/old_docs/`
- 创建用户友好的文档结构
- 新增多个指南文档

### ✅ 核心改进
1. **完整残基选择**: 0% → 100%残基完整度
2. **末端原子清理**: 自动处理tleap类型错误
3. **鲁棒错误处理**: 预检查和自动回退
4. **测试验证**: 3/3复合物成功处理（100%）

---

## 2025-01-22 - 短期任务完成

### ✅ 新功能实现

#### 1. 配体参数化 (antechamber + GAFF)
- **新增函数**: `parameterize_ligand_gaff()` (`scripts/01_process_data.py:353-488`)
- **工作流程**:
  ```
  PDB → antechamber (GAFF2 atom types + AM1-BCC charges)
      → parmchk2 (missing parameters)
      → tleap (prmtop/inpcrd)
  ```
- **特性**:
  - 自动原子类型分配
  - AM1-BCC电荷计算（可配置）
  - 生成缺失力场参数
  - 完整AMBER拓扑输出

#### 2. 修饰RNA残基处理
- **新增函数**: `parameterize_modified_rna()` (`scripts/01_process_data.py:491-646`)
- **支持的修饰**: PSU, 5MU, 5MC, 1MA, 7MG, M2G, OMC, OMG, H2U, 2MG, M7G, OMU, YYG, YG, 6MZ, IU, I
- **策略**:
  - 每个修饰残基独立参数化（GAFF）
  - 使用tleap combine合并拓扑
  - 自动电荷计算

### 🔧 集成和测试
- 自动分类系统更新：识别标准RNA、修饰RNA、配体、蛋白
- 新增测试脚本：`scripts/test_new_features.py`
- 新增示例脚本：`scripts/demo_new_features.py`
- 详细文档：`docs/NEW_FEATURES.md`
- 更新README：反映新功能

### 📊 验证状态
```bash
✓ antechamber found
✓ parmchk2 found
✓ tleap found
✓ All dependencies ready
```

### 📁 输出文件扩展
现在每个复合物生成：
```
data/amber/
├── {pdb}_{lig}_rna.prmtop/inpcrd           # 标准RNA
├── {pdb}_{lig}_modified_rna.prmtop/inpcrd  # 修饰RNA
├── {pdb}_{lig}_ligand.prmtop/inpcrd        # 配体
└── {pdb}_{lig}_protein.prmtop/inpcrd       # 蛋白（如有）
```

---

## 未来计划

### 短期
- [x] ~~实现配体参数化（antechamber + GAFF）~~ ✅ 已完成
- [x] ~~处理修饰RNA残基~~ ✅ 已完成
- [ ] 完善训练脚本中的对比学习损失函数
- [ ] 添加评估指标（Hit Rate@K, Enrichment Factor）

### 中期
- [ ] 批量处理完整HARIBOSS数据集
- [ ] 优化训练超参数
- [ ] 构建标准配体库

### 长期
- [ ] 集成实验验证结果
- [ ] 发布预训练模型
- [ ] Web服务接口
