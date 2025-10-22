#!/usr/bin/env python3
"""
快速诊断和修复 splits.json 问题
"""
import glob
from pathlib import Path
import h5py
import json
import numpy as np

print("="*80)
print("诊断和修复 splits.json")
print("="*80)

# 步骤 1: 检查图文件
print("\n1. 检查图文件...")
graph_dir = Path("data/processed/graphs")
if not graph_dir.exists():
    print(f"   ❌ 目录不存在: {graph_dir}")
    exit(1)

graph_files = list(graph_dir.glob("*.pt"))
print(f"   ✓ 找到 {len(graph_files)} 个图文件")

if len(graph_files) == 0:
    print(f"   ❌ 没有图文件！请先运行: python scripts/03_build_dataset.py")
    exit(1)

print(f"   示例文件:")
for f in sorted(graph_files)[:5]:
    print(f"     - {f.name}")
if len(graph_files) > 5:
    print(f"     ... 还有 {len(graph_files) - 5} 个文件")

# 步骤 2: 检查 embedding 文件
print("\n2. 检查 ligand embeddings...")
embeddings_path = Path("data/processed/ligand_embeddings.h5")

if not embeddings_path.exists():
    print(f"   ❌ 文件不存在: {embeddings_path}")
    print(f"   请先运行: python scripts/02_embed_ligands.py")
    exit(1)

try:
    with h5py.File(str(embeddings_path), 'r') as f:
        embedding_keys = set(f.keys())
        print(f"   ✓ 找到 {len(embedding_keys)} 个 embedding")

        print(f"   示例键:")
        for k in sorted(embedding_keys)[:5]:
            print(f"     - {k}")
        if len(embedding_keys) > 5:
            print(f"     ... 还有 {len(embedding_keys) - 5} 个键")
except Exception as e:
    print(f"   ❌ 无法读取 HDF5 文件: {e}")
    exit(1)

# 步骤 3: 尝试匹配
print("\n3. 匹配图文件和 embeddings...")
valid_ids = []
unmatched = []

for graph_file in graph_files:
    graph_id = graph_file.stem  # e.g., "1aju_ARG_model0"

    # 提取 base ID (去掉 model 编号)
    if '_model' in graph_id:
        base_id = '_'.join(graph_id.split('_model')[0].split('_'))
    else:
        base_id = graph_id

    # 尝试匹配
    if base_id in embedding_keys:
        valid_ids.append(graph_id)
    elif graph_id in embedding_keys:
        valid_ids.append(graph_id)
    else:
        unmatched.append({
            'graph_id': graph_id,
            'base_id': base_id
        })

print(f"   ✓ 匹配成功: {len(valid_ids)}")
print(f"   ✗ 未匹配: {len(unmatched)}")

if unmatched:
    print(f"\n   未匹配的示例 (前5个):")
    for item in unmatched[:5]:
        print(f"     Graph ID: {item['graph_id']}")
        print(f"     Base ID:  {item['base_id']}")
        # 检查相似的键
        similar = [k for k in embedding_keys if item['base_id'].split('_')[0] in k]
        if similar:
            print(f"     可能的匹配: {similar[:3]}")
        print()

if len(valid_ids) == 0:
    print("\n❌ 错误: 没有找到任何匹配的样本！")
    print("\n可能的原因:")
    print("  1. Ligand embeddings 还没有生成")
    print("  2. 图文件和 embedding 的命名格式不一致")
    print("\n建议:")
    print("  1. 检查图文件命名: ls data/processed/graphs/*.pt | head -5")
    print("  2. 检查 embedding 键:")
    print("     python -c \"import h5py; f=h5py.File('data/processed/ligand_embeddings.h5','r'); print(list(f.keys())[:5])\"")
    print("  3. 如果命名不匹配，可能需要重新运行 02_embed_ligands.py")
    exit(1)

# 步骤 4: 创建数据集划分
print("\n4. 创建数据集划分...")
np.random.seed(42)
valid_ids_shuffled = valid_ids.copy()
np.random.shuffle(valid_ids_shuffled)

# 划分比例
train_ratio = 0.8
val_ratio = 0.1

n_train = int(len(valid_ids_shuffled) * train_ratio)
n_val = int(len(valid_ids_shuffled) * val_ratio)

train_ids = valid_ids_shuffled[:n_train]
val_ids = valid_ids_shuffled[n_train:n_train+n_val]
test_ids = valid_ids_shuffled[n_train+n_val:]

print(f"   训练集: {len(train_ids)} ({len(train_ids)/len(valid_ids)*100:.1f}%)")
print(f"   验证集: {len(val_ids)} ({len(val_ids)/len(valid_ids)*100:.1f}%)")
print(f"   测试集: {len(test_ids)} ({len(test_ids)/len(valid_ids)*100:.1f}%)")

# 步骤 5: 保存
print("\n5. 保存 splits...")
splits = {
    'train': train_ids,
    'val': val_ids,
    'test': test_ids
}

splits_file = Path("data/splits/splits.json")
splits_file.parent.mkdir(parents=True, exist_ok=True)

with open(splits_file, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"   ✓ 保存到: {splits_file}")

# 步骤 6: 显示示例
print("\n6. 示例样本:")
print(f"\n   训练集 (前5个):")
for id in train_ids[:5]:
    print(f"     - {id}")

if val_ids:
    print(f"\n   验证集 (前5个):")
    for id in val_ids[:5]:
        print(f"     - {id}")

if test_ids:
    print(f"\n   测试集 (前5个):")
    for id in test_ids[:5]:
        print(f"     - {id}")

print("\n" + "="*80)
print("✓ 修复完成！")
print("="*80)
print("\n现在可以运行:")
print("  python scripts/04_train_model.py --batch_size 16 --num_epochs 100")
