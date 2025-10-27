#!/usr/bin/env python3
"""
诊断可学习权重问题

检查：
1. 数据是否包含多跳路径
2. 权重初始化是否正确
3. 权重梯度是否正常
"""
import sys
from pathlib import Path
import torch
from torch_geometric.data import Data, Batch
import glob

sys.path.insert(0, str(Path(__file__).parent))
from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
from scripts.amber_vocabulary import get_global_encoder


def check_data_format():
    """检查数据是否包含多跳路径"""
    print("=" * 80)
    print("检查数据格式")
    print("=" * 80)

    graph_dir = Path("data/processed/graphs")
    graph_files = list(graph_dir.glob("*.pt"))

    if not graph_files:
        print("❌ 未找到图数据文件")
        return False

    print(f"找到 {len(graph_files)} 个图文件")

    # 检查前5个文件
    has_multi_hop = True
    for i, graph_file in enumerate(graph_files[:5]):
        print(f"\n检查文件 {i+1}: {graph_file.name}")
        data = torch.load(graph_file)

        print(f"  节点特征: {data.x.shape}")
        print(f"  1-hop edges: {data.edge_index.shape[1]}")

        # 检查多跳路径
        if hasattr(data, 'triple_index'):
            print(f"  2-hop angles: {data.triple_index.shape[1]} ✅")
        else:
            print(f"  2-hop angles: 缺失 ❌")
            has_multi_hop = False

        if hasattr(data, 'quadra_index'):
            print(f"  3-hop dihedrals: {data.quadra_index.shape[1]} ✅")
        else:
            print(f"  3-hop dihedrals: 缺失 ❌")
            has_multi_hop = False

        if hasattr(data, 'nonbonded_edge_index'):
            print(f"  Non-bonded edges: {data.nonbonded_edge_index.shape[1]} ✅")
        else:
            print(f"  Non-bonded edges: 缺失 ❌")

    print("\n" + "=" * 80)
    if has_multi_hop:
        print("✅ 数据格式正确，包含多跳路径")
    else:
        print("❌ 数据缺少多跳路径！需要重新生成数据")
        print("运行: python scripts/03_build_dataset.py")
    print("=" * 80)

    return has_multi_hop


def check_model_initialization():
    """检查模型初始化"""
    print("\n" + "=" * 80)
    print("检查模型初始化")
    print("=" * 80)

    encoder = get_global_encoder()

    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=3,
        use_multi_hop=True,
        use_nonbonded=True
    )

    print(f"\n可学习权重初始值:")
    print(f"  angle_weight: {model.angle_weight.item():.4f}")
    print(f"  dihedral_weight: {model.dihedral_weight.item():.4f}")
    print(f"  nonbonded_weight: {model.nonbonded_weight.item():.4f}")

    # 检查权重是否需要梯度
    print(f"\n权重是否需要梯度:")
    print(f"  angle_weight.requires_grad: {model.angle_weight.requires_grad}")
    print(f"  dihedral_weight.requires_grad: {model.dihedral_weight.requires_grad}")
    print(f"  nonbonded_weight.requires_grad: {model.nonbonded_weight.requires_grad}")

    # 检查权重是否在优化器中
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=1e-4)

    weight_params = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.numel() == 1:
            weight_params.append(name)

    print(f"\n找到的可学习权重参数:")
    for name in weight_params:
        print(f"  {name}")

    return model


def test_forward_pass(model):
    """测试前向传播和梯度"""
    print("\n" + "=" * 80)
    print("测试前向传播和梯度")
    print("=" * 80)

    # 创建测试数据
    encoder = get_global_encoder()
    num_nodes = 50

    x = torch.zeros(num_nodes, 4)
    x[:, 0] = torch.randint(1, encoder.num_atom_types + 1, (num_nodes,)).float()
    x[:, 1] = torch.randn(num_nodes) * 0.5
    x[:, 2] = torch.randint(1, encoder.num_residues + 1, (num_nodes,)).float()
    x[:, 3] = torch.randint(1, 20, (num_nodes,)).float()

    data = Data(
        x=x,
        pos=torch.randn(num_nodes, 3),
        edge_index=torch.randint(0, num_nodes, (2, 150)),
        edge_attr=torch.randn(150, 2).abs() + 0.1,
        triple_index=torch.randint(0, num_nodes, (3, 80)),
        triple_attr=torch.randn(80, 2).abs() + 0.1,
        quadra_index=torch.randint(0, num_nodes, (4, 40)),
        quadra_attr=torch.randn(40, 3),
        nonbonded_edge_index=torch.randint(0, num_nodes, (2, 100)),
        nonbonded_edge_attr=torch.cat([
            torch.randn(100, 2).abs(),
            torch.rand(100, 1) * 6.0
        ], dim=-1)
    )

    # 前向传播
    model.train()
    output = model(data)

    # 计算损失
    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)

    print(f"\n前向传播:")
    print(f"  输出形状: {output.shape}")
    print(f"  损失: {loss.item():.6f}")

    # 反向传播
    loss.backward()

    # 检查梯度
    print(f"\n梯度检查:")
    print(f"  angle_weight.grad: {model.angle_weight.grad}")
    print(f"  dihedral_weight.grad: {model.dihedral_weight.grad}")
    print(f"  nonbonded_weight.grad: {model.nonbonded_weight.grad}")

    # 检查梯度是否为0
    if model.angle_weight.grad is None or model.angle_weight.grad.item() == 0:
        print("\n⚠️  警告: angle_weight 的梯度为 0 或 None!")
        print("   可能原因: triple_index 为空或消息传递有问题")

    if model.dihedral_weight.grad is None or model.dihedral_weight.grad.item() == 0:
        print("\n⚠️  警告: dihedral_weight 的梯度为 0 或 None!")
        print("   可能原因: quadra_index 为空或消息传递有问题")

    return model


def test_with_real_data():
    """使用真实数据测试"""
    print("\n" + "=" * 80)
    print("使用真实数据测试")
    print("=" * 80)

    graph_dir = Path("data/processed/graphs")
    graph_files = list(graph_dir.glob("*.pt"))

    if not graph_files:
        print("❌ 未找到图数据文件")
        return

    # 加载一个真实样本
    data = torch.load(graph_files[0])
    print(f"使用文件: {graph_files[0].name}")

    # 检查数据
    print(f"\n数据检查:")
    print(f"  节点数: {data.x.shape[0]}")
    print(f"  特征维度: {data.x.shape[1]}")

    if data.x.shape[1] != 4:
        print(f"\n❌ 错误: 特征维度应该是4，实际是 {data.x.shape[1]}")
        print("   需要重新生成数据: python scripts/03_build_dataset.py")
        return

    print(f"  1-hop edges: {data.edge_index.shape[1]}")

    if hasattr(data, 'triple_index'):
        print(f"  2-hop angles: {data.triple_index.shape[1]}")
        if data.triple_index.shape[1] == 0:
            print("    ⚠️  警告: 角度路径数量为 0!")
    else:
        print(f"  2-hop angles: 缺失 ❌")

    if hasattr(data, 'quadra_index'):
        print(f"  3-hop dihedrals: {data.quadra_index.shape[1]}")
        if data.quadra_index.shape[1] == 0:
            print("    ⚠️  警告: 二面角路径数量为 0!")
    else:
        print(f"  3-hop dihedrals: 缺失 ❌")

    if hasattr(data, 'nonbonded_edge_index'):
        print(f"  Non-bonded edges: {data.nonbonded_edge_index.shape[1]}")
    else:
        print(f"  Non-bonded edges: 缺失 ❌")

    # 测试前向传播
    encoder = get_global_encoder()
    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        hidden_irreps="16x0e + 8x1o",  # 小模型快速测试
        output_dim=128,
        num_layers=2,
        use_multi_hop=True,
        use_nonbonded=True
    )

    print(f"\n初始权重:")
    print(f"  angle_weight: {model.angle_weight.item():.4f}")
    print(f"  dihedral_weight: {model.dihedral_weight.item():.4f}")
    print(f"  nonbonded_weight: {model.nonbonded_weight.item():.4f}")

    # 前向传播
    model.train()
    try:
        output = model(data)
        print(f"\n✅ 前向传播成功")
        print(f"  输出形状: {output.shape}")

        # 计算损失并反向传播
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        print(f"\n梯度:")
        print(f"  angle_weight.grad: {model.angle_weight.grad.item() if model.angle_weight.grad is not None else None}")
        print(f"  dihedral_weight.grad: {model.dihedral_weight.grad.item() if model.dihedral_weight.grad is not None else None}")
        print(f"  nonbonded_weight.grad: {model.nonbonded_weight.grad.item() if model.nonbonded_weight.grad is not None else None}")

    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有诊断"""
    print("\n" + "=" * 80)
    print("可学习权重问题诊断工具")
    print("=" * 80)

    # 1. 检查数据格式
    has_multi_hop = check_data_format()

    # 2. 检查模型初始化
    model = check_model_initialization()

    # 3. 测试前向传播和梯度
    test_forward_pass(model)

    # 4. 使用真实数据测试
    if has_multi_hop:
        test_with_real_data()

    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

    # 给出建议
    print("\n💡 问题排查建议:")
    print("1. 检查数据是否包含 triple_index 和 quadra_index")
    print("2. 检查这些索引是否为空（shape[1] == 0）")
    print("3. 如果为空，需要重新生成数据: python scripts/03_build_dataset.py")
    print("4. 检查权重是否有负梯度导致被优化到0")
    print("5. 考虑添加权重约束（例如限制在 [0, 1] 范围内）")


if __name__ == "__main__":
    main()
