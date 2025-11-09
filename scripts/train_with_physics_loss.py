#!/usr/bin/env python3
"""
训练脚本示例：使用物理约束loss

展示如何在MSE loss基础上添加物理约束正则化
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3
from models.improved_components import PhysicsConstraintLoss


def train_epoch_with_physics(
    model,
    loader,
    optimizer,
    device,
    target_embeddings,  # 目标embedding (配体)
    physics_loss_fn,
    physics_weight=0.1,  # 物理loss的权重
    grad_clip=0.0
):
    """
    训练一个epoch，使用MSE loss + 物理约束loss

    Args:
        model: RNAPocketEncoderV3模型
        loader: DataLoader
        optimizer: 优化器
        device: 设备
        target_embeddings: dict, {sample_id: target_embedding}
        physics_loss_fn: PhysicsConstraintLoss
        physics_weight: 物理loss的权重系数
        grad_clip: 梯度裁剪阈值

    Returns:
        epoch_metrics: dict with losses
    """
    model.train()

    total_mse_loss = 0.0
    total_physics_loss = 0.0
    total_combined_loss = 0.0

    physics_components = {
        'bond_energy': 0.0,
        'angle_energy': 0.0,
        'dihedral_energy': 0.0
    }

    num_batches = 0

    for batch_idx, data in enumerate(loader):
        data = data.to(device)

        # Forward pass
        pocket_embedding = model(data)  # [batch_size, 512]

        # ========== 1. 主任务loss (MSE) ==========
        # 获取target embeddings
        batch_size = pocket_embedding.size(0)
        target_emb_list = []
        for i in range(batch_size):
            # 假设data中有sample_id
            sample_id = data.sample_id[i] if hasattr(data, 'sample_id') else i
            target_emb = target_embeddings.get(sample_id, torch.zeros(512, device=device))
            target_emb_list.append(target_emb)

        target_emb = torch.stack(target_emb_list)  # [batch_size, 512]

        # MSE loss
        mse_loss = nn.MSELoss()(pocket_embedding, target_emb)

        # ========== 2. 物理约束loss (正则化) ==========
        # 注意: 物理loss使用data.pos，不需要梯度
        # 我们只是用它来约束学习的embedding更符合物理
        physics_loss, physics_dict = physics_loss_fn(data)

        # ========== 3. 组合loss ==========
        combined_loss = mse_loss + physics_weight * physics_loss

        # Backward
        optimizer.zero_grad()
        combined_loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # 记录
        total_mse_loss += mse_loss.item()
        total_physics_loss += physics_loss.item()
        total_combined_loss += combined_loss.item()

        for key in physics_components:
            if key in physics_dict:
                physics_components[key] += physics_dict[key]

        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)}: "
                  f"MSE={mse_loss.item():.4f}, "
                  f"Physics={physics_loss.item():.4f}, "
                  f"Total={combined_loss.item():.4f}")

    # 计算平均
    epoch_metrics = {
        'mse_loss': total_mse_loss / num_batches,
        'physics_loss': total_physics_loss / num_batches,
        'combined_loss': total_combined_loss / num_batches,
    }

    for key in physics_components:
        epoch_metrics[key] = physics_components[key] / num_batches

    return epoch_metrics


def main():
    """主训练流程示例"""

    # ========== 配置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========== 创建模型 ==========
    print("\n1. Creating model...")
    model = RNAPocketEncoderV3(
        output_dim=512,
        num_layers=4,
        use_geometric_mp=True,        # ✅ 使用几何增强MP
        use_enhanced_invariants=True,  # ✅ 使用增强不变量
        pooling_type='multihead_attention',  # ✅ 使用多头注意力
        num_attention_heads=4,
        dropout=0.1
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 创建物理约束loss ==========
    print("\n2. Creating physics constraint loss...")
    physics_loss_fn = PhysicsConstraintLoss(
        use_bond=True,       # ✅ 键伸缩能量
        use_angle=True,      # ✅ 键角弯曲能量
        use_dihedral=True,   # ✅ 二面角扭转能量
        use_nonbonded=False, # ❌ 非键能量（通常不用，计算量大）
        reduction='mean'
    )

    # ========== 优化器 ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # ========== 加载数据 (示例) ==========
    print("\n3. Loading data...")
    # 这里应该加载你的实际数据
    # train_loader = GeometricDataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = GeometricDataLoader(val_dataset, batch_size=32)

    # 示例：假设target_embeddings
    target_embeddings = {}
    # 在实际使用中，这应该是配体的embedding
    # target_embeddings[sample_id] = ligand_embedding

    # ========== 训练循环 ==========
    print("\n4. Training...")
    num_epochs = 100
    best_val_loss = float('inf')

    # 物理loss权重策略
    physics_weight_schedule = {
        0: 0.01,   # 初期很小
        10: 0.05,  # 逐渐增大
        20: 0.1,   # 稳定值
    }

    for epoch in range(num_epochs):
        # 动态调整physics loss权重
        physics_weight = physics_weight_schedule.get(
            epoch,
            physics_weight_schedule[max([k for k in physics_weight_schedule.keys() if k <= epoch])]
        )

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Physics loss weight: {physics_weight:.4f}")
        print(f"{'='*80}")

        # 训练
        # train_metrics = train_epoch_with_physics(
        #     model=model,
        #     loader=train_loader,
        #     optimizer=optimizer,
        #     device=device,
        #     target_embeddings=target_embeddings,
        #     physics_loss_fn=physics_loss_fn,
        #     physics_weight=physics_weight,
        #     grad_clip=1.0
        # )

        # print(f"\nTraining Metrics:")
        # print(f"  MSE Loss:      {train_metrics['mse_loss']:.4f}")
        # print(f"  Physics Loss:  {train_metrics['physics_loss']:.4f}")
        # print(f"  Combined Loss: {train_metrics['combined_loss']:.4f}")
        # print(f"  - Bond Energy:     {train_metrics['bond_energy']:.4f}")
        # print(f"  - Angle Energy:    {train_metrics['angle_energy']:.4f}")
        # print(f"  - Dihedral Energy: {train_metrics['dihedral_energy']:.4f}")

        # 验证 (不使用physics loss)
        # val_loss = evaluate(model, val_loader, device, target_embeddings)
        # print(f"\nValidation MSE Loss: {val_loss:.4f}")

        # # 学习率调度
        # scheduler.step(val_loss)

        # # 保存最佳模型
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': val_loss,
        #     }, 'best_model_v3.pt')
        #     print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)


def evaluate(model, loader, device, target_embeddings):
    """验证函数（不使用physics loss）"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            pocket_embedding = model(data)

            # 获取target
            batch_size = pocket_embedding.size(0)
            target_emb_list = []
            for i in range(batch_size):
                sample_id = data.sample_id[i] if hasattr(data, 'sample_id') else i
                target_emb = target_embeddings.get(sample_id, torch.zeros(512, device=device))
                target_emb_list.append(target_emb)

            target_emb = torch.stack(target_emb_list)

            loss = nn.MSELoss()(pocket_embedding, target_emb)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# ============================================================================
# 使用说明和最佳实践
# ============================================================================

"""
📖 使用指南

1. **物理loss权重调优**:
   - 初期 (epoch 0-10): 0.01-0.05 (让模型先学主任务)
   - 中期 (epoch 10-20): 0.05-0.1 (逐渐增加约束)
   - 后期 (epoch 20+): 0.1-0.2 (稳定正则化)

   如果physics_loss >> mse_loss:
   - 减小physics_weight
   - 或者对physics_loss各项加权

2. **何时使用物理loss**:
   ✅ 适用场景:
   - 训练数据较少时（帮助泛化）
   - 模型倾向过拟合时
   - 需要物理可解释性时

   ❌ 不适用:
   - 数据充足且质量高
   - 主任务loss已经很好收敛

3. **监控指标**:
   - 如果physics_loss持续下降，说明模型学到了物理规律 ✓
   - 如果physics_loss上升但mse_loss下降，可能需要调整权重
   - 观察验证集性能来判断是否过度正则化

4. **变体策略**:

   Strategy A - 仅前期使用:
   ```python
   if epoch < 20:
       physics_weight = 0.1
   else:
       physics_weight = 0.0  # 后期关闭
   ```

   Strategy B - 分项加权:
   ```python
   physics_loss = (
       1.0 * bond_energy +      # 键很重要
       0.5 * angle_energy +     # 角度次要
       0.3 * dihedral_energy    # 二面角最不重要
   )
   ```

   Strategy C - 只在特定样本使用:
   ```python
   # 对大分子使用更强的物理约束
   if data.num_nodes > 100:
       physics_weight = 0.2
   else:
       physics_weight = 0.05
   ```

5. **调试建议**:
   ```python
   # 打印各项能量，确保数值合理
   print(f"Bond energy: {bond_energy:.2f} kcal/mol")
   print(f"Angle energy: {angle_energy:.2f} kcal/mol")
   print(f"Dihedral energy: {dihedral_energy:.2f} kcal/mol")

   # 如果某项特别大，检查:
   # - 输入数据的单位是否正确
   # - 力常数是否异常
   # - 平衡值是否合理
   ```

6. **性能优化**:
   - 物理loss计算不需要反向传播到data.pos
   - 可以每N个batch计算一次physics loss
   ```python
   if batch_idx % 5 == 0:  # 每5个batch计算一次
       physics_loss, _ = physics_loss_fn(data)
   ```
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("This is a template script. Modify it for your actual use case.")
    print("="*80)
    # Uncomment to run:
    # main()
