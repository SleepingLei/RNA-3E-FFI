#!/usr/bin/env python3
"""
诊断脚本: 检查模型权重和梯度问题

用法:
    python scripts/diagnose_gradient_issue.py --checkpoint models/checkpoints_v2_normalized_1351_4/best_model.pt
"""
import argparse
import json
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3


def diagnose_checkpoint(checkpoint_path, config_path=None):
    """诊断检查点文件"""
    print("=" * 80)
    print("🔍 Gradient & Weight Diagnostic Tool")
    print("=" * 80)

    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Show basic info
    print(f"\n📊 Checkpoint Information:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")

    # Extract state dict
    state_dict = checkpoint['model_state_dict']

    # Check for learnable weights
    print(f"\n🔎 Checking Learnable Weights:")
    print("-" * 80)

    # Old format (直接存储)
    if 'angle_weight' in state_dict:
        print("  ⚠️  OLD FORMAT DETECTED (无约束权重)")
        print(f"    angle_weight:     {state_dict['angle_weight'].item():.6f}")
        print(f"    dihedral_weight:  {state_dict['dihedral_weight'].item():.6f}")
        print(f"    nonbonded_weight: {state_dict['nonbonded_weight'].item():.6f}")

        # Check if weights are growing
        if state_dict['angle_weight'].item() > 1.0:
            print(f"    🚨 PROBLEM: angle_weight > 1.0 (无约束增长!)")
        if state_dict['dihedral_weight'].item() > 1.0:
            print(f"    🚨 PROBLEM: dihedral_weight > 1.0 (无约束增长!)")
        if state_dict['nonbonded_weight'].item() > 1.0:
            print(f"    🚨 PROBLEM: nonbonded_weight > 1.0 (无约束增长!)")

    # New format (log-space)
    elif 'log_angle_weight' in state_dict:
        print("  ✅ NEW FORMAT DETECTED (有约束权重)")
        log_angle = state_dict['log_angle_weight'].item()
        log_dihedral = state_dict['log_dihedral_weight'].item()
        log_nonbonded = state_dict['log_nonbonded_weight'].item()

        # Compute actual weights (sigmoid)
        angle_w = torch.sigmoid(torch.tensor(log_angle)).item()
        dihedral_w = torch.sigmoid(torch.tensor(log_dihedral)).item()
        nonbonded_w = torch.sigmoid(torch.tensor(log_nonbonded)).item()

        print(f"    Log-space parameters:")
        print(f"      log_angle_weight:     {log_angle:.6f} → weight={angle_w:.6f}")
        print(f"      log_dihedral_weight:  {log_dihedral:.6f} → weight={dihedral_w:.6f}")
        print(f"      log_nonbonded_weight: {log_nonbonded:.6f} → weight={nonbonded_w:.6f}")

        print(f"\n    Actual weights (sigmoid约束到 [0, 1]):")
        print(f"      angle_weight:     {angle_w:.6f}")
        print(f"      dihedral_weight:  {dihedral_w:.6f}")
        print(f"      nonbonded_weight: {nonbonded_w:.6f}")
    else:
        print("  ⚠️  No learnable weights found in checkpoint")

    # Check training history if available
    history_path = Path(checkpoint_path).parent / "training_history.json"
    if history_path.exists():
        print(f"\n📈 Loading training history: {history_path}")
        with open(history_path, 'r') as f:
            history = json.load(f)

        if 'learnable_weights' in history:
            weights_history = history['learnable_weights']

            print(f"\n📊 Weight Evolution Over Training:")
            print("-" * 80)

            # Analyze angle weight
            if 'angle_weight' in weights_history and weights_history['angle_weight']:
                angles = weights_history['angle_weight']
                print(f"  Angle Weight:")
                print(f"    Initial:  {angles[0]:.6f}")
                print(f"    Final:    {angles[-1]:.6f}")
                print(f"    Change:   {angles[-1] - angles[0]:+.6f}")
                print(f"    Max:      {max(angles):.6f}")
                print(f"    Min:      {min(angles):.6f}")

                # Check for growth
                if angles[-1] > 2 * angles[0]:
                    print(f"    🚨 PROBLEM: Weight doubled during training!")

            # Analyze dihedral weight
            if 'dihedral_weight' in weights_history and weights_history['dihedral_weight']:
                dihedrals = weights_history['dihedral_weight']
                print(f"\n  Dihedral Weight:")
                print(f"    Initial:  {dihedrals[0]:.6f}")
                print(f"    Final:    {dihedrals[-1]:.6f}")
                print(f"    Change:   {dihedrals[-1] - dihedrals[0]:+.6f}")
                print(f"    Max:      {max(dihedrals):.6f}")
                print(f"    Min:      {min(dihedrals):.6f}")

                if dihedrals[-1] > 2 * dihedrals[0]:
                    print(f"    🚨 PROBLEM: Weight doubled during training!")

            # Analyze nonbonded weight
            if 'nonbonded_weight' in weights_history and weights_history['nonbonded_weight']:
                nonbondeds = weights_history['nonbonded_weight']
                print(f"\n  Nonbonded Weight:")
                print(f"    Initial:  {nonbondeds[0]:.6f}")
                print(f"    Final:    {nonbondeds[-1]:.6f}")
                print(f"    Change:   {nonbondeds[-1] - nonbondeds[0]:+.6f}")
                print(f"    Max:      {max(nonbondeds):.6f}")
                print(f"    Min:      {min(nonbondeds):.6f}")

                if nonbondeds[-1] > 2 * nonbondeds[0]:
                    print(f"    🚨 PROBLEM: Weight doubled during training!")

        # Analyze loss history
        if 'train_loss' in history and 'val_loss' in history:
            train_losses = history['train_loss']
            val_losses = history['val_loss']

            print(f"\n📉 Loss Evolution:")
            print("-" * 80)
            print(f"  Train Loss:")
            print(f"    Initial: {train_losses[0]:.6f}")
            print(f"    Final:   {train_losses[-1]:.6f}")
            print(f"    Min:     {min(train_losses):.6f} (epoch {train_losses.index(min(train_losses))+1})")

            print(f"\n  Val Loss:")
            print(f"    Initial: {val_losses[0]:.6f}")
            print(f"    Final:   {val_losses[-1]:.6f}")
            print(f"    Min:     {min(val_losses):.6f} (epoch {val_losses.index(min(val_losses))+1})")

    # Recommendations
    print(f"\n💡 Recommendations:")
    print("-" * 80)

    if 'angle_weight' in state_dict and state_dict['angle_weight'].item() > 1.0:
        print("  1. ⚠️  权重无约束增长，建议：")
        print("     - 使用修改后的 V3 模型 (带 sigmoid 约束)")
        print("     - 重新训练模型")
        print("     - 或者降低学习率")

    if 'log_angle_weight' in state_dict:
        print("  1. ✅ 已使用权重约束，继续训练应该稳定")

    print("  2. 建议启用以下监控选项:")
    print("     --monitor_gradients  (监控梯度范数)")
    print("     --grad_clip 1.0      (对MSE loss使用更严格的梯度裁剪)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="诊断模型权重和梯度问题")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="检查点文件路径")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (可选)")

    args = parser.parse_args()

    diagnose_checkpoint(args.checkpoint, args.config)


if __name__ == "__main__":
    main()
