#!/usr/bin/env python3
"""
验证MSE和Cosine Similarity的理论范围

针对归一化嵌入（z-score normalized），验证：
1. 随机向量的MSE期望 ≈ 2
2. 随机向量的Cosine Similarity期望 ≈ 0
3. MSE ≈ 2(1 - cosine_similarity) 的关系

Usage:
    python scripts/verify_loss_ranges.py --dim 1536
    python scripts/verify_loss_ranges.py --dim 256
"""

import numpy as np
import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def verify_random_embeddings(dim=1536, n_samples=10000):
    """
    验证随机归一化嵌入的统计特性。

    Args:
        dim: 嵌入维度
        n_samples: 采样数量
    """
    print("="*80)
    print(f"验证随机归一化嵌入的统计特性 (维度={dim})")
    print("="*80)

    np.random.seed(42)
    torch.manual_seed(42)

    # 生成随机归一化嵌入 (标准正态分布 N(0,1))
    print(f"\n生成 {n_samples} 对随机嵌入...")
    embeddings_a = np.random.randn(n_samples, dim).astype(np.float32)
    embeddings_b = np.random.randn(n_samples, dim).astype(np.float32)

    # 验证归一化
    print(f"\n检查归一化状态:")
    print(f"  embeddings_a: mean={embeddings_a.mean():.6f}, std={embeddings_a.std():.6f}")
    print(f"  embeddings_b: mean={embeddings_b.mean():.6f}, std={embeddings_b.std():.6f}")

    # 计算MSE
    print(f"\n计算MSE...")
    mse_values = ((embeddings_a - embeddings_b) ** 2).mean(axis=1)

    print(f"\nMSE统计 (理论期望=2.0):")
    print(f"  均值: {mse_values.mean():.4f}")
    print(f"  标准差: {mse_values.std():.4f}")
    print(f"  最小值: {mse_values.min():.4f}")
    print(f"  最大值: {mse_values.max():.4f}")
    print(f"  中位数: {np.median(mse_values):.4f}")
    print(f"  95%分位数: [{np.percentile(mse_values, 2.5):.4f}, {np.percentile(mse_values, 97.5):.4f}]")

    # 判断
    if abs(mse_values.mean() - 2.0) < 0.05:
        print(f"  ✓ MSE均值接近理论值2.0")
    else:
        print(f"  ⚠️ MSE均值偏离理论值: {mse_values.mean():.4f} vs 2.0")

    # 计算Cosine Similarity
    print(f"\n计算Cosine Similarity...")
    embeddings_a_t = torch.tensor(embeddings_a)
    embeddings_b_t = torch.tensor(embeddings_b)
    cosine_values = F.cosine_similarity(embeddings_a_t, embeddings_b_t, dim=1).numpy()

    print(f"\nCosine Similarity统计 (理论期望=0.0):")
    print(f"  均值: {cosine_values.mean():.6f}")
    print(f"  标准差: {cosine_values.std():.6f}")
    print(f"  最小值: {cosine_values.min():.6f}")
    print(f"  最大值: {cosine_values.max():.6f}")
    print(f"  中位数: {np.median(cosine_values):.6f}")
    print(f"  95%分位数: [{np.percentile(cosine_values, 2.5):.6f}, {np.percentile(cosine_values, 97.5):.6f}]")

    # 判断
    if abs(cosine_values.mean()) < 0.01:
        print(f"  ✓ Cosine均值接近理论值0.0")
    else:
        print(f"  ⚠️ Cosine均值偏离理论值: {cosine_values.mean():.6f} vs 0.0")

    # 验证关系: MSE ≈ 2(1 - cosine_similarity)
    print(f"\n验证关系: MSE ≈ 2(1 - cosine_similarity)...")
    predicted_mse = 2 * (1 - cosine_values)
    actual_mse = mse_values

    correlation = np.corrcoef(predicted_mse, actual_mse)[0, 1]
    mae = np.abs(predicted_mse - actual_mse).mean()
    rmse = np.sqrt(((predicted_mse - actual_mse) ** 2).mean())

    print(f"\n关系验证:")
    print(f"  相关系数: {correlation:.6f} (理论=1.0)")
    print(f"  平均绝对误差: {mae:.6f}")
    print(f"  均方根误差: {rmse:.6f}")

    if correlation > 0.99:
        print(f"  ✓ 关系式高度准确 (r={correlation:.6f})")
    else:
        print(f"  ⚠️ 关系式偏差较大 (r={correlation:.6f})")

    return {
        'mse': mse_values,
        'cosine': cosine_values,
        'predicted_mse': predicted_mse,
        'dim': dim
    }


def verify_extreme_cases(dim=1536):
    """
    验证极端情况。
    """
    print("\n" + "="*80)
    print(f"验证极端情况 (维度={dim})")
    print("="*80)

    # Case 1: 完全相同
    print("\n情况1: 完全相同的向量")
    a = torch.randn(1, dim)
    b = a.clone()
    mse = F.mse_loss(a, b)
    cosine = F.cosine_similarity(a, b, dim=1).item()
    print(f"  MSE: {mse.item():.10f} (理论=0)")
    print(f"  Cosine: {cosine:.10f} (理论=1)")

    # Case 2: 完全相反
    print("\n情况2: 完全相反的向量")
    a = torch.randn(1, dim)
    b = -a
    mse = F.mse_loss(a, b)
    cosine = F.cosine_similarity(a, b, dim=1).item()
    predicted_mse = 2 * (1 - cosine)
    print(f"  MSE: {mse.item():.4f} (理论≈4)")
    print(f"  Cosine: {cosine:.6f} (理论=-1)")
    print(f"  预测MSE: {predicted_mse:.4f}")

    # Case 3: 正交向量
    print("\n情况3: 正交向量")
    # 生成两个正交向量（在高维空间中近似正交）
    a = torch.randn(1, dim)
    b = torch.randn(1, dim)
    # 正交化: b = b - (a·b)/(a·a) * a
    a_norm = a / a.norm()
    b = b - (a_norm * b).sum() * a_norm

    mse = F.mse_loss(a, b)
    cosine = F.cosine_similarity(a, b, dim=1).item()
    predicted_mse = 2 * (1 - cosine)
    print(f"  MSE: {mse.item():.4f}")
    print(f"  Cosine: {cosine:.6f} (理论≈0)")
    print(f"  预测MSE: {predicted_mse:.4f} (理论≈2)")


def plot_relationship(results, save_dir="analysis_results"):
    """
    绘制MSE和Cosine Similarity的关系图。
    """
    print(f"\n生成可视化图表...")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mse = results['mse']
    cosine = results['cosine']
    predicted_mse = results['predicted_mse']
    dim = results['dim']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. MSE分布
    ax = axes[0, 0]
    ax.hist(mse, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(2.0, color='red', linestyle='--', linewidth=2, label='理论期望=2.0')
    ax.axvline(mse.mean(), color='blue', linestyle='--', linewidth=2, label=f'实际均值={mse.mean():.4f}')
    ax.set_xlabel('MSE', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title(f'MSE分布 (维度={dim})', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Cosine Similarity分布
    ax = axes[0, 1]
    ax.hist(cosine, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(0.0, color='red', linestyle='--', linewidth=2, label='理论期望=0.0')
    ax.axvline(cosine.mean(), color='blue', linestyle='--', linewidth=2, label=f'实际均值={cosine.mean():.6f}')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title(f'Cosine Similarity分布 (维度={dim})', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. MSE vs Cosine散点图
    ax = axes[1, 0]
    # 采样以避免过多点
    sample_idx = np.random.choice(len(mse), size=min(1000, len(mse)), replace=False)
    ax.scatter(cosine[sample_idx], mse[sample_idx], alpha=0.3, s=10)

    # 理论曲线
    cosine_theory = np.linspace(-1, 1, 100)
    mse_theory = 2 * (1 - cosine_theory)
    ax.plot(cosine_theory, mse_theory, 'r--', linewidth=2, label='理论: MSE=2(1-cos)')

    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE vs Cosine Similarity', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. 预测MSE vs 实际MSE
    ax = axes[1, 1]
    sample_idx = np.random.choice(len(mse), size=min(1000, len(mse)), replace=False)
    ax.scatter(mse[sample_idx], predicted_mse[sample_idx], alpha=0.3, s=10, color='green')

    # 完美预测线 (y=x)
    min_val = min(mse.min(), predicted_mse.min())
    max_val = max(mse.max(), predicted_mse.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测 (y=x)')

    ax.set_xlabel('实际MSE', fontsize=12)
    ax.set_ylabel('预测MSE (from cosine)', fontsize=12)
    ax.set_title('MSE预测精度', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # 保存
    save_path = save_dir / f"loss_relationship_dim{dim}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 图表保存至: {save_path}")

    # 也显示
    # plt.show()
    plt.close()


def create_reference_table():
    """
    创建参考对照表。
    """
    print("\n" + "="*80)
    print("MSE ↔ Cosine Similarity 参考对照表")
    print("="*80)

    print("\n{:<20} {:<15} {:<20}".format("Cosine Similarity", "MSE", "质量评估"))
    print("-" * 60)

    reference = [
        (1.00, 0.00, "完美 (不可达)"),
        (0.95, 0.10, "优秀"),
        (0.90, 0.20, "优秀"),
        (0.85, 0.30, "优秀"),
        (0.80, 0.40, "良好"),
        (0.75, 0.50, "良好"),
        (0.70, 0.60, "良好"),
        (0.60, 0.80, "中等"),
        (0.50, 1.00, "中等"),
        (0.40, 1.20, "学习不足"),
        (0.30, 1.40, "学习不足"),
        (0.20, 1.60, "很差"),
        (0.10, 1.80, "很差"),
        (0.00, 2.00, "随机基线"),
        (-0.50, 3.00, "异常"),
        (-1.00, 4.00, "极端异常"),
    ]

    for cosine, mse, quality in reference:
        print(f"{cosine:>8.2f}            {mse:>6.2f}          {quality}")


def main():
    parser = argparse.ArgumentParser(description="验证MSE和Cosine Similarity的理论范围")

    parser.add_argument("--dim", type=int, default=1536,
                       help="嵌入维度 (default: 1536)")
    parser.add_argument("--n_samples", type=int, default=10000,
                       help="采样数量 (default: 10000)")
    parser.add_argument("--save_dir", type=str, default="analysis_results",
                       help="图表保存目录")
    parser.add_argument("--no_plot", action="store_true",
                       help="不生成图表")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MSE和Cosine Similarity理论范围验证工具")
    print("="*80)
    print(f"\n配置:")
    print(f"  嵌入维度: {args.dim}")
    print(f"  采样数量: {args.n_samples}")

    # 验证随机嵌入
    results = verify_random_embeddings(dim=args.dim, n_samples=args.n_samples)

    # 验证极端情况
    verify_extreme_cases(dim=args.dim)

    # 创建参考表
    create_reference_table()

    # 绘图
    if not args.no_plot:
        plot_relationship(results, save_dir=args.save_dir)

    print("\n" + "="*80)
    print("✓ 验证完成！")
    print("="*80)

    print("\n关键结论:")
    print("  1. 随机归一化嵌入的MSE期望 ≈ 2.0")
    print("  2. 随机归一化嵌入的Cosine Similarity期望 ≈ 0.0")
    print("  3. MSE ≈ 2(1 - cosine_similarity) 关系成立")
    print("\n训练建议:")
    print("  - 如果验证MSE > 2.0 → 模型未学习（不如随机）")
    print("  - 如果验证Cosine < 0.0 → 模型学习错误方向")
    print("  - 目标: MSE < 0.5, Cosine > 0.75")


if __name__ == "__main__":
    main()
