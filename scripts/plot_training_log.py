import re
import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件
log_file = '/Users/ldw/Desktop/software/RNA-3E-FFI/docs/noohup-v2-1000-sota.out'

with open(log_file, 'r') as f:
    lines = f.readlines()

# 用于存储数据
epochs = []
train_losses = []
angle_weights = []
dihedral_weights = []
nonbonded_weights = []
val_losses = []

# 临时变量
current_epoch = None
current_train_loss = None
current_angle_weight = None
current_dihedral_weight = None
current_nonbonded_weight = None
current_val_loss = None

# 用于检测训练中断和重新开始
last_epoch = -1
epoch_offset = 0

# 解析日志
i = 0
while i < len(lines):
    line = lines[i].strip()

    # 匹配 Epoch
    epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
    if epoch_match:
        epoch_num = int(epoch_match.group(1))

        # 检测训练是否中断重启（epoch 数字变小了）
        if epoch_num <= last_epoch:
            # 训练重新开始，更新 offset
            epoch_offset = epochs[-1] if epochs else 0
            print(f"检测到训练中断重启: Epoch {epoch_num}, offset设为 {epoch_offset}")

        current_epoch = epoch_num + epoch_offset
        last_epoch = epoch_num

    # 匹配 Train Loss
    train_loss_match = re.search(r'Train Loss:\s*([\d.]+)', line)
    if train_loss_match:
        current_train_loss = float(train_loss_match.group(1))

    # 匹配 Angle weight
    angle_weight_match = re.search(r'Angle weight:\s*([\d.]+)', line)
    if angle_weight_match:
        current_angle_weight = float(angle_weight_match.group(1))

    # 匹配 Dihedral weight
    dihedral_weight_match = re.search(r'Dihedral weight:\s*([\d.]+)', line)
    if dihedral_weight_match:
        current_dihedral_weight = float(dihedral_weight_match.group(1))

    # 匹配 Nonbonded weight
    nonbonded_weight_match = re.search(r'Nonbonded weight:\s*([\d.]+)', line)
    if nonbonded_weight_match:
        current_nonbonded_weight = float(nonbonded_weight_match.group(1))

    # 匹配 Val Loss
    val_loss_match = re.search(r'Val Loss:\s*([\d.]+)', line)
    if val_loss_match:
        current_val_loss = float(val_loss_match.group(1))

        # 当我们读取到 Val Loss 时，说明这个 epoch 的所有数据都读取完了
        if all([current_epoch is not None,
                current_train_loss is not None,
                current_angle_weight is not None,
                current_dihedral_weight is not None,
                current_nonbonded_weight is not None]):

            epochs.append(current_epoch)
            train_losses.append(current_train_loss)
            angle_weights.append(current_angle_weight)
            dihedral_weights.append(current_dihedral_weight)
            nonbonded_weights.append(current_nonbonded_weight)
            val_losses.append(current_val_loss)

            # 重置临时变量
            current_train_loss = None
            current_angle_weight = None
            current_dihedral_weight = None
            current_nonbonded_weight = None
            current_val_loss = None

    i += 1

# 打印统计信息
print(f"\n提取数据统计:")
print(f"总共提取了 {len(epochs)} 个 epoch 的数据")
print(f"Epoch 范围: {min(epochs)} - {max(epochs)}")
print(f"Train Loss 范围: {min(train_losses):.4f} - {max(train_losses):.4f}")
print(f"Val Loss 范围: {min(val_losses):.4f} - {max(val_losses):.4f}")

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Train Loss 和 Val Loss
axes[0, 0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
axes[0, 0].plot(epochs, val_losses, label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 所有 Weights
axes[0, 1].plot(epochs, angle_weights, label='Angle Weight', linewidth=2)
axes[0, 1].plot(epochs, dihedral_weights, label='Dihedral Weight', linewidth=2)
axes[0, 1].plot(epochs, nonbonded_weights, label='Nonbonded Weight', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Weight')
axes[0, 1].set_title('Loss Weights Over Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Train Loss (单独)
axes[1, 0].plot(epochs, train_losses, color='blue', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Train Loss')
axes[1, 0].set_title('Training Loss')
axes[1, 0].grid(True, alpha=0.3)

# 4. Val Loss (单独)
axes[1, 1].plot(epochs, val_losses, color='orange', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Val Loss')
axes[1, 1].set_title('Validation Loss')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/ldw/Desktop/software/RNA-3E-FFI/docs/training_curves.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存到: /Users/ldw/Desktop/software/RNA-3E-FFI/docs/training_curves.png")

plt.show()

# 额外保存数据到 CSV 文件
import csv
csv_file = '/Users/ldw/Desktop/software/RNA-3E-FFI/docs/training_data.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train Loss', 'Angle Weight', 'Dihedral Weight', 'Nonbonded Weight', 'Val Loss'])
    for i in range(len(epochs)):
        writer.writerow([epochs[i], train_losses[i], angle_weights[i], dihedral_weights[i],
                        nonbonded_weights[i], val_losses[i]])

print(f"数据已保存到: {csv_file}")
