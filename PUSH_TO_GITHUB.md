# 推送到GitHub - 完整操作指南

你的代码已经提交到本地Git仓库，现在需要推送到GitHub。由于遇到权限问题，请按照以下步骤操作：

## ✅ 当前状态

- ✓ 本地Git仓库已初始化
- ✓ 所有文件已提交 (commit 9360f18)
- ✓ 远程仓库已配置：`git@github.com:SleepingLei/RNA-3E-FFI.git`
- ✗ SSH认证需要配置

---

## 🔑 方法1：配置SSH密钥（推荐，一次设置永久使用）

### 步骤1：复制SSH公钥

你的SSH公钥是：
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIF+N5undzWHSDoI3vVrP1R5BkQIhZqRckhMr0kTQ+boJ leidingwei_ssrmaker
```

### 步骤2：添加到GitHub

1. 访问：https://github.com/settings/ssh/new
2. 在 "Title" 输入：`RNA-3E-FFI-macbook`
3. 在 "Key" 粘贴上面的公钥
4. 点击 "Add SSH key"

### 步骤3：测试连接

打开终端，执行：
```bash
ssh -T git@github.com
```

如果看到类似这样的消息就成功了：
```
Hi SleepingLei! You've successfully authenticated, but GitHub does not provide shell access.
```

### 步骤4：推送代码

```bash
cd /Users/ldw/Desktop/software/RNA-3E-FFI
git push
```

---

## 🎫 方法2：使用Personal Access Token（快速方法）

### 步骤1：生成Token

1. 访问：https://github.com/settings/tokens/new
2. 填写：
   - **Note**: `RNA-3E-FFI Project`
   - **Expiration**: 选择期限（建议90天或No expiration）
   - **Select scopes**: 勾选 `repo`（所有子选项都会自动勾选）
3. 点击底部 "Generate token"
4. **立即复制token**（格式类似：`ghp_xxxxxxxxxxxxxxxxxxxx`）⚠️ 只显示一次！

### 步骤2：更新远程仓库URL

打开终端，执行：
```bash
cd /Users/ldw/Desktop/software/RNA-3E-FFI

# 将YOUR_TOKEN替换为刚才复制的token
git remote set-url origin https://YOUR_TOKEN@github.com/SleepingLei/RNA-3E-FFI.git
```

示例（假设token是 `ghp_abc123`）：
```bash
git remote set-url origin https://ghp_abc123@github.com/SleepingLei/RNA-3E-FFI.git
```

### 步骤3：推送代码

```bash
git push
```

---

## 📱 方法3：使用GitHub Desktop（图形界面）

### 步骤1：安装GitHub Desktop

下载：https://desktop.github.com/

### 步骤2：登录

1. 打开GitHub Desktop
2. Sign in to GitHub.com
3. 使用浏览器登录你的GitHub账户

### 步骤3：添加仓库

1. File → Add Local Repository
2. 选择路径：`/Users/ldw/Desktop/software/RNA-3E-FFI`
3. 点击 "Add Repository"

### 步骤4：推送

1. 点击顶部的 "Push origin"
2. 等待推送完成

---

## 🚀 快速命令参考

### 检查当前状态
```bash
cd /Users/ldw/Desktop/software/RNA-3E-FFI
git status
git log --oneline -5
git remote -v
```

### 重新推送（如果失败后需要重试）
```bash
git push
# 或强制推送（谨慎使用）
git push -f origin main
```

---

## ❓ 常见问题

### Q1: 推送时提示 "Permission denied"
**解决**：使用方法1或方法2重新配置认证

### Q2: 推送时提示 "Repository not found"
**解决**：检查仓库名称是否正确
```bash
git remote -v
# 应该显示：git@github.com:SleepingLei/RNA-3E-FFI.git
```

### Q3: 提示 "failed to push some refs"
**解决**：可能远程有新内容，先拉取：
```bash
git pull origin main --rebase
git push
```

### Q4: 我的GitHub用户名不是SleepingLei
**解决**：更新远程仓库URL
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/RNA-3E-FFI.git
# 或创建新仓库
gh repo create RNA-3E-FFI --public --source=. --push
```

---

## ✅ 验证推送成功

推送成功后：

1. 访问：https://github.com/SleepingLei/RNA-3E-FFI
2. 应该能看到所有文件：
   - README.md
   - requirements.txt
   - models/
   - scripts/
   - 等等

3. 检查commit历史：
   - 点击 "X commits"
   - 应该能看到你的提交记录

---

## 📝 推送成功后的后续操作

### 1. 添加项目描述

在GitHub仓库页面：
1. 点击 "Add description"
2. 输入：`E(3) Equivariant Graph Neural Network for RNA-Ligand Virtual Screening`

### 2. 添加Topics（标签）

点击设置图标，添加以下topics：
- `deep-learning`
- `graph-neural-networks`
- `e3nn`
- `drug-discovery`
- `rna-binding`
- `pytorch-geometric`
- `computational-biology`

### 3. 考虑添加徽章

在README.md顶部添加：
```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

---

## 💡 推荐工作流

日常开发时：
```bash
# 1. 修改代码
# 2. 查看更改
git status
git diff

# 3. 添加更改
git add .

# 4. 提交
git commit -m "feat: add new feature"

# 5. 推送
git push

# 或者一次性提交并推送
git add . && git commit -m "your message" && git push
```

---

需要帮助？请查看：
- GitHub文档：https://docs.github.com/
- Git教程：https://git-scm.com/book/zh/v2
- 或在终端运行：`git help`
