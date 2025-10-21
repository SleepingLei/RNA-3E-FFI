# 如何将项目推送到GitHub

## 方法一：使用命令行（推荐）

### 1. 初始化Git仓库

```bash
cd /Users/ldw/Desktop/software/RNA-3E-FFI
git init
```

### 2. 添加所有文件到暂存区

```bash
git add .
```

### 3. 创建第一个提交

```bash
git commit -m "Initial commit: RNA-3E-FFI project with E(3) GNN implementation"
```

### 4. 在GitHub上创建新仓库

1. 打开浏览器访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `RNA-3E-FFI`
   - **Description**: `E(3) Equivariant Graph Neural Network for RNA-Ligand Virtual Screening`
   - **Public/Private**: 根据需要选择
   - **不要**勾选 "Initialize this repository with a README"（我们已经有了）
3. 点击 "Create repository"

### 5. 连接本地仓库到GitHub

复制GitHub页面上显示的仓库URL（例如：`https://github.com/your-username/RNA-3E-FFI.git`），然后执行：

```bash
# 添加远程仓库（替换成你的URL）
git remote add origin https://github.com/your-username/RNA-3E-FFI.git

# 查看远程仓库配置
git remote -v
```

### 6. 推送到GitHub

```bash
# 推送到main分支
git push -u origin main

# 如果你的默认分支是master，使用：
# git push -u origin master
```

如果遇到认证问题，GitHub现在需要使用个人访问令牌（Personal Access Token）：

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token" -> "Generate new token (classic)"
3. 设置权限：至少勾选 `repo` 相关权限
4. 生成并复制令牌
5. 推送时使用令牌作为密码

---

## 方法二：使用GitHub CLI（更简单）

### 1. 安装GitHub CLI

```bash
# macOS
brew install gh

# 或访问 https://cli.github.com/ 下载
```

### 2. 登录GitHub

```bash
gh auth login
```

按照提示完成登录（选择 HTTPS，使用浏览器登录）

### 3. 初始化并推送

```bash
cd /Users/ldw/Desktop/software/RNA-3E-FFI

# 初始化git（如果还没做）
git init
git add .
git commit -m "Initial commit: RNA-3E-FFI project"

# 创建GitHub仓库并推送
gh repo create RNA-3E-FFI --public --source=. --push
```

选项说明：
- `--public`: 创建公开仓库（使用 `--private` 创建私有仓库）
- `--source=.`: 使用当前目录作为源
- `--push`: 立即推送

---

## 方法三：使用GitHub Desktop（图形界面）

### 1. 下载并安装GitHub Desktop

访问 https://desktop.github.com/ 下载

### 2. 添加本地仓库

1. 打开GitHub Desktop
2. File -> Add Local Repository
3. 选择 `/Users/ldw/Desktop/software/RNA-3E-FFI`
4. 如果提示"这不是Git仓库"，点击 "Create a Repository"

### 3. 提交更改

1. 在左侧查看更改的文件
2. 在左下角输入提交信息
3. 点击 "Commit to main"

### 4. 发布到GitHub

1. 点击顶部的 "Publish repository"
2. 填写仓库名称和描述
3. 选择公开或私有
4. 点击 "Publish Repository"

---

## 后续更新代码

当你修改代码后，使用以下命令更新：

```bash
# 查看更改
git status

# 添加更改的文件
git add .

# 提交更改
git commit -m "描述你的更改"

# 推送到GitHub
git push
```

---

## 常用Git命令

```bash
# 查看状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 拉取最新代码
git pull

# 创建新分支
git checkout -b feature-branch

# 切换分支
git checkout main

# 合并分支
git merge feature-branch

# 查看差异
git diff
```

---

## 注意事项

### ⚠️ 数据文件

`.gitignore` 已配置忽略大文件：
- `data/` 目录下的所有数据文件
- `.h5`、`.pdb`、`.cif` 等大型文件
- 模型checkpoint文件

如果需要共享数据：
1. 使用 Git LFS (Large File Storage)
2. 或上传到云存储（Google Drive、Zenodo等）并在README中提供链接

### 🔒 敏感信息

确保不要提交：
- API密钥
- 密码
- 个人访问令牌
- 任何敏感配置

### 📝 许可证

考虑添加LICENSE文件：

```bash
# 查看常用许可证
gh repo license-list

# 使用MIT许可证
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy...
EOF
```

---

## 推荐的提交信息格式

遵循规范的提交信息：

```
feat: 添加新功能
fix: 修复bug
docs: 更新文档
style: 代码格式调整
refactor: 代码重构
test: 添加测试
chore: 构建/工具更改
```

示例：
```bash
git commit -m "feat: add attention mechanism to pooling layer"
git commit -m "fix: resolve atom ordering issue in graph construction"
git commit -m "docs: update installation instructions"
```

---

## 快速参考

```bash
# 完整流程（首次推送）
cd /Users/ldw/Desktop/software/RNA-3E-FFI
git init
git add .
git commit -m "Initial commit: RNA-3E-FFI project"
git remote add origin https://github.com/your-username/RNA-3E-FFI.git
git push -u origin main

# 日常更新
git add .
git commit -m "your message"
git push
```
