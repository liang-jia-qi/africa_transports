# gsync_africa 配置指南

## 第一步：在 GitHub 新建仓库

1. 打开 https://github.com/new
2. 填写仓库名（比如 `Africa-Transport-Research`）
3. 设为 **Private** 或 Public（按需）
4. **不要**勾选 "Add a README" 或 "Add .gitignore"（我们自己有）
5. 点击 **Create repository**，复制仓库名备用

---

## 第二步：安装依赖

```bash
# 安装 Git LFS（Ubuntu/Debian）
sudo apt install git-lfs

# 或 Mac
brew install git-lfs
```

---

## 第三步：放置配置文件

把下载的三个文件放到 Africa_Transports 文件夹根目录：

```bash
cp .gitignore    /home/liang/Documents/Research/Africa_Transports/
cp .gitattributes /home/liang/Documents/Research/Africa_Transports/
```

---

## 第四步：填写脚本信息并安装

打开 `gsync_africa`，填入第 14-15 行：

```bash
GITHUB_USER="你的GitHub用户名"
GITHUB_REPO="你的仓库名"
```

然后安装为全局命令：

```bash
sudo cp gsync_africa /usr/local/bin/
sudo chmod +x /usr/local/bin/gsync_africa
```

---

## 第五步：首次运行

```bash
gsync_africa
```

首次运行会自动：
- 初始化 Git 仓库
- 启用 Git LFS
- 添加 GitHub remote
- 提交并推送所有文件

**注意**：GitHub 会弹出登录验证。推荐使用 Personal Access Token（PAT）作为密码：
- 生成地址：https://github.com/settings/tokens
- 权限勾选：`repo` 即可

---

## 日常使用

以后每次只需运行：

```bash
gsync_africa
```

脚本会自动检测变更、询问提交信息、推送。

---

## 忽略规则说明

以下内容**不会**上传到 GitHub：

| 路径 | 原因 |
|------|------|
| `input/Africapolis_GIS_2024/` | 大型 GIS 数据集 |
| `input/Density_Scenarios_Paper/` | 大型文件夹 |
| `input/global_pop_2025_CN_1km_R2025A_UA_v1.tif` | 大型栅格文件 |
| `input/World-Base-Map-Shapefiles.zip` | 大型压缩包 |

演示文稿（`.pptx`、`.pdf`）等通过 **Git LFS** 上传，几十 MB 没问题。
