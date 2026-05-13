# 网站部署说明（无需本地安装包）

你本地不需要安装 Python 依赖，直接用云平台构建即可。  
本仓库已准备好：

- `Dockerfile`
- `render.yaml`
- `requirements.txt`

## 方案 A（推荐）：Render 一键部署

1. 打开 Render 并连接你的 GitHub 仓库。
2. 选择本仓库，Render 会识别 `render.yaml` 与 `Dockerfile`。
3. 部署后访问生成的 URL，即可在线使用 Dashboard。

启动命令已经写在 Dockerfile 中（容器内）：

```bash
python3 ligand_interactive_dashboard.py --model-dir ligand_outputs --excel-path "Kraken monophosphine coordinates AD Descriptors.xlsx" --host 0.0.0.0 --port ${PORT}
```

## 方案 B：任何支持 Docker 的平台

只要平台支持 Docker（Railway/Fly.io/云主机容器），直接使用仓库根目录 Dockerfile 部署即可。

## 功能确认

部署后访问：

- `/`：演示模式 Dashboard（中英文切换、自动讲解词、预测）
- `/report?lang=zh`：中文 PDF 汇报页
- `/report?lang=en`：英文 PDF 汇报页

在 `/report` 页面点击“打印/导出 PDF”即可得到可分享版本。
