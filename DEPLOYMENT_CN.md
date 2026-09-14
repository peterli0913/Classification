# 网站部署说明（无需本地安装包）

你本地不需要安装 Python 依赖，直接用云平台构建即可。  
本仓库已准备好：

- `Dockerfile`
- `render.yaml`
- `requirements.txt`
- `yield_bo_rdkit_pyg_pipeline.py`（RDKit+PyG 训练）
- `data/bh-reactions.csv`
- `data/Dreher_and_Doyle_input_data.xlsx`
- `data/aap9112_Data_File_S1.xlsx`

## 方案 A（推荐）：Render 一键部署

1. 打开 Render 并连接你的 GitHub 仓库。
2. 选择本仓库，Render 会识别 `render.yaml` 与 `Dockerfile`。
3. 部署后访问生成的 URL，即可在线使用 Dashboard。

启动命令已经写在 Dockerfile 中（容器内）：

```bash
python3 render_dashboard_lite.py --base-summary-path ligand_outputs/results_summary.json --bo2-output-dir yield_bo_pyg_outputs --bo2-candidate-data data/bh-reactions.csv --host 0.0.0.0 --port ${PORT}
```

## 方案 B：任何支持 Docker 的平台

只要平台支持 Docker（Railway/Fly.io/云主机容器），直接使用仓库根目录 Dockerfile 部署即可。

## 功能确认

部署后访问：

- `/`：Render 轻量版 Dashboard（基础指标 + BO2 预测/推荐）
- `/api/bo2/predict`：RDKit+PyG 预测接口
- `/api/bo2/suggest`：离散 BO 候选推荐接口

> 说明：Render 免费实例内存较小，线上默认使用 `render_dashboard_lite.py`（轻量版）保证稳定运行与 BO2 功能可用。完整版界面可在本地/更高内存实例运行 `ligand_interactive_dashboard.py`。
