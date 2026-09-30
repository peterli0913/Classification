# RDKit + PyG 升级版：用法与结果说明（含 Render 交互）

## 1) 你关心的能力是否已落地

已落地，包含四块：

1. **标准分子图建模**（RDKit 解析分子，PyG-GCN 训练）
2. **标准 scaffold 验证**（Bemis-Murcko scaffold + GroupKFold）
3. **不确定性与离散 BO 推荐**（bootstrap std + EI/UCB）
4. **接入 Dashboard**（网页交互预测 + 候选推荐）

对应代码：

- `yield_bo_rdkit_pyg_pipeline.py`
- `ligand_interactive_dashboard.py`（新增 BO2 区域）

---

## 2) 训练命令（真实收率）

```bash
python3 yield_bo_rdkit_pyg_pipeline.py train \
  --data-path data/bh-reactions.csv \
  --output-dir yield_bo_pyg_outputs \
  --reaction-id 0 \
  --kfolds 3 \
  --n-bootstrap 10 \
  --gnn-epochs 8 \
  --bo-iter 8 \
  --top-k 8
```

输出：

- `yield_bo_pyg_outputs/summary.json`
- `yield_bo_pyg_outputs/top_recommendations.csv`
- `yield_bo_pyg_outputs/next_candidates.csv`
- `yield_bo_pyg_outputs/figures/rdkit_bootstrap_uncertainty_scatter.png`
- `yield_bo_pyg_outputs/figures/rdkit_discrete_bo_curve.png`

数据文件默认已在仓库：

- `data/bh-reactions.csv`
- `data/Dreher_and_Doyle_input_data.xlsx`
- `data/aap9112_Data_File_S1.xlsx`

---

## 3) 核心结果（当前实验）

测试集（reaction=0）：

- Bootstrap: `RMSE=7.9264`, `R²=0.9301`
- Quantile mean: `RMSE=10.5055`, `R²=0.8772`
- GNN(PyG): `RMSE=30.3484`, `R²=-0.0249`

Scaffold GroupKFold：

- `RMSE(mean)=13.5107`
- `R²(mean)=0.5173`

说明：

- 现阶段 **bootstrap + RDKit 指纹** 仍是主力；
- GNN 已接入标准图建模流程，但在当前小数据与参数下还需进一步调优；
- scaffold 分割结果可用于评估“新骨架外推”风险。

---

## 4) 预测与推荐命令

### 4.1 新 ligand 预测

```bash
python3 yield_bo_rdkit_pyg_pipeline.py predict \
  --output-dir yield_bo_pyg_outputs \
  --smiles "YOUR_LIGAND_SMILES" \
  --reaction "0" \
  --additive "..." \
  --base "..." \
  --aryl_halide "..."
```

输出字段：

- `bootstrap_mean`
- `bootstrap_std`
- `gnn_pyg`
- `ensemble_avg`

### 4.2 离散候选推荐（下一批实验）

```bash
python3 yield_bo_rdkit_pyg_pipeline.py suggest \
  --output-dir yield_bo_pyg_outputs \
  --candidate-data data/bh-reactions.csv \
  --reaction-id 0 \
  --top-k 20 \
  --out-csv yield_bo_pyg_outputs/next_candidates.csv
```

---

## 5) Dashboard 内如何使用（Render 上同样）

启动：

```bash
# 本地/高内存环境（完整版）
python3 ligand_interactive_dashboard.py ...

# Render 免费实例（轻量版）
python3 render_dashboard_lite.py \
  --base-summary-path ligand_outputs/results_summary.json \
  --bo2-output-dir yield_bo_pyg_outputs \
  --bo2-candidate-data data/bh-reactions.csv \
  --host 0.0.0.0 \
  --port 8765
```

网页中新增区域：

- **“四、RDKit + PyG + BO（新）”**
  - 输入 `SMILES + reaction + additive + base + aryl halide`
  - 点击“运行 RDKit+PyG 预测”
  - 点击“推荐下一批候选”

后端接口：

- `POST /api/bo2/predict`
- `POST /api/bo2/suggest`

---

## 6) Render 部署要点

- `Dockerfile` 已更新：首次部署若无 `yield_bo_pyg_outputs/meta.json`，会自动下载数据并完成一轮训练，然后启动 Dashboard；
- 这意味着你本地无需安装 RDKit / PyG，也能在线看到新功能。
