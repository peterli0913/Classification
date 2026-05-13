# 真实收率数据来源与使用说明

你当前没有本地真实收率列时，可直接使用以下公开数据（我已在代码中接入）：

## 已验证可用的数据源

1. **Buchwald-Hartwig（CSV，含 SMILES 与 yield）**
   - URL: `https://raw.githubusercontent.com/schwallergroup/ai4chem_course/main/notebooks/10%20-%20Bayesian%20optimization/bh-reactions.csv`
   - 字段示例：`ligand`, `additive`, `base`, `aryl halide`, `yield`
   - 用途：直接用于本项目“真实收率重训 + 离散 BO”

2. **Dreher & Doyle HTE（Excel）**
   - URL: `https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx`
   - 字段示例：`Ligand`, `Additive`, `Base`, `Aryl halide`, `Output`
   - 用途：可作为外部补充/对照数据

3. **Suzuki-Miyaura HTE（Excel）**
   - URL: `https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx`
   - 字段示例：`Ligand_Short_Hand`, `Product_Yield_PCT_Area_UV`
   - 用途：可做迁移验证或跨反应类型评估

---

## 一键下载

```bash
python3 fetch_public_yield_data.py --output-dir data
```

---

## 新增管线（已实现）

脚本：`yield_bo_discrete_pipeline.py`

支持能力：

1. 真实收率建模（替代 AF 映射列）
2. 不确定性建模
   - bootstrap ensemble（均值 + std）
   - quantile 回归（0.1/0.9 区间）
3. 外推验证
   - K-fold
   - scaffold group K-fold（无 RDKit 依赖的轻量 scaffold 签名）
4. 图模型
   - SMILES -> 轻量图解析 -> GNN 训练
5. 离散 BO 闭环
   - `acq = mu + beta * sigma` / EI
   - 从离散候选空间推荐下一批分子/条件组合

---

## 典型命令

### 训练 + 验证 + BO 模拟

```bash
python3 yield_bo_discrete_pipeline.py train \
  --data-path data/bh-reactions.csv \
  --output-dir yield_bo_outputs \
  --reaction-id 0 \
  --kfolds 5 \
  --gnn-epochs 20
```

### 预测新 ligand 的收率（给定离散条件）

```bash
python3 yield_bo_discrete_pipeline.py predict \
  --output-dir yield_bo_outputs \
  --smiles "YOUR_LIGAND_SMILES" \
  --reaction "0" \
  --additive "..." \
  --base "..." \
  --aryl_halide "..."
```

### 从离散候选池推荐下一批实验

```bash
python3 yield_bo_discrete_pipeline.py suggest \
  --output-dir yield_bo_outputs \
  --candidate-data data/bh-reactions.csv \
  --reaction-id 0 \
  --top-k 20 \
  --out-csv yield_bo_outputs/next_candidates.csv
```

---

## 说明

- 你提到“feature 是离散不是连续”，该脚本已按离散候选空间 BO 设计；
- scaffold split 使用轻量近似（不依赖 RDKit），优点是部署轻、运行快；如果后续允许加 RDKit，可再升级为标准 Bemis-Murcko scaffold。
