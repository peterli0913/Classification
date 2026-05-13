# Ligand Feature 分析与预测系统（专家展示版）

## 1. 项目目标与当前完成度

本项目围绕高通量实验中的 ligand 结构/描述符与目标 label 的关系建模，目标是：

1. 从描述符中挖掘对 label（当前数据中为 `vbur_ratio_vbur_vtot`）影响较深的特征模式；
2. 用 attention 编码挖掘组合特征（group-level + sequence-level）；
3. 将 ligand 投影到二维空间，观察“相似 label 是否在表示空间聚集”；
4. 建立新 ligand 到 label 的预测系统，为后续 Reaction BO 提供先验。

当前已完成：

- 多模型系统（XGBoost + 描述符 Attention + SMILES-Descriptor 混合网络 + 集成）；
- 统一评估（RMSE / MAE / R²）；
- 可视化输出（5 张核心结果图）；
- 新 ligand 交互预测接口；
- “superpower skill” 方法增强（Meta-Pattern Recognition）在建模逻辑中落地。

---

## 2. 数据与特征结构

- 数据集：`Kraken monophosphine coordinates AD Descriptors.xlsx`
- 样本数：237
- 特征：按列位映射 O-AE（17 维）
- SMILES：J 列
- Label：AF 列（当前映射列名 `vbur_ratio_vbur_vtot`）

特征分组（用于组级 attention）：

- Group-1: O-Q（3维）
- Group-2: R-T（3维）
- Group-3: U-W（3维）
- Group-4: X-AE（8维）

---

## 3. 模型体系与方法亮点

### 3.1 XGBoost baseline

- 作用：构建强基线，快速得到可解释特征重要性排序；
- 优点：小样本、非线性关系下稳定，便于诊断。

### 3.2 Descriptor Attention Regressor

- 对四个 descriptor group 编码为 token；
- 使用 self-attention 学习组间关系；
- 输出 group attention 权重，解释“哪类 descriptor 组更关键”。

### 3.3 SMILES + Descriptor Hybrid

- SMILES 字符序列经双向 GRU + attention 提取结构语义；
- 与标准化 descriptor 分支融合后回归；
- 兼顾结构信息与工程描述符。

### 3.4 集成策略

- 对三模型预测取均值（ensemble_avg），降低单模型偏差风险。

---

## 4. 当前 performance（测试集）

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| XGBoost | 0.0182 | 0.0145 | 0.8574 |
| Descriptor Attention | 0.0327 | 0.0230 | 0.5424 |
| SMILES Hybrid | 0.0237 | 0.0173 | 0.7599 |
| Ensemble Avg | 0.0201 | 0.0151 | 0.8261 |

### 4.1 结果解读

1. **XGBoost 当前最优**（R² 0.8574），说明在当前样本规模下，树模型对 descriptor 非线性关系的拟合能力最强；
2. **SMILES Hybrid 次优**（R² 0.7599），证明加入结构序列信息是有效的；
3. **Descriptor Attention 单模较弱**，但其强项是“可解释性”（可输出组权重），适合与其他模型联合诊断；
4. **Ensemble 稳定但未超过最优单模**，提示可改为加权集成（按验证集表现学习权重）而非均值。

### 4.2 关键特征与深层模式

XGBoost Top 特征（按 importance）：

1. `sterimol_B1` (0.2598)
2. `E_solv_total` (0.1846)
3. `sterimol_B5` (0.1180)
4. `sterimol_burL` (0.0761)
5. `vbur_vbur.2` (0.0728)

Attention 组权重均值 `[O-Q, R-T, U-W, X-AE]`：

- `[0.2494, 0.1490, 0.3216, 0.2800]`

说明：

- **U-W 组（Group-3）与 X-AE 组（Group-4）贡献更高**；
- 结合 XGBoost importance，可推断形状相关 + 溶剂/电子相关特征共同影响 label。

---

## 5. 结果图片逐图说明（用于汇报）

> 图像目录：`ligand_outputs/figures/`

### 5.1 `ligand_embedding_tsne.png`

- 含义：将 attention latent 投影到二维；
- 观察点：颜色（label）是否出现局部聚类与连续梯度；
- 结论范式：若同色团簇明显，说明表示学习捕获了与 label 相关的结构化信息，可用于候选筛选。

### 5.2 `actual_vs_predicted.png`

- 含义：每个模型的真实值 vs 预测值散点；
- 观察点：点越贴近对角线越好，离群点反映模型难点区域；
- 结论范式：XGBoost 子图最接近对角线，SMILES 次之，attention 单模分散更大。

### 5.3 `model_performance.png`

- 含义：RMSE/MAE/R² 的并列对比；
- 观察点：误差最小与解释度最高模型；
- 结论范式：当前生产优先 XGBoost 或加权集成；attention 模型承担解释补充角色。

### 5.4 `xgboost_feature_importance.png`

- 含义：描述符的重要性排序；
- 观察点：头部特征是否符合化学先验；
- 结论范式：可用于后续实验特征优先级设计与降维保留策略。

### 5.5 `descriptor_group_attention_heatmap.png`

- 含义：测试样本维度上的组权重热图；
- 观察点：不同样本上主导组是否切换（反映机制异质性）；
- 结论范式：若存在样本簇对应不同主导组，可按机制分簇做局部 BO。

---

## 6. 下一步建议（面向 BO 和实验设计）

1. **切换到真实收率 label（yield）重训**
   - 当前 AF 列并非显式 yield；
   - 一旦提供真实收率列，优先进行全流程复训与复评估。

2. **不确定性建模 + 主动学习**
   - 增加 quantile regression / bootstrap ensemble；
   - 输出均值 + 方差，用于 BO 的 acquisition function（EI/UCB）。

3. **多折交叉验证 + 外推验证**
   - 小样本建议 5-fold CV；
   - 增加 scaffold split 检查对新化学空间的泛化。

4. **从字符 SMILES 升级到图神经网络（GNN）**
   - 目前是字符序列编码；
   - 建议引入分子图（节点/键）消息传递，提升结构归纳能力。

5. **BO 闭环集成**
   - 将预测器作为 surrogate model；
   - 每轮实验后增量更新模型，实现“实验-建模-建议”闭环。

---

## 7. 互动展示入口

已新增 Python 交互式 Dashboard：`ligand_interactive_dashboard.py`

启动方式：

```bash
python3 ligand_interactive_dashboard.py \
  --model-dir ligand_outputs \
  --excel-path "Kraken monophosphine coordinates AD Descriptors.xlsx" \
  --host 127.0.0.1 \
  --port 8765
```

打开浏览器访问：

`http://127.0.0.1:8765`

功能：

- 展示项目概览、指标、图像与专业解释；
- 输入新 ligand 的 SMILES 与 17 维 descriptor；
- 实时返回三模型与集成预测，以及 descriptor group attention 权重。

## 8. 演示模式升级（本次新增）

Dashboard 已升级为“演示模式”，新增：

1. **中英文一键切换**（适配内部/外部汇报）
2. **自动讲解词**（项目负责人口吻，避免“AI腔”）
3. **PDF 导出页**（`/report?lang=zh` 或 `/report?lang=en`）

可部署说明见：`DEPLOYMENT_CN.md`（支持无需本地装包的云端部署流程）。

## 9. 真实收率与 BO 闭环升级（本次新增）

新增脚本：`yield_bo_discrete_pipeline.py`，用于你当前最关键的下一步：

1. 用真实收率数据重训（默认 `data/bh-reactions.csv`）
2. 不确定性建模（bootstrap std + quantile 区间）
3. scaffold group K-fold + 常规 K-fold 验证
4. SMILES 图模型（轻量 GNN）
5. 离散候选 BO 推荐（适配离散 feature 空间）

参考文档：

- `REAL_YIELD_DATA_SOURCES_CN.md`（真实数据来源与命令）
- `fetch_public_yield_data.py`（一键下载公开数据）
