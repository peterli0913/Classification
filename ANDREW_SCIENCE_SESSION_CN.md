# Andrew 科学讨论稿（公开集验证，可展示）

数字全部来自公开 HTE，不是内部工艺数据。随机划分只说明“格子里插值”，**对外该报 leave-one-component**。

配套：`hte_scientific_outputs/report.html` 与 `hte_scientific_outputs/figures/`。

---

## 0. 开场（约 45 秒）

上一版管线有两个会误导人的地方：只给配体做指纹，却主要报随机划分 R²。Doyle 这套数据只有 4 个配体，随机划分几乎等于“见过的配体再插值一次”。

这次我在两个公开集上把评估改成化学上站得住的划分，并对比了五种模型：主效应、one-hot XGB、旧的配体指纹、多组分指纹、主效应+残差提升。结论可以收成三句：

1. **插值已经够强**：BH 随机 5-fold，多组分 / 残差 XGB 的 R² ≈ **0.94**，RMSE ≈ **6.7**，比旧管线（约 8.3 / 0.91）明显好。
2. **外推要看“拿掉的是什么”**：新添加剂上，多组分指纹 22 个里有 **17/22** 个 RMSE 更低；换配体则完全取决于拿掉的是哪一个。
3. **GNN 现在还不是答案**：配体只有 4 张图，图模型 R² 只有 0.08–0.19，远低于树模型。

---

## 1. 数据（请先对齐，避免被问“你训的是什么”）

| 集合 | 来源 | n | 组成 | label |
|---|---|---:|---|---|
| Buchwald–Hartwig | Ahneman / Doyle *Science* 2018，公开 CSV | 3955 | 4 ligands × 22 additives × 3 bases × 15 aryl halides × 5 products | isolated-style yield |
| Suzuki–Miyaura | Perera et al. *Science* 2018 (aap9112) | 5760 | 11 named ligands + no-ligand，7 electrophile / nucleophile 组合，溶剂/碱 | HPLC area% yield |

四个 BH 配体（按 SMILES 对应）：**tBuBrettPhos, tBuXPhos, XPhos, AdBrettPhos**。不是 237 个 Kraken 配体，也不是真实车间收率。

---

## 2. 模型改了什么（对着方法问）

| 模型 | 含义 | 为什么要有它 |
|---|---|---|
| Main effects | 配体/添加剂/碱/卤代物均值相加 | ANOVA 基线：没有交互时能解释多少 |
| One-hot XGB | 所有组分 one-hot | 文献插值上限（格子见过就能拟合） |
| Ligand-FP XGB | 只指纹配体，其余 one-hot | **上一版管线** |
| Multi-FP XGB | 配体+添加剂+芳基卤的 Morgan + RDKit 描述符，碱/反应 one-hot | 结构多样性其实在添加剂和卤代物上 |
| Residual XGB | 主效应 + 对残差做多组分 XGB | 把“这个配体普遍好”和交互分开 |

没有新依赖。GNN 仍用 PyG，但只作为对照，不当主力。

---

## 3. 该往黑板上写的数字

### 3.1 Buchwald–Hartwig

| 划分 | Main eff. R² | One-hot R² | 旧 ligand-FP R² | Multi-FP R² | Residual R² | Multi-FP RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Random 5-fold | 0.601 | 0.906 | 0.908 | 0.938 | 0.939 | 6.77 |
| Leave-one-ligand (mean) | -0.047 | 0.361 | 0.388 | 0.365 | 0.375 | 14.64 |
| Leave-one-product | -0.127 | -0.340 | -0.293 | **0.084** | -0.198 | 21.64 |
| Unseen additives (GroupKFold) | 0.421 | 0.626 | 0.625 | **0.718** | 0.666 | 14.25 |
| Leave-one-additive ×22 (mean) | — | 0.223 | 0.219 | **0.451** | — | 13.58 |
| Unseen aryl halide | -0.523 | -0.373 | -0.366 | **0.152** | -0.111 | 17.27 |

Leave-one-additive **中位数**（比均值老实）：one-hot R² 0.69 / RMSE 14.3；multi-FP R² **0.77** / RMSE **12.5**。22 个添加剂里 multi-FP RMSE 更好的有 **17/22**。

### 3.2 换配体：不要只报平均

平均 R² ~0.36 会被 XPhos 那一折（R²≈−1，RMSE≈25）拉垮。拆开看：

| Held-out ligand | One-hot RMSE / R² | Multi-FP RMSE / R² | 怎么读 |
|---|---|---|---|
| tBuBrettPhos | 10.2 / 0.87 | **7.3 / 0.94** | BrettPhos 家族互相像，指纹能迁 |
| AdBrettPhos | 10.9 / 0.85 | **8.7 / 0.91** | 同上（都有 OMe） |
| tBuXPhos | **13.8 / 0.76** | 17.9 / 0.59 | 去掉 OMe 后结构迁过去会偏 |
| XPhos (PCy₂) | 25.1 / −1.04 | 24.6 / −0.98 | 立体完全不同，4 个配体里迁不过去 |

开口句：

> 4 个 Buchwald 配体不够支撑“通用配体 GNN”。能迁的是 BrettPhos 家族内部；XPhos 必须当成新化学实体，不能靠指纹硬外推。

### 3.3 Suzuki–Miyaura（第二反应类型）

| 划分 | One-hot R² | 旧 ligand-FP R² | Multi-FP R² | Residual R² |
|---|---:|---:|---:|---:|
| Random 5-fold | 0.840 | 0.852 | 0.855 | 0.850 |
| Leave-one-ligand (mean) | **0.200** | -0.193 | -0.045 | 0.165 |
| Leave-one-reagent | 0.755 | 0.759 | 0.762 | 0.763 |
| Leave-one-solvent | 0.194 | 0.183 | 0.204 | **0.272** |

Suzuki 换配体时，**指纹经常更差**（Xantphos、PPh₃、CataCXium A）。Xantphos 是双膦，机制就和其他单膦不一样，所有模型 R² 都是负的。这里 one-hot 其实是在用溶剂/碱/底物插值，不是真的“认识新配体”。

开口句：

> 名字配体 + Morgan 不能当通用配体编码器。双膦和单膦要分开，不能混在一个指纹空间里外推。

---

## 4. 不确定性与 BO

Split conformal（multi-FP XGB，随机 hold-out）：

- 80% 区间：目标 0.80，实际覆盖 **0.796**，半宽 8.1 收率点
- 90% 区间：目标 0.90，实际覆盖 **0.891**，半宽 11.4

覆盖率是校准过的，比上一版只报 bootstrap std 更硬。

离散 BO（BH reaction 0，8 seeds，init 8，再走 36 步，池子 n=790，池最大收率 86.6）：

| 策略 | 结束时平均最佳收率 |
|---|---:|
| random | 82.4 ± 1.8 |
| greedy | 85.0 ± 1.4 |
| UCB | 85.0 ± 1.5 |
| EI | 84.0 ± 2.2 |

这是回顾式池模拟，不是湿实验。池子里高收率条件不少，随机 init 就已经不低；UCB 仍稳定高于随机。不要说成“已经能替代 DOE”。

---

## 5. GNN 为什么先放下

同一随机划分：ligand GIN R² = 0.185，additive GIN R² = 0.081。树模型是 0.94。

配体只有 4 个独特图，GNN 学的是 4 类 embedding，不是膦化学。添加剂有 22 个分子，理论上更合适，但当前训练预算下也打不过指纹树模型。会上可以主动说：“图模型接口还在，但这个数据尺度上不是正确工具。”

---

## 6. 和上一版比，到底新在哪

1. 评估从随机划分改成 leave-one-ligand / additive / product / electrophile。
2. 特征从“只指纹配体”改成多组分结构；**新添加剂、新产品、新卤代物**上 RMSE 下降是真的。
3. 主效应 + 残差把“加和效应”和交互分开，BH 插值 R² 从 0.91 到 0.94。
4. 第二反应类型（Suzuki）用来防 overfit 一条 BH 曲线。
5. conformal 覆盖率对得上名义水平。

上一版随机 R² 0.93 和这次 0.94 不是同一个故事：以前那是单反应、配体指纹、偏乐观划分；这次是全表 + 化学划分。

---

## 7. 他们可能会问

**Q：为什么不用 DFT 描述符（Ahneman 原文）？**  
A：公开 CSV 没有现成 DFT。多组分指纹是零计算描述符基线。下一步如果要对齐原文，应该复现 Ahneman RF + DFT，而不是再加一层 GNN。

**Q：4 个配体的 LOLO 有意义吗？**  
A：有，而且正好说明问题。BrettPhos 家族能迁，XPhos 不能。要做通用配体模型需要更宽的配体库（Kraken / 内部库），不能指望 Doyle 四配体。

**Q：Suzuki 的 label 是 area%？**  
A：是，和 BH 的 yield 不是同一物理量，所以两套数字不能横比绝对值，只能比趋势。

**Q：下一步最值得做的算法是什么？**  
A：不是更深的 GNN。是 (i) 配体家族内的局部模型 / 相似度门控；(ii) 添加剂指纹 + 协同克里金 / 分层贝叶斯；(iii) 和 DFT 描述符拼在一起做 ablation。BO 则应改成带杂质约束的多目标，一旦有内部表。

---

## 8. 建议讨论顺序（15 分钟）

1. 两张图：`bh_rmse_by_split.png`、`bh_lolo_per_ligand.png`（4 分钟）
2. `bh_loao_delta_rmse.png`：17/22 添加剂结构有用（3 分钟）
3. Suzuki 配体条形图：Xantphos 是机制外点（2 分钟）
4. conformal + BO 遗憾曲线（3 分钟）
5. 问他们：下一组公开或内部配体库，要不要按家族分层，而不是再训一个全局 GNN（3 分钟）

---

## 9. English, if they switch

> Random-split R² of 0.94 is interpolation. The chemically relevant tests are leave-one-component-out. Multi-component fingerprints beat ligand-only and one-hot on unseen additives (17/22) and unseen products. They do not invent a universal ligand encoder: BrettPhos transfers, XPhos does not, and Xantphos on Suzuki breaks every model. GNNs are not competitive with four unique ligand graphs. Uncertainty is split-conformal and empirically calibrated. Discrete BO beats random modestly in a retrospective pool — not a wet-lab claim.

---

## 10. 文件

- 报告：`hte_scientific_outputs/report.html`
- 图：`hte_scientific_outputs/figures/`
- 原数字：`hte_scientific_outputs/summary.json`
- 复现：`python3 hte_scientific_benchmark.py`
