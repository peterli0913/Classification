# Andrew 会议讨论稿（HTE / Ligand / BO + 凯莱英生产 AI）

适用场景：15–25 分钟同步会。近期带宽有限、新实验进展不多时，用“已落地能力 + 诚实缺口 + 要拍板的事”把会开顺。

---

## 0. 开场（约 60 秒，建议原话）

最近这块我推进得慢一些，主要时间花在把已有工作收成一条能演示、能讨论的链路，而不是再堆新模型。

今天我想对齐三件事：

1. **现在手里到底有什么**：ligand 描述符分析、真实收率代理模型、离散贝叶斯优化推荐。
2. **哪些结论可以信、哪些还不能拿去指导实验**。
3. **下一步如果要接到凯莱英内部数据或生产场景，优先做哪一条**。

我先用两分钟把现状说清楚，然后想听你们对数据、验证方式和落地场景的判断。

---

## 1. 一句话现状（给 Andrew 的 mental model）

我们现在做的不是“再训一个更高分的黑盒”，而是一条 **HTE 条件推荐的最小闭环**：

`公开 Buchwald–Hartwig 收率数据 → 可解释/可校准的代理模型 → 不确定性 → 在离散候选池里用 EI/UCB 推荐下一组实验`

这条链路已经能跑、能演示。还没接到凯莱英自己的反应和产线数据，所以今天更适合讨论 **方法是否对、数据怎么接、生产上值不值得做**，而不是报最终工艺结论。

---

## 2. 已经做成的事（只讲能站得住的）

### 2.1 Ligand / 描述符侧（Kraken monophosphine）

- 数据：237 个单膦配体，O–AE 描述符 + SMILES。
- 模型：XGBoost、组 attention、SMILES+描述符混合、简单集成。
- 测试集（当前 label 是 `vbur_ratio_vbur_vtot`，**不是真实收率**）：
  - XGBoost：RMSE 0.0182，R² 0.857
  - SMILES hybrid：R² 0.760
  - Attention：R² 0.542（弱，但能看组权重）
- 可解释信号：`sterimol_B1`、`E_solv_total`、`sterimol_B5` 较重要；组 attention 更偏向 U–W / X–AE。
- 用途：讲“空间/溶剂化描述符可能比单纯电子项更敏感”，**不要把它说成收率机理结论**。

开口句：

> 描述符这条线证明了小样本下树模型仍然最稳，attention 更适合解释而不是当主力预测。但当时的 label 是 buried volume ratio，不是 yield，所以它只是方法验证，不是工艺结论。

### 2.2 真实收率 + 离散 BO（Doyle BH 公开数据）

- 数据：`bh-reactions.csv`，约 3955 条 Buchwald–Hartwig，label 是真实 yield。
- 当前主力：RDKit 指纹 + XGBoost bootstrap（均值 + 标准差）。
- Reaction 0 上一次完整训练大约是：
  - Bootstrap：RMSE ≈ 7.9，R² ≈ 0.93（随机划分，偏乐观）
  - Quantile：R² ≈ 0.88
  - Murcko scaffold GroupKFold：R² ≈ 0.52（这个数字更接近“新骨架外推”）
  - PyG-GCN：目前没有超过指纹树模型，小数据下 GNN 还不是主力
- BO：在离散 combinatorial 池里按 EI / UCB 排序，输出 next experiments，不是生成新分子。
- 演示：本地完整 dashboard；Render 免费实例内存只有 512MB，线上是轻量版（指纹 + bootstrap + EI）。

开口句：

> 随机划分 R² 0.93 只能说明“见过的化学空间拟合得住”。scaffold 掉到 0.5 左右，说明换骨架后不确定性会明显上去。这对 BO 其实是好事：模型会把没把握的区域标出来，而不是假装很准。

### 2.3 现在能给实验同事看的产品形态

- 输入：ligand SMILES + aryl halide / additive / base 等条件
- 输出：预测收率、不确定性、EI/UCB 推荐的下一组条件
- 还不能承诺：新配体发明、跨反应类型迁移、直接上生产批

---

## 3. 诚实缺口（主动说，避免被问穿）

| 缺口 | 怎么说 | 为什么现在先不硬补 |
|---|---|---|
| 近期没有新的内部实验闭环 | 公开数据把方法跑通了，还没接到我们自己的 HTE / 工艺批次 | 没有内部表，优化目标会对不准 |
| GNN 还没打赢 XGB | 图模型已经接上，但当前样本和训练预算下不如指纹树模型 | 生产上先要稳和可解释，不先追架构 |
| 线上是轻量版 | Render 512MB 跑不动 PyTorch/PyG | 演示用轻量，完整训练放本地/内网 |
| 离散 BO 不是分子生成 | 现在是在已有配体/添加剂/碱/卤代物组合里排序 | 先把“下一锅做什么”做对，再生新结构 |
| 全池打分偏笨 | 候选池一大，每次对全池算 EI 会慢 | 下一步可以两阶段：XGB 粗筛 → 不确定度精排 |

如果被问 “那你最近做出新 SOTA 了吗”：

> 没有。最近我更在意闭环能不能被工艺同事用，而不是再刷一个公开集分数。公开集上树模型已经够强，再堆模型对接到我们自己的反应帮助有限。

---

## 4. 建议向 Andrew 要的 3 个决定

不要让会议停在“看看挺有意思”。直接要这三件事：

### 决定 A：下一阶段用哪套内部数据

优先顺序建议你这样提：

1. **一条已经做完的 HTE 反应**（配体/碱/溶剂/温度都扫过，有收率和杂质）
2. 其次：**同一反应的历史工艺批次**（参数 + IPC + 收率/纯度）
3. 先不要一上来要全公司数据湖

你要的最小字段：

- 反应标识、批次/孔位
- 条件：ligand、base、solvent、T、当量、浓度、时间
- 结构：至少 SMILES 或内部 ID 能对上结构
- 结果：assay yield、isolated yield、关键杂质
- 可选：HPLC、PAT、安全/放热

### 决定 B：优化目标怎么定义

生产上很少是“只追最高收率”。建议当场问：

- 主目标：收率、纯度、还是杂质上限？
- 硬约束：成本、周期、催化剂载量、溶剂回收、安全窗口
- 成功标准：比现有 DOE / 专家经验 **少做多少组、多拿到多少合格条件**

可提一个务实目标：

> 先做单反应、有约束的离散 BO：10–20 组建议实验里，打到现有最优附近，并把失败条件的原因说清楚。

### 决定 C：落地放在工艺开发，还是先不碰 GMP 生产

建议你明确主张：

> **先 PD / HTE，不先上 GMP 放行。**  
> 开发阶段容错高、数据密、决策快；生产上模型要进质量体系，现在还没到。

---

## 5. 讨论题（把空气递给他们）

1. 你们现在一条典型 HTE / 工艺优化，大概做多少组、周期多长、谁拍板下一组条件？
2. 配体筛选和条件筛选，哪个更痛？是“候选太多”还是“做了也不知道为什么好”？
3. 内部有没有已经结构化的表，还是还在 ELN / Excel？
4. 如果模型推荐和专家直觉冲突，你们希望系统怎么表现：强提醒、只排序、还是必须给理由？
5. 有没有一条即将启动的反应，适合做 4–6 周的对照试点（专家选 vs 模型选）？

---

## 6. 凯莱英生产 / 工艺上的 AI，可以怎么谈

不要一上来讲大模型平台。按 **离生产越近、越要保守；离筛选越近、越能快试** 来说。

### 6.1 和今天工作直接接得上的（优先谈）

**反应条件 / 配体筛选的闭环优化（PD）**

- 痛点：HTE 组合爆炸，专家经验强，但下一组实验常靠直觉。
- 我们已有：代理模型 + 不确定性 + 离散推荐。
- 接到凯莱英后多出来的价值：内部底物、内部配体库、杂质约束，而不是再预测一次 Doyle 公开集。
- 讨论点：先选 **1 条金属催化偶联或已有 HTE 的项目** 做对照。

**失败模式学习，不只预测高收率**

- 生产/中试更关心：为什么放大后掉收率、哪个杂质超标。
- 思路：把“低收率 / 高杂质”当单独标签，模型输出风险，而不是只给一个 yield 数字。
- 讨论点：历史失败批次有没有被结构化留下来。

**描述符 + 结构双轨，给工艺同事解释**

- 树模型给重要性，attention / 基团给出“立体还是电子在起作用”。
- 这对凯莱英有用：配体采购和库存决策需要理由，不能只给 SMILES 排名。

### 6.2 中期、值得探一下口风的

**工艺参数放大（lab → kilo → plant）**

- 实验室最优不等于车间可执行。
- AI 更适合做：给定设备约束（传热、搅拌、加料时间、溶剂回收），预测“会不会偏出质量窗口”，而不是直接改 SOP。
- 需要：同一反应跨尺度的批次数据。没有这个，不承诺放大模型。

**杂质谱 / 规格风险**

- CDMO 放行看的是规格，不是最高收率。
- 可做：关键杂质超标概率、哪类条件会推高某杂质。
- 比生成新路线更贴近现有质量语言。

**溶剂 / 试剂替代（绿色化学 + 供应链）**

- 缺溶剂、涨价、EHS 限制时，用模型在“相近选择性”的溶剂里找替代。
- 凯莱英连续流和绿色工艺本身就有积累，适合做成受限推荐，而不是开放生成。

### 6.3 可以提、但不要承诺排期的

- ELN / 实验记录转成可训练表（这往往是真正瓶颈）
- 文献/专利条件检索后的可执行 recipe 草案
- 车间排产、campaign 切换（更偏运筹，不是化学模型）
- GMP 放行辅助、偏差调查（质量体系门槛高，第二阶段再说）
- 通用大模型直接“开处方”（厅里听听就好，不作为本季度目标）

### 6.4 你可以用的判断句

> 凯莱英的优势不在再做一个公开反应的预测冠军，而在 **自己的反应库、设备约束和杂质规格**。AI 应该嵌进“下一组实验 / 下一批参数”的决策，而不是另做一套演示系统。

> 我的建议是三层：  
> 1）开发阶段：HTE + BO，现在就能试点；  
> 2）中试：参数窗口和杂质风险，有跨批次数据再做；  
> 3）GMP 生产：先做监测和预警，不做自动改工艺。

---

## 7. 如果他们问“你觉得最该做什么”

按这个优先级答：

1. **选 1 条内部反应，把现有离散 BO 接上真实表**（4–6 周能看出有没有少做实验）。
2. **把目标从纯收率改成收率 + 杂质/成本约束**（否则推荐结果工艺上不可用）。
3. **两阶段推荐**：指纹模型粗筛，再对短名单算不确定度和多样性，避免全池穷举。
4. GNN / 生成新配体先放后面；等内部数据稳定、树模型基线站稳再加。

依据：公开数据上树模型已经明显强于当前 GNN；生产决策更吃可解释和约束，不吃架构新鲜感。

不确定的地方：内部 HTE 的样本量、标签质量、以及工艺同事是否接受“模型先推荐、人再否决”的工作流。这些要会上问。

---

## 8. 时间不够时的 8 分钟版

1. 开场 60 秒（上面原文）
2. 描述符线：方法通了，label 不是 yield（40 秒）
3. 收率 + BO：随机 R² 高、scaffold R² 约 0.5，所以必须带不确定度（90 秒）
4. 缺口：没内部数据、GNN 未超 XGB、线上轻量（40 秒）
5. 要 3 个决定：数据、目标、先 PD 不先 GMP（90 秒）
6. 生产 AI：先闭环筛选，再杂质/放大，最后放行（90 秒）
7. 留问题：哪条反应适合试点（剩下时间）

---

## 9. 英文提要（Andrew 若切英文可直接用）

**Opener**

> I have been slower on new experiments recently. What I did instead is package the current work into a closed loop we can actually discuss: public Buchwald–Hartwig yields, an uncertainty-aware surrogate, and discrete BO suggestions for the next experiments. I want to align on what is trustworthy, what is not, and which internal dataset or process problem is worth connecting next.

**What works**

> Tree models on fingerprints are the workhorse. Random-split R² looks strong (~0.93) but scaffold split drops to ~0.5, so we should not treat this as ready for new ligand cores. GNN is wired, not winning. BO ranks existing condition combinations; it does not invent molecules.

**Ask**

> If we do one thing next, I would pick a single internal HTE or process-development reaction, optimize yield under impurity and cost constraints, and compare model-chosen vs expert-chosen experiments. I would not start with GMP release.

**Production AI**

> Asymchem’s advantage is internal chemistry, equipment constraints, and spec limits. The useful AI is decision support for the next experiment or the next parameter window, not another public-leaderboard model.

---

## 10. 会后可以发的三行纪要（留白）

- 今天对齐的现状：________________________________
- 内部试点反应 / 数据接口人：____________________
- 两周内下一步：________________________________
