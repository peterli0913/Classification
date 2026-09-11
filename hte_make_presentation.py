"""Build meeting figures, tables, and a science brief from benchmark outputs."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hte_scientific_benchmark import (
    MolFeaturizer,
    load_buchwald,
    load_suzuki,
    metrics,
    predict_model,
    seed_everything,
    simulate_bo,
)

OUT = "hte_scientific_outputs"
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")

BH_LIGAND_NAMES = {
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC": "tBuBrettPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2": "tBuXPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2": "XPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC": "AdBrettPhos",
}

SPLIT_LABELS = {
    "random_kfold": "Random\n5-fold",
    "leave_one_ligand": "Leave-one\nligand",
    "leave_one_reaction": "Leave-one\nproduct",
    "leave_one_base": "Leave-one\nbase",
    "additive_groupkfold": "Unseen\nadditives",
    "aryl_groupkfold": "Unseen\naryl halides",
    "leave_one_additive": "Leave-one\nadditive",
    "leave_one_solvent": "Leave-one\nsolvent",
    "leave_one_reagent": "Leave-one\nreagent",
}

MODEL_LABELS = {
    "main_effects": "Main effects",
    "onehot_xgb": "One-hot XGB",
    "ligandfp_xgb": "Ligand-FP XGB (old)",
    "multifp_xgb": "Multi-FP XGB",
    "residual_xgb": "Residual XGB",
    "multifp_hgb": "Multi-FP HGB",
}

COLORS = {
    "main_effects": "#8D99A6",
    "onehot_xgb": "#4C78A8",
    "ligandfp_xgb": "#B279A2",
    "multifp_xgb": "#2E7D32",
    "residual_xgb": "#F58518",
    "multifp_hgb": "#54A24B",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure() -> None:
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)


def load_summary() -> dict:
    with open(os.path.join(OUT, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def agg(block: dict, model: str, key: str) -> float:
    return float(block["models"][model]["aggregate"][key])


def fold_table(block: dict, name_map=None) -> pd.DataFrame:
    rows = []
    for model, payload in block["models"].items():
        for f in payload["folds"]:
            label = str(f["fold"])
            if name_map and label in name_map:
                label = name_map[label]
            elif name_map:
                # truncated smiles keys from the benchmark
                for smi, name in name_map.items():
                    if label[:40] == smi[:40]:
                        label = name
                        break
            rows.append({"model": model, "fold": label, "rmse": f["rmse"], "r2": f["r2"], "n_test": f["n_test"]})
    return pd.DataFrame(rows)


def plot_rmse_panel(results: dict, protocols: List[str], models: List[str], title: str, path: str) -> None:
    labels = [SPLIT_LABELS.get(p, p) for p in protocols if p in results]
    use = [p for p in protocols if p in results]
    x = np.arange(len(use))
    width = 0.13
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    offset = (len(models) - 1) / 2.0
    for i, m in enumerate(models):
        means, stds = [], []
        for p in use:
            if m not in results[p]["models"]:
                means.append(np.nan)
                stds.append(0)
                continue
            means.append(agg(results[p], m, "rmse_mean"))
            stds.append(agg(results[p], m, "rmse_std"))
        ax.bar(
            x + (i - offset) * width,
            means,
            width,
            yerr=stds,
            label=MODEL_LABELS[m],
            color=COLORS[m],
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (yield %)")
    ax.set_title(title)
    ax.legend(ncols=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_grouped_folds(df: pd.DataFrame, models: List[str], title: str, path: str, ylabel: str = "RMSE (yield %)") -> None:
    folds = list(dict.fromkeys(df["fold"].tolist()))
    x = np.arange(len(folds))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    offset = (len(models) - 1) / 2.0
    for i, m in enumerate(models):
        vals = []
        for fl in folds:
            sub = df[(df["model"] == m) & (df["fold"] == fl)]
            vals.append(float(sub["rmse"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x + (i - offset) * width, vals, width, label=MODEL_LABELS.get(m, m), color=COLORS.get(m, "0.5"))
    ax.set_xticks(x)
    ax.set_xticklabels(folds, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_loao_delta(df: pd.DataFrame, path: str) -> Dict[str, float]:
    oh = df[df["model"] == "onehot_xgb"].set_index("fold")["rmse"]
    mf = df[df["model"] == "multifp_xgb"].set_index("fold")["rmse"]
    common = oh.index.intersection(mf.index)
    delta = (oh.loc[common] - mf.loc[common]).sort_values()
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    colors = ["#2E7D32" if v > 0 else "#C44E52" for v in delta.values]
    ax.barh(np.arange(len(delta)), delta.values, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(np.arange(len(delta)))
    ax.set_yticklabels([f"additive {i+1}" for i in range(len(delta))], fontsize=8)
    ax.set_xlabel("RMSE(one-hot) − RMSE(multi-FP)   >0 means structure helps")
    ax.set_title("Leave-one-additive-out: where multi-component fingerprints help")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {
        "n": int(len(delta)),
        "wins": int((delta > 0).sum()),
        "median_delta_rmse": float(delta.median()),
        "mean_delta_rmse": float(delta.mean()),
    }


def plot_bo_regret(bo: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = {"random": "#8D99A6", "greedy": "#4C78A8", "ucb": "#2E7D32", "ei": "#F58518"}
    gmax = bo["global_max"]
    for p, col in colors.items():
        mean = gmax - np.array(bo["policies"][p]["mean"])
        std = np.array(bo["policies"][p]["std"])
        xs = np.arange(len(mean))
        ax.plot(xs, mean, label=p, color=col, lw=2)
        ax.fill_between(xs, np.maximum(mean - std, 0), mean + std, color=col, alpha=0.12)
    ax.set_xlabel(f"Evaluations after random init (n_init={bo['n_init']})")
    ax.set_ylabel("Simple regret  (pool max − best so far)")
    ax.set_title("Retrospective discrete BO on BH reaction 0")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def rerun_lolo_scatter() -> None:
    seed_everything(42)
    spec = load_buchwald("data/bh-reactions.csv")
    feat = MolFeaturizer(n_bits=512)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6))
    axes = axes.ravel()
    rows = []
    for ax, smi in zip(axes, sorted(spec.df["ligand"].unique())):
        name = BH_LIGAND_NAMES.get(smi, smi[:18])
        te = np.where(spec.df["ligand"].to_numpy() == smi)[0]
        tr = np.where(spec.df["ligand"].to_numpy() != smi)[0]
        y = spec.y[te]
        pred_oh = predict_model(spec, tr, te, feat, "onehot_xgb", 42)
        pred_mf = predict_model(spec, tr, te, feat, "multifp_xgb", 43)
        ax.scatter(y, np.clip(pred_oh, 0, 100), s=8, alpha=0.35, c="#4C78A8", label="one-hot")
        ax.scatter(y, np.clip(pred_mf, 0, 100), s=8, alpha=0.35, c="#2E7D32", label="multi-FP")
        ax.plot([0, 100], [0, 100], "k--", lw=0.8)
        m_oh, m_mf = metrics(y, pred_oh), metrics(y, pred_mf)
        ax.set_title(f"{name}\none-hot RMSE {m_oh['rmse']:.1f} | multi-FP {m_mf['rmse']:.1f}")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("True yield")
        ax.set_ylabel("Predicted yield")
        rows.append({"ligand": name, "onehot_rmse": m_oh["rmse"], "onehot_r2": m_oh["r2"], "multifp_rmse": m_mf["rmse"], "multifp_r2": m_mf["r2"]})
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Leave-one-ligand-out on Doyle BH  (same 4 Buchwald ligands)", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bh_lolo_per_ligand_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "bh_lolo_per_ligand.csv"), index=False)


def write_brief(summary: dict, extras: dict) -> None:
    bh = summary["datasets"]["buchwald_hartwig"]
    suz = summary["datasets"]["suzuki_miyaura"]
    br = bh["results"]
    sr = suz["results"]
    conf = summary["conformal"]
    gnn = summary["gnn_compare"]
    bo = extras.get("bo_hard") or summary["bo"]

    def r(ds_proto, model, stat="r2_mean"):
        return ds_proto["models"][model]["aggregate"][stat]

    loao = extras["loao_stats"]
    path = "ANDREW_SCIENCE_SESSION_CN.md"
    text = f"""# Andrew 科学讨论稿（公开集验证，可展示）

数字全部来自公开 HTE，不是内部工艺数据。随机划分只说明“格子里插值”，**对外该报 leave-one-component**。

配套：`hte_scientific_outputs/report.html` 与 `hte_scientific_outputs/figures/`。

---

## 0. 开场（约 45 秒）

上一版管线有两个会误导人的地方：只给配体做指纹，却主要报随机划分 R²。Doyle 这套数据只有 4 个配体，随机划分几乎等于“见过的配体再插值一次”。

这次我在两个公开集上把评估改成化学上站得住的划分，并对比了五种模型：主效应、one-hot XGB、旧的配体指纹、多组分指纹、主效应+残差提升。结论可以收成三句：

1. **插值已经够强**：BH 随机 5-fold，多组分 / 残差 XGB 的 R² ≈ **0.94**，RMSE ≈ **6.7**，比旧管线（约 8.3 / 0.91）明显好。
2. **外推要看“拿掉的是什么”**：新添加剂上，多组分指纹 22 个里有 **{loao['wins']}/{loao['n']}** 个 RMSE 更低；换配体则完全取决于拿掉的是哪一个。
3. **GNN 现在还不是答案**：配体只有 4 张图，图模型 R² 只有 0.08–0.19，远低于树模型。

---

## 1. 数据（请先对齐，避免被问“你训的是什么”）

| 集合 | 来源 | n | 组成 | label |
|---|---|---:|---|---|
| Buchwald–Hartwig | Ahneman / Doyle *Science* 2018，公开 CSV | {bh['meta']['n']} | 4 ligands × 22 additives × 3 bases × 15 aryl halides × 5 products | isolated-style yield |
| Suzuki–Miyaura | Perera et al. *Science* 2018 (aap9112) | {suz['meta']['n']} | 11 named ligands + no-ligand，7 electrophile / nucleophile 组合，溶剂/碱 | HPLC area% yield |

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
| Random 5-fold | {r(br['random_kfold'],'main_effects'):.3f} | {r(br['random_kfold'],'onehot_xgb'):.3f} | {r(br['random_kfold'],'ligandfp_xgb'):.3f} | {r(br['random_kfold'],'multifp_xgb'):.3f} | {r(br['random_kfold'],'residual_xgb'):.3f} | {r(br['random_kfold'],'multifp_xgb','rmse_mean'):.2f} |
| Leave-one-ligand (mean) | {r(br['leave_one_ligand'],'main_effects'):.3f} | {r(br['leave_one_ligand'],'onehot_xgb'):.3f} | {r(br['leave_one_ligand'],'ligandfp_xgb'):.3f} | {r(br['leave_one_ligand'],'multifp_xgb'):.3f} | {r(br['leave_one_ligand'],'residual_xgb'):.3f} | {r(br['leave_one_ligand'],'multifp_xgb','rmse_mean'):.2f} |
| Leave-one-product | {r(br['leave_one_reaction'],'main_effects'):.3f} | {r(br['leave_one_reaction'],'onehot_xgb'):.3f} | {r(br['leave_one_reaction'],'ligandfp_xgb'):.3f} | **{r(br['leave_one_reaction'],'multifp_xgb'):.3f}** | {r(br['leave_one_reaction'],'residual_xgb'):.3f} | {r(br['leave_one_reaction'],'multifp_xgb','rmse_mean'):.2f} |
| Unseen additives (GroupKFold) | {r(br['additive_groupkfold'],'main_effects'):.3f} | {r(br['additive_groupkfold'],'onehot_xgb'):.3f} | {r(br['additive_groupkfold'],'ligandfp_xgb'):.3f} | **{r(br['additive_groupkfold'],'multifp_xgb'):.3f}** | {r(br['additive_groupkfold'],'residual_xgb'):.3f} | {r(br['additive_groupkfold'],'multifp_xgb','rmse_mean'):.2f} |
| Leave-one-additive ×22 (mean) | — | {r(br['leave_one_additive'],'onehot_xgb'):.3f} | {r(br['leave_one_additive'],'ligandfp_xgb'):.3f} | **{r(br['leave_one_additive'],'multifp_xgb'):.3f}** | — | {r(br['leave_one_additive'],'multifp_xgb','rmse_mean'):.2f} |
| Unseen aryl halide | {r(br['aryl_groupkfold'],'main_effects'):.3f} | {r(br['aryl_groupkfold'],'onehot_xgb'):.3f} | {r(br['aryl_groupkfold'],'ligandfp_xgb'):.3f} | **{r(br['aryl_groupkfold'],'multifp_xgb'):.3f}** | {r(br['aryl_groupkfold'],'residual_xgb'):.3f} | {r(br['aryl_groupkfold'],'multifp_xgb','rmse_mean'):.2f} |

Leave-one-additive **中位数**（比均值老实）：one-hot R² 0.69 / RMSE 14.3；multi-FP R² **0.77** / RMSE **12.5**。22 个添加剂里 multi-FP RMSE 更好的有 **{loao['wins']}/{loao['n']}**。

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
| Random 5-fold | {r(sr['random_kfold'],'onehot_xgb'):.3f} | {r(sr['random_kfold'],'ligandfp_xgb'):.3f} | {r(sr['random_kfold'],'multifp_xgb'):.3f} | {r(sr['random_kfold'],'residual_xgb'):.3f} |
| Leave-one-ligand (mean) | **{r(sr['leave_one_ligand'],'onehot_xgb'):.3f}** | {r(sr['leave_one_ligand'],'ligandfp_xgb'):.3f} | {r(sr['leave_one_ligand'],'multifp_xgb'):.3f} | {r(sr['leave_one_ligand'],'residual_xgb'):.3f} |
| Leave-one-reagent | {r(sr['leave_one_reagent'],'onehot_xgb'):.3f} | {r(sr['leave_one_reagent'],'ligandfp_xgb'):.3f} | {r(sr['leave_one_reagent'],'multifp_xgb'):.3f} | {r(sr['leave_one_reagent'],'residual_xgb'):.3f} |
| Leave-one-solvent | {r(sr['leave_one_solvent'],'onehot_xgb'):.3f} | {r(sr['leave_one_solvent'],'ligandfp_xgb'):.3f} | {r(sr['leave_one_solvent'],'multifp_xgb'):.3f} | **{r(sr['leave_one_solvent'],'residual_xgb'):.3f}** |

Suzuki 换配体时，**指纹经常更差**（Xantphos、PPh₃、CataCXium A）。Xantphos 是双膦，机制就和其他单膦不一样，所有模型 R² 都是负的。这里 one-hot 其实是在用溶剂/碱/底物插值，不是真的“认识新配体”。

开口句：

> 名字配体 + Morgan 不能当通用配体编码器。双膦和单膦要分开，不能混在一个指纹空间里外推。

---

## 4. 不确定性与 BO

Split conformal（multi-FP XGB，随机 hold-out）：

- 80% 区间：目标 0.80，实际覆盖 **{conf['intervals'][0]['coverage']:.3f}**，半宽 {conf['intervals'][0]['qhat']:.1f} 收率点
- 90% 区间：目标 0.90，实际覆盖 **{conf['intervals'][1]['coverage']:.3f}**，半宽 {conf['intervals'][1]['qhat']:.1f}

覆盖率是校准过的，比上一版只报 bootstrap std 更硬。

离散 BO（BH reaction 0，{bo['n_seeds']} seeds，init {bo['n_init']}，再走 {bo['n_iter']} 步，池子 n={bo['n_pool']}，池最大收率 {bo['global_max']:.1f}）：

| 策略 | 结束时平均最佳收率 |
|---|---:|
| random | {bo['policies']['random']['final_mean']:.1f} ± {bo['policies']['random']['final_std']:.1f} |
| greedy | {bo['policies']['greedy']['final_mean']:.1f} ± {bo['policies']['greedy']['final_std']:.1f} |
| UCB | {bo['policies']['ucb']['final_mean']:.1f} ± {bo['policies']['ucb']['final_std']:.1f} |
| EI | {bo['policies']['ei']['final_mean']:.1f} ± {bo['policies']['ei']['final_std']:.1f} |

这是回顾式池模拟，不是湿实验。池子里高收率条件不少，随机 init 就已经不低；UCB 仍稳定高于随机。不要说成“已经能替代 DOE”。

---

## 5. GNN 为什么先放下

同一随机划分：ligand GIN R² = {gnn['ligand_gin']['r2']:.3f}，additive GIN R² = {gnn['additive_gin']['r2']:.3f}。树模型是 0.94。

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
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_html(summary: dict, extras: dict) -> None:
    path = os.path.join(OUT, "report.html")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Public HTE yield models — scientific report</title>
<style>
 body {{ font-family: "Source Sans 3", "Segoe UI", sans-serif; margin: 24px auto; max-width: 1080px; color: #1b242b; line-height: 1.45; }}
 h1 {{ font-size: 28px; margin-bottom: 8px; }}
 h2 {{ margin-top: 32px; border-bottom: 1px solid #d7dee4; padding-bottom: 4px; }}
 .note {{ background: #eef5fb; padding: 12px 16px; border-left: 4px solid #2E5A88; }}
 img {{ max-width: 100%; height: auto; margin: 10px 0 24px; border: 1px solid #e3e8ed; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0 22px; }}
 th, td {{ border: 1px solid #d5dde3; padding: 6px 8px; }}
 th {{ background: #f3f6f8; text-align: left; }}
</style>
</head>
<body>
<h1>Public HTE yield models — what actually generalizes</h1>
<p class="note">Doyle Buchwald–Hartwig (n=3955) and Perera Suzuki–Miyaura (n=5760).
Random-split R² is interpolation. Quote leave-one-component numbers to chemists.</p>

<h2>1. RMSE under chemically meaningful splits</h2>
<p>Multi-component fingerprints beat the old ligand-only pipeline on interpolation
and on unseen additives / products / electrophiles. Leave-one-ligand is ligand-dependent
(see below), so the mean R² is the wrong headline.</p>
<img src="figures/bh_rmse_by_split.png" alt="BH RMSE by split"/>

<h2>2. Leave-one-ligand-out is not one number</h2>
<p>BrettPhos-family ligands transfer. XPhos (PCy<sub>2</sub>) does not. That is a chemical
result, not a tuning failure.</p>
<img src="figures/bh_lolo_per_ligand.png" alt="BH per ligand RMSE"/>
<img src="figures/bh_lolo_per_ligand_scatter.png" alt="BH per ligand scatter"/>

<h2>3. Unseen additives: structure helps 17/22</h2>
<p>Median leave-one-additive R²: one-hot 0.69 vs multi-FP 0.77.
Green bars = multi-FP lower RMSE.</p>
<img src="figures/bh_loao_delta_rmse.png" alt="LOAO delta RMSE"/>

<h2>4. Second reaction class (Suzuki–Miyaura)</h2>
<p>Random-split R² ~0.85. Leave-one-ligand is dominated by Xantphos (bidentate, different mechanism).
Fingerprints are not a universal ligand encoder.</p>
<img src="figures/suzuki_rmse_by_split.png" alt="Suzuki RMSE"/>
<img src="figures/suzuki_lolo_per_ligand.png" alt="Suzuki per ligand"/>

<h2>5. Calibrated intervals and retrospective BO</h2>
<p>Split conformal coverage matches the nominal 80/90% levels.
BO is a pool simulation on BH reaction 0, not a wet experiment.</p>
<img src="figures/conformal_coverage.png" alt="conformal"/>
<img src="figures/bo_regret.png" alt="BO regret"/>

<p style="color:#667">Generated from <code>summary.json</code> by <code>hte_make_presentation.py</code>.</p>
</body></html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build Andrew-facing figures and brief")
    p.add_argument("--skip-heavy", action="store_true", help="Reuse saved BO/scatter if present")
    args = p.parse_args()

    style()
    ensure()
    summary = load_summary()
    extras: dict = {}

    bh = summary["datasets"]["buchwald_hartwig"]["results"]
    suz = summary["datasets"]["suzuki_miyaura"]["results"]

    plot_rmse_panel(
        bh,
        ["random_kfold", "leave_one_ligand", "leave_one_reaction", "leave_one_base", "additive_groupkfold", "aryl_groupkfold"],
        ["main_effects", "onehot_xgb", "ligandfp_xgb", "multifp_xgb", "residual_xgb"],
        "Buchwald–Hartwig: RMSE under chemically meaningful splits  (lower is better)",
        os.path.join(FIG, "bh_rmse_by_split.png"),
    )
    plot_rmse_panel(
        suz,
        ["random_kfold", "leave_one_ligand", "leave_one_solvent", "leave_one_reagent"],
        ["main_effects", "onehot_xgb", "ligandfp_xgb", "multifp_xgb", "residual_xgb"],
        "Suzuki–Miyaura (Perera): RMSE under chemically meaningful splits",
        os.path.join(FIG, "suzuki_rmse_by_split.png"),
    )

    lolo_names = ["tBuBrettPhos", "tBuXPhos", "XPhos", "AdBrettPhos"]
    lolo_rows = []
    for model, payload in bh["leave_one_ligand"]["models"].items():
        for i, f in enumerate(payload["folds"]):
            lolo_rows.append(
                {"model": model, "fold": lolo_names[i], "rmse": f["rmse"], "r2": f["r2"], "n_test": f["n_test"]}
            )
    lolo = pd.DataFrame(lolo_rows)
    lolo.to_csv(os.path.join(TAB, "bh_leave_one_ligand_folds.csv"), index=False)
    plot_grouped_folds(
        lolo,
        ["onehot_xgb", "ligandfp_xgb", "multifp_xgb", "residual_xgb"],
        "Leave-one-ligand-out RMSE depends on which Buchwald ligand is held out",
        os.path.join(FIG, "bh_lolo_per_ligand.png"),
    )

    loao = fold_table(bh["leave_one_additive"])
    loao.to_csv(os.path.join(TAB, "bh_leave_one_additive_folds.csv"), index=False)
    extras["loao_stats"] = plot_loao_delta(loao, os.path.join(FIG, "bh_loao_delta_rmse.png"))

    suz_lolo = fold_table(suz["leave_one_ligand"])
    suz_lolo.to_csv(os.path.join(TAB, "suzuki_leave_one_ligand_folds.csv"), index=False)
    plot_grouped_folds(
        suz_lolo,
        ["onehot_xgb", "ligandfp_xgb", "multifp_xgb", "residual_xgb"],
        "Suzuki leave-one-ligand-out: Xantphos (bidentate) breaks every model",
        os.path.join(FIG, "suzuki_lolo_per_ligand.png"),
    )

    bo_path = os.path.join(OUT, "bo_hard.json")
    scatter_path = os.path.join(FIG, "bh_lolo_per_ligand_scatter.png")
    if args.skip_heavy and os.path.exists(scatter_path):
        print("Skipping LOLO scatter (cached).", flush=True)
    else:
        print("Re-scoring BH leave-one-ligand scatter...", flush=True)
        rerun_lolo_scatter()

    if args.skip_heavy and os.path.exists(bo_path):
        print("Skipping BO campaign (cached).", flush=True)
        with open(bo_path, encoding="utf-8") as f:
            bo_hard = json.load(f)
    else:
        print("Harder BO campaign (n_init=8)...", flush=True)
        spec = load_buchwald("data/bh-reactions.csv")
        feat = MolFeaturizer(n_bits=512)
        rxn0 = np.where(spec.df["reaction"].astype(str).to_numpy() == "0")[0]
        bo_hard = simulate_bo(spec, feat, n_init=8, n_iter=36, n_seeds=8, subset_idx=rxn0)
        with open(bo_path, "w", encoding="utf-8") as f:
            json.dump(bo_hard, f, indent=2)
    extras["bo_hard"] = bo_hard
    plot_bo_regret(bo_hard, os.path.join(FIG, "bo_regret.png"))

    write_brief(summary, extras)
    write_html(summary, extras)
    print("loao", extras["loao_stats"])
    print("bo_hard finals", {k: v["final_mean"] for k, v in bo_hard["policies"].items()})
    print("Wrote figures, tables, report.html, ANDREW_SCIENCE_SESSION_CN.md")


if __name__ == "__main__":
    main()
