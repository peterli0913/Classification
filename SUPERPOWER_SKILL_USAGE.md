# Superpower Skill 安装与项目应用说明

## 1) 在 GitHub 上选择高好评 superpower skill

通过 `gh search repos "superpower skill" --sort stars --order desc` 检索，当前最高星项目为：

- Repo: `obra/superpowers-skills`
- URL: https://github.com/obra/superpowers-skills
- Stars: 653（检索时）

本项目采用其中的 **Meta-Pattern Recognition** 作为方法增强技能。

## 2) “装上并使用”的落地方式

在 `ligand_superpower_pipeline.py` 中已将该技能落成到建模结构：

1. 把特征按用户定义分组（O-Q / R-T / U-W / X-AE）；
2. 使用 `DescriptorAttentionRegressor` 在组级别做 attention 编码；
3. 使用 `SmilesDescriptorRegressor` 在 SMILES 序列层做 attention 编码；
4. 同时保留 XGBoost 基线，并做多模型集成；
5. 用 2D 降维可视化检查“相似 label 的 ligand 是否聚集”。

这等价于把 “meta-pattern recognition” 的思想应用到模型中：在不同表征域（分组描述符、序列表征、树模型）中寻找共性影响模式，并输出可解释可视化。

## 3) 运行命令

训练与可视化：

```bash
python3 ligand_superpower_pipeline.py train \
  --excel-path "Kraken monophosphine coordinates AD Descriptors.xlsx" \
  --output-dir "ligand_outputs"
```

新 ligand 预测（示例）：

```bash
python3 ligand_superpower_pipeline.py predict \
  --model-dir "ligand_outputs" \
  --smiles "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1" \
  --descriptor-values "12.131655,5.013689,9.844823,8.238802,5.013689,7.835474,71.473986,3.014584,56.456887,306.760233,-0.121678,0.193131,0.762849,-0.061654,-18.908945,-0.439429,51.270247"
```

> 说明：当前脚本按 Excel 的 A-AF 列位映射默认把 `AF` 作为 label 列。如果你后续提供真实收率列名称，可在脚本中将 `label_col` 替换为该列进行复训。
