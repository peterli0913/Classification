"""Interactive expert dashboard with legacy + RDKit/PyG BO features."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from ligand_superpower_pipeline import (
    DescriptorAttentionRegressor,
    SmilesDescriptorRegressor,
    infer_descriptor_attention,
    infer_smiles_hybrid,
)
from yield_bo_rdkit_pyg_pipeline import (
    build_tabular_features,
    expected_improvement,
    load_artifacts,
    predict_bootstrap,
    predict_gnn,
)


class InferenceService:
    def __init__(
        self,
        model_dir: str,
        excel_path: str,
        bo2_output_dir: str,
        bo2_candidate_data: str,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.excel_path = Path(excel_path)
        self.fig_dir = self.model_dir / "figures"
        self.bo2_output_dir = Path(bo2_output_dir)
        self.bo2_candidate_data = Path(bo2_candidate_data)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        artifact_path = self.model_dir / "artifacts.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing artifacts file: {artifact_path}")

        artifacts = joblib.load(artifact_path)
        self.feature_cols: List[str] = artifacts["feature_cols"]
        self.scaler = artifacts["scaler"]
        self.vocab = artifacts["vocab"]
        self.max_smiles_len = int(artifacts["max_smiles_len"])
        self.group_indices = artifacts["descriptor_group_indices"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(str(self.model_dir / "xgb_model.json"))
        self.desc_model = DescriptorAttentionRegressor(
            feature_dim=len(self.feature_cols), group_indices=self.group_indices
        ).to(self.device)
        self.desc_model.load_state_dict(
            torch.load(self.model_dir / "descriptor_attention_model.pt", map_location=self.device)
        )
        self.desc_model.eval()
        self.smiles_model = SmilesDescriptorRegressor(
            vocab_size=len(self.vocab), descriptor_dim=len(self.feature_cols)
        ).to(self.device)
        self.smiles_model.load_state_dict(
            torch.load(self.model_dir / "smiles_hybrid_model.pt", map_location=self.device)
        )
        self.smiles_model.eval()

        summary_path = self.model_dir / "results_summary.json"
        self.summary = {}
        if summary_path.exists():
            self.summary = json.loads(summary_path.read_text(encoding="utf-8"))

        # RDKit + PyG artifacts (optional)
        self.bo2_ready = False
        self.bo2_summary: Dict[str, object] = {}
        self.bo2_top_recs: pd.DataFrame | None = None
        self.bo2_candidate_df: pd.DataFrame | None = None
        self.bo2_art = None
        self.bo2_boot = None
        self.bo2_gnn = None
        try:
            if (self.bo2_output_dir / "meta.json").exists():
                self.bo2_art, self.bo2_boot, self.bo2_gnn = load_artifacts(str(self.bo2_output_dir))
                if (self.bo2_output_dir / "summary.json").exists():
                    self.bo2_summary = json.loads((self.bo2_output_dir / "summary.json").read_text(encoding="utf-8"))
                if (self.bo2_output_dir / "top_recommendations.csv").exists():
                    self.bo2_top_recs = pd.read_csv(self.bo2_output_dir / "top_recommendations.csv")
                if self.bo2_candidate_data.exists():
                    self.bo2_candidate_df = pd.read_csv(self.bo2_candidate_data)
                self.bo2_ready = True
        except Exception:
            self.bo2_ready = False

    def _build_top_features(self) -> List[Dict[str, float]]:
        imp = self.xgb_model.feature_importances_
        order = np.argsort(imp)[::-1]
        return [{"feature": self.feature_cols[i], "importance": float(imp[i])} for i in order[:8]]

    def _load_sample_input(self) -> Dict[str, str]:
        if not self.excel_path.exists():
            return {"smiles": "", "descriptor_values": ""}
        df = pd.read_excel(self.excel_path)
        if df.empty:
            return {"smiles": "", "descriptor_values": ""}
        smiles = str(df.iloc[0, 9])
        values = df.iloc[0, 14:31].astype(float).tolist()
        return {"smiles": smiles, "descriptor_values": ",".join(str(v) for v in values)}

    def _bo2_summary_payload(self) -> Dict[str, object]:
        if not self.bo2_ready:
            return {"ready": False}
        top_rows = []
        if self.bo2_top_recs is not None and not self.bo2_top_recs.empty:
            keep_cols = [c for c in ["ligand", "additive", "base", "aryl halide", "pred_mean", "pred_std", "acq_ei"] if c in self.bo2_top_recs.columns]
            top_rows = self.bo2_top_recs[keep_cols].head(8).to_dict(orient="records")
        sample = {}
        if self.bo2_candidate_df is not None and not self.bo2_candidate_df.empty:
            row = self.bo2_candidate_df.iloc[0].to_dict()
            sample = {
                "smiles": str(row.get(self.bo2_art.smiles_col, "")),
                "reaction": str(row.get("reaction", "0")),
                "additive": str(row.get("additive", "")),
                "base": str(row.get("base", "")),
                "aryl_halide": str(row.get("aryl halide", "")),
            }
        return {
            "ready": True,
            "metrics": self.bo2_summary.get("test_metrics", {}),
            "scaffold_kfold": self.bo2_summary.get("scaffold_group_kfold", {}).get("aggregate", {}),
            "top_recommendations": top_rows,
            "sample_input": sample,
        }

    def summary_payload(self) -> Dict[str, object]:
        return {
            "feature_cols": self.feature_cols,
            "metrics": self.summary.get("metrics", {}),
            "label_col": self.summary.get("label_col", "label"),
            "top_features": self._build_top_features(),
            "sample_input": self._load_sample_input(),
            "bo2": self._bo2_summary_payload(),
        }

    def predict(self, smiles: str, descriptor_values: List[float]) -> Dict[str, object]:
        if len(descriptor_values) != len(self.feature_cols):
            raise ValueError(f"Expected {len(self.feature_cols)} descriptor values, got {len(descriptor_values)}")
        x_raw = np.array(descriptor_values, dtype=float).reshape(1, -1)
        x_scaled = self.scaler.transform(x_raw)
        xgb_pred = float(self.xgb_model.predict(x_raw)[0])
        desc_pred, group_weights, _ = infer_descriptor_attention(self.desc_model, x_scaled, self.device)
        smiles_pred, _ = infer_smiles_hybrid(
            self.smiles_model,
            [smiles],
            x_scaled,
            self.vocab,
            self.max_smiles_len,
            self.device,
        )
        return {
            "prediction": {
                "xgboost": xgb_pred,
                "descriptor_attention": float(desc_pred[0]),
                "smiles_hybrid": float(smiles_pred[0]),
                "ensemble_avg": float((xgb_pred + float(desc_pred[0]) + float(smiles_pred[0])) / 3.0),
            },
            "group_attention": {
                "O-Q": float(group_weights[0][0]),
                "R-T": float(group_weights[0][1]),
                "U-W": float(group_weights[0][2]),
                "X-AE": float(group_weights[0][3]),
            },
        }

    def bo2_predict(self, smiles: str, reaction: str, additive: str, base: str, aryl_halide: str) -> Dict[str, float]:
        if not self.bo2_ready:
            raise ValueError("RDKit+PyG artifacts not ready.")
        one = {
            self.bo2_art.smiles_col: smiles,
            "reaction": reaction,
            "additive": additive,
            "base": base,
            "aryl halide": aryl_halide,
        }
        df = pd.DataFrame([one])
        x = build_tabular_features(
            df=df,
            cond_cols=self.bo2_art.cond_cols,
            cond_maps=self.bo2_art.cond_maps,
            smiles_col=self.bo2_art.smiles_col,
            fp_bits=self.bo2_art.fp_bits,
        )
        mu, std = predict_bootstrap(self.bo2_boot, x)
        gnn = predict_gnn(
            self.bo2_gnn,
            df=df,
            cond_cols=self.bo2_art.cond_cols,
            cond_maps=self.bo2_art.cond_maps,
            smiles_col=self.bo2_art.smiles_col,
        )
        return {
            "bootstrap_mean": float(mu[0]),
            "bootstrap_std": float(std[0]),
            "gnn_pyg": float(gnn[0]),
            "ensemble_avg": float((mu[0] + gnn[0]) / 2.0),
        }

    def bo2_suggest(self, reaction: str, top_k: int) -> List[Dict[str, object]]:
        if not self.bo2_ready or self.bo2_candidate_df is None:
            raise ValueError("Candidate pool or artifacts not ready.")
        df = self.bo2_candidate_df.copy()
        if "reaction" in df.columns:
            df = df[df["reaction"].astype(str) == str(reaction)].reset_index(drop=True)
        if df.empty:
            return []
        x = build_tabular_features(
            df=df,
            cond_cols=self.bo2_art.cond_cols,
            cond_maps=self.bo2_art.cond_maps,
            smiles_col=self.bo2_art.smiles_col,
            fp_bits=self.bo2_art.fp_bits,
        )
        mu, std = predict_bootstrap(self.bo2_boot, x)
        gnn = predict_gnn(
            self.bo2_gnn,
            df=df,
            cond_cols=self.bo2_art.cond_cols,
            cond_maps=self.bo2_art.cond_maps,
            smiles_col=self.bo2_art.smiles_col,
        )
        blend = 0.7 * mu + 0.3 * gnn
        ei = expected_improvement(blend, std, best=float(np.max(blend)))
        out = df.copy()
        out["pred_blend"] = blend
        out["pred_std"] = std
        out["acq_ei"] = ei
        out = out.sort_values(["acq_ei", "pred_blend"], ascending=False).head(top_k)
        keep = [c for c in ["ligand", "additive", "base", "aryl halide", "pred_blend", "pred_std", "acq_ei"] if c in out.columns]
        return out[keep].to_dict(orient="records")


def build_index_html(summary_payload: Dict[str, object]) -> str:
    payload = json.dumps(summary_payload, ensure_ascii=False)
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ligand Demo Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; background:#f6f8fb; color:#1f2937; }
    .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
    .hero { background: linear-gradient(135deg, #0f172a, #1e3a8a); color: #fff; border-radius: 14px; padding: 22px; }
    .hero h1 { margin: 0 0 8px 0; font-size: 28px; }
    .hero p { margin: 4px 0; opacity: 0.95; }
    .toolbar { margin-top: 14px; display:flex; gap:10px; flex-wrap:wrap; }
    .chip { border:1px solid rgba(255,255,255,0.3); border-radius:16px; padding:4px 10px; font-size:12px; }
    .grid { display: grid; gap: 16px; }
    .metrics { grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); margin-top: 16px; }
    .row2 { display:grid; grid-template-columns: 1.2fr 1fr; gap:16px; }
    .fig-grid { display:grid; gap:16px; grid-template-columns:1fr 1fr; }
    .card { background:#fff; border-radius:12px; padding:16px; box-shadow: 0 2px 14px rgba(15,23,42,0.06); }
    .card h3 { margin:0 0 10px 0; font-size:16px; }
    .muted { color:#6b7280; font-size:13px; }
    .section-title { margin:26px 0 10px 0; font-size:20px; }
    .kpi { font-size:24px; font-weight:700; margin-top:8px; }
    .fig-grid img { width:100%; border:1px solid #e5e7eb; border-radius: 10px; }
    textarea, input { width:100%; padding:10px; border:1px solid #d1d5db; border-radius:8px; font-size:14px; box-sizing: border-box; }
    button { background:#1d4ed8; color:white; border:none; padding:10px 16px; border-radius:8px; cursor:pointer; font-weight:600; }
    button:hover { background:#1e40af; }
    .secondary { background:#334155; }
    .secondary:hover { background:#1f2937; }
    table { border-collapse: collapse; width:100%; }
    th, td { border-bottom:1px solid #e5e7eb; padding:8px; text-align:left; font-size:14px; }
    .talk { white-space: pre-wrap; background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:12px; line-height:1.6; }
    .mini-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:10px; }
    @media (max-width: 900px) { .row2,.fig-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1 id="heroTitle"></h1>
      <p id="heroSub1"></p>
      <p id="heroSub2"></p>
      <div class="toolbar">
        <button id="langBtn"></button>
        <button id="scriptBtn" class="secondary"></button>
        <button id="pdfBtn" class="secondary"></button>
        <span class="chip" id="langChip"></span>
      </div>
    </div>

    <h2 class="section-title" id="secPerf"></h2>
    <div class="grid metrics" id="metricCards"></div>
    <div class="row2">
      <div class="card">
        <h3 id="chartTitle"></h3>
        <canvas id="metricChart" height="160"></canvas>
      </div>
      <div class="card">
        <h3 id="topTitle"></h3>
        <table>
          <thead><tr><th id="thFeature"></th><th id="thImportance"></th></tr></thead>
          <tbody id="featureTable"></tbody>
        </table>
      </div>
    </div>

    <h2 class="section-title" id="secFig"></h2>
    <div class="fig-grid">
      <div class="card"><h3 id="fig1t"></h3><img src="/figures/ligand_embedding_tsne.png" /><p class="muted" id="fig1d"></p></div>
      <div class="card"><h3 id="fig2t"></h3><img src="/figures/actual_vs_predicted.png" /><p class="muted" id="fig2d"></p></div>
      <div class="card"><h3 id="fig3t"></h3><img src="/figures/model_performance.png" /><p class="muted" id="fig3d"></p></div>
      <div class="card"><h3 id="fig4t"></h3><img src="/figures/descriptor_group_attention_heatmap.png" /><p class="muted" id="fig4d"></p></div>
    </div>

    <h2 class="section-title" id="secPred"></h2>
    <div class="card">
      <h3 id="predTitle"></h3>
      <label id="smilesLabel"></label>
      <textarea id="smilesInput" rows="2"></textarea>
      <p class="muted"><span id="orderPrefix"></span><span id="featureOrder"></span></p>
      <label id="descLabel"></label>
      <textarea id="descInput" rows="3"></textarea>
      <div style="margin-top:10px; display:flex; gap:10px;">
        <button id="sampleBtn"></button>
        <button id="predictBtn"></button>
      </div>
      <div id="predictResult" style="margin-top:14px;"></div>
      <div style="margin-top:10px;"><canvas id="attentionChart" height="120"></canvas></div>
    </div>

    <h2 class="section-title" id="secBo2"></h2>
    <div class="card" id="bo2Card">
      <h3 id="bo2Title"></h3>
      <div id="bo2NotReady" class="muted" style="display:none;"></div>
      <div id="bo2ReadyWrap">
        <div class="mini-grid" id="bo2MetricCards"></div>
        <div class="row2" style="margin-top:12px;">
          <div>
            <label id="bo2SmilesLabel"></label>
            <textarea id="bo2SmilesInput" rows="2"></textarea>
          </div>
          <div>
            <label id="bo2ReactionLabel"></label>
            <input id="bo2ReactionInput" />
          </div>
        </div>
        <div class="mini-grid" style="margin-top:10px;">
          <div><label id="bo2AdditiveLabel"></label><textarea id="bo2AdditiveInput" rows="2"></textarea></div>
          <div><label id="bo2BaseLabel"></label><textarea id="bo2BaseInput" rows="2"></textarea></div>
          <div><label id="bo2ArylLabel"></label><textarea id="bo2ArylInput" rows="2"></textarea></div>
        </div>
        <div style="margin-top:10px; display:flex; gap:10px;">
          <button id="bo2SampleBtn"></button>
          <button id="bo2PredictBtn"></button>
          <button id="bo2SuggestBtn"></button>
        </div>
        <div id="bo2PredictResult" style="margin-top:12px;"></div>
        <h3 id="bo2RecTitle" style="margin-top:16px;"></h3>
        <table>
          <thead><tr><th>Ligand</th><th>Additive</th><th>Base</th><th>Aryl halide</th><th>Pred</th><th>Std</th><th>EI</th></tr></thead>
          <tbody id="bo2RecTable"></tbody>
        </table>
      </div>
    </div>

    <h2 class="section-title" id="secScript"></h2>
    <div class="card">
      <h3 id="scriptTitle"></h3>
      <div id="talkTrack" class="talk"></div>
      <p class="muted" id="scriptHint"></p>
    </div>
  </div>
  <script>
    const summary = __PAYLOAD__;
    const state = { lang: "zh", metricChart: null, attentionChart: null };
    const text = {
      zh: {
        heroTitle: "Ligand 演示模式看板",
        heroSub1: "这页用于专业汇报：预测、解释、BO 推荐都在一个界面里完成。",
        heroSub2: "基础模型 + RDKit/PyG 升级模型双路线展示",
        langBtn: "Switch to English", scriptBtn: "自动生成讲解词", pdfBtn: "导出 PDF 汇报页",
        langChip: "当前语言：中文",
        secPerf: "一、基础模型性能总览", chartTitle: "模型性能对比（RMSE / MAE / R²）", topTitle: "Top Features（XGBoost）",
        thFeature: "Feature", thImportance: "Importance",
        secFig: "二、基础模型结果图", fig1t: "2D Embedding（TSNE）", fig1d: "看同类标签是否在二维空间聚在一起，聚得越明显，表示学习越有用。",
        fig2t: "Actual vs Predicted", fig2d: "点越贴近对角线越好；偏离明显的样本通常是后续排查重点。",
        fig3t: "Performance Comparison", fig3d: "当前 XGBoost 最稳，SMILES 分支提供结构补充，attention 负责解释视角。",
        fig4t: "Descriptor Group Attention", fig4d: "不同样本依赖不同特征组，这能指导后续局部 BO 和分簇实验。",
        secPred: "三、基础模型交互预测", predTitle: "输入 SMILES 与 17 维描述符", smilesLabel: "SMILES", orderPrefix: "Descriptor 顺序：",
        descLabel: "Descriptor values（逗号分隔）", sampleBtn: "填充示例", predictBtn: "运行预测",
        secBo2: "四、RDKit + PyG + BO（新）", bo2Title: "真实收率建模与离散候选推荐", bo2NotReady: "未检测到 RDKit+PyG 训练产物，请先运行训练命令。",
        bo2SmilesLabel: "Ligand SMILES", bo2ReactionLabel: "Reaction", bo2AdditiveLabel: "Additive", bo2BaseLabel: "Base", bo2ArylLabel: "Aryl halide",
        bo2SampleBtn: "填充 BO 示例", bo2PredictBtn: "运行 RDKit+PyG 预测", bo2SuggestBtn: "推荐下一批候选", bo2RecTitle: "离散候选推荐（Top）",
        secScript: "五、讲解词（口播）", scriptTitle: "自动讲解词", scriptHint: "语气偏项目负责人，尽量不“AI化”。",
        lead: "当前结果", modelCol: "模型", predCol: "预测值", predFail: "预测失败：", groupLabel: "特征组权重"
      },
      en: {
        heroTitle: "Ligand Demo Dashboard",
        heroSub1: "Built for technical demos: prediction, interpretation, and BO recommendation in one page.",
        heroSub2: "Base model + RDKit/PyG upgraded route",
        langBtn: "切换到中文", scriptBtn: "Generate talk track", pdfBtn: "Export PDF report page",
        langChip: "Language: English",
        secPerf: "1) Base-model performance", chartTitle: "Model comparison (RMSE / MAE / R²)", topTitle: "Top Features (XGBoost)",
        thFeature: "Feature", thImportance: "Importance",
        secFig: "2) Base-model figures", fig1t: "2D Embedding (TSNE)", fig1d: "Check whether similar labels cluster in 2D.",
        fig2t: "Actual vs Predicted", fig2d: "Closer to diagonal is better; large outliers are high-priority debug targets.",
        fig3t: "Performance Comparison", fig3d: "XGBoost is currently most stable, with sequence and attention branches as complements.",
        fig4t: "Descriptor Group Attention", fig4d: "Group-level importance shift helps local BO and mechanism-aware batching.",
        secPred: "3) Base-model interactive prediction", predTitle: "Input SMILES and 17 descriptors", smilesLabel: "SMILES", orderPrefix: "Descriptor order: ",
        descLabel: "Descriptor values (comma-separated)", sampleBtn: "Load sample", predictBtn: "Run prediction",
        secBo2: "4) RDKit + PyG + BO (new)", bo2Title: "True-yield modeling and discrete candidate recommendation", bo2NotReady: "RDKit+PyG artifacts not found. Please run training first.",
        bo2SmilesLabel: "Ligand SMILES", bo2ReactionLabel: "Reaction", bo2AdditiveLabel: "Additive", bo2BaseLabel: "Base", bo2ArylLabel: "Aryl halide",
        bo2SampleBtn: "Load BO sample", bo2PredictBtn: "Run RDKit+PyG prediction", bo2SuggestBtn: "Recommend next candidates", bo2RecTitle: "Top discrete recommendations",
        secScript: "5) Talk Track", scriptTitle: "Auto-generated speaking notes", scriptHint: "Project-owner tone, concise and practical.",
        lead: "Current snapshot", modelCol: "Model", predCol: "Prediction", predFail: "Prediction failed: ", groupLabel: "Group attention"
      }
    };
    function t(k){return text[state.lang][k]||k;}
    function applyLanguage(){
      Object.keys(text[state.lang]).forEach((k)=>{ const el=document.getElementById(k); if(el && ["lead","modelCol","predCol","predFail","groupLabel"].indexOf(k)===-1){ el.textContent=text[state.lang][k]; }});
      document.documentElement.lang = state.lang==="zh"?"zh-CN":"en";
    }
    function renderMetrics(){
      const wrap=document.getElementById("metricCards"); wrap.innerHTML="";
      const m=summary.metrics||{};
      Object.entries(m).forEach(([name,v])=>{ const card=document.createElement("div"); card.className="card"; card.innerHTML=`<h3>${name}</h3><div class="muted">${t("lead")}</div><div class="kpi">${Number(v.rmse).toFixed(4)}</div><div class="muted">MAE: ${Number(v.mae).toFixed(4)} | R²: ${Number(v.r2).toFixed(4)}</div>`; wrap.appendChild(card);});
    }
    function renderFeatureTable(){
      const tbody=document.getElementById("featureTable"); tbody.innerHTML="";
      (summary.top_features||[]).forEach((r)=>{ const tr=document.createElement("tr"); tr.innerHTML=`<td>${r.feature}</td><td>${Number(r.importance).toFixed(4)}</td>`; tbody.appendChild(tr);});
    }
    function renderMetricChart(){
      const m=summary.metrics||{}; const names=Object.keys(m);
      if(state.metricChart) state.metricChart.destroy();
      state.metricChart = new Chart(document.getElementById("metricChart"), { type:"bar", data:{ labels:names, datasets:[ {label:"RMSE",data:names.map(k=>m[k].rmse),backgroundColor:"rgba(59,130,246,0.7)"}, {label:"MAE",data:names.map(k=>m[k].mae),backgroundColor:"rgba(249,115,22,0.7)"}, {label:"R²",data:names.map(k=>m[k].r2),backgroundColor:"rgba(16,185,129,0.7)"} ] } });
    }
    function renderAttention(obj){
      const labels=Object.keys(obj||{}); const values=labels.map(k=>obj[k]);
      if(state.attentionChart) state.attentionChart.destroy();
      state.attentionChart = new Chart(document.getElementById("attentionChart"), { type:"bar", data:{ labels, datasets:[{label:t("groupLabel"),data:values,backgroundColor:"rgba(99,102,241,0.75)"}] }, options:{scales:{y:{beginAtZero:true,max:1}}}});
    }
    function formatPred(pred){
      return `<table><thead><tr><th>${t("modelCol")}</th><th>${t("predCol")}</th></tr></thead><tbody>
      <tr><td>xgboost</td><td>${pred.xgboost.toFixed(6)}</td></tr>
      <tr><td>descriptor_attention</td><td>${pred.descriptor_attention.toFixed(6)}</td></tr>
      <tr><td>smiles_hybrid</td><td>${pred.smiles_hybrid.toFixed(6)}</td></tr>
      <tr><td><b>ensemble_avg</b></td><td><b>${pred.ensemble_avg.toFixed(6)}</b></td></tr></tbody></table>`;
    }
    async function runPredict(){
      const smiles=document.getElementById("smilesInput").value.trim();
      const values=document.getElementById("descInput").value.trim().split(",").map(v=>Number(v.trim())).filter(v=>!Number.isNaN(v));
      const res=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({smiles,descriptor_values:values})});
      const data=await res.json(); const box=document.getElementById("predictResult");
      if(!res.ok){ box.innerHTML=`<div style="color:#b91c1c;font-weight:600;">${t("predFail")}${data.error}</div>`; return; }
      box.innerHTML=formatPred(data.prediction); renderAttention(data.group_attention);
    }
    function renderBo2Summary(){
      const bo2 = summary.bo2 || {ready:false};
      const notReady=document.getElementById("bo2NotReady");
      const wrap=document.getElementById("bo2ReadyWrap");
      if(!bo2.ready){ notReady.style.display="block"; notReady.textContent=t("bo2NotReady"); wrap.style.display="none"; return; }
      notReady.style.display="none"; wrap.style.display="block";
      const m=bo2.metrics||{};
      const scaffold=bo2.scaffold_kfold||{};
      const cards=document.getElementById("bo2MetricCards");
      cards.innerHTML=`
        <div class="card"><h3>Bootstrap</h3><div class="muted">RMSE</div><div class="kpi">${Number((m.bootstrap_mean||{}).rmse||0).toFixed(4)}</div><div class="muted">R²: ${Number((m.bootstrap_mean||{}).r2||0).toFixed(4)}</div></div>
        <div class="card"><h3>PyG GNN</h3><div class="muted">RMSE</div><div class="kpi">${Number((m.gnn_pyg||{}).rmse||0).toFixed(4)}</div><div class="muted">R²: ${Number((m.gnn_pyg||{}).r2||0).toFixed(4)}</div></div>
        <div class="card"><h3>Scaffold CV</h3><div class="muted">RMSE(mean)</div><div class="kpi">${Number(scaffold.rmse_mean||0).toFixed(4)}</div><div class="muted">R²(mean): ${Number(scaffold.r2_mean||0).toFixed(4)}</div></div>
      `;
    }
    function fillBo2Sample(){
      const s=(summary.bo2||{}).sample_input||{};
      document.getElementById("bo2SmilesInput").value=s.smiles||"";
      document.getElementById("bo2ReactionInput").value=s.reaction||"0";
      document.getElementById("bo2AdditiveInput").value=s.additive||"";
      document.getElementById("bo2BaseInput").value=s.base||"";
      document.getElementById("bo2ArylInput").value=s.aryl_halide||"";
    }
    async function runBo2Predict(){
      const body={
        smiles:document.getElementById("bo2SmilesInput").value.trim(),
        reaction:document.getElementById("bo2ReactionInput").value.trim(),
        additive:document.getElementById("bo2AdditiveInput").value.trim(),
        base:document.getElementById("bo2BaseInput").value.trim(),
        aryl_halide:document.getElementById("bo2ArylInput").value.trim()
      };
      const res=await fetch("/api/bo2/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
      const data=await res.json();
      const box=document.getElementById("bo2PredictResult");
      if(!res.ok){ box.innerHTML=`<div style="color:#b91c1c;font-weight:600;">${t("predFail")}${data.error}</div>`; return; }
      box.innerHTML=`<table><thead><tr><th>${t("modelCol")}</th><th>${t("predCol")}</th></tr></thead><tbody>
      <tr><td>bootstrap_mean</td><td>${data.bootstrap_mean.toFixed(6)}</td></tr>
      <tr><td>bootstrap_std</td><td>${data.bootstrap_std.toFixed(6)}</td></tr>
      <tr><td>gnn_pyg</td><td>${data.gnn_pyg.toFixed(6)}</td></tr>
      <tr><td><b>ensemble_avg</b></td><td><b>${data.ensemble_avg.toFixed(6)}</b></td></tr></tbody></table>`;
    }
    async function runBo2Suggest(){
      const reaction=document.getElementById("bo2ReactionInput").value.trim()||"0";
      const res=await fetch("/api/bo2/suggest",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reaction,top_k:8})});
      const data=await res.json();
      const tbody=document.getElementById("bo2RecTable"); tbody.innerHTML="";
      if(!res.ok){ tbody.innerHTML=`<tr><td colspan="7" style="color:#b91c1c;">${t("predFail")}${data.error}</td></tr>`; return; }
      (data.rows||[]).forEach((r)=>{
        const tr=document.createElement("tr");
        tr.innerHTML=`<td>${(r.ligand||"").slice(0,28)}...</td><td>${(r.additive||"").slice(0,16)}...</td><td>${(r.base||"").slice(0,16)}...</td><td>${(r["aryl halide"]||"").slice(0,16)}...</td><td>${Number(r.pred_blend||0).toFixed(3)}</td><td>${Number(r.pred_std||0).toFixed(3)}</td><td>${Number(r.acq_ei||0).toFixed(3)}</td>`;
        tbody.appendChild(tr);
      });
    }
    function buildTalkTrack(){
      const metrics=summary.metrics||{}; const top3=(summary.top_features||[]).slice(0,3).map(x=>`${x.feature}(${x.importance.toFixed(3)})`).join(", ");
      const bo2=summary.bo2||{}; const scaff=((bo2.scaffold_kfold||{}).rmse_mean||0).toFixed(3);
      if(state.lang==="zh"){
        return [
          "我先给结论：现在我们有两条线，基础模型负责快速解释，RDKit+PyG 线负责真实收率建模和离散 BO 推荐。",
          `基础模型里当前最稳的是 XGBoost 路线，关键特征主要集中在：${top3}。`,
          `新线里我们加入了标准 Murcko scaffold 验证，当前 scaffold RMSE 均值约 ${scaff}。`,
          "如果现场要做下一步实验，我会直接用下方 BO 推荐表挑 Top 候选去做。"
        ].join("\\n");
      }
      return [
        "Bottom line: we now run two tracks in one interface — base interpretation and RDKit/PyG true-yield modeling.",
        `Base track remains stable, with top explanatory features: ${top3}.`,
        `The upgraded track now uses Murcko scaffold validation, with mean scaffold RMSE around ${scaff}.`,
        "For immediate lab planning, I would take the top BO-ranked candidates from the recommendation table below."
      ].join("\\n");
    }
    function setTalkTrack(){ document.getElementById("talkTrack").textContent=buildTalkTrack(); }
    function init(){
      applyLanguage(); renderMetrics(); renderFeatureTable(); renderMetricChart(); renderBo2Summary(); setTalkTrack();
      document.getElementById("featureOrder").textContent=(summary.feature_cols||[]).join(", ");
      document.getElementById("sampleBtn").addEventListener("click",()=>{ const s=summary.sample_input||{}; document.getElementById("smilesInput").value=s.smiles||""; document.getElementById("descInput").value=s.descriptor_values||""; });
      document.getElementById("predictBtn").addEventListener("click",runPredict);
      document.getElementById("bo2SampleBtn").addEventListener("click",fillBo2Sample);
      document.getElementById("bo2PredictBtn").addEventListener("click",runBo2Predict);
      document.getElementById("bo2SuggestBtn").addEventListener("click",runBo2Suggest);
      document.getElementById("scriptBtn").addEventListener("click",setTalkTrack);
      document.getElementById("pdfBtn").addEventListener("click",()=>window.open(`/report?lang=${state.lang}`,"_blank"));
      document.getElementById("langBtn").addEventListener("click",()=>{ state.lang=state.lang==="zh"?"en":"zh"; applyLanguage(); renderMetrics(); renderMetricChart(); renderBo2Summary(); setTalkTrack(); });
      if((summary.bo2||{}).ready){ fillBo2Sample(); runBo2Suggest(); }
    }
    init();
  </script>
</body>
</html>"""
    return template.replace("__PAYLOAD__", payload)


def build_report_html(summary_payload: Dict[str, object], lang: str) -> str:
    lang = "en" if lang == "en" else "zh"
    metrics = summary_payload.get("metrics", {})
    top_features = summary_payload.get("top_features", [])[:8]
    bo2 = summary_payload.get("bo2", {})
    if lang == "zh":
        title = "Ligand 汇报页（PDF 导出版）"
        subtitle = "可用于项目周会、技术评审、对外沟通"
        metric_head = ("模型", "RMSE", "MAE", "R²")
        export_label = "点击打印/导出 PDF"
        sec1 = "1) 基础模型表现"
        sec2 = "2) 关键特征"
        sec3 = "3) 结果图"
        sec4 = "4) RDKit+PyG 升级结果"
    else:
        title = "Ligand Report Page (PDF Export)"
        subtitle = "For technical review and stakeholder meetings"
        metric_head = ("Model", "RMSE", "MAE", "R²")
        export_label = "Click to print/export PDF"
        sec1 = "1) Base-model performance"
        sec2 = "2) Key features"
        sec3 = "3) Figures"
        sec4 = "4) RDKit+PyG upgrade summary"
    metric_rows = "".join(
        f"<tr><td>{k}</td><td>{v.get('rmse', 0):.4f}</td><td>{v.get('mae', 0):.4f}</td><td>{v.get('r2', 0):.4f}</td></tr>"
        for k, v in metrics.items()
    )
    feature_rows = "".join(f"<tr><td>{x['feature']}</td><td>{x['importance']:.4f}</td></tr>" for x in top_features)
    bo2_boot = bo2.get("metrics", {}).get("bootstrap_mean", {})
    bo2_gnn = bo2.get("metrics", {}).get("gnn_pyg", {})
    bo2_scaf = bo2.get("scaffold_kfold", {})
    return f"""<!doctype html>
<html lang="{ 'zh-CN' if lang == 'zh' else 'en' }">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color:#111827; }}
    h1 {{ margin-bottom: 6px; }}
    .muted {{ color:#6b7280; margin-bottom: 14px; }}
    .btn {{ background:#1d4ed8; color:#fff; border:none; border-radius:8px; padding:10px 14px; cursor:pointer; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border:1px solid #d1d5db; padding:8px; text-align:left; font-size:14px; }}
    .fig-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .fig-grid img {{ width:100%; border:1px solid #d1d5db; }}
    @media print {{ .no-print {{ display:none; }} body {{ margin:10mm; }} }}
  </style>
</head>
<body>
  <div class="no-print"><button class="btn" onclick="window.print()">{export_label}</button></div>
  <h1>{title}</h1>
  <div class="muted">{subtitle}</div>
  <h2>{sec1}</h2>
  <table><thead><tr><th>{metric_head[0]}</th><th>{metric_head[1]}</th><th>{metric_head[2]}</th><th>{metric_head[3]}</th></tr></thead><tbody>{metric_rows}</tbody></table>
  <h2>{sec2}</h2>
  <table><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>{feature_rows}</tbody></table>
  <h2>{sec3}</h2>
  <div class="fig-grid">
    <img src="/figures/ligand_embedding_tsne.png" />
    <img src="/figures/actual_vs_predicted.png" />
    <img src="/figures/model_performance.png" />
    <img src="/figures/descriptor_group_attention_heatmap.png" />
  </div>
  <h2>{sec4}</h2>
  <table>
    <thead><tr><th>Item</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Bootstrap RMSE</td><td>{bo2_boot.get('rmse', 0):.4f}</td></tr>
      <tr><td>Bootstrap R²</td><td>{bo2_boot.get('r2', 0):.4f}</td></tr>
      <tr><td>GNN(PYG) RMSE</td><td>{bo2_gnn.get('rmse', 0):.4f}</td></tr>
      <tr><td>Scaffold RMSE(mean)</td><td>{bo2_scaf.get('rmse_mean', 0):.4f}</td></tr>
      <tr><td>Scaffold R²(mean)</td><td>{bo2_scaf.get('r2_mean', 0):.4f}</td></tr>
    </tbody>
  </table>
</body></html>"""


def build_handler(service: InferenceService):
    summary_payload = service.summary_payload()

    class DashboardHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, object], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str, status: int = 200) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, file_path: Path, content_type: str) -> None:
            if not file_path.exists() or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query)
            if path == "/":
                self._send_html(build_index_html(summary_payload))
                return
            if path == "/report":
                lang = query.get("lang", ["zh"])[0]
                self._send_html(build_report_html(summary_payload, lang))
                return
            if path == "/api/summary":
                self._send_json(summary_payload)
                return
            if path.startswith("/figures/"):
                name = path.split("/figures/", 1)[1]
                target = (service.fig_dir / name).resolve()
                if service.fig_dir.resolve() not in target.parents and target != service.fig_dir.resolve():
                    self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                    return
                ctype = "image/png" if name.lower().endswith(".png") else "application/octet-stream"
                self._send_file(target, ctype)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if path == "/api/predict":
                    smiles = str(payload.get("smiles", "")).strip()
                    values = payload.get("descriptor_values", [])
                    if not isinstance(values, list):
                        raise ValueError("descriptor_values must be a list")
                    result = service.predict(smiles=smiles, descriptor_values=[float(v) for v in values])
                    self._send_json(result, status=200)
                    return
                if path == "/api/bo2/predict":
                    result = service.bo2_predict(
                        smiles=str(payload.get("smiles", "")).strip(),
                        reaction=str(payload.get("reaction", "0")).strip(),
                        additive=str(payload.get("additive", "")).strip(),
                        base=str(payload.get("base", "")).strip(),
                        aryl_halide=str(payload.get("aryl_halide", "")).strip(),
                    )
                    self._send_json(result, status=200)
                    return
                if path == "/api/bo2/suggest":
                    reaction = str(payload.get("reaction", "0")).strip()
                    top_k = int(payload.get("top_k", 8))
                    rows = service.bo2_suggest(reaction=reaction, top_k=top_k)
                    self._send_json({"rows": rows}, status=200)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:  # pylint: disable=broad-except
                self._send_json({"error": str(exc)}, status=400)

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive expert dashboard for ligand modeling")
    parser.add_argument("--model-dir", type=str, default="ligand_outputs")
    parser.add_argument("--excel-path", type=str, default="Kraken monophosphine coordinates AD Descriptors.xlsx")
    parser.add_argument("--bo2-output-dir", type=str, default="yield_bo_pyg_outputs")
    parser.add_argument("--bo2-candidate-data", type=str, default="data/bh-reactions.csv")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = InferenceService(
        model_dir=args.model_dir,
        excel_path=args.excel_path,
        bo2_output_dir=args.bo2_output_dir,
        bo2_candidate_data=args.bo2_candidate_data,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(service))
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
