"""
Professional interactive dashboard for ligand modeling results.

No extra dependency is required beyond the current project environment.
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import unquote, urlparse

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


class InferenceService:
    def __init__(self, model_dir: str, excel_path: str) -> None:
        self.model_dir = Path(model_dir)
        self.excel_path = Path(excel_path)
        self.fig_dir = self.model_dir / "figures"

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        artifact_path = self.model_dir / "artifacts.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing artifacts file: {artifact_path}")

        self.artifacts = joblib.load(artifact_path)
        self.feature_cols: List[str] = self.artifacts["feature_cols"]
        self.scaler = self.artifacts["scaler"]
        self.vocab = self.artifacts["vocab"]
        self.max_smiles_len = int(self.artifacts["max_smiles_len"])
        self.group_indices = self.artifacts["descriptor_group_indices"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(str(self.model_dir / "xgb_model.json"))

        self.desc_model = DescriptorAttentionRegressor(
            feature_dim=len(self.feature_cols),
            group_indices=self.group_indices,
        ).to(self.device)
        self.desc_model.load_state_dict(
            torch.load(self.model_dir / "descriptor_attention_model.pt", map_location=self.device)
        )
        self.desc_model.eval()

        self.smiles_model = SmilesDescriptorRegressor(
            vocab_size=len(self.vocab),
            descriptor_dim=len(self.feature_cols),
        ).to(self.device)
        self.smiles_model.load_state_dict(
            torch.load(self.model_dir / "smiles_hybrid_model.pt", map_location=self.device)
        )
        self.smiles_model.eval()

        summary_path = self.model_dir / "results_summary.json"
        self.summary = {}
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                self.summary = json.load(f)

        self.top_features = self._build_top_features()
        self.sample_input = self._load_sample_input()

    def _build_top_features(self) -> List[Dict[str, float]]:
        importance = self.xgb_model.feature_importances_
        order = np.argsort(importance)[::-1]
        return [
            {"feature": self.feature_cols[i], "importance": float(importance[i])}
            for i in order[:8]
        ]

    def _load_sample_input(self) -> Dict[str, str]:
        if not self.excel_path.exists():
            return {"smiles": "", "descriptor_values": ""}
        df = pd.read_excel(self.excel_path)
        if df.empty:
            return {"smiles": "", "descriptor_values": ""}
        smiles = str(df.iloc[0, 9])  # J
        values = df.iloc[0, 14:31].astype(float).tolist()
        return {"smiles": smiles, "descriptor_values": ",".join(str(v) for v in values)}

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
        desc_pred_val = float(desc_pred[0])
        smiles_pred_val = float(smiles_pred[0])
        ensemble = float((xgb_pred + desc_pred_val + smiles_pred_val) / 3.0)

        return {
            "prediction": {
                "xgboost": xgb_pred,
                "descriptor_attention": desc_pred_val,
                "smiles_hybrid": smiles_pred_val,
                "ensemble_avg": ensemble,
            },
            "group_attention": {
                "O-Q": float(group_weights[0][0]),
                "R-T": float(group_weights[0][1]),
                "U-W": float(group_weights[0][2]),
                "X-AE": float(group_weights[0][3]),
            },
        }

    def summary_payload(self) -> Dict[str, object]:
        return {
            "feature_cols": self.feature_cols,
            "metrics": self.summary.get("metrics", {}),
            "label_col": self.summary.get("label_col", "label"),
            "top_features": self.top_features,
            "sample_input": self.sample_input,
            "figures": [
                "ligand_embedding_tsne.png",
                "actual_vs_predicted.png",
                "model_performance.png",
                "xgboost_feature_importance.png",
                "descriptor_group_attention_heatmap.png",
            ],
        }


def build_index_html(summary_payload: Dict[str, object]) -> str:
    payload_json = json.dumps(summary_payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ligand AI Expert Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background:#f6f8fb; color:#1f2937; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: linear-gradient(135deg, #0f172a, #1e3a8a); color: white; border-radius: 14px; padding: 22px; }}
    .hero h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
    .hero p {{ margin: 6px 0; opacity: 0.95; }}
    .grid {{ display: grid; gap: 16px; }}
    .metrics {{ grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); margin-top: 16px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 14px rgba(15, 23, 42, 0.06); }}
    .card h3 {{ margin: 0 0 10px 0; font-size: 16px; color:#111827; }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .section-title {{ margin: 26px 0 10px 0; font-size: 20px; }}
    .fig-grid {{ display:grid; gap:16px; grid-template-columns:1fr 1fr; }}
    .fig-grid img {{ width: 100%; border:1px solid #e5e7eb; border-radius: 10px; }}
    .kpi {{ font-size: 24px; font-weight: 700; margin-top: 8px; }}
    textarea, input {{ width: 100%; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 14px; box-sizing: border-box; }}
    button {{ background:#1d4ed8; color:white; border:none; padding:10px 16px; border-radius:8px; cursor:pointer; font-weight:600; }}
    button:hover {{ background:#1e40af; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 14px; }}
    .row2 {{ display:grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }}
    @media (max-width: 900px) {{
      .fig-grid {{ grid-template-columns: 1fr; }}
      .row2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>Ligand Feature Intelligence Dashboard</h1>
      <p>面向专业汇报：深层特征分析、模型性能评估、结果可视化与新配体预测。</p>
      <p>方法栈：XGBoost + Descriptor Attention + SMILES Hybrid + Ensemble</p>
    </div>

    <h2 class="section-title">一、性能总览</h2>
    <div class="grid metrics" id="metricCards"></div>

    <div class="row2">
      <div class="card">
        <h3>模型性能对比（RMSE / MAE / R²）</h3>
        <canvas id="metricChart" height="160"></canvas>
      </div>
      <div class="card">
        <h3>Top Features（XGBoost）</h3>
        <table>
          <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
          <tbody id="featureTable"></tbody>
        </table>
      </div>
    </div>

    <h2 class="section-title">二、结果图与专业解读</h2>
    <div class="fig-grid">
      <div class="card">
        <h3>2D Embedding（TSNE）</h3>
        <img src="/figures/ligand_embedding_tsne.png" alt="tsne" />
        <p class="muted">用于观察表示空间中不同 label 的聚集性。若颜色形成局部簇，说明模型学习到与 label 相关的结构信息。</p>
      </div>
      <div class="card">
        <h3>Actual vs Predicted</h3>
        <img src="/figures/actual_vs_predicted.png" alt="scatter" />
        <p class="muted">散点越贴近对角线，拟合越准确；偏离明显的点可作为异常样本或机制异质性候选。</p>
      </div>
      <div class="card">
        <h3>Performance Comparison</h3>
        <img src="/figures/model_performance.png" alt="performance" />
        <p class="muted">当前 XGBoost 在该数据规模下表现最强，SMILES 混合模型提供结构维度补充，attention 模型强化解释能力。</p>
      </div>
      <div class="card">
        <h3>Descriptor Group Attention</h3>
        <img src="/figures/descriptor_group_attention_heatmap.png" alt="attention" />
        <p class="muted">展示不同样本对 descriptor group 的依赖差异，可用于后续局部 BO 与机制分簇实验设计。</p>
      </div>
    </div>

    <h2 class="section-title">三、交互预测（新 Ligand）</h2>
    <div class="card">
      <h3>输入 SMILES 与 17 维描述符</h3>
      <label>SMILES</label>
      <textarea id="smilesInput" rows="2"></textarea>
      <p class="muted">Descriptor 顺序：<span id="featureOrder"></span></p>
      <label>Descriptor values（逗号分隔）</label>
      <textarea id="descInput" rows="3"></textarea>
      <div style="margin-top:10px; display:flex; gap:10px;">
        <button id="fillSampleBtn">填充示例</button>
        <button id="predictBtn">运行预测</button>
      </div>
      <div id="predictResult" style="margin-top:14px;"></div>
      <div style="margin-top:10px;">
        <canvas id="attentionChart" height="120"></canvas>
      </div>
    </div>
  </div>

  <script>
    const summary = {payload_json};

    function renderMetrics() {{
      const metrics = summary.metrics || {{}};
      const cardContainer = document.getElementById('metricCards');
      cardContainer.innerHTML = '';
      Object.entries(metrics).forEach(([name, v]) => {{
        const el = document.createElement('div');
        el.className = 'card';
        el.innerHTML = `
          <h3>${{name}}</h3>
          <div class="muted">RMSE</div><div class="kpi">${{Number(v.rmse).toFixed(4)}}</div>
          <div class="muted">MAE: ${{Number(v.mae).toFixed(4)}} | R²: ${{Number(v.r2).toFixed(4)}}</div>
        `;
        cardContainer.appendChild(el);
      }});
    }}

    function renderFeatureTable() {{
      const tbody = document.getElementById('featureTable');
      tbody.innerHTML = '';
      (summary.top_features || []).forEach((row) => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{row.feature}}</td><td>${{Number(row.importance).toFixed(4)}}</td>`;
        tbody.appendChild(tr);
      }});
    }}

    function renderMetricChart() {{
      const m = summary.metrics || {{}};
      const names = Object.keys(m);
      const rmse = names.map(k => m[k].rmse);
      const mae = names.map(k => m[k].mae);
      const r2 = names.map(k => m[k].r2);
      new Chart(document.getElementById('metricChart'), {{
        type: 'bar',
        data: {{
          labels: names,
          datasets: [
            {{ label: 'RMSE', data: rmse, backgroundColor: 'rgba(59,130,246,0.7)' }},
            {{ label: 'MAE', data: mae, backgroundColor: 'rgba(249,115,22,0.7)' }},
            {{ label: 'R²', data: r2, backgroundColor: 'rgba(16,185,129,0.7)' }}
          ]
        }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: 'top' }} }} }}
      }});
    }}

    let attentionChart = null;
    function renderAttentionChart(attentionObj) {{
      const labels = Object.keys(attentionObj || {{}});
      const values = labels.map(k => attentionObj[k]);
      if (attentionChart) attentionChart.destroy();
      attentionChart = new Chart(document.getElementById('attentionChart'), {{
        type: 'bar',
        data: {{
          labels,
          datasets: [{{ label: 'Group Attention', data: values, backgroundColor: 'rgba(99,102,241,0.75)' }}]
        }},
        options: {{
          scales: {{ y: {{ beginAtZero: true, max: 1 }} }}
        }}
      }});
    }}

    function formatPrediction(pred) {{
      return `
        <table>
          <thead><tr><th>Model</th><th>Prediction</th></tr></thead>
          <tbody>
            <tr><td>xgboost</td><td>${{pred.xgboost.toFixed(6)}}</td></tr>
            <tr><td>descriptor_attention</td><td>${{pred.descriptor_attention.toFixed(6)}}</td></tr>
            <tr><td>smiles_hybrid</td><td>${{pred.smiles_hybrid.toFixed(6)}}</td></tr>
            <tr><td><b>ensemble_avg</b></td><td><b>${{pred.ensemble_avg.toFixed(6)}}</b></td></tr>
          </tbody>
        </table>
      `;
    }}

    async function runPredict() {{
      const smiles = document.getElementById('smilesInput').value.trim();
      const descText = document.getElementById('descInput').value.trim();
      const values = descText.split(',').map(v => Number(v.trim())).filter(v => !Number.isNaN(v));

      const res = await fetch('/api/predict', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ smiles, descriptor_values: values }})
      }});
      const data = await res.json();
      const box = document.getElementById('predictResult');
      if (!res.ok) {{
        box.innerHTML = `<div style="color:#b91c1c; font-weight:600;">预测失败：${{data.error}}</div>`;
        return;
      }}
      box.innerHTML = formatPrediction(data.prediction);
      renderAttentionChart(data.group_attention);
    }}

    function initInputs() {{
      document.getElementById('featureOrder').innerText = (summary.feature_cols || []).join(', ');
      document.getElementById('fillSampleBtn').addEventListener('click', () => {{
        const sample = summary.sample_input || {{}};
        document.getElementById('smilesInput').value = sample.smiles || '';
        document.getElementById('descInput').value = sample.descriptor_values || '';
      }});
      document.getElementById('predictBtn').addEventListener('click', runPredict);
    }}

    renderMetrics();
    renderFeatureTable();
    renderMetricChart();
    initInputs();
  </script>
</body>
</html>"""


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
            if path == "/":
                self._send_html(build_index_html(summary_payload))
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
            parsed = urlparse(self.path)
            if parsed.path != "/api/predict":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8"))
                smiles = str(payload.get("smiles", "")).strip()
                values = payload.get("descriptor_values", [])
                if not isinstance(values, list):
                    raise ValueError("descriptor_values must be a list")
                descriptor_values = [float(v) for v in values]
                result = service.predict(smiles=smiles, descriptor_values=descriptor_values)
                self._send_json(result, status=200)
            except Exception as exc:  # pylint: disable=broad-except
                self._send_json({"error": str(exc)}, status=400)

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive expert dashboard for ligand modeling")
    parser.add_argument("--model-dir", type=str, default="ligand_outputs")
    parser.add_argument("--excel-path", type=str, default="Kraken monophosphine coordinates AD Descriptors.xlsx")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = InferenceService(model_dir=args.model_dir, excel_path=args.excel_path)
    handler = build_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
