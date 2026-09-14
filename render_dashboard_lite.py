"""Memory-light dashboard for Render free tier (BO2-focused)."""

from __future__ import annotations

import argparse
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-8)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * normal_cdf(z) + sigma * normal_pdf(z)


class LiteService:
    def __init__(
        self,
        base_summary_path: str,
        bo2_output_dir: str,
        candidate_data_path: str,
    ) -> None:
        self.base_summary_path = Path(base_summary_path)
        self.bo2_output_dir = Path(bo2_output_dir)
        self.candidate_data_path = Path(candidate_data_path)

        self.base_summary: Dict[str, object] = {}
        if self.base_summary_path.exists():
            self.base_summary = json.loads(self.base_summary_path.read_text(encoding="utf-8"))

        self.ready = False
        self.meta: Dict[str, object] = {}
        self.bo2_summary: Dict[str, object] = {}
        self.boot_models: List[xgb.XGBRegressor] = []
        self.candidate_df: pd.DataFrame | None = None
        self.morgan_gen = None
        self.cached_suggest_df: pd.DataFrame | None = None

        try:
            meta_path = self.bo2_output_dir / "meta.json"
            summary_path = self.bo2_output_dir / "summary.json"
            boot_dir = self.bo2_output_dir / "bootstrap_models"
            if not (meta_path.exists() and summary_path.exists() and boot_dir.exists()):
                return
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.bo2_summary = json.loads(summary_path.read_text(encoding="utf-8"))

            for name in sorted([x for x in boot_dir.iterdir() if x.suffix == ".json"]):
                m = xgb.XGBRegressor()
                m.load_model(str(name))
                self.boot_models.append(m)
            if not self.boot_models:
                return

            if not self.candidate_data_path.exists():
                return
            self.candidate_df = pd.read_csv(self.candidate_data_path)
            fp_bits = int(self.meta.get("fp_bits", 1024))
            self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_bits)
            self.ready = True
        except Exception:
            self.ready = False

    def _morgan_fp(self, smiles: str) -> np.ndarray:
        fp_bits = int(self.meta.get("fp_bits", 1024))
        arr = np.zeros((fp_bits,), dtype=np.float32)
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None or self.morgan_gen is None:
            return arr
        bv = self.morgan_gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr

    def _encode_conditions(self, df: pd.DataFrame) -> np.ndarray:
        cond_cols = list(self.meta.get("cond_cols", []))
        cond_maps = self.meta.get("cond_maps", {})
        parts = []
        for c in cond_cols:
            vocab = cond_maps.get(c, {})
            one = np.zeros((len(df), len(vocab)), dtype=np.float32)
            vals = df[c].astype(str).tolist() if c in df.columns else [""] * len(df)
            for i, v in enumerate(vals):
                idx = vocab.get(v)
                if idx is not None:
                    one[i, int(idx)] = 1.0
            parts.append(one)
        return np.concatenate(parts, axis=1) if parts else np.zeros((len(df), 0), dtype=np.float32)

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        smiles_col = str(self.meta.get("smiles_col", "ligand"))
        cond_x = self._encode_conditions(df)
        fp_x = np.stack([self._morgan_fp(s) for s in df[smiles_col].astype(str).tolist()], axis=0)
        return np.concatenate([cond_x, fp_x], axis=1)

    def _predict_bootstrap(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = np.stack([m.predict(x) for m in self.boot_models], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def summary_payload(self) -> Dict[str, object]:
        bo2 = {"ready": self.ready}
        if self.ready:
            sample = {}
            if self.candidate_df is not None and not self.candidate_df.empty:
                r = self.candidate_df.iloc[0].to_dict()
                sample = {
                    "smiles": str(r.get(self.meta.get("smiles_col", "ligand"), "")),
                    "reaction": str(r.get("reaction", "0")),
                    "additive": str(r.get("additive", "")),
                    "base": str(r.get("base", "")),
                    "aryl_halide": str(r.get("aryl halide", "")),
                }
            bo2.update(
                {
                    "metrics": self.bo2_summary.get("test_metrics", {}),
                    "scaffold_kfold": self.bo2_summary.get("scaffold_group_kfold", {}).get("aggregate", {}),
                    "sample_input": sample,
                }
            )
        return {
            "base_metrics": self.base_summary.get("metrics", {}),
            "bo2": bo2,
        }

    def bo2_predict(self, payload: Dict[str, object]) -> Dict[str, float]:
        if not self.ready:
            raise ValueError("BO2 not ready")
        smiles_col = str(self.meta.get("smiles_col", "ligand"))
        row = {
            smiles_col: str(payload.get("smiles", "")).strip(),
            "reaction": str(payload.get("reaction", "0")).strip(),
            "additive": str(payload.get("additive", "")).strip(),
            "base": str(payload.get("base", "")).strip(),
            "aryl halide": str(payload.get("aryl_halide", "")).strip(),
        }
        df = pd.DataFrame([row])
        x = self._build_features(df)
        mu, std = self._predict_bootstrap(x)
        return {
            "bootstrap_mean": float(mu[0]),
            "bootstrap_std": float(std[0]),
            "ensemble_avg": float(mu[0]),
            "note": "Lite mode uses bootstrap model for online inference.",
        }

    def bo2_suggest(self, reaction: str, top_k: int) -> List[Dict[str, object]]:
        if not self.ready or self.candidate_df is None:
            raise ValueError("BO2 not ready")
        if self.cached_suggest_df is None:
            df = self.candidate_df.copy()
            x = self._build_features(df)
            mu, std = self._predict_bootstrap(x)
            ei = expected_improvement(mu, std, best=float(np.max(mu)))
            df["pred_blend"] = mu
            df["pred_std"] = std
            df["acq_ei"] = ei
            self.cached_suggest_df = df
        out = self.cached_suggest_df.copy()
        if "reaction" in out.columns:
            out = out[out["reaction"].astype(str) == str(reaction)].reset_index(drop=True)
        out = out.sort_values(["acq_ei", "pred_blend"], ascending=False).head(top_k)
        keep = [c for c in ["ligand", "additive", "base", "aryl halide", "pred_blend", "pred_std", "acq_ei"] if c in out.columns]
        return out[keep].to_dict(orient="records")


def build_index_html(payload: Dict[str, object]) -> str:
    pj = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Classification Dashboard (Lite)</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin:0; background:#f6f8fb; color:#111827; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 22px; }}
    .hero {{ background: linear-gradient(135deg,#0f172a,#1e3a8a); color:#fff; border-radius:12px; padding:18px; }}
    .card {{ background:#fff; border-radius:10px; padding:14px; box-shadow:0 1px 10px rgba(0,0,0,0.06); margin-top:14px; }}
    .grid {{ display:grid; gap:10px; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); }}
    table {{ width:100%; border-collapse: collapse; }}
    th,td {{ border-bottom:1px solid #e5e7eb; padding:7px; font-size:13px; text-align:left; }}
    textarea,input {{ width:100%; box-sizing:border-box; padding:8px; border:1px solid #d1d5db; border-radius:8px; }}
    button {{ background:#1d4ed8; color:#fff; border:none; border-radius:8px; padding:9px 12px; cursor:pointer; }}
    .muted {{ color:#6b7280; font-size:13px; }}
  </style>
</head>
<body>
<div class="container">
  <div class="hero">
    <h2>Classification Dashboard（Render Lite）</h2>
    <p>基础指标 + RDKit/BO 推荐（轻量部署模式，适配 512MB 内存）。</p>
  </div>

  <div class="card">
    <h3>基础模型指标</h3>
    <div id="baseMetrics" class="grid"></div>
  </div>

  <div class="card">
    <h3>RDKit + BO 模块</h3>
    <div id="bo2Status" class="muted"></div>
    <div id="bo2Wrap" style="display:none;">
      <div class="grid">
        <div><label>Ligand SMILES</label><textarea id="smiles" rows="2"></textarea></div>
        <div><label>Reaction</label><input id="reaction" /></div>
        <div><label>Additive</label><textarea id="additive" rows="2"></textarea></div>
        <div><label>Base</label><textarea id="base" rows="2"></textarea></div>
        <div><label>Aryl halide</label><textarea id="aryl" rows="2"></textarea></div>
      </div>
      <div style="margin-top:10px;display:flex;gap:8px;">
        <button id="fillSample">填充示例</button>
        <button id="runPredict">运行 BO2 预测</button>
        <button id="runSuggest">推荐下一批候选</button>
      </div>
      <div id="predOut" style="margin-top:10px;"></div>
      <h4>推荐结果</h4>
      <table><thead><tr><th>Ligand</th><th>Additive</th><th>Base</th><th>Aryl halide</th><th>Pred</th><th>Std</th><th>EI</th></tr></thead><tbody id="recRows"></tbody></table>
    </div>
  </div>
</div>
<script>
const summary = {pj};
function renderBase() {{
  const m = summary.base_metrics || {{}};
  const box = document.getElementById('baseMetrics');
  box.innerHTML='';
  Object.entries(m).forEach(([k,v])=>{{
    const d=document.createElement('div');
    d.className='card';
    d.innerHTML=`<b>${{k}}</b><div class="muted">RMSE: ${{Number(v.rmse||0).toFixed(4)}} | MAE: ${{Number(v.mae||0).toFixed(4)}} | R²: ${{Number(v.r2||0).toFixed(4)}}</div>`;
    box.appendChild(d);
  }});
}}
function fillBo2Sample() {{
  const s=((summary.bo2||{{}}).sample_input||{{}});
  document.getElementById('smiles').value=s.smiles||'';
  document.getElementById('reaction').value=s.reaction||'0';
  document.getElementById('additive').value=s.additive||'';
  document.getElementById('base').value=s.base||'';
  document.getElementById('aryl').value=s.aryl_halide||'';
}}
async function runBo2Predict() {{
  const body={{
    smiles:document.getElementById('smiles').value.trim(),
    reaction:document.getElementById('reaction').value.trim(),
    additive:document.getElementById('additive').value.trim(),
    base:document.getElementById('base').value.trim(),
    aryl_halide:document.getElementById('aryl').value.trim(),
  }};
  const r=await fetch('/api/bo2/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});
  const j=await r.json();
  const out=document.getElementById('predOut');
  if(!r.ok) {{ out.innerHTML=`<div style="color:#b91c1c;">${{j.error}}</div>`; return; }}
  out.innerHTML=`<table><tbody>
    <tr><td>bootstrap_mean</td><td>${{Number(j.bootstrap_mean).toFixed(6)}}</td></tr>
    <tr><td>bootstrap_std</td><td>${{Number(j.bootstrap_std).toFixed(6)}}</td></tr>
    <tr><td>ensemble_avg</td><td>${{Number(j.ensemble_avg).toFixed(6)}}</td></tr>
  </tbody></table><div class="muted">${{j.note||''}}</div>`;
}}
async function runBo2Suggest() {{
  const reaction=document.getElementById('reaction').value.trim()||'0';
  const r=await fetch('/api/bo2/suggest',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{reaction,top_k:8}})}});
  const j=await r.json();
  const tbody=document.getElementById('recRows'); tbody.innerHTML='';
  if(!r.ok) {{ tbody.innerHTML=`<tr><td colspan="7" style="color:#b91c1c;">${{j.error}}</td></tr>`; return; }}
  (j.rows||[]).forEach(row=>{{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${{String(row.ligand||'').slice(0,24)}}...</td><td>${{String(row.additive||'').slice(0,16)}}...</td><td>${{String(row.base||'').slice(0,16)}}...</td><td>${{String(row['aryl halide']||'').slice(0,16)}}...</td><td>${{Number(row.pred_blend||0).toFixed(3)}}</td><td>${{Number(row.pred_std||0).toFixed(3)}}</td><td>${{Number(row.acq_ei||0).toFixed(3)}}</td>`;
    tbody.appendChild(tr);
  }});
}}
function init() {{
  renderBase();
  const bo2=summary.bo2||{{}};
  const status=document.getElementById('bo2Status');
  const wrap=document.getElementById('bo2Wrap');
  if(!bo2.ready) {{
    status.textContent='BO2 模块未就绪：请检查 yield_bo_pyg_outputs 和 data/bh-reactions.csv。';
    wrap.style.display='none';
    return;
  }}
  status.textContent='BO2 模块已就绪。';
  wrap.style.display='block';
  document.getElementById('fillSample').addEventListener('click',fillBo2Sample);
  document.getElementById('runPredict').addEventListener('click',runBo2Predict);
  document.getElementById('runSuggest').addEventListener('click',runBo2Suggest);
  fillBo2Sample();
  runBo2Suggest();
}}
init();
</script>
</body></html>"""


def build_handler(service: LiteService):
    summary = service.summary_payload()

    class Handler(BaseHTTPRequestHandler):
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

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            _ = parse_qs(parsed.query)
            if path == "/":
                self._send_html(build_index_html(summary))
                return
            if path == "/api/summary":
                self._send_json(summary)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            try:
                path = urlparse(self.path).path
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if path == "/api/bo2/predict":
                    self._send_json(service.bo2_predict(payload))
                    return
                if path == "/api/bo2/suggest":
                    reaction = str(payload.get("reaction", "0"))
                    top_k = int(payload.get("top_k", 8))
                    self._send_json({"rows": service.bo2_suggest(reaction=reaction, top_k=top_k)})
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:  # pylint: disable=broad-except
                self._send_json({"error": str(exc)}, status=400)

    return Handler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render lite dashboard")
    p.add_argument("--base-summary-path", type=str, default="ligand_outputs/results_summary.json")
    p.add_argument("--bo2-output-dir", type=str, default="yield_bo_pyg_outputs")
    p.add_argument("--bo2-candidate-data", type=str, default="data/bh-reactions.csv")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    service = LiteService(
        base_summary_path=args.base_summary_path,
        bo2_output_dir=args.bo2_output_dir,
        candidate_data_path=args.bo2_candidate_data,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(service))
    print(f"Lite dashboard at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
