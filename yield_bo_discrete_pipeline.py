"""
Reaction-yield modeling and discrete BO pipeline.

Key capabilities:
1) Use real yield dataset (default: data/bh-reactions.csv)
2) Uncertainty modeling (bootstrap ensemble + quantile regression)
3) K-fold and scaffold-style grouped validation
4) SMILES graph model (lightweight GNN without extra dependencies)
5) Discrete BO loop and next-candidate recommendation
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def smiles_hash_features(smiles: str, fp_dim: int = 256, ngram_range: Tuple[int, int] = (2, 4)) -> np.ndarray:
    vec = np.zeros(fp_dim, dtype=np.float32)
    s = smiles or ""
    if len(s) == 0:
        return vec
    for n in range(ngram_range[0], ngram_range[1] + 1):
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            token = s[i : i + n]
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % fp_dim
            vec[h] += 1.0
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def build_tabular_features(
    df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    fp_dim: int,
    fit_columns: Sequence[str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    cond_df = pd.get_dummies(df[list(cond_cols)].astype(str), prefix=list(cond_cols), dtype=np.float32)
    if fit_columns is None:
        cond_columns = cond_df.columns.tolist()
    else:
        cond_columns = list(fit_columns)
        cond_df = cond_df.reindex(columns=cond_columns, fill_value=0.0)
    cond_x = cond_df.to_numpy(dtype=np.float32)
    smiles_x = np.stack([smiles_hash_features(s, fp_dim=fp_dim) for s in df[smiles_col].astype(str)], axis=0)
    x = np.concatenate([cond_x, smiles_x], axis=1)
    return x, cond_columns


def train_bootstrap_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_models: int = 25,
    seed: int = 42,
) -> List[xgb.XGBRegressor]:
    models: List[xgb.XGBRegressor] = []
    n = len(x_train)
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        idx = rng.integers(0, n, n)
        m = xgb.XGBRegressor(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed + i,
        )
        m.fit(x_train[idx], y_train[idx], verbose=False)
        models.append(m)
    return models


def bootstrap_predict(models: Sequence[xgb.XGBRegressor], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.stack([m.predict(x) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def train_quantile_models(
    x_train: np.ndarray, y_train: np.ndarray
) -> Tuple[GradientBoostingRegressor, GradientBoostingRegressor, GradientBoostingRegressor]:
    mean_model = GradientBoostingRegressor(loss="squared_error", n_estimators=300, learning_rate=0.03, max_depth=3)
    low_model = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=300, learning_rate=0.03, max_depth=3)
    high_model = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=300, learning_rate=0.03, max_depth=3)
    mean_model.fit(x_train, y_train)
    low_model.fit(x_train, y_train)
    high_model.fit(x_train, y_train)
    return mean_model, low_model, high_model


def cycle_nodes_from_edges(n_nodes: int, edges: Sequence[Tuple[int, int]]) -> set[int]:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges)
    cycles = nx.cycle_basis(g)
    out: set[int] = set()
    for cyc in cycles:
        out.update(cyc)
    return out


def tokenize_smiles(smiles: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    two_char = {"Br", "Cl", "Si", "Na", "Li", "Ca", "Fe", "Cu", "Zn", "Mg", "Al", "Sn", "Hg", "Ag", "Au", "Pt", "Pd", "Ir", "Rh", "Ru", "Os", "Co", "Ni", "Ti", "Cr", "Mo"}
    bond_tokens = {"-", "=", "#", ":"}
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            j = i + 1
            while j < len(smiles) and smiles[j] != "]":
                j += 1
            tokens.append(smiles[i : j + 1] if j < len(smiles) else smiles[i:])
            i = j + 1
            continue
        if ch == "%":
            tokens.append(smiles[i : i + 3])
            i += 3
            continue
        if i + 1 < len(smiles) and smiles[i : i + 2] in two_char:
            tokens.append(smiles[i : i + 2])
            i += 2
            continue
        if ch in bond_tokens or ch in {"(", ")", ".", "/", "\\", "@", "+", "-", "*"}:
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit():
            tokens.append(ch)
            i += 1
            continue
        tokens.append(ch)
        i += 1
    return tokens


def atom_symbol(token: str) -> str:
    if token.startswith("[") and token.endswith("]"):
        inside = token[1:-1]
        if not inside:
            return "C"
        if len(inside) >= 2 and inside[1].islower():
            return inside[:2]
        return inside[0]
    if len(token) >= 1 and token[0].isalpha():
        if len(token) == 2 and token[1].islower():
            return token
        return token[0]
    return "C"


def is_atom_token(token: str) -> bool:
    if token.startswith("[") and token.endswith("]"):
        return True
    return token and token[0].isalpha() and token not in {"c", "n", "o", "s", "p", "b", "B", "C", "N", "O", "P", "S", "F", "I", "Cl", "Br"} or token in {
        "c",
        "n",
        "o",
        "s",
        "p",
        "b",
        "B",
        "C",
        "N",
        "O",
        "P",
        "S",
        "F",
        "I",
        "Cl",
        "Br",
    }


def parse_smiles_graph(smiles: str) -> Tuple[List[str], List[Tuple[int, int, float]], int]:
    tokens = tokenize_smiles(smiles)
    atoms: List[str] = []
    edges: List[Tuple[int, int, float]] = []
    branch_stack: List[int] = []
    ring_map: Dict[str, Tuple[int, float]] = {}
    cur_idx: int | None = None
    bond_order = 1.0
    bond_map = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5}
    ring_count = 0

    for tok in tokens:
        if tok in bond_map:
            bond_order = bond_map[tok]
            continue
        if tok == "(":
            if cur_idx is not None:
                branch_stack.append(cur_idx)
            continue
        if tok == ")":
            if branch_stack:
                cur_idx = branch_stack.pop()
            continue
        if tok == ".":
            cur_idx = None
            continue
        if tok in {"/", "\\", "@", "+", "-", "*"}:
            continue
        if tok.isdigit() or tok.startswith("%"):
            if cur_idx is None:
                continue
            key = tok
            if key not in ring_map:
                ring_map[key] = (cur_idx, bond_order)
            else:
                prev_idx, prev_bo = ring_map[key]
                bo = bond_order if bond_order != 1.0 else prev_bo
                edges.append((prev_idx, cur_idx, bo))
                ring_count += 1
                del ring_map[key]
            bond_order = 1.0
            continue
        if is_atom_token(tok):
            sym = atom_symbol(tok)
            atoms.append(sym)
            new_idx = len(atoms) - 1
            if cur_idx is not None:
                edges.append((cur_idx, new_idx, bond_order))
            cur_idx = new_idx
            bond_order = 1.0
            continue
    return atoms, edges, ring_count


def scaffold_signature(smiles: str) -> str:
    atoms, edges_w, ring_count = parse_smiles_graph(smiles)
    if not atoms:
        return "empty"
    edges = [(u, v) for u, v, _ in edges_w]
    cyc_nodes = cycle_nodes_from_edges(len(atoms), edges)
    if cyc_nodes:
        core_symbols = sorted([atoms[i].lower() for i in cyc_nodes])
    else:
        degrees = np.zeros(len(atoms), dtype=int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        core = [i for i, d in enumerate(degrees.tolist()) if d >= 2]
        if not core:
            core = list(range(min(3, len(atoms))))
        core_symbols = sorted([atoms[i].lower() for i in core])
    aromatic = sum(1 for a in atoms if a.islower())
    uniq = "-".join(sorted(set([a.lower() for a in atoms])))
    return f"r{ring_count}|a{aromatic}|c{''.join(core_symbols)}|u{uniq}|n{len(atoms)}"


class GraphYieldDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cond_cols: Sequence[str], cond_columns_fit: Sequence[str], y_col: str, atom_vocab: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.y = self.df[y_col].astype(float).to_numpy(dtype=np.float32)
        cond = pd.get_dummies(self.df[list(cond_cols)].astype(str), prefix=list(cond_cols), dtype=np.float32)
        cond = cond.reindex(columns=list(cond_columns_fit), fill_value=0.0)
        self.cond = cond.to_numpy(dtype=np.float32)
        self.atom_vocab = atom_vocab
        self.graphs = [parse_smiles_graph(s) for s in self.df["ligand"].astype(str)]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        atoms, edges, _ = self.graphs[idx]
        atom_ids = [self.atom_vocab.get(a.lower(), 1) for a in atoms]
        aromatic = [1.0 if a.islower() else 0.0 for a in atoms]
        n = len(atoms)
        degrees = [0.0] * n
        for u, v, _ in edges:
            degrees[u] += 1.0
            degrees[v] += 1.0
        aux = [[aromatic[i], degrees[i] / 4.0] for i in range(n)]
        return {
            "atom_ids": atom_ids,
            "aux": aux,
            "edges": edges,
            "cond": self.cond[idx],
            "y": float(self.y[idx]),
        }


def collate_graph_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    bsz = len(batch)
    max_nodes = max(len(x["atom_ids"]) for x in batch)
    cond_dim = len(batch[0]["cond"])
    atom_ids = torch.zeros((bsz, max_nodes), dtype=torch.long)
    aux = torch.zeros((bsz, max_nodes, 2), dtype=torch.float32)
    adj = torch.zeros((bsz, max_nodes, max_nodes), dtype=torch.float32)
    mask = torch.zeros((bsz, max_nodes), dtype=torch.float32)
    cond = torch.zeros((bsz, cond_dim), dtype=torch.float32)
    y = torch.zeros((bsz,), dtype=torch.float32)

    for i, item in enumerate(batch):
        ids = item["atom_ids"]
        n = len(ids)
        atom_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        aux[i, :n] = torch.tensor(item["aux"], dtype=torch.float32)
        mask[i, :n] = 1.0
        cond[i] = torch.tensor(item["cond"], dtype=torch.float32)
        y[i] = torch.tensor(item["y"], dtype=torch.float32)
        for u, v, bo in item["edges"]:
            adj[i, u, v] += float(bo)
            adj[i, v, u] += float(bo)
        for j in range(n):
            adj[i, j, j] += 1.0
        deg = adj[i, :n, :n].sum(dim=1, keepdim=True) + 1e-8
        adj[i, :n, :n] = adj[i, :n, :n] / deg
    return {"atom_ids": atom_ids, "aux": aux, "adj": adj, "mask": mask, "cond": cond, "y": y}


class SimpleGraphYieldNet(nn.Module):
    def __init__(self, atom_vocab_size: int, cond_dim: int, hidden_dim: int = 96, n_layers: int = 3):
        super().__init__()
        self.atom_emb = nn.Embedding(atom_vocab_size, hidden_dim, padding_idx=0)
        self.aux_proj = nn.Linear(2, hidden_dim)
        self.self_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.nei_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, atom_ids: torch.Tensor, aux: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.atom_emb(atom_ids) + self.aux_proj(aux)
        for i in range(len(self.self_linears)):
            m = torch.bmm(adj, h)
            h_new = self.self_linears[i](h) + self.nei_linears[i](m)
            h_new = torch.relu(self.norms[i](h_new))
            h = self.dropout(h_new)
        mask_e = mask.unsqueeze(-1)
        pooled = (h * mask_e).sum(dim=1) / (mask_e.sum(dim=1) + 1e-8)
        out = self.head(torch.cat([pooled, cond], dim=1)).squeeze(-1)
        return out


def build_atom_vocab(smiles_list: Sequence[str]) -> Dict[str, int]:
    symbols = set()
    for s in smiles_list:
        atoms, _, _ = parse_smiles_graph(str(s))
        symbols.update([a.lower() for a in atoms])
    vocab = {"<pad>": 0, "<unk>": 1}
    for sym in sorted(symbols):
        vocab[sym] = len(vocab)
    return vocab


def train_graph_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cond_cols: Sequence[str],
    cond_columns_fit: Sequence[str],
    y_col: str,
    atom_vocab: Dict[str, int],
    epochs: int = 20,
    lr: float = 8e-4,
) -> SimpleGraphYieldNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = GraphYieldDataset(train_df, cond_cols=cond_cols, cond_columns_fit=cond_columns_fit, y_col=y_col, atom_vocab=atom_vocab)
    val_ds = GraphYieldDataset(val_df, cond_cols=cond_cols, cond_columns_fit=cond_columns_fit, y_col=y_col, atom_vocab=atom_vocab)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_graph_batch)

    model = SimpleGraphYieldNet(atom_vocab_size=len(atom_vocab), cond_dim=len(cond_columns_fit)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    best_rmse = float("inf")
    best_state = None
    patience = 6
    bad = 0

    for _ in range(epochs):
        model.train()
        for b in train_loader:
            optim.zero_grad()
            pred = model(
                b["atom_ids"].to(device),
                b["aux"].to(device),
                b["adj"].to(device),
                b["mask"].to(device),
                b["cond"].to(device),
            )
            loss = loss_fn(pred, b["y"].to(device))
            loss.backward()
            optim.step()

        model.eval()
        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []
        with torch.no_grad():
            for b in val_loader:
                p = model(
                    b["atom_ids"].to(device),
                    b["aux"].to(device),
                    b["adj"].to(device),
                    b["mask"].to(device),
                    b["cond"].to(device),
                )
                preds.append(p.cpu().numpy())
                trues.append(b["y"].cpu().numpy())
        val_rmse = rmse(np.concatenate(trues), np.concatenate(preds))
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_graph_model(
    model: SimpleGraphYieldNet,
    df: pd.DataFrame,
    cond_cols: Sequence[str],
    cond_columns_fit: Sequence[str],
    atom_vocab: Dict[str, int],
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GraphYieldDataset(df, cond_cols=cond_cols, cond_columns_fit=cond_columns_fit, y_col="yield", atom_vocab=atom_vocab)
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_graph_batch)
    model = model.to(device)
    model.eval()
    preds = []
    for b in loader:
        p = model(
            b["atom_ids"].to(device),
            b["aux"].to(device),
            b["adj"].to(device),
            b["mask"].to(device),
            b["cond"].to(device),
        )
        preds.append(p.cpu().numpy())
    return np.concatenate(preds)


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-8)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * normal_cdf(z) + sigma * normal_pdf(z)


@dataclass
class ModelBundle:
    cond_columns: List[str]
    bootstrap_models: List[xgb.XGBRegressor]
    quantile_mean: GradientBoostingRegressor
    quantile_low: GradientBoostingRegressor
    quantile_high: GradientBoostingRegressor
    atom_vocab: Dict[str, int]
    gnn_model: SimpleGraphYieldNet
    cond_cols: List[str]
    fp_dim: int


def fit_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    y_col: str,
    fp_dim: int,
    seed: int,
    gnn_epochs: int,
) -> ModelBundle:
    x_train, cond_columns = build_tabular_features(train_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim)
    x_val, _ = build_tabular_features(val_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim, fit_columns=cond_columns)
    y_train = train_df[y_col].astype(float).to_numpy()
    _ = x_val

    boot = train_bootstrap_ensemble(x_train, y_train, n_models=25, seed=seed)
    q_mean, q_low, q_high = train_quantile_models(x_train, y_train)
    atom_vocab = build_atom_vocab(train_df[smiles_col].astype(str).tolist())
    gnn_model = train_graph_model(
        train_df.rename(columns={smiles_col: "ligand", y_col: "yield"}),
        val_df.rename(columns={smiles_col: "ligand", y_col: "yield"}),
        cond_cols=cond_cols,
        cond_columns_fit=cond_columns,
        y_col="yield",
        atom_vocab=atom_vocab,
        epochs=gnn_epochs,
    )
    return ModelBundle(
        cond_columns=cond_columns,
        bootstrap_models=boot,
        quantile_mean=q_mean,
        quantile_low=q_low,
        quantile_high=q_high,
        atom_vocab=atom_vocab,
        gnn_model=gnn_model,
        cond_cols=list(cond_cols),
        fp_dim=fp_dim,
    )


def evaluate_bundle(bundle: ModelBundle, test_df: pd.DataFrame, smiles_col: str, y_col: str) -> Dict[str, Dict[str, float]]:
    x_test, _ = build_tabular_features(
        test_df,
        cond_cols=bundle.cond_cols,
        smiles_col=smiles_col,
        fp_dim=bundle.fp_dim,
        fit_columns=bundle.cond_columns,
    )
    y = test_df[y_col].astype(float).to_numpy()
    mu, std = bootstrap_predict(bundle.bootstrap_models, x_test)
    q_mean = bundle.quantile_mean.predict(x_test)
    q_low = bundle.quantile_low.predict(x_test)
    q_high = bundle.quantile_high.predict(x_test)
    gnn_pred = predict_graph_model(
        bundle.gnn_model,
        test_df.rename(columns={smiles_col: "ligand", y_col: "yield"}),
        cond_cols=bundle.cond_cols,
        cond_columns_fit=bundle.cond_columns,
        atom_vocab=bundle.atom_vocab,
    )
    coverage = float(np.mean((y >= q_low) & (y <= q_high)))
    return {
        "bootstrap_mean": regression_metrics(y, mu),
        "quantile_mean": regression_metrics(y, q_mean),
        "gnn_graph": regression_metrics(y, gnn_pred),
        "uncertainty": {
            "bootstrap_std_mean": float(np.mean(std)),
            "quantile_interval_mean_width": float(np.mean(q_high - q_low)),
            "quantile_80pct_coverage": coverage,
        },
    }


def run_kfold_eval(
    df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    y_col: str,
    fp_dim: int,
    seed: int,
    n_splits: int = 5,
) -> Dict[str, object]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        x_train, cond_cols_fit = build_tabular_features(train_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim)
        x_test, _ = build_tabular_features(
            test_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim, fit_columns=cond_cols_fit
        )
        y_train = train_df[y_col].to_numpy()
        y_test = test_df[y_col].to_numpy()

        boot = train_bootstrap_ensemble(x_train, y_train, n_models=20, seed=seed + fold)
        mu, std = bootstrap_predict(boot, x_test)
        q_mean, q_low, q_high = train_quantile_models(x_train, y_train)
        q_pred = q_mean.predict(x_test)
        row = {
            "fold": fold,
            "bootstrap_rmse": rmse(y_test, mu),
            "bootstrap_r2": float(r2_score(y_test, mu)),
            "bootstrap_std_mean": float(np.mean(std)),
            "quantile_rmse": rmse(y_test, q_pred),
            "quantile_80pct_coverage": float(np.mean((y_test >= q_low.predict(x_test)) & (y_test <= q_high.predict(x_test)))),
        }
        rows.append(row)
    agg = {
        "bootstrap_rmse_mean": float(np.mean([r["bootstrap_rmse"] for r in rows])),
        "bootstrap_rmse_std": float(np.std([r["bootstrap_rmse"] for r in rows])),
        "bootstrap_r2_mean": float(np.mean([r["bootstrap_r2"] for r in rows])),
        "quantile_rmse_mean": float(np.mean([r["quantile_rmse"] for r in rows])),
        "quantile_coverage_mean": float(np.mean([r["quantile_80pct_coverage"] for r in rows])),
    }
    return {"folds": rows, "aggregate": agg}


def run_scaffold_group_kfold(
    df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    y_col: str,
    fp_dim: int,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    _ = seed
    groups = df[smiles_col].astype(str).map(scaffold_signature).to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(df, groups=groups), start=1):
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        x_train, cond_fit = build_tabular_features(train_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim)
        x_test, _ = build_tabular_features(test_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim, fit_columns=cond_fit)
        y_train = train_df[y_col].to_numpy()
        y_test = test_df[y_col].to_numpy()
        boot = train_bootstrap_ensemble(x_train, y_train, n_models=20, seed=100 + fold)
        mu, std = bootstrap_predict(boot, x_test)
        rows.append(
            {
                "fold": fold,
                "rmse": rmse(y_test, mu),
                "r2": float(r2_score(y_test, mu)),
                "std_mean": float(np.mean(std)),
                "n_test": int(len(test_df)),
            }
        )
    agg = {
        "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
        "rmse_std": float(np.std([r["rmse"] for r in rows])),
        "r2_mean": float(np.mean([r["r2"] for r in rows])),
    }
    return {"folds": rows, "aggregate": agg}


def discrete_bo_loop(
    candidate_df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    y_col: str,
    fp_dim: int,
    n_init: int = 20,
    n_iter: int = 40,
    beta: float = 1.5,
    seed: int = 42,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n = len(candidate_df)
    all_idx = np.arange(n)
    observed = rng.choice(all_idx, size=n_init, replace=False).tolist()
    remaining = [i for i in all_idx.tolist() if i not in observed]
    best_so_far = float(candidate_df.iloc[observed][y_col].max())
    history = [{"iter": 0, "best_yield": best_so_far}]

    for it in range(1, n_iter + 1):
        train_df = candidate_df.iloc[observed].reset_index(drop=True)
        rem_df = candidate_df.iloc[remaining].reset_index(drop=True)
        x_train, cond_fit = build_tabular_features(train_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim)
        y_train = train_df[y_col].to_numpy()
        x_rem, _ = build_tabular_features(rem_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim, fit_columns=cond_fit)
        models = train_bootstrap_ensemble(x_train, y_train, n_models=15, seed=seed + it)
        mu, std = bootstrap_predict(models, x_rem)
        acq = mu + beta * std
        idx_local = int(np.argmax(acq))
        idx_global = remaining[idx_local]
        observed.append(idx_global)
        remaining.pop(idx_local)
        best_so_far = max(best_so_far, float(candidate_df.iloc[idx_global][y_col]))
        history.append({"iter": it, "best_yield": best_so_far})
        if not remaining:
            break
    return {"history": history, "best_yield": best_so_far}


def recommend_from_discrete_space(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    y_col: str,
    fp_dim: int,
    top_k: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    x_train, cond_fit = build_tabular_features(train_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim)
    y_train = train_df[y_col].to_numpy()
    models = train_bootstrap_ensemble(x_train, y_train, n_models=30, seed=seed)
    x_cand, _ = build_tabular_features(candidate_df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim, fit_columns=cond_fit)
    mu, std = bootstrap_predict(models, x_cand)
    best = float(y_train.max())
    ei = expected_improvement(mu, std, best=best, xi=0.01)
    out = candidate_df.copy()
    out["pred_mean"] = mu
    out["pred_std"] = std
    out["acq_ucb"] = mu + 1.5 * std
    out["acq_ei"] = ei
    out = out.sort_values(["acq_ei", "pred_mean"], ascending=False).head(top_k).reset_index(drop=True)
    return out


def plot_uncertainty_scatter(y_true: np.ndarray, y_pred: np.ndarray, std: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, c=std, cmap="viridis", s=30, alpha=0.9)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--")
    cbar = plt.colorbar()
    cbar.set_label("Predictive std")
    plt.xlabel("True yield")
    plt.ylabel("Predicted yield")
    plt.title("Bootstrap prediction with uncertainty")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_bo_curve(history: List[Dict[str, float]], save_path: str) -> None:
    xs = [h["iter"] for h in history]
    ys = [h["best_yield"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("BO iteration")
    plt.ylabel("Best observed yield")
    plt.title("Discrete BO closed-loop progress")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def default_condition_columns(df: pd.DataFrame) -> List[str]:
    candidates = ["reaction", "additive", "base", "aryl halide"]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        raise ValueError("No default condition columns found.")
    return cols


def run_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "figures"))

    df = pd.read_csv(args.data_path)
    if args.reaction_id is not None and "reaction" in df.columns:
        df = df[df["reaction"] == args.reaction_id].reset_index(drop=True)
    if args.y_col not in df.columns or args.smiles_col not in df.columns:
        raise ValueError("Configured y_col/smiles_col not found in dataset.")

    cond_cols = default_condition_columns(df) if args.cond_cols == "auto" else [x.strip() for x in args.cond_cols.split(",")]
    df = df.dropna(subset=[args.y_col, args.smiles_col] + cond_cols).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=args.val_size / (1 - args.test_size), random_state=args.seed)
    bundle = fit_all_models(
        train_df=train_df,
        val_df=val_df,
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        fp_dim=args.fp_dim,
        seed=args.seed,
        gnn_epochs=args.gnn_epochs,
    )
    eval_out = evaluate_bundle(bundle, test_df=test_df, smiles_col=args.smiles_col, y_col=args.y_col)

    x_test, _ = build_tabular_features(
        test_df,
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        fp_dim=args.fp_dim,
        fit_columns=bundle.cond_columns,
    )
    y_test = test_df[args.y_col].to_numpy()
    mu, std = bootstrap_predict(bundle.bootstrap_models, x_test)

    kfold = run_kfold_eval(
        df=df,
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        fp_dim=args.fp_dim,
        seed=args.seed,
        n_splits=args.kfolds,
    )
    scaffold_kfold = run_scaffold_group_kfold(
        df=df,
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        fp_dim=args.fp_dim,
        n_splits=args.kfolds,
        seed=args.seed,
    )
    bo_out = discrete_bo_loop(
        candidate_df=df.copy(),
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        fp_dim=args.fp_dim,
        n_init=args.bo_init,
        n_iter=args.bo_iter,
        beta=args.bo_beta,
        seed=args.seed,
    )
    recs = recommend_from_discrete_space(
        train_df=df.copy(),
        candidate_df=df.copy(),
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        fp_dim=args.fp_dim,
        top_k=args.top_k,
        seed=args.seed,
    )

    plot_uncertainty_scatter(
        y_true=y_test,
        y_pred=mu,
        std=std,
        save_path=os.path.join(args.output_dir, "figures", "bootstrap_uncertainty_scatter.png"),
    )
    plot_bo_curve(
        history=bo_out["history"],
        save_path=os.path.join(args.output_dir, "figures", "discrete_bo_curve.png"),
    )

    recs.to_csv(os.path.join(args.output_dir, "top_recommendations.csv"), index=False)

    payload = {
        "data_path": args.data_path,
        "n_samples": int(len(df)),
        "condition_columns": cond_cols,
        "smiles_col": args.smiles_col,
        "target_col": args.y_col,
        "test_metrics": eval_out,
        "random_kfold": kfold,
        "scaffold_group_kfold": scaffold_kfold,
        "bo": bo_out,
        "notes": {
            "scaffold_method": "Lightweight scaffold signature from SMILES graph parser (RDKit-free approximation).",
            "uncertainty": "Bootstrap predictive std + quantile interval (0.1, 0.9).",
            "bo_space": "Discrete candidate set; acquisition supports UCB/EI ranking.",
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # save core artifacts for predict/suggest subcommands
    torch.save(bundle.gnn_model.state_dict(), os.path.join(args.output_dir, "gnn_model.pt"))
    meta = {
        "cond_columns": bundle.cond_columns,
        "cond_cols": bundle.cond_cols,
        "atom_vocab": bundle.atom_vocab,
        "fp_dim": bundle.fp_dim,
        "smiles_col": args.smiles_col,
        "y_col": args.y_col,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # xgboost bootstrap
    boot_dir = os.path.join(args.output_dir, "bootstrap_models")
    ensure_dir(boot_dir)
    for i, m in enumerate(bundle.bootstrap_models):
        m.save_model(os.path.join(boot_dir, f"model_{i}.json"))
    # quantile models
    import joblib

    joblib.dump(bundle.quantile_mean, os.path.join(args.output_dir, "quantile_mean.joblib"))
    joblib.dump(bundle.quantile_low, os.path.join(args.output_dir, "quantile_low.joblib"))
    joblib.dump(bundle.quantile_high, os.path.join(args.output_dir, "quantile_high.joblib"))

    print("Training done.")
    print("Output dir:", args.output_dir)
    print("Test metrics:", json.dumps(eval_out, indent=2))
    print("Scaffold GroupKFold aggregate:", json.dumps(scaffold_kfold["aggregate"], indent=2))


def load_trained_for_inference(output_dir: str) -> Tuple[Dict[str, object], List[xgb.XGBRegressor]]:
    meta = json.loads(open(os.path.join(output_dir, "meta.json"), "r", encoding="utf-8").read())
    boot_dir = os.path.join(output_dir, "bootstrap_models")
    names = sorted([n for n in os.listdir(boot_dir) if n.endswith(".json")])
    models = []
    for n in names:
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(boot_dir, n))
        models.append(m)
    return meta, models


def run_predict(args: argparse.Namespace) -> None:
    meta, models = load_trained_for_inference(args.output_dir)
    cond_cols = meta["cond_cols"]
    cond_columns = meta["cond_columns"]
    fp_dim = int(meta["fp_dim"])
    smiles_col = meta["smiles_col"]
    one_row = {smiles_col: args.smiles}
    for c in cond_cols:
        one_row[c] = getattr(args, c.replace(" ", "_"))
    df = pd.DataFrame([one_row])
    x, _ = build_tabular_features(df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim, fit_columns=cond_columns)
    mu, std = bootstrap_predict(models, x)
    out = {"pred_mean": float(mu[0]), "pred_std": float(std[0]), "ucb_beta_1_5": float(mu[0] + 1.5 * std[0])}
    print(json.dumps(out, ensure_ascii=False, indent=2))


def run_suggest(args: argparse.Namespace) -> None:
    meta, models = load_trained_for_inference(args.output_dir)
    cond_cols = meta["cond_cols"]
    cond_columns = meta["cond_columns"]
    fp_dim = int(meta["fp_dim"])
    smiles_col = meta["smiles_col"]
    df = pd.read_csv(args.candidate_data)
    if args.reaction_id is not None and "reaction" in df.columns:
        df = df[df["reaction"] == args.reaction_id].reset_index(drop=True)
    x, _ = build_tabular_features(df, cond_cols=cond_cols, smiles_col=smiles_col, fp_dim=fp_dim, fit_columns=cond_columns)
    mu, std = bootstrap_predict(models, x)
    ei = expected_improvement(mu, std, best=float(np.max(mu)), xi=0.01)
    out = df.copy()
    out["pred_mean"] = mu
    out["pred_std"] = std
    out["acq_ucb"] = mu + 1.5 * std
    out["acq_ei"] = ei
    out = out.sort_values(["acq_ei", "pred_mean"], ascending=False).head(args.top_k)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved top-{args.top_k} suggestions to {args.out_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-yield modeling, uncertainty, scaffold CV, graph GNN, and discrete BO.")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train and evaluate full pipeline")
    t.add_argument("--data-path", type=str, default="data/bh-reactions.csv")
    t.add_argument("--output-dir", type=str, default="yield_bo_outputs")
    t.add_argument("--y-col", type=str, default="yield")
    t.add_argument("--smiles-col", type=str, default="ligand")
    t.add_argument("--cond-cols", type=str, default="auto", help="Comma-separated condition columns or auto")
    t.add_argument("--reaction-id", type=int, default=0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--fp-dim", type=int, default=256)
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--val-size", type=float, default=0.2)
    t.add_argument("--kfolds", type=int, default=5)
    t.add_argument("--gnn-epochs", type=int, default=20)
    t.add_argument("--bo-init", type=int, default=20)
    t.add_argument("--bo-iter", type=int, default=40)
    t.add_argument("--bo-beta", type=float, default=1.5)
    t.add_argument("--top-k", type=int, default=20)

    pred = sub.add_parser("predict", help="Predict yield for one new ligand-condition tuple")
    pred.add_argument("--output-dir", type=str, default="yield_bo_outputs")
    pred.add_argument("--smiles", type=str, required=True)
    pred.add_argument("--reaction", type=str, default="0")
    pred.add_argument("--additive", type=str, required=True)
    pred.add_argument("--base", type=str, required=True)
    pred.add_argument("--aryl_halide", type=str, required=True)

    sug = sub.add_parser("suggest", help="Rank discrete candidate molecules from candidate pool")
    sug.add_argument("--output-dir", type=str, default="yield_bo_outputs")
    sug.add_argument("--candidate-data", type=str, default="data/bh-reactions.csv")
    sug.add_argument("--reaction-id", type=int, default=0)
    sug.add_argument("--top-k", type=int, default=20)
    sug.add_argument("--out-csv", type=str, default="yield_bo_outputs/next_candidates.csv")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        setattr(args, "reaction", getattr(args, "reaction", "0"))
        run_predict(args)
    elif args.command == "suggest":
        run_suggest(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
