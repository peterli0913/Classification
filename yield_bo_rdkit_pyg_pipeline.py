"""
RDKit + PyG based reaction-yield modeling pipeline.

Upgrades over baseline:
1) Standard Bemis-Murcko scaffold grouping from RDKit
2) Ligand graph neural network (PyTorch Geometric)
3) Bootstrap and quantile uncertainty
4) Discrete BO recommendations on candidate pool
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


RDLogger.DisableLog("rdApp.warning")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def safe_mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol


def morgan_fp(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    mol = safe_mol_from_smiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    on_bits = list(fp.GetOnBits())
    arr[on_bits] = 1.0
    return arr


def murcko_scaffold(smiles: str) -> str:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return "invalid"
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaf:
        return scaf
    # fallback for molecules without rings
    return Chem.MolToSmiles(mol, canonical=True)


def build_condition_maps(df: pd.DataFrame, cond_cols: Sequence[str]) -> Dict[str, Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    for c in cond_cols:
        uniq = sorted(df[c].astype(str).unique().tolist())
        maps[c] = {v: i for i, v in enumerate(uniq)}
    return maps


def encode_conditions(df: pd.DataFrame, cond_cols: Sequence[str], cond_maps: Dict[str, Dict[str, int]]) -> np.ndarray:
    parts = []
    for c in cond_cols:
        vocab = cond_maps[c]
        one_hot = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for i, v in enumerate(df[c].astype(str).tolist()):
            idx = vocab.get(v)
            if idx is not None:
                one_hot[i, idx] = 1.0
        parts.append(one_hot)
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(df), 0), dtype=np.float32)


def build_tabular_features(
    df: pd.DataFrame,
    cond_cols: Sequence[str],
    cond_maps: Dict[str, Dict[str, int]],
    smiles_col: str,
    fp_bits: int,
) -> np.ndarray:
    cond_x = encode_conditions(df, cond_cols=cond_cols, cond_maps=cond_maps)
    fp_x = np.stack([morgan_fp(s, n_bits=fp_bits) for s in df[smiles_col].astype(str).tolist()], axis=0)
    return np.concatenate([cond_x, fp_x], axis=1)


def train_bootstrap(
    x_train: np.ndarray, y_train: np.ndarray, n_models: int = 30, seed: int = 42
) -> List[xgb.XGBRegressor]:
    rng = np.random.default_rng(seed)
    models: List[xgb.XGBRegressor] = []
    n = len(x_train)
    for i in range(n_models):
        idx = rng.integers(0, n, n)
        model = xgb.XGBRegressor(
            n_estimators=450,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed + i,
        )
        model.fit(x_train[idx], y_train[idx], verbose=False)
        models.append(model)
    return models


def predict_bootstrap(models: Sequence[xgb.XGBRegressor], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.stack([m.predict(x) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def train_quantiles(
    x_train: np.ndarray, y_train: np.ndarray
) -> Tuple[GradientBoostingRegressor, GradientBoostingRegressor, GradientBoostingRegressor]:
    mean_model = GradientBoostingRegressor(loss="squared_error", n_estimators=320, learning_rate=0.03, max_depth=3)
    low_model = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=320, learning_rate=0.03, max_depth=3)
    high_model = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=320, learning_rate=0.03, max_depth=3)
    mean_model.fit(x_train, y_train)
    low_model.fit(x_train, y_train)
    high_model.fit(x_train, y_train)
    return mean_model, low_model, high_model


def atom_feature_vector(atom: Chem.Atom) -> List[float]:
    hyb = atom.GetHybridization()
    hyb_one_hot = [
        1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0,
    ]
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 5.0,
        atom.GetFormalCharge() / 4.0,
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
    ] + hyb_one_hot


def smiles_to_pyg_data(smiles: str, cond_vec: np.ndarray, y: float | None = None) -> Data:
    mol = safe_mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        x = torch.zeros((1, 8), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        x = torch.tensor([atom_feature_vector(a) for a in mol.GetAtoms()], dtype=torch.float32)
        edges = []
        for b in mol.GetBonds():
            u = b.GetBeginAtomIdx()
            v = b.GetEndAtomIdx()
            edges.append((u, v))
            edges.append((v, u))
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        cond=torch.tensor(cond_vec, dtype=torch.float32),
    )
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data


class LigandGCN(nn.Module):
    def __init__(self, node_dim: int, cond_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.gcn1 = GCNConv(node_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.gcn3(x, edge_index))
        pooled = global_mean_pool(x, batch)
        cond = data.cond.view(pooled.size(0), -1)
        fused = torch.cat([pooled, cond], dim=1)
        out = self.head(fused).squeeze(-1)
        return out


def train_gnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cond_cols: Sequence[str],
    cond_maps: Dict[str, Dict[str, int]],
    smiles_col: str,
    y_col: str,
    epochs: int,
    lr: float = 9e-4,
) -> LigandGCN:
    cond_train = encode_conditions(train_df, cond_cols=cond_cols, cond_maps=cond_maps)
    cond_val = encode_conditions(val_df, cond_cols=cond_cols, cond_maps=cond_maps)
    train_data = [
        smiles_to_pyg_data(
            train_df.iloc[i][smiles_col],
            cond_train[i],
            y=float(train_df.iloc[i][y_col]),
        )
        for i in range(len(train_df))
    ]
    val_data = [
        smiles_to_pyg_data(
            val_df.iloc[i][smiles_col],
            cond_val[i],
            y=float(val_df.iloc[i][y_col]),
        )
        for i in range(len(val_df))
    ]

    train_loader = PyGDataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LigandGCN(node_dim=8, cond_dim=cond_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    best_rmse = float("inf")
    best_state = None
    bad = 0
    patience = 6

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                p = model(batch).detach().cpu().numpy()
                preds.append(p)
                trues.append(batch.y.view(-1).cpu().numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        cur_rmse = rmse(y_true, y_pred)
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
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
def predict_gnn(
    model: LigandGCN,
    df: pd.DataFrame,
    cond_cols: Sequence[str],
    cond_maps: Dict[str, Dict[str, int]],
    smiles_col: str,
) -> np.ndarray:
    cond_x = encode_conditions(df, cond_cols=cond_cols, cond_maps=cond_maps)
    data_list = [smiles_to_pyg_data(df.iloc[i][smiles_col], cond_x[i], y=None) for i in range(len(df))]
    loader = PyGDataLoader(data_list, batch_size=128, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).detach().cpu().numpy())
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
class TrainedArtifacts:
    cond_cols: List[str]
    cond_maps: Dict[str, Dict[str, int]]
    fp_bits: int
    smiles_col: str
    y_col: str


def run_kfold(
    df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    y_col: str,
    fp_bits: int,
    seed: int,
    n_splits: int,
) -> Dict[str, object]:
    rows = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        cond_maps = build_condition_maps(train_df, cond_cols=cond_cols)
        x_train = build_tabular_features(train_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=smiles_col, fp_bits=fp_bits)
        x_test = build_tabular_features(test_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=smiles_col, fp_bits=fp_bits)
        y_train = train_df[y_col].to_numpy()
        y_test = test_df[y_col].to_numpy()
        boot = train_bootstrap(x_train, y_train, n_models=20, seed=seed + fold)
        mu, std = predict_bootstrap(boot, x_test)
        rows.append(
            {
                "fold": fold,
                "rmse": rmse(y_test, mu),
                "r2": float(r2_score(y_test, mu)),
                "std_mean": float(np.mean(std)),
                "n_test": int(len(test_df)),
            }
        )
    return {
        "folds": rows,
        "aggregate": {
            "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
            "rmse_std": float(np.std([r["rmse"] for r in rows])),
            "r2_mean": float(np.mean([r["r2"] for r in rows])),
        },
    }


def run_scaffold_group_kfold(
    df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    y_col: str,
    fp_bits: int,
    n_splits: int,
    seed: int,
) -> Dict[str, object]:
    _ = seed
    groups = df[smiles_col].astype(str).map(murcko_scaffold).to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(df, groups=groups), start=1):
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        cond_maps = build_condition_maps(train_df, cond_cols=cond_cols)
        x_train = build_tabular_features(train_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=smiles_col, fp_bits=fp_bits)
        x_test = build_tabular_features(test_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=smiles_col, fp_bits=fp_bits)
        y_train = train_df[y_col].to_numpy()
        y_test = test_df[y_col].to_numpy()
        boot = train_bootstrap(x_train, y_train, n_models=20, seed=100 + fold)
        mu, std = predict_bootstrap(boot, x_test)
        rows.append(
            {
                "fold": fold,
                "rmse": rmse(y_test, mu),
                "r2": float(r2_score(y_test, mu)),
                "std_mean": float(np.mean(std)),
                "n_test": int(len(test_df)),
            }
        )
    return {
        "folds": rows,
        "aggregate": {
            "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
            "rmse_std": float(np.std([r["rmse"] for r in rows])),
            "r2_mean": float(np.mean([r["r2"] for r in rows])),
        },
    }


def discrete_bo(
    df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    y_col: str,
    fp_bits: int,
    n_init: int,
    n_iter: int,
    beta: float,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(df))
    observed = rng.choice(idx_all, size=n_init, replace=False).tolist()
    remain = [i for i in idx_all.tolist() if i not in observed]
    best_y = float(df.iloc[observed][y_col].max())
    history = [{"iter": 0, "best_yield": best_y}]

    for it in range(1, n_iter + 1):
        tr_df = df.iloc[observed].reset_index(drop=True)
        re_df = df.iloc[remain].reset_index(drop=True)
        cond_maps = build_condition_maps(tr_df, cond_cols=cond_cols)
        x_train = build_tabular_features(tr_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=smiles_col, fp_bits=fp_bits)
        y_train = tr_df[y_col].to_numpy()
        x_re = build_tabular_features(re_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=smiles_col, fp_bits=fp_bits)
        models = train_bootstrap(x_train, y_train, n_models=15, seed=seed + it)
        mu, std = predict_bootstrap(models, x_re)
        acq = mu + beta * std
        local = int(np.argmax(acq))
        global_idx = remain[local]
        observed.append(global_idx)
        remain.pop(local)
        best_y = max(best_y, float(df.iloc[global_idx][y_col]))
        history.append({"iter": it, "best_yield": best_y})
        if not remain:
            break
    return {"history": history, "best_yield": best_y}


def recommend_candidates(
    train_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cond_cols: Sequence[str],
    smiles_col: str,
    y_col: str,
    fp_bits: int,
    top_k: int,
    seed: int,
) -> pd.DataFrame:
    cond_maps = build_condition_maps(train_df, cond_cols=cond_cols)
    x_train = build_tabular_features(train_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=smiles_col, fp_bits=fp_bits)
    y_train = train_df[y_col].to_numpy()
    x_cand = build_tabular_features(cand_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=smiles_col, fp_bits=fp_bits)
    models = train_bootstrap(x_train, y_train, n_models=30, seed=seed)
    mu, std = predict_bootstrap(models, x_cand)
    best = float(np.max(y_train))
    ei = expected_improvement(mu, std, best=best, xi=0.01)
    out = cand_df.copy()
    out["pred_mean"] = mu
    out["pred_std"] = std
    out["acq_ucb"] = mu + 1.5 * std
    out["acq_ei"] = ei
    out = out.sort_values(["acq_ei", "pred_mean"], ascending=False).head(top_k).reset_index(drop=True)
    return out


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, std: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, c=std, cmap="viridis", s=30)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--")
    cbar = plt.colorbar()
    cbar.set_label("Predictive std")
    plt.xlabel("True yield")
    plt.ylabel("Predicted yield")
    plt.title("RDKit+Bootstrap uncertainty scatter")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_bo(history: List[Dict[str, float]], save_path: str) -> None:
    xs = [h["iter"] for h in history]
    ys = [h["best_yield"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("BO iteration")
    plt.ylabel("Best observed yield")
    plt.title("Discrete BO progress")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def default_cond_cols(df: pd.DataFrame) -> List[str]:
    cand = ["reaction", "additive", "base", "aryl halide"]
    out = [c for c in cand if c in df.columns]
    if not out:
        raise ValueError("No default condition columns found.")
    return out


def run_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "figures"))

    df = pd.read_csv(args.data_path)
    if args.reaction_id is not None and "reaction" in df.columns:
        df = df[df["reaction"] == args.reaction_id].reset_index(drop=True)
    cond_cols = default_cond_cols(df) if args.cond_cols == "auto" else [x.strip() for x in args.cond_cols.split(",")]
    df = df.dropna(subset=[args.smiles_col, args.y_col] + cond_cols).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=args.val_size / (1 - args.test_size), random_state=args.seed)

    cond_maps = build_condition_maps(train_df, cond_cols=cond_cols)
    x_train = build_tabular_features(train_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=args.smiles_col, fp_bits=args.fp_bits)
    x_test = build_tabular_features(test_df, cond_cols=cond_cols, cond_maps=cond_maps, smiles_col=args.smiles_col, fp_bits=args.fp_bits)
    y_train = train_df[args.y_col].to_numpy()
    y_test = test_df[args.y_col].to_numpy()

    boot_models = train_bootstrap(x_train, y_train, n_models=args.n_bootstrap, seed=args.seed)
    mu, std = predict_bootstrap(boot_models, x_test)
    q_mean, q_low, q_high = train_quantiles(x_train, y_train)
    q_pred = q_mean.predict(x_test)

    gnn_model = train_gnn(
        train_df=train_df,
        val_df=val_df,
        cond_cols=cond_cols,
        cond_maps=cond_maps,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        epochs=args.gnn_epochs,
    )
    gnn_pred = predict_gnn(
        gnn_model,
        df=test_df,
        cond_cols=cond_cols,
        cond_maps=cond_maps,
        smiles_col=args.smiles_col,
    )

    kfold = run_kfold(
        df=df,
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        fp_bits=args.fp_bits,
        seed=args.seed,
        n_splits=args.kfolds,
    )
    scaffold_kfold = run_scaffold_group_kfold(
        df=df,
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        fp_bits=args.fp_bits,
        n_splits=args.kfolds,
        seed=args.seed,
    )
    bo_out = discrete_bo(
        df=df.copy(),
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        fp_bits=args.fp_bits,
        n_init=args.bo_init,
        n_iter=args.bo_iter,
        beta=args.bo_beta,
        seed=args.seed,
    )
    recs = recommend_candidates(
        train_df=df.copy(),
        cand_df=df.copy(),
        cond_cols=cond_cols,
        smiles_col=args.smiles_col,
        y_col=args.y_col,
        fp_bits=args.fp_bits,
        top_k=args.top_k,
        seed=args.seed,
    )

    test_metrics = {
        "bootstrap_mean": metrics(y_test, mu),
        "quantile_mean": metrics(y_test, q_pred),
        "gnn_pyg": metrics(y_test, gnn_pred),
        "uncertainty": {
            "bootstrap_std_mean": float(np.mean(std)),
            "quantile_80pct_coverage": float(np.mean((y_test >= q_low.predict(x_test)) & (y_test <= q_high.predict(x_test)))),
            "quantile_interval_mean_width": float(np.mean(q_high.predict(x_test) - q_low.predict(x_test))),
        },
    }

    plot_scatter(
        y_true=y_test,
        y_pred=mu,
        std=std,
        save_path=os.path.join(args.output_dir, "figures", "rdkit_bootstrap_uncertainty_scatter.png"),
    )
    plot_bo(
        history=bo_out["history"],
        save_path=os.path.join(args.output_dir, "figures", "rdkit_discrete_bo_curve.png"),
    )

    recs.to_csv(os.path.join(args.output_dir, "top_recommendations.csv"), index=False)

    summary = {
        "data_path": args.data_path,
        "n_samples": int(len(df)),
        "condition_columns": cond_cols,
        "smiles_col": args.smiles_col,
        "target_col": args.y_col,
        "test_metrics": test_metrics,
        "kfold": kfold,
        "scaffold_group_kfold": scaffold_kfold,
        "bo": bo_out,
        "notes": {
            "scaffold": "RDKit Bemis-Murcko scaffold via rdkit.Chem.Scaffolds.MurckoScaffold",
            "gnn": "PyTorch Geometric GCN over ligand molecular graph + one-hot condition context",
            "bo": "Discrete BO recommendation over candidate pool using uncertainty-aware acquisition",
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save artifacts for interactive inference
    boot_dir = os.path.join(args.output_dir, "bootstrap_models")
    ensure_dir(boot_dir)
    for i, model in enumerate(boot_models):
        model.save_model(os.path.join(boot_dir, f"model_{i}.json"))
    joblib.dump(q_mean, os.path.join(args.output_dir, "quantile_mean.joblib"))
    joblib.dump(q_low, os.path.join(args.output_dir, "quantile_low.joblib"))
    joblib.dump(q_high, os.path.join(args.output_dir, "quantile_high.joblib"))

    torch.save(gnn_model.state_dict(), os.path.join(args.output_dir, "gnn_pyg.pt"))
    meta = {
        "cond_cols": cond_cols,
        "cond_maps": cond_maps,
        "fp_bits": args.fp_bits,
        "smiles_col": args.smiles_col,
        "y_col": args.y_col,
        "n_bootstrap": args.n_bootstrap,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("RDKit+PyG training done.")
    print("Output dir:", args.output_dir)
    print(json.dumps(test_metrics, indent=2))
    print("Scaffold GroupKFold aggregate:", json.dumps(scaffold_kfold["aggregate"], indent=2))


def load_artifacts(output_dir: str) -> Tuple[TrainedArtifacts, List[xgb.XGBRegressor], LigandGCN]:
    meta_path = os.path.join(output_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
    art = TrainedArtifacts(
        cond_cols=meta["cond_cols"],
        cond_maps=meta["cond_maps"],
        fp_bits=int(meta["fp_bits"]),
        smiles_col=meta["smiles_col"],
        y_col=meta["y_col"],
    )

    boot_models: List[xgb.XGBRegressor] = []
    boot_dir = os.path.join(output_dir, "bootstrap_models")
    for name in sorted([x for x in os.listdir(boot_dir) if x.endswith(".json")]):
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(boot_dir, name))
        boot_models.append(m)

    cond_dim = sum(len(art.cond_maps[c]) for c in art.cond_cols)
    gnn = LigandGCN(node_dim=8, cond_dim=cond_dim)
    gnn.load_state_dict(torch.load(os.path.join(output_dir, "gnn_pyg.pt"), map_location="cpu"))
    gnn.eval()
    return art, boot_models, gnn


def run_predict(args: argparse.Namespace) -> None:
    art, boot, gnn = load_artifacts(args.output_dir)
    one = {art.smiles_col: args.smiles}
    for c in art.cond_cols:
        key = c.replace(" ", "_")
        one[c] = getattr(args, key)
    df = pd.DataFrame([one])
    x = build_tabular_features(df, cond_cols=art.cond_cols, cond_maps=art.cond_maps, smiles_col=art.smiles_col, fp_bits=art.fp_bits)
    mu, std = predict_bootstrap(boot, x)
    gnn_pred = predict_gnn(gnn, df=df, cond_cols=art.cond_cols, cond_maps=art.cond_maps, smiles_col=art.smiles_col)
    out = {
        "bootstrap_mean": float(mu[0]),
        "bootstrap_std": float(std[0]),
        "gnn_pyg": float(gnn_pred[0]),
        "ensemble_avg": float((mu[0] + gnn_pred[0]) / 2.0),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def run_suggest(args: argparse.Namespace) -> None:
    art, boot, gnn = load_artifacts(args.output_dir)
    df = pd.read_csv(args.candidate_data)
    if args.reaction_id is not None and "reaction" in df.columns:
        df = df[df["reaction"] == args.reaction_id].reset_index(drop=True)
    x = build_tabular_features(df, cond_cols=art.cond_cols, cond_maps=art.cond_maps, smiles_col=art.smiles_col, fp_bits=art.fp_bits)
    mu, std = predict_bootstrap(boot, x)
    gnn_pred = predict_gnn(gnn, df=df, cond_cols=art.cond_cols, cond_maps=art.cond_maps, smiles_col=art.smiles_col)
    blend = 0.7 * mu + 0.3 * gnn_pred
    ei = expected_improvement(blend, std, best=float(np.max(blend)))
    out = df.copy()
    out["pred_bootstrap_mean"] = mu
    out["pred_bootstrap_std"] = std
    out["pred_gnn"] = gnn_pred
    out["pred_blend"] = blend
    out["acq_ucb"] = blend + 1.5 * std
    out["acq_ei"] = ei
    out = out.sort_values(["acq_ei", "pred_blend"], ascending=False).head(args.top_k).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved top-{args.top_k} suggestions to {args.out_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RDKit + PyG yield modeling and discrete BO pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train full RDKit+PyG pipeline")
    t.add_argument("--data-path", type=str, default="data/bh-reactions.csv")
    t.add_argument("--output-dir", type=str, default="yield_bo_pyg_outputs")
    t.add_argument("--reaction-id", type=int, default=0)
    t.add_argument("--y-col", type=str, default="yield")
    t.add_argument("--smiles-col", type=str, default="ligand")
    t.add_argument("--cond-cols", type=str, default="auto")
    t.add_argument("--fp-bits", type=int, default=1024)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--val-size", type=float, default=0.2)
    t.add_argument("--kfolds", type=int, default=5)
    t.add_argument("--n-bootstrap", type=int, default=30)
    t.add_argument("--gnn-epochs", type=int, default=28)
    t.add_argument("--bo-init", type=int, default=20)
    t.add_argument("--bo-iter", type=int, default=40)
    t.add_argument("--bo-beta", type=float, default=1.5)
    t.add_argument("--top-k", type=int, default=20)

    pred = sub.add_parser("predict", help="Predict one ligand-condition tuple")
    pred.add_argument("--output-dir", type=str, default="yield_bo_pyg_outputs")
    pred.add_argument("--smiles", type=str, required=True)
    pred.add_argument("--reaction", type=str, default="0")
    pred.add_argument("--additive", type=str, required=True)
    pred.add_argument("--base", type=str, required=True)
    pred.add_argument("--aryl_halide", type=str, required=True)

    sug = sub.add_parser("suggest", help="Rank candidate pool with acquisition functions")
    sug.add_argument("--output-dir", type=str, default="yield_bo_pyg_outputs")
    sug.add_argument("--candidate-data", type=str, default="data/bh-reactions.csv")
    sug.add_argument("--reaction-id", type=int, default=0)
    sug.add_argument("--top-k", type=int, default=20)
    sug.add_argument("--out-csv", type=str, default="yield_bo_pyg_outputs/next_candidates.csv")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "suggest":
        run_suggest(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
