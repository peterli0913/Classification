"""
Scientific benchmark for public HTE yield datasets.

What this improves versus the previous ligand-only / random-split pipeline:
1) Multi-component fingerprints (ligand + additive + electrophile), not ligand only
2) Chemically meaningful splits (leave-one-ligand / additive / aryl / reaction)
3) Main-effects ANOVA baseline + residual booster (explicit interaction model)
4) Second public set: Perera Suzuki-Miyaura HTE
5) Split conformal intervals
6) Multi-seed discrete BO vs random / greedy

No new third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINConv, global_add_pool

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


# ---------------------------------------------------------------------------
# Ligand / reactant SMILES for the Perera Suzuki set (names in the Excel file)
# ---------------------------------------------------------------------------
SUZUKI_LIGAND_SMILES = {
    "P(tBu)3": "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
    "P(Ph)3": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
    "AmPhos": "CN(C)c1ccc(P(C(C)(C)C)C(C)(C)C)cc1",
    "P(Cy)3": "C1CCC(P(C2CCCCC2)C2CCCCC2)CC1",
    "P(o-Tol)3": "Cc1ccccc1P(c1ccccc1C)c1ccccc1C",
    "CataCXium A": "CCCCP(C12CC3CC(CC(C3)C1)C2)C12CC3CC(CC(C3)C1)C2",
    "SPhos": "COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1",
    "XPhos": "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1",
    "Xantphos": "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21",
    "dppf": "[Fe].c1ccc(P(c2ccccc2)C2C=CC=C2)cc1.c1ccc(P(c2ccccc2)C2C=CC=C2)cc1",
    "dtbpf": "[Fe].CC(C)(C)P(C1C=CC=C1)C(C)(C)C.CC(C)(C)P(C1C=CC=C1)C(C)(C)C",
}

SUZUKI_R1_SMILES = {
    "6-chloroquinoline": "Clc1ccc2ncccc2c1",
    "6-Bromoquinoline": "Brc1ccc2ncccc2c1",
    "6-triflatequinoline": "O=S(=O)(Oc1ccc2ncccc2c1)C(F)(F)F",
    "6-Iodoquinoline": "Ic1ccc2ncccc2c1",
    "6-quinoline-boronic acid hydrochloride": "OB(O)c1ccc2ncccc2c1",
    "Potassium quinoline-6-trifluoroborate": "F[B-](F)(F)c1ccc2ncccc2c1.[K+]",
    "6-Quinolineboronic acid pinacol ester": "CC1(C)OB(c2ccc3ncccc3c2)OC1(C)C",
}

DESC_FUNCS: List[Tuple[str, Callable]] = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("BertzCT", Descriptors.BertzCT),
    ("RingCount", Descriptors.RingCount),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("LabuteASA", Descriptors.LabuteASA),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
]

MODEL_COLORS = {
    "main_effects": "#7f8c8d",
    "onehot_xgb": "#4C78A8",
    "ligandfp_xgb": "#B279A2",
    "multifp_xgb": "#54A24B",
    "residual_xgb": "#F58518",
    "multifp_hgb": "#72B7B2",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    pred = np.clip(y_pred, 0.0, 100.0)
    return {
        "rmse": rmse(y_true, pred),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


def aggregate_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"n_folds": 0, "rmse_mean": float("nan"), "rmse_std": float("nan"), "r2_mean": float("nan"), "r2_std": float("nan")}
    return {
        "n_folds": int(len(rows)),
        "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
        "rmse_std": float(np.std([r["rmse"] for r in rows])),
        "mae_mean": float(np.mean([r["mae"] for r in rows])),
        "r2_mean": float(np.mean([r["r2"] for r in rows])),
        "r2_std": float(np.std([r["r2"] for r in rows])),
    }


def safe_mol(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


class MolFeaturizer:
    def __init__(self, n_bits: int = 512, radius: int = 2):
        self.n_bits = n_bits
        self.radius = radius
        self._fp: Dict[str, np.ndarray] = {}
        self._desc: Dict[str, np.ndarray] = {}

    def morgan(self, smiles: str) -> np.ndarray:
        key = smiles if isinstance(smiles, str) else ""
        if key not in self._fp:
            arr = np.zeros((self.n_bits,), dtype=np.float32)
            mol = safe_mol(key)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.n_bits)
                arr[list(fp.GetOnBits())] = 1.0
            self._fp[key] = arr
        return self._fp[key]

    def desc(self, smiles: str) -> np.ndarray:
        key = smiles if isinstance(smiles, str) else ""
        if key not in self._desc:
            mol = safe_mol(key)
            if mol is None:
                self._desc[key] = np.zeros((len(DESC_FUNCS),), dtype=np.float32)
            else:
                vals = []
                for _, fn in DESC_FUNCS:
                    try:
                        vals.append(float(fn(mol)))
                    except Exception:
                        vals.append(0.0)
                self._desc[key] = np.asarray(vals, dtype=np.float32)
        return self._desc[key]


def onehot_maps(series_list: Dict[str, pd.Series]) -> Dict[str, Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    for name, s in series_list.items():
        uniq = sorted({str(v) for v in s.tolist()})
        maps[name] = {v: i for i, v in enumerate(uniq)}
    return maps


def apply_onehot(series_list: Dict[str, pd.Series], maps: Dict[str, Dict[str, int]]) -> np.ndarray:
    parts = []
    n = len(next(iter(series_list.values())))
    for name, s in series_list.items():
        vocab = maps[name]
        mat = np.zeros((n, len(vocab)), dtype=np.float32)
        for i, v in enumerate(s.astype(str).tolist()):
            j = vocab.get(v)
            if j is not None:
                mat[i, j] = 1.0
        parts.append(mat)
    return np.concatenate(parts, axis=1) if parts else np.zeros((n, 0), dtype=np.float32)


def stack_mol(feat: MolFeaturizer, smiles: Sequence[str], kind: str) -> np.ndarray:
    if kind == "fp":
        return np.stack([feat.morgan(s) for s in smiles], axis=0)
    return np.stack([feat.desc(s) for s in smiles], axis=0)


class MainEffectsRegressor:
    """Additive ANOVA-style baseline: y ~ ligand + additive + base + electrophile."""

    def __init__(self):
        self.global_mean = 0.0
        self.effects: Dict[str, Dict[str, float]] = {}
        self.cols: List[str] = []

    def fit(self, df: pd.DataFrame, cols: Sequence[str], y: np.ndarray) -> "MainEffectsRegressor":
        self.cols = list(cols)
        self.global_mean = float(np.mean(y))
        tmp = df[self.cols].copy()
        tmp["_y"] = y
        self.effects = {}
        for c in self.cols:
            means = tmp.groupby(tmp[c].astype(str))["_y"].mean()
            self.effects[c] = {k: float(v - self.global_mean) for k, v in means.items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = np.full(len(df), self.global_mean, dtype=np.float64)
        for c in self.cols:
            eff = self.effects[c]
            pred += df[c].astype(str).map(eff).fillna(0.0).to_numpy(dtype=np.float64)
        return pred


def fit_xgb(x: np.ndarray, y: np.ndarray, seed: int) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=320,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.7,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=seed,
        tree_method="hist",
    )
    try:
        model.fit(x, y, verbose=False)
    except TypeError:
        model.fit(x, y)
    return model


def fit_hgb(x: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=280,
        l2_regularization=0.1,
        random_state=seed,
    )
    model.fit(x, y)
    return model


@dataclass
class DatasetSpec:
    name: str
    df: pd.DataFrame
    y: np.ndarray
    effect_cols: List[str]
    smiles_map: Dict[str, str]
    onehot_all: List[str]
    onehot_context: List[str]
    fp_cols: List[str]


def load_buchwald(path: str) -> DatasetSpec:
    df = pd.read_csv(path)
    df = df.dropna(subset=["ligand", "additive", "base", "aryl halide", "yield"]).copy()
    for c in ["ligand", "additive", "base", "aryl halide", "reaction"]:
        df[c] = df[c].astype(str)
    df["reaction"] = df["reaction"].astype(str)
    df = df.reset_index(drop=True)
    return DatasetSpec(
        name="buchwald_hartwig",
        df=df,
        y=df["yield"].to_numpy(dtype=np.float64),
        effect_cols=["ligand", "additive", "base", "aryl halide", "reaction"],
        smiles_map={"ligand": "ligand", "additive": "additive", "aryl halide": "aryl halide"},
        onehot_all=["ligand", "additive", "base", "aryl halide", "reaction"],
        onehot_context=["base", "reaction"],
        fp_cols=["ligand", "additive", "aryl halide"],
    )


def load_suzuki(path: str) -> DatasetSpec:
    raw = pd.read_excel(path)
    df = raw.copy()
    df["ligand"] = df["Ligand_Short_Hand"].astype(str).str.strip().replace({"nan": "none", "None": "none"})
    df.loc[df["Ligand_Short_Hand"].isna(), "ligand"] = "none"
    df["reagent"] = df["Reagent_1_Short_Hand"].astype(str).str.strip().replace({"nan": "none"})
    df.loc[df["Reagent_1_Short_Hand"].isna(), "reagent"] = "none"
    df["solvent"] = df["Solvent_1_Short_Hand"].astype(str).str.strip()
    df["reactant_1"] = df["Reactant_1_Name"].astype(str)
    df["reactant_2"] = df["Reactant_2_Name"].astype(str)
    df["yield"] = pd.to_numeric(df["Product_Yield_PCT_Area_UV"], errors="coerce")
    df = df.dropna(subset=["yield"]).reset_index(drop=True)
    df["ligand_smi"] = df["ligand"].map(SUZUKI_LIGAND_SMILES).fillna("")
    df["r1_smi"] = df["reactant_1"].map(SUZUKI_R1_SMILES).fillna("")
    return DatasetSpec(
        name="suzuki_miyaura",
        df=df,
        y=df["yield"].to_numpy(dtype=np.float64),
        effect_cols=["ligand", "reagent", "solvent", "reactant_1", "reactant_2"],
        smiles_map={"ligand": "ligand_smi", "reactant_1": "r1_smi"},
        onehot_all=["ligand", "reagent", "solvent", "reactant_1", "reactant_2"],
        onehot_context=["reagent", "solvent", "reactant_2"],
        fp_cols=["ligand", "reactant_1"],
    )


def smiles_series(spec: DatasetSpec, col_key: str) -> pd.Series:
    src = spec.smiles_map[col_key]
    return spec.df[src].astype(str)


def make_features(
    spec: DatasetSpec,
    idx: np.ndarray,
    feat: MolFeaturizer,
    maps: Dict[str, Dict[str, int]] | None,
    kind: str,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    df = spec.df.iloc[idx]
    if kind == "onehot":
        cols = {c: df[c] for c in spec.onehot_all}
        if maps is None:
            maps = onehot_maps(cols)
        return apply_onehot(cols, maps), maps

    if kind == "ligandfp":
        # Previous pipeline: fingerprint the ligand only, one-hot everything else.
        other = [c for c in spec.onehot_all if c != spec.fp_cols[0]]
        cols = {c: df[c] for c in other}
        if maps is None:
            maps = onehot_maps(cols)
        oh = apply_onehot(cols, maps)
        fp = stack_mol(feat, smiles_series(spec, spec.fp_cols[0]).iloc[idx].tolist(), "fp")
        return np.concatenate([fp, oh], axis=1), maps

    if kind in {"multifp", "residual"}:
        cols = {c: df[c] for c in spec.onehot_context}
        if maps is None:
            maps = onehot_maps(cols)
        oh = apply_onehot(cols, maps)
        parts = [oh]
        for key in spec.fp_cols:
            smi = smiles_series(spec, key).iloc[idx].tolist()
            parts.append(stack_mol(feat, smi, "fp"))
            parts.append(stack_mol(feat, smi, "desc"))
        return np.concatenate(parts, axis=1), maps

    raise ValueError(kind)


def predict_model(
    spec: DatasetSpec,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feat: MolFeaturizer,
    model_name: str,
    seed: int,
) -> np.ndarray:
    y_tr = spec.y[train_idx]
    df_tr = spec.df.iloc[train_idx]
    df_te = spec.df.iloc[test_idx]

    if model_name == "main_effects":
        me = MainEffectsRegressor().fit(df_tr, spec.effect_cols, y_tr)
        return me.predict(df_te)

    kind = "onehot" if model_name == "onehot_xgb" else ("ligandfp" if model_name == "ligandfp_xgb" else "multifp")
    x_tr, maps = make_features(spec, train_idx, feat, None, kind)
    x_te, _ = make_features(spec, test_idx, feat, maps, kind)

    if model_name == "residual_xgb":
        me = MainEffectsRegressor().fit(df_tr, spec.effect_cols, y_tr)
        resid = y_tr - me.predict(df_tr)
        model = fit_xgb(x_tr, resid, seed)
        return me.predict(df_te) + model.predict(x_te)

    if model_name == "multifp_hgb":
        model = fit_hgb(x_tr, y_tr, seed)
        return model.predict(x_te)

    model = fit_xgb(x_tr, y_tr, seed)
    return model.predict(x_te)


def iter_random_kfold(n: int, n_splits: int, seed: int) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, te) in enumerate(kf.split(np.arange(n)), start=1):
        yield f"fold{i}", tr, te


def iter_leave_one(df: pd.DataFrame, col: str) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
    values = df[col].astype(str)
    for g in sorted(values.unique()):
        te = np.where(values.to_numpy() == g)[0]
        tr = np.where(values.to_numpy() != g)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        yield str(g)[:48], tr, te


def iter_group_kfold(df: pd.DataFrame, col: str, n_splits: int, seed: int) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
    groups = df[col].astype(str).to_numpy()
    n_groups = len(set(groups))
    splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=splits)
    _ = seed
    for i, (tr, te) in enumerate(gkf.split(np.arange(len(df)), groups=groups), start=1):
        yield f"fold{i}", tr, te


def evaluate_split_protocol(
    spec: DatasetSpec,
    feat: MolFeaturizer,
    protocol: str,
    splitter: Iterable[Tuple[str, np.ndarray, np.ndarray]],
    model_names: Sequence[str],
    seed: int,
) -> Dict[str, object]:
    out: Dict[str, object] = {"protocol": protocol, "models": {}}
    fold_store = {m: [] for m in model_names}
    pred_store = {m: {"y_true": [], "y_pred": [], "fold": []} for m in model_names}
    t0 = time.time()
    n_folds = 0
    for fold_name, tr, te in splitter:
        n_folds += 1
        y_te = spec.y[te]
        for m in model_names:
            pred = predict_model(spec, tr, te, feat, m, seed + n_folds)
            met = metrics(y_te, pred)
            met["fold"] = fold_name
            met["n_test"] = int(len(te))
            fold_store[m].append(met)
            pred_store[m]["y_true"].extend(y_te.tolist())
            pred_store[m]["y_pred"].extend(np.clip(pred, 0, 100).tolist())
            pred_store[m]["fold"].extend([fold_name] * len(te))
        print(f"    [{spec.name}/{protocol}] {fold_name}  n_train={len(tr)} n_test={len(te)}", flush=True)
    for m in model_names:
        out["models"][m] = {
            "aggregate": aggregate_rows(fold_store[m]),
            "folds": fold_store[m],
        }
    out["n_folds"] = n_folds
    out["seconds"] = round(time.time() - t0, 1)
    # keep concatenated predictions only for the two models we plot
    keep = [m for m in ("onehot_xgb", "multifp_xgb", "residual_xgb") if m in pred_store]
    out["concat_preds"] = {m: pred_store[m] for m in keep}
    return out


# ---------------------------------------------------------------------------
# Light GNN comparison: ligand graph (old idea) vs additive graph (more unique mols)
# ---------------------------------------------------------------------------
def atom_features(atom: Chem.Atom) -> List[float]:
    hyb = atom.GetHybridization()
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 4.0,
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
        float(hyb == Chem.rdchem.HybridizationType.SP),
        float(hyb == Chem.rdchem.HybridizationType.SP2),
        float(hyb == Chem.rdchem.HybridizationType.SP3),
    ]


def mol_to_data(smiles: str, cond: np.ndarray, y: float | None) -> Data:
    mol = safe_mol(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        x = torch.zeros((1, 8), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
        edges = []
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges.extend([(u, v), (v, u)])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, cond=torch.tensor(cond, dtype=torch.float32))
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data


class CondGIN(nn.Module):
    def __init__(self, cond_dim: int, hidden: int = 64):
        super().__init__()

        def mlp():
            return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.embed = nn.Linear(8, hidden)
        self.conv1 = GINConv(mlp())
        self.conv2 = GINConv(mlp())
        self.head = nn.Sequential(
            nn.Linear(hidden + cond_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = torch.relu(self.embed(data.x))
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        g = global_add_pool(x, data.batch)
        cond = data.cond.view(g.size(0), -1)
        return self.head(torch.cat([g, cond], dim=1)).squeeze(-1)


def run_gnn_compare(spec: DatasetSpec, train_idx: np.ndarray, test_idx: np.ndarray, epochs: int = 12) -> Dict[str, object]:
    """Compare ligand-graph vs additive-graph GIN on the same split."""
    if spec.name != "buchwald_hartwig":
        return {}
    df_tr, df_te = spec.df.iloc[train_idx], spec.df.iloc[test_idx]
    cond_cols = ["base", "reaction"]
    maps = onehot_maps({c: df_tr[c] for c in cond_cols + ["ligand", "additive", "aryl halide"]})

    def cond_block(df: pd.DataFrame, graph_col: str) -> np.ndarray:
        # condition = everything except the molecule that is given as a graph
        use = [c for c in ["ligand", "additive", "base", "aryl halide", "reaction"] if c != graph_col]
        return apply_onehot({c: df[c] for c in use}, {c: maps[c] for c in use})

    def train_one(graph_col: str) -> np.ndarray:
        c_tr = cond_block(df_tr, graph_col)
        c_te = cond_block(df_te, graph_col)
        tr_data = [mol_to_data(str(df_tr.iloc[i][graph_col]), c_tr[i], float(spec.y[train_idx[i]])) for i in range(len(df_tr))]
        te_data = [mol_to_data(str(df_te.iloc[i][graph_col]), c_te[i], float(spec.y[test_idx[i]])) for i in range(len(df_te))]
        loader = PyGDataLoader(tr_data, batch_size=128, shuffle=True)
        te_loader = PyGDataLoader(te_data, batch_size=256, shuffle=False)
        model = CondGIN(cond_dim=c_tr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.SmoothL1Loss()
        model.train()
        for _ in range(epochs):
            for batch in loader:
                opt.zero_grad()
                pred = model(batch)
                loss = loss_fn(pred, batch.y.view(-1))
                loss.backward()
                opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in te_loader:
                preds.append(model(batch).numpy())
        return np.concatenate(preds)

    y_te = spec.y[test_idx]
    ligand_pred = train_one("ligand")
    additive_pred = train_one("additive")
    return {
        "ligand_gin": metrics(y_te, ligand_pred),
        "additive_gin": metrics(y_te, additive_pred),
        "n_unique_ligand": int(spec.df["ligand"].nunique()),
        "n_unique_additive": int(spec.df["additive"].nunique()),
        "epochs": epochs,
        "note": "Ligand GNN sees only 4 unique graphs; additive GNN sees 22 unique graphs.",
    }


# ---------------------------------------------------------------------------
# Conformal + BO
# ---------------------------------------------------------------------------
def run_conformal(spec: DatasetSpec, feat: MolFeaturizer, seed: int = 42) -> Dict[str, object]:
    n = len(spec.df)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr, n_ca = int(0.55 * n), int(0.20 * n)
    tr, ca, te = idx[:n_tr], idx[n_tr : n_tr + n_ca], idx[n_tr + n_ca :]
    x_tr, maps = make_features(spec, tr, feat, None, "multifp")
    x_ca, _ = make_features(spec, ca, feat, maps, "multifp")
    x_te, _ = make_features(spec, te, feat, maps, "multifp")
    model = fit_xgb(x_tr, spec.y[tr], seed)
    pred_ca = model.predict(x_ca)
    pred_te = model.predict(x_te)
    scores = np.abs(spec.y[ca] - pred_ca)
    rows = []
    for alpha, name in [(0.20, "80pct"), (0.10, "90pct")]:
        q_level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
        qhat = float(np.quantile(scores, q_level, method="higher"))
        cover = float(np.mean(np.abs(spec.y[te] - pred_te) <= qhat))
        rows.append({"interval": name, "alpha": alpha, "qhat": qhat, "coverage": cover, "mean_width": 2 * qhat})
    return {
        "n_train": int(len(tr)),
        "n_cal": int(len(ca)),
        "n_test": int(len(te)),
        "test_metrics": metrics(spec.y[te], pred_te),
        "intervals": rows,
        "method": "split conformal on |y - mu|, multifp_xgb",
    }


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.25) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    z = (mu - best - xi) / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (mu - best - xi) * cdf + sigma * pdf


def campaign_features(spec: DatasetSpec, feat: MolFeaturizer) -> np.ndarray:
    """Compact features for low-N BO: one-hot + descriptors only."""
    maps = onehot_maps({c: spec.df[c] for c in spec.onehot_all})
    oh = apply_onehot({c: spec.df[c] for c in spec.onehot_all}, maps)
    parts = [oh]
    for key in spec.fp_cols:
        parts.append(stack_mol(feat, smiles_series(spec, key).tolist(), "desc"))
    return np.concatenate(parts, axis=1)


def simulate_bo(
    spec: DatasetSpec,
    feat: MolFeaturizer,
    n_init: int,
    n_iter: int,
    n_seeds: int,
    subset_idx: np.ndarray | None = None,
) -> Dict[str, object]:
    if subset_idx is None:
        subset_idx = np.arange(len(spec.df))
    y = spec.y[subset_idx]
    x = campaign_features(spec, feat)[subset_idx]
    global_max = float(np.max(y))
    policies = ["random", "greedy", "ucb", "ei"]
    curves = {p: [] for p in policies}

    def acquire(policy: str, mu: np.ndarray, std: np.ndarray, best: float, remain_local: List[int], rng: np.random.Generator) -> int:
        if policy == "random":
            return int(rng.integers(0, len(remain_local)))
        if policy == "greedy":
            return int(np.argmax(mu))
        if policy == "ucb":
            return int(np.argmax(mu + 1.2 * std))
        return int(np.argmax(expected_improvement(mu, std, best=best)))

    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        init = rng.choice(len(subset_idx), size=n_init, replace=False)
        for policy in policies:
            observed = init.tolist()
            remain = [i for i in range(len(subset_idx)) if i not in set(observed)]
            best = float(np.max(y[observed]))
            hist = [best]
            for t in range(n_iter):
                if not remain:
                    break
                x_tr, y_tr = x[observed], y[observed]
                # ExtraTrees gives cheap tree-wise uncertainty and fits small N quickly.
                et = ExtraTreesRegressor(
                    n_estimators=80,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=seed + t,
                    n_jobs=2,
                )
                et.fit(x_tr, y_tr)
                tree_preds = np.stack([est.predict(x[remain]) for est in et.estimators_], axis=0)
                mu, std = tree_preds.mean(axis=0), tree_preds.std(axis=0)
                loc = acquire(policy, mu, std, best, remain, rng)
                chosen = remain.pop(loc)
                observed.append(chosen)
                best = max(best, float(y[chosen]))
                hist.append(best)
            curves[policy].append(hist)

    summary = {}
    for p, runs in curves.items():
        arr = np.array(runs, dtype=float)
        summary[p] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "final_mean": float(arr[:, -1].mean()),
            "final_std": float(arr[:, -1].std()),
            "evals_to_90pct_max_median": float(
                np.median([next((i for i, v in enumerate(r) if v >= 0.9 * global_max), len(r) - 1) for r in runs])
            ),
        }
    return {
        "n_pool": int(len(subset_idx)),
        "n_init": n_init,
        "n_iter": n_iter,
        "n_seeds": n_seeds,
        "global_max": global_max,
        "policies": summary,
    }


# ---------------------------------------------------------------------------
# Figures + HTML
# ---------------------------------------------------------------------------
def _bar_with_err(ax, labels, model_names, means, stds) -> None:
    x = np.arange(len(labels))
    width = 0.14
    offset = (len(model_names) - 1) / 2.0
    for i, m in enumerate(model_names):
        ax.bar(
            x + (i - offset) * width,
            [means[m][j] for j in range(len(labels))],
            width,
            yerr=[stds[m][j] for j in range(len(labels))],
            label=m,
            color=MODEL_COLORS.get(m, "0.5"),
            capsize=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("R²")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.legend(fontsize=8, ncol=2, frameon=False)


def plot_r2_panel(results: Dict[str, object], model_names: List[str], title: str, path: str, protocols: List[str]) -> None:
    labels, means, stds = [], {m: [] for m in model_names}, {m: [] for m in model_names}
    for proto in protocols:
        block = results.get(proto)
        if not block:
            continue
        labels.append(proto.replace("_", "\n"))
        for m in model_names:
            agg = block["models"][m]["aggregate"]
            means[m].append(agg["r2_mean"])
            stds[m].append(agg["r2_std"])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _bar_with_err(ax, labels, model_names, means, stds)
    ax.set_title(title)
    ax.set_ylim(min(-0.2, min(min(v) for v in means.values()) - 0.1), 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_scatter_compare(proto_block: Dict[str, object], path: str, title: str) -> None:
    preds = proto_block.get("concat_preds", {})
    names = [m for m in ("onehot_xgb", "multifp_xgb") if m in preds]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.8), squeeze=False)
    for ax, m in zip(axes[0], names):
        yt = np.array(preds[m]["y_true"])
        yp = np.array(preds[m]["y_pred"])
        ax.scatter(yt, yp, s=10, alpha=0.35, c=MODEL_COLORS.get(m, "0.3"))
        ax.plot([0, 100], [0, 100], "k--", lw=1)
        met = metrics(yt, yp)
        ax.set_title(f"{m}\nR²={met['r2']:.3f}  RMSE={met['rmse']:.1f}")
        ax.set_xlabel("True yield")
        ax.set_ylabel("Predicted yield")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bo(bo: Dict[str, object], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"random": "#7f8c8d", "greedy": "#4C78A8", "ucb": "#54A24B", "ei": "#F58518"}
    for p, col in colors.items():
        mean = np.array(bo["policies"][p]["mean"])
        std = np.array(bo["policies"][p]["std"])
        xs = np.arange(len(mean))
        ax.plot(xs, mean, label=p, color=col, lw=2)
        ax.fill_between(xs, mean - std, mean + std, color=col, alpha=0.15)
    ax.axhline(bo["global_max"], color="k", ls="--", lw=1, label="pool max")
    ax.set_xlabel("Evaluations after initialization (init = %d)" % bo["n_init"])
    ax.set_ylabel("Best observed yield")
    ax.set_title("Discrete BO on Buchwald–Hartwig reaction 0  (mean ± std over seeds)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_conformal(conf: Dict[str, object], path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    names = [r["interval"] for r in conf["intervals"]]
    cover = [r["coverage"] for r in conf["intervals"]]
    target = [1 - r["alpha"] for r in conf["intervals"]]
    x = np.arange(len(names))
    ax.bar(x - 0.18, target, 0.36, label="target", color="#9ecae1")
    ax.bar(x + 0.18, cover, 0.36, label="empirical coverage", color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage")
    ax.set_title("Split conformal calibration (multi-component XGB)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_html(summary: Dict[str, object], fig_dir: str, path: str) -> None:
    def tbl(ds_key: str) -> str:
        block = summary["datasets"].get(ds_key, {})
        protocols = [k for k in block.keys() if k not in {"meta"}]
        if not protocols:
            return "<p>No results.</p>"
        models = list(next(iter(block.values()))["models"].keys()) if protocols else []
        # protocols are stored under results
        return ""

    bh = summary["datasets"]["buchwald_hartwig"]
    suz = summary["datasets"].get("suzuki_miyaura", {})
    model_order = ["main_effects", "onehot_xgb", "ligandfp_xgb", "multifp_xgb", "residual_xgb", "multifp_hgb"]

    def rows_for(ds):
        html = []
        results = ds.get("results", {})
        html.append("<table><thead><tr><th>Split</th>")
        for m in model_order:
            html.append(f"<th>{m}<br/>R²</th>")
        html.append("</tr></thead><tbody>")
        for proto, block in results.items():
            html.append(f"<tr><td>{proto}</td>")
            for m in model_order:
                if m not in block["models"]:
                    html.append("<td>—</td>")
                    continue
                agg = block["models"][m]["aggregate"]
                html.append(f"<td>{agg['r2_mean']:.3f} ± {agg['r2_std']:.3f}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return "".join(html)

    gnn = summary.get("gnn_compare", {})
    conf = summary.get("conformal", {})
    bo = summary.get("bo", {})

    rel = os.path.relpath(fig_dir, os.path.dirname(path))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>HTE yield models — public-set scientific report</title>
<style>
 body {{ font-family: "IBM Plex Sans", "Segoe UI", sans-serif; margin: 28px auto; max-width: 1040px; color: #1f2a33; }}
 h1,h2,h3 {{ color: #12202b; }}
 .note {{ background: #f4f7fa; padding: 12px 16px; border-left: 4px solid #2171b5; }}
 table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
 th,td {{ border: 1px solid #d5dde3; padding: 6px 8px; text-align: center; }}
 th {{ background: #eef3f7; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; height: auto; margin: 8px 0 22px; border: 1px solid #e4e9ee; }}
 code {{ background: #f2f4f6; padding: 1px 4px; }}
</style>
</head>
<body>
<h1>Public HTE yield modeling — results for discussion</h1>
<p class="note">
These numbers are from public sets only (Doyle Buchwald–Hartwig and Perera Suzuki–Miyaura).
Random-split R² is interpolation. Leave-one-component-out is the chemically relevant test.
</p>
<h2>1. Buchwald–Hartwig (Doyle / Ahneman, n={bh.get("meta", {}).get("n", "?")})</h2>
<p>4 ligands, 22 additives, 3 bases, 15 aryl halides, 5 products. Previous work fingerprinted the ligand only
and mostly reported random splits. That overstates generalization: a new ligand is an unseen one-hot column.</p>
{rows_for(bh)}
<img src="{rel}/bh_r2_by_split.png" alt="BH R2 by split"/>
<img src="{rel}/bh_lolo_scatter.png" alt="BH leave-one-ligand scatter"/>
<img src="{rel}/bh_loao_scatter.png" alt="BH leave-one-additive scatter"/>
<h2>2. Why the old ligand GNN failed</h2>
<p>There are only 4 unique ligand graphs. A GNN cannot learn phosphine chemistry from 4 molecules.
Additives (22 unique) are the diverse organic component. Same split, same budget:</p>
<pre>{json.dumps(gnn, indent=2)}</pre>
<h2>3. Suzuki–Miyaura (Perera et al., n={suz.get("meta", {}).get("n", "?")})</h2>
<p>Second reaction class, 11 named ligands (plus a no-ligand control). Tests whether the same modeling
choices transfer when ligand names, not DFT descriptors, are all we have.</p>
{rows_for(suz)}
<img src="{rel}/suzuki_r2_by_split.png" alt="Suzuki R2 by split"/>
<h2>4. Uncertainty (split conformal)</h2>
<pre>{json.dumps(conf, indent=2)}</pre>
<img src="{rel}/conformal_coverage.png" alt="conformal coverage"/>
<h2>5. Discrete BO vs random / greedy</h2>
<p>Campaign on Buchwald–Hartwig reaction 0. ExtraTrees surrogate, 4 policies, multiple seeds.
This is a retrospective pool simulation, not a prospective wet-lab campaign.</p>
<pre>{json.dumps({k: bo.get(k) for k in ("n_pool", "n_init", "n_iter", "n_seeds", "global_max") if k in bo}, indent=2)}</pre>
<img src="{rel}/bo_campaign.png" alt="BO campaign"/>
<h2>6. What is actually new</h2>
<ul>
<li><b>multi-component fingerprints</b> — additive and aryl halide carry most of the structural diversity.</li>
<li><b>main-effects + residual XGB</b> — separates “this ligand is generally good” from interactions.</li>
<li><b>leave-one-component-out</b> — the number we should quote to chemists, not random-split R².</li>
<li><b>conformal intervals</b> — coverage is checked, not assumed from bootstrap std.</li>
</ul>
<p style="color:#667">Generated by <code>hte_scientific_benchmark.py</code>.</p>
</body></html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def dataset_meta(spec: DatasetSpec) -> Dict[str, object]:
    meta = {"n": int(len(spec.df)), "target": "yield", "effect_cols": spec.effect_cols}
    for c in spec.effect_cols:
        meta[f"n_{c}"] = int(spec.df[c].nunique())
    meta["yield_mean"] = float(np.mean(spec.y))
    meta["yield_std"] = float(np.std(spec.y))
    return meta


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    fig_dir = os.path.join(args.output_dir, "figures")
    ensure_dir(fig_dir)
    feat = MolFeaturizer(n_bits=args.fp_bits)

    kfolds = 2 if args.quick else args.kfolds
    models_full = ["main_effects", "onehot_xgb", "ligandfp_xgb", "multifp_xgb", "residual_xgb", "multifp_hgb"]
    models_ood = ["main_effects", "onehot_xgb", "ligandfp_xgb", "multifp_xgb", "residual_xgb"]
    if args.quick:
        models_full = ["main_effects", "onehot_xgb", "multifp_xgb"]
        models_ood = models_full

    summary: Dict[str, object] = {"seed": args.seed, "quick": bool(args.quick), "datasets": {}}

    # ---- Buchwald–Hartwig ----
    bh = load_buchwald(args.bh_path)
    print(f"BH n={len(bh.df)} ligands={bh.df['ligand'].nunique()} additives={bh.df['additive'].nunique()}", flush=True)
    bh_results = {}
    bh_results["random_kfold"] = evaluate_split_protocol(
        bh, feat, "random_kfold", iter_random_kfold(len(bh.df), kfolds, args.seed), models_full, args.seed
    )
    bh_results["leave_one_ligand"] = evaluate_split_protocol(
        bh, feat, "leave_one_ligand", iter_leave_one(bh.df, "ligand"), models_ood, args.seed
    )
    if not args.quick:
        bh_results["leave_one_reaction"] = evaluate_split_protocol(
            bh, feat, "leave_one_reaction", iter_leave_one(bh.df, "reaction"), models_ood, args.seed
        )
        bh_results["leave_one_base"] = evaluate_split_protocol(
            bh, feat, "leave_one_base", iter_leave_one(bh.df, "base"), models_ood, args.seed
        )
        bh_results["additive_groupkfold"] = evaluate_split_protocol(
            bh, feat, "additive_groupkfold", iter_group_kfold(bh.df, "additive", kfolds, args.seed), models_ood, args.seed
        )
        bh_results["aryl_groupkfold"] = evaluate_split_protocol(
            bh, feat, "aryl_groupkfold", iter_group_kfold(bh.df, "aryl halide", kfolds, args.seed), models_ood, args.seed
        )
        # True leave-one-additive for the two models that differ most (cheaper than all models × 22)
        bh_results["leave_one_additive"] = evaluate_split_protocol(
            bh,
            feat,
            "leave_one_additive",
            iter_leave_one(bh.df, "additive"),
            ["onehot_xgb", "ligandfp_xgb", "multifp_xgb"],
            args.seed,
        )
    summary["datasets"]["buchwald_hartwig"] = {"meta": dataset_meta(bh), "results": bh_results}

    # ---- Suzuki ----
    if not args.skip_suzuki:
        suz = load_suzuki(args.suzuki_path)
        print(f"Suzuki n={len(suz.df)} ligands={suz.df['ligand'].nunique()}", flush=True)
        suz_results = {}
        suz_results["random_kfold"] = evaluate_split_protocol(
            suz, feat, "random_kfold", iter_random_kfold(len(suz.df), kfolds, args.seed), models_full, args.seed
        )
        suz_results["leave_one_ligand"] = evaluate_split_protocol(
            suz, feat, "leave_one_ligand", iter_leave_one(suz.df, "ligand"), models_ood, args.seed
        )
        if not args.quick:
            suz_results["leave_one_solvent"] = evaluate_split_protocol(
                suz, feat, "leave_one_solvent", iter_leave_one(suz.df, "solvent"), models_ood, args.seed
            )
            suz_results["leave_one_reagent"] = evaluate_split_protocol(
                suz, feat, "leave_one_reagent", iter_leave_one(suz.df, "reagent"), models_ood, args.seed
            )
        summary["datasets"]["suzuki_miyaura"] = {"meta": dataset_meta(suz), "results": suz_results}

    # ---- GNN / conformal / BO ----
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(bh.df))
    n_te = int(0.2 * len(bh.df))
    te_idx, tr_idx = perm[:n_te], perm[n_te:]
    if not args.skip_gnn:
        print("GNN comparison...", flush=True)
        summary["gnn_compare"] = run_gnn_compare(bh, tr_idx, te_idx, epochs=8 if args.quick else 14)
    if not args.skip_conformal:
        print("Conformal...", flush=True)
        summary["conformal"] = run_conformal(bh, feat, seed=args.seed)
    if not args.skip_bo:
        print("BO campaigns...", flush=True)
        rxn0 = np.where(bh.df["reaction"].astype(str).to_numpy() == "0")[0]
        summary["bo"] = simulate_bo(
            bh,
            feat,
            n_init=12 if args.quick else 20,
            n_iter=8 if args.quick else 28,
            n_seeds=3 if args.quick else 8,
            subset_idx=rxn0,
        )

    # ---- figures ----
    model_plot = [m for m in models_ood if m in bh_results["random_kfold"]["models"]]
    plot_r2_panel(
        bh_results,
        model_plot,
        "Buchwald–Hartwig: R² under chemically meaningful splits",
        os.path.join(fig_dir, "bh_r2_by_split.png"),
        [p for p in ["random_kfold", "leave_one_ligand", "leave_one_reaction", "leave_one_base", "additive_groupkfold", "aryl_groupkfold"] if p in bh_results],
    )
    plot_scatter_compare(bh_results["leave_one_ligand"], os.path.join(fig_dir, "bh_lolo_scatter.png"), "Leave-one-ligand-out (concatenated folds)")
    if "leave_one_additive" in bh_results:
        plot_scatter_compare(bh_results["leave_one_additive"], os.path.join(fig_dir, "bh_loao_scatter.png"), "Leave-one-additive-out (concatenated folds)")
    if "suzuki_miyaura" in summary["datasets"]:
        suz_res = summary["datasets"]["suzuki_miyaura"]["results"]
        plot_r2_panel(
            suz_res,
            [m for m in model_plot if m in suz_res["random_kfold"]["models"]],
            "Suzuki–Miyaura (Perera): R² under chemically meaningful splits",
            os.path.join(fig_dir, "suzuki_r2_by_split.png"),
            [p for p in ["random_kfold", "leave_one_ligand", "leave_one_solvent", "leave_one_reagent"] if p in suz_res],
        )
    if "conformal" in summary:
        plot_conformal(summary["conformal"], os.path.join(fig_dir, "conformal_coverage.png"))
    if "bo" in summary:
        plot_bo(summary["bo"], os.path.join(fig_dir, "bo_campaign.png"))

    # persist (drop bulky concat_preds from json except we already used them)
    slim = json.loads(json.dumps(summary))
    for ds in slim["datasets"].values():
        for proto in ds.get("results", {}).values():
            proto.pop("concat_preds", None)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)
    write_html(slim, fig_dir, os.path.join(args.output_dir, "report.html"))
    print("Wrote", args.output_dir)
    print(json.dumps({k: {p: {m: b["models"][m]["aggregate"] for m in b["models"]} for p, b in v.get("results", {}).items()} for k, v in slim["datasets"].items()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Public-set scientific HTE yield benchmark")
    p.add_argument("--bh-path", default="data/bh-reactions.csv")
    p.add_argument("--suzuki-path", default="data/aap9112_Data_File_S1.xlsx")
    p.add_argument("--output-dir", default="hte_scientific_outputs")
    p.add_argument("--fp-bits", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-suzuki", action="store_true")
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument("--skip-bo", action="store_true")
    p.add_argument("--skip-conformal", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
