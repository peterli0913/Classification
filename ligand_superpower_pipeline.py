"""
Ligand feature analysis and yield-style label prediction pipeline.

This script implements:
1) XGBoost baseline on descriptor features
2) Descriptor-group attention regressor (interpretable group weights)
3) SMILES + descriptor hybrid regressor (sequence attention + descriptor branch)
4) 2D embedding visualization and model performance plots
5) Production-like prediction interface for new ligands

Superpower skill integrated:
- Source: https://github.com/obra/superpowers-skills (highest-star "superpower skill" match)
- Skill used: Meta-Pattern Recognition
- Practical usage here: modeling descriptors as grouped patterns (O-Q / R-T / U-W / X-AE),
  then learning group-level and sequence-level attention to uncover deeper patterns.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_ligand_table(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    return df.copy()


def build_feature_layout(df: pd.DataFrame) -> Tuple[List[str], str, List[List[int]]]:
    """
    Layout assumption based on user-provided Excel mapping:
    - J: smiles (index 9)
    - O-Q: descriptor group 1
    - R-T: descriptor group 2
    - U-W: descriptor group 3
    - X-AE: descriptor group 4
    - AF: label

    In zero-based indexing:
    - O is 14, Q is 16
    - R is 17, T is 19
    - U is 20, W is 22
    - X is 23, AE is 30
    - AF is 31
    """
    if df.shape[1] < 32:
        raise ValueError(f"Expected at least 32 columns (A-AF), got {df.shape[1]}")

    feature_cols = list(df.columns[14:31])  # O-AE, 17 features
    label_col = df.columns[31]  # AF
    group_indices = [
        [0, 1, 2],  # O-Q
        [3, 4, 5],  # R-T
        [6, 7, 8],  # U-W
        list(range(9, 17)),  # X-AE
    ]
    return feature_cols, label_col, group_indices


def safe_float_array(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    arr = df[cols].astype(float).to_numpy()
    if np.isnan(arr).any():
        col_means = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_means, inds[1])
    return arr


def build_smiles_vocab(smiles_list: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for s in smiles_list:
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            vocab[ch] = len(vocab)
    return vocab


def encode_smiles(smiles: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], List[int]]:
    ids = [vocab.get(ch, 1) for ch in smiles[:max_len]]
    mask = [1] * len(ids)
    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids.extend([0] * pad_len)
        mask.extend([0] * pad_len)
    return ids, mask


class DescriptorAttentionRegressor(nn.Module):
    def __init__(self, feature_dim: int, group_indices: List[List[int]], hidden_dim: int = 64) -> None:
        super().__init__()
        self.group_indices = group_indices
        self.group_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(len(idx), hidden_dim),
                    nn.ReLU(),
                )
                for idx in group_indices
            ]
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.group_weight_head = nn.Linear(hidden_dim, 1)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = []
        for idx, encoder in zip(self.group_indices, self.group_encoders):
            tokens.append(encoder(x[:, idx]))
        token_tensor = torch.stack(tokens, dim=1)  # [B, groups, H]
        attn_out, _ = self.attn(token_tensor, token_tensor, token_tensor)
        token_tensor = self.norm1(token_tensor + attn_out)
        token_tensor = self.norm2(token_tensor + self.ffn(token_tensor))

        group_logits = self.group_weight_head(token_tensor).squeeze(-1)  # [B, groups]
        group_weights = torch.softmax(group_logits, dim=1)
        pooled = (token_tensor * group_weights.unsqueeze(-1)).sum(dim=1)
        pred = self.reg_head(pooled).squeeze(-1)
        return {"pred": pred, "group_weights": group_weights, "embedding": pooled}


class SmilesDescriptorDataset(Dataset):
    def __init__(
        self,
        smiles: List[str],
        descriptor_x: np.ndarray,
        y: np.ndarray,
        vocab: Dict[str, int],
        max_len: int,
    ) -> None:
        self.smiles = smiles
        self.descriptor_x = descriptor_x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids, mask = encode_smiles(self.smiles[idx], self.vocab, self.max_len)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "desc": torch.tensor(self.descriptor_x[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


class SmilesDescriptorRegressor(nn.Module):
    def __init__(self, vocab_size: int, descriptor_dim: int, embed_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.seq_attn = nn.Linear(hidden_dim * 2, 1)
        self.desc_branch = nn.Sequential(
            nn.Linear(descriptor_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, desc: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.embedding(ids)
        seq_out, _ = self.gru(emb)
        attn_logits = self.seq_attn(seq_out).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn_logits, dim=1)
        seq_feat = (seq_out * attn.unsqueeze(-1)).sum(dim=1)
        desc_feat = self.desc_branch(desc)
        fused = torch.cat([seq_feat, desc_feat], dim=1)
        pred = self.reg_head(fused).squeeze(-1)
        return {"pred": pred, "embedding": fused, "seq_attention": attn}


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model


def train_descriptor_attention(
    model: DescriptorAttentionRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 250,
    lr: float = 1e-3,
    patience: int = 35,
) -> DescriptorAttentionRegressor:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    best_rmse = float("inf")
    best_state = None
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out["pred"], yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                preds.append(out["pred"].cpu().numpy())
                trues.append(yb.cpu().numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        val_rmse = rmse(true, pred)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_smiles_hybrid(
    model: SmilesDescriptorRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 40,
) -> SmilesDescriptorRegressor:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_rmse = float("inf")
    best_state = None
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            desc = batch["desc"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            out = model(ids, mask, desc)
            loss = criterion(out["pred"], y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["ids"].to(device)
                mask = batch["mask"].to(device)
                desc = batch["desc"].to(device)
                y = batch["y"].to(device)
                out = model(ids, mask, desc)
                preds.append(out["pred"].cpu().numpy())
                trues.append(y.cpu().numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        val_rmse = rmse(true, pred)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def infer_descriptor_attention(
    model: DescriptorAttentionRegressor,
    x: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    xb = torch.tensor(x, dtype=torch.float32).to(device)
    out = model(xb)
    pred = out["pred"].cpu().numpy()
    group_weights = out["group_weights"].cpu().numpy()
    embedding = out["embedding"].cpu().numpy()
    return pred, group_weights, embedding


@torch.no_grad()
def infer_smiles_hybrid(
    model: SmilesDescriptorRegressor,
    smiles: List[str],
    desc: np.ndarray,
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_ids, all_mask = [], []
    for s in smiles:
        ids, mask = encode_smiles(s, vocab, max_len)
        all_ids.append(ids)
        all_mask.append(mask)
    ids_t = torch.tensor(all_ids, dtype=torch.long).to(device)
    mask_t = torch.tensor(all_mask, dtype=torch.float32).to(device)
    desc_t = torch.tensor(desc, dtype=torch.float32).to(device)
    out = model(ids_t, mask_t, desc_t)
    pred = out["pred"].cpu().numpy()
    emb = out["embedding"].cpu().numpy()
    return pred, emb


def fit_tsne(embedding: np.ndarray, seed: int) -> np.ndarray:
    n = embedding.shape[0]
    if n < 10:
        return embedding[:, :2]
    perplexity = min(30, max(5, (n - 1) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")
    return tsne.fit_transform(embedding)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_visualizations(
    output_dir: str,
    y_true: np.ndarray,
    pred_dict: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    xgb_feature_names: List[str],
    xgb_feature_importance: np.ndarray,
    attn_group_weights: np.ndarray,
    attn_embedding: np.ndarray,
    seed: int,
    label_name: str,
) -> None:
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    # 1) Embedding projection: similar labels should cluster if representation is meaningful.
    coords = fit_tsne(attn_embedding, seed)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=y_true, cmap="viridis", s=35, alpha=0.85)
    plt.colorbar(sc, label=label_name)
    plt.title("Ligand 2D embedding (Descriptor Attention latent)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ligand_embedding_tsne.png"), dpi=180)
    plt.close()

    # 2) Actual vs predicted for all models.
    model_names = list(pred_dict.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        pred = pred_dict[name]
        ax.scatter(y_true, pred, alpha=0.8, s=30)
        lo = min(float(y_true.min()), float(pred.min()))
        hi = max(float(y_true.max()), float(pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
        ax.set_title(f"{name}\nRMSE={metrics[name]['rmse']:.4f}, R2={metrics[name]['r2']:.4f}")
        ax.set_xlabel(f"True {label_name}")
        ax.set_ylabel(f"Predicted {label_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "actual_vs_predicted.png"), dpi=180)
    plt.close()

    # 3) Performance bars.
    names = list(metrics.keys())
    rmse_vals = [metrics[n]["rmse"] for n in names]
    mae_vals = [metrics[n]["mae"] for n in names]
    r2_vals = [metrics[n]["r2"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(names, rmse_vals, color="#4C78A8")
    axes[0].set_title("RMSE (lower is better)")
    axes[1].bar(names, mae_vals, color="#F58518")
    axes[1].set_title("MAE (lower is better)")
    axes[2].bar(names, r2_vals, color="#54A24B")
    axes[2].set_title("R2 (higher is better)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_performance.png"), dpi=180)
    plt.close()

    # 4) XGBoost feature importance.
    order = np.argsort(xgb_feature_importance)[::-1]
    ordered_names = [xgb_feature_names[i] for i in order]
    ordered_imp = xgb_feature_importance[order]
    plt.figure(figsize=(8, 6))
    plt.barh(ordered_names[::-1], ordered_imp[::-1], color="#72B7B2")
    plt.title("XGBoost feature importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "xgboost_feature_importance.png"), dpi=180)
    plt.close()

    # 5) Attention group weight heatmap (test samples).
    group_labels = ["O-Q group", "R-T group", "U-W group", "X-AE group"]
    plt.figure(figsize=(10, 4))
    subset = attn_group_weights[: min(120, attn_group_weights.shape[0])]
    plt.imshow(subset.T, aspect="auto", cmap="magma")
    plt.colorbar(label="Attention weight")
    plt.yticks(range(len(group_labels)), group_labels)
    plt.xlabel("Sample index (test subset)")
    plt.title("Descriptor-group attention map")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "descriptor_group_attention_heatmap.png"), dpi=180)
    plt.close()


@dataclass
class TrainArtifacts:
    feature_cols: List[str]
    label_col: str
    scaler: StandardScaler
    vocab: Dict[str, int]
    max_smiles_len: int
    descriptor_group_indices: List[List[int]]


def save_artifacts(
    output_dir: str,
    xgb_model: xgb.XGBRegressor,
    desc_model: DescriptorAttentionRegressor,
    smiles_model: SmilesDescriptorRegressor,
    artifacts: TrainArtifacts,
) -> None:
    ensure_dir(output_dir)
    xgb_model.save_model(os.path.join(output_dir, "xgb_model.json"))
    torch.save(desc_model.state_dict(), os.path.join(output_dir, "descriptor_attention_model.pt"))
    torch.save(smiles_model.state_dict(), os.path.join(output_dir, "smiles_hybrid_model.pt"))
    joblib.dump(
        {
            "feature_cols": artifacts.feature_cols,
            "label_col": artifacts.label_col,
            "scaler": artifacts.scaler,
            "vocab": artifacts.vocab,
            "max_smiles_len": artifacts.max_smiles_len,
            "descriptor_group_indices": artifacts.descriptor_group_indices,
            "superpower_skill": {
                "repo": "https://github.com/obra/superpowers-skills",
                "skill": "Meta-Pattern Recognition",
            },
        },
        os.path.join(output_dir, "artifacts.joblib"),
    )


def train_pipeline(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_ligand_table(args.excel_path)
    feature_cols, label_col, group_indices = build_feature_layout(df)

    smiles_col = df.columns[9]  # J
    smiles_all = df[smiles_col].fillna("").astype(str).tolist()
    x_all = safe_float_array(df, feature_cols)
    y_all = df[label_col].astype(float).to_numpy()

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=args.val_size / (1 - args.test_size),
        random_state=args.seed,
    )

    x_train, x_val, x_test = x_all[train_idx], x_all[val_idx], x_all[test_idx]
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]
    smiles_train = [smiles_all[i] for i in train_idx]
    smiles_val = [smiles_all[i] for i in val_idx]
    smiles_test = [smiles_all[i] for i in test_idx]

    # -------- Model 1: XGBoost --------
    xgb_model = train_xgboost(x_train, y_train, x_val, y_val, args.seed)
    xgb_pred = xgb_model.predict(x_test)

    # Standardize descriptors for neural models.
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    # -------- Model 2: Descriptor Attention --------
    train_ds = TensorDataset(
        torch.tensor(x_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val_s, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    desc_model = DescriptorAttentionRegressor(feature_dim=x_train_s.shape[1], group_indices=group_indices).to(device)
    desc_model = train_descriptor_attention(
        desc_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.attn_epochs,
        lr=args.attn_lr,
        patience=40,
    )
    attn_pred, attn_group_weights, attn_embedding = infer_descriptor_attention(desc_model, x_test_s, device)

    # -------- Model 3: SMILES + Descriptor Hybrid --------
    vocab = build_smiles_vocab(smiles_train, min_freq=1)
    max_len = args.max_smiles_len
    smiles_train_ds = SmilesDescriptorDataset(smiles_train, x_train_s, y_train, vocab=vocab, max_len=max_len)
    smiles_val_ds = SmilesDescriptorDataset(smiles_val, x_val_s, y_val, vocab=vocab, max_len=max_len)
    smiles_train_loader = DataLoader(smiles_train_ds, batch_size=32, shuffle=True)
    smiles_val_loader = DataLoader(smiles_val_ds, batch_size=64, shuffle=False)

    smiles_model = SmilesDescriptorRegressor(vocab_size=len(vocab), descriptor_dim=x_train_s.shape[1]).to(device)
    smiles_model = train_smiles_hybrid(
        smiles_model,
        train_loader=smiles_train_loader,
        val_loader=smiles_val_loader,
        device=device,
        epochs=args.smiles_epochs,
        lr=args.smiles_lr,
        patience=45,
    )
    smiles_pred, _ = infer_smiles_hybrid(smiles_model, smiles_test, x_test_s, vocab, max_len, device)

    # Ensemble prediction.
    ensemble_pred = (xgb_pred + attn_pred + smiles_pred) / 3.0

    metrics = {
        "xgboost": evaluate_regression(y_test, xgb_pred),
        "descriptor_attention": evaluate_regression(y_test, attn_pred),
        "smiles_hybrid": evaluate_regression(y_test, smiles_pred),
        "ensemble_avg": evaluate_regression(y_test, ensemble_pred),
    }

    pred_dict = {
        "xgboost": xgb_pred,
        "descriptor_attention": attn_pred,
        "smiles_hybrid": smiles_pred,
        "ensemble_avg": ensemble_pred,
    }

    save_visualizations(
        output_dir=args.output_dir,
        y_true=y_test,
        pred_dict=pred_dict,
        metrics=metrics,
        xgb_feature_names=feature_cols,
        xgb_feature_importance=xgb_model.feature_importances_,
        attn_group_weights=attn_group_weights,
        attn_embedding=attn_embedding,
        seed=args.seed,
        label_name=label_col,
    )

    results_payload = {
        "dataset_size": int(len(df)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "feature_cols": feature_cols,
        "label_col": label_col,
        "metrics": metrics,
        "superpower_skill": {
            "repo": "https://github.com/obra/superpowers-skills",
            "skill": "Meta-Pattern Recognition",
            "how_used": "Descriptor groups and dual-attention encoding to identify deeper yield-related patterns.",
        },
    }

    with open(os.path.join(args.output_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    artifact_obj = TrainArtifacts(
        feature_cols=feature_cols,
        label_col=label_col,
        scaler=scaler,
        vocab=vocab,
        max_smiles_len=max_len,
        descriptor_group_indices=group_indices,
    )
    save_artifacts(args.output_dir, xgb_model, desc_model, smiles_model, artifact_obj)

    avg_group_weights = attn_group_weights.mean(axis=0).tolist()
    print("=== Training completed ===")
    print("Label column:", label_col)
    print("Metrics:", json.dumps(metrics, indent=2))
    print("Average descriptor-group attention weights [O-Q, R-T, U-W, X-AE]:", avg_group_weights)
    print("Artifacts saved to:", args.output_dir)


def predict_one(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_path = os.path.join(args.model_dir, "artifacts.joblib")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Missing artifacts: {artifact_path}")
    art = joblib.load(artifact_path)

    feature_cols: List[str] = art["feature_cols"]
    scaler: StandardScaler = art["scaler"]
    vocab: Dict[str, int] = art["vocab"]
    max_len: int = art["max_smiles_len"]
    group_indices: List[List[int]] = art["descriptor_group_indices"]

    values = [float(x.strip()) for x in args.descriptor_values.split(",") if x.strip()]
    if len(values) != len(feature_cols):
        raise ValueError(f"Expected {len(feature_cols)} descriptor values, got {len(values)}")
    x_raw = np.array(values, dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x_raw)
    smiles = args.smiles.strip()

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(args.model_dir, "xgb_model.json"))
    xgb_pred = float(xgb_model.predict(x_raw)[0])

    desc_model = DescriptorAttentionRegressor(feature_dim=len(feature_cols), group_indices=group_indices).to(device)
    desc_model.load_state_dict(torch.load(os.path.join(args.model_dir, "descriptor_attention_model.pt"), map_location=device))
    desc_pred, group_weights, _ = infer_descriptor_attention(desc_model, x_scaled, device)
    desc_pred = float(desc_pred[0])

    smiles_model = SmilesDescriptorRegressor(vocab_size=len(vocab), descriptor_dim=len(feature_cols)).to(device)
    smiles_model.load_state_dict(torch.load(os.path.join(args.model_dir, "smiles_hybrid_model.pt"), map_location=device))
    smiles_pred, _ = infer_smiles_hybrid(smiles_model, [smiles], x_scaled, vocab, max_len, device)
    smiles_pred = float(smiles_pred[0])

    ensemble_pred = float((xgb_pred + desc_pred + smiles_pred) / 3.0)
    out = {
        "prediction": {
            "xgboost": xgb_pred,
            "descriptor_attention": desc_pred,
            "smiles_hybrid": smiles_pred,
            "ensemble_avg": ensemble_pred,
        },
        "descriptor_group_attention": {
            "O-Q": float(group_weights[0][0]),
            "R-T": float(group_weights[0][1]),
            "U-W": float(group_weights[0][2]),
            "X-AE": float(group_weights[0][3]),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ligand feature modeling with XGBoost + Attention + SMILES hybrid")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train models and generate visualizations")
    train_p.add_argument("--excel-path", type=str, default="Kraken monophosphine coordinates AD Descriptors.xlsx")
    train_p.add_argument("--output-dir", type=str, default="ligand_outputs")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--test-size", type=float, default=0.2)
    train_p.add_argument("--val-size", type=float, default=0.2)
    train_p.add_argument("--attn-epochs", type=int, default=260)
    train_p.add_argument("--attn-lr", type=float, default=1e-3)
    train_p.add_argument("--smiles-epochs", type=int, default=320)
    train_p.add_argument("--smiles-lr", type=float, default=8e-4)
    train_p.add_argument("--max-smiles-len", type=int, default=120)

    pred_p = sub.add_parser("predict", help="Predict label for a new ligand")
    pred_p.add_argument("--model-dir", type=str, default="ligand_outputs")
    pred_p.add_argument("--smiles", type=str, required=True)
    pred_p.add_argument(
        "--descriptor-values",
        type=str,
        required=True,
        help="Comma-separated descriptor values aligned with O-AE feature order used in training.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_pipeline(args)
    elif args.command == "predict":
        predict_one(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
