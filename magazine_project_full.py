"""
Complete project code for multimodal conflict detection on Magazine_final.csv.

What this code does
-------------------
1. Reads the user-provided CSV format:
   - parent_asin
   - title
   - original_description
   - original_features
   - image_url
   - image_variant
   - ai_generated_description
2. Builds a pseudo conflict label from the semantic similarity between
   the original text metadata and the AI-generated image description.
3. Splits the data into train / validation / test.
4. Supports two image loading modes:
   - url: read image directly from `image_url` during training/inference
   - local: download images to a folder first, then read from disk
5. Trains a PyTorch multimodal model with:
   - text encoder for metadata text
   - text encoder for ai_generated_description
   - image encoder for product image
   - conflict-aware gating module inspired by CMAG
6. Evaluates with accuracy / macro-F1 / weighted-F1 / confusion matrix.

Important note
--------------
This CSV does NOT contain an explicit rating or human conflict label.
Therefore this code uses pseudo labels for a conflict-detection task.
If later you add a true label column (for example `rating` or `label`),
you can replace `build_pseudo_labels` with your real supervised target.

Dependencies
------------
pip install pandas numpy scikit-learn pillow requests torch

Examples
--------
# 1) Direct URL image loading
python magazine_project_full.py \
    --csv-path Magazine_final.csv \
    --image-mode url \
    --epochs 5

# 2) First download images locally, then train from local folder
python magazine_project_full.py \
    --csv-path Magazine_final.csv \
    --download-images \
    --images-dir ./magazine_images \
    --image-mode local \
    --epochs 5

# 3) Notebook/local usage
from magazine_project_full import download_images_from_csv, run_pipeline

download_images_from_csv(
    csv_path="Magazine_final.csv",
    images_dir="./magazine_images",
)

run_pipeline(
    csv_path="Magazine_final.csv",
    image_mode="local",
    images_dir="./magazine_images",
    epochs=5,
)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def build_main_text(df: pd.DataFrame) -> pd.Series:
    cols = ["title", "original_description", "original_features"]
    texts = []
    for _, row in df.iterrows():
        parts = [clean_text(row.get(c, "")) for c in cols]
        text = " ".join([p for p in parts if p])
        texts.append(text)
    return pd.Series(texts)


def image_to_tensor(img: Image.Image, image_size: int = 128) -> torch.Tensor:
    img = img.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32)


def blank_image_tensor(image_size: int = 128) -> torch.Tensor:
    return torch.zeros((3, image_size, image_size), dtype=torch.float32)


def safe_filename(parent_asin: str, image_url: str) -> str:
    ext = os.path.splitext(image_url.split("?")[0])[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    url_hash = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:8]
    return f"{parent_asin}_{url_hash}{ext}"


# -----------------------------
# Pseudo-label generation
# -----------------------------

def build_pseudo_labels(df: pd.DataFrame, n_classes: int = 3) -> pd.DataFrame:
    """
    Build pseudo conflict labels from similarity between:
    - original text metadata: title + original_description + original_features
    - ai_generated_description

    We compute TF-IDF cosine similarity row-wise.
    Then we convert it into conflict labels.

    Label semantics:
    - 0: low conflict (high similarity)
    - 1: medium conflict
    - 2: high conflict (low similarity)
    """
    main_text = build_main_text(df).fillna("")
    ai_text = df["ai_generated_description"].fillna("").map(clean_text)

    vectorizer = TfidfVectorizer(max_features=15000, stop_words="english")
    stacked = list(main_text) + list(ai_text)
    X = vectorizer.fit_transform(stacked)
    A = X[: len(df)]
    B = X[len(df):]

    # cosine for paired rows using sparse elementwise multiplication
    sim = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    df = df.copy()
    df["similarity_score"] = sim

    if n_classes == 2:
        thr = np.quantile(sim, 0.5)
        # lower similarity => higher conflict
        df["conflict_label"] = (sim < thr).astype(int)
    else:
        q1 = np.quantile(sim, 1 / 3)
        q2 = np.quantile(sim, 2 / 3)
        labels = []
        for s in sim:
            if s >= q2:
                labels.append(0)  # low conflict
            elif s >= q1:
                labels.append(1)  # medium conflict
            else:
                labels.append(2)  # high conflict
        df["conflict_label"] = labels
    return df


# -----------------------------
# Train/Val/Test split
# -----------------------------

def make_splits(
    df: pd.DataFrame,
    output_dir: str,
    label_col: str = "conflict_label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=seed,
    )

    # val is taken from the remaining training part
    relative_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=relative_val,
        stratify=train_df[label_col],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = os.path.join(output_dir, "magazine_train.csv")
    val_path = os.path.join(output_dir, "magazine_val.csv")
    test_path = os.path.join(output_dir, "magazine_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, val_df, test_df


# -----------------------------
# Image download helper
# -----------------------------

def _download_one(row: pd.Series, images_dir: str, timeout: int = 20) -> Tuple[str, bool, str]:
    parent_asin = str(row["parent_asin"])
    url = str(row["image_url"])
    filename = safe_filename(parent_asin, url)
    filepath = os.path.join(images_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return filepath, True, "cached"

    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.save(filepath)
        return filepath, True, "downloaded"
    except Exception as e:
        return filepath, False, str(e)


def download_images_from_csv(
    csv_path: str,
    images_dir: str,
    num_workers: int = 8,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Download all images from the `image_url` column and save them to a folder.
    Returns a dataframe with an added `local_image_path` column.
    """
    os.makedirs(images_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.iloc[:limit].copy()
    df["local_image_path"] = ""
    df["download_ok"] = False
    df["download_status"] = ""

    futures = {}
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for idx, row in df.iterrows():
            futures[ex.submit(_download_one, row, images_dir)] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            path, ok, status = fut.result()
            df.at[idx, "local_image_path"] = path
            df.at[idx, "download_ok"] = ok
            df.at[idx, "download_status"] = status

    out_path = os.path.join(images_dir, "download_manifest.csv")
    df.to_csv(out_path, index=False)
    return df


# -----------------------------
# Tokenizer / Vocabulary
# -----------------------------

def simple_tokenize(text: str) -> List[str]:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return [t for t in text.split() if t]


def build_vocab(texts: List[str], min_freq: int = 2, max_vocab: int = 20000) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for text in texts:
        for tok in simple_tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1

    sorted_items = sorted(
        [(tok, c) for tok, c in freq.items() if c >= min_freq],
        key=lambda x: (-x[1], x[0]),
    )[:max_vocab]

    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _ in sorted_items:
        vocab[tok] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 128) -> List[int]:
    toks = simple_tokenize(text)[:max_len]
    return [vocab.get(t, 1) for t in toks]


# -----------------------------
# Dataset
# -----------------------------

class MagazineDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Dict[str, int],
        image_mode: str = "url",
        images_dir: Optional[str] = None,
        image_size: int = 128,
        timeout: int = 15,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.vocab = vocab
        self.image_mode = image_mode
        self.images_dir = images_dir
        self.image_size = image_size
        self.timeout = timeout
        self.main_texts = build_main_text(self.df).tolist()
        self.ai_texts = self.df["ai_generated_description"].fillna("").map(clean_text).tolist()
        self.labels = self.df["conflict_label"].astype(int).tolist()
        self._session = None

    def __len__(self) -> int:
        return len(self.df)

    def _get_session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _load_image_url(self, url: str) -> torch.Tensor:
        try:
            session = self._get_session()
            r = session.get(url, timeout=self.timeout)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            return image_to_tensor(img, self.image_size)
        except Exception:
            return blank_image_tensor(self.image_size)

    def _load_image_local(self, row: pd.Series) -> torch.Tensor:
        path = None
        if "local_image_path" in row and isinstance(row["local_image_path"], str) and row["local_image_path"]:
            path = row["local_image_path"]
        elif self.images_dir is not None:
            filename = safe_filename(str(row["parent_asin"]), str(row["image_url"]))
            path = os.path.join(self.images_dir, filename)

        try:
            if path and os.path.exists(path):
                img = Image.open(path).convert("RGB")
                return image_to_tensor(img, self.image_size)
        except Exception:
            pass
        return blank_image_tensor(self.image_size)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        main_ids = torch.tensor(encode_text(self.main_texts[idx], self.vocab), dtype=torch.long)
        ai_ids = torch.tensor(encode_text(self.ai_texts[idx], self.vocab), dtype=torch.long)

        if self.image_mode == "url":
            image = self._load_image_url(str(row["image_url"]))
        else:
            image = self._load_image_local(row)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "main_ids": main_ids,
            "ai_ids": ai_ids,
            "image": image,
            "label": label,
            "parent_asin": str(row["parent_asin"]),
        }


class Collator:
    def __call__(self, batch: List[Dict]):
        main_ids = [x["main_ids"] for x in batch]
        ai_ids = [x["ai_ids"] for x in batch]
        labels = torch.stack([x["label"] for x in batch], dim=0)
        images = torch.stack([x["image"] for x in batch], dim=0)

        main_lens = torch.tensor([len(x) for x in main_ids], dtype=torch.long)
        ai_lens = torch.tensor([len(x) for x in ai_ids], dtype=torch.long)
        main_pad = nn.utils.rnn.pad_sequence(main_ids, batch_first=True, padding_value=0)
        ai_pad = nn.utils.rnn.pad_sequence(ai_ids, batch_first=True, padding_value=0)

        return {
            "main_ids": main_pad,
            "main_lens": main_lens,
            "ai_ids": ai_pad,
            "ai_lens": ai_lens,
            "images": images,
            "labels": labels,
        }


# -----------------------------
# Model
# -----------------------------

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        # h shape: (2, B, hidden//2) -> concat directions
        h = h.permute(1, 0, 2).reshape(ids.size(0), -1)
        return h


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return self.fc(x)


class ConflictAwareGate(nn.Module):
    """
    Inspired by the proposal's CMAG idea:
    - cross-modal similarity
    - polarity/divergence proxy
    - dynamic fusion weights

    Here we use:
    - sim(text, ai_text): semantic consistency proxy
    - sim(text, image): learned alignment proxy
    - abs(text-ai_text) mean
    - abs(text-image) mean
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-8

    def _cos(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a / (a.norm(dim=1, keepdim=True) + self.eps)
        b = b / (b.norm(dim=1, keepdim=True) + self.eps)
        return (a * b).sum(dim=1, keepdim=True)

    def forward(self, text_feat: torch.Tensor, ai_feat: torch.Tensor, image_feat: torch.Tensor):
        sim_text_ai = self._cos(text_feat, ai_feat)
        sim_text_img = self._cos(text_feat, image_feat)
        div_text_ai = torch.abs(text_feat - ai_feat).mean(dim=1, keepdim=True)
        div_text_img = torch.abs(text_feat - image_feat).mean(dim=1, keepdim=True)

        gate_inp = torch.cat([sim_text_ai, sim_text_img, div_text_ai, div_text_img], dim=1)
        weights = self.softmax(self.mlp(gate_inp))
        w_text = weights[:, 0:1]
        w_img = weights[:, 1:2]
        fused = w_text * text_feat + w_img * image_feat
        return fused, w_text.squeeze(1), w_img.squeeze(1), gate_inp


class MultimodalConflictModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=128, hidden_dim=hidden_dim)
        self.ai_text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=128, hidden_dim=hidden_dim)
        self.image_encoder = ImageEncoder(out_dim=hidden_dim)
        self.gate = ConflictAwareGate(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        text_feat = self.text_encoder(batch["main_ids"], batch["main_lens"])
        ai_feat = self.ai_text_encoder(batch["ai_ids"], batch["ai_lens"])
        image_feat = self.image_encoder(batch["images"])
        fused, w_text, w_img, gate_stats = self.gate(text_feat, ai_feat, image_feat)
        logits = self.classifier(torch.cat([fused, ai_feat], dim=1))
        return {
            "logits": logits,
            "w_text": w_text,
            "w_img": w_img,
            "gate_stats": gate_stats,
        }


# -----------------------------
# Train / Evaluate
# -----------------------------

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    all_preds = []
    all_labels = []
    text_weights = []
    img_weights = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        preds = out["logits"].argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())
        text_weights.extend(out["w_text"].detach().cpu().tolist())
        img_weights.extend(out["w_img"].detach().cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "avg_text_weight": float(np.mean(text_weights)) if text_weights else None,
        "avg_img_weight": float(np.mean(img_weights)) if img_weights else None,
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out["logits"], batch["labels"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> Tuple[nn.Module, Dict[str, object]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    best_metrics = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


# -----------------------------
# End-to-end pipeline
# -----------------------------

def run_pipeline(
    csv_path: str,
    image_mode: str = "url",
    images_dir: Optional[str] = None,
    output_dir: str = "./project_outputs",
    epochs: int = 5,
    batch_size: int = 32,
    min_freq: int = 2,
    max_vocab: int = 20000,
    seed: int = 42,
    num_workers: int = 0,
) -> Dict[str, object]:
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = build_pseudo_labels(df, n_classes=3)

    train_df, val_df, test_df = make_splits(
        df,
        output_dir=output_dir,
        label_col="conflict_label",
        test_size=0.15,
        val_size=0.15,
        seed=seed,
    )

    # Build vocab on training text only
    train_texts = build_main_text(train_df).tolist() + train_df["ai_generated_description"].fillna("").map(clean_text).tolist()
    vocab = build_vocab(train_texts, min_freq=min_freq, max_vocab=max_vocab)

    train_set = MagazineDataset(train_df, vocab=vocab, image_mode=image_mode, images_dir=images_dir)
    val_set = MagazineDataset(val_df, vocab=vocab, image_mode=image_mode, images_dir=images_dir)
    test_set = MagazineDataset(test_df, vocab=vocab, image_mode=image_mode, images_dir=images_dir)

    collator = Collator()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalConflictModel(vocab_size=len(vocab), hidden_dim=256, num_classes=3).to(device)

    model, best_val_metrics = fit(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=1e-3,
    )

    test_metrics = evaluate(model, test_loader, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {
                "image_mode": image_mode,
                "images_dir": images_dir,
                "epochs": epochs,
                "batch_size": batch_size,
            },
        },
        os.path.join(output_dir, "best_model.pt"),
    )

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_f1": test_metrics["weighted_f1"],
                "avg_text_weight": test_metrics["avg_text_weight"],
                "avg_img_weight": test_metrics["avg_img_weight"],
                "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
                "classification_report": test_metrics["classification_report"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nBest validation metrics:")
    print(best_val_metrics["classification_report"])
    print("Average validation gate weights:", best_val_metrics["avg_text_weight"], best_val_metrics["avg_img_weight"])

    print("\nTest metrics:")
    print(test_metrics["classification_report"])
    print("Confusion matrix:\n", test_metrics["confusion_matrix"])
    print("Average test gate weights:", test_metrics["avg_text_weight"], test_metrics["avg_img_weight"])

    return {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "output_dir": output_dir,
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./project_outputs")
    p.add_argument("--image-mode", type=str, choices=["url", "local"], default="url")
    p.add_argument("--images-dir", type=str, default="./magazine_images")
    p.add_argument("--download-images", action="store_true")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.download_images:
        print("Downloading images...")
        download_images_from_csv(args.csv_path, args.images_dir)

    run_pipeline(
        csv_path=args.csv_path,
        image_mode=args.image_mode,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
