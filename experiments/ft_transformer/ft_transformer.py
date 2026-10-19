"""
ft_transformer.py
------------------
A self-contained Feature Tokenizer + Transformer (FT-Transformer) that
works with any of the prepared train/test CSV splits.

Architecture (Gorishniy et al., 2021, https://arxiv.org/abs/2106.11959):
  Numeric cols      -> per-feature linear projection -> token
  Categorical cols  -> per-feature embedding lookup  -> token
  [CLS] + tokens    -> Transformer Encoder -> CLS token -> head -> prediction

HOW TO USE:
  python ft_transformer.py --train train_student.csv --test test_student.csv
  python ft_transformer.py --train train_hetero.csv  --test test_hetero.csv --target treatment

REQUIREMENTS:
  pip install torch scikit-learn pandas numpy
"""

import argparse
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_TARGET  = None
EPOCHS          = 30
BATCH_SIZE      = 256
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
D_TOKEN         = 192
N_HEADS         = 8
N_LAYERS        = 3
FFN_FACTOR      = 4 / 3
DROPOUT         = 0.1
MAX_CARDINALITY = 50          # >this many uniques on a non-float col -> treat as numeric
PATIENCE        = 7
SEED            = 42
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Identity / admin columns never used as features OR targets (unless explicitly chosen).
DROP_COLS = {
    "survey_year", "id", "_row_id", "Timestamp", "timestamp", "index", "Unnamed: 0",
}

# `dataset_source` is admin too, but kept separately so the hetero files can fall back
# to predicting it. It is NEVER used as a feature.
ADMIN_NON_FEATURE = DROP_COLS | {"dataset_source"}

# Groups of mutually-leaking columns. If one is chosen as the target, the others in
# its group must be removed from the feature set (they encode the same outcome).
LEAKAGE_GROUPS = [
    {"depression_score", "depression_level"},
    {"anxiety_score",    "anxiety_level"},
    {"stress_score",     "stress_level"},
]


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class DataPreprocessor:
    """Automatic column-type detection, encoding, and scaling.
    Fit on train, transform both train and test."""

    def __init__(self, target_col, max_cardinality=MAX_CARDINALITY, exclude_cols=None):
        self.target_col      = target_col
        self.max_cardinality = max_cardinality
        self.exclude_cols    = set(exclude_cols or [])     # leakage + admin features
        self.num_cols        = []
        self.cat_cols        = []
        self.label_encoders  = {}
        self.cat_vocab_sizes = {}
        self.num_medians     = {}      # train medians, reused on test (no train/test leak)
        self.scaler          = StandardScaler()
        self.target_encoder  = LabelEncoder()
        self.target_median   = 0.0     # for regression NaN fill on test
        self.task_type       = None
        self.n_classes       = None

    # ---- column detection ------------------------------------------------------
    def _is_excluded(self, col):
        return (col == self.target_col
                or col in self.exclude_cols
                or col in ADMIN_NON_FEATURE)

    def _detect_columns(self, df):
        self.num_cols, self.cat_cols = [], []
        for col in df.columns:
            if self._is_excluded(col):
                continue
            ser = df[col]
            n_unique = ser.nunique(dropna=True)
            is_object = (ser.dtype == object) or isinstance(ser.dtype,
                                                             pd.CategoricalDtype)
            is_float = pd.api.types.is_float_dtype(ser)
            # Categorical if it's text/category, OR a low-cardinality non-float (coded ints).
            if is_object or (not is_float and n_unique <= self.max_cardinality):
                self.cat_cols.append(col)
            else:
                self.num_cols.append(col)

    # ---- target encoding -------------------------------------------------------
    def _infer_task_and_encode_target(self, df, fit):
        raw = df[self.target_col]
        n_unique = raw.nunique(dropna=True)

        # Decide task type (only on fit; reuse on transform)
        if fit:
            if n_unique == 2:
                self.task_type = "binary"
            elif n_unique <= 20 and not pd.api.types.is_float_dtype(raw):
                self.task_type = "multiclass"
            else:
                # try numeric -> regression, else treat as multiclass
                numeric = pd.to_numeric(raw, errors="coerce")
                if numeric.notna().mean() > 0.5 and n_unique > 20:
                    self.task_type = "regression"
                else:
                    self.task_type = "multiclass"

        if self.task_type == "regression":
            y = pd.to_numeric(raw, errors="coerce")
            if fit:
                self.target_median = float(y.median())
            y = y.fillna(self.target_median).values.astype(np.float32)
            if fit:
                self.n_classes = 1
            return y

        # classification: encode string labels
        y_str = raw.astype(str)
        if fit:
            self.target_encoder.fit(y_str)
            self.n_classes = len(self.target_encoder.classes_)
        known = set(self.target_encoder.classes_)
        y_safe = y_str.map(lambda v: v if v in known else self.target_encoder.classes_[0])
        y = self.target_encoder.transform(y_safe)
        if self.task_type == "binary":
            y = y.astype(np.float32)
        return y

    # ---- features --------------------------------------------------------------
    def _encode_categoricals(self, df, fit):
        out = []
        for col in self.cat_cols:
            filled = df[col].fillna("__MISSING__").astype(str)
            if fit:
                le = LabelEncoder()
                enc = le.fit_transform(filled)
                self.label_encoders[col] = le
                self.cat_vocab_sizes[col] = len(le.classes_)
            else:
                le = self.label_encoders[col]
                known = set(le.classes_)
                safe = filled.map(lambda v: v if v in known else le.classes_[0])
                enc = le.transform(safe)
            out.append(enc)
        if out:
            return np.stack(out, axis=1).astype(np.int64)
        return np.zeros((len(df), 0), dtype=np.int64)

    def _encode_numerics(self, df, fit):
        if not self.num_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        X = df[self.num_cols].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in self.num_cols:
            if fit:
                med = X[col].median()
                self.num_medians[col] = float(med) if not np.isnan(med) else 0.0
            X[col] = X[col].fillna(self.num_medians[col])
        if fit:
            arr = self.scaler.fit_transform(X)
        else:
            arr = self.scaler.transform(X)
        arr = np.clip(arr.astype(np.float32), -10.0, 10.0)
        return arr

    def fit_transform(self, df):
        self._detect_columns(df)
        y = self._infer_task_and_encode_target(df, fit=True)
        X_cat = self._encode_categoricals(df, fit=True)
        X_num = self._encode_numerics(df, fit=True)
        return X_num, X_cat, y

    def transform(self, df):
        y = self._infer_task_and_encode_target(df, fit=False)
        X_cat = self._encode_categoricals(df, fit=False)
        X_num = self._encode_numerics(df, fit=False)
        return X_num, X_cat, y


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """Each feature -> a d_token vector. Prepends a learnable [CLS] token."""

    def __init__(self, n_num, cat_vocab_sizes, d_token):
        super().__init__()
        self.d_token = d_token
        self.n_num   = n_num
        self.n_cat   = len(cat_vocab_sizes)

        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            self.num_bias   = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))
            nn.init.zeros_(self.num_bias)

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(v + 1, d_token) for v in cat_vocab_sizes   # +1 safety slot
        ])
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.01)

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.01)

    def forward(self, x_num, x_cat):
        # Determine batch size robustly from whichever input is populated.
        B = x_num.shape[0] if self.n_num > 0 else x_cat.shape[0]
        tokens = []
        if self.n_num > 0:
            x = x_num.unsqueeze(-1)                                   # (B, n_num, 1)
            tokens.append(x * self.num_weight.unsqueeze(0)
                          + self.num_bias.unsqueeze(0))               # (B, n_num, d)
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))              # (B, 1, d)
        feats = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, feats], dim=1)


class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_vocab_sizes, d_token=D_TOKEN, n_heads=N_HEADS,
                 n_layers=N_LAYERS, ffn_factor=FFN_FACTOR, dropout=DROPOUT, n_classes=2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_vocab_sizes, d_token)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=int(d_token * ffn_factor),
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             norm=nn.LayerNorm(d_token))
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, n_classes),
        )

    def forward(self, x_num, x_cat):
        tokens = self.tokenizer(x_num, x_cat)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]
        return self.head(cls_out)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(X_num, X_cat, y, batch_size, shuffle=True):
    t_num = torch.tensor(X_num, dtype=torch.float32)
    t_cat = torch.tensor(X_cat, dtype=torch.long)
    if y.dtype == np.float32:
        t_y = torch.tensor(y, dtype=torch.float32)
    else:
        t_y = torch.tensor(y.astype(np.int64), dtype=torch.long)
    ds = TensorDataset(t_num, t_cat, t_y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE == "cuda"))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, task_type):
    model.eval()
    total, n = 0.0, 0
    preds, probs, true = [], [], []
    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        logits = model(x_num, x_cat)
        if task_type == "regression":
            logits = logits.squeeze(1)
            loss = criterion(logits, y)
            preds.extend(logits.cpu().numpy())
            true.extend(y.cpu().numpy())
        else:
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)
                loss = criterion(logits, y.float())
                p = torch.sigmoid(logits)
                preds.extend((p >= 0.5).long().cpu().numpy())
                probs.extend(p.cpu().numpy())
            else:
                loss = criterion(logits, y)
                p = F.softmax(logits, dim=1)
                preds.extend(logits.argmax(1).cpu().numpy())
                probs.extend(p.cpu().numpy())
            true.extend(y.cpu().numpy())
        total += loss.item() * len(y)
        n += len(y)
    return (total / max(n, 1),
            np.array(preds),
            np.array(probs) if probs else None,
            np.array(true))


def print_performance(y_true, y_pred, y_probs, task_type, label_classes=None):
    import datetime
    sep = "-" * 55
    results = {"task_type": task_type,
               "timestamp": datetime.datetime.now().isoformat()}

    if task_type == "regression":
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        results.update({"rmse": round(rmse, 6), "mae": round(mae, 6),
                        "r2": round(r2, 6)})
        print(f"\n{'PERFORMANCE REPORT':^55}\n{sep}")
        print(f"  Task type   : Regression")
        print(f"  RMSE        : {rmse:.4f}")
        print(f"  MAE         : {mae:.4f}")
        print(f"  R2 Score    : {r2:.4f}\n{sep}")
        return results

    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results.update({"accuracy": round(float(acc), 6),
                    "f1_weighted": round(float(f1_w), 6),
                    "f1_macro": round(float(f1_m), 6)})

    print(f"\n{'PERFORMANCE REPORT':^55}\n{sep}")
    print(f"  Task type   : {task_type.capitalize()}")
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Weighted : {f1_w:.4f}")
    print(f"  F1 Macro    : {f1_m:.4f}")

    if task_type == "binary" and y_probs is not None:
        try:
            auc = roc_auc_score(y_true, y_probs)
            results["roc_auc"] = round(float(auc), 6)
            print(f"  ROC-AUC     : {auc:.4f}")
        except ValueError:
            pass

    names = [str(c) for c in label_classes] if label_classes is not None else None
    rep = classification_report(y_true, y_pred, target_names=names,
                                zero_division=0, output_dict=True)
    results["per_class_metrics"] = {
        k: ({mk: round(mv, 6) for mk, mv in v.items()} if isinstance(v, dict)
            else round(v, 6))
        for k, v in rep.items()
    }
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = {"labels": names or list(range(len(cm))),
                                   "matrix": cm.tolist()}

    print(sep + "\n  Classification Report:")
    for line in classification_report(y_true, y_pred, target_names=names,
                                      zero_division=0).splitlines():
        print("    " + line)
    print(sep)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TARGET SELECTION
# ══════════════════════════════════════════════════════════════════════════════

PREFERRED_TARGETS = [
    "Depression", "treatment", "sought_treatment",
    "stress_level", "depression_level", "anxiety_level",
    "care_options", "Growing_Stress", "Coping_Struggles",
    "Work_Interest", "Social_Weakness", "productivity_affected",
    "current_mh_disorder",
    "dataset_source",   # last-resort fallback for hetero mixes
]


def auto_detect_target(df):
    for col in PREFERRED_TARGETS:
        if col in df.columns and col != "dataset_source":
            # require it to have at least some non-missing labels
            if df[col].notna().mean() > 0.05:
                return col
    # heuristic: last short, dense, low-cardinality column
    for col in reversed(df.columns):
        if col in ADMIN_NON_FEATURE or len(col) > 80:
            continue
        if df[col].isna().mean() > 0.5:
            continue
        if df[col].nunique(dropna=True) > 20:
            continue
        return col
    if "dataset_source" in df.columns:
        print("  [INFO] Falling back to dataset_source as target.")
        return "dataset_source"
    return df.columns[0]


def leakage_excludes(target_col):
    """Columns dropped as features because they leak `target_col`.
    If the target is one of the DASS outcome columns, drop every other
    score/level column (they are alternate encodings of the same outcomes)."""
    all_outcome = set().union(*LEAKAGE_GROUPS)
    if target_col in all_outcome:
        return all_outcome - {target_col}
    return set()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(train_path, test_path, target_col=None, epochs=EPOCHS,
        batch_size=BATCH_SIZE, lr=LEARNING_RATE, d_token=D_TOKEN, n_layers=N_LAYERS):

    set_seed()
    print("\n" + "=" * 55)
    print("  FT-TRANSFORMER  -  Tabular Mental-Health Training")
    print("=" * 55)
    print(f"  Train : {os.path.basename(train_path)}")
    print(f"  Test  : {os.path.basename(test_path)}")
    print(f"  Device: {DEVICE.upper()}")

    # ---- load -----------------------------------------------------------------
    print("\n[1/5] Loading data ...")
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df  = pd.read_csv(test_path,  low_memory=False)

    # keep only columns present in both (so encoders match)
    common = [c for c in train_df.columns if c in test_df.columns]
    train_df, test_df = train_df[common], test_df[common]

    # ---- target ---------------------------------------------------------------
    if target_col is None:
        target_col = auto_detect_target(train_df)
        print(f"  Auto-detected target column: '{target_col}'")
    elif target_col not in train_df.columns:
        sys.exit(f"  ERROR: target '{target_col}' not found. "
                 f"Available: {train_df.columns.tolist()[:30]} ...")

    # CRITICAL FIX: drop rows whose target is missing (they corrupt training).
    n_before = len(train_df)
    train_df = train_df[train_df[target_col].notna()].reset_index(drop=True)
    test_df  = test_df[test_df[target_col].notna()].reset_index(drop=True)
    dropped = n_before - len(train_df)
    if dropped:
        print(f"  Dropped {dropped:,} train rows with missing target.")

    if len(train_df) == 0 or len(test_df) == 0:
        sys.exit("  ERROR: no rows left after dropping missing targets.")

    # CRITICAL FIX: exclude leakage siblings (e.g. stress_score when target=stress_level).
    exclude = leakage_excludes(target_col)
    if exclude:
        print(f"  Excluding leakage columns from features: {sorted(exclude)}")

    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test rows  : {len(test_df):,}")
    print(f"  Total cols : {len(train_df.columns)}")

    # ---- preprocess -----------------------------------------------------------
    print("\n[2/5] Preprocessing ...")
    prep = DataPreprocessor(target_col, exclude_cols=exclude)
    X_num_tr, X_cat_tr, y_tr = prep.fit_transform(train_df)
    X_num_te, X_cat_te, y_te = prep.transform(test_df)

    print(f"  Numeric features    : {len(prep.num_cols)}  {prep.num_cols[:5]}"
          f"{'...' if len(prep.num_cols) > 5 else ''}")
    print(f"  Categorical features: {len(prep.cat_cols)}  {prep.cat_cols[:5]}"
          f"{'...' if len(prep.cat_cols) > 5 else ''}")
    print(f"  Task type           : {prep.task_type}")
    if prep.task_type != "regression":
        print(f"  Classes ({prep.n_classes})         : {prep.target_encoder.classes_}")

    # ---- build model ----------------------------------------------------------
    print("\n[3/5] Building model ...")
    n_features = len(prep.num_cols) + len(prep.cat_cols)
    if n_features == 0:
        sys.exit("  ERROR: no usable features after preprocessing.")

    cat_vocab_sizes = [prep.cat_vocab_sizes[c] for c in prep.cat_cols]

    # ensure n_heads divides d_token
    n_heads = N_HEADS
    while d_token % n_heads != 0 and n_heads > 1:
        n_heads -= 1

    if prep.task_type == "regression":
        out_dim = 1
    elif prep.task_type == "binary":
        out_dim = 1
    else:
        out_dim = prep.n_classes

    model = FTTransformer(
        n_num=len(prep.num_cols), cat_vocab_sizes=cat_vocab_sizes,
        d_token=d_token, n_heads=n_heads, n_layers=n_layers,
        dropout=DROPOUT, n_classes=out_dim,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters  : {n_params:,}")
    print(f"  d_token     : {d_token}   n_heads: {n_heads}   n_layers: {n_layers}")

    # ---- loss -----------------------------------------------------------------
    if prep.task_type == "regression":
        criterion = nn.MSELoss()
    elif prep.task_type == "binary":
        pos_rate = float(np.mean(y_tr)) if len(y_tr) else 0.5
        pos_rate = min(max(pos_rate, 1e-6), 1 - 1e-6)
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate],
                                  dtype=torch.float32, device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        counts = np.bincount(y_tr.astype(int), minlength=prep.n_classes)
        w = 1.0 / (counts + 1e-6)
        w = w / w.sum() * prep.n_classes
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = make_loader(X_num_tr, X_cat_tr, y_tr, batch_size, shuffle=True)
    test_loader  = make_loader(X_num_te, X_cat_te, y_te, batch_size, shuffle=False)

    # ---- train ----------------------------------------------------------------
    print(f"\n[4/5] Training ({epochs} epochs, early stop patience={PATIENCE}) ...")
    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>10}  {'LR':>10}")
    print("  " + "-" * 42)

    best_val, best_state, patience = float("inf"), None, 0
    epoch = 0
    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl, _, _, _ = evaluate(model, test_loader, criterion, DEVICE, prep.task_type)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        improved = vl < best_val - 1e-4
        print(f"  {epoch:>5}  {tr:>11.4f}  {vl:>10.4f}  {lr_now:>10.2e}"
              f"{' *' if improved else ''}")
        if improved:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # ---- evaluate -------------------------------------------------------------
    print("\n[5/5] Evaluating on test set ...")
    _, y_pred, y_probs, y_true = evaluate(model, test_loader, criterion,
                                          DEVICE, prep.task_type)
    label_classes = (prep.target_encoder.classes_
                     if prep.task_type != "regression" else None)
    probs_report = y_probs if prep.task_type == "binary" else (
        y_probs if prep.task_type == "regression" else None)
    results = print_performance(y_true, y_pred, probs_report,
                                prep.task_type, label_classes)

    # ---- save -----------------------------------------------------------------
    train_stem = os.path.splitext(os.path.basename(train_path))[0]
    test_stem  = os.path.splitext(os.path.basename(test_path))[0]
    base_dir   = os.path.dirname(os.path.abspath(train_path))

    model_path = os.path.join(base_dir, f"ft_transformer_{train_stem}.pt")
    torch.save({
        "model_state": best_state or model.state_dict(),
        "config": {"n_num": len(prep.num_cols), "cat_vocab_sizes": cat_vocab_sizes,
                   "d_token": d_token, "n_heads": n_heads, "n_layers": n_layers,
                   "n_classes": out_dim, "task_type": prep.task_type,
                   "target_col": target_col},
    }, model_path)

    import json, datetime
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{train_stem}_vs_{test_stem}.json")
    results["run_info"] = {
        "train_file": os.path.basename(train_path),
        "test_file": os.path.basename(test_path),
        "target_column": target_col,
        "excluded_leakage_cols": sorted(exclude),
        "train_rows": len(train_df), "test_rows": len(test_df),
        "n_features": n_features,
        "numeric_features": prep.num_cols,
        "categorical_features": prep.cat_cols,
        "model_params": n_params, "epochs_run": epoch,
        "best_val_loss": round(best_val, 6),
        "config": {"d_token": d_token, "n_heads": n_heads, "n_layers": n_layers,
                   "batch_size": batch_size, "lr": lr, "epochs": epochs},
        "saved_at": datetime.datetime.now().isoformat(),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Model saved   -> {model_path}")
    print(f"  Results saved -> {result_path}")
    print("=" * 55 + "\n")
    return results


# ---- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FT-Transformer for tabular datasets")
    p.add_argument("--train",  required=True)
    p.add_argument("--test",   required=True)
    p.add_argument("--target", default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch",  type=int, default=BATCH_SIZE)
    p.add_argument("--lr",     type=float, default=LEARNING_RATE)
    p.add_argument("--d_token", type=int, default=D_TOKEN)
    p.add_argument("--layers", type=int, default=N_LAYERS)
    a = p.parse_args()

    run(train_path=a.train, test_path=a.test, target_col=a.target,
        epochs=a.epochs, batch_size=a.batch, lr=a.lr,
        d_token=a.d_token, n_layers=a.layers)
