import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
BASE = ROOT_DIR / "data" / "processed" / "tabular" / "results"   # FT-Transformer result JSONs
OUT  = Path.cwd() / "tabular_figures"                             # figures are written here
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "Student":    "train_student_vs_test_student.json",
    "OSMI":       "train_osmi_vs_test_osmi.json",
    "Mod-Hetero": "train_mod_hetero_vs_test_mod_hetero.json",
    "Hetero":     "train_hetero_vs_test_hetero.json",
    "General":    "train_general_vs_test_general.json",
    "DASS":       "train_dass_vs_test_dass.json",
}

data = {}
for name, fname in FILES.items():
    with open(BASE / fname) as f:
        data[name] = json.load(f)

# ── Shared style ─────────────────────────────────────────────────────────────
BG       = "#0D0D1A"
GRID_COL = "#1E1E3A"
TXT      = "white"
PALETTE  = ["#4361EE", "#F72585", "#4CC9F0", "#7209B7", "#3A0CA3", "#F77F00"]
NAMES    = list(data.keys())
CMAP     = LinearSegmentedColormap.from_list("bp", [BG, "#4361EE", "#4CC9F0"])

def style_ax(ax, title=""):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.yaxis.label.set_color(TXT)
    ax.xaxis.label.set_color(TXT)
    if title:
        ax.set_title(title, color=TXT, fontsize=11, fontweight="bold", pad=8)

def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

def get(n, key):
    return data[n].get(key) or data[n]["run_info"].get(key)

records = {n: dict(
    acc     = data[n]["accuracy"],
    f1w     = data[n]["f1_weighted"],
    f1m     = data[n]["f1_macro"],
    roc     = data[n].get("roc_auc"),
    task    = data[n]["task_type"],
    train   = data[n]["run_info"]["train_rows"],
    test    = data[n]["run_info"]["test_rows"],
    feats   = data[n]["run_info"]["n_features"],
    params  = data[n]["run_info"]["model_params"],
    epochs  = data[n]["run_info"]["epochs_run"],
    val_loss= data[n]["run_info"]["best_val_loss"],
) for n in NAMES}

print("Generating figures…")

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Accuracy / F1 grouped bar
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
style_ax(ax, "Classification Performance: Accuracy · F1 Weighted · F1 Macro")

accs = [records[n]["acc"] for n in NAMES]
f1ws = [records[n]["f1w"] for n in NAMES]
f1ms = [records[n]["f1m"] for n in NAMES]
x, w = np.arange(len(NAMES)), 0.25

b1 = ax.bar(x - w,  accs, w, color=PALETTE[0], alpha=0.9, label="Accuracy")
b2 = ax.bar(x,      f1ws, w, color=PALETTE[1], alpha=0.9, label="F1 Weighted")
b3 = ax.bar(x + w,  f1ms, w, color=PALETTE[2], alpha=0.9, label="F1 Macro")
ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=10)
ax.set_ylim(0.6, 1.06)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.axhline(1.0, color=GRID_COL, lw=0.8, ls="--")
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color=TXT)
ax.legend(fontsize=9, facecolor=BG, labelcolor=TXT, edgecolor=GRID_COL)
ax.yaxis.grid(True, color=GRID_COL, lw=0.5); ax.set_axisbelow(True)
save(fig, "fig1_accuracy_f1.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — ROC-AUC (binary tasks)
# ════════════════════════════════════════════════════════════════════════════
roc_names = [n for n in NAMES if records[n]["roc"] is not None]
roc_vals  = [records[n]["roc"] for n in roc_names]
roc_cols  = [PALETTE[NAMES.index(n)] for n in roc_names]

fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
style_ax(ax, "ROC-AUC Score (Binary Classification Tasks)")
bars = ax.barh(roc_names, roc_vals, color=roc_cols, alpha=0.9, height=0.5)
ax.set_xlim(0.5, 1.05)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
for bar, v in zip(bars, roc_vals):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=10, color=TXT, fontweight="bold")
ax.xaxis.grid(True, color=GRID_COL, lw=0.5); ax.set_axisbelow(True)
save(fig, "fig2_roc_auc.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Confusion matrices (all 6, 2×3 grid)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=BG)
fig.suptitle("Confusion Matrices — All Datasets", color=TXT,
             fontsize=14, fontweight="bold", y=1.01)

for ax, n in zip(axes.flat, NAMES):
    d      = data[n]
    labels = d["confusion_matrix"]["labels"]
    mat    = np.array(d["confusion_matrix"]["matrix"])
    mat_n  = mat.astype(float) / mat.sum(axis=1, keepdims=True)
    style_ax(ax, n)
    ax.imshow(mat_n, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    short = [str(l)[:9] for l in labels]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=8 if len(labels) <= 3 else 7,
                    color="white" if mat_n[i, j] < 0.55 else "#0D0D1A",
                    fontweight="bold")

plt.tight_layout()
save(fig, "fig3_confusion_matrices.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Training & Test set sizes
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle("Dataset Sizes", color=TXT, fontsize=13, fontweight="bold")

train_sizes = [records[n]["train"] for n in NAMES]
test_sizes  = [records[n]["test"]  for n in NAMES]

for ax, sizes, label in zip(axes, [train_sizes, test_sizes], ["Training Rows", "Test Rows"]):
    style_ax(ax, label)
    bars = ax.bar(NAMES, sizes, color=PALETTE, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticklabels(NAMES, rotation=25, ha="right", fontsize=9)
    ax.yaxis.grid(True, color=GRID_COL, lw=0.5); ax.set_axisbelow(True)
    ax.yaxis.label.set_color(TXT)
    for bar, v in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.2,
                f"{v:,}", ha="center", va="bottom", fontsize=8, color=TXT)

plt.tight_layout()
save(fig, "fig4_dataset_sizes.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Features vs Model Parameters (scatter)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
style_ax(ax, "Number of Features vs Model Parameters")

feats  = [records[n]["feats"]  for n in NAMES]
params = [records[n]["params"] for n in NAMES]
for i, n in enumerate(NAMES):
    ax.scatter(feats[i], params[i], color=PALETTE[i], s=180, zorder=3,
               edgecolors="white", linewidths=0.5, label=n)
    ax.annotate(n, (feats[i], params[i]),
                textcoords="offset points", xytext=(8, 5),
                fontsize=9, color=PALETTE[i], fontweight="bold")
ax.set_xlabel("Number of Features", fontsize=10)
ax.set_ylabel("Model Parameters", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.2f}M"))
ax.xaxis.grid(True, color=GRID_COL, lw=0.5)
ax.yaxis.grid(True, color=GRID_COL, lw=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=9, facecolor=BG, labelcolor=TXT, edgecolor=GRID_COL)
save(fig, "fig5_features_vs_params.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — Best Validation Loss
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
style_ax(ax, "Best Validation Loss per Dataset")

val_losses = [records[n]["val_loss"] for n in NAMES]
bars = ax.barh(NAMES, val_losses, color=PALETTE, alpha=0.9, height=0.55)
ax.xaxis.grid(True, color=GRID_COL, lw=0.5); ax.set_axisbelow(True)
for bar, v in zip(bars, val_losses):
    ax.text(v + 0.004, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=10, color=TXT, fontweight="bold")
ax.set_xlabel("Validation Loss", fontsize=10)
save(fig, "fig6_val_loss.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 7 — Epochs run vs Val Loss (connected scatter)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
style_ax(ax, "Epochs Run vs Best Validation Loss")

epochs_run = [records[n]["epochs"] for n in NAMES]
for i, n in enumerate(NAMES):
    ax.scatter(epochs_run[i], val_losses[i], color=PALETTE[i], s=180, zorder=3,
               edgecolors="white", linewidths=0.5)
    ax.annotate(f"{n}\n({epochs_run[i]} ep, {val_losses[i]:.3f})",
                (epochs_run[i], val_losses[i]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8.5, color=PALETTE[i])
ax.set_xlabel("Epochs Run (early stopped at best val loss)", fontsize=10)
ax.set_ylabel("Best Validation Loss", fontsize=10)
ax.xaxis.grid(True, color=GRID_COL, lw=0.5)
ax.yaxis.grid(True, color=GRID_COL, lw=0.5)
ax.set_axisbelow(True)
save(fig, "fig7_epochs_vs_val_loss.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 8 — Per-class F1 heatmap (all datasets side by side)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(NAMES), figsize=(18, 5), facecolor=BG)
fig.suptitle("Per-Class F1 Score — All Datasets", color=TXT,
             fontsize=13, fontweight="bold")

for ax, n in zip(axes, NAMES):
    d       = data[n]
    pcm     = d["per_class_metrics"]
    classes = [k for k in pcm if k not in ("accuracy", "macro avg", "weighted avg")]
    prec    = [pcm[c]["precision"] for c in classes]
    rec     = [pcm[c]["recall"]    for c in classes]
    f1s     = [pcm[c]["f1-score"]  for c in classes]

    mat = np.array([prec, rec, f1s])   # shape (3, n_classes)
    style_ax(ax, n)
    im = ax.imshow(mat, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Precision", "Recall", "F1"], fontsize=8)
    short = [str(c)[:9] for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    for i in range(3):
        for j in range(len(classes)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v < 0.6 else "#0D0D1A",
                    fontweight="bold")

plt.tight_layout()
save(fig, "fig8_per_class_metrics.png")

print("\nAll 8 figures saved successfully.")