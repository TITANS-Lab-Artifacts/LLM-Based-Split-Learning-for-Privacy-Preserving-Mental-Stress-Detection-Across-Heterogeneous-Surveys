import json
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_RESULTS = os.path.join(ROOT_DIR, "data", "processed", "text", "results")      # Zero_Shot.py / Few_Shot.py output
LORA_RESULTS = os.path.join(ROOT_DIR, "experiments", "results")                    # finetune_*.py output
SPLIT_RESULTS = os.path.join(ROOT_DIR, "experiments", "result_split")              # split-learning output
FT_RESULTS = os.path.join(ROOT_DIR, "data", "processed", "tabular", "results")     # FT-Transformer output

BG, BLUE, PINK, CYAN, FG, GRID = "#ffffff", "#3d58d9", "#df237a", "#45b5da", "#000000", "#cccccc"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "font.size": 20, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    # --- Fix: embed fonts as TrueType (type 42) instead of Type 3 outlines ---
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

with open(os.path.join(FT_RESULTS, "batch_summary.json")) as f:
    summary = json.load(f)

DISPLAY = [("general", "MHD"), ("osmi", "OST"), ("dass", "D-42"),
           ("student", "SD"), ("mod_hetero", "MHg"), ("hetero", "Hg")]

by_name = {row["dataset"]: row for row in summary}
labels, acc, f1m, roc = [], [], [], []
for key, lab in DISPLAY:
    r = by_name[key]
    labels.append(lab)
    acc.append(r["accuracy"])
    f1m.append(r["f1_macro"])
    roc.append(r["roc_auc"] if r["roc_auc"] is not None else np.nan)

x = np.arange(len(labels))
w = 0.26
fig, ax = plt.subplots(figsize=(15.6, 7))

b1 = ax.bar(x - w, acc, w, label="Accuracy", color=BLUE, edgecolor=BG)
b2 = ax.bar(x, f1m, w, label="F1 Macro", color=PINK, edgecolor=BG)
b3 = ax.bar(x + w, np.nan_to_num(roc), w, label="ROC-AUC", color=CYAN, edgecolor=BG)

ymin = 0.60

def annotate(bars, values):
    for b, v in zip(bars, values):
        if np.isnan(v) or v == 0:
            continue
        cx = b.get_x() + b.get_width() / 2
        cy = ymin + (v - ymin) / 2
        ax.text(cx, cy, f"{v:.2f}", ha="center", va="center",
                fontsize=22, color="#ffffff", fontweight="bold", rotation=90)

annotate(b1, acc); annotate(b2, f1m); annotate(b3, roc)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=24, fontweight="bold")
ax.set_ylim(ymin, 1.04)
yt = np.arange(0.60, 1.01, 0.10)
ax.set_yticks(yt)
ax.set_yticklabels([f"{int(round(v*100))}%" for v in yt], fontsize=24, fontweight="bold")
ax.set_ylabel("Score", fontsize=24, fontweight="bold")

ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_color(GRID)

leg = ax.legend(loc="upper left", facecolor=BG, edgecolor=GRID, fontsize=24)
for t in leg.get_texts():
    t.set_color(FG)
    t.set_fontweight("bold")

plt.tight_layout()
plt.savefig("fig1_ft_performance.png", dpi=200)
plt.savefig("fig1_ft_performance.pdf")
print("saved fig1")