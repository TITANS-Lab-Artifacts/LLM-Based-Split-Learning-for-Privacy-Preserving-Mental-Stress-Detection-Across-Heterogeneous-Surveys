import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_RESULTS = os.path.join(ROOT_DIR, "data", "processed", "text", "results")      # Zero_Shot.py / Few_Shot.py output
LORA_RESULTS = os.path.join(ROOT_DIR, "experiments", "results")                    # finetune_*.py output
SPLIT_RESULTS = os.path.join(ROOT_DIR, "experiments", "result_split")              # split-learning output
FT_RESULTS = os.path.join(ROOT_DIR, "data", "processed", "tabular", "results")     # FT-Transformer output

BG, FG, GRID = "#ffffff", "#000000", "#cccccc"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "font.size": 22, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATASETS = [("mh_general_dataset", "MHD"), ("mh_tech_dataset", "OST"),
            ("mh_dass_dataset", "D-42"), ("mh_student_dataset", "SD"),
            ("mod_hetero", "MHg"), ("hetero", "Hg")]
KEYS = [k for k, _ in DATASETS]
LABELS = [l for _, l in DATASETS]

MODEL = "meta_llama_Llama_3.2_3B_Instruct"
LORA_TAG = "llama"
LORA_DIRS = {
    "mh_student_dataset": "train_mh_student_dataset__test_mh_student_dataset",
    "mh_tech_dataset": "train_mh_tech_dataset__test_mh_tech_dataset",
    "mod_hetero": "train_mod_hetero__test_mod_hetero",
    "hetero": "train_hetero__test_hetero",
    "mh_general_dataset": "train_mh_general_dataset__test_mh_general_dataset",
    "mh_dass_dataset": "train_mh_dass_dataset__test_mh_dass_dataset",
}
GEN_FILE = {
    "mh_student_dataset": "test_mh_student_dataset.jsonl",
    "mh_tech_dataset": "test_mh_tech_dataset.jsonl",
    "mod_hetero": "test_mod_hetero.jsonl",
    "hetero": "test_hetero.jsonl",
    "mh_general_dataset": "test_mh_general_dataset.jsonl",
    "mh_dass_dataset": "test_mh_dass_dataset.jsonl",
}

def anls(path):
    return json.load(open(path))["metrics"]["anls"]

gen = {}
with open(os.path.join(SPLIT_RESULTS,
                       "result_split_generalization.csv"), newline="") as f:
    for row in csv.DictReader(f):
        gen[row["test_file"].strip()] = float(row["anls"])
general = np.array([gen[GEN_FILE[k]] for k in KEYS])

zero = np.array([anls(os.path.join(TEXT_RESULTS, f"zero_shot_{MODEL}_test_{k}.json")) for k in KEYS])
few = np.array([anls(os.path.join(TEXT_RESULTS, f"few_shot_10_{MODEL}_train_{k}_vs_test_{k}.json")) for k in KEYS])
lora = np.array([anls(os.path.join(LORA_RESULTS, LORA_DIRS[k], f"results.{LORA_TAG}.json")) for k in KEYS])

methods = ["Zero-Shot", "Few-Shot", "LoRA"]
gain = np.vstack([general - zero, general - few, general - lora]).T

vmax = np.max(np.abs(gain))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cmap = plt.get_cmap("RdYlGn")

fig, ax = plt.subplots(figsize=(8, 7.5))
im = ax.imshow(gain, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(np.arange(len(methods)))
ax.set_xticklabels(methods, fontsize=20, fontweight="bold")
ax.set_yticks(np.arange(len(LABELS)))
ax.set_yticklabels(LABELS, fontsize=20, fontweight="bold")
#ax.set_xlabel("Baseline", fontsize=26, labelpad=10, fontweight="bold")
#ax.set_ylabel("Test dataset", fontsize=26, labelpad=10, fontweight="bold")

for i in range(gain.shape[0]):
    for j in range(gain.shape[1]):
        ax.text(j, i, f"{gain[i, j]:+.3f}", ha="center", va="center",
                fontsize=20, color="#111111", fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("ANLS gain (generalist \u2212 baseline)", color=FG, fontsize=20, fontweight="bold")
cbar.ax.yaxis.set_tick_params(color=FG, labelsize=20)
plt.setp(cbar.ax.get_yticklabels(), color=FG, fontweight="bold")
cbar.outline.set_edgecolor(GRID)

for s in ax.spines.values():
    s.set_color(GRID)

plt.tight_layout()
plt.savefig("fig5_general_gain_heatmap.png", dpi=200)
plt.savefig("fig5_general_gain_heatmap.pdf")
print("saved fig5")