import csv
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_RESULTS = os.path.join(ROOT_DIR, "data", "processed", "text", "results")      # Zero_Shot.py / Few_Shot.py output
LORA_RESULTS = os.path.join(ROOT_DIR, "experiments", "results")                    # finetune_*.py output
SPLIT_RESULTS = os.path.join(ROOT_DIR, "experiments", "result_split")              # split-learning output
FT_RESULTS = os.path.join(ROOT_DIR, "data", "processed", "tabular", "results")     # FT-Transformer output

BG, FG, GRID = "#ffffff", "#000000", "#cccccc"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "font.size": 20, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

path = os.path.join(SPLIT_RESULTS, "result_split_generalization.csv")
LABELS = {
    "test_mh_general_dataset.jsonl": "MHD", "test_mh_tech_dataset.jsonl": "OST",
    "test_mh_dass_dataset.jsonl": "D-42", "test_mh_student_dataset.jsonl": "SD",
    "test_mod_hetero.jsonl": "MHg", "test_hetero.jsonl": "Hg",
}
ORDER = ["MHD", "OST", "D-42", "SD", "MHg", "Hg"]

DS_COLORS = {"MHD": "#e07b39", "OST": "#2e8b57", "D-42": "#8e44ad", "SD": "#c0392b"}
STRIPE = {
    "MHg": ["#2e8b57", "#c0392b"],
    "Hg":  ["#e07b39", "#2e8b57", "#8e44ad", "#c0392b"],
}

vals = {}
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        lab = LABELS.get(row["test_file"].strip())
        if lab:
            vals[lab] = float(row["anls"])

labels = ORDER
anls = [vals[l] for l in labels]


def diagonal_stripe(ax, left, bottom, width, height, colors,
                    band_px=0.05, aspect=None):
    """Diagonal striped bar, clipped to its rectangle. `band_px` is the band
    width as a fraction of the bar width; stripes lean at ~45 deg visually."""
    clip = mpatches.Rectangle((left, bottom), width, height,
                              transform=ax.transData,
                              facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    k = len(colors)
    band = width * band_px
    # visual 45 deg: skew in x equals height scaled by the data->display aspect
    skew = height * aspect
    o = -skew - band
    idx = 0
    while o < width + skew + band:
        c = colors[idx % k]
        x0 = left + o
        verts = [(x0, bottom), (x0 + band, bottom),
                 (x0 + band + skew, bottom + height), (x0 + skew, bottom + height)]
        poly = mpatches.Polygon(verts, closed=True, facecolor=c,
                                edgecolor="none", zorder=3)
        ax.add_patch(poly)
        poly.set_clip_path(clip)
        o += band
        idx += 1
    ax.add_patch(mpatches.Rectangle((left, bottom), width, height,
                 facecolor="none", edgecolor=BG, linewidth=1.4, zorder=6))


x = np.arange(len(labels))
bw = 0.6
fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_xlim(-0.6, len(labels) - 0.4)
ax.set_ylim(0, max(anls) * 1.18)

# data->display aspect so 1 x-unit and 1 y-unit map appropriately for ~45deg
bbox = ax.get_window_extent()
xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
aspect = (xspan / bbox.width) / (yspan / bbox.height)

for xi, lab, v in zip(x, labels, anls):
    left = xi - bw / 2
    if lab in DS_COLORS:
        ax.add_patch(mpatches.Rectangle((left, 0), bw, v,
                     facecolor=DS_COLORS[lab], edgecolor=BG, linewidth=1.2, zorder=3))
    else:
        diagonal_stripe(ax, left, 0, bw, v, STRIPE[lab], band_px=0.11, aspect=aspect)
    ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
            fontsize=24, color=FG, fontweight="bold")

ax.set_ylabel("ANLS", fontsize=24, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=26, fontweight="bold")
ax.tick_params(axis="y", labelsize=26)
for t in ax.get_yticklabels():
    t.set_fontweight("bold")
ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_color(GRID)

plt.tight_layout()
plt.savefig("fig4_generalization.png", dpi=200)
plt.savefig("fig4_generalization.pdf")
print("saved fig4")