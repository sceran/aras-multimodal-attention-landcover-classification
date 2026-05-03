# =============================================================================
# 06_analysis_figures.py — Production figures for the Phase-2 report
# =============================================================================
# Loads cached JSON results and produces 4 publication-quality figures:
#
#   fig1_fusion_heatmap.png      — Path A: fusion (rows) × caption (cols) mAP
#   fig2_ca_vs_late_delta.png    — CA-vs-Late Δ across captions, both tasks
#   fig3_backbone_compare.png    — Vanilla CLIP vs RemoteCLIP per fusion
#   fig4_perclass_minor.png      — Per-class minor class F1/IoU gain
#
# Plus, the segmentation-related figures already saved by 05_pathB:
#   fig_seg_qualitative.png
#   fig_seg_attention_overlay.png
#
# Wall-clock: ~30 sec total.
# =============================================================================


# %% CELL 1 — Load all cached results
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use a clean serif font that works well in IEEE LaTeX context.
mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FEAT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/DI725/phase2_features")
FIG_DIR  = FEAT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Tree", "Shrub", "Grass", "Crop", "Built-up", "Barren", "Water"]
CAPTION_COLS = ["hybrid_gemma3-4b", "hybrid_qwen3-vl-8b", "text_qwen3-4b",
                "vision_gemma3-4b", "vision_qwen3-vl-8b"]
CAPTION_SHORT = {  # display labels
    "hybrid_gemma3-4b":   "hyb-gem",
    "hybrid_qwen3-vl-8b": "hyb-qwn",
    "text_qwen3-4b":      "txt-qwn",
    "vision_gemma3-4b":   "vis-gem",
    "vision_qwen3-vl-8b": "vis-qwn",
}

with open(FEAT_DIR / "pathA_results_multi_seed.json") as f:
    pathA = json.load(f)
with open(FEAT_DIR / "pathA_backbone_ablation.json") as f:
    backbone = json.load(f)
with open(FEAT_DIR / "pathA_fusion_variants.json") as f:
    fusion_variants = json.load(f)
with open(FEAT_DIR / "pathB_seg_results.json") as f:
    pathB = json.load(f)

print("Loaded:")
print(f"  pathA       — {len(pathA)} keys")
print(f"  backbone    — vanilla:{len(backbone['vanilla'])}, remoteclip:{len(backbone['remoteclip'])}")
print(f"  fusion_vars — {len(fusion_variants)} keys")
print(f"  pathB       — {len(pathB)} keys")


# %% CELL 2 — Figure 1: fusion × caption mAP heatmap (Path A)
fusion_order = ["late", "film", "gated", "cross_attn"]
fusion_label = {"late": "Late", "film": "FiLM", "gated": "Gated", "cross_attn": "Cross-Attn"}

H = np.zeros((len(fusion_order), len(CAPTION_COLS)))
for i, f in enumerate(fusion_order):
    for j, c in enumerate(CAPTION_COLS):
        H[i, j] = fusion_variants[f"{f}__{c}"]["mAP_mean"]

fig, ax = plt.subplots(figsize=(7.0, 3.0))
im = ax.imshow(H, cmap="viridis", aspect="auto", vmin=0.80, vmax=0.95)

ax.set_xticks(range(len(CAPTION_COLS)))
ax.set_xticklabels([CAPTION_SHORT[c] for c in CAPTION_COLS], rotation=0)
ax.set_yticks(range(len(fusion_order)))
ax.set_yticklabels([fusion_label[f] for f in fusion_order])
ax.set_xlabel("Caption strategy")
ax.set_ylabel("Fusion")

# Annotate each cell with the mAP value
for i in range(len(fusion_order)):
    for j in range(len(CAPTION_COLS)):
        v = H[i, j]
        color = "white" if v < 0.88 else "black"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=color, fontsize=9)

# Image-only baseline as reference annotation
img_baseline = pathA["image_only"]["mAP_mean"]
ax.set_title(f"Path A multi-label mAP (3-seed mean) — image-only baseline: {img_baseline:.3f}",
             fontsize=10)
fig.colorbar(im, ax=ax, label="mAP", shrink=0.8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_fusion_heatmap.png")
plt.show()
print(f"Saved: {FIG_DIR/'fig1_fusion_heatmap.png'}")


# %% CELL 3 — Figure 2: CA vs Late Δ (Path A mAP + Path B mIoU)

# Path A: 5 captions, Δ in mAP
pathA_deltas = []
pathA_late_stds = []
pathA_ca_stds = []
for cap in CAPTION_COLS:
    L = pathA[f"late__{cap}"]
    C = pathA[f"cross_attn__{cap}"]
    pathA_deltas.append(C["mAP_mean"] - L["mAP_mean"])
    pathA_late_stds.append(L["mAP_std"])
    pathA_ca_stds.append(C["mAP_std"])

# Path B: 3 captions (subset that we ran)
SEG_CAPTIONS = ["hybrid_qwen3-vl-8b", "text_qwen3-4b", "vision_qwen3-vl-8b"]
pathB_deltas = []
for cap in SEG_CAPTIONS:
    L = pathB[f"seg_late__{cap}"]
    C = pathB[f"seg_cross_attn__{cap}"]
    pathB_deltas.append(C["mIoU_mean"] - L["mIoU_mean"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6), sharey=False)

x1 = np.arange(len(CAPTION_COLS))
ax1.bar(x1, pathA_deltas, color="#2c7fb8", alpha=0.85)
ax1.axhline(0, color="black", linewidth=0.6)
ax1.set_xticks(x1)
ax1.set_xticklabels([CAPTION_SHORT[c] for c in CAPTION_COLS], rotation=0)
ax1.set_ylabel(r"$\Delta$ mAP (CA $-$ Late)")
ax1.set_title("Path A: multi-label classification")
ax1.grid(axis="y", linestyle="--", alpha=0.3)
for xi, d in zip(x1, pathA_deltas):
    ax1.text(xi, d + 0.001, f"+{d:.3f}", ha="center", va="bottom", fontsize=8)

x2 = np.arange(len(SEG_CAPTIONS))
ax2.bar(x2, pathB_deltas, color="#d95f0e", alpha=0.85)
ax2.axhline(0, color="black", linewidth=0.6)
ax2.set_xticks(x2)
ax2.set_xticklabels([CAPTION_SHORT[c] for c in SEG_CAPTIONS], rotation=0)
ax2.set_ylabel(r"$\Delta$ mIoU (CA $-$ Late)")
ax2.set_title("Path B: segmentation (preliminary)")
ax2.grid(axis="y", linestyle="--", alpha=0.3)
for xi, d in zip(x2, pathB_deltas):
    ax2.text(xi, d + 0.001, f"+{d:.3f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Cross-attention consistently beats late fusion across captions and tasks",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_ca_vs_late_delta.png")
plt.show()
print(f"Saved: {FIG_DIR/'fig2_ca_vs_late_delta.png'}")


# %% CELL 4 — Figure 3: Vanilla CLIP vs RemoteCLIP (per fusion mode)

bb_v = backbone["vanilla"]
bb_r = backbone["remoteclip"]

fusion_groups = {
    "image-only": ["image_only"],
    "Late":       [f"late__{c}" for c in CAPTION_COLS],
    "Cross-Attn": [f"cross_attn__{c}" for c in CAPTION_COLS],
}

# Compute means and stds per group
group_names, vanilla_means, vanilla_stds, rc_means, rc_stds = [], [], [], [], []
for gname, keys in fusion_groups.items():
    v_means = [bb_v[k]["mAP_mean"] for k in keys]
    r_means = [bb_r[k]["mAP_mean"] for k in keys]
    group_names.append(gname)
    vanilla_means.append(float(np.mean(v_means)))
    vanilla_stds.append(float(np.std(v_means)))
    rc_means.append(float(np.mean(r_means)))
    rc_stds.append(float(np.std(r_means)))

x = np.arange(len(group_names))
w = 0.35
fig, ax = plt.subplots(figsize=(5.5, 2.8))
b1 = ax.bar(x - w/2, vanilla_means, w, yerr=vanilla_stds,
            label="Vanilla CLIP", color="#7f7f7f", capsize=3,
            hatch='////', edgecolor='black', linewidth=0.6)
b2 = ax.bar(x + w/2, rc_means, w, yerr=rc_stds,
            label="RemoteCLIP", color="#2ca02c", capsize=3,
            hatch='\\\\\\\\', edgecolor='black', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(group_names)
ax.set_ylabel("mAP (mean over caption strategies)")
ax.set_ylim(0.65, 0.97)
ax.set_title("Backbone ablation: RemoteCLIP wins on every fusion mode")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(loc="lower right")
for bb in (b1, b2):
    for rect in bb:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_backbone_compare.png")
plt.show()
print(f"Saved: {FIG_DIR/'fig3_backbone_compare.png'}")


# %% CELL 5 — Figure 4: Per-class minor-class gains (Path A F1 + Path B IoU)

# Best CA condition per task
best_ca_pathA = "cross_attn__hybrid_qwen3-vl-8b"
best_ca_pathB = "seg_cross_attn__hybrid_qwen3-vl-8b"

img_pathA = pathA["image_only"]["f1_per_class_mean"]
ca_pathA  = pathA[best_ca_pathA]["f1_per_class_mean"]

img_pathB = pathB["seg_image_only"]["iou_per_class_mean"]
ca_pathB  = pathB[best_ca_pathB]["iou_per_class_mean"]

x = np.arange(len(CLASSES))
w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.8), sharey=False)

# Path A F1
ax1.bar(x - w/2, img_pathA, w, label="image-only", color="#bdbdbd",
        hatch='////', edgecolor='black', linewidth=0.5)
ax1.bar(x + w/2, ca_pathA,  w, label="best CA (hyb-qwn)", color="#2c7fb8",
        hatch='\\\\\\\\', edgecolor='black', linewidth=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(CLASSES, rotation=30, ha="right")
ax1.set_ylabel("Val F1 (per class)")
ax1.set_title("Path A: per-class F1")
ax1.set_ylim(0, 1.0)
ax1.grid(axis="y", linestyle="--", alpha=0.3)
ax1.legend(loc="lower right", fontsize=8)

# Path B IoU
ax2.bar(x - w/2, img_pathB, w, label="image-only seg", color="#bdbdbd",
        hatch='////', edgecolor='black', linewidth=0.5)
ax2.bar(x + w/2, ca_pathB,  w, label="best CA seg (hyb-qwn)", color="#d95f0e",
        hatch='\\\\\\\\', edgecolor='black', linewidth=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(CLASSES, rotation=30, ha="right")
ax2.set_ylabel("Val IoU (per class)")
ax2.set_title("Path B: per-class IoU")
ax2.set_ylim(0, 1.0)
ax2.grid(axis="y", linestyle="--", alpha=0.3)
ax2.legend(loc="lower right", fontsize=8)

fig.suptitle("Cross-attention's largest gains are on minor classes", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_perclass_minor.png")
plt.show()
print(f"Saved: {FIG_DIR/'fig4_perclass_minor.png'}")


# %% CELL 6 — List all figures + report-ready summary printout
print("\n=== Generated figures (Drive) ===")
for f in sorted(FIG_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
print(f"\nAlso available from 05_pathB_segmentation:")
print(f"  fig_seg_qualitative.png")
print(f"  fig_seg_attention_overlay.png")

print("\n=== Headline numbers for the report ===")
print(f"Path A:")
print(f"  image-only baseline: mAP {pathA['image_only']['mAP_mean']:.3f} ± {pathA['image_only']['mAP_std']:.3f}")
print(f"  best CA (hybrid_qwen): mAP {pathA[best_ca_pathA]['mAP_mean']:.3f} ± {pathA[best_ca_pathA]['mAP_std']:.3f}")
print(f"  fusion gain: +{pathA[best_ca_pathA]['mAP_mean'] - pathA['image_only']['mAP_mean']:.3f} mAP")
print(f"\nPath B:")
print(f"  image-only seg: mIoU {pathB['seg_image_only']['mIoU_mean']:.3f} ± {pathB['seg_image_only']['mIoU_std']:.3f}")
print(f"  best CA seg (hybrid_qwen): mIoU {pathB[best_ca_pathB]['mIoU_mean']:.3f} ± {pathB[best_ca_pathB]['mIoU_std']:.3f}")
print(f"  fusion gain: +{pathB[best_ca_pathB]['mIoU_mean'] - pathB['seg_image_only']['mIoU_mean']:.3f} mIoU")

print(f"\nBackbone ablation:")
v_means = [bb_v[k]['mAP_mean'] for k in bb_v]
r_means = [bb_r[k]['mAP_mean'] for k in bb_r]
deltas = [r - v for v, r in zip(v_means, r_means)]
print(f"  RemoteCLIP > Vanilla CLIP on {sum(1 for d in deltas if d > 0)}/{len(deltas)} conditions")
print(f"  avg Δ mAP: +{np.mean(deltas):.3f}  (best +{max(deltas):.3f}, worst +{min(deltas):.3f})")

print(f"\nFusion family means (mAP):")
for fusion in ["late", "film", "gated", "cross_attn"]:
    means = [fusion_variants[f"{fusion}__{c}"]["mAP_mean"] for c in CAPTION_COLS]
    print(f"  {fusion:12s} {np.mean(means):.3f}  (range {min(means):.3f}–{max(means):.3f})")


# %% CELL 7 — Re-display saved figures inline (in case earlier cells didn't render)
# Uses IPython.display.Image so the saved PNGs show even if matplotlib state was cleared.

from IPython.display import Image, display, Markdown

display(Markdown("## Generated figures"))
for fname in ["fig1_fusion_heatmap.png", "fig2_ca_vs_late_delta.png",
              "fig3_backbone_compare.png", "fig4_perclass_minor.png"]:
    p = FIG_DIR / fname
    if p.exists():
        display(Markdown(f"### {fname}"))
        display(Image(filename=str(p)))

display(Markdown("## Segmentation figures (from 05_pathB_segmentation)"))
for fname in ["fig_seg_qualitative.png", "fig_seg_attention_overlay.png"]:
    p = FEAT_DIR / fname
    if p.exists():
        display(Markdown(f"### {fname}"))
        display(Image(filename=str(p)))


# %% CELL 8 — Bundle figures + result JSONs into a zip and trigger browser download
# Downloads to your local Mac. Excludes large feature .pt files (those stay on Drive).

import zipfile
from google.colab import files

zip_path = "/content/phase2_outputs.zip"

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    # Report figures
    for f in (FEAT_DIR / "figures").iterdir():
        zf.write(f, f"figures/{f.name}")
    # Segmentation figures (saved at top level of FEAT_DIR by 05_pathB)
    for fname in ["fig_seg_qualitative.png", "fig_seg_attention_overlay.png"]:
        p = FEAT_DIR / fname
        if p.exists():
            zf.write(p, f"figures/{fname}")
    # All results JSONs
    for fname in ["pathA_results.json", "pathA_results_multi_seed.json",
                  "pathA_backbone_ablation.json", "pathA_fusion_variants.json",
                  "pathB_seg_results.json"]:
        p = FEAT_DIR / fname
        if p.exists():
            zf.write(p, f"results/{fname}")

# Print contents of the zip
print(f"Zip: {zip_path}  ({Path(zip_path).stat().st_size / 1024:.1f} KB)")
with zipfile.ZipFile(zip_path) as zf:
    for n in zf.namelist():
        info = zf.getinfo(n)
        print(f"  {n}  ({info.file_size / 1024:.1f} KB)")

print("\nTriggering browser download (check your Downloads folder)...")
files.download(zip_path)
