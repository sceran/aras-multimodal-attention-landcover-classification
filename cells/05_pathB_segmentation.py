# =============================================================================
# 05_pathB_segmentation.py — Path B preliminary semantic segmentation
# =============================================================================
# Goal: Test whether the multi-label fusion findings generalize to dense
# prediction. Captions cannot give pixel-level layout, so the fusion gain
# pattern (CA's noise robustness, FiLM-vs-late) may differ from Path A.
#
# Task: 7-class semantic segmentation at patch resolution (7×7 = 49 patches).
# Mask is downsampled by majority class per 32×32 image patch.
#
# Conditions: 1 image-only seg + 3 captions × 2 fusion (late, CA) = 7 conds.
# 7 × 3 seeds = 21 runs (~5-10 min compute).
#
# Note: For CA in segmentation, image patches are Q and text tokens are K,V
# (OPPOSITE direction from Path A classification, where text was Q). Each
# image patch asks "what does the caption say about my region?".
#
# PREREQUISITES:
#   - 02_pathA_multilabel.py CELL 1-9 already run; results_agg in memory
#     (image_patches_gpu, text_tokens, text_pooled, train/val_idx).
#   - 00_health_check shows [ALL GOOD].
# =============================================================================


# %% CELL 1 — Mask diagnostic + class palette
import numpy as np
from PIL import Image

MASKS_DIR = DATA_ROOT / "masks"
IMG_SIZE  = 224     # match RemoteCLIP preprocess (256 source -> 224 input)
PATCH     = 32      # ViT-B/32 stride; gives 7x7 patch grid at 224

# From spec PDF — Class RGB palette
CLASS_COLORS = np.array([
    [0,   100, 0],     # 0 Tree
    [255, 182, 193],   # 1 Shrub
    [154, 205, 50],    # 2 Grass
    [255, 215, 0],     # 3 Crop
    [139, 69,  19],    # 4 Built-up
    [211, 211, 211],   # 5 Barren
    [0,   0,   255],   # 6 Water
], dtype=np.int32)

# Sanity check on first mask — masks are stored at 256×256 but RemoteCLIP
# preprocess resizes images to 224×224. We resize masks the same way (NEAREST
# to preserve class colors exactly) so patch indexing aligns with image features.
sample_name = df["filename"].iloc[0]
sample_mask_orig = Image.open(MASKS_DIR / sample_name)
sample_mask_resized = sample_mask_orig.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
sample_mask = np.array(sample_mask_resized)

print(f"Sample mask: {sample_name}")
print(f"  original size: {sample_mask_orig.size}")
print(f"  resized to:    {sample_mask.shape[:2]}  (NEAREST, preserves class colors)")
print(f"  unique colors after resize:")
unique_colors = np.unique(sample_mask.reshape(-1, sample_mask.shape[-1]), axis=0)
for c in unique_colors:
    matched = np.where((CLASS_COLORS == c).all(axis=1))[0]
    cls = CLASSES[matched[0]] if len(matched) else "UNKNOWN"
    print(f"    {tuple(c.tolist())}  -> {cls}")

assert sample_mask.shape[:2] == (IMG_SIZE, IMG_SIZE), f"Resize failed: {sample_mask.shape[:2]}"
unmatched = sum(1 for c in unique_colors if not (CLASS_COLORS == c).all(axis=1).any())
assert unmatched == 0, f"{unmatched} unique colors don't match palette — NEAREST resize may have created new colors"
print(f"\nMask -> 7x7 patch grid OK (PATCH={PATCH}, IMG_SIZE={IMG_SIZE})")


# %% CELL 2 — Convert all masks to per-patch class labels (cache to Drive)
# Algorithm:
#   RGB mask [224,224,3] -> class index mask [224,224] (exact RGB match)
#   -> reshape to [7,32,7,32] -> majority class per 32x32 patch -> [49] flat
#
# Vectorized over batches; one-hot histogram per patch -> argmax.

PATCH_LABELS_PATH = FEAT_DIR / "patch_labels_7x7.pt"

if PATCH_LABELS_PATH.exists():
    print(f"Patch labels already cached: {PATCH_LABELS_PATH}")
    cache = torch.load(PATCH_LABELS_PATH, map_location="cpu")
    patch_labels_seg = cache["patch_labels"]   # [10000, 49] long
    print(f"Loaded: {patch_labels_seg.shape}  {patch_labels_seg.dtype}")
else:
    def rgb_to_class_idx(rgb_mask: np.ndarray) -> np.ndarray:
        """[H,W,3] uint8 -> [H,W] int64 with class indices 0..6.
        Uses exact RGB match; pixels not matching any class get -1 (will be ignored)."""
        H, W, _ = rgb_mask.shape
        out = np.full((H, W), -1, dtype=np.int64)
        for idx, color in enumerate(CLASS_COLORS):
            mask = ((rgb_mask[..., 0] == color[0]) &
                    (rgb_mask[..., 1] == color[1]) &
                    (rgb_mask[..., 2] == color[2]))
            out[mask] = idx
        return out

    def majority_per_patch(class_mask: np.ndarray, patch=32) -> np.ndarray:
        """[H,W] -> [49] majority class per patch (replace -1 by mode of valid pixels;
        if all -1 in a patch, default to class 0 — should not happen with clean masks)."""
        H, W = class_mask.shape
        nh, nw = H // patch, W // patch  # 7, 7
        # Replace -1 with a sentinel that won't be in the valid range
        clean = class_mask.copy()
        # Use bincount per patch
        out = np.zeros(nh * nw, dtype=np.int64)
        idx = 0
        for h in range(nh):
            for w in range(nw):
                pat = clean[h*patch:(h+1)*patch, w*patch:(w+1)*patch].flatten()
                valid = pat[pat >= 0]
                if len(valid) == 0:
                    out[idx] = 0
                else:
                    out[idx] = np.bincount(valid, minlength=7).argmax()
                idx += 1
        return out

    print("Processing 10000 masks -> patch labels...")
    all_patches = np.zeros((len(df), 49), dtype=np.int64)
    unmatched_total = 0
    for i, name in enumerate(tqdm(df["filename"].tolist(), desc="masks")):
        # Resize 256x256 mask -> 224x224 with NEAREST (matches image preprocess)
        rgb = np.array(Image.open(MASKS_DIR / name).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        cls = rgb_to_class_idx(rgb)
        unmatched_total += (cls == -1).sum()
        all_patches[i] = majority_per_patch(cls, patch=PATCH)

    patch_labels_seg = torch.tensor(all_patches, dtype=torch.long)
    print(f"\nUnmatched pixels across dataset: {unmatched_total} (expect near 0)")
    print(f"Patch labels: {patch_labels_seg.shape}  {patch_labels_seg.dtype}")
    print("Class distribution (across all 10K × 49 patches):")
    for c, n in enumerate(np.bincount(patch_labels_seg.flatten().numpy(), minlength=7)):
        print(f"  {CLASSES[c]:10s}: {int(n):8d}  ({100*n/(10000*49):.2f}%)")

    torch.save({"patch_labels": patch_labels_seg, "filenames": df["filename"].tolist(),
                "patch_grid": (7, 7), "patch_size": 32}, PATCH_LABELS_PATH)
    print(f"Saved: {PATCH_LABELS_PATH}")

# Move to GPU
patch_labels_gpu = patch_labels_seg.to(DEVICE)
print(f"patch_labels on GPU: {patch_labels_gpu.shape}")


# %% CELL 3 — Segmentation modules
import torch.nn as nn

class ImageOnlySeg(nn.Module):
    """Per-patch classification from image features."""
    def __init__(self, dim=512, hidden=256, n_classes=7, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, image_patches):       # [B, 49, 512]
        return self.head(image_patches)     # [B, 49, 7]


class LateSeg(nn.Module):
    """Broadcast pooled text to each patch, concat, then classify per patch."""
    def __init__(self, dim=512, hidden=256, n_classes=7, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim*2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, image_patches, text_pooled):   # [B,49,512], [B,512]
        text_b = text_pooled.unsqueeze(1).expand(-1, image_patches.shape[1], -1)
        cat = torch.cat([image_patches, text_b], dim=-1)
        return self.head(cat)                         # [B, 49, 7]


class CASeg(nn.Module):
    """Image patches Q over text tokens K,V (opposite direction from Path A classification).
    Each patch asks: what does the caption say about my region?"""
    def __init__(self, dim=512, n_heads=8, hidden=256, n_classes=7, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, image_patches, text_tokens):   # [B,49,512], [B,77,512]
        attended, attn_weights = self.attn(query=image_patches, key=text_tokens, value=text_tokens,
                                           need_weights=True, average_attn_weights=True)
        attended = self.norm(attended + image_patches)
        return self.head(attended), attn_weights      # logits [B,49,7], weights [B,49,77]

for name, m in [("ImageOnlySeg", ImageOnlySeg()),
                ("LateSeg", LateSeg()),
                ("CASeg", CASeg())]:
    n = sum(p.numel() for p in m.parameters())
    print(f"{name:14s} trainable params: {n:,}")


# %% CELL 4 — Training/eval helper for segmentation
import wandb

def compute_iou(preds: np.ndarray, targets: np.ndarray, n_classes=7):
    """preds, targets: [N*49] int. Returns per-class IoU and mIoU."""
    ious = np.zeros(n_classes)
    for c in range(n_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        inter = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        ious[c] = inter / union if union > 0 else float('nan')
    miou = np.nanmean(ious)
    return ious, miou

def train_seg(condition, caption_col, seed, epochs=30, batch_size=128,
              lr=1e-3, weight_decay=1e-4, wandb_project="di725-phase2-seg"):
    set_seeds(seed)
    name = f"seg_{condition}" + ("" if condition == "image_only" else f"__{caption_col}") + f"__s{seed}"
    print(f"=== {name} ===")

    # Features
    if condition == "image_only":
        feats = (image_patches_gpu.float(),)
        net = ImageOnlySeg().to(DEVICE)
        is_ca = False
    elif condition == "late":
        text_p = text_pooled[caption_col].to(DEVICE).float()
        feats = (image_patches_gpu.float(), text_p)
        net = LateSeg().to(DEVICE)
        is_ca = False
    elif condition == "cross_attn":
        text_t = text_tokens[caption_col].to(DEVICE).float()
        feats = (image_patches_gpu.float(), text_t)
        net = CASeg().to(DEVICE)
        is_ca = True
    else:
        raise ValueError(condition)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    yt = patch_labels_gpu[train_idx]   # [Ntr, 49]
    yv_np = patch_labels_gpu[val_idx].cpu().numpy()  # [Nv, 49]

    wandb.init(project=wandb_project, name=name, reinit=True,
               tags=[f"seed={seed}", f"fusion={condition}", f"caption={caption_col or 'none'}"],
               config={"condition": condition, "caption": caption_col, "seed": seed,
                       "epochs": epochs, "lr": lr, "wd": weight_decay,
                       "batch_size": batch_size})

    best = {"val_mIoU": 0.0, "val_iou_per_class": None, "epoch": -1}

    tr_idx_t = torch.tensor(train_idx, device=DEVICE)
    v_idx_t  = torch.tensor(val_idx, device=DEVICE)

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(train_idx), device=DEVICE)
        epoch_loss = 0.0; nbatch = 0
        for i in range(0, len(train_idx), batch_size):
            b = perm[i:i+batch_size]
            idx_b = tr_idx_t[b]
            inputs = tuple(f[idx_b] for f in feats)
            opt.zero_grad()
            if is_ca:
                logits, _ = net(*inputs)
            else:
                logits = net(*inputs)                        # [B, 49, 7]
            # CE expects [B, C, *] and target [B, *]
            loss = loss_fn(logits.permute(0, 2, 1), yt[b])   # logits -> [B,7,49]
            loss.backward(); opt.step()
            epoch_loss += loss.item(); nbatch += 1

        net.eval()
        with torch.no_grad():
            v_inputs = tuple(f[v_idx_t] for f in feats)
            if is_ca:
                logits, _ = net(*v_inputs)
            else:
                logits = net(*v_inputs)
            preds = logits.argmax(dim=-1).cpu().numpy().reshape(-1)   # [Nv*49]
            targs = yv_np.reshape(-1)
            ious, miou = compute_iou(preds, targs)

        log = {"epoch": ep, "train/loss": epoch_loss/nbatch, "val/mIoU": miou}
        for c, v in zip(CLASSES, ious):
            log[f"val/IoU_{c}"] = float(v)
        wandb.log(log)

        if miou > best["val_mIoU"]:
            best = {"val_mIoU": float(miou),
                    "val_iou_per_class": [float(x) for x in ious],
                    "epoch": ep}

    wandb.finish()
    return best


# %% CELL 5 — Run 7 conditions × 3 seeds = 21 runs
SEEDS = [42, 1337, 2024]
SEG_CAPTIONS = ["hybrid_qwen3-vl-8b", "text_qwen3-4b", "vision_qwen3-vl-8b"]   # 3-caption sample (high/qual/low leakage)

seg_results_multi = {}
for seed in SEEDS:
    print(f"\n##### SEG SEED {seed} #####")
    seg_results_multi.setdefault("seg_image_only", []).append(
        train_seg("image_only", None, seed))
    for cap in SEG_CAPTIONS:
        seg_results_multi.setdefault(f"seg_late__{cap}", []).append(
            train_seg("late", cap, seed))
    for cap in SEG_CAPTIONS:
        seg_results_multi.setdefault(f"seg_cross_attn__{cap}", []).append(
            train_seg("cross_attn", cap, seed))

print(f"\nTotal runs: {sum(len(v) for v in seg_results_multi.values())}")


# %% CELL 6 — Aggregate + table + save
import json

def aggregate_seg(results_multi):
    agg = {}
    for name, runs in results_multi.items():
        miou = np.array([r["val_mIoU"] for r in runs])
        iouc = np.array([r["val_iou_per_class"] for r in runs])
        agg[name] = {
            "n_seeds": len(runs),
            "mIoU_mean": float(miou.mean()), "mIoU_std": float(miou.std()),
            "iou_per_class_mean": iouc.mean(0).tolist(),
            "iou_per_class_std":  iouc.std(0).tolist(),
        }
    return agg

seg_agg = aggregate_seg(seg_results_multi)

print("\n" + "="*120)
print(f"{'Condition':38s} {'mIoU±std':>14s}   " + " ".join(f"{c[:5]:>10s}" for c in CLASSES))
print("="*120)
for name, a in seg_agg.items():
    pc = " ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(a['iou_per_class_mean'], a['iou_per_class_std']))
    print(f"{name:38s} {a['mIoU_mean']:.3f}±{a['mIoU_std']:.3f}   {pc}")

# CA vs late head-to-head per caption
print("\n=== CA vs LATE head-to-head (Δ mIoU, segmentation) ===")
for cap in SEG_CAPTIONS:
    late_a = seg_agg[f"seg_late__{cap}"]
    ca_a   = seg_agg[f"seg_cross_attn__{cap}"]
    delta  = ca_a["mIoU_mean"] - late_a["mIoU_mean"]
    print(f"  {cap:25s}  late {late_a['mIoU_mean']:.3f}±{late_a['mIoU_std']:.3f}  "
          f"→  CA {ca_a['mIoU_mean']:.3f}±{ca_a['mIoU_std']:.3f}   Δ={delta:+.3f}")

# Compare against image-only baseline
img = seg_agg["seg_image_only"]
print(f"\nimage-only seg mIoU: {img['mIoU_mean']:.3f}±{img['mIoU_std']:.3f}")
for fusion in ["seg_late", "seg_cross_attn"]:
    for cap in SEG_CAPTIONS:
        a = seg_agg[f"{fusion}__{cap}"]
        delta = a["mIoU_mean"] - img["mIoU_mean"]
        print(f"  {fusion+'__'+cap:38s}  Δ over image-only seg: {delta:+.3f}")

out_path = FEAT_DIR / "pathB_seg_results.json"
with open(out_path, "w") as f:
    json.dump(seg_agg, f, indent=2)
print(f"\nSaved: {out_path}")


# %% CELL 7 — Qualitative visualization (4 examples)
# Picks 4 val images, runs the best CA model + image-only model, overlays predicted seg + GT.
# Skipped figures saved to FEAT_DIR for later inclusion in report.

import matplotlib.pyplot as plt

# Use the seed-42 best CA model from the last run (re-train one model for viz)
print("\nTraining one CA model (seed=42, hybrid_qwen) for visualization...")
viz_caption = "hybrid_qwen3-vl-8b"
set_seeds(42)
text_t = text_tokens[viz_caption].to(DEVICE).float()
viz_net = CASeg().to(DEVICE)
opt = torch.optim.AdamW(viz_net.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
yt = patch_labels_gpu[train_idx]
tr_idx_t = torch.tensor(train_idx, device=DEVICE)
for ep in range(30):
    viz_net.train()
    perm = torch.randperm(len(train_idx), device=DEVICE)
    for i in range(0, len(train_idx), 128):
        b = perm[i:i+128]
        idx_b = tr_idx_t[b]
        opt.zero_grad()
        logits, _ = viz_net(image_patches_gpu[idx_b].float(), text_t[idx_b])
        loss = loss_fn(logits.permute(0, 2, 1), yt[b])
        loss.backward(); opt.step()

# Pick 4 val images, run viz model
viz_net.eval()
viz_idx = val_idx[:4]
with torch.no_grad():
    v_idx_t = torch.tensor(viz_idx, device=DEVICE)
    logits, attn_w = viz_net(image_patches_gpu[v_idx_t].float(), text_t[v_idx_t])
    preds = logits.argmax(dim=-1).cpu().numpy()      # [4, 49]
    attn = attn_w.cpu().numpy()                      # [4, 49, 77]

# Color palette for visualization (matches CLASS_COLORS)
PALETTE = CLASS_COLORS / 255.0

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for row, vi in enumerate(viz_idx):
    name = df["filename"].iloc[vi]
    img = np.array(Image.open(DATA_ROOT/"images"/name))
    mask_rgb = np.array(Image.open(MASKS_DIR/name))

    # GT patch labels reshaped to 7x7
    gt_patches = patch_labels_seg[vi].numpy().reshape(7, 7)
    # Pred patch labels
    pred_patches = preds[row].reshape(7, 7)

    axes[row, 0].imshow(img); axes[row, 0].set_title(f"{name}"); axes[row, 0].axis('off')
    axes[row, 1].imshow(mask_rgb); axes[row, 1].set_title("GT mask (RGB)"); axes[row, 1].axis('off')
    axes[row, 2].imshow(PALETTE[gt_patches]); axes[row, 2].set_title("GT patch (7×7)"); axes[row, 2].axis('off')
    axes[row, 3].imshow(PALETTE[pred_patches]); axes[row, 3].set_title("CA seg pred"); axes[row, 3].axis('off')

plt.tight_layout()
fig_path = FEAT_DIR / "fig_seg_qualitative.png"
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# Attention map figure: for one image, show how much each text token attends to each patch
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for row, vi in enumerate(viz_idx):
    name = df["filename"].iloc[vi]
    img = np.array(Image.open(DATA_ROOT/"images"/name))
    # Attention from all 49 patches to the average text position
    avg_attn = attn[row].mean(axis=1)        # [49] — how much each patch attends to text on avg
    avg_attn = avg_attn.reshape(7, 7)

    axes[row].imshow(img, alpha=0.6)
    # Upsample attention to image resolution
    import scipy.ndimage as ndi
    attn_up = ndi.zoom(avg_attn, 32, order=0)
    axes[row].imshow(attn_up, cmap='jet', alpha=0.4)
    axes[row].set_title(f"{name}\nattention overlay"); axes[row].axis('off')

plt.tight_layout()
fig_path2 = FEAT_DIR / "fig_seg_attention_overlay.png"
plt.savefig(fig_path2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path2}")
