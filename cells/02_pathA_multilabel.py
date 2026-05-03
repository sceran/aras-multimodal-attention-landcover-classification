# =============================================================================
# 02_pathA_multilabel.py — Multi-label classification with 3 fusion strategies
# =============================================================================
# Goal: Train 15 conditions (5 caption types × 3 fusion modes) on cached
# RemoteCLIP features. Multi-label target (>=10% composition threshold).
#
# Fusion modes:
#   - image_only: image CLS [B,512] -> MLP -> 7
#   - late      : [image_CLS || text_CLS] [B,1024] -> MLP -> 7
#   - cross_attn: text tokens (Q) attend over image patches (K,V) -> mean pool
#                 -> MLP -> 7
#
# Wall-clock: ~3-5 min per condition on cached features = ~50-75 min for 15 runs.
#
# PREREQUISITES:
#   - 01_patch_features.py CELL 1-6 successfully run; patch features saved to
#     /content/drive/MyDrive/Colab Notebooks/DI725/phase2_features/
#   - If session was lost, re-run 01_patch_features.py CELLs 1, 2, 3 (mount,
#     install, load model) before starting here.
# =============================================================================


# %% CELL 1 — Load cached patch features, captions, labels, splits
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

DATA_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/DI725/DI725_project_dataset")
FEAT_DIR  = Path("/content/drive/MyDrive/Colab Notebooks/DI725/phase2_features")
DEVICE = "cuda"
SEED = 42
TAU = 10
CLASSES = ["Tree","Shrub","Grass","Crop","Built-up","Barren","Water"]
CAPTION_COLS = ["hybrid_gemma3-4b","hybrid_qwen3-vl-8b","text_qwen3-4b",
                "vision_gemma3-4b","vision_qwen3-vl-8b"]

torch.manual_seed(SEED); np.random.seed(SEED)

# Load patch features
patch_data = torch.load(FEAT_DIR / "patch_features_remoteclip_vitb32.pt", map_location="cpu")
patch_features = patch_data["patch_features"]   # [10000, 50, 512] FP16
filenames_cached = patch_data["filenames"]
print(f"Patch features: {patch_features.shape}  dtype={patch_features.dtype}")

# Load captions
df = pd.read_csv(DATA_ROOT / "captions.csv")
assert df["filename"].tolist() == filenames_cached, "filename order drift between CSV and cached features!"
print(f"Captions CSV: {len(df)} rows, order verified.")

# Build multi-label
labels = (df[CLASSES].values >= TAU).astype(np.float32)
print(f"Label positives per class: {dict(zip(CLASSES, labels.sum(0).astype(int).tolist()))}")

# Stratified 80/20 split on dominant class (matches sanity check)
dominant = df[CLASSES].values.argmax(1)
train_idx, val_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=dominant)
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# Move features to GPU as FP16 once (~0.5 GB)
patch_features_gpu = patch_features.to(DEVICE)
image_cls_gpu      = patch_features_gpu[:, 0, :]                      # [10000, 512]
image_patches_gpu  = patch_features_gpu[:, 1:, :]                     # [10000, 49, 512]
labels_gpu         = torch.tensor(labels, device=DEVICE)

print(f"image_cls on GPU:     {image_cls_gpu.shape}  {image_cls_gpu.dtype}")
print(f"image_patches on GPU: {image_patches_gpu.shape}  {image_patches_gpu.dtype}")


# %% CELL 2 — Define hook-based text token encoder
# If you skipped 01 cells in this session, re-run 01_patch_features.py CELLs
# 1-3 first to load the model + tokenizer.

import open_clip
from huggingface_hub import hf_hub_download

# Reload model if not present (idempotent)
if "model" not in globals():
    print("Reloading RemoteCLIP-ViT-B/32...")
    ckpt_path = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
else:
    print("Model already loaded.")

# Diagnostic for text path
print(f"text_pool_type: {getattr(model, 'text_pool_type', '?')}")
print(f"text_projection shape: {None if model.text_projection is None else tuple(model.text_projection.shape)}")
print(f"ln_final: {model.ln_final}")

@torch.no_grad()
def encode_text_tokens_and_pooled(captions, batch_size=256, normalize=True):
    """
    Hook-based text encoder. Returns:
      tokens [N, 77, 512] - all tokens, projected, normalized (for cross-attn Q)
      pooled [N, 512]     - EOS-pooled (matches model.encode_text), normalized
    """
    all_tokens, all_pooled = [], []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        text_ids = tokenizer(batch).to(DEVICE)

        captured = {}
        def hook(_, __, out):
            captured['tokens'] = out  # [B, 77, transformer_width]

        h = model.ln_final.register_forward_hook(hook)
        try:
            pooled = model.encode_text(text_ids)
        finally:
            h.remove()

        tokens_pre_proj = captured['tokens']
        if model.text_projection is not None:
            tokens = tokens_pre_proj @ model.text_projection   # [B, 77, 512]
        else:
            tokens = tokens_pre_proj

        if normalize:
            tokens = tokens / tokens.norm(dim=-1, keepdim=True)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)

        all_tokens.append(tokens.cpu().half())
        all_pooled.append(pooled.cpu().half())

    return torch.cat(all_tokens), torch.cat(all_pooled)

# Sanity check
sample = ["A satellite image of cropland and forests.", "Predominantly grass with some water."]
t, p = encode_text_tokens_and_pooled(sample)
print(f"\nSanity: tokens {t.shape} pooled {p.shape}")
assert t.shape == (2, 77, 512) and p.shape == (2, 512)


# %% CELL 3 — Encode all 5 caption sets (cache to RAM)
# Total ~5 captions × 10K × 77 × 512 × 2 bytes = ~3.9 GB FP16, fits comfortably.
# Pooled: 5 × 10K × 512 × 2 = 50 MB.
# Compute: ~30 sec total on A100.

text_tokens = {}   # caption_col -> [10000, 77, 512] FP16, on CPU
text_pooled = {}   # caption_col -> [10000, 512]    FP16, on CPU

for col in CAPTION_COLS:
    print(f"\nEncoding {col}...")
    captions = df[col].fillna("").tolist()
    tk, pl = encode_text_tokens_and_pooled(captions)
    text_tokens[col] = tk
    text_pooled[col] = pl
    print(f"  tokens {tk.shape}  pooled {pl.shape}")

# Verify storage
total_gb = sum(t.numel() * 2 for t in text_tokens.values()) / 1e9
print(f"\nTotal text token storage in CPU RAM: {total_gb:.2f} GB")


# %% CELL 4 — Define fusion modules
import torch.nn as nn

class ImageOnlyHead(nn.Module):
    def __init__(self, dim=512, hidden=256, n_classes=7, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, image_cls):
        return self.net(image_cls)

class LateFusion(nn.Module):
    def __init__(self, dim=512, hidden=256, n_classes=7, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim*2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, image_cls, text_pooled):
        return self.net(torch.cat([image_cls, text_pooled], dim=-1))

class CrossAttentionFusion(nn.Module):
    """Text tokens (Q) attend over image patches (K,V). Pool attended Q and classify."""
    def __init__(self, dim=512, n_heads=8, hidden=256, n_classes=7, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, image_patches, text_tokens):
        # image_patches: [B, 49, 512] (K,V), text_tokens: [B, 77, 512] (Q)
        attended, _ = self.attn(query=text_tokens, key=image_patches, value=image_patches)
        attended = self.norm(attended + text_tokens)             # residual
        pooled = attended.mean(dim=1)                            # [B, 512]
        return self.head(pooled)

# Quick parameter count
for name, m in [("ImageOnly", ImageOnlyHead()), ("Late", LateFusion()), ("CrossAttn", CrossAttentionFusion())]:
    n = sum(p.numel() for p in m.parameters())
    print(f"{name:12s} trainable params: {n:,}")


# %% CELL 5 — Training / eval helper
import wandb
from sklearn.metrics import f1_score, average_precision_score

def make_features_for(condition, caption_col=None):
    """Returns a tuple of GPU tensors used by the module's forward."""
    if condition == "image_only":
        return (image_cls_gpu.float(),)
    elif condition == "late":
        text_p = text_pooled[caption_col].to(DEVICE).float()      # [10000, 512]
        return (image_cls_gpu.float(), text_p)
    elif condition == "cross_attn":
        text_t = text_tokens[caption_col].to(DEVICE).float()      # [10000, 77, 512]
        return (image_patches_gpu.float(), text_t)
    raise ValueError(condition)

def make_module(condition):
    if condition == "image_only": return ImageOnlyHead().to(DEVICE)
    if condition == "late":       return LateFusion().to(DEVICE)
    if condition == "cross_attn": return CrossAttentionFusion().to(DEVICE)
    raise ValueError(condition)

def train_condition(condition, caption_col, epochs=30, batch_size=256,
                    lr=1e-3, weight_decay=1e-4, wandb_project="di725-phase2-main"):
    name = f"{condition}" + ("" if condition == "image_only" else f"__{caption_col}")
    print(f"\n=== {name} ===")

    feats = make_features_for(condition, caption_col)
    net = make_module(condition)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    yt = labels_gpu[train_idx]
    yv = labels_gpu[val_idx]
    yv_np = yv.cpu().numpy()

    wandb.init(project=wandb_project, name=name, reinit=True,
               config={"condition": condition, "caption": caption_col,
                       "epochs": epochs, "lr": lr, "wd": weight_decay,
                       "batch_size": batch_size, "tau": TAU, "seed": SEED})
    best = {"val_mAP": 0.0, "val_f1_macro": 0.0, "val_f1_per_class": None, "epoch": -1}

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(train_idx), device=DEVICE)
        epoch_loss = 0.0; nbatch = 0
        for i in range(0, len(train_idx), batch_size):
            b = perm[i:i+batch_size]
            tr_idx = torch.tensor(train_idx, device=DEVICE)[b]
            inputs = tuple(f[tr_idx] for f in feats)
            opt.zero_grad()
            logits = net(*inputs)
            loss = loss_fn(logits, yt[b])
            loss.backward()
            opt.step()
            epoch_loss += loss.item(); nbatch += 1

        net.eval()
        with torch.no_grad():
            v_idx = torch.tensor(val_idx, device=DEVICE)
            v_inputs = tuple(f[v_idx] for f in feats)
            logits = net(*v_inputs).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs > 0.5).astype(int)
            mAP = average_precision_score(yv_np, probs, average='macro')
            f1m = f1_score(yv_np, preds, average='macro', zero_division=0)
            f1pc = f1_score(yv_np, preds, average=None, zero_division=0)

        log = {"epoch": ep, "train/loss": epoch_loss/nbatch,
               "val/mAP": mAP, "val/f1_macro": f1m}
        for c, f in zip(CLASSES, f1pc):
            log[f"val/f1_{c}"] = f
        wandb.log(log)

        if mAP > best["val_mAP"]:
            best = {"val_mAP": float(mAP), "val_f1_macro": float(f1m),
                    "val_f1_per_class": [float(x) for x in f1pc], "epoch": ep}

    wandb.summary.update({"best_" + k: v for k, v in best.items() if k != "val_f1_per_class"})
    for c, f in zip(CLASSES, best["val_f1_per_class"]):
        wandb.summary[f"best_val_f1_{c}"] = f
    wandb.finish()
    return name, best


# %% CELL 6 — Run all 15 conditions
results = {}

# 1 image-only baseline
name, best = train_condition("image_only", caption_col=None)
results[name] = best

# 5 late fusion (one per caption)
for cap in CAPTION_COLS:
    name, best = train_condition("late", caption_col=cap)
    results[name] = best

# 5 cross-attention (one per caption)
for cap in CAPTION_COLS:
    name, best = train_condition("cross_attn", caption_col=cap)
    results[name] = best

print(f"\n{len(results)} conditions complete.")


# %% CELL 7 — Aggregate results table + save
import json

print("\n" + "="*110)
header = f"{'Condition':38s} {'mAP':>6s} {'F1m':>6s}   " + " ".join(f"{c[:5]:>5s}" for c in CLASSES)
print(header)
print("="*110)
for name, best in results.items():
    pc = " ".join(f"{f:.2f}" for f in best["val_f1_per_class"])
    print(f"{name:38s} {best['val_mAP']:.3f}  {best['val_f1_macro']:.3f}   {pc}")

# Save results
out_json = FEAT_DIR / "pathA_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

# Quick story: image-only vs best fusion gain on minor classes
img = results["image_only"]
print("\n=== Fusion gain on MINOR classes (Shrub, Built-up, Barren, Water) ===")
minor_idx = [CLASSES.index(c) for c in ["Shrub", "Built-up", "Barren", "Water"]]
for name, best in results.items():
    if name == "image_only": continue
    gain = [best["val_f1_per_class"][i] - img["val_f1_per_class"][i] for i in minor_idx]
    avg_gain = sum(gain) / len(gain)
    if avg_gain > 0.02:
        print(f"  {name:38s} avg minor F1 gain: +{avg_gain:.3f}")


# %% CELL 8 — Multi-seed re-run for variance estimation
# Redefines train_condition to accept a seed and properly seeds RNGs at run start.
# Re-runs all 11 conditions with 3 seeds = 33 runs (~5 min on A100, cached features).
# Aggregates mean ± std for each metric.

import random as _random

def set_seeds(seed: int):
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_condition_seeded(condition, caption_col, seed, epochs=30, batch_size=256,
                            lr=1e-3, weight_decay=1e-4, wandb_project="di725-phase2-main"):
    """Like train_condition but resets RNGs at start and includes seed in run name."""
    set_seeds(seed)

    name = f"{condition}" + ("" if condition == "image_only" else f"__{caption_col}") + f"__s{seed}"
    print(f"=== {name} ===")

    feats = make_features_for(condition, caption_col)
    net = make_module(condition)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    yt = labels_gpu[train_idx]
    yv_np = labels_gpu[val_idx].cpu().numpy()

    wandb.init(project=wandb_project, name=name, reinit=True,
               tags=[f"seed={seed}", f"fusion={condition}", f"caption={caption_col or 'none'}"],
               config={"condition": condition, "caption": caption_col, "seed": seed,
                       "epochs": epochs, "lr": lr, "wd": weight_decay,
                       "batch_size": batch_size, "tau": TAU})
    best = {"val_mAP": 0.0, "val_f1_macro": 0.0, "val_f1_per_class": None, "epoch": -1}

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(train_idx), device=DEVICE)
        epoch_loss = 0.0; nbatch = 0
        tr_idx_t = torch.tensor(train_idx, device=DEVICE)
        for i in range(0, len(train_idx), batch_size):
            b = perm[i:i+batch_size]
            inputs = tuple(f[tr_idx_t[b]] for f in feats)
            opt.zero_grad()
            loss = loss_fn(net(*inputs), yt[b])
            loss.backward(); opt.step()
            epoch_loss += loss.item(); nbatch += 1

        net.eval()
        with torch.no_grad():
            v_idx = torch.tensor(val_idx, device=DEVICE)
            v_inputs = tuple(f[v_idx] for f in feats)
            logits = net(*v_inputs).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs > 0.5).astype(int)
            mAP = average_precision_score(yv_np, probs, average='macro')
            f1m = f1_score(yv_np, preds, average='macro', zero_division=0)
            f1pc = f1_score(yv_np, preds, average=None, zero_division=0)

        log = {"epoch": ep, "train/loss": epoch_loss/nbatch,
               "val/mAP": mAP, "val/f1_macro": f1m}
        for c, f in zip(CLASSES, f1pc):
            log[f"val/f1_{c}"] = f
        wandb.log(log)

        if mAP > best["val_mAP"]:
            best = {"val_mAP": float(mAP), "val_f1_macro": float(f1m),
                    "val_f1_per_class": [float(x) for x in f1pc], "epoch": ep}

    wandb.finish()
    return best


SEEDS = [42, 1337, 2024]
results_multi = {}  # condition_key -> list of best dicts (one per seed)

for seed in SEEDS:
    print(f"\n##### SEED {seed} #####")
    # image-only
    key = "image_only"
    results_multi.setdefault(key, []).append(
        train_condition_seeded("image_only", None, seed))
    # late × 5 captions
    for cap in CAPTION_COLS:
        key = f"late__{cap}"
        results_multi.setdefault(key, []).append(
            train_condition_seeded("late", cap, seed))
    # cross_attn × 5 captions
    for cap in CAPTION_COLS:
        key = f"cross_attn__{cap}"
        results_multi.setdefault(key, []).append(
            train_condition_seeded("cross_attn", cap, seed))

print(f"\nTotal runs: {sum(len(v) for v in results_multi.values())}")


# %% CELL 9 — Aggregate mean ± std across seeds, print table, save
results_agg = {}
for name, runs in results_multi.items():
    mAPs = np.array([r["val_mAP"] for r in runs])
    f1ms = np.array([r["val_f1_macro"] for r in runs])
    f1pcs = np.array([r["val_f1_per_class"] for r in runs])  # [n_seeds, 7]
    results_agg[name] = {
        "n_seeds": len(runs),
        "seeds": SEEDS,
        "mAP_mean": float(mAPs.mean()), "mAP_std": float(mAPs.std()),
        "f1_macro_mean": float(f1ms.mean()), "f1_macro_std": float(f1ms.std()),
        "f1_per_class_mean": f1pcs.mean(0).tolist(),
        "f1_per_class_std": f1pcs.std(0).tolist(),
        "raw_runs": runs,
    }

# Print compact table
print("\n" + "="*125)
header = f"{'Condition':38s} {'mAP (mean±std)':>16s}  {'F1m (mean±std)':>16s}   " + " ".join(f"{c[:5]:>10s}" for c in CLASSES)
print(header)
print("="*125)
for name, agg in results_agg.items():
    pc = " ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(agg['f1_per_class_mean'], agg['f1_per_class_std']))
    print(f"{name:38s}   {agg['mAP_mean']:.3f}±{agg['mAP_std']:.3f}    {agg['f1_macro_mean']:.3f}±{agg['f1_macro_std']:.3f}   {pc}")

# Save aggregated
out_json = FEAT_DIR / "pathA_results_multi_seed.json"
with open(out_json, "w") as f:
    json.dump(results_agg, f, indent=2)
print(f"\nSaved: {out_json}")

# Headline numbers for the report
print("\n=== Headline (3-seed mean ± std) ===")
img = results_agg["image_only"]
print(f"image-only:                          mAP {img['mAP_mean']:.3f}±{img['mAP_std']:.3f}  F1m {img['f1_macro_mean']:.3f}±{img['f1_macro_std']:.3f}")
best_late = max((k for k in results_agg if k.startswith("late__")), key=lambda k: results_agg[k]["mAP_mean"])
best_ca   = max((k for k in results_agg if k.startswith("cross_attn__")), key=lambda k: results_agg[k]["mAP_mean"])
for k, label in [(best_late, "best LATE     "), (best_ca, "best CROSS-ATTN")]:
    a = results_agg[k]
    print(f"{label}: {k:40s} mAP {a['mAP_mean']:.3f}±{a['mAP_std']:.3f}  F1m {a['f1_macro_mean']:.3f}±{a['f1_macro_std']:.3f}")

# CA vs late head-to-head per caption
print("\n=== CA vs LATE head-to-head per caption (Δ mAP) ===")
for cap in CAPTION_COLS:
    late_a = results_agg[f"late__{cap}"]
    ca_a   = results_agg[f"cross_attn__{cap}"]
    delta  = ca_a["mAP_mean"] - late_a["mAP_mean"]
    print(f"  {cap:25s}  late {late_a['mAP_mean']:.3f}±{late_a['mAP_std']:.3f}  →  CA {ca_a['mAP_mean']:.3f}±{ca_a['mAP_std']:.3f}   Δ={delta:+.3f}")
