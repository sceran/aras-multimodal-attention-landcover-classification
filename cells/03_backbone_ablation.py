# =============================================================================
# 03_backbone_ablation.py — Vanilla CLIP vs RemoteCLIP backbone ablation
# =============================================================================
# Goal: Address Phase-1 feedback #2 ("vanilla CLIP not justified") empirically.
# Re-extract features with OpenAI's vanilla CLIP-ViT-B/32 (same architecture,
# different weights). Re-run the SAME 11×3-seed sweep with vanilla features.
# Compare backbone × fusion × caption.
#
# Wall-clock: ~30 min Drive I/O (vanilla image extraction) + ~5 min compute.
#
# PREREQUISITES:
#   - 02_pathA_multilabel.py CELL 1-9 already run; results_agg in memory.
#   - 00_health_check shows [ALL GOOD].
# =============================================================================


# %% CELL 1 — Load vanilla CLIP and define backbone-agnostic encoders
# We keep RemoteCLIP loaded as `model` (used by 02 cells); vanilla goes in
# `vanilla_model` namespace. Same ViT-B/32 architecture so feature shape and
# pipeline match — only weights differ.

import torch
import open_clip

print("Loading vanilla CLIP-ViT-B/32 (OpenAI weights)...")
vanilla_model, _, vanilla_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
vanilla_model = vanilla_model.to(DEVICE).eval()
vanilla_tokenizer = open_clip.get_tokenizer("ViT-B-32")
print("Vanilla CLIP loaded.")
print(f"  pool_type: {getattr(vanilla_model.visual, 'pool_type', '?')}")
print(f"  proj shape: {tuple(vanilla_model.visual.proj.shape) if vanilla_model.visual.proj is not None else None}")

@torch.no_grad()
def encode_image_patches_with(net, images, normalize=True):
    """Hook-based patch encoder for any open_clip ViT model."""
    visual = net.visual
    captured = {}
    def hook(_, __, out): captured['tokens'] = out
    h = visual.ln_post.register_forward_hook(hook)
    try:
        _ = net.encode_image(images)
    finally:
        h.remove()
    tokens = captured['tokens']
    if visual.proj is not None:
        tokens = tokens @ visual.proj
    if normalize:
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
    return tokens

@torch.no_grad()
def encode_text_with(net, tok_fn, captions, batch_size=256, normalize=True):
    """Hook-based text encoder; returns (tokens [N,77,512], pooled [N,512])."""
    all_t, all_p = [], []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        ids = tok_fn(batch).to(DEVICE)
        captured = {}
        def hook(_, __, out): captured['t'] = out
        h = net.ln_final.register_forward_hook(hook)
        try:
            pooled = net.encode_text(ids)
        finally:
            h.remove()
        tk = captured['t']
        if net.text_projection is not None:
            tk = tk @ net.text_projection
        if normalize:
            tk = tk / tk.norm(dim=-1, keepdim=True)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        all_t.append(tk.cpu().half()); all_p.append(pooled.cpu().half())
    return torch.cat(all_t), torch.cat(all_p)


# %% CELL 2 — Extract vanilla CLIP patch features (THE 30-MIN STEP)
# Saves to Drive so we never pay this I/O cost again.

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

VANILLA_PATCH_PATH = FEAT_DIR / "patch_features_vanillaclip_vitb32.pt"

if VANILLA_PATCH_PATH.exists():
    print(f"Vanilla patch features already on Drive: {VANILLA_PATCH_PATH}")
    print("Loading from cache...")
    vanilla_patch_data = torch.load(VANILLA_PATCH_PATH, map_location="cpu")
    vanilla_patch_features = vanilla_patch_data["patch_features"]
    print(f"Loaded: {vanilla_patch_features.shape}  {vanilla_patch_features.dtype}")
else:
    class ImgDS(Dataset):
        def __init__(self, names, transform):
            self.names = names; self.t = transform
        def __len__(self): return len(self.names)
        def __getitem__(self, i):
            return self.t(Image.open(DATA_ROOT/"images"/self.names[i]).convert("RGB"))

    ds = ImgDS(df["filename"].tolist(), vanilla_preprocess)
    loader = DataLoader(ds, batch_size=128, num_workers=4, shuffle=False, pin_memory=True)

    chunks = []
    with torch.no_grad():
        for x in tqdm(loader, desc="vanilla-patch-encode"):
            x = x.to(DEVICE, non_blocking=True)
            f = encode_image_patches_with(vanilla_model, x)
            chunks.append(f.cpu().half())
    vanilla_patch_features = torch.cat(chunks, dim=0)
    print(f"Vanilla patch features: {vanilla_patch_features.shape}  {vanilla_patch_features.dtype}")

    torch.save({
        "patch_features": vanilla_patch_features,
        "filenames": df["filename"].tolist(),
        "model": "vanilla CLIP-ViT-B/32 (OpenAI)",
        "shape_note": "[N_images, 1+49 tokens, 512 dim]; token 0 = CLS, 1-49 = 7x7 patches",
    }, VANILLA_PATCH_PATH)
    print(f"Saved: {VANILLA_PATCH_PATH}")


# %% CELL 3 — Extract vanilla CLIP text features for all 5 captions
print("Encoding text with vanilla CLIP...")
vanilla_text_tokens = {}
vanilla_text_pooled = {}
for col in CAPTION_COLS:
    print(f"  {col}...")
    tk, pl = encode_text_with(vanilla_model, vanilla_tokenizer, df[col].fillna("").tolist())
    vanilla_text_tokens[col] = tk
    vanilla_text_pooled[col] = pl

print("Done. RAM footprint: ~3.94 GB tokens + 50 MB pooled.")

# Move to GPU once (FP16, ~0.5 GB)
vanilla_patch_features_gpu = vanilla_patch_features.to(DEVICE)
vanilla_image_cls_gpu      = vanilla_patch_features_gpu[:, 0, :]
vanilla_image_patches_gpu  = vanilla_patch_features_gpu[:, 1:, :]
print(f"vanilla_image_cls on GPU:     {vanilla_image_cls_gpu.shape}  {vanilla_image_cls_gpu.dtype}")
print(f"vanilla_image_patches on GPU: {vanilla_image_patches_gpu.shape}")


# %% CELL 4 — Backbone-aware sweep using vanilla features
# Override globals temporarily inside a helper so train_condition_seeded picks
# up vanilla features without needing refactor. Restore after the sweep.

_remoteclip_state = {
    "image_cls_gpu": image_cls_gpu,
    "image_patches_gpu": image_patches_gpu,
    "text_tokens": text_tokens,
    "text_pooled": text_pooled,
}

def run_backbone_sweep(backbone_label, img_cls, img_patches, txt_tokens, txt_pooled, seeds=(42, 1337, 2024)):
    global image_cls_gpu, image_patches_gpu, text_tokens, text_pooled
    image_cls_gpu     = img_cls
    image_patches_gpu = img_patches
    text_tokens       = txt_tokens
    text_pooled       = txt_pooled

    results = {}
    for seed in seeds:
        print(f"\n##### {backbone_label} SEED {seed} #####")
        results.setdefault("image_only", []).append(
            train_condition_seeded("image_only", None, seed,
                                   wandb_project=f"di725-phase2-backbone-{backbone_label}"))
        for cap in CAPTION_COLS:
            results.setdefault(f"late__{cap}", []).append(
                train_condition_seeded("late", cap, seed,
                                       wandb_project=f"di725-phase2-backbone-{backbone_label}"))
        for cap in CAPTION_COLS:
            results.setdefault(f"cross_attn__{cap}", []).append(
                train_condition_seeded("cross_attn", cap, seed,
                                       wandb_project=f"di725-phase2-backbone-{backbone_label}"))
    return results

vanilla_results_multi = run_backbone_sweep(
    "vanilla", vanilla_image_cls_gpu, vanilla_image_patches_gpu,
    vanilla_text_tokens, vanilla_text_pooled
)

# Restore RemoteCLIP globals
image_cls_gpu     = _remoteclip_state["image_cls_gpu"]
image_patches_gpu = _remoteclip_state["image_patches_gpu"]
text_tokens       = _remoteclip_state["text_tokens"]
text_pooled       = _remoteclip_state["text_pooled"]
print("\nRestored RemoteCLIP globals.")
print(f"Total runs: {sum(len(v) for v in vanilla_results_multi.values())}")


# %% CELL 5 — Aggregate vanilla results + comparison table
import json

def aggregate(results_multi):
    agg = {}
    for name, runs in results_multi.items():
        mAPs = np.array([r["val_mAP"] for r in runs])
        f1ms = np.array([r["val_f1_macro"] for r in runs])
        f1pcs = np.array([r["val_f1_per_class"] for r in runs])
        agg[name] = {
            "n_seeds": len(runs),
            "mAP_mean": float(mAPs.mean()), "mAP_std": float(mAPs.std()),
            "f1_macro_mean": float(f1ms.mean()), "f1_macro_std": float(f1ms.std()),
            "f1_per_class_mean": f1pcs.mean(0).tolist(),
            "f1_per_class_std": f1pcs.std(0).tolist(),
        }
    return agg

vanilla_agg = aggregate(vanilla_results_multi)

# Side-by-side comparison
print("\n" + "="*120)
print(f"{'Condition':38s}  {'Vanilla CLIP':>22s}    {'RemoteCLIP':>22s}    {'Δ mAP':>8s}")
print(f"{'':38s}  {'mAP±std':>10s} {'F1m':>10s}   {'mAP±std':>10s} {'F1m':>10s}")
print("="*120)
for name in vanilla_agg:
    v = vanilla_agg[name]
    r = results_agg[name]
    delta = r["mAP_mean"] - v["mAP_mean"]
    print(f"{name:38s}  "
          f"{v['mAP_mean']:.3f}±{v['mAP_std']:.3f} {v['f1_macro_mean']:.3f}     "
          f"{r['mAP_mean']:.3f}±{r['mAP_std']:.3f} {r['f1_macro_mean']:.3f}     "
          f"{delta:+.3f}")

# Save comparison
out = {
    "vanilla": vanilla_agg,
    "remoteclip": {k: {kk: v for kk, v in vv.items() if kk != "raw_runs"}
                   for k, vv in results_agg.items()},
}
out_path = FEAT_DIR / "pathA_backbone_ablation.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")

# Headline: average improvement of RemoteCLIP over vanilla
deltas = [results_agg[k]["mAP_mean"] - vanilla_agg[k]["mAP_mean"] for k in vanilla_agg]
print(f"\n=== Headline ===")
print(f"RemoteCLIP outperforms vanilla CLIP by avg Δ mAP = {sum(deltas)/len(deltas):+.3f}")
print(f"  best gain: {max(deltas):+.3f}  worst: {min(deltas):+.3f}")
print(f"  positive gain on {sum(1 for d in deltas if d > 0)}/{len(deltas)} conditions")
