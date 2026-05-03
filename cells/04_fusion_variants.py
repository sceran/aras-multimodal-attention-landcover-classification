# =============================================================================
# 04_fusion_variants.py — FiLM and Gated fusion variants
# =============================================================================
# Goal: Extend Path A fusion comparison with two more variants beyond the
# existing image-only / late / cross-attention. This makes the "which fusion
# wins?" question non-trivial — comparing 5 fusion modes instead of 2.
#
# New modules:
#   - FiLM: text -> (gamma, beta) modulate image_cls -> head
#   - Gated: sigmoid gate interpolates between image_cls and text_pooled
#
# Both use pooled features (same input as late fusion), so they cost the same
# at training time. 2 new fusions × 5 captions × 3 seeds = 30 runs (~5 min).
#
# PREREQUISITES:
#   - 02_pathA_multilabel.py CELL 1-9 already run; results_agg in memory.
#   - 00_health_check shows [ALL GOOD].
# =============================================================================


# %% CELL 1 — Define FiLM and Gated modules
import torch
import torch.nn as nn

class FiLMFusion(nn.Module):
    """Text generates feature-wise (gamma, beta) that modulate image CLS.
    Initialized so gamma starts near 1 and beta near 0 (identity at start)."""
    def __init__(self, dim=512, hidden=256, n_classes=7, dropout=0.1):
        super().__init__()
        self.modulator = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim*2),
        )
        # Initialize the final modulator layer to zero so gamma=1, beta=0 at start
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

        self.head = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, image_cls, text_pooled):
        gb = self.modulator(text_pooled)
        gamma, beta = gb.chunk(2, dim=-1)
        modulated = (1.0 + gamma) * image_cls + beta
        return self.head(modulated)


class GatedFusion(nn.Module):
    """Sigmoid gate interpolates between image and text features."""
    def __init__(self, dim=512, hidden=256, n_classes=7, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, image_cls, text_pooled):
        cat = torch.cat([image_cls, text_pooled], dim=-1)
        g = self.gate(cat)                                # [B, 512] in (0, 1)
        fused = g * image_cls + (1.0 - g) * text_pooled
        return self.head(fused)


# Param counts
for name, m in [("FiLM", FiLMFusion()), ("Gated", GatedFusion())]:
    n = sum(p.numel() for p in m.parameters())
    print(f"{name:8s} trainable params: {n:,}")


# %% CELL 2 — Extend make_module and make_features_for to handle new types
# We monkey-patch the existing helpers (defined in 02 CELL 5) so the existing
# train_condition_seeded picks up the new fusions automatically.

_orig_make_module = make_module
_orig_make_features_for = make_features_for

def make_module(condition):
    if condition == "film":  return FiLMFusion().to(DEVICE)
    if condition == "gated": return GatedFusion().to(DEVICE)
    return _orig_make_module(condition)

def make_features_for(condition, caption_col=None):
    # FiLM and Gated both use (image_cls, text_pooled) like late fusion
    if condition in ("film", "gated"):
        text_p = text_pooled[caption_col].to(DEVICE).float()
        return (image_cls_gpu.float(), text_p)
    return _orig_make_features_for(condition, caption_col)

# Make sure train_condition_seeded uses the new globals
import builtins as _b
print("Extended make_module and make_features_for to handle 'film' and 'gated'.")


# %% CELL 3 — Run FiLM and Gated sweeps (2 fusion × 5 caption × 3 seed = 30 runs)
SEEDS = [42, 1337, 2024]

variant_results_multi = {}
for seed in SEEDS:
    print(f"\n##### VARIANTS SEED {seed} #####")
    for cap in CAPTION_COLS:
        key = f"film__{cap}"
        variant_results_multi.setdefault(key, []).append(
            train_condition_seeded("film", cap, seed))
    for cap in CAPTION_COLS:
        key = f"gated__{cap}"
        variant_results_multi.setdefault(key, []).append(
            train_condition_seeded("gated", cap, seed))

print(f"\nTotal new runs: {sum(len(v) for v in variant_results_multi.values())}")


# %% CELL 4 — Aggregate variants, combine with results_agg, print full table
import json
import numpy as np

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

variant_agg = aggregate(variant_results_multi)

# Combined table: all 5 fusion modes × 5 captions + image-only
all_fusion_agg = {}
all_fusion_agg["image_only"] = {k: v for k, v in results_agg["image_only"].items() if k != "raw_runs"}
for fusion in ["late", "film", "gated", "cross_attn"]:
    for cap in CAPTION_COLS:
        key = f"{fusion}__{cap}"
        if fusion in ("film", "gated"):
            all_fusion_agg[key] = variant_agg[key]
        else:
            all_fusion_agg[key] = {k: v for k, v in results_agg[key].items() if k != "raw_runs"}

# Save
out_path = FEAT_DIR / "pathA_fusion_variants.json"
with open(out_path, "w") as f:
    json.dump(all_fusion_agg, f, indent=2)
print(f"Saved: {out_path}")

# Print compact comparison
print("\n" + "="*100)
print(f"{'Condition':38s}  {'mAP±std':>14s}  {'F1m±std':>14s}  {'best epoch?':>10s}")
print("="*100)
for name in all_fusion_agg:
    a = all_fusion_agg[name]
    print(f"{name:38s}  {a['mAP_mean']:.3f}±{a['mAP_std']:.3f}    {a['f1_macro_mean']:.3f}±{a['f1_macro_std']:.3f}")

# 2D summary: fusion × caption mAP
print("\n=== mAP heatmap-style: fusion (rows) × caption (cols) ===")
print(f"{'fusion / caption':22s}  " + "  ".join(f"{c[:12]:>12s}" for c in CAPTION_COLS) + "  mean")
for fusion in ["late", "film", "gated", "cross_attn"]:
    row = []
    for cap in CAPTION_COLS:
        m = all_fusion_agg[f"{fusion}__{cap}"]["mAP_mean"]
        row.append(m)
    mean_r = sum(row) / len(row)
    print(f"{fusion:22s}  " + "  ".join(f"{m:>12.3f}" for m in row) + f"  {mean_r:.3f}")

# Best per fusion family
print("\n=== Best mAP per fusion family ===")
for fusion in ["late", "film", "gated", "cross_attn"]:
    best_key = max((k for k in all_fusion_agg if k.startswith(f"{fusion}__")),
                   key=lambda k: all_fusion_agg[k]["mAP_mean"])
    a = all_fusion_agg[best_key]
    print(f"  {fusion:12s}  {best_key:30s}  mAP {a['mAP_mean']:.3f}±{a['mAP_std']:.3f}  F1m {a['f1_macro_mean']:.3f}")

# Headline gain over late fusion (per caption)
print("\n=== Δ mAP over Late fusion (same caption) ===")
print(f"{'caption':22s}  " + "  ".join(f"{f:>10s}" for f in ["late", "film", "gated", "cross_attn"]))
for cap in CAPTION_COLS:
    base = all_fusion_agg[f"late__{cap}"]["mAP_mean"]
    deltas = [all_fusion_agg[f"{fusion}__{cap}"]["mAP_mean"] - base
              for fusion in ["late", "film", "gated", "cross_attn"]]
    print(f"{cap:22s}  " + "  ".join(f"{d:+10.3f}" for d in deltas))
