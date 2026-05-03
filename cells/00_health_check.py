# =============================================================================
# 00_health_check.py — Verify Colab session state
# =============================================================================
# Run this cell after any disconnect/idle period. It checks:
#   - GPU available
#   - Drive mounted + cached features visible on disk
#   - In-memory globals (model, tokenizer, features, labels, splits, results)
#
# If anything is MISSING, the output tells you exactly which previous cells
# from 01_patch_features.py and 02_pathA_multilabel.py to re-run to recover.
# =============================================================================


# %% CELL — Health check
from pathlib import Path

print("="*70)
print("COLAB SESSION HEALTH CHECK")
print("="*70)

issues = []

# ---- 1. GPU ----
try:
    import torch
    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.device_count()} device(s))")
    else:
        print("[FAIL] CUDA not available")
        issues.append("Restart runtime with A100/T4 enabled")
except Exception as e:
    print(f"[FAIL] torch import: {e}")
    issues.append("Re-import torch (re-run 01_patch_features.py CELL 3)")

# ---- 2. Drive mount + on-disk caches ----
DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/DI725")
DATA_ROOT  = DRIVE_ROOT / "DI725_project_dataset"
FEAT_DIR   = DRIVE_ROOT / "phase2_features"

if not DATA_ROOT.exists():
    print(f"[FAIL] Dataset not visible: {DATA_ROOT}")
    issues.append("Re-mount Drive (01_patch_features.py CELL 1)")
else:
    print(f"[OK] Dataset visible: {DATA_ROOT}")

cache_files = {
    "patch_features": FEAT_DIR / "patch_features_remoteclip_vitb32.pt",
    "pathA_results_multi": FEAT_DIR / "pathA_results_multi_seed.json",
}
for name, p in cache_files.items():
    if p.exists():
        sz = p.stat().st_size / 1e6
        print(f"[OK] {name:25s} on Drive: {sz:.1f} MB")
    else:
        print(f"[--] {name:25s} not on Drive yet (will be created later)")

# ---- 3. In-memory globals ----
required_globals = {
    "model":              "open_clip RemoteCLIP model — re-run 01 CELL 3 (or 02 CELL 2)",
    "tokenizer":          "open_clip tokenizer — re-run 01 CELL 3 (or 02 CELL 2)",
    "preprocess":         "image preprocess transform — re-run 01 CELL 3",
    "patch_features":     "[10000,50,512] FP16 — re-run 02 CELL 1",
    "image_cls_gpu":      "image CLS on GPU — re-run 02 CELL 1",
    "image_patches_gpu":  "image patches on GPU — re-run 02 CELL 1",
    "labels_gpu":         "labels tensor on GPU — re-run 02 CELL 1",
    "train_idx":          "train indices — re-run 02 CELL 1",
    "val_idx":            "val indices — re-run 02 CELL 1",
    "text_tokens":        "dict of [10000,77,512] per caption — re-run 02 CELL 3",
    "text_pooled":        "dict of [10000,512] per caption — re-run 02 CELL 3",
    "results_agg":        "Path A multi-seed aggregated results — re-run 02 CELL 8 + 9",
}

print("\n--- In-memory state ---")
import builtins
g = globals()
missing = []
for name, hint in required_globals.items():
    if name in g:
        v = g[name]
        if hasattr(v, "shape"):
            print(f"[OK] {name:20s} {tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"[OK] {name:20s} dict[{len(v)}]")
        elif isinstance(v, (list, tuple)):
            print(f"[OK] {name:20s} {type(v).__name__}({len(v)})")
        else:
            print(f"[OK] {name:20s} {type(v).__name__}")
    else:
        print(f"[MISSING] {name:20s}  → {hint}")
        missing.append((name, hint))

# ---- 4. Verdict ----
print("\n" + "="*70)
if not missing and not issues:
    print("[ALL GOOD] Session is healthy — proceed with new cells (backbone ablation, etc.)")
elif missing:
    cells_to_rerun = set()
    for name, hint in missing:
        if "01 CELL 3" in hint: cells_to_rerun.add("01_patch_features.py CELL 3")
        if "01 CELL 1" in hint: cells_to_rerun.add("01_patch_features.py CELL 1")
        if "02 CELL 1" in hint: cells_to_rerun.add("02_pathA_multilabel.py CELL 1")
        if "02 CELL 2" in hint: cells_to_rerun.add("02_pathA_multilabel.py CELL 2")
        if "02 CELL 3" in hint: cells_to_rerun.add("02_pathA_multilabel.py CELL 3")
        if "02 CELL 8" in hint: cells_to_rerun.add("02_pathA_multilabel.py CELL 8 + 9 (only if you need results_agg in memory; the JSON on Drive can be reloaded instead)")
    print(f"[ACTION REQUIRED] Re-run these cells in order:")
    for c in sorted(cells_to_rerun):
        print(f"  - {c}")
else:
    print("[WARNING] Issues:")
    for i in issues:
        print(f"  - {i}")
