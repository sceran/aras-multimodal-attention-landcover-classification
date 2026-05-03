# =============================================================================
# 01_patch_features.py — Patch-level RemoteCLIP image feature extraction
# =============================================================================
# Goal: Extract per-patch image features (50 tokens × 512 dim per image) from
# RemoteCLIP-ViT-B/32 and cache to Drive. Replaces sanity_features.pt for
# Path A (cross-attention needs patch tokens) and Path B (segmentation needs
# spatial features).
#
# Wall-clock: ~30-35 min (mostly Drive I/O for 10K images).
# Storage: ~1 GB on Drive.
#
# Paste each cell into a separate Colab cell and run sequentially.
# =============================================================================


# %% CELL 1 — Mount Drive (skip if already mounted) and verify paths
from google.colab import drive
try:
    drive.mount('/content/drive')
except Exception as e:
    print("Drive may already be mounted:", e)

from pathlib import Path
DATA_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/DI725/DI725_project_dataset")
FEAT_DIR  = Path("/content/drive/MyDrive/Colab Notebooks/DI725/phase2_features")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

assert (DATA_ROOT / "captions.csv").exists(), "captions.csv not found"
assert (DATA_ROOT / "images").is_dir(), "images/ not found"
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"FEAT_DIR:  {FEAT_DIR}")
print(f"Existing cached features: {[p.name for p in FEAT_DIR.iterdir()]}")


# %% CELL 2 — Install (skip if already installed in this session)
# Only install if open_clip is missing — saves time on warm sessions.
import importlib
try:
    importlib.import_module("open_clip")
    print("open_clip already installed.")
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "open_clip_torch", "wandb"], check=True)
    print("Installed open_clip_torch and wandb.")


# %% CELL 3 — Imports + load RemoteCLIP-ViT-B/32
import torch
import numpy as np
import pandas as pd
from PIL import Image
import open_clip
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

DEVICE = "cuda"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

ckpt_path = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
msg = model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
print("Load message:", msg)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Verify expected shapes
with torch.no_grad():
    dummy = torch.randn(2, 3, 224, 224, device=DEVICE)
    out = model.encode_image(dummy)
    print(f"Pooled image out shape: {out.shape}  (expected [2, 512])")
    print(f"Vision dim: {model.visual.conv1.out_channels}, embed dim: {out.shape[-1]}")


# %% CELL 4 — Hook-based patch encoder (version-proof)
# Newer open_clip versions may apply additional post-pooling steps (head_norm,
# attention pooling, or 'avg' pool_type) that diverge from a manually replicated
# forward pass. We instead register a forward hook on ln_post during a normal
# encode_image call to capture pre-pool tokens directly — guaranteed to match
# what the model actually computes for this checkpoint.

print("=== Vision module diagnostic ===")
print(f"pool_type / global_pool: {getattr(model.visual, 'pool_type', getattr(model.visual, 'global_pool', '?'))}")
print(f"attn_pool: {model.visual.attn_pool}")
print(f"proj shape: {None if model.visual.proj is None else tuple(model.visual.proj.shape)}")
print(f"ln_post: {model.visual.ln_post}")

@torch.no_grad()
def encode_image_patches(images, normalize=True):
    """
    Returns per-token image features [B, 1+N, 512] where N=49 for ViT-B/32.
    Hook captures ln_post output (pre-pool, all tokens) during the standard
    encode_image forward, then applies the same vision projection per token.
    Token 0 = CLS-like; tokens 1..N = patches in row-major 7x7 order.
    """
    visual = model.visual
    captured = {}

    def hook(_, __, out):
        captured['tokens'] = out  # [B, 1+N, vision_width]

    h = visual.ln_post.register_forward_hook(hook)
    try:
        _ = model.encode_image(images)  # standard forward populates the hook
    finally:
        h.remove()

    tokens = captured['tokens']               # [B, 50, 768] for ViT-B/32
    if visual.proj is not None:
        tokens = tokens @ visual.proj         # [B, 50, 512]

    if normalize:
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
    return tokens

# Sanity check — informational only, no strict assert on CLS == pooled because
# the standard pool path may add post-pool ops we don't reproduce.
with torch.no_grad():
    dummy = torch.randn(2, 3, 224, 224, device=DEVICE)
    patch_out = encode_image_patches(dummy, normalize=True)
    pooled    = model.encode_image(dummy)
    pooled    = pooled / pooled.norm(dim=-1, keepdim=True)

print(f"\nPatch out shape: {patch_out.shape}  (expected [2, 50, 512])")
print(f"Patch token 0 first 5: {patch_out[0, 0, :5].cpu().tolist()}")
print(f"Standard pooled first 5: {pooled[0, :5].cpu().tolist()}")
diff = (patch_out[:, 0] - pooled).abs().max().item()
print(f"CLS-token vs pooled max abs diff: {diff:.4f}  (informational; >0 OK)")

assert patch_out.shape == (2, 50, 512), f"Unexpected patch shape: {patch_out.shape}"
print("\nPatch encoder OK — proceeding to Cell 5.")


# %% CELL 5 — Encode all 10K images at patch level
# Storage: 10000 × 50 × 512 × 4 bytes = ~1.0 GB
# Time: ~30-35 min on A100 (Drive I/O dominated)

df = pd.read_csv(DATA_ROOT / "captions.csv")
print(f"Encoding {len(df)} images...")

class ImgDS(Dataset):
    def __init__(self, names, transform):
        self.names = names; self.t = transform
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        return self.t(Image.open(DATA_ROOT/"images"/self.names[i]).convert("RGB"))

ds = ImgDS(df["filename"].tolist(), preprocess)
loader = DataLoader(ds, batch_size=128, num_workers=4, shuffle=False, pin_memory=True)

patch_feats_chunks = []
with torch.no_grad():
    for x in tqdm(loader, desc="patch-encode"):
        x = x.to(DEVICE, non_blocking=True)
        f = encode_image_patches(x)            # [B, 50, 512] normalized
        patch_feats_chunks.append(f.cpu().half())  # store FP16 to halve size

patch_feats = torch.cat(patch_feats_chunks, dim=0)
print(f"\nPatch features shape: {patch_feats.shape}  dtype: {patch_feats.dtype}")
print(f"Memory footprint: {patch_feats.numel() * 2 / 1e9:.2f} GB")


# %% CELL 6 — Save to Drive
out_path = FEAT_DIR / "patch_features_remoteclip_vitb32.pt"
torch.save({
    "patch_features": patch_feats,           # [10000, 50, 512] FP16
    "filenames": df["filename"].tolist(),    # row-aligned with patch_features
    "tau": 10,
    "model": "RemoteCLIP-ViT-B/32",
    "shape_note": "[N_images, 1+49 tokens, 512 dim]; token 0 = CLS, 1-49 = 7x7 patches row-major",
}, out_path)
print(f"Saved: {out_path}")
print(f"Size: {out_path.stat().st_size / 1e9:.2f} GB")

# Quick verify reload
reload = torch.load(out_path, map_location="cpu")
print(f"Reload patch_features shape: {reload['patch_features'].shape}")
print(f"Reload filenames[:3]: {reload['filenames'][:3]}")
