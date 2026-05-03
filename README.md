# ARAS Multi-Modal Attention Fusion for Land-Cover Classification

Comparing four cross-modal fusion strategies (late concatenation, FiLM,
gated, and cross-attention) for **multi-label land-cover classification** of
remote-sensing imagery, using the [ARAS400K dataset](https://zenodo.org/records/18890661)
and a frozen [RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP) backbone with five
LLM-generated caption strategies.

## Key findings (3-seed mean ± std)

- **Cross-attention (CA) wins on every caption type**, beating late
  concatenation by Δ mAP **+0.036 to +0.051** consistently. Best run:
  `cross-attention + hybrid_qwen3-vl-8b` → **mAP 0.948 ± 0.003** vs
  image-only baseline 0.817 ± 0.001.
- **CA is robust to noisy text.** Vision-only captions (which we found to
  *hallucinate* on this dataset — describing 92%-grass scenes as "barren")
  give *no* gain to late fusion (mAP 0.815 ≈ image-only) but recover
  **+0.049 mAP** under cross-attention. CA's selectivity downweights
  unreliable text tokens.
- **Caption strategy interacts with fusion.** Highly leaky hybrid captions
  saturate late fusion; cross-attention extracts marginal gains *on top of*
  leakage. The largest CA-vs-late gap appears with vision_qwen (most noisy):
  Δ mAP +0.051.
- **Backbone ablation: RemoteCLIP > vanilla CLIP on 11/11 conditions**,
  avg Δ mAP +0.031. Backbone gap is wider for late fusion (+0.04–0.05) and
  narrower for CA (+0.014–0.029): strong fusion partly compensates for a
  weaker backbone.
- **Fusion-mechanism comparison:** mean mAP across captions —
  CA 0.908 > FiLM 0.903 > Late 0.864 > **Gated 0.857** (gated *under*-performs
  late: sigmoid gate is too aggressive and collapses one modality).
- **Preliminary segmentation extension** (7-class patch-level seg at 7×7):
  fusion-direction switches in CA (image patches as Q over text tokens as
  K,V) — see `cells/05_pathB_segmentation.py`.

## Repository layout

```
di725_phase2.ipynb          # Main notebook with all cell outputs visible
cells/                      # Source-of-truth Colab-compatible cell scripts
├── 00_health_check.py      # Verify session state
├── 01_patch_features.py    # Extract patch-level RemoteCLIP image features
├── 02_pathA_multilabel.py  # Multi-label classification, 11 conds × 3 seeds
├── 03_backbone_ablation.py # Vanilla CLIP vs RemoteCLIP
├── 04_fusion_variants.py   # Add FiLM and Gated fusion modules
├── 05_pathB_segmentation.py# Preliminary patch-level segmentation
└── 06_analysis_figures.py  # Aggregate JSONs into report-ready figures
report/                     # Phase-2 IEEE-format report (LaTeX, to follow)
```

## Tracked experiments (Weights & Biases)

All training runs are logged publicly. Each run has full per-epoch metrics,
hyperparameters, and tags by seed / fusion type / caption strategy:

| Project | Contents |
|---|---|
| [di725-phase2-sanity](https://wandb.ai/sceran/di725-phase2-sanity) | Initial 8 baselines used to motivate multi-label as the target task |
| [di725-phase2-main](https://wandb.ai/sceran/di725-phase2-main) | Path A: 33 multi-seed runs (image-only / late / CA × 5 captions) + 30 fusion variant runs (FiLM, Gated) |
| [di725-phase2-backbone-vanilla](https://wandb.ai/sceran/di725-phase2-backbone-vanilla) | Vanilla CLIP backbone ablation, 33 runs |
| [di725-phase2-seg](https://wandb.ai/sceran/di725-phase2-seg) | Path B preliminary segmentation, 21 runs |

## Reproducing the experiments

1. **Environment.** `pip install -r requirements.txt`. Tested on Colab Pro with
   an A100 GPU; CPU fallback works for small batches.
2. **Data.** Download the [ARAS400K dataset](https://zenodo.org/records/18890661)
   and place `images/`, `masks/`, and `captions.csv` under a single root
   directory. Update `DATA_ROOT` in `cells/01_patch_features.py` to point to it.
3. **Run order.**
   1. `00_health_check.py` — confirm GPU + Drive (or local) paths.
   2. `01_patch_features.py` — encode and cache patch-level image features
      (~30 min Drive I/O; ~30 sec on warm cache).
   3. `02_pathA_multilabel.py` — main multi-label sweep (11 conds × 3 seeds,
      ~5 min on cached features).
   4. `03_backbone_ablation.py` — repeat with vanilla CLIP weights.
   5. `04_fusion_variants.py` — add FiLM + Gated fusion modules.
   6. `05_pathB_segmentation.py` — preliminary segmentation extension.

All training runs log to the Weights & Biases projects listed above.

## Method summary

- **Backbone:** RemoteCLIP-ViT-B/32 (frozen). Patch features extracted via a
  forward-hook on `ln_post` to recover the per-token, projected representation.
- **Multi-label target:** `composition_pct(class_i) ≥ 10%` → 7-bit binary
  vector. Captures presence of dominant *and* minor classes — fusion's gain
  is concentrated on the latter.
- **Fusion modules** (all share a 256→7 MLP head, BCE loss, AdamW lr=1e-3,
  30 epochs):
  - *Image-only* — image CLS → MLP.
  - *Late* — `[image_CLS ‖ text_CLS]` → MLP.
  - *FiLM* — text generates per-feature `(γ, β)`; modulated image → MLP.
  - *Gated* — sigmoid gate `g = σ(W[image‖text])`; `g·image + (1-g)·text` → MLP.
  - *Cross-attention* — text tokens (Q) over image patches (K, V), residual,
    LayerNorm, mean-pool, MLP. Direction is **inverted for segmentation**
    (image patches as Q, text tokens as K, V).
- **Captions:** five LLM-generated variants from the dataset
  (`hybrid_gemma3-4b`, `hybrid_qwen3-vl-8b`, `text_qwen3-4b`,
  `vision_gemma3-4b`, `vision_qwen3-vl-8b`) covering a spectrum from
  composition-leaky to image-only (and noisy).

## Acknowledgments

Course project for **DI 725 — Transformers and Attention-Based Deep Networks**,
Middle East Technical University, Graduate School of Informatics. The
ARAS400K dataset is the work of M. Caglar (2026, [Zenodo](https://zenodo.org/records/18890661)).
RemoteCLIP weights from Liu et al., 2024 (HuggingFace
[`chendelong/RemoteCLIP`](https://huggingface.co/chendelong/RemoteCLIP)).
