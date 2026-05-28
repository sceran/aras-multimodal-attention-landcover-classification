# ARAS Multi-Modal Attention Fusion for Land-Cover Classification and Segmentation

Comparing four cross-modal fusion strategies (late concatenation, FiLM, gated,
and cross-attention) for **caption-guided multi-label land-cover classification
and patch-level segmentation** of remote-sensing imagery, using a 10,024-image
subset of the [ARAS400K dataset](https://zenodo.org/records/18890661) and a
frozen [RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP) backbone with
five LLM-generated caption strategies. Course project for **DI 725 —
Transformers and Attention-Based Deep Networks**, METU Graduate School of
Informatics, Spring 2026.

## Key findings (3-seed mean ± std)

- **Cross-attention (CA) consistently outperforms late fusion** on every
  caption type, every seed, every threshold, and on both tasks. Best Path A:
  `cross-attention + hybrid_qwen3-vl-8b` → **mAP 0.948 ± 0.003** vs image-only
  0.817 ± 0.001 (Δ +0.131). Best Path B segmentation: **mIoU 0.711 ± 0.003**
  vs image-only 0.574 ± 0.003 (Δ +0.137). Cross-task fusion gain ≈ +0.13 in
  both tasks.
- **CA is robust to noisy text.** Vision-only LLM captions hallucinate on this
  subset (describing 92%-grass scenes as "barren"); late fusion gains nothing
  over image-only (mAP 0.815 ≈ 0.817), but CA recovers **+0.049 mAP**.
- **Fusion-mechanism ranking** (mean across captions): CA 0.908 ≈ FiLM 0.903 >
  Late 0.864 ≈ Gated 0.857. **FiLM slightly exceeds CA at τ=5%**
  (0.944 vs 0.938), so we do not claim CA dominates FiLM unconditionally.
  **Gated underperforms even simple late concatenation** — a reported
  negative result.
- **Threshold sensitivity (Phase-3 ablation):** CA-vs-baseline gap grows
  monotonically with τ, from +0.097 at τ=5% to +0.153 at τ=20%.
- **Backbone-source ablation (Phase-2):** RemoteCLIP > vanilla CLIP on 11/11
  conditions, avg Δ mAP +0.031. Strong fusion partly compensates for a weaker
  backbone.
- **Backbone-capacity ablation (Phase-3, severely affects results):**
  Replacing ViT-B/32 with the larger, higher-resolution **ViT-L/14**
  *decreases* segmentation mIoU by 0.041 (best CA 0.711 → 0.670). The frozen
  backbone cannot exploit the extra capacity for patch-level prediction;
  partial fine-tuning (e.g. LoRA) is a natural next step.
- **Caption leakage caveat:** the hybrid and text-only captions encode the
  same composition percentages that define the multi-label targets. Results
  in those settings should be read as oracle-text upper bounds rather than
  fully independent language supervision. The vision-only-caption results
  provide the cleaner test of caption-guided fusion.

## Repository layout

```
di725_phase2.ipynb          # Phase-2 notebook (outputs visible)
cells/                      # Phase-2 source-of-truth cell scripts
report/                     # Phase-2 IEEE 2-page report (LaTeX + PDF + figures)
phase3/                     # Phase-3 final deliverables
├── di725_phase3.ipynb              # Phase-3 notebook (outputs visible)
├── requirements.txt                # Pinned environment
├── report/
│   ├── phase3_report.tex
│   ├── di725_phase3_report.pdf     # 4-page IEEE report
│   └── figures/                    # 4 publication figures, 300 dpi, B&W readable
└── results/                        # Aggregated JSON outputs
    ├── tau_ablation.json
    ├── seg_full_b32.json
    ├── pathA_l14.json
    ├── seg_l14.json
    └── pathA_*.json, pathB_*.json  # Phase-2 reference aggregates
```

## Tracked experiments (Weights & Biases)

All ~500 training runs are public. Each run has per-epoch metrics and tags by
seed / fusion type / caption strategy.

| Project | Contents |
|---|---|
| [di725-phase2-sanity](https://wandb.ai/sceran/di725-phase2-sanity) | 8 sanity baselines used to motivate the multi-label task |
| [di725-phase2-main](https://wandb.ai/sceran/di725-phase2-main) | Path A: 33 multi-seed (image-only / late / CA × 5 captions) + 30 fusion variants (FiLM, Gated) |
| [di725-phase2-backbone-vanilla](https://wandb.ai/sceran/di725-phase2-backbone-vanilla) | Vanilla CLIP backbone ablation, 33 runs |
| [di725-phase2-seg](https://wandb.ai/sceran/di725-phase2-seg) | Path B preliminary segmentation, 21 runs |
| [di725-phase3-tau](https://wandb.ai/sceran/di725-phase3-tau) | Threshold sensitivity, 4 fusion × 5 caption × 3 τ × 3 seed = 189 runs |
| [di725-phase3-seg-full](https://wandb.ai/sceran/di725-phase3-seg-full) | Full B/32 segmentation matrix, 21 cond × 3 seed = 63 runs |
| [di725-phase3-l14-pathA](https://wandb.ai/sceran/di725-phase3-l14-pathA) | Path A on ViT-L/14, 21 cond × 3 seed = 63 runs |
| [di725-phase3-l14-seg](https://wandb.ai/sceran/di725-phase3-l14-seg) | Path B on ViT-L/14, 21 cond × 3 seed = 63 runs |

## Reproducing the experiments

1. **Environment.** `pip install -r phase3/requirements.txt`. Tested on Colab
   Pro with A100; CPU fallback works for small batches.
2. **Data.** Download the
   [ARAS400K validation subset](https://zenodo.org/records/18890661) (10,024
   images) and place `images/`, `masks/`, and `captions.csv` under a single
   root directory. Update the `data_root` entry in the notebook's `CONFIG`
   dict.
3. **Run order (Phase-3 notebook).** Setup → `01 Tau Ablation` →
   `02 Seg Full Matrix` → `03A Extract L/14 Features` → `03B Path A L/14` →
   `03C Seg L/14` → `04 Example Samples` → `05 Phase3 Figures`. Features are
   cached after a single backbone pass, so subsequent sweeps run on the cache.

## Method summary

- **Backbone:** RemoteCLIP-ViT-B/32 (main) and ViT-L/14 (ablation), both
  frozen. Per-token features extracted via forward hooks on `ln_post` /
  `ln_final`.
- **Task A (multi-label classification):** `composition_pct(class_i) ≥ τ%` →
  7-bit binary vector; default τ=10%. BCE loss, mAP and per-class F1.
- **Task B (patch-level segmentation):** majority-pool the mask onto the patch
  grid (7×7 at B/32, 16×16 at L/14); per-patch cross-entropy, mIoU and
  per-class IoU.
- **Fusion modules** (all share a 256-unit ReLU–Dropout–Linear head, AdamW
  lr=1e-3, 30 epochs):
  - **Image-only** — image CLS → MLP.
  - **Late** — `[image_CLS ‖ text_pooled]` → MLP.
  - **FiLM** — text generates per-feature `(γ, β)` that modulate image CLS;
    final modulator layer zero-initialised so the model starts at the
    image-only baseline.
  - **Gated** — sigmoid gate `g = σ(W[image ‖ text])`, fused = `g·image + (1-g)·text`.
  - **Cross-attention** — text tokens (Q) over image patches (K, V) for
    classification; direction inverted for segmentation (image patches Q over
    text tokens K, V) so each patch produces its own logits.
- **Captions:** five LLM-generated streams covering a leakage / quality
  spectrum: `hybrid_gemma3-4b`, `hybrid_qwen3-vl-8b`, `text_qwen3-4b`,
  `vision_gemma3-4b`, `vision_qwen3-vl-8b`.

## Acknowledgments

The ARAS400K dataset is the work of M. Caglar (2026,
[Zenodo](https://zenodo.org/records/18890661)). RemoteCLIP weights from Liu
et al., 2024 (HuggingFace
[`chendelong/RemoteCLIP`](https://huggingface.co/chendelong/RemoteCLIP)).

GitHub: <https://github.com/sceran/aras-multimodal-attention-landcover-classification>
