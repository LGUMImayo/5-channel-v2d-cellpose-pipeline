# V2d Cellpose Cell Detection Pipeline

End-to-end pipeline for NeuN+ cell detection in multiplex immunofluorescence brain tissue images using finetuned Cellpose.

## Pipeline Overview

```
Raw TIFFs + GT XMLs
        │
        ▼
┌─────────────────────┐
│ Stage 0: V0 Base    │  finetune_cellpose.py
│ Cellpose finetuning │  (Cellpose pretrained → V0 on NeuN+DAPI)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Stage 1: Bootstrap  │  bootstrap_cellpose_v2.py
│ V1 silver masks     │  (V0 inference on all 32 cases → silver masks)
│ + low-thresh merge  │  (merge default + low-threshold runs)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Stage 2: Pseudo     │  create_pseudo_masks.py
│ mask augmentation   │  (fill GT false negatives with intensity-guided
│                     │   pseudo masks → augmented training data)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Stage 3: V2d        │  train_v2d_pseudo_masks.py
│ finetuning          │  (V1-merged base → train on augmented masks)
│                     │  400 epochs, LR=1e-4, WD=0.1, batch=8
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Stage 4: Inference  │  run_v2d_inference_all.py
│ + Evaluation        │  Single-pass & TTA (24 passes)
│                     │  sweep_cellprob_threshold.py
│                     │  sweep_tta_min_votes.py
└─────────────────────┘
```

## Data Description

### Raw Input Data (32 cases)
- **15 LR C12.5 cases** — Leica scanner, 5-channel: [NeuN, CAMKII, PHF1, BEX1, DAPI]
- **5 MAYO C12.5 cases** — Mayo scanner, 5-channel: [NeuN, DAPI, CAMKII, PHF1, BEX1]
- **12 MAYO C12 cases** — Mayo scanner, same channel order as MAYO C12.5

Each case has:
- Multi-frame TIFF (5 fluorescence channels)
- CellCounter XML (manual NeuN+ cell annotations as ground truth)

**Note:** MAYO C12 cases have significantly weaker NeuN signal (CLAHE NeuN mean: 48–63) compared to LR and MAYO C12.5 cases (mean: 78–83), leading to lower detection performance.

### Channel Maps
```
LR:   Frame 0=NeuN, 1=CAMKII, 2=PHF1, 3=BEX1, 4=DAPI
MAYO: Frame 0=NeuN, 1=DAPI, 2=CAMKII, 3=PHF1, 4=BEX1
```

## Model Lineage

| Model | Base | Training Data | Notes |
|-------|------|---------------|-------|
| **V0 (base)** | Cellpose pretrained `cyto3` | Subset of cases, GT masks | Initial NeuN+DAPI finetuning |
| **V1 (finetuned)** | V0 | 26 train / 4 test, silver masks from V0 | Round 2 bootstrapping |
| **V1 (merged)** | V0 | Silver masks from V1 default + low-threshold merged | Better recall base |
| **V2d (final)** | V1-merged | 28 train / 4 test, augmented masks (silver + pseudo) | Best model, 400 epochs |

## Production Configuration

```python
# V2d Inference
model_path = "models/v2d_final/cellpose_finetuned_v2d"
cellprob_threshold = -0.5   # optimal from threshold sweep
diameter = 35
channels = [1, 2]           # NeuN=cytoplasm, DAPI=nucleus
preprocessing = CLAHE       # via prepare_cellpose_input_clahe()

# TTA (Test-Time Augmentation)
scales = [30, 35, 40]
intensity_augs = 8          # identity, bright+15/30%, contrast+30/60%,
                            # gamma 0.6/0.8, CLAHE clip=5.0
total_passes = 24           # 8 intensity × 3 scales
min_votes = 4               # optimal from sweep
nms_dist = 15               # NMS merge radius in pixels
```

## Performance Summary

### Single-Pass V2d (cellprob_threshold = -0.5)

| Group | Cases | Mean F1 | Mean Precision | Mean Recall |
|-------|-------|---------|----------------|-------------|
| LR C12.5 | 15 | 0.755 | 0.824 | 0.718 |
| MAYO C12.5 | 5 | 0.785 | 0.820 | 0.788 |
| MAYO C12 | 12 | 0.568 | 0.755 | 0.519 |
| **All** | **32** | **0.689** | **0.797** | **0.654** |

### TTA with min_votes=4 (early results, 8 LR cases)
- F1: 0.775 (+0.037 vs single-pass)
- Precision: 0.788, Recall: 0.793

## Directory Structure

```
v2d_pipeline_share/
├── README.md
├── code/                           # All pipeline Python scripts
│   ├── config.py                   # Data paths, channel maps, hyperparams
│   ├── data_utils.py               # TIFF I/O, XML parsing, data discovery
│   ├── finetune_cellpose.py        # V0 base model training
│   ├── bootstrap_cellpose_v2.py    # V1 bootstrapping + silver mask generation
│   ├── create_pseudo_masks.py      # Pseudo mask creation for V2d
│   ├── train_v2d_pseudo_masks.py   # V2d training script
│   ├── run_v2d_inference_all.py    # Inference + evaluation (single-pass & TTA)
│   ├── sweep_cellprob_threshold.py # Cellprob threshold optimization
│   ├── sweep_tta_min_votes.py      # TTA min_votes optimization
│   ├── stage1_detection.py         # Stage 1 detection module
│   ├── run_pipeline.py             # End-to-end Stage 1+2 pipeline
│   ├── visualize_preprocessing.py  # Preprocessing visualization
│   └── test_dry_run.sh             # Quick sanity check script
│
├── data/
│   ├── raw_tiffs/
│   │   ├── LR/                     # 15 LR cases (multi-frame TIFFs)
│   │   └── MAYO/                   # 17 MAYO cases (multi-frame TIFFs)
│   ├── ground_truth/
│   │   ├── LR/                     # CellCounter XMLs for LR cases
│   │   └── MAYO/                   # CellCounter XMLs for MAYO cases
│   ├── Cases counted.xlsx          # Case tracking spreadsheet
│   └── Counting Protocol.docx      # Manual counting protocol
│
├── intermediate/
│   ├── silver_masks_v1/            # V1 inference masks (default threshold)
│   │   └── {case}_img.npy, {case}_mask.npy  (32 × 2 = 64 files)
│   ├── silver_masks_v2_merged/     # V1 merged (default + lowthresh)
│   │   └── {case}_img.npy, {case}_mask.npy  (32 × 2 = 64 files)
│   ├── silver_masks_v2_lowthresh/  # V1 low-threshold inference
│   │   └── {case}_img.npy, {case}_mask.npy  (32 × 2 = 64 files)
│   └── augmented_masks/            # Final V2d training data
│       ├── augmented_masks_manifest.json  # Training manifest
│       └── {case}_augmented_masks.npy     # 32 augmented mask files
│
├── models/
│   ├── v0_base/                    # cellpose_finetuned_neun_dapi (582 MB)
│   ├── v1_finetuned/               # cellpose_finetuned_v2 + info JSON (581 MB)
│   ├── v1_merged/                  # cellpose_finetuned_v2 merged (582 MB)
│   └── v2d_final/                  # cellpose_finetuned_v2d (501 MB) ← BEST
│
└── results/
    ├── single_pass/                # V2d single-pass inference on all 32 cases
    │   └── {case}_centroids.npy, {case}_overlay.png, etc.
    ├── tta/                        # V2d TTA inference (24 passes)
    │   └── {case}_centroids.npy, {case}_raw_centroids.npy, etc.
    ├── threshold_sweep/            # Cellprob threshold sweep JSONs
    └── preprocessing_viz/          # CLAHE preprocessing visualizations
```

## How to Run

### Prerequisites
```bash
conda activate rhizonet
# Requires: cellpose, numpy, scipy, scikit-image, matplotlib, tifffile
```

### Inference on a single image
```bash
python code/run_v2d_inference_all.py  # runs all 32 cases
```

### TTA inference
```bash
python code/run_v2d_inference_all.py --tta
```

### Post-hoc min_votes sweep (after TTA)
```bash
python code/sweep_tta_min_votes.py --votes 2 3 4 5 6 8 10 12
```

### Threshold sweep
```bash
python code/sweep_cellprob_threshold.py
```

**Note:** The scripts reference absolute paths from the original workspace. Update paths in `config.py` to point to the new location before running.

## Key Design Decisions

1. **CLAHE preprocessing**: Applied to NeuN channel before Cellpose to normalize intensity variation across cases
2. **Pseudo mask augmentation**: Ground truth false negatives are filled with intensity-guided pseudo masks to improve recall during training
3. **Intensity-only TTA**: Geometric augmentations (flip/rotate) don't help for roughly circular cells; intensity variations directly address weak-NeuN scenarios
4. **cellprob_threshold = -0.5**: More permissive than Cellpose default (0.0), trades precision for recall — optimal for this application

## Environment
- Python 3.x (conda env: `rhizonet`)
- Cellpose 2.x
- SLURM cluster (GPU: A100-40GB)
