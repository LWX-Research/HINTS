# HINTS — 3D Multimodal MRI Survival Analysis

Hierarchically Disentangling Subregional Heterogeneity with Structural Priors, applied to 3D brain tumor MRI (BraTS, NPC) for survival prediction.

## Overview

HINTS builds **modality-specific graphs** from 3D supervoxel regions within the tumor ROI, then learns via a **prototype-based disentanglement** that separates shared and modality-specific features. A **quaternion fusion** with QSVD captures high-order interactions across modalities before final survival prediction.

### Pipeline

```
3D MRI (T1/T1c/T2) → ROI crop → 3D SLIC supervoxels → per-modality graphs
    → SubregionEncoder (GCN) → PrototypeDisentangler → QuaternionFusion (QSVD)
    → SurvivalPredictionHead → risk score
```

Key design choices:
- All three modalities share the same supervoxel topology (single SLIC run on stacked volumes)
- 6-dimensional node features: `(cz, cy, cx, voxel_count, mean, std)`
- 6-neighborhood edges (Z, Y, X axes)
- Contrastive JSD loss pushes shared and modality-specific features apart

## Installation

```bash
pip install -r requirements.txt
```

> **Note**: `torch` and `torch-geometric` wheels depend on your CUDA version. The versions in `requirements.txt` target CUDA 12.1. If your environment differs, install PyTorch first following the [official guide](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.

## Quick Start

### 1. Convert raw data to unified format

Supported sources: BraTS NIfTI, NPC NIfTI, NPC NPY.

```bash
# BraTS 2020
python -m hints.cli convert \
  --raw-dir /path/to/MICCAI_BraTS2020_TrainingData \
  --output-dir /path/to/unified-brats

# NPC NIfTI
python -m hints.cli convert \
  --raw-dir /path/to/NPC-1352/nii \
  --output-dir /path/to/unified-npc \
  --source-type npc_nifti
```

**Unified format** structure:
```
unified_root/
  survival.csv        # MRN, dead, deadtime (months)
  {case_id}/
    t1.npy            # (D,H,W) float32
    t1c.npy           # (D,H,W) float32
    t2.npy            # (D,H,W) float32
    mask.npy          # (D,H,W) bool — tumor ROI
```

### 2. Train

Online mode (build graphs on-the-fly from unified format, recommended):

```bash
python -m hints.cli train \
  --unified-dir /path/to/unified-brats \
  --epochs 50 --batch-size 4 --num-workers 8
```

Offline mode (pre-compute graphs first, then train from cache):

```bash
# Step A: preprocess
python -m hints.cli preprocess-3d \
  --unified-dir /path/to/unified-brats \
  --output-dir /path/to/Hints-3D-BraTS \
  --n-segments 200 --num-nodes 128

# Step B: train from cached graphs
HINTS_PREPROCESS_OUTPUT_DIR=/path/to/Hints-3D-BraTS \
  python -m hints.cli train --epochs 50 --batch-size 4
```

### 3. Evaluate

```bash
python -m hints.cli eval --fold 1 \
  --unified-dir /path/to/unified-brats \
  --pretrained-path /path/to/checkpoint.pth
```

## CLI Reference

```
python -m hints.cli {command} [options]
```

| Command | Description |
|---------|-------------|
| `convert` | Convert raw dataset → unified canonical format |
| `train` | Run K-fold cross-validation training |
| `eval` | Evaluate a single fold with pretrained weights |
| `preprocess-3d` | Offline 3D graph preprocessing (cache .pt files) |

### `convert`

| Option | Description |
|--------|-------------|
| `--raw-dir` | Raw dataset directory (BraTS NIfTI / NPC NIfTI / NPC NPY) |
| `--output-dir` | Output unified directory |
| `--source-type` | `npc_nifti` / `npc_npy` / `brats` (auto-detected if omitted) |
| `--case-id` | Convert a single case |
| `--max-cases` | Limit number of cases |

### `train` / `eval`

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--num-workers` | 4 | DataLoader workers |
| `--k-fold` | 5 | K-fold splits |
| `--eval-interval` | 10 | Evaluate every N epochs |
| `--unified-dir` | — | Unified format directory (online mode) |
| `--preprocess-output-dir` | — | Precomputed graphs directory (offline mode) |
| `--pretrained-path` | — | Path to pretrained checkpoint |
| `--num-nodes` | 128 | Graph node capacity |
| `--n-segments` | 200 | Target supervoxel count |
| `--hidden-dim` | 32 | GCN hidden dimension |
| `--lr` | 0.01 | Learning rate (SGD) |
| `--contra-weight` | 1.0 | Contrastive loss weight |
| `--graph-build-mode` | auto | `offline` / `online` / `auto` |
| `--seed` | 111111 | Random seed |

### `preprocess-3d`

| Option | Default | Description |
|--------|---------|-------------|
| `--unified-dir` | (required) | Input unified format directory |
| `--output-dir` | (required) | Output directory for .pt graphs |
| `--n-segments` | 200 | Target supervoxel count |
| `--num-nodes` | n_segments | Graph node capacity |
| `--roi-margin` | 5 | ROI bounding box margin (voxels) |
| `--compactness` | 0.1 | SLIC compactness |
| `--sigma` | 1.0 | SLIC sigma |
| `--overwrite` | false | Overwrite existing graphs |

## Environment Variables

All settings are overridable via environment variables (priority: CLI args > env vars > defaults).

| Variable | Default | Description |
|----------|---------|-------------|
| `HINTS_UNIFIED_DIR` | — | Unified format directory |
| `HINTS_PREPROCESS_OUTPUT_DIR` | `../dataset/Hints-3D-BraTS` | Offline graph output |
| `HINTS_EPOCHS` | 50 | Training epochs |
| `HINTS_BATCH_SIZE` | 4 | Batch size |
| `HINTS_NUM_WORKERS` | 4 | DataLoader workers |
| `HINTS_K_FOLD` | 5 | K-fold splits |
| `HINTS_SEED` | 111111 | Random seed |
| `HINTS_LR` | 0.01 | Learning rate |
| `HINTS_N_SEGMENTS` | 200 | Supervoxel target count |
| `HINTS_NUM_NODES` | 128 | Graph node capacity |
| `HINTS_NUM_PROTOTYPES` | 3 | Prototypes per modality |
| `HINTS_CONTRA_WEIGHT` | 1.0 | Contrastive loss weight |
| `HINTS_HIDDEN_DIM` | 32 | GCN hidden dimension |
| `HINTS_ROI_MARGIN` | 5 | ROI margin (voxels) |
| `HINTS_SLIC_COMPACTNESS` | 0.1 | SLIC compactness |
| `HINTS_SLIC_SIGMA` | 1.0 | SLIC sigma |
| `HINTS_GRAPH_BUILD_MODE` | auto | `offline` / `online` / `auto` |
| `HINTS_EVAL_INTERVAL` | 10 | Evaluate every N epochs |
| `HINTS_PRETRAINED_PATH` | — | Pretrained weights path |
| `HINTS_TAU` | 0.01 | Contrastive JSD temperature |
| `HINTS_ETA_MIN` | 0.1 | LR scheduler min ratio |

## Model Architecture

```
MultiModalGCN
├── SubregionEncoder        # 3× independent GCN branches (t1, t1c, t2)
│   └── 2-layer GCNConv per modality + BatchNorm
├── PrototypeDisentangler   # Learnable prototypes → shared + specific features
│   └── Cross-attention + momentum update + MLP pooling
├── StructuralQuaternionFusion  # Quaternion mapping → QSVD → attention fusion
│   └── Hamilton matrix construction → truncated SVD reconstruction
└── SurvivalPredictionHead  # MLP: dim → 1024 → 128 → 1
```

**Loss**: `total = Cox PH loss + λ₁·L1_reg + contra_weight · Contrastive JSD loss`

**Metrics**: Concordance Index (C-index), Time-dependent AUC

## Tuning Guide

Recommended sweep order for improving performance:

1. **Supervoxel parameters**: `HINTS_N_SEGMENTS` → `HINTS_NUM_NODES` → `HINTS_SLIC_COMPACTNESS` → `HINTS_SLIC_SIGMA` → `HINTS_ROI_MARGIN`
2. **Training hyperparams**: `lr` → `batch_size` → `contra_weight` → `num_prototypes`

Change 1–2 parameters at a time; record Val CI, Val AUC, node statistics, and GPU memory.

## License

This project is for research purposes. See the paper for details.
