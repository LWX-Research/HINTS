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

### 1. Download dataset

```bash
pip install modelscope
modelscope download --dataset wxl6519/BraTS2020
```

### 2. Convert raw data to unified format

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

### 3. Train

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

### 4. Evaluate

```bash
python -m hints.cli eval --fold 1 \
  --unified-dir /path/to/unified-brats \
  --pretrained-path /path/to/checkpoint.pth
```

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

## License

This project is for research purposes. See the paper for details.
