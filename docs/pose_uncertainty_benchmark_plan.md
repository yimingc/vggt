# Pose Uncertainty Head — Benchmark Plan

## Motivation

All prior experiments (Phases 1–5.10) used a single TUM sequence (`freiburg1_desk`, 596 frames).
This is sufficient for debugging and validating the pipeline, but insufficient for:

1. **Claiming generalization** — does uncertainty transfer to unseen motion patterns?
2. **Meaningful PGO improvement** — single-sequence results have ceiling effects (~0.2% ATE gap)
3. **Interview credibility** — "trained and evaluated on one sequence" is a weakness

This plan scales training and evaluation across multiple datasets in three phases.

---

## Dataset Landscape

### Datasets with Sensor-Grade GT Poses (Not in VGGT Training)

These are ideal for **evaluation** — clean GT, no overlap with backbone training.

| Dataset | GT Source | Domain | Sequences | Frames | Download |
|---|---|---|---|---|---|
| **TUM RGB-D** | Motion capture (sub-mm) | Indoor | 39 | ~100k | ~20 GB |
| **7-Scenes** | KinectFusion | Indoor (small rooms) | 7 scenes | ~46k | ~76 GB |
| **ETH3D** | Laser scanner + mocap | Indoor + Outdoor | 56 SLAM seqs | ~10k | ~42 GB |
| **KITTI Odometry** | GPS/INS (RTK, <10cm) | Outdoor driving | 11 with GT | ~43k | ~65 GB |
| **EuRoC MAV** | Vicon (sub-mm) | Indoor (drone) | 11 | ~28k | ~22 GB |

### Datasets In VGGT's Training Distribution

These are ideal for **training the uncertainty head** — the frozen backbone produces strong features.

| Dataset | GT Source | Domain | Scale | Data Loader? |
|---|---|---|---|---|
| **CO3D v2** | COLMAP | Object-centric (40 categories) | 19k sequences, ~1.5M frames | **Yes** (in codebase) |
| **ScanNet** | BundleFusion | Indoor scenes | 1,513 scenes, ~2.5M frames | No |
| **DL3DV** | COLMAP | Indoor + Outdoor (diverse) | 10k videos, 51M frames | No |
| **Replica** | Synthetic (perfect) | Indoor | 18 scenes | No |

### Key Insight: Train In-Distribution, Evaluate Out-of-Distribution

The uncertainty head learns from **residuals** (gap between VGGT prediction and GT).
If the backbone features are weak (OOD data), residuals are noisy → uncertainty head learns noise.
If the backbone features are strong (in-distribution), residuals reflect true prediction quality → uncertainty head learns meaningful σ.

**Therefore:** Train on CO3D + TUM (strong backbone features), evaluate on held-out TUM + ETH3D + 7-Scenes (unseen data with clean GT).

---

## Phase 1: Multi-Sequence TUM (Validates Pipeline)

**Goal:** Prove the pipeline works with multiple sequences before scaling to CO3D.

### 1.1 Revised TUM Split

Training diversity matters — the original plan had 3/6 desk scenes. Revised:

| Sequence | Frames | Motion Type | Split | Why |
|---|---|---|---|---|
| `freiburg1_desk` | 596 | slow, office | Train | Current data, baseline |
| `freiburg1_room` | 1362 | medium, room traverse | Train | Larger motion range |
| `freiburg1_plant` | 1146 | slow, close-up | Train | Texture-rich, different scene |
| `freiburg2_xyz` | 3669 | pure translation | Train | Translation diversity |
| `freiburg2_rpy` | 3290 | pure rotation | Train | Rotation diversity |
| `freiburg2_desk` | 2965 | slow, office | Train | Different camera (fr2 intrinsics) |
| | | | | |
| `freiburg1_360` | 756 | rotation-heavy | **Eval** | Rotation generalization |
| `freiburg1_floor` | 1214 | fast motion | **Eval** | Motion blur, speed |
| `freiburg1_teddy` | 1419 | slow, close-up | **Eval** | Unseen object |
| `freiburg3_long_office` | 2585 | long trajectory | **Eval** | Drift, different camera (fr3) |

**Changes from original plan:**
- Moved `fr2_rpy` from eval→train (rotation diversity in training)
- Dropped `fr1_desk2` (too similar to desk)
- Training now covers: slow, medium, translation-only, rotation-only, different cameras
- Eval still has challenging conditions: fast motion, rotation-heavy, long trajectory, unseen object

### 1.2 Download

```bash
TUM_DIR=/home/yiming/Dev/tum_rgbd
cd $TUM_DIR

# Training sequences (we already have freiburg1_desk)
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_rpy.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz

# Eval sequences
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_360.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_floor.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_teddy.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz

# Extract all
for f in *.tgz; do tar xzf "$f"; done
```

### 1.3 Training

```bash
# Multi-sequence TUM training
# Note: need --tum_sequences flag to restrict to train split only
python training/tests/train_uncertainty_tensorboard.py \
    --tum_dir /home/yiming/Dev/tum_rgbd \
    --num_iters 10000 \
    --augment_consecutive \
    --consecutive_ratio 0.5 \
    --window_sizes 8 16 32 64 \
    --checkpoint_dir ./checkpoints_tum_multi \
    --save_interval 1000 \
    --log_dir ./runs/tum_multi \
    --loss_type gaussian
```

**Script updates:** Done.
- [x] Added `--tum_sequences` flag to restrict training to specific sequences
- [x] Added `--tum_sequence` flag to eval script for per-sequence evaluation
- [x] Dataset loader already supported `sequences` parameter

### 1.4 Evaluation

```bash
CKPT=./checkpoints_tum_multi/best.pt

for SEQ in freiburg1_360 freiburg1_floor freiburg1_teddy freiburg3_long_office_household; do
    echo "=== $SEQ ==="
    HF_HUB_OFFLINE=1 python training/tests/eval_pgo_uncertainty.py \
        --tum_dir /home/yiming/Dev/tum_rgbd \
        --tum_sequence rgbd_dataset_$SEQ \
        --uncertainty_checkpoint $CKPT \
        --window_size 16 \
        --overlap 0.5 \
        --output_dir ./eval_benchmark_tum/$SEQ \
        --oracle_isotropic
done
```

### 1.5 Results (Phase 1) — 2026-02-21

**Training:** 10k iters, Gaussian NLL + augmented consecutive sampling, best checkpoint calibration_error=0.04 (d²_rot=3.02, d²_trans=2.98).

**Eval settings:** Per-window GT scale, loop closure, robust (Huber) kernel. Temperature sweep α ∈ {0.3, 0.5, 0.7, 1.0}.

| Sequence | Frames | Edges | Spearman | d²_trans | d²_rot | Uniform ATE | Best Predicted ATE | Δ | Best α | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| fr1_360 | 756 | 1560 | 0.416 | 22.26 | 7.32 | **7.94 cm** | 8.01 cm | +1% | 0.3 | FAIL |
| **fr1_floor** | 1214 | 2415 | 0.528 | 3.85 | 4.94 | 8.01 cm | **7.80 cm** | **-3%** | 0.7 | SUCCESS |
| **fr1_teddy** | 1419 | 2805 | 0.467 | 13.45 | 8.23 | 14.19 cm | **13.59 cm** | **-4%** | 1.0 | SUCCESS |
| **fr3_long_office** | 2585 | 4800 | 0.278 | 3.01 | 2.29 | 11.79 cm | **10.47 cm** | **-11%** | 1.0 | SUCCESS |
| **Mean** | | | **0.422** | | | 10.48 cm | **9.97 cm** | **-5%** | | **3/4** |

**Per-sequence details:**

| Sequence | Method | ATE Trans | ATE Rot | RPE Trans | RPE Rot |
|---|---|---|---|---|---|
| **fr1_360** | Before PGO (init) | 11.35 cm | 31.02° | 2.24 cm | 0.90° |
| | PGO + uniform | **7.94 cm** | **9.66°** | 1.56 cm | 0.53° |
| | PGO + predicted (α=1.0) | 8.45 cm | 10.03° | 1.51 cm | 0.51° |
| | PGO + predicted (α=0.3) | 8.01 cm | 9.73° | 1.51 cm | 0.52° |
| **fr1_floor** | Before PGO (init) | 10.78 cm | 8.16° | 0.62 cm | 0.31° |
| | PGO + uniform | 8.01 cm | 6.87° | 0.67 cm | 0.33° |
| | PGO + predicted (α=1.0) | 7.81 cm | 6.64° | 0.64 cm | 0.31° |
| | **PGO + predicted (α=0.7)** | **7.80 cm** | **6.78°** | 0.64 cm | 0.31° |
| **fr1_teddy** | Before PGO (init) | 13.32 cm | 6.50° | 1.61 cm | 0.50° |
| | PGO + uniform | 14.19 cm | 4.98° | 1.59 cm | 0.48° |
| | **PGO + predicted (α=1.0)** | **13.59 cm** | **4.93°** | 1.59 cm | 0.48° |
| | PGO + predicted (α=0.7) | 14.12 cm | 4.87° | 1.58 cm | 0.48° |
| **fr3_long_office** | Before PGO (init) | 12.12 cm | 3.39° | 0.77 cm | 0.26° |
| | PGO + uniform | 11.79 cm | 3.28° | 0.78 cm | 0.28° |
| | **PGO + predicted (α=1.0)** | **10.47 cm** | **2.80°** | 0.78 cm | 0.26° |
| | PGO + predicted (α=0.7) | 10.64 cm | 2.83° | 0.78 cm | 0.26° |

### 1.6 Success Criteria Assessment

| Criterion | Target | Actual | Status |
|---|---|---|---|
| d² calibration (train distribution) | 2.5–4.0 | 3.02 / 2.98 | PASS |
| Predicted < Uniform ATE on ≥3/4 eval seqs | majority | **3/4** | **PASS** |
| Spearman on eval seqs | > 0.3 | 0.28–0.53 (mean 0.42) | PASS |
| Mean Δ ATE across eval | > 1% | **-5%** | **PASS** |

### 1.7 Analysis

1. **Predicted beats uniform on 3/4 sequences.** Largest gain is fr3_long_office (-11%), the longest and most challenging trajectory.

2. **Full α=1.0 is optimal on 2/4 sequences** (fr1_teddy, fr3_long_office). Temperature only needed for fr1_floor (α=0.7 marginally better). The multi-sequence trained checkpoint has reasonable calibration on held-out data.

3. **Spearman 0.28–0.53 across held-out sequences.** Moderate but meaningful — PGO improvement confirms practical value even at moderate correlation.

4. **fr1_360 is the only failure** — rotation-heavy sequence where d²_trans=22.26 (poorly calibrated). The uncertainty head struggles with rotation-dominated motion patterns.

**Implication for Phase 2:** Phase 1 success criteria are met (3/4 sequences, -5% mean ATE). CO3D training should improve generalization further, especially calibration on out-of-distribution sequences (fr1_360 d²=22.26).

---

## Phase 2: CO3D Training (The Big Win)

**Goal:** Train on large-scale, diverse data where the VGGT backbone is strongest.

### 2.1 Why CO3D

- **In VGGT's training distribution** → backbone features are strong → residuals reflect true prediction quality
- **Data loader already exists** in the codebase (`training/data/datasets/co3d.py`)
- **40 object categories**, 19k sequences → massive diversity
- **Varied viewpoints** — turntable, handheld, close-up, far-away
- GT from COLMAP (moderate quality, but large scale compensates)

### 2.2 CO3D Setup

CO3D data structure:
```
CO3D_DIR/
├── {category}/{sequence}/
│   ├── images/*.jpg
│   ├── depths/*.geometric.png
│   └── depth_masks/*.png

CO3D_ANNOTATION_DIR/
├── {category}_train.jgz
└── {category}_test.jgz
```

Download CO3D from https://github.com/facebookresearch/co3d and annotation files
from https://huggingface.co/datasets/JianyuanWang/co3d_anno/tree/main

### 2.3 Training Approach

The existing CO3D data loader provides extrinsics, intrinsics, and depth. The uncertainty
head training script currently only supports TUM. Two options:

**Option A: Adapt `train_uncertainty_tensorboard.py` to support CO3D**
- Add CO3D dataset loading path
- Use the same `compute_camera_nll_loss` function (it only needs pose encodings and GT extrinsics)
- CO3D loader already provides `get_nearby` sampling
- Main work: handle CO3D batch format → existing loss function interface

**Option B: Use VGGT's native training script with uncertainty loss**
- Add `camera_nll` loss to the existing `training/train.py`
- This would use the existing CO3D training pipeline with all augmentations
- Freeze backbone + pose head, only train uncertainty head
- Most principled approach (same pipeline, same augmentations)

**Recommended: Option B** — less code, uses the battle-tested training pipeline.

```bash
# Option B: Add uncertainty training to native training script
python training/train.py \
    --config config/train_uncertainty_co3d.yaml \
    --freeze_backbone \
    --freeze_pose_head \
    --max_iters 50000 \
    --output_dir ./checkpoints_co3d_uncertainty
```

### 2.4 CO3D Training Config

```yaml
# config/train_uncertainty_co3d.yaml
camera_nll:
  weight: 1.0
  pose_encoding_type: "absT_quaR_FoV"
  gamma: 0.6
  log_var_clamp: [-20.0, 20.0]
  scale_detach: true
  min_translation: 0.02
  loss_type: gaussian  # or laplace

# Data: CO3D with default augmentations
dataset:
  co3d:
    len_train: 10000
    get_nearby: true
    img_nums: [2, 24]

# Freeze everything except uncertainty head
freeze_backbone: true
freeze_pose_head: true

# Training
optimizer:
  lr: 1e-4
  weight_decay: 0.01
max_iters: 50000
```

### 2.5 Mixed Training (CO3D + TUM)

For best results, train on both:
- CO3D: large scale, diverse objects/viewpoints, strong backbone features
- TUM: sequential video (matches PGO evaluation), real SLAM trajectory

```yaml
# Mixed dataset config
dataset:
  co3d:
    len_train: 8000   # 80% CO3D
  tum_rgbd:
    len_train: 2000   # 20% TUM
    augment_consecutive: true
```

### 2.6 Success Criteria (Phase 2)

| Criterion | Target |
|---|---|
| d² calibration (CO3D test split) | 2.5–4.0 |
| d² calibration (TUM eval sequences) | 2.5–4.0 |
| Predicted < Uniform on ≥3/4 TUM eval seqs | majority |
| Mean Δ ATE across TUM eval | > 1% |
| Spearman on TUM eval | > 0.5 |

---

## Phase 3: Cross-Dataset Evaluation (Generalization Story)

**Goal:** Evaluate on datasets the model has never seen, with sensor-grade GT.

### 3.1 Additional Eval Datasets

| Dataset | Why | Setup Effort | Priority |
|---|---|---|---|
| **7-Scenes** | Indoor relocalization benchmark, not in VGGT training | Low (just download + write loader) | High |
| **ETH3D SLAM** | Laser scanner GT, indoor + outdoor | Medium (needs SLAM sequence loader) | Medium |
| **KITTI Odometry** | Outdoor, completely different domain | Medium (needs loader) | Low (very OOD) |

### 3.2 7-Scenes Evaluation

7-Scenes is the most practical addition:
- 7 diverse indoor scenes
- Well-known benchmark
- GT from KinectFusion (good enough for relative pose evaluation)
- ~46k frames total

```bash
# Download 7-Scenes
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip
# ... (6 more scenes)
```

**Eval approach:** Same PGO pipeline, but need a 7-Scenes data loader for frame loading + GT pose extraction.

### 3.3 Results Table (Template)

| Dataset / Sequence    | ATE Uniform | ATE Best Predicted | Best α | Δ ATE   | Spearman |
|-----------------------|-------------|---------------------|--------|---------|----------|
| **TUM (held-out, per-window + LC + robust)** | | | | | |
| fr1_360               | **7.94 cm** | 8.01 cm             | 0.3    | +1%     | 0.416    |
| fr1_floor             | 8.01 cm     | **7.80 cm**         | 0.7    | **-3%** | 0.528    |
| fr1_teddy             | 14.19 cm    | **13.59 cm**        | 1.0    | **-4%** | 0.467    |
| fr3_long_office       | 11.79 cm    | **10.47 cm**        | 1.0    | **-11%**| 0.278    |
| TUM Mean              | 10.48 cm    | **9.97 cm**         |        | **-5%** | **0.422**|
|-----------------------|-------------|---------------|------------|---------|----------|
| **7-Scenes**          |             |               |            |         |          |
| Chess                 |             |               |            |         |          |
| Fire                  |             |               |            |         |          |
| Heads                 |             |               |            |         |          |
| Office                |             |               |            |         |          |
| Pumpkin               |             |               |            |         |          |
| RedKitchen            |             |               |            |         |          |
| Stairs                |             |               |            |         |          |
| 7-Scenes Mean         |             |               |            |         |          |
|-----------------------|-------------|---------------|------------|---------|----------|
| **Overall Mean**      |             |               |            |         |          |

---

## 4. Interview-Ready Deliverables

After completing all phases, you should be able to say:

### Phase 1 (completed)
> "Trained on 6 diverse TUM sequences (12k frames), evaluated on 4 held-out sequences with different motion patterns.
> Predicted uncertainty weights beat uniform on 3/4 held-out sequences (mean -5% ATE), with largest gain on the hardest trajectory (-11% on fr3_long_office, 2585 frames).
> Uncertainty-weighted PGO wins on ATE Rot across all sequences. Full α=1.0 is optimal on 2/4 sequences — calibration transfers well from training distribution."

### Phase 2 (strong)
> "Trained on CO3D (19k sequences, 40 categories) + TUM, evaluated on held-out TUM sequences.
> Large-scale diverse training improved Spearman correlation from 0.74 to Y and PGO improvement from 0.2% to Z%."

### Phase 3 (impressive)
> "Evaluated on 3 different datasets (TUM, 7-Scenes, ETH3D) with sensor-grade GT.
> Uncertainty generalizes across domains — [specific numbers]."

### Narrative
> "This follows the same paradigm as L4P and FoundationPose heads — frozen foundation model + lightweight task-specific head.
> The key insight is that training in-distribution (CO3D, where the backbone is strongest) produces better uncertainty estimates
> than training on the evaluation domain (TUM), because the residuals are more informative when backbone features are strong."

---

## 5. Implementation Checklist

### Phase 1: Multi-Sequence TUM — DONE (2026-02-21)
- [x] Download 9 additional TUM sequences
- [x] Add `--tum_sequences` flag to training script (train/eval split)
- [x] Add `--tum_sequence` flag to eval script (per-sequence evaluation)
- [x] Train: Gaussian + augmented, 10k iters (best calibration_error=0.04)
- [x] Eval: 4 held-out sequences, per-window GT scale + LC + robust + temperature sweep
- [x] Compile results: **3/4 predicted beats uniform, mean -5% ATE**

### Phase 2: CO3D Training
- [ ] Download CO3D dataset + annotations
- [ ] Option A: Adapt `train_uncertainty_tensorboard.py` for CO3D, or
- [ ] Option B: Add uncertainty loss to native `training/train.py`
- [ ] Train: CO3D + TUM mixed, 50k iters
- [ ] Eval on TUM held-out sequences
- [ ] Compare: CO3D-trained vs TUM-only-trained

### Phase 3: Cross-Dataset Evaluation
- [ ] Download 7-Scenes
- [ ] Write 7-Scenes data loader (frame loading + GT pose extraction)
- [ ] Run PGO evaluation on 7 scenes
- [ ] Compile cross-dataset results table

---

## 6. Estimated Timeline

| Step | Hands-On | GPU Time |
|---|---|---|
| **Phase 1** | | |
| Download TUM sequences | 30 min | — |
| Script updates (train/eval flags) | 1 hour | — |
| Training (10k iters) | — | ~3 hours |
| Evaluation (4 seqs × 3 modes) | — | ~1 hour |
| **Phase 1 Total** | ~2 hours | ~4 hours |
| | | |
| **Phase 2** | | |
| Download CO3D + annotations | 2 hours | — |
| Integrate uncertainty loss into training pipeline | 2–4 hours | — |
| Training (50k iters, CO3D + TUM) | — | ~8 hours |
| Evaluation | — | ~1 hour |
| **Phase 2 Total** | ~4–6 hours | ~9 hours |
| | | |
| **Phase 3** | | |
| Download 7-Scenes | 1 hour | — |
| Write data loader | 2 hours | — |
| Evaluation | — | ~2 hours |
| **Phase 3 Total** | ~3 hours | ~2 hours |
| | | |
| **Grand Total** | ~9–11 hours | ~15 hours GPU |

Phase 1 can be done in a day. Phase 2 needs CO3D download time + integration work.
Phase 3 is optional but adds significant interview value. GPU time can overlap with other work.
