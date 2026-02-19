# Pose Uncertainty Head — Benchmark Plan

## Motivation

All prior experiments (Phases 1–5.10) used a single TUM sequence (`freiburg1_desk`, 596 frames).
This is sufficient for debugging and validating the pipeline, but insufficient for:

1. **Claiming generalization** — does uncertainty transfer to unseen motion patterns?
2. **Meaningful PGO improvement** — single-sequence results have ceiling effects (~0.2% ATE gap)
3. **Interview credibility** — "trained and evaluated on one sequence" is a weakness

This plan scales training and evaluation to multiple sequences with proper train/eval splits.

---

## 1. Data

### 1.1 TUM RGB-D Sequences

Download from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download

| Sequence | Frames | Motion Type | Split | Notes |
|---|---|---|---|---|
| `freiburg1_desk` | 596 | slow, office | Train | Current data |
| `freiburg1_desk2` | 620 | slow, office | Train | Similar to desk |
| `freiburg1_room` | 1362 | medium, room traverse | Train | Larger motion |
| `freiburg1_plant` | 1146 | slow, close-up | Train | Texture-rich |
| `freiburg2_desk` | 2965 | slow, office | Train | Different camera (fr2) |
| `freiburg2_xyz` | 3669 | pure translation | Train | Tests translation uncertainty |
| `freiburg1_360` | 756 | rotation-heavy | **Eval** | Tests rotation uncertainty |
| `freiburg1_floor` | 1214 | fast motion | **Eval** | Motion blur, challenging |
| `freiburg1_teddy` | 1419 | slow, close-up | **Eval** | Different object |
| `freiburg2_rpy` | 3290 | pure rotation | **Eval** | Tests rotation-only |
| `freiburg3_long_office` | 2585 | long trajectory | **Eval** | Tests drift |

**Train:** 6 sequences (~10k frames) — diverse motion types and cameras
**Eval:** 5 sequences (~9k frames) — held out, includes challenging conditions

### 1.2 Download Commands

```bash
TUM_DIR=/home/yiming/Dev/tum_rgbd
cd $TUM_DIR

# Training sequences
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz

# Eval sequences
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_360.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_floor.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_teddy.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_rpy.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz

# Extract all
for f in *.tgz; do tar xzf "$f"; done
```

### 1.3 Verification

```bash
# Should see 11 sequences (6 train + 5 eval + the original desk we already have = 11 total, but desk is in train)
ls -d $TUM_DIR/rgbd_dataset_freiburg*/ | wc -l  # expect: 11

# Each should have rgb.txt, depth.txt, groundtruth.txt
for d in $TUM_DIR/rgbd_dataset_freiburg*/; do
    echo "$(basename $d): $(ls $d/{rgb,depth,groundtruth}.txt 2>/dev/null | wc -l)/3 files"
done
```

---

## 2. Training

### 2.1 Training Configuration

```bash
# Multi-sequence augmented training
python training/tests/train_uncertainty_tensorboard.py \
    --tum_dir /home/yiming/Dev/tum_rgbd \
    --num_iters 10000 \
    --augment_consecutive \
    --consecutive_ratio 0.5 \
    --window_sizes 8 16 32 64 \
    --checkpoint_dir ./checkpoints_multi \
    --save_interval 1000 \
    --log_dir ./runs/multi_seq \
    --loss_type gaussian
```

**Key differences from single-sequence:**
- Dataset loader auto-detects all sequences in `tum_dir`
- 10k iterations (vs 2k/5k before) — more data needs more training
- Same augmented sampling: 50% varied + 50% consecutive

### 2.2 Training Variants

Run all three for comparison:

| Variant | Command Flag | Checkpoint Dir | Purpose |
|---|---|---|---|
| Gaussian (baseline) | `--loss_type gaussian` | `checkpoints_multi_gaussian` | Main model |
| Laplace | `--loss_type laplace` | `checkpoints_multi_laplace` | Ablation |
| Gaussian (varied only) | (no `--augment_consecutive`) | `checkpoints_multi_varied` | Ablation: does augmented sampling help? |

### 2.3 Monitoring

Watch for:
- d²_rot, d²_trans should approach 3.0 and stay stable
- Loss should decrease across different sequence types
- Check TensorBoard: `tensorboard --logdir ./runs/multi_seq`

---

## 3. Evaluation Protocol

### 3.1 Per-Sequence PGO Evaluation

For each eval sequence, run PGO with ws=16:

```bash
# Template for each eval sequence
CKPT=./checkpoints_multi_gaussian/best.pt

for SEQ in freiburg1_360 freiburg1_floor freiburg1_teddy freiburg2_rpy freiburg3_long_office_household; do
    echo "=== $SEQ ==="
    HF_HUB_OFFLINE=1 python training/tests/eval_pgo_uncertainty.py \
        --tum_dir /home/yiming/Dev/tum_rgbd \
        --tum_sequence rgbd_dataset_$SEQ \
        --uncertainty_checkpoint $CKPT \
        --window_size 16 \
        --overlap 0.5 \
        --output_dir ./eval_benchmark/$SEQ \
        --oracle_isotropic
done
```

**Note:** The eval script may need a `--tum_sequence` flag to select a specific sequence. Check and add if needed.

### 3.2 Metrics per Sequence

For each eval sequence, report:

| Metric | Description |
|---|---|
| ATE (Uniform) | PGO baseline |
| ATE (Predicted) | Our method |
| ATE (Oracle iso) | Upper bound |
| Spearman | Correlation quality |
| d² (full) | Calibration on this sequence |
| Δ ATE (%) | (Uniform - Predicted) / Uniform × 100 |

### 3.3 Aggregate Metrics

Report mean and per-sequence breakdown:

```
| Sequence         | ATE Uniform | ATE Predicted | ATE Oracle | Δ ATE | Spearman |
|------------------|-------------|---------------|------------|-------|----------|
| fr1_360          |             |               |            |       |          |
| fr1_floor        |             |               |            |       |          |
| fr1_teddy        |             |               |            |       |          |
| fr2_rpy          |             |               |            |       |          |
| fr3_long_office  |             |               |            |       |          |
|------------------|-------------|---------------|------------|-------|----------|
| **Mean**         |             |               |            |       |          |
```

### 3.4 Cross-Validation (Optional)

If time permits: leave-one-out on training sequences to check for overfitting.

---

## 4. Success Criteria

### Must Pass

| Criterion | Target | Why |
|---|---|---|
| d² calibration (train dist) | 2.5–4.0 | Model is calibrated |
| Predicted < Uniform ATE on ≥3/5 eval seqs | majority | Generalization |
| Mean Δ ATE across eval seqs | > 0% | Net positive |
| Spearman on eval seqs | > 0.3 | Uncertainty is informative |

### Should Pass

| Criterion | Target | Why |
|---|---|---|
| Mean Δ ATE | > 1% | Meaningful improvement |
| Oracle < Predicted on most seqs | yes | Confirms headroom |
| Laplace vs Gaussian comparison | one is consistently better | Informative ablation |

### Interview-Ready Deliverables

After this benchmark, you should be able to say:

1. "Trained on 6 TUM sequences (~10k frames), evaluated on 5 held-out sequences"
2. "Predicted weights improve PGO ATE on X/5 held-out sequences, mean improvement Y%"
3. "Oracle upper bound shows Z% headroom — motivates full covariance prediction"
4. "Gaussian vs Laplace: [one] is better because [reason]"
5. "Failure modes: [sequence X] shows [specific failure] because [reason]"

---

## 5. Implementation Checklist

### Data Preparation
- [ ] Download 10 additional TUM sequences (5 train + 5 eval)
- [ ] Verify all sequences load correctly (rgb.txt, depth.txt, groundtruth.txt)
- [ ] Verify dataset loader auto-detects all sequences

### Training Script Updates
- [ ] Verify training script works with multiple sequences (may already work via auto-detection)
- [ ] Add `--tum_sequences` flag if needed (to specify train-only subset)
- [ ] Add per-sequence d² logging to monitor calibration across sequence types

### Evaluation Script Updates
- [ ] Add `--tum_sequence` flag to eval script (run PGO on a specific sequence)
- [ ] Add aggregate results output (mean across sequences)
- [ ] Generate per-sequence + aggregate summary table

### Runs
- [ ] Train: Gaussian + augmented (10k iters)
- [ ] Train: Laplace + augmented (10k iters)
- [ ] Train: Gaussian + varied-only (10k iters, ablation)
- [ ] Eval: All 3 checkpoints × 5 eval sequences × {uniform, predicted, oracle}
- [ ] Compile results table

### Documentation
- [ ] Update test plan with benchmark results
- [ ] Update interview prep doc with new numbers

---

## 6. Estimated Timeline

| Step | Time | GPU Time |
|---|---|---|
| Download + extract data | 30 min | — |
| Verify data loading | 15 min | — |
| Script updates (if needed) | 30 min | — |
| Training (3 variants × 10k iters) | — | ~3 hours each, sequential |
| Evaluation (3 × 5 × ws=16) | — | ~30 min each, ~4 hours total |
| Results compilation | 30 min | — |
| **Total** | ~2 hours hands-on | ~13 hours GPU |

GPU time can be reduced by running training overnight.
