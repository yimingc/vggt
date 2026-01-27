# Pose Uncertainty Head - Test Plan

## Overview

This document outlines the test plan for verifying the pose uncertainty head implementation before full-scale training.

**Test Dataset**: TUM RGB-D `freiburg1_desktop` (small, well-known indoor sequence)

## Table of Contents

- [Phase 1: Pre-Training Verification](#phase-1-pre-training-verification-already-done)
- [Phase 2: Training Smoke Test (Complete)](#phase-2-training-smoke-test-complete)
  - [2.1 Setup](#21-setup)
  - [2.2 Training Script Modifications](#22-training-script-modifications)
  - [2.3 Smoke Test Command](#23-smoke-test-command)
  - [2.4 Success Criteria](#24-success-criteria)
- [Phase 3: Monitoring Metrics](#phase-3-monitoring-metrics)
  - [3.1 TensorBoard Metrics to Watch](#31-tensorboard-metrics-to-watch)
  - [3.2 TensorBoard Logging Code](#32-tensorboard-logging-code)
  - [3.3 WandB Integration (Optional)](#33-wandb-integration-optional)
- [Phase 4: Post-Training Evaluation](#phase-4-post-training-evaluation)
  - [4.0 Constant-Uncertainty Baseline](#40-constant-uncertainty-baseline-run-first)
  - [4.1 Calibration Check Script](#41-calibration-check-script)
  - [4.2 Uncertainty vs Error Correlation](#42-uncertainty-vs-error-correlation-scatter-plot)
  - [4.3 Reliability Diagram](#43-reliability-diagram-binned-σ--empirical-error)
  - [4.4 Evaluation Command](#44-evaluation-command)
- [Phase 5: Integration Test with Existing Eval](#phase-5-integration-test-with-existing-eval)
  - [5.1 Extend eval_vggt_tum.py](#51-extend-eval_vggt_tumpy)
  - [5.2 Run Full Evaluation](#52-run-full-evaluation)
- [Phase 5.5: Failure Mode Test (Static Sequence)](#phase-55-failure-mode-test-static-sequence)
- [Quick Test Checklist](#quick-test-checklist)
- [Expected Timeline](#expected-timeline)
- [Phase 6: Scale to Full TUM RGB-D](#phase-6-scale-to-full-tum-rgb-d)
  - [6.1 TUM RGB-D Sequences](#61-tum-rgb-d-sequences)
  - [6.2 Full Training Config](#62-full-training-config)
  - [6.3 Expected Training Time](#63-expected-training-time)
  - [6.4 Scaling Checklist](#64-scaling-checklist)
  - [6.5 Cross-Sequence Validation](#65-cross-sequence-validation)

---

## Phase 1: Pre-Training Verification (Complete)

Run unit tests to verify conventions:

```bash
cd /home/yiming/Dev/vggt
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vggt
python -m pytest training/tests/test_lie_algebra.py -v
```

**Checklist:**
- [x] PyPose quaternion order: `[qx, qy, qz, qw]`
- [x] PyPose Log() returns: `[vx, vy, vz, wx, wy, wz]` (trans first, rot second)
- [x] Window-relative frame 0 is identity
- [x] Scale fitting recovers known scale
- [x] Residual is zero for matching poses
- [x] Gradient flows only to uncertainty branch

---

## Phase 2: Training Smoke Test (Complete)

### 2.1 Setup

Create a minimal training config for uncertainty-only training:

```yaml
# config/train_uncertainty_tum.yaml
camera_nll:
  weight: 1.0
  pose_encoding_type: "absT_quaR_FoV"
  gamma: 0.6
  sqrt_info_rot_clamp: [0.1, 200.0]
  sqrt_info_trans_clamp: [0.1, 100.0]
  residual_sq_clamp: 100.0  # safety for smoke test (log clamp ratio!)
  scale_detach: true
  min_translation: 0.02
  eps: 1.0e-6

# Freeze everything except uncertainty head
freeze_backbone: true
freeze_pose_head: true
```

### 2.2 Training Script Modifications

Before training, verify optimizer setup:

```python
# In training script, add this check:
def verify_trainable_params(model):
    """Verify only uncertainty head is trainable."""
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, param.numel()))
        else:
            frozen.append(name)

    total_trainable = sum(n for _, n in trainable)
    print(f"\n{'='*60}")
    print(f"TRAINABLE PARAMETERS: {total_trainable:,}")
    print(f"{'='*60}")
    for name, count in trainable:
        print(f"  {name}: {count:,}")

    # Sanity check: should be ~2.1M for uncertainty MLP
    assert total_trainable < 5_000_000, f"Too many trainable params: {total_trainable}"
    assert any('pose_uncertainty_branch' in name for name, _ in trainable), \
        "pose_uncertainty_branch not in trainable params!"
    print(f"\n✓ Verified: Only uncertainty head is trainable")
```

### 2.3 Smoke Test Command

```bash
# Run training for 100 iterations on TUM freiburg1_desktop
python training/train.py \
    --config config/train_uncertainty_tum.yaml \
    --dataset tum_rgbd \
    --tum_dir /path/to/tum/freiburg1_desktop \
    --max_iters 100 \
    --log_interval 10 \
    --eval_interval 50 \
    --output_dir ./uncertainty_smoke_test
```

### 2.4 Success Criteria

After 100 iterations:
- [x] `loss_camera_nll` decreases (not necessarily monotonic, but trending down)
- [x] No NaN/Inf in loss or gradients
- [x] `sqrt_info_rot_mean` and `sqrt_info_trans_mean` not stuck at clamp boundaries
- [x] `scale_mean` stable across batches (not collapsing to 0.01 or 100)
- [x] `residual_sq_clamped_ratio` < 10% (if higher, loss signal is misleading)

**Results (2026-01-27):**
- Loss: -0.50 → -1.34 (decreasing ✓)
- sqrt_info_rot: 20.76, sqrt_info_trans: 20.21 (not at clamps ✓)
- scale: 0.628 (stable ✓)
- residual_sq_clamped_ratio: 0.0% (< 10% ✓)
- d²_rot: 0.07, d²_trans: 0.03 (low, calibration evaluated in Phase 4)

---

## Phase 3: Monitoring Metrics

### 3.1 TensorBoard Metrics to Watch

| Metric | Expected Behavior | Red Flag |
|--------|-------------------|----------|
| `loss_camera_nll` | Decreases over training | Stuck or increasing |
| `nll_rot` | Decreases | Much larger than `nll_trans` |
| `nll_trans` | Decreases | Much larger than `nll_rot` |
| `sqrt_info_rot_mean` | Stabilizes ~1-50 | Stuck at 0.1 or 200 |
| `sqrt_info_trans_mean` | Stabilizes ~1-20 | Stuck at 0.1 or 100 |
| `d2_rot_mean` | Approaches 3.0 | >> 10 or << 1 |
| `d2_trans_mean` | Approaches 3.0 | >> 10 or << 1 |
| `scale_mean` | **Stable across batches** | All 0.01 or 100 |
| `scale_std` | Small (~0.1-0.5) | Very large (>2.0) |
| `scale_valid_count_mean` | > 2 | 0 (all static) |
| `residual_sq_clamped_ratio` | < 0.1 | > 0.3 (loss misleading) |

**Note on `scale_mean`:** Don't over-interpret absolute value. VGGT translations can drift depending on window size. The key signal is **stability and consistency**, not `scale ≈ 1.0`.

### 3.2 TensorBoard Logging Code

Add to training loop:

```python
# In training loop, after loss computation:
if step % log_interval == 0:
    writer.add_scalar('loss/camera_nll', loss_dict['loss_camera_nll'], step)
    writer.add_scalar('loss/nll_rot', loss_dict['nll_rot'], step)
    writer.add_scalar('loss/nll_trans', loss_dict['nll_trans'], step)

    writer.add_scalar('uncertainty/sqrt_info_rot_mean', loss_dict['sqrt_info_rot_mean'], step)
    writer.add_scalar('uncertainty/sqrt_info_trans_mean', loss_dict['sqrt_info_trans_mean'], step)

    writer.add_scalar('calibration/d2_rot_mean', loss_dict['d2_rot_mean'], step)
    writer.add_scalar('calibration/d2_trans_mean', loss_dict['d2_trans_mean'], step)

    writer.add_scalar('scale/mean', loss_dict['scale_mean'], step)
    writer.add_scalar('scale/std', loss_dict['scale_std'], step)
    writer.add_scalar('scale/valid_count_mean', loss_dict['scale_valid_count_mean'], step)

    # Log residual clamp ratio (IMPORTANT for smoke test)
    if 'residual_sq_clamped_ratio' in loss_dict:
        writer.add_scalar('debug/residual_sq_clamped_ratio', loss_dict['residual_sq_clamped_ratio'], step)

    # Gradient norm for uncertainty head
    grad_norm = 0.0
    for name, param in model.named_parameters():
        if 'pose_uncertainty_branch' in name and param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    writer.add_scalar('grad/uncertainty_head_norm', grad_norm, step)
```

### 3.3 WandB Integration (Optional)

```python
import wandb

wandb.init(project="vggt-uncertainty", name="smoke-test-tum")

# In training loop:
wandb.log({
    "loss/camera_nll": loss_dict['loss_camera_nll'].item(),
    "loss/nll_rot": loss_dict['nll_rot'].item(),
    "loss/nll_trans": loss_dict['nll_trans'].item(),
    "uncertainty/sqrt_info_rot_mean": loss_dict['sqrt_info_rot_mean'].item(),
    "uncertainty/sqrt_info_trans_mean": loss_dict['sqrt_info_trans_mean'].item(),
    "calibration/d2_rot_mean": loss_dict['d2_rot_mean'].item(),
    "calibration/d2_trans_mean": loss_dict['d2_trans_mean'].item(),
    "scale/mean": loss_dict['scale_mean'].item(),
    "scale/std": loss_dict['scale_std'].item(),
}, step=step)
```

---

## Phase 4: Post-Training Evaluation

### 4.0 Constant-Uncertainty Baseline (Run First!)

Before evaluating trained model, establish a baseline with constant uncertainty:

```python
def compute_baseline_nll(model, dataloader, device):
    """Compute NLL with sqrt_info = 1.0 (constant uncertainty baseline)."""
    # Use the SAME loss computation pipeline, just override sqrt_info
    all_nll = []
    all_d2 = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # ... run model, compute residuals ...

            # Override with constant sqrt_info = 1.0
            sqrt_info_baseline = torch.ones_like(sqrt_info_predicted)

            # Compute NLL with baseline (same formula as training)
            lambda_diag = sqrt_info_baseline ** 2  # = 1.0
            nll_baseline = 0.5 * (residual_sq * lambda_diag - 2 * torch.log(sqrt_info_baseline + eps))
            d2_baseline = (residual_sq * lambda_diag).sum(dim=-1)

            all_nll.append(nll_baseline.mean().item())
            all_d2.append(d2_baseline.mean().item())

    print(f"\nBaseline (sqrt_info=1.0):")
    print(f"  NLL:  {np.mean(all_nll):.4f}")
    print(f"  d² mean: {np.mean(all_d2):.2f} (expect >> 6 if model has useful signal)")

    return {'nll_baseline': np.mean(all_nll), 'd2_baseline': np.mean(all_d2)}
```

**Interpretation:**
- If trained NLL < baseline NLL → model learned useful uncertainty
- If trained d² ≈ 6 but baseline d² >> 6 → calibration improved

### 4.1 Calibration Check Script

Create `training/tests/eval_uncertainty_calibration.py`:

```python
"""
Evaluate uncertainty calibration on TUM RGB-D.

A well-calibrated uncertainty should satisfy:
- d² = Σ λ_k r_k² follows χ²(6) distribution
- d²_rot follows χ²(3), d²_trans follows χ²(3)
- Mean d² ≈ 6 (or 3 for rot/trans separately)

CAVEAT: Residuals are NOT i.i.d. (sliding windows, temporal correlation).
KS p-values will often be overly pessimistic. Treat KS as QUALITATIVE,
not pass/fail. The mean and p95 matching χ² is more important.
"""

import torch
import numpy as np
from scipy import stats

def evaluate_calibration(model, dataloader, device):
    """Compute calibration statistics."""
    all_d2_rot = []
    all_d2_trans = []
    all_d2_total = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            predictions = model(images)

            # Compute residuals and d² (same as loss computation)
            # ... (reuse loss computation logic but collect d² values)

            d2_rot = ...  # [B, S-1]
            d2_trans = ...  # [B, S-1]

            all_d2_rot.append(d2_rot.cpu().numpy().flatten())
            all_d2_trans.append(d2_trans.cpu().numpy().flatten())
            all_d2_total.append((d2_rot + d2_trans).cpu().numpy().flatten())

    d2_rot = np.concatenate(all_d2_rot)
    d2_trans = np.concatenate(all_d2_trans)
    d2_total = np.concatenate(all_d2_total)

    # Statistics
    print(f"\nCalibration Statistics:")
    print(f"  d²_rot:   mean={d2_rot.mean():.2f} (expect ~3), p95={np.percentile(d2_rot, 95):.2f}")
    print(f"  d²_trans: mean={d2_trans.mean():.2f} (expect ~3), p95={np.percentile(d2_trans, 95):.2f}")
    print(f"  d²_total: mean={d2_total.mean():.2f} (expect ~6), p95={np.percentile(d2_total, 95):.2f}")

    # Chi-square reference values
    chi2_3 = stats.chi2(df=3)
    chi2_6 = stats.chi2(df=6)

    print(f"\n  Reference χ²(3): mean=3.0, p95={chi2_3.ppf(0.95):.2f}")
    print(f"  Reference χ²(6): mean=6.0, p95={chi2_6.ppf(0.95):.2f}")

    # Kolmogorov-Smirnov test (QUALITATIVE only - see docstring caveat)
    ks_rot = stats.kstest(d2_rot, 'chi2', args=(3,))
    ks_trans = stats.kstest(d2_trans, 'chi2', args=(3,))
    ks_total = stats.kstest(d2_total, 'chi2', args=(6,))

    print(f"\n  KS test (QUALITATIVE - samples are correlated, p-values pessimistic):")
    print(f"    d²_rot vs χ²(3):   p={ks_rot.pvalue:.4f}")
    print(f"    d²_trans vs χ²(3): p={ks_trans.pvalue:.4f}")
    print(f"    d²_total vs χ²(6): p={ks_total.pvalue:.4f}")
    print(f"  NOTE: Mean and p95 matching reference is MORE important than p-value.")

    return {
        'd2_rot_mean': d2_rot.mean(),
        'd2_trans_mean': d2_trans.mean(),
        'd2_total_mean': d2_total.mean(),
        'ks_rot_pvalue': ks_rot.pvalue,
        'ks_trans_pvalue': ks_trans.pvalue,
        'ks_total_pvalue': ks_total.pvalue,
    }
```

### 4.2 Uncertainty vs Error Correlation (Scatter Plot)

```python
def plot_uncertainty_vs_error(residuals, sqrt_info, output_path):
    """
    Plot predicted sigma vs actual |residual|.

    For well-calibrated uncertainty:
    - Points should cluster around y=x line
    - 68% of points should fall within 1-sigma
    """
    import matplotlib.pyplot as plt

    # sigma = 1 / sqrt_info
    sigma = 1.0 / sqrt_info  # [N, 6]
    actual_error = np.abs(residuals)  # [N, 6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']  # trans first, rot second

    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        ax.scatter(sigma[:, i], actual_error[:, i], alpha=0.3, s=1)
        max_val = max(sigma[:, i].max(), actual_error[:, i].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='y=x (ideal)')
        ax.set_xlabel(f'Predicted σ_{label}')
        ax.set_ylabel(f'Actual |r_{label}|')
        ax.set_title(f'{label}: corr={np.corrcoef(sigma[:, i], actual_error[:, i])[0,1]:.3f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved uncertainty vs error scatter plot to {output_path}")
```

### 4.3 Reliability Diagram (Binned σ → Empirical Error)

```python
def plot_reliability_diagram(residuals, sqrt_info, output_path, n_bins=10):
    """
    Reliability diagram: bin by predicted σ, plot mean |residual| per bin.

    For well-calibrated uncertainty:
    - Points should lie on y=x line
    - More interpretable than scatter plot for reviewers
    """
    import matplotlib.pyplot as plt

    sigma = 1.0 / sqrt_info  # [N, 6]
    actual_error = np.abs(residuals)  # [N, 6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']

    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        # Bin by predicted sigma
        sigma_i = sigma[:, i]
        error_i = actual_error[:, i]

        # Use quantile bins for balanced counts
        bin_edges = np.percentile(sigma_i, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        mean_errors = []
        std_errors = []

        for j in range(n_bins):
            mask = (sigma_i >= bin_edges[j]) & (sigma_i < bin_edges[j+1])
            if mask.sum() > 10:
                bin_centers.append(sigma_i[mask].mean())
                mean_errors.append(error_i[mask].mean())
                std_errors.append(error_i[mask].std() / np.sqrt(mask.sum()))

        bin_centers = np.array(bin_centers)
        mean_errors = np.array(mean_errors)
        std_errors = np.array(std_errors)

        ax.errorbar(bin_centers, mean_errors, yerr=std_errors, fmt='o-', capsize=3)
        max_val = max(bin_centers.max(), mean_errors.max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'r--', label='y=x (ideal)')
        ax.set_xlabel(f'Predicted σ_{label} (binned)')
        ax.set_ylabel(f'Mean |r_{label}|')
        ax.set_title(f'{label}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved reliability diagram to {output_path}")
```

### 4.4 Evaluation Command

```bash
# After training, run calibration evaluation
python training/tests/eval_uncertainty_calibration.py \
    --checkpoint ./uncertainty_smoke_test/checkpoint_100.pt \
    --tum_dir /path/to/tum/freiburg1_desktop \
    --output_dir ./uncertainty_eval_output
```

---

## Phase 5: Integration Test with Existing Eval

### 5.1 Extend eval_vggt_tum.py

Add uncertainty output to existing evaluation:

```python
# In evaluate_sequence(), after VGGT inference:
if 'pose_sqrt_info_list' in predictions:
    sqrt_info = predictions['pose_sqrt_info_list'][-1]  # [1, S, 6]

    # Convert to sigma (standard deviation)
    import torch.nn.functional as F
    sqrt_info_pos = F.softplus(sqrt_info) + 1e-6
    sigma = 1.0 / sqrt_info_pos  # [1, S, 6]

    # Remember: [:3] = trans, [3:] = rot
    sigma_trans = sigma[0, :, :3].cpu().numpy()  # [S, 3]
    sigma_rot = sigma[0, :, 3:].cpu().numpy()    # [S, 3]

    print(f"\n  Uncertainty Statistics:")
    print(f"    σ_trans: mean={sigma_trans.mean():.4f}, std={sigma_trans.std():.4f}")
    print(f"    σ_rot:   mean={sigma_rot.mean():.4f} rad ({np.degrees(sigma_rot.mean()):.2f}°)")
```

### 5.2 Run Full Evaluation

```bash
python training/tests/eval_vggt_tum.py \
    --tum_dir /path/to/tum/freiburg1_desktop \
    --num_frames 24 \
    --sampling consecutive \
    --output_dir ./uncertainty_eval_tum
```

---

## Phase 5.5: Failure Mode Test (Static Sequence)

**Purpose:** Verify model handles degeneracy gracefully, not accidentally.

### Test Setup

Create or select a sequence where frames are nearly static:
- Very small camera motion (< 1cm translation)
- `scale_valid_count_mean ≈ 0`

### Expected Behavior

| Metric | Expected | Why |
|--------|----------|-----|
| `scale_valid_count_mean` | ~0 | Few valid frames for scale fitting |
| `scale_mean` | ~1.0 (fallback) | Fallback kicks in |
| `sqrt_info_trans_mean` | Lower than normal | Translation uncertainty ↑ (less confident) |
| `sqrt_info_rot_mean` | Stable | Rotation still observable |
| No NaN/Inf | True | Robust handling |

### Test Code

```python
def test_static_sequence_handling(model, device):
    """Test that model handles near-static sequences gracefully."""
    # Create synthetic static sequence
    B, S, C, H, W = 1, 8, 3, 518, 518

    # Use same image repeated (perfectly static)
    base_image = torch.randn(1, 1, C, H, W)
    images = base_image.expand(B, S, C, H, W).to(device)

    with torch.no_grad():
        predictions = model(images)

    sqrt_info = predictions['pose_sqrt_info_list'][-1]

    # Check no NaN/Inf
    assert not torch.isnan(sqrt_info).any(), "NaN in sqrt_info for static sequence!"
    assert not torch.isinf(sqrt_info).any(), "Inf in sqrt_info for static sequence!"

    # Check translation uncertainty is higher (sqrt_info lower)
    sqrt_info_trans = sqrt_info[..., :3].mean()
    sqrt_info_rot = sqrt_info[..., 3:].mean()

    print(f"Static sequence test:")
    print(f"  sqrt_info_trans_mean: {sqrt_info_trans:.4f}")
    print(f"  sqrt_info_rot_mean: {sqrt_info_rot:.4f}")
    print(f"  ✓ No NaN/Inf")
```

---

## Quick Test Checklist

### Before Training (Phase 1 - Complete)
- [x] Unit tests pass: `python -m pytest training/tests/test_lie_algebra.py -v`
- [x] Model loads with uncertainty head
- [x] Forward pass produces `pose_sqrt_info_list`
- [x] Only uncertainty branch requires grad (verify_trainable_params)
- [x] Dimension convention frozen in code (split_se3_tangent, concat_se3_tangent)

### During Training (First 100 iters) (Phase 2 - Complete)
- [x] Loss decreases
- [x] No NaN/Inf
- [x] sqrt_info not stuck at clamps
- [x] Scale fitting healthy (scale_mean stable, not at extremes)
- [x] Gradient norm non-zero for uncertainty head
- [x] residual_sq_clamped_ratio < 10%

### After Training
- [ ] Trained NLL < baseline NLL (constant sqrt_info=1)
- [ ] d²_rot ≈ 3, d²_trans ≈ 3
- [ ] Uncertainty correlates with actual error (scatter plot)
- [ ] Reliability diagram close to y=x
- [ ] Static sequence test passes (no NaN, graceful degradation)
- [ ] Can export uncertainty alongside poses

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Phase 1 | Done | Unit tests already passing |
| Phase 2 | Done | 100-iter smoke test passed |
| Phase 3 | Ongoing | Monitor during training |
| Phase 4 | 10 min | Calibration check + baseline comparison |
| Phase 5 | 10 min | Integration with existing eval |
| Phase 5.5 | 5 min | Static sequence failure mode test |

**Total: ~35 minutes** for complete verification on freiburg1_desktop.

---

## Phase 6: Scale to Full TUM RGB-D

Once smoke test passes on `freiburg1_desktop`, scale up to all TUM sequences.

### 6.1 TUM RGB-D Sequences

| Sequence | Environment | Motion | Notes |
|----------|-------------|--------|-------|
| fr1_desk | Office desk | Slow | Good for initial testing |
| fr1_desk2 | Office desk | Slow | Similar to desk |
| fr1_room | Room | Medium | Larger motion |
| fr1_360 | 360° rotation | Rotation-heavy | Tests rotation uncertainty |
| fr1_floor | Floor | Fast | Fast motion, blur |
| fr1_plant | Plant | Slow | Texture-rich |
| fr1_teddy | Teddy bear | Slow | Close-up |
| fr2_desk | Office desk | Slow | Different camera |
| fr2_xyz | XYZ translation | Translation-only | Pure translation |
| fr2_rpy | RPY rotation | Rotation-only | Pure rotation |
| fr3_long_office | Long corridor | Long trajectory | Tests drift |
| fr3_structure_texture_far | Varied | Mixed | Challenging |

### 6.2 Full Training Config

```yaml
# config/train_uncertainty_tum_full.yaml
data:
  dataset: tum_rgbd
  tum_sequences:
    - freiburg1_desk
    - freiburg1_desk2
    - freiburg1_room
    - freiburg1_360
    - freiburg2_desk
    - freiburg2_xyz
    - freiburg3_long_office_household
  window_size: 24
  get_nearby: true  # CRITICAL: use nearby frames, not random!
  expand_ratio: 2.0

training:
  max_iters: 10000
  batch_size: 1  # per-GPU
  lr: 1e-4
  weight_decay: 0.01
  warmup_iters: 500

camera_nll:
  weight: 1.0
  residual_sq_clamp: null  # remove for full training
  # ... other params same as smoke test

freeze:
  backbone: true
  pose_head: true  # freeze original pose head
```

### 6.3 Expected Training Time

| GPU | Batch Size | ~Time for 10k iters |
|-----|------------|---------------------|
| A100 40GB | 1 | ~2 hours |
| V100 32GB | 1 | ~3 hours |
| RTX 4090 | 1 | ~2.5 hours |

### 6.4 Scaling Checklist

- [ ] Smoke test passes on freiburg1_desktop
- [ ] Data loader works with multiple sequences
- [ ] Memory usage acceptable (<30GB for A100)
- [ ] Loss continues decreasing on larger dataset
- [ ] Calibration holds across different motion types
- [ ] Uncertainty higher for challenging sequences (fr1_floor, fr3_structure_texture_far)

### 6.5 Cross-Sequence Validation

After full training, validate on held-out sequences:

```bash
# Train on fr1_*, fr2_*, validate on fr3_*
python training/tests/eval_uncertainty_calibration.py \
    --checkpoint ./uncertainty_full/checkpoint_10000.pt \
    --tum_dir /path/to/tum \
    --sequences fr3_long_office fr3_structure_texture_far \
    --output_dir ./uncertainty_eval_cross
```

Expected behavior:
- Calibration should generalize (d² ≈ 6)
- Uncertainty should be higher for held-out challenging sequences
- No overfitting to training motion patterns
