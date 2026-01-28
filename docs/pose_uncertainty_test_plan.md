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
  log_var_clamp: [-20.0, 20.0]  # very loose clamp on log(σ²)
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
    assert any('pose_log_var_branch' in name for name, _ in trainable), \
        "pose_log_var_branch not in trainable params!"
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
- [x] `pose_uncertainty_nll` decreases (not necessarily monotonic, but trending down)
- [x] No NaN/Inf in loss or gradients
- [x] `log_var_at_min` and `log_var_at_max` are 0 (no clamp collapse)
- [x] `scale_mean` stable across batches (not collapsing to 0.01 or 100)
- [x] `residual_sq_clamped_ratio` < 10% (if higher, loss signal is misleading)

**Results (2026-01-27, log-variance parameterization):**
- Loss: -0.01 → -2.19 (decreasing ✓)
- sigma_rot: 0.014 rad, sigma_trans: 0.011 m (reasonable values ✓)
- log_var clamp hit: 0% (no clamp collapse ✓)
- scale: stable (✓)
- residual_sq_clamped_ratio: 0.0% (< 10% ✓)
- d²_rot: 1.68, d²_trans: 3.38 (trans close to target 3! ✓)

---

## Phase 3: Monitoring Metrics

### 3.1 TensorBoard Metrics to Watch

| Metric | Expected Behavior | Red Flag |
|--------|-------------------|----------|
| `pose_uncertainty_nll` | Decreases over training | Stuck or increasing |
| `rot_uncertainty_nll` | Decreases | Much larger than `trans_uncertainty_nll` |
| `trans_uncertainty_nll` | Decreases | Much larger than `rot_uncertainty_nll` |
| `sigma_rot_mean` | Stabilizes ~0.01-0.1 rad | Near 0 or very large |
| `sigma_trans_mean` | Stabilizes ~0.01-0.1 m | Near 0 or very large |
| `log_var_rot_at_min/max` | 0 | > 0 (clamp collapse) |
| `log_var_trans_at_min/max` | 0 | > 0 (clamp collapse) |
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
    writer.add_scalar('loss/pose_uncertainty_nll', loss_dict['pose_uncertainty_nll'], step)
    writer.add_scalar('loss/rot_uncertainty_nll', loss_dict['rot_uncertainty_nll'], step)
    writer.add_scalar('loss/trans_uncertainty_nll', loss_dict['trans_uncertainty_nll'], step)

    writer.add_scalar('uncertainty/sigma_rot_mean', loss_dict['sigma_rot_mean'], step)
    writer.add_scalar('uncertainty/sigma_trans_mean', loss_dict['sigma_trans_mean'], step)

    writer.add_scalar('calibration/d2_rot_mean', loss_dict['d2_rot_mean'], step)
    writer.add_scalar('calibration/d2_trans_mean', loss_dict['d2_trans_mean'], step)

    writer.add_scalar('scale/mean', loss_dict['scale_mean'], step)
    writer.add_scalar('scale/std', loss_dict['scale_std'], step)
    writer.add_scalar('scale/valid_count_mean', loss_dict['scale_valid_count_mean'], step)

    # Log residual clamp ratio (IMPORTANT for smoke test)
    if 'residual_sq_clamped_ratio' in loss_dict:
        writer.add_scalar('debug/residual_sq_clamped_ratio', loss_dict['residual_sq_clamped_ratio'], step)

    # Log log_var clamp hit rate (should be 0)
    writer.add_scalar('diagnostic/log_var_rot_at_min', loss_dict['log_var_rot_at_min'], step)
    writer.add_scalar('diagnostic/log_var_rot_at_max', loss_dict['log_var_rot_at_max'], step)

    # Gradient norm for uncertainty head
    grad_norm = 0.0
    for name, param in model.named_parameters():
        if 'pose_log_var_branch' in name and param.grad is not None:
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
    "loss/pose_uncertainty_nll": loss_dict['pose_uncertainty_nll'].item(),
    "loss/rot_uncertainty_nll": loss_dict['rot_uncertainty_nll'].item(),
    "loss/trans_uncertainty_nll": loss_dict['trans_uncertainty_nll'].item(),
    "uncertainty/sigma_rot_mean": loss_dict['sigma_rot_mean'].item(),
    "uncertainty/sigma_trans_mean": loss_dict['sigma_trans_mean'].item(),
    "calibration/d2_rot_mean": loss_dict['d2_rot_mean'].item(),
    "calibration/d2_trans_mean": loss_dict['d2_trans_mean'].item(),
    "scale/mean": loss_dict['scale_mean'].item(),
    "scale/std": loss_dict['scale_std'].item(),
    "diagnostic/log_var_rot_at_min": loss_dict['log_var_rot_at_min'].item(),
    "diagnostic/log_var_rot_at_max": loss_dict['log_var_rot_at_max'].item(),
}, step=step)
```

### 3.4 Phase 3 Results (2000 iterations)

**Training Run:** 2000 iterations on TUM freiburg1_desk (596 frames)

**Checkpoints saved:**
- `checkpoints/best.pt` - iteration 544, calibration_error=0.11
- `checkpoints/final.pt` - iteration 2000
- Periodic checkpoints at 500, 1000, 1500, 2000

**Final Training Metrics:**

| Metric | Value | Target |
|--------|-------|--------|
| NLL Loss | -2.19 | ↓ decreasing |
| d²_rot | 3.11 (best) | ~3 |
| d²_trans | 3.00 (best) | ~3 |
| σ_rot mean | 0.014 rad (0.8°) | stable |
| σ_trans mean | 0.011 m | stable |
| log_var clamp hit | 0% | 0% |

**Best checkpoint** achieved near-perfect calibration at iteration 544.

**Why iteration 544 outperforms later iterations (1k, 2k):**

The NLL loss has two competing terms: `0.5 * (r² * exp(-log_var) + log_var)`
- First term (`r² * exp(-log_var)`): Penalizes underconfidence (σ too large)
- Second term (`log_var`): Penalizes overconfidence (σ too small)

Training dynamics:
1. **Early phase (0-500)**: Model starts underconfident, rapidly increases confidence
2. **Sweet spot (~544)**: Model achieves good calibration (d² ≈ 3)
3. **Late phase (500-2000)**: Model becomes progressively overconfident on training data

The model overfits to the single TUM sequence (596 frames), learning to be overly confident on familiar motion patterns. This is expected behavior when training on limited data - the model memorizes rather than generalizes.

**Mitigation strategies for future work:**
- Early stopping based on d² (stop when |d² - 3| < threshold)
- Train on multiple sequences for better generalization
- Add regularization to penalize extreme log_var values

For detailed training analysis, loss curves, and diagnostic plots, see:
**[Pose Uncertainty Training Analysis](pose_uncertainty_training_analysis.md)**

---

## Phase 4: Post-Training Evaluation

### 4.0 Baseline Comparison (Homoscedastic MLE)

Compare trained heteroscedastic model against proper baselines:

**Two baselines:**
1. **Weak baseline (σ=1)**: Arbitrary, for sanity check only
2. **Strong baseline (Homoscedastic MLE)**: Best possible constant σ per dimension

The MLE baseline fits optimal constant uncertainty from the data:
```python
# MLE for constant σ: σ_k² = E[r_k²]
sigma_sq_mle = residual_sq.mean(axis=0)  # [6] per dimension
log_var_mle = np.log(sigma_sq_mle + 1e-12)
```

**Why MLE baseline matters:**
- σ=1 is arbitrary and not comparable (residuals may be ~0.01, making σ=1 meaningless)
- MLE baseline is the "strongest homoscedastic" - fair comparison
- If trained NLL < MLE NLL → heteroscedastic model learned **useful per-sample uncertainty**

**Interpretation:**
- `NLL_trained < NLL_mle` → Model learned useful heteroscedastic uncertainty ✓
- `NLL_trained ≈ NLL_mle` → Per-sample uncertainty not adding value
- `NLL_trained > NLL_mle` → Model is worse than constant σ (problem!)

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

### 4.2 Coverage Test (Primary Calibration Metric)

The most rigorous calibration test uses **normalized residuals** z = r/σ:

```python
def compute_coverage_statistics(residuals, log_var):
    """
    For well-calibrated Gaussian: z = r/σ ~ N(0,1)

    Expected coverage:
    - P(|z| < 1) ≈ 0.6827 (68.27%)
    - P(|z| < 2) ≈ 0.9545 (95.45%)
    - P(|z| < 3) ≈ 0.9973 (99.73%)
    """
    sigma = np.exp(0.5 * log_var)
    z = residuals / sigma  # Should be ~N(0,1)

    coverage_1sigma = (np.abs(z) < 1).mean()  # Target: 0.6827
    coverage_2sigma = (np.abs(z) < 2).mean()  # Target: 0.9545
    coverage_3sigma = (np.abs(z) < 3).mean()  # Target: 0.9973
```

**Also compute quantile-coverage reliability:**
- For d² ~ χ²(df), check P(d² ≤ χ²_p) vs theoretical p
- Plot empirical quantile vs theoretical quantile (should be y=x)

### 4.3 Reliability Diagram (Corrected)

**IMPORTANT:** For Gaussian r ~ N(0, σ²), E[|r|] = σ × √(2/π) ≈ **0.798σ**, NOT σ!

```python
def plot_reliability_diagram(residuals, log_var, output_path, n_bins=10):
    """
    Reliability diagram: bin by predicted σ, plot mean |residual| per bin.

    CORRECTED: Ideal line is y = 0.798x (not y = x)
    because E[|r|] = σ * sqrt(2/π) for Gaussian.
    """
    GAUSSIAN_FACTOR = np.sqrt(2 / np.pi)  # ≈ 0.798

    # ... binning code ...

    # Plot CORRECT ideal line
    ax.plot(x, GAUSSIAN_FACTOR * x, 'r--', label='y=0.798x (Gaussian ideal)')
```

### 4.4 Whitened Covariance (Diagonal Assumption Check)

Check if diagonal uncertainty is sufficient:

```python
def compute_whitened_covariance(residuals, log_var):
    """
    Compute Cov(z) where z = r/σ (whitened residuals).

    If calibrated AND diagonal assumption valid:
    - Diagonal elements ≈ 1
    - Off-diagonal elements ≈ 0

    Large off-diagonal → coupling exists → consider full covariance
    """
    sigma = np.exp(0.5 * log_var)
    z = residuals / sigma
    cov_z = np.cov(z, rowvar=False)  # [6, 6]

    # Check: diag ≈ 1, off-diag ≈ 0
    max_off_diag = np.abs(cov_z - np.diag(np.diag(cov_z))).max()
```

### 4.6 Evaluation Command

```bash
# After training, run calibration evaluation
python training/tests/eval_uncertainty_calibration.py \
    --checkpoint ./checkpoints/best.pt \
    --tum_dir /path/to/tum/freiburg1_desk \
    --output_dir ./eval_uncertainty
```

### 4.7 Phase 4 Results (Calibration Evaluation)

**Checkpoint evaluated:** `checkpoints/best.pt` (iteration 544, from Phase 3 training)

> **Note:** Phase 3 and Phase 4 use the same trained checkpoint. The slight difference in d² values (3.11/3.00 in Phase 3 vs 2.53/2.77 in Phase 4) is due to different data sampling - Phase 3 reports training-time metrics while Phase 4 evaluates on freshly sampled windows.

**Calibration Statistics (d² vs χ²):**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| d²_rot mean | 2.53 | 3.0 | ✓ Well-calibrated |
| d²_trans mean | 2.77 | 3.0 | ✓ Well-calibrated |
| d²_total mean | 5.30 | 6.0 | ✓ Well-calibrated |

**Baseline Comparison (Homoscedastic MLE):**

| Model | NLL | Notes |
|-------|-----|-------|
| **Trained (heteroscedastic)** | **-4.108** | Per-sample uncertainty |
| MLE baseline (best constant σ) | -4.053 | σ_k² = E[r_k²] |
| Unit baseline (σ=1) | 0.0001 | Sanity check only |

**✓ NLL improvement vs MLE: 0.055** - Heteroscedastic model learned useful per-sample uncertainty beyond best constant σ.

**Coverage Statistics (z = r/σ vs N(0,1)):**

| Metric | Empirical | Target | Status |
|--------|-----------|--------|--------|
| P(\|z\| < 1) | 0.744 | 0.683 | ~ Slightly underconfident |
| P(\|z\| < 2) | 0.956 | 0.955 | ✓ Excellent |
| P(\|z\| < 3) | 0.993 | 0.997 | ✓ Excellent |

The 1σ coverage is slightly high (σ too large), but 2σ and 3σ coverage are nearly perfect.

**Diagonal Assumption Check (Cov(z) analysis):**

| Metric | Value | Assessment |
|--------|-------|------------|
| max \|off-diag\| | 0.557 | ✗ Significant coupling |
| mean \|off-diag\| | 0.144 | Moderate |
| Largest coupling | vx-wy = -0.557 | Translation X ↔ Rotation Y |

**Finding:** The diagonal assumption is limiting. There's significant coupling between translation-x and rotation-y that the diagonal uncertainty cannot capture. This motivates future work on full 6×6 covariance prediction.

**Generated Plots** (in `./eval_uncertainty_v2/`):
- `d2_histogram.png` - d² distribution vs χ² reference
- `coverage_reliability.png` - Quantile-coverage diagram (primary calibration plot)
- `normalized_residual_histogram.png` - z=r/σ vs N(0,1)
- `whitened_covariance.png` - Cov(z) heatmap (diagonal assumption check)
- `reliability_diagram.png` - Binned σ vs mean|r| (ideal line: y=0.798x)
- `uncertainty_vs_error.png` - σ vs |r| scatter (correlation check)

**Conclusion:**
- ✓ **Well-calibrated**: d² close to target, coverage statistics good
- ✓ **Heteroscedastic value proven**: Beats MLE baseline
- ~ **Slightly underconfident**: 1σ coverage 74% vs 68% expected
- ✗ **Diagonal assumption limiting**: vx-wy coupling suggests full covariance would help

---

## Phase 5: Integration Test with Existing Eval

### 5.1 Extend eval_vggt_tum.py

Add uncertainty output to existing evaluation:

```python
# In evaluate_sequence(), after VGGT inference:
if 'pose_log_var_list' in predictions:
    log_var = predictions['pose_log_var_list'][-1]  # [1, S, 6]

    # Convert to sigma (standard deviation): σ = exp(0.5 * log_var)
    sigma = torch.exp(0.5 * log_var)  # [1, S, 6]

    # Remember: [:3] = trans, [3:] = rot (PyPose se(3) convention)
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
| `sigma_trans_mean` | Higher than normal | Translation uncertainty ↑ (less confident) |
| `sigma_rot_mean` | Stable | Rotation still observable |
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

    log_var = predictions['pose_log_var_list'][-1]

    # Check no NaN/Inf
    assert not torch.isnan(log_var).any(), "NaN in log_var for static sequence!"
    assert not torch.isinf(log_var).any(), "Inf in log_var for static sequence!"

    # Compute sigma: σ = exp(0.5 * log_var)
    sigma = torch.exp(0.5 * log_var)
    sigma_trans = sigma[..., :3].mean()
    sigma_rot = sigma[..., 3:].mean()

    print(f"Static sequence test:")
    print(f"  sigma_trans_mean: {sigma_trans:.4f}")
    print(f"  sigma_rot_mean: {sigma_rot:.4f}")
    print(f"  ✓ No NaN/Inf")
```

---

## Quick Test Checklist

### Before Training (Phase 1 - Complete)
- [x] Unit tests pass: `python -m pytest training/tests/test_lie_algebra.py -v`
- [x] Model loads with uncertainty head
- [x] Forward pass produces `pose_log_var_list`
- [x] Only uncertainty branch requires grad (verify_trainable_params)
- [x] Dimension convention frozen in code (split_se3_tangent, concat_se3_tangent)

### During Training (First 100 iters) (Phase 2 - Complete)
- [x] Loss decreases
- [x] No NaN/Inf
- [x] log_var clamp hit rate = 0 (no clamp collapse)
- [x] Scale fitting healthy (scale_mean stable, not at extremes)
- [x] Gradient norm non-zero for uncertainty head
- [x] residual_sq_clamped_ratio < 10%

### After Training (Phase 4 - Complete)
- [x] Trained NLL < MLE baseline NLL - NLL: -4.108 vs -4.053 (improvement: 0.055) ✓
- [x] d²_rot ≈ 3, d²_trans ≈ 3 - d²_rot=2.53, d²_trans=2.77 ✓
- [x] Coverage statistics reasonable - 1σ: 74.4% (target 68.3%), 2σ: 95.6% ✓
- [x] Whitened covariance analyzed - max off-diag: 0.557 (vx-wy coupling found)
- [x] Reliability diagram with corrected y=0.798x line ✓
- [x] Static sequence test passes (no NaN, graceful degradation) ✓

### Integration (Phase 5 - In Progress)
- [x] eval_vggt_tum.py extended to output uncertainty statistics
- [x] --uncertainty_checkpoint argument added to load trained weights
- [ ] Verified: Can export uncertainty alongside poses (needs testing)

---

## Expected Timeline

| Phase | Status | Description | Results |
|-------|--------|-------------|---------|
| Phase 1 | Done | Unit tests passing | See [test_lie_algebra.py](../training/tests/test_lie_algebra.py) |
| Phase 2 | Done | 100-iter smoke test | Loss ↓, no clamp collapse |
| Phase 3 | Done | 2000-iter training | Best: d²_rot=3.11, d²_trans=3.00. See [§3.4](#34-phase-3-results-2000-iterations) |
| Phase 4 | Done | Calibration evaluation | d²_rot=2.53, d²_trans=2.77, NLL beats MLE. See [§4.7](#47-phase-4-results-calibration-evaluation) |
| Phase 5 | In Progress | Integration with eval | eval_vggt_tum.py extended |
| Phase 5.5 | Done | Static sequence test | No NaN/Inf ✓ |

**Detailed Results:**
- Training analysis & plots: [pose_uncertainty_training_analysis.md](pose_uncertainty_training_analysis.md)
- Evaluation plots: `./eval_uncertainty_v2/` (d2_histogram.png, coverage_reliability.png, whitened_covariance.png, etc.)

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
  log_var_clamp: [-20.0, 20.0]
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
