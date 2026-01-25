# Pose Uncertainty Head Design for VGGT

> **Design Document** - Last updated: 2026-01-21
>
> This document describes the design and implementation of a pose uncertainty prediction head for VGGT using SE(3) Lie algebra formulation.

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Key Design Decisions](#key-design-decisions)
4. [PyPose Conventions](#pypose-conventions)
5. [Implementation Details](#implementation-details)
6. [Numerical Stability](#numerical-stability)
7. [Verification Plan](#verification-plan)
8. [Training Configuration](#training-configuration)
9. [Implementation Gotchas](#implementation-gotchas)

---

## Overview

Add a 6-dim diagonal information matrix uncertainty prediction to VGGT's pose estimation using **SE(3) Lie algebra (se(3))** formulation with NLL loss.

**Key principle: Minimal correct loop first, then iterate.**

### What We're Building

- **Input**: VGGT pose encoding (9-dim: `[tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]`)
- **Output**: 6-dim √λ per frame (3 rotation + 3 translation information matrix diagonal)
- **Loss**: Negative log-likelihood in se(3) tangent space
- **Constraint Type**: Star-graph (frame 0 → frame i) for v1 PGO compatibility

### Files to Modify

| File | Change |
|------|--------|
| `vggt/utils/lie_algebra.py` | **NEW**: PyPose wrapper (window-relative, scale, residual) |
| `vggt/heads/camera_head.py` | Add uncertainty MLP branch, return `sqrt_info_list` |
| `training/loss.py` | Add `compute_camera_nll_loss()`, update `MultitaskLoss` |
| `vggt/models/vggt.py` | Pass `pose_sqrt_info_list` through forward |
| `requirements.txt` | Add `pypose` dependency |

---

## Mathematical Foundation

### NLL Loss Formulation

For a Gaussian distribution in se(3) tangent space with diagonal information matrix Λ:

```
L_nll = 0.5 * Σᵢ (rᵢ² * λᵢ - log(λᵢ))
      = 0.5 * Σᵢ (rᵢ² * (√λᵢ)² - 2*log(√λᵢ))
```

Where:
- `rᵢ` is the i-th component of the SE(3) residual (6-dim)
- `λᵢ` is the i-th diagonal element of the information matrix
- We predict `√λᵢ` directly for automatic positive definiteness

### Residual Computation (PGO-consistent)

Use `inv(GT) @ Pred` convention:
```python
r_i = Log(T_rel_gt_i.Inv() @ T_rel_pred_i)  # 6-dim
```

This means "how much Pred deviates from GT", consistent with factor graph/PGO conventions where the residual measures the error in the predicted quantity.

### Why Window-Relative Poses?

VGGT uses frame 0 as reference within each window. We convert both GT and Pred to **window-relative** form:

```python
# T_rel_i = T_0^{-1} @ T_i  (relative to frame 0)
T_rel_gt = T_gt_0.Inv() @ T_gt_i
T_rel_pred = T_pred_0.Inv() @ T_pred_i
```

**Benefits:**
- No need for global Umeyama/Sim3 alignment
- Consistent with sliding-window VIO/SLAM setting
- Skip frame 0 naturally (T_rel_0 = Identity, no meaningful uncertainty)
- Enables star-graph PGO constraints (0→1, 0→2, ..., 0→S)

### Per-Window Scale Fitting

VGGT doesn't provide metric scale. We fit scale per window using **relative translations**:

```python
# s* = argmin Σ ||s * t_pred - t_gt||²
# Closed-form: s* = Σ(t_gt · t_pred) / Σ||t_pred||²

scale = (gt_t_rel * pred_t_rel).sum() / (pred_t_rel ** 2).sum()
```

**Critical**: Use `scale.detach()` to prevent gradient flow through scale fitting (avoids "cheating channel" where model learns to manipulate scale instead of uncertainty).

---

## Key Design Decisions

### 1. Window-Relative Poses (Critical!)

```python
def compute_window_relative_poses(pose_enc: torch.Tensor) -> pp.SE3:
    T_abs = pose_encoding_to_se3(pose_enc)  # [B, S]
    T_0 = T_abs[:, 0:1]  # [B, 1]
    T_rel = T_0.Inv() @ T_abs  # [B, S]
    return T_rel  # T_rel[:, 0] = Identity
```

### 2. Uncertainty Semantics: Star-Graph Constraints (0→i)

**Current design:** Per-frame uncertainty relative to frame 0.
- Residual: `r_i = Log(inv(T_rel_gt_i) @ T_rel_pred_i)` where `T_rel = T_0^{-1} @ T_i`
- Corresponds to **star-graph constraints** (edges: 0→1, 0→2, ..., 0→S)

**For v1 PGO:**
- Use star-graph: nodes = frames, edges = 0→i with information from √λ_i
- Simple and consistent with training

**Future (v2):**
- Per-edge uncertainty (i-1→i or arbitrary i→j)
- Requires different head architecture or covariance propagation

### 3. Training Strategy: Freeze Original Pose Head

**Phase 1 (this implementation):**
- Freeze backbone + freeze original pose head
- Only train uncertainty head
- Clear ablation: uncertainty head doesn't affect pose quality

```python
# At training start, verify:
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params}")
# Should be ~small (only uncertainty MLP)
```

### 4. Separate Clamping for Rotation/Translation

Rotation and translation have different units and scales:
- Rotation residuals are in radians (typically < 0.5 rad)
- Translation residuals are in meters (scale-dependent)

```python
# PyPose convention: [:3] = translation, [3:] = rotation
sqrt_info_trans = sqrt_info[..., :3].clamp(min=0.1, max=100.0) # adjust based on scale
sqrt_info_rot = sqrt_info[..., 3:].clamp(min=0.1, max=200.0)   # ~σ_r in [0.005, 10] rad
```

### 5. What We're NOT Doing in v1

- `char_length` normalization (add later if needed for cross-dataset consistency)
- Joint training with pose head (stick with frozen pose head)
- Full covariance matrix (stick with 6-dim diagonal)
- Per-edge uncertainty (stick with per-frame relative to frame 0)

---

## PyPose Conventions

### SE3 Input Format (IMPORTANT!)

```python
import pypose as pp

# PyPose SE3 takes 7-dim: [tx, ty, tz, qx, qy, qz, qw]
# NOTE: Translation FIRST, then quaternion!
t = pose_enc[..., :3]      # translation
q = pose_enc[..., 3:7]     # quaternion [qx, qy, qz, qw]
se3_vec = torch.cat([t, q], dim=-1)  # [t, q] order!
T = pp.SE3(se3_vec)
```

### Log Map Output Format

```python
# Log map returns 6-dim: [vx, vy, vz, wx, wy, wz]
xi = T.Log()
#     ├─ :3 ──┤  ├─ 3: ──┤
#     translation  rotation
#     (meters)     (radians)
```

### Dimension Convention (MUST MATCH!)

```
residual = [vx, vy, vz, wx, wy, wz]
           ├─ :3 ──┤  ├─ 3: ──┤
           translation  rotation
           (meters)     (radians)

sqrt_info = [√λ_vx, √λ_vy, √λ_vz, √λ_wx, √λ_wy, √λ_wz]
            ├──── :3 ─────┤  ├──── 3: ─────┤
            translation info   rotation info
```

**Rule:** Always slice as `[..., :3]` for trans, `[..., 3:]` for rot. **NEVER mix!**

---

## Implementation Details

### Step 0: SE(3) Utilities (`vggt/utils/lie_algebra.py`)

```python
import torch
import pypose as pp


def pose_encoding_to_se3(pose_enc: torch.Tensor) -> pp.SE3:
    """
    Convert VGGT pose encoding to PyPose SE3.

    Args:
        pose_enc: (..., 9) tensor [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]

    Returns:
        pp.SE3 object
    """
    t = pose_enc[..., :3]      # translation [tx, ty, tz]
    q = pose_enc[..., 3:7]     # quaternion [qx, qy, qz, qw]

    # PyPose SE3 takes [tx, ty, tz, qx, qy, qz, qw] - t first!
    se3_vec = torch.cat([t, q], dim=-1)
    return pp.SE3(se3_vec)


def compute_window_relative_poses(pose_enc: torch.Tensor) -> pp.SE3:
    """
    Convert absolute poses to window-relative (relative to frame 0).

    Args:
        pose_enc: [B, S, 9] pose encodings

    Returns:
        T_rel: pp.SE3 of shape [B, S] where T_rel[b, 0] = Identity
    """
    T_abs = pose_encoding_to_se3(pose_enc)  # [B, S]
    T_0 = T_abs[:, 0:1]  # [B, 1] - take directly from T_abs, more stable

    # T_rel_i = T_0^{-1} @ T_i
    T_rel = T_0.Inv() @ T_abs  # [B, S]
    return T_rel


def compute_window_scale_batched(
    pred_t_rel: torch.Tensor,
    gt_t_rel: torch.Tensor,
    detach: bool = True,
    min_translation: float = 0.02  # Filter out near-stationary frames (2cm)
) -> torch.Tensor:
    """
    Compute per-window scale using relative translations (vectorized).

    s* = Σ(t_gt · t_pred) / Σ||t_pred||²

    Uses ROBUST filtering to exclude near-stationary frames.

    Args:
        pred_t_rel: [B, S, 3] predicted relative translations
        gt_t_rel: [B, S, 3] GT relative translations
        min_translation: threshold to filter small motions (prevents scale blowup)

    Returns:
        scale: [B] optimal scale per window
    """
    # Skip frame 0 (identity, t=0)
    pred_t = pred_t_rel[:, 1:]  # [B, S-1, 3]
    gt_t = gt_t_rel[:, 1:]      # [B, S-1, 3]

    # ROBUST: Filter out near-stationary frames to prevent scale blowup
    # Use GT-only for mask (pred is up-to-scale, threshold might be too strict/loose)
    gt_norm = gt_t.norm(dim=-1)      # [B, S-1]
    valid_mask = gt_norm > min_translation  # [B, S-1]

    # Compute masked sums (explicit dtype conversion for AMP stability)
    dot_product = (gt_t * pred_t).sum(dim=-1)  # [B, S-1]
    pred_sq = (pred_t ** 2).sum(dim=-1)        # [B, S-1]
    m = valid_mask.to(dot_product.dtype)       # bool -> float for stable masking

    numerator = (dot_product * m).sum(dim=-1)    # [B]
    denominator = (pred_sq * m).sum(dim=-1)      # [B]

    # Count valid frames per batch
    valid_count = valid_mask.sum(dim=-1)  # [B]

    scale = numerator / denominator.clamp(min=1e-8)
    scale = scale.clamp(min=0.01, max=100.0)

    # FALLBACK: If no valid frames (all stationary), use scale=1.0
    scale = torch.where(valid_count >= 2, scale, torch.ones_like(scale))

    if detach:
        scale = scale.detach()  # Prevent gradient flow through scale fitting

    return scale


def compute_se3_residual(T_rel_pred: pp.SE3, T_rel_gt: pp.SE3) -> torch.Tensor:
    """
    Compute SE(3) residual: r = Log(inv(T_gt) @ T_pred).

    This is "how much Pred deviates from GT" - PGO convention.

    Args:
        T_rel_pred: pp.SE3 [B, S] predicted relative poses (scale-corrected)
        T_rel_gt: pp.SE3 [B, S] GT relative poses

    Returns:
        residual: [B, S, 6] in se(3) [vx, vy, vz, wx, wy, wz]
                  [:3] = translation (meters), [3:] = rotation (radians)
    """
    T_err = T_rel_gt.Inv() @ T_rel_pred
    residual = T_err.Log()  # [B, S, 6]
    return residual
```

### Step 1: CameraHead Modifications (`vggt/heads/camera_head.py`)

```python
# In __init__:
self.pose_uncertainty_branch = Mlp(
    in_features=dim_in,
    hidden_features=dim_in // 2,
    out_features=6,  # 3 rotation + 3 translation in se(3)
    drop=0
)

# In trunk_fn, after pose prediction:
pred_sqrt_info_raw = self.pose_uncertainty_branch(self.trunk_norm(pose_tokens_modulated))
# Output RAW values! No clamp here - done in loss via softplus+clamp
pred_sqrt_info_list.append(pred_sqrt_info_raw)

# Return both lists
return pred_pose_enc_list, pred_sqrt_info_list
```

**NOTE:** CameraHead outputs RAW unconstrained values. The loss function applies:
1. `F.softplus(raw) + eps` → ensures positive
2. Separate rot/trans clamping → safety rails

### Step 2: NLL Loss Function (`training/loss.py`)

```python
import torch
import torch.nn.functional as F
import pypose as pp
from vggt.utils.lie_algebra import (
    pose_encoding_to_se3,
    compute_window_relative_poses,
    compute_window_scale_batched,
    compute_se3_residual,
)
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


def compute_camera_nll_loss(
    pred_dict,
    batch_data,
    pose_encoding_type="absT_quaR_FoV",
    gamma=0.6,
    sqrt_info_rot_clamp=(0.1, 200.0),
    sqrt_info_trans_clamp=(0.1, 100.0),  # tighter for stability
    residual_sq_clamp=100.0,  # set to None for formal training
    scale_detach=True,  # detach scale to avoid cheating channel
    eps=1e-6,  # for log stability
    **kwargs
):
    """
    NLL loss for pose uncertainty in se(3) space.

    Pipeline:
    1. Convert GT/Pred to window-relative poses (relative to frame 0)
    2. Fit per-window scale using relative translations
    3. Compute SE(3) residual: r = Log(inv(T_rel_gt) @ T_rel_pred)
    4. NLL loss with diagonal information matrix

    L = 0.5 * Σᵢ (rᵢ² * λᵢ - log(λᵢ))
    """
    pred_pose_encodings = pred_dict['pose_enc_list']
    pred_sqrt_info_list = pred_dict['pose_sqrt_info_list']
    n_stages = len(pred_pose_encodings)

    # Get GT pose encoding
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]
    gt_pose_enc = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type
    )  # [B, S, 9]

    B, S, _ = gt_pose_enc.shape

    # Step 1: Convert GT to window-relative
    T_rel_gt = compute_window_relative_poses(gt_pose_enc)  # pp.SE3 [B, S]
    gt_t_rel = T_rel_gt.translation()  # [B, S, 3]

    total_nll = 0
    for stage_idx in range(n_stages):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_enc = pred_pose_encodings[stage_idx]  # [B, S, 9]
        sqrt_info = pred_sqrt_info_list[stage_idx]       # [B, S, 6]

        # Step 2: Convert Pred to window-relative
        T_rel_pred = compute_window_relative_poses(pred_pose_enc)
        pred_t_rel = T_rel_pred.translation()  # [B, S, 3]

        # Step 3: Fit per-window scale (using relative translations)
        scale = compute_window_scale_batched(pred_t_rel, gt_t_rel, detach=scale_detach)  # [B]

        # Apply scale to pred relative translations
        pred_t_rel_scaled = scale[:, None, None] * pred_t_rel  # [B, S, 3]

        # Reconstruct scaled SE3 (keep rotation, scale translation)
        pred_r_rel = T_rel_pred.rotation()  # pp.SO3 [B, S]

        # IMPORTANT: Normalize quaternion to prevent numerical drift!
        q = pred_r_rel.tensor()  # [B, S, 4] in [qx, qy, qz, qw] order
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # IMPORTANT: Hemisphere canonicalization (q and -q are same rotation)
        # Force qw >= 0 to avoid Log() branch jumps near π rotation
        sign = torch.where(q[..., 3:4] < 0, -1.0, 1.0).to(q.dtype)
        q = q * sign

        # Build new SE3 with scaled translation
        T_rel_pred_scaled = pp.SE3(torch.cat([pred_t_rel_scaled, q], dim=-1))

        # Step 4: Compute SE(3) residual: r = Log(inv(T_gt) @ T_pred)
        residual = compute_se3_residual(T_rel_pred_scaled, T_rel_gt)  # [B, S, 6]
        # DIMENSION CONVENTION (PyPose Log output):
        #   residual[..., :3] = translation (vx, vy, vz) in meters
        #   residual[..., 3:] = rotation (wx, wy, wz) in radians

        # Step 5: Skip frame 0 (identity, residual=0)
        residual = residual[:, 1:]      # [B, S-1, 6]
        sqrt_info_raw = sqrt_info[:, 1:]    # [B, S-1, 6]

        # Step 6: SOFTPLUS + CLAMP for stable positive sqrt_info
        # softplus ensures positive, clamp is just safety rail
        sqrt_info_positive = F.softplus(sqrt_info_raw) + eps  # Always positive!

        # DIMENSION CONVENTION (must match residual!):
        #   sqrt_info[..., :3] = translation uncertainty (√λ for vx, vy, vz)
        #   sqrt_info[..., 3:] = rotation uncertainty (√λ for wx, wy, wz)
        sqrt_info_trans = sqrt_info_positive[..., :3].clamp(
            min=sqrt_info_trans_clamp[0], max=sqrt_info_trans_clamp[1]
        )
        sqrt_info_rot = sqrt_info_positive[..., 3:].clamp(
            min=sqrt_info_rot_clamp[0], max=sqrt_info_rot_clamp[1]
        )
        sqrt_info_clamped = torch.cat([sqrt_info_trans, sqrt_info_rot], dim=-1)

        # Step 7: NLL loss
        residual_sq = residual ** 2
        if residual_sq_clamp is not None:
            residual_sq = residual_sq.clamp(max=residual_sq_clamp)  # Only for smoke test
        lambda_diag = sqrt_info_clamped ** 2
        nll = 0.5 * (residual_sq * lambda_diag - 2 * torch.log(sqrt_info_clamped + eps))

        total_nll += stage_weight * nll.mean()

        # Save last-stage variables for logging
        if stage_idx == n_stages - 1:
            residual_sq_last = residual_sq
            lambda_diag_last = lambda_diag
            sqrt_info_last = sqrt_info_clamped
            scale_last = scale

    # Compute separate rot/trans NLL for logging (last stage only)
    # Remember: [:3] = translation, [3:] = rotation
    nll_trans = 0.5 * (residual_sq_last[..., :3] * lambda_diag_last[..., :3]
                       - 2 * torch.log(sqrt_info_last[..., :3] + eps))
    nll_rot = 0.5 * (residual_sq_last[..., 3:] * lambda_diag_last[..., 3:]
                     - 2 * torch.log(sqrt_info_last[..., 3:] + eps))

    # Calibration statistics: d² = Σ λ_k r_k² (expect 3 for rot, 3 for trans)
    d2_trans = (residual_sq_last[..., :3] * lambda_diag_last[..., :3]).sum(dim=-1)
    d2_rot = (residual_sq_last[..., 3:] * lambda_diag_last[..., 3:]).sum(dim=-1)

    return {
        "loss_camera_nll": total_nll / n_stages,
        # For TensorBoard logging:
        "nll_rot": nll_rot.mean().detach(),
        "nll_trans": nll_trans.mean().detach(),
        "sqrt_info_rot_mean": sqrt_info_last[..., 3:].mean().detach(),
        "sqrt_info_trans_mean": sqrt_info_last[..., :3].mean().detach(),
        "d2_rot_mean": d2_rot.mean().detach(),    # expect ~3 if calibrated
        "d2_trans_mean": d2_trans.mean().detach(), # expect ~3 if calibrated
        "scale_mean": scale_last.mean().detach(),
        "scale_std": scale_last.std().detach(),
    }
```

### Step 3: MultitaskLoss Integration

```python
# In MultitaskLoss.__init__, add:
self.camera_nll = camera_nll  # config dict with 'weight', etc.

# In MultitaskLoss.forward():
if "pose_sqrt_info_list" in predictions and self.camera_nll is not None:
    nll_loss_dict = compute_camera_nll_loss(predictions, batch, **self.camera_nll)
    nll_loss = nll_loss_dict["loss_camera_nll"] * self.camera_nll["weight"]
    total_loss = total_loss + nll_loss
    loss_dict.update(nll_loss_dict)
```

---

## Numerical Stability

| Issue | Solution |
|-------|----------|
| MLP outputs negative | `F.softplus(raw) + eps` ensures positive √λ |
| √λ too small (σ → ∞) | Clamp √λ_rot ≥ 0.1, √λ_trans ≥ 0.1 |
| √λ too large (σ → 0) | Clamp √λ_rot ≤ 200, √λ_trans ≤ 100 |
| Log singularity | `log(sqrt_info + eps)` with eps=1e-6 |
| Quaternion drift | Normalize q before SE3 reconstruction |
| Quaternion double-cover | Hemisphere canonicalization (force qw ≥ 0) |
| AMP dtype mismatch | Explicit `valid_mask.to(dtype)` in scale fitting |
| Scale blowup (static frames) | Filter GT ‖t‖ < 2cm in scale fitting |
| All frames stationary | Fallback to scale=1.0 when valid_count < 2 |
| SE(3) log at identity | PyPose handles Taylor expansion internally |
| Extreme residuals | Clamp residual² ≤ 100 (smoke test only) |

---

## Verification Plan

### Pre-Training (MUST DO FIRST!)

1. **Convention verification**: Run `test_pose_encoding_convention()` with REAL model output
2. **SO3 quaternion order**: Verify SO3.tensor() returns `[qx, qy, qz, qw]`
3. **Window-relative test**: Verify T_rel[0] = Identity for all batches
4. **Residual sanity check**: Print rot/trans norm statistics
   ```python
   # Expect: rot_norm mean < 0.5 rad (~30°), trans_norm mean < 0.3 m
   # If rot p90 > 2 rad, check window span or convention!
   ```
5. **Constant sqrt_info baseline**: Compute NLL/χ² with fixed sqrt_info=1 as reference

### Unit Tests

```python
def test_pose_encoding_convention(model, images_tensor):
    """Verify pose_encoding_to_se3 matches pose_encoding_to_extri_intri.

    IMPORTANT: Use real model output, not random input!
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # Get REAL pose_enc from model (not random!)
    with torch.no_grad():
        predictions = model(images_tensor)  # Direct tensor input!
    pose_enc = predictions["pose_enc_list"][-1]  # [B, S, 9]

    # Use ACTUAL image size from input (not hardcoded!)
    image_hw = images_tensor.shape[-2:]  # (H, W)

    # Compare R, t separately (not whole matrix)
    extri_vggt, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_hw)
    T = pose_encoding_to_se3(pose_enc)
    T_matrix = T.matrix()

    R_vggt = extri_vggt[..., :3, :3]
    t_vggt = extri_vggt[..., :3, 3]
    R_pypose = T_matrix[..., :3, :3]
    t_pypose = T_matrix[..., :3, 3]

    assert torch.allclose(R_vggt, R_pypose, atol=1e-5)
    assert torch.allclose(t_vggt, t_pypose, atol=1e-5)
    print("✓ Convention verified")


def test_so3_quaternion_order():
    """Verify SO3.tensor() returns [qx, qy, qz, qw]."""
    q_input = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
    q_input = F.normalize(q_input, dim=-1)

    T = pp.SE3(torch.cat([torch.zeros(1, 3), q_input], dim=-1))
    q_output = T.rotation().tensor()

    assert torch.allclose(q_input, q_output, atol=1e-6)
    print("✓ SO3.tensor() returns [qx, qy, qz, qw]")
```

### Training Verification

- **Optimizer check**: Print trainable params (should be only uncertainty MLP)
- **Smoke test**: Train 100 iterations, verify NLL loss decreases
- **Grad norm monitoring**: Log uncertainty head grad_norm in TensorBoard

### Post-Training Calibration

```python
# Chi-square calibration: d² = Σ_k λ_k * r_k² should follow χ²(6)
d_sq = (lambda_diag * residual_sq).sum(dim=-1)
print(f"mean(d²) = {d_sq.mean():.2f} (expect ~6)")
print(f"p95(d²) = {d_sq.quantile(0.95):.2f} (expect ~12.59)")

# Separate rot/trans (each should be χ²(3)):
# Remember: [:3] = translation, [3:] = rotation
d2_trans = (lambda_diag[..., :3] * residual_sq[..., :3]).sum(dim=-1)
d2_rot = (lambda_diag[..., 3:] * residual_sq[..., 3:]).sum(dim=-1)
print(f"d²_rot mean: {d2_rot.mean():.2f} (expect ~3)")
print(f"d²_trans mean: {d2_trans.mean():.2f} (expect ~3)")
```

---

## Training Configuration

### Data Sampling (CRITICAL!)

**Window sampling MUST use nearby frames (like VGGT's `get_nearby_ids`), NOT uniform random sampling!**

Rationale:
- Uncertainty learning requires coherent motion between frames
- Uniform sampling creates huge baseline jumps → residuals dominated by noise
- Star-graph constraint (0→i) assumes smooth trajectory within window

Recommended settings:
- `get_nearby=True` in dataset config
- `expand_ratio=2.0` (sample within 2x window span)
- Window size S=24 (or whatever VGGT default is)

### Loss Configuration

```yaml
camera_nll:
  weight: 1.0
  pose_encoding_type: "absT_quaR_FoV"
  gamma: 0.6
  sqrt_info_rot_clamp: [0.1, 200.0]
  sqrt_info_trans_clamp: [0.1, 100.0]
  residual_sq_clamp: 100.0  # set to null for formal training
  scale_detach: true
  eps: 1.0e-6
```

---

## Implementation Gotchas

### 1. Last-stage logging variables

The `residual_sq`, `lambda_diag`, `sqrt_info_clamped` used for logging are inside the loop. Explicitly save them:
```python
if stage_idx == n_stages - 1:
    residual_sq_last = residual_sq
    lambda_diag_last = lambda_diag
    sqrt_info_last = sqrt_info_clamped
```

### 2. Baseline sqrt_info=1 must use same pipeline

When computing baseline NLL for comparison, use the SAME NLL computation code (same eps, same clamp logic). Don't compute baseline separately with different constants.

### 3. model(images_tensor) vs model(batch_dict)

Check VGGT's actual forward path. If training uses `model(batch)` with dict input, unit tests should use dict too. Match the exact forward entry point.

### 4. Scale statistics to TensorBoard

Log scale fitting health:
```python
"scale/mean": scale.mean().detach(),
"scale/std": scale.std().detach(),
"scale/valid_count_mean": valid_count.float().mean().detach(),
```
Helps debug if translation uncertainty diverges.

---

## Summary

**Minimal Correct Loop:**
1. **Convention verification** - MUST run unit tests before training
2. Window-relative poses (T_rel = T_0^{-1} @ T_i)
3. Per-window scale on relative translations + **detach()**
4. SE(3) residual: `Log(inv(T_gt) @ T_pred)` (PGO-consistent)
5. 6-dim √λ output with **separate rot/trans clamping**
6. NLL loss with **eps for log stability**, skip frame 0
7. **Freeze original pose head** (verify optimizer only updates uncertainty)
8. **Star-graph constraints** (0→i) for v1 PGO

**Key Safety Measures:**
- `scale.detach()` - prevent cheating channel
- `F.softplus(raw) + eps` - smooth positive sqrt_info (not hard clamp!)
- `q = q / q.norm()` - quaternion normalization before SE3 reconstruction
- `q *= sign(qw)` - hemisphere canonicalization to avoid Log() branch jumps
- Robust scale fitting - filter frames with GT ‖t‖ < 2cm (GT-only, not pred)
- Scale fallback - use scale=1.0 when valid_count < 2 (all stationary)
- `valid_mask.to(dtype)` - explicit dtype for AMP stability
- Dimension consistency - `[..., :3]` = trans, `[..., 3:]` = rot (NEVER mix!)
- Unit tests with REAL model output (not random!)
- Residual sanity check - verify rot/trans magnitudes before training
- Constant sqrt_info baseline - compare NLL/χ² against fixed sqrt_info=1

---

## Suggested Commit Structure

For better review/debug, split implementation into 3 commits:

1. **`lie_algebra.py` + unit tests**
   - `pose_encoding_to_se3`, `compute_window_relative_poses`, `compute_window_scale_batched`, `compute_se3_residual`
   - Unit tests: convention verification, Log/Exp roundtrip, scale fitting

2. **CameraHead output `sqrt_info_list` + forward pass**
   - Add `pose_uncertainty_branch` MLP
   - Return raw sqrt_info (no activation)
   - Update VGGT forward to pass through

3. **Loss integration + TensorBoard logging**
   - `compute_camera_nll_loss()` with softplus+clamp
   - Update `MultitaskLoss` with config switch
   - Add rot/trans separate logging + calibration stats
