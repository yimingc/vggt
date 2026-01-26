# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SE(3) Lie algebra utilities for pose uncertainty estimation using PyPose.

This module provides utilities for:
- Converting VGGT pose encodings to PyPose SE3 objects
- Computing window-relative poses (relative to frame 0)
- Fitting per-window scale using relative translations
- Computing SE(3) residuals for NLL loss

Dimension Convention (IMPORTANT!):
    PyPose SE3.Log() returns 6-dim vector: [vx, vy, vz, wx, wy, wz]
                                           |--- :3 ---|  |--- 3: ---|
                                           translation   rotation
                                           (meters)      (radians)

    Use TRANS_SLICE and ROT_SLICE constants to avoid hardcoding indices.
"""

import torch
import pypose as pp
from typing import Tuple


def split_se3_tangent(xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split se(3) tangent vector into translation and rotation components.

    PyPose SE3.Log() returns [vx, vy, vz, wx, wy, wz] - translation first, rotation second.
    This function makes the convention explicit and avoids hardcoded indices.

    Args:
        xi: (..., 6) se(3) tangent vector

    Returns:
        v: (..., 3) translation component [vx, vy, vz] in meters
        w: (..., 3) rotation component [wx, wy, wz] in radians
    """
    v = xi[..., :3]  # translation
    w = xi[..., 3:]  # rotation
    return v, w


def concat_se3_tangent(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Concatenate translation and rotation into se(3) tangent vector.

    Inverse of split_se3_tangent.

    Args:
        v: (..., 3) translation component [vx, vy, vz] in meters
        w: (..., 3) rotation component [wx, wy, wz] in radians

    Returns:
        xi: (..., 6) se(3) tangent vector [v, w]
    """
    return torch.cat([v, w], dim=-1)


def pose_encoding_to_se3(pose_enc: torch.Tensor) -> pp.SE3:
    """
    Convert VGGT pose encoding to PyPose SE3.

    VGGT uses quaternion convention [qx, qy, qz, qw] (scalar-last),
    which matches PyPose's convention.

    Args:
        pose_enc: (..., 9) tensor [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]

    Returns:
        pp.SE3 object with shape (...)
    """
    t = pose_enc[..., :3]      # translation [tx, ty, tz]
    q = pose_enc[..., 3:7]     # quaternion [qx, qy, qz, qw]

    # Normalize quaternion for numerical stability
    # (VGGT's activate_pose doesn't normalize, so we do it here)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # PyPose SE3 takes [tx, ty, tz, qx, qy, qz, qw] - t first!
    se3_vec = torch.cat([t, q], dim=-1)
    return pp.SE3(se3_vec)


def extract_camera_positions(T: pp.SE3) -> torch.Tensor:
    """
    Extract camera positions from SE3 poses.

    For world-to-camera transformation T = [R|t]:
        Camera position in world = -R^T @ t

    For relative transformation T_rel (frame_i relative to frame_0):
        This gives camera_i's position in frame_0's coordinate system.

    Args:
        T: pp.SE3 of shape [B, S]

    Returns:
        positions: [B, S, 3] camera positions
    """
    R = T.rotation().matrix()  # [B, S, 3, 3]
    t = T.translation()        # [B, S, 3]
    # Camera position = -R^T @ t
    positions = -torch.einsum('...ij,...j->...i', R.transpose(-1, -2), t)
    return positions


def compute_window_relative_poses(pose_enc: torch.Tensor) -> pp.SE3:
    """
    Convert absolute poses to window-relative (relative to frame 0).

    This avoids the need for global Umeyama/Sim3 alignment and is consistent
    with sliding-window settings. Frame 0 becomes identity.

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

    s* = sum(t_gt . t_pred) / sum(||t_pred||^2)

    Uses ROBUST filtering to exclude near-stationary frames, which prevents
    scale blowup on static sequences.

    Args:
        pred_t_rel: [B, S, 3] predicted relative translations
        gt_t_rel: [B, S, 3] GT relative translations
        detach: If True, detach scale to prevent gradient flow (prevents cheating channel)
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

    scale_raw = numerator / denominator.clamp(min=1e-8)

    # FALLBACK for negative scale: If dot product is negative, GT and Pred motion
    # directions are fundamentally misaligned (possibly due to coordinate system
    # differences or prediction errors). Scale fitting doesn't make sense in this
    # case, so fall back to scale=1.0.
    # NOTE: This can happen when VGGT predictions have rotational offset from GT.
    scale = torch.where(scale_raw > 0, scale_raw, torch.ones_like(scale_raw))
    scale = scale.clamp(min=0.01, max=100.0)

    # FALLBACK: Require at least 2 valid motion frames for robust scale estimation.
    # With only 1 frame, scale fitting is degenerate (single point, no overdetermination).
    # For short windows (S<=2), this means scale=1.0 fallback, which is conservative
    # but avoids unreliable scale estimates.
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
                  |--- :3 ---|  |--- 3: ---|
                  translation   rotation
                  (meters)      (radians)
    """
    T_err = T_rel_gt.Inv() @ T_rel_pred
    residual = T_err.Log()  # [B, S, 6]
    return residual


def reconstruct_scaled_se3(T_rel: pp.SE3, scale: torch.Tensor) -> pp.SE3:
    """
    Reconstruct SE3 with scaled translation while keeping rotation.

    Applies quaternion normalization and hemisphere canonicalization
    for numerical stability.

    Args:
        T_rel: pp.SE3 [B, S] relative poses
        scale: [B] per-window scale factors

    Returns:
        T_rel_scaled: pp.SE3 [B, S] with scaled translations
    """
    # Get translation and scale it
    t_rel = T_rel.translation()  # [B, S, 3]
    t_rel_scaled = scale[:, None, None] * t_rel  # [B, S, 3]

    # Get rotation quaternion
    r_rel = T_rel.rotation()  # pp.SO3 [B, S]
    q = r_rel.tensor()  # [B, S, 4] in [qx, qy, qz, qw] order

    # IMPORTANT: Normalize quaternion to prevent numerical drift!
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # IMPORTANT: Hemisphere canonicalization (q and -q are same rotation)
    # Force qw >= 0 to avoid Log() branch jumps near pi rotation
    sign = torch.where(q[..., 3:4] < 0, -torch.ones_like(q[..., 3:4]), torch.ones_like(q[..., 3:4]))
    q = q * sign

    # Build new SE3 with scaled translation
    T_rel_scaled = pp.SE3(torch.cat([t_rel_scaled, q], dim=-1))
    return T_rel_scaled
