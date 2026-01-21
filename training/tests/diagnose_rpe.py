#!/usr/bin/env python3
"""
Diagnostic script to identify RPE issues.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vggt_tum import compute_ate, compute_rpe, rotation_error


def check_rotation_validity(R, name="R"):
    """Check if rotation matrix is valid (det=1, R@R.T=I)."""
    det = np.linalg.det(R)
    orthogonality_error = np.max(np.abs(R @ R.T - np.eye(3)))
    print(f"  {name}: det={det:.6f}, orthogonality_error={orthogonality_error:.2e}")
    return np.isclose(det, 1.0, atol=1e-3) and orthogonality_error < 1e-3


def diagnose_rpe(pred_poses, gt_poses, timestamps=None, delta=1):
    """
    Diagnose RPE issues.
    """
    n_poses = len(pred_poses)
    print(f"\n{'='*60}")
    print("RPE DIAGNOSTICS")
    print(f"{'='*60}")

    # A. Check timestamps
    print(f"\n[A] TIMESTAMPS (frame gaps)")
    if timestamps is not None:
        print(f"  Total frames: {len(timestamps)}")
        gaps = np.diff(timestamps)
        print(f"  Time gaps: min={gaps.min():.3f}s, max={gaps.max():.3f}s, mean={gaps.mean():.3f}s")
        print(f"  First 5 timestamps: {timestamps[:5]}")
        print(f"  First 5 gaps: {gaps[:5]}")
    else:
        print("  No timestamps provided - using index-based delta")

    # B. Check rotation validity
    print(f"\n[B] ROTATION VALIDITY")
    print("  GT rotations:")
    gt_valid = all(check_rotation_validity(gt_poses[i, :3, :3], f"frame_{i}") for i in [0, n_poses//2, -1])
    print("  Pred rotations:")
    pred_valid = all(check_rotation_validity(pred_poses[i, :3, :3], f"frame_{i}") for i in [0, n_poses//2, -1])

    # C. Check translation magnitudes
    print(f"\n[C] TRANSLATION MAGNITUDES")
    gt_trans = np.linalg.norm(gt_poses[:, :3, 3], axis=1)
    pred_trans = np.linalg.norm(pred_poses[:, :3, 3], axis=1)
    print(f"  GT |t|: min={gt_trans.min():.3f}, max={gt_trans.max():.3f}, mean={gt_trans.mean():.3f}")
    print(f"  Pred |t|: min={pred_trans.min():.3f}, max={pred_trans.max():.3f}, mean={pred_trans.mean():.3f}")
    print(f"  Ratio pred/gt: {pred_trans.mean() / gt_trans.mean():.3f}")

    # D. Check relative motion magnitudes
    print(f"\n[D] RELATIVE MOTION MAGNITUDES (delta={delta})")
    gt_rel_trans = []
    gt_rel_rot = []
    pred_rel_trans = []
    pred_rel_rot = []

    for i in range(n_poses - delta):
        # GT relative
        R1_gt, t1_gt = gt_poses[i, :3, :3], gt_poses[i, :3, 3]
        R2_gt, t2_gt = gt_poses[i+delta, :3, :3], gt_poses[i+delta, :3, 3]
        R_rel_gt = R1_gt.T @ R2_gt
        t_rel_gt = R1_gt.T @ (t2_gt - t1_gt)

        gt_rel_trans.append(np.linalg.norm(t_rel_gt))
        angle_gt = rotation_error(R_rel_gt, np.eye(3))
        gt_rel_rot.append(angle_gt)

        # Pred relative
        R1_pred, t1_pred = pred_poses[i, :3, :3], pred_poses[i, :3, 3]
        R2_pred, t2_pred = pred_poses[i+delta, :3, :3], pred_poses[i+delta, :3, 3]
        R_rel_pred = R1_pred.T @ R2_pred
        t_rel_pred = R1_pred.T @ (t2_pred - t1_pred)

        pred_rel_trans.append(np.linalg.norm(t_rel_pred))
        angle_pred = rotation_error(R_rel_pred, np.eye(3))
        pred_rel_rot.append(angle_pred)

    gt_rel_trans = np.array(gt_rel_trans)
    gt_rel_rot = np.array(gt_rel_rot)
    pred_rel_trans = np.array(pred_rel_trans)
    pred_rel_rot = np.array(pred_rel_rot)

    print(f"  GT relative trans: mean={gt_rel_trans.mean()*100:.2f}cm, max={gt_rel_trans.max()*100:.2f}cm")
    print(f"  GT relative rot: mean={gt_rel_rot.mean():.2f}°, max={gt_rel_rot.max():.2f}°")
    print(f"  Pred relative trans: mean={pred_rel_trans.mean()*100:.2f}cm, max={pred_rel_trans.max()*100:.2f}cm")
    print(f"  Pred relative rot: mean={pred_rel_rot.mean():.2f}°, max={pred_rel_rot.max():.2f}°")

    # E. Compare individual relative poses
    print(f"\n[E] INDIVIDUAL RELATIVE POSE ERRORS (first 5)")
    for i in range(min(5, n_poses - delta)):
        R1_gt, t1_gt = gt_poses[i, :3, :3], gt_poses[i, :3, 3]
        R2_gt, t2_gt = gt_poses[i+delta, :3, :3], gt_poses[i+delta, :3, 3]
        R_rel_gt = R1_gt.T @ R2_gt
        t_rel_gt = R1_gt.T @ (t2_gt - t1_gt)

        R1_pred, t1_pred = pred_poses[i, :3, :3], pred_poses[i, :3, 3]
        R2_pred, t2_pred = pred_poses[i+delta, :3, :3], pred_poses[i+delta, :3, 3]
        R_rel_pred = R1_pred.T @ R2_pred
        t_rel_pred = R1_pred.T @ (t2_pred - t1_pred)

        trans_err = np.linalg.norm(t_rel_pred - t_rel_gt)
        rot_err = rotation_error(R_rel_pred, R_rel_gt)

        print(f"  Frame {i}->{i+delta}:")
        print(f"    GT rel trans: {t_rel_gt}")
        print(f"    Pred rel trans: {t_rel_pred}")
        print(f"    Trans error: {trans_err*100:.2f}cm, Rot error: {rot_err:.2f}°")

    # F. Check if Sim3 alignment rotation affects RPE
    print(f"\n[F] EFFECT OF ALIGNMENT ROTATION ON RPE")
    result = compute_ate(pred_poses, gt_poses, align='sim3')
    R_align = result['R_align']
    scale = result['scale']

    print(f"  Alignment rotation angle: {rotation_error(R_align, np.eye(3)):.2f}°")
    print(f"  Alignment scale: {scale:.4f}")

    # Apply alignment to pred poses
    pred_poses_aligned = np.zeros_like(pred_poses)
    for i in range(n_poses):
        R_pred = pred_poses[i, :3, :3]
        t_pred = pred_poses[i, :3, 3]
        # Get camera position, apply Sim3, then convert back
        pos_pred = -R_pred.T @ t_pred
        pos_aligned = scale * (R_align @ pos_pred) + result['aligned_positions'][i] - result['gt_positions'][i]
        # Actually, let's properly align:
        # For rotations: R_aligned = R_pred @ R_align.T
        R_aligned = R_pred @ R_align.T
        # For positions, use the aligned positions from ATE
        pos_aligned = result['aligned_positions'][i]
        t_aligned = -R_aligned @ pos_aligned
        pred_poses_aligned[i, :3, :3] = R_aligned
        pred_poses_aligned[i, :3, 3] = t_aligned

    # Compute RPE on aligned poses
    rpe_trans_raw, rpe_rot_raw = compute_rpe(pred_poses, gt_poses, delta=delta, scale=1.0)
    rpe_trans_aligned, rpe_rot_aligned = compute_rpe(pred_poses_aligned, gt_poses, delta=delta, scale=1.0)

    print(f"\n  RPE without alignment:")
    print(f"    Trans: {rpe_trans_raw*100:.2f}cm, Rot: {rpe_rot_raw:.2f}°")
    print(f"  RPE with full Sim3 alignment (including rotation):")
    print(f"    Trans: {rpe_trans_aligned*100:.2f}cm, Rot: {rpe_rot_aligned:.2f}°")

    # G. Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if gt_rel_rot.mean() > 5:
        print(f"  ⚠ Large GT relative rotations ({gt_rel_rot.mean():.1f}°) - check frame sampling!")
    if not gt_valid:
        print(f"  ⚠ GT rotations may not be valid rotation matrices!")
    if not pred_valid:
        print(f"  ⚠ Pred rotations may not be valid rotation matrices!")
    if rotation_error(R_align, np.eye(3)) > 10:
        print(f"  ⚠ Large alignment rotation ({rotation_error(R_align, np.eye(3)):.1f}°) - coordinate frame mismatch?")


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing with synthetic data (circular trajectory)...")

    n_poses = 24
    gt_poses = np.zeros((n_poses, 3, 4))
    for i in range(n_poses):
        t = i / n_poses
        angle = t * 2 * np.pi * 0.5  # Half circle
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        pos = np.array([np.sin(t * np.pi) * 2, t * 3, 0.5])
        gt_poses[i, :3, :3] = R
        gt_poses[i, :3, 3] = -R @ pos

    # Add noise to create pred
    pred_poses = gt_poses.copy()
    np.random.seed(42)
    for i in range(n_poses):
        # Add small rotation noise
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.randn() * 0.05  # ~3 degrees
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        pred_poses[i, :3, :3] = R_noise @ gt_poses[i, :3, :3]
        # Add translation noise
        pred_poses[i, :3, 3] += np.random.randn(3) * 0.02

    timestamps = np.linspace(0, 10, n_poses)
    diagnose_rpe(pred_poses, gt_poses, timestamps, delta=1)
