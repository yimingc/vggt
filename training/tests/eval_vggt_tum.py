#!/usr/bin/env python3
"""
Evaluate VGGT pose predictions on TUM RGB-D dataset.

Computes:
- ATE (Absolute Trajectory Error) after Umeyama alignment
- RPE (Relative Pose Error) for translation and rotation

Usage:
    python training/tests/eval_vggt_tum.py --tum_dir /path/to/tum --num_frames 8
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least-squares solution for the Umeyama alignment.
    Aligns y to x: finds R, t, s such that x ≈ s * R @ y + t

    For row vectors (N, 3), the aligned result is: s * (y @ R.T) + t

    Args:
        x: (N, 3) reference/target points
        y: (N, 3) source points to align
        with_scale: whether to estimate scale

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scale factor
    """
    assert x.shape == y.shape
    n, dim = x.shape

    # Center the points
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Compute covariance: H = Y.T @ X (for Procrustes: find R s.t. X ≈ Y @ R.T)
    # This is equivalent to sum_i (y_i.T @ x_i) where y_i, x_i are column vectors
    H = y_centered.T @ x_centered

    # SVD: H = U @ S @ V.T
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Rotation: R = V @ U.T (standard Procrustes solution)
    # Handle reflection case
    d = np.ones(dim)
    if np.linalg.det(V @ U.T) < 0:
        d[dim - 1] = -1

    R = V @ np.diag(d) @ U.T

    # Compute scale
    if with_scale:
        var_y = np.sum(y_centered ** 2)
        s = np.sum(S * d) / var_y
    else:
        s = 1.0

    # Compute translation: x = s * R @ y + t => t = x_mean - s * R @ y_mean
    t = x_mean - s * R @ y_mean

    return R, t, s


def extract_camera_positions(poses, convention='w2c'):
    """Extract camera positions from extrinsic matrices.

    Args:
        poses: (N, 3, 4) pose matrices
        convention: 'w2c' - World-to-Camera: P_cam = R @ P_world + t
                           Camera position = -R^T @ t
                    'c2w' - Camera-to-World: P_world = R @ P_cam + t
                           Camera position = t (translation is camera center)
    """
    positions = []
    for i in range(len(poses)):
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        if convention == 'w2c':
            positions.append(-R.T @ t)
        else:  # c2w
            positions.append(t)
    return np.array(positions)


def compute_ate(pred_poses, gt_poses, align='sim3',
                gt_convention='w2c', pred_convention='w2c'):
    """
    Compute Absolute Trajectory Error (translation and rotation).

    Reference: Zhang & Scaramuzza, "A Tutorial on Quantitative Trajectory
    Evaluation for Visual(-Inertial) Odometry", IROS 2018.

    Args:
        pred_poses: (N, 3, 4) predicted poses
        gt_poses: (N, 3, 4) ground truth poses
        align: 'sim3' for Sim3 alignment (with scale),
               'se3' for SE3 alignment (no scale),
               'none' for no alignment
        gt_convention: 'w2c' or 'c2w' for ground truth poses
        pred_convention: 'w2c' or 'c2w' for predicted poses

    Returns:
        dict with keys:
            'trans_rmse': RMSE of position errors (meters)
            'trans_mean': mean position error (meters)
            'rot_rmse': RMSE of rotation errors (degrees)
            'rot_mean': mean rotation error (degrees)
            'aligned_positions': aligned predicted positions
            'gt_positions': ground truth positions
            'scale': scale factor (1.0 if not using Sim3)
            'R_align': alignment rotation matrix
            'is_degenerate': whether degenerate case was detected
    """
    # Assert w2c convention - the rotation alignment logic assumes w2c
    # (right-multiply R_rot.T, and t = -R @ C for aligned_poses)
    assert gt_convention == 'w2c' and pred_convention == 'w2c', \
        f"compute_ate currently only supports w2c convention, got gt={gt_convention}, pred={pred_convention}"

    pred_positions = extract_camera_positions(pred_poses, pred_convention)
    gt_positions = extract_camera_positions(gt_poses, gt_convention)

    # =========================================================================
    # Degenerate detection using GT extent (max distance from centroid)
    # More robust than first-last distance for loop trajectories
    # Note: Only use GT extent - pred extent is unreliable (VGGT is scale-agnostic)
    # =========================================================================
    gt_centroid = gt_positions.mean(axis=0)
    pred_centroid = pred_positions.mean(axis=0)
    extent_gt = np.max(np.linalg.norm(gt_positions - gt_centroid, axis=1))
    extent_pred = np.max(np.linalg.norm(pred_positions - pred_centroid, axis=1))

    # Degenerate if GT trajectory extent is too small (< 30cm)
    # Only check GT - pred scale is arbitrary before alignment
    is_degenerate = extent_gt < 0.3

    # Store original requested alignment mode
    requested_align = align

    # =========================================================================
    # Alignment (with fallback for degenerate cases)
    # =========================================================================
    if align == 'sim3':
        # Try Sim3 alignment first
        R_align, t_align, s_align = umeyama_alignment(gt_positions, pred_positions, with_scale=True)

        # Check for degenerate scale (ill-conditioned Sim3)
        scale_degenerate = s_align < 0.1 or s_align > 10.0 or np.isnan(s_align)

        if is_degenerate or scale_degenerate:
            # Fallback to SE3 (no scale) for degenerate cases
            if is_degenerate:
                print(f"  [Warning] Degenerate trajectory (GT extent={extent_gt*100:.1f} cm < 30 cm): "
                      f"falling back to SE3 alignment (no scale)")
            if scale_degenerate:
                print(f"  [Warning] Degenerate scale ({s_align:.3f}): "
                      f"falling back to SE3 alignment (no scale)")
            R_align, t_align, _ = umeyama_alignment(gt_positions, pred_positions, with_scale=False)
            s_align = 1.0
            align = 'se3'  # Mark actual alignment used

        aligned_positions = s_align * (pred_positions @ R_align.T) + t_align

    elif align == 'se3':
        # SE3 alignment (no scale)
        R_align, t_align, _ = umeyama_alignment(gt_positions, pred_positions, with_scale=False)
        aligned_positions = (pred_positions @ R_align.T) + t_align
        s_align = 1.0

        if is_degenerate:
            print(f"  [Warning] Degenerate trajectory (GT extent={extent_gt*100:.1f} cm < 30 cm): "
                  f"rotation alignment may be unreliable")
    else:
        R_align = np.eye(3)
        aligned_positions = pred_positions
        s_align = 1.0

    # Translation errors
    trans_errors = np.linalg.norm(aligned_positions - gt_positions, axis=1)
    trans_rmse = np.sqrt(np.mean(trans_errors ** 2))
    trans_mean = np.mean(trans_errors)

    # =========================================================================
    # Rotation alignment (with fallback for degenerate cases)
    # For degenerate trajectories, position-based R_align is ill-conditioned
    # Use first-frame rotation alignment as fallback
    # =========================================================================
    if is_degenerate:
        # Degenerate: use first-frame rotation alignment
        # R_rot such that R_pred[0] @ R_rot.T = R_gt[0]
        # => R_rot = R_gt[0].T @ R_pred[0]
        R_pred_0 = pred_poses[0, :3, :3]
        R_gt_0 = gt_poses[0, :3, :3]
        R_rot = R_gt_0.T @ R_pred_0
        print(f"  [Warning] Using first-frame rotation alignment fallback")
    else:
        # Non-degenerate: position-based R_align works well
        R_rot = R_align

    # Compute rotation errors using R_rot
    rot_errors = []
    for i in range(len(pred_poses)):
        R_pred = pred_poses[i, :3, :3]
        R_gt = gt_poses[i, :3, :3]
        # Apply rotation alignment (w2c: right-multiply)
        R_pred_aligned = R_pred @ R_rot.T
        # Compute rotation error
        rot_err = rotation_error(R_pred_aligned, R_gt)
        rot_errors.append(rot_err)

    rot_errors = np.array(rot_errors)
    rot_rmse = np.sqrt(np.mean(rot_errors ** 2))
    rot_mean = np.mean(rot_errors)

    # Compute fully aligned poses (for RPE computation)
    # Use R_rot for rotation alignment, aligned_positions for translation
    # Note: This assumes w2c convention where t = -R @ C
    aligned_poses = np.zeros_like(pred_poses)
    for i in range(len(pred_poses)):
        R_pred = pred_poses[i, :3, :3]
        R_aligned = R_pred @ R_rot.T
        pos_aligned = aligned_positions[i]
        t_aligned = -R_aligned @ pos_aligned  # w2c: t = -R @ C
        aligned_poses[i, :3, :3] = R_aligned
        aligned_poses[i, :3, 3] = t_aligned

    return {
        'trans_rmse': trans_rmse,
        'trans_mean': trans_mean,
        'trans_p50': np.percentile(trans_errors, 50),
        'trans_p90': np.percentile(trans_errors, 90),
        'trans_p99': np.percentile(trans_errors, 99),
        'rot_rmse': rot_rmse,
        'rot_mean': rot_mean,
        'rot_p50': np.percentile(rot_errors, 50),
        'rot_p90': np.percentile(rot_errors, 90),
        'rot_p99': np.percentile(rot_errors, 99),
        'aligned_positions': aligned_positions,
        'gt_positions': gt_positions,
        'scale': s_align,
        'R_align': R_align,
        'aligned_poses': aligned_poses,  # Fully aligned w2c poses
        'is_degenerate': is_degenerate,
        'extent_gt': extent_gt,
        'extent_pred': extent_pred,
        'requested_align': requested_align,
        'actual_align': align,
    }


def rotation_error(R1, R2):
    """Compute rotation error in degrees between two rotation matrices."""
    R_diff = R1.T @ R2
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    angle = np.arccos(trace) * 180 / np.pi
    return angle


def compute_rpe(pred_poses, gt_poses, delta=1, verbose=False):
    """
    Compute Relative Pose Error.

    Args:
        pred_poses: (N, 3, 4) predicted poses (should be aligned to GT frame if needed)
        gt_poses: (N, 3, 4) ground truth poses
        delta: frame interval for computing relative poses
        verbose: if True, print diagnostic information

    Returns:
        dict with keys:
            'trans_rmse': RMSE of relative translation errors (meters)
            'trans_mean': mean relative translation error (meters)
            'trans_p50/p90/p99': percentiles of translation errors
            'rot_rmse': RMSE of relative rotation errors (degrees)
            'rot_mean': mean relative rotation error (degrees)
            'rot_p50/p90/p99': percentiles of rotation errors
    """
    trans_errors = []
    rot_errors = []
    gt_rel_trans_mags = []
    pred_rel_trans_mags = []

    for i in range(len(pred_poses) - delta):
        # Predicted relative pose
        R1_pred = pred_poses[i, :3, :3]
        t1_pred = pred_poses[i, :3, 3]
        R2_pred = pred_poses[i + delta, :3, :3]
        t2_pred = pred_poses[i + delta, :3, 3]

        R_rel_pred = R1_pred.T @ R2_pred
        t_rel_pred = R1_pred.T @ (t2_pred - t1_pred)

        # GT relative pose
        R1_gt = gt_poses[i, :3, :3]
        t1_gt = gt_poses[i, :3, 3]
        R2_gt = gt_poses[i + delta, :3, :3]
        t2_gt = gt_poses[i + delta, :3, 3]

        R_rel_gt = R1_gt.T @ R2_gt
        t_rel_gt = R1_gt.T @ (t2_gt - t1_gt)

        # Errors
        trans_errors.append(np.linalg.norm(t_rel_pred - t_rel_gt))
        rot_errors.append(rotation_error(R_rel_pred, R_rel_gt))

        # Magnitudes for diagnostics
        gt_rel_trans_mags.append(np.linalg.norm(t_rel_gt))
        pred_rel_trans_mags.append(np.linalg.norm(t_rel_pred))

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    if verbose:
        gt_rel_mean = np.mean(gt_rel_trans_mags)
        pred_rel_mean = np.mean(pred_rel_trans_mags)
        print(f"    [RPE Diagnostics] GT rel trans: {gt_rel_mean*100:.1f} cm, "
              f"Pred rel trans: {pred_rel_mean*100:.1f} cm, "
              f"Ratio pred/gt: {pred_rel_mean/gt_rel_mean:.3f}")

    return {
        'trans_rmse': np.sqrt(np.mean(trans_errors ** 2)),
        'trans_mean': np.mean(trans_errors),
        'trans_p50': np.percentile(trans_errors, 50),
        'trans_p90': np.percentile(trans_errors, 90),
        'trans_p99': np.percentile(trans_errors, 99),
        'rot_rmse': np.sqrt(np.mean(rot_errors ** 2)),
        'rot_mean': np.mean(rot_errors),
        'rot_p50': np.percentile(rot_errors, 50),
        'rot_p90': np.percentile(rot_errors, 90),
        'rot_p99': np.percentile(rot_errors, 99),
    }


class MockCommonConf:
    """Mock configuration for dataset."""
    def __init__(self):
        self.img_size = 518
        self.patch_size = 14
        self.debug = False
        self.training = False
        self.get_nearby = True
        self.load_depth = True
        self.inside_random = False
        self.allow_duplicate_img = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.augs = type('obj', (object,), {'scales': None})()


def evaluate_sequence(dataset, seq_index, num_frames, model, device, dtype,
                       sampling='uniform', start_frame=0, end_frame=None,
                       dryrun=False, return_vggt_predictions=False):
    """Evaluate VGGT on a single sequence.

    Args:
        sampling: 'uniform' - evenly spread within [start_frame, end_frame]
                  'consecutive' - consecutive frames starting from start_frame
                  'random' - random frames within [start_frame, end_frame]
        start_frame: starting frame index (default: 0)
        end_frame: ending frame index, inclusive (default: last frame)
        dryrun: if True, use GT as prediction (skip VGGT inference) to verify pipeline
        return_vggt_predictions: if True, also return full VGGT predictions dict for viser
    """
    # Get sequence name and length
    seq_name = dataset.sequence_list[seq_index]
    seq_len = len(dataset.data_store[seq_name])

    # Set default end_frame to last frame
    if end_frame is None:
        end_frame = seq_len - 1

    # Clamp to valid range
    start_frame = max(0, min(start_frame, seq_len - 1))
    end_frame = max(start_frame, min(end_frame, seq_len - 1))
    frame_range = end_frame - start_frame + 1

    # Determine frame IDs based on sampling strategy
    if sampling == 'uniform':
        # Evenly spread num_frames within [start_frame, end_frame]
        ids = np.linspace(start_frame, end_frame, num_frames, dtype=int).tolist()
    elif sampling == 'consecutive':
        # Consecutive frames starting from start_frame
        actual_num = min(num_frames, frame_range)
        ids = list(range(start_frame, start_frame + actual_num))
        if actual_num < num_frames:
            print(f"  Warning: Only {actual_num} frames available in range [{start_frame}, {end_frame}]")
    else:  # random
        # Random frames within [start_frame, end_frame]
        available = list(range(start_frame, end_frame + 1))
        actual_num = min(num_frames, len(available))
        ids = sorted(np.random.choice(available, actual_num, replace=False).tolist())

    # Get data with specified IDs
    batch = dataset.get_data(seq_index=seq_index, img_per_seq=len(ids), ids=ids, aspect_ratio=1.0)

    print(f"\nSequence: {batch['seq_name']} (total {seq_len} frames)")
    print(f"Frame range: [{start_frame}, {end_frame}], Sampling: {sampling}, Frames: {len(ids)}")
    print(f"IDs: {ids[:5]}{'...' + str(ids[-1]) if len(ids) > 5 else ''}")
    if dryrun:
        print("  [DRYRUN MODE] Using GT as prediction")

    # Get GT poses (our loader converts TUM to w2c convention)
    gt_poses = np.stack(batch['extrinsics'], axis=0)

    vggt_predictions = None  # Will be set if return_vggt_predictions=True

    if dryrun:
        # Use GT as prediction to verify pipeline
        pred_poses = gt_poses.copy()
    else:
        # Prepare images for VGGT
        # Note: VGGT model does normalization internally (aggregator.py:200-201)
        # So we only convert to [0, 1] range, do NOT apply mean/std normalization here
        images = np.stack(batch['images'], axis=0)
        images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        images_tensor = torch.from_numpy(images).to(device).to(dtype).unsqueeze(0)

        # Run VGGT
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = model(images_tensor)

        # Get predicted poses
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        pred_extri, pred_intri = pose_encoding_to_extri_intri(
            predictions['pose_enc'], images_tensor.shape[-2:]
        )
        pred_poses = pred_extri[0].float().cpu().numpy()

        # Output uncertainty statistics if available
        if 'pose_log_var_list' in predictions:
            log_var = predictions['pose_log_var_list'][-1]  # [1, S, 6]

            # Convert to sigma (standard deviation): σ = exp(0.5 * log_var)
            sigma = torch.exp(0.5 * log_var)  # [1, S, 6]

            # PyPose se(3) convention: [:3] = trans, [3:] = rot
            sigma_trans = sigma[0, :, :3].cpu().numpy()  # [S, 3]
            sigma_rot = sigma[0, :, 3:].cpu().numpy()    # [S, 3]

            print(f"\n  Uncertainty (from pose_log_var_list - trained uncertainty head):")
            print(f"    σ_trans: mean={sigma_trans.mean():.4f} m, std={sigma_trans.std():.4f} m")
            print(f"    σ_rot:   mean={sigma_rot.mean():.4f} rad ({np.degrees(sigma_rot.mean()):.2f}°)")
            print(f"    σ_trans range: [{sigma_trans.min():.4f}, {sigma_trans.max():.4f}] m")
            print(f"    σ_rot range:   [{sigma_rot.min():.4f}, {sigma_rot.max():.4f}] rad")

        # Store full predictions for viser visualization if requested
        if return_vggt_predictions:
            predictions["extrinsic"] = pred_extri
            predictions["intrinsic"] = pred_intri
            # Convert predictions to numpy for viser
            vggt_predictions = {}
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    # Convert to float32 first (bfloat16 can't be converted to numpy directly)
                    vggt_predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)
                else:
                    vggt_predictions[key] = predictions[key]
            # Add un-normalized images (0-1 range) for viser
            vggt_predictions["images"] = images

    # Both GT (from TUM loader) and VGGT predictions use w2c (world-to-camera) convention
    # Reference: vggt/utils/pose_enc.py - "representing camera from world transformation"
    pose_convention = 'w2c'

    # Compute ATE with Sim3 alignment (includes both translation and rotation errors)
    ate_sim3 = compute_ate(pred_poses, gt_poses, align='sim3',
                           gt_convention=pose_convention, pred_convention=pose_convention)
    scale = ate_sim3['scale']
    R_align = ate_sim3['R_align']

    # Compute ATE with SE3 alignment (no scale)
    ate_se3 = compute_ate(pred_poses, gt_poses, align='se3',
                          gt_convention=pose_convention, pred_convention=pose_convention)

    # RPE (relative pose error) - without and with Sim3 alignment
    # Raw: compare pred vs GT directly (no alignment)
    # Sim3: compare fully aligned pred vs GT
    rpe_raw = compute_rpe(pred_poses, gt_poses, delta=1)
    print(f"\n  Relative Pose Error (δ=1, with Sim3 alignment):")
    rpe_sim3 = compute_rpe(ate_sim3['aligned_poses'], gt_poses, delta=1, verbose=True)

    print(f"\n  ATE - SE3 Alignment (no scale):")
    print(f"    Trans RMSE: {ate_se3['trans_rmse'] * 100:.2f} cm  "
          f"(p50={ate_se3['trans_p50']*100:.2f}, p90={ate_se3['trans_p90']*100:.2f}, p99={ate_se3['trans_p99']*100:.2f})")
    print(f"    Rot RMSE:   {ate_se3['rot_rmse']:.2f}°  "
          f"(p50={ate_se3['rot_p50']:.2f}, p90={ate_se3['rot_p90']:.2f}, p99={ate_se3['rot_p99']:.2f})")

    print(f"\n  ATE - Sim3 Alignment (with scale={scale:.3f}):")
    print(f"    Trans RMSE: {ate_sim3['trans_rmse'] * 100:.2f} cm  "
          f"(p50={ate_sim3['trans_p50']*100:.2f}, p90={ate_sim3['trans_p90']*100:.2f}, p99={ate_sim3['trans_p99']*100:.2f})")
    print(f"    Rot RMSE:   {ate_sim3['rot_rmse']:.2f}°  "
          f"(p50={ate_sim3['rot_p50']:.2f}, p90={ate_sim3['rot_p90']:.2f}, p99={ate_sim3['rot_p99']:.2f})")

    print(f"\n  RPE (δ=1) - No Alignment:")
    print(f"    Trans RMSE: {rpe_raw['trans_rmse'] * 100:.2f} cm  "
          f"(p50={rpe_raw['trans_p50']*100:.2f}, p90={rpe_raw['trans_p90']*100:.2f}, p99={rpe_raw['trans_p99']*100:.2f})")
    print(f"    Rot RMSE:   {rpe_raw['rot_rmse']:.2f}°  "
          f"(p50={rpe_raw['rot_p50']:.2f}, p90={rpe_raw['rot_p90']:.2f}, p99={rpe_raw['rot_p99']:.2f})")

    print(f"\n  RPE (δ=1) - Sim3 Alignment (with scale={scale:.3f}):")
    print(f"    Trans RMSE: {rpe_sim3['trans_rmse'] * 100:.2f} cm  "
          f"(p50={rpe_sim3['trans_p50']*100:.2f}, p90={rpe_sim3['trans_p90']*100:.2f}, p99={rpe_sim3['trans_p99']*100:.2f})")
    print(f"    Rot RMSE:   {rpe_sim3['rot_rmse']:.2f}°  "
          f"(p50={rpe_sim3['rot_p50']:.2f}, p90={rpe_sim3['rot_p90']:.2f}, p99={rpe_sim3['rot_p99']:.2f})")

    results = {
        # ATE SE3
        'ate_trans_rmse_se3': ate_se3['trans_rmse'],
        'ate_trans_mean_se3': ate_se3['trans_mean'],
        'ate_trans_p50_se3': ate_se3['trans_p50'],
        'ate_trans_p90_se3': ate_se3['trans_p90'],
        'ate_trans_p99_se3': ate_se3['trans_p99'],
        'ate_rot_rmse_se3': ate_se3['rot_rmse'],
        'ate_rot_mean_se3': ate_se3['rot_mean'],
        'ate_rot_p50_se3': ate_se3['rot_p50'],
        'ate_rot_p90_se3': ate_se3['rot_p90'],
        'ate_rot_p99_se3': ate_se3['rot_p99'],
        # ATE Sim3
        'ate_trans_rmse_sim3': ate_sim3['trans_rmse'],
        'ate_trans_mean_sim3': ate_sim3['trans_mean'],
        'ate_trans_p50_sim3': ate_sim3['trans_p50'],
        'ate_trans_p90_sim3': ate_sim3['trans_p90'],
        'ate_trans_p99_sim3': ate_sim3['trans_p99'],
        'ate_rot_rmse_sim3': ate_sim3['rot_rmse'],
        'ate_rot_mean_sim3': ate_sim3['rot_mean'],
        'ate_rot_p50_sim3': ate_sim3['rot_p50'],
        'ate_rot_p90_sim3': ate_sim3['rot_p90'],
        'ate_rot_p99_sim3': ate_sim3['rot_p99'],
        # RPE raw
        'rpe_trans_rmse_raw': rpe_raw['trans_rmse'],
        'rpe_trans_p50_raw': rpe_raw['trans_p50'],
        'rpe_trans_p90_raw': rpe_raw['trans_p90'],
        'rpe_trans_p99_raw': rpe_raw['trans_p99'],
        'rpe_rot_rmse_raw': rpe_raw['rot_rmse'],
        'rpe_rot_p50_raw': rpe_raw['rot_p50'],
        'rpe_rot_p90_raw': rpe_raw['rot_p90'],
        'rpe_rot_p99_raw': rpe_raw['rot_p99'],
        # RPE Sim3
        'rpe_trans_rmse_sim3': rpe_sim3['trans_rmse'],
        'rpe_trans_p50_sim3': rpe_sim3['trans_p50'],
        'rpe_trans_p90_sim3': rpe_sim3['trans_p90'],
        'rpe_trans_p99_sim3': rpe_sim3['trans_p99'],
        'rpe_rot_rmse_sim3': rpe_sim3['rot_rmse'],
        'rpe_rot_p50_sim3': rpe_sim3['rot_p50'],
        'rpe_rot_p90_sim3': rpe_sim3['rot_p90'],
        'rpe_rot_p99_sim3': rpe_sim3['rot_p99'],
        # Other
        'pose_convention': pose_convention,
        'scale': scale,
        'pred_positions': ate_sim3['aligned_positions'],
        'gt_positions': ate_sim3['gt_positions'],
        'vggt_predictions': vggt_predictions,
    }

    # Add uncertainty statistics if available
    if not dryrun and 'pose_log_var_list' in predictions:
        log_var = predictions['pose_log_var_list'][-1]  # [1, S, 6]
        sigma = torch.exp(0.5 * log_var)
        sigma_trans = sigma[0, :, :3].cpu().numpy()
        sigma_rot = sigma[0, :, 3:].cpu().numpy()
        results['sigma_trans'] = sigma_trans  # [S, 3]
        results['sigma_rot'] = sigma_rot      # [S, 3]
        results['sigma_trans_mean'] = float(sigma_trans.mean())
        results['sigma_rot_mean'] = float(sigma_rot.mean())

    return results


def load_uncertainty_checkpoint(model, checkpoint_path, device):
    """Load trained uncertainty head weights from checkpoint.

    Args:
        model: VGGT model
        checkpoint_path: Path to uncertainty checkpoint (e.g., best.pt)
        device: Device to load to

    Returns:
        dict with checkpoint metadata (iteration, d2_rot_mean, etc.)
    """
    print(f"\nLoading uncertainty checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load uncertainty head state dict
    uncertainty_state_dict = checkpoint['uncertainty_head_state_dict']

    # Load into model
    model_state = model.state_dict()
    loaded_count = 0
    for name, param in uncertainty_state_dict.items():
        if name in model_state:
            model_state[name].copy_(param.to(device))
            loaded_count += 1
        else:
            print(f"  Warning: {name} not found in model")

    print(f"  Loaded {loaded_count} uncertainty head parameters")
    print(f"  Checkpoint iteration: {checkpoint.get('iteration', 'unknown')}")
    print(f"  Checkpoint d²_rot: {checkpoint.get('d2_rot_mean', 'unknown'):.3f}")
    print(f"  Checkpoint d²_trans: {checkpoint.get('d2_trans_mean', 'unknown'):.3f}")

    return {
        'iteration': checkpoint.get('iteration'),
        'd2_rot_mean': checkpoint.get('d2_rot_mean'),
        'd2_trans_mean': checkpoint.get('d2_trans_mean'),
        'sigma_rot_mean': checkpoint.get('sigma_rot_mean'),
        'sigma_trans_mean': checkpoint.get('sigma_trans_mean'),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate VGGT on TUM RGB-D')
    parser.add_argument('--tum_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./tum_eval_output')
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'consecutive'],
                        help='Frame sampling strategy within [start_frame, end_frame]: uniform (default), random, or consecutive')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Starting frame index (default: 0)')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='Ending frame index, inclusive (default: last frame)')
    parser.add_argument('--dryrun', action='store_true',
                        help='Dry run: use GT as prediction to verify evaluation pipeline (skip VGGT inference)')
    parser.add_argument('--uncertainty_checkpoint', type=str, default=None,
                        help='Path to trained uncertainty head checkpoint (e.g., checkpoints/best.pt)')
    parser.add_argument('--viser', action='store_true', default=True,
                        help='Enable viser 3D visualization (default: enabled)')
    parser.add_argument('--no-viser', dest='viser', action='store_false',
                        help='Disable viser 3D visualization')
    parser.add_argument('--viser_port', type=int, default=8080,
                        help='Port for viser server (default: 8080)')
    parser.add_argument('--viser_conf_threshold', type=float, default=25.0,
                        help='Initial confidence threshold percentage for viser (default: 25.0)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("="*60)
    print("VGGT Evaluation on TUM RGB-D")
    print("="*60)
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Frames per trial: {args.num_frames}")
    end_frame_str = str(args.end_frame) if args.end_frame is not None else "last"
    print(f"Frame range: [{args.start_frame}, {end_frame_str}]")
    print(f"Sampling strategy: {args.sampling}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Uncertainty checkpoint: {args.uncertainty_checkpoint if args.uncertainty_checkpoint else 'None (using default init)'}")
    print(f"Viser visualization: {'enabled (port ' + str(args.viser_port) + ')' if args.viser else 'disabled'}")
    if args.dryrun:
        print(f"*** DRYRUN MODE: Using GT as prediction (skipping VGGT) ***")

    # Load dataset
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.datasets.tum_rgbd import TUMRGBDDataset

    common_conf = MockCommonConf()
    dataset = TUMRGBDDataset(
        common_conf=common_conf,
        split="test",
        TUM_DIR=args.tum_dir,
        min_num_images=args.num_frames,
    )

    # Load VGGT model (skip in dryrun mode)
    uncertainty_checkpoint_meta = None
    if args.dryrun:
        print("\n[DRYRUN] Skipping VGGT model loading")
        model = None
    else:
        print("\nLoading VGGT model...")
        from vggt.models.vggt import VGGT
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()

        # Load uncertainty checkpoint if provided
        if args.uncertainty_checkpoint:
            uncertainty_checkpoint_meta = load_uncertainty_checkpoint(
                model, args.uncertainty_checkpoint, device
            )

    # Run evaluation
    all_results = []
    for trial in range(args.num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{args.num_trials}")
        print("="*60)

        # Request VGGT predictions for viser on the last trial
        return_vggt = args.viser and not args.dryrun and (trial == args.num_trials - 1)
        results = evaluate_sequence(dataset, 0, args.num_frames, model, device, dtype,
                                     sampling=args.sampling, start_frame=args.start_frame,
                                     end_frame=args.end_frame, dryrun=args.dryrun,
                                     return_vggt_predictions=return_vggt)
        all_results.append(results)

    # Summary
    print("\n" + "="*60)
    if args.dryrun:
        print("SUMMARY [DRYRUN - GT vs GT]")
    else:
        print("SUMMARY")
    print("="*60)

    # Extract metrics from results (use last trial for percentiles since they're per-sequence)
    ate_trans_se3 = [r['ate_trans_rmse_se3'] for r in all_results]
    ate_rot_se3 = [r['ate_rot_rmse_se3'] for r in all_results]
    ate_trans_sim3 = [r['ate_trans_rmse_sim3'] for r in all_results]
    ate_rot_sim3 = [r['ate_rot_rmse_sim3'] for r in all_results]
    rpe_trans_raw = [r['rpe_trans_rmse_raw'] for r in all_results]
    rpe_rots_raw = [r['rpe_rot_rmse_raw'] for r in all_results]
    rpe_trans_sim3 = [r['rpe_trans_rmse_sim3'] for r in all_results]
    rpe_rots_sim3 = [r['rpe_rot_rmse_sim3'] for r in all_results]
    scales = [r['scale'] for r in all_results]

    # Get percentiles from last trial (representative)
    last = all_results[-1]

    print(f"\n[ATE - SE3 Alignment (no scale)]")
    print(f"  Trans RMSE: {np.mean(ate_trans_se3)*100:.2f} ± {np.std(ate_trans_se3)*100:.2f} cm")
    print(f"       (p50={last['ate_trans_p50_se3']*100:.2f}, p90={last['ate_trans_p90_se3']*100:.2f}, p99={last['ate_trans_p99_se3']*100:.2f} cm)")
    print(f"  Rot RMSE:   {np.mean(ate_rot_se3):.2f} ± {np.std(ate_rot_se3):.2f}°")
    print(f"       (p50={last['ate_rot_p50_se3']:.2f}, p90={last['ate_rot_p90_se3']:.2f}, p99={last['ate_rot_p99_se3']:.2f}°)")

    print(f"\n[ATE - Sim3 Alignment (with scale)]")
    print(f"  Trans RMSE: {np.mean(ate_trans_sim3)*100:.2f} ± {np.std(ate_trans_sim3)*100:.2f} cm")
    print(f"       (p50={last['ate_trans_p50_sim3']*100:.2f}, p90={last['ate_trans_p90_sim3']*100:.2f}, p99={last['ate_trans_p99_sim3']*100:.2f} cm)")
    print(f"  Rot RMSE:   {np.mean(ate_rot_sim3):.2f} ± {np.std(ate_rot_sim3):.2f}°")
    print(f"       (p50={last['ate_rot_p50_sim3']:.2f}, p90={last['ate_rot_p90_sim3']:.2f}, p99={last['ate_rot_p99_sim3']:.2f}°)")
    print(f"  Scale:      {np.mean(scales):.3f} ± {np.std(scales):.3f}")

    print(f"\n[RPE - No Alignment]")
    print(f"  Trans RMSE: {np.mean(rpe_trans_raw)*100:.2f} ± {np.std(rpe_trans_raw)*100:.2f} cm")
    print(f"       (p50={last['rpe_trans_p50_raw']*100:.2f}, p90={last['rpe_trans_p90_raw']*100:.2f}, p99={last['rpe_trans_p99_raw']*100:.2f} cm)")
    print(f"  Rot RMSE:   {np.mean(rpe_rots_raw):.2f} ± {np.std(rpe_rots_raw):.2f}°")
    print(f"       (p50={last['rpe_rot_p50_raw']:.2f}, p90={last['rpe_rot_p90_raw']:.2f}, p99={last['rpe_rot_p99_raw']:.2f}°)")

    print(f"\n[RPE - Sim3 Alignment (with scale)]")
    print(f"  Trans RMSE: {np.mean(rpe_trans_sim3)*100:.2f} ± {np.std(rpe_trans_sim3)*100:.2f} cm")
    print(f"       (p50={last['rpe_trans_p50_sim3']*100:.2f}, p90={last['rpe_trans_p90_sim3']*100:.2f}, p99={last['rpe_trans_p99_sim3']*100:.2f} cm)")
    print(f"  Rot RMSE:   {np.mean(rpe_rots_sim3):.2f} ± {np.std(rpe_rots_sim3):.2f}°")
    print(f"       (p50={last['rpe_rot_p50_sim3']:.2f}, p90={last['rpe_rot_p90_sim3']:.2f}, p99={last['rpe_rot_p99_sim3']:.2f}°)")

    # Uncertainty summary (if available) - from uncertainty head we added
    if 'sigma_trans_mean' in all_results[0]:
        sigma_trans_means = [r['sigma_trans_mean'] for r in all_results]
        sigma_rot_means = [r['sigma_rot_mean'] for r in all_results]
        print(f"\n[Uncertainty (from trained uncertainty head)]")
        print(f"  σ_trans: {np.mean(sigma_trans_means)*100:.2f} ± {np.std(sigma_trans_means)*100:.2f} cm")
        print(f"  σ_rot:   {np.mean(sigma_rot_means):.4f} ± {np.std(sigma_rot_means):.4f} rad "
              f"({np.degrees(np.mean(sigma_rot_means)):.2f}°)")

    # Save trajectory plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot last trial
    pred_pos = all_results[-1]['pred_positions']
    gt_pos = all_results[-1]['gt_positions']

    pred_label = 'GT (dryrun)' if args.dryrun else 'VGGT (aligned)'
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'g-', linewidth=2, label='GT')
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', linewidth=2, label=pred_label)

    ax.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], c='g', s=100, marker='^')
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], c='r', s=100, marker='^')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    title_prefix = '[DRYRUN] GT vs GT' if args.dryrun else 'VGGT vs GT'
    ate_trans = all_results[-1]["ate_trans_rmse_sim3"] * 100
    ate_rot = all_results[-1]["ate_rot_rmse_sim3"]
    ax.set_title(f'{title_prefix} Trajectory (Sim3 aligned)\nATE: {ate_trans:.1f}cm trans, {ate_rot:.1f}° rot')

    plt.savefig(os.path.join(args.output_dir, 'trajectory_comparison.png'), dpi=150)
    plt.close()
    print(f"\nSaved trajectory plot to: {args.output_dir}/trajectory_comparison.png")

    # Viser 3D visualization
    if args.viser and not args.dryrun:
        vggt_predictions = all_results[-1].get('vggt_predictions')
        if vggt_predictions is not None:
            print(f"\n{'='*60}")
            print("Starting Viser 3D Visualization")
            print("="*60)
            print(f"Open http://localhost:{args.viser_port} in your browser")

            # Import viser_wrapper from demo_viser
            from demo_viser import viser_wrapper

            viser_wrapper(
                vggt_predictions,
                port=args.viser_port,
                init_conf_threshold=args.viser_conf_threshold,
                use_point_map=False,
                background_mode=False,
                mask_sky=False,
                image_folder=None,
            )
        else:
            print("\n[Warning] No VGGT predictions available for viser visualization")
    elif args.viser and args.dryrun:
        print("\n[Note] Viser visualization is not available in dryrun mode")


if __name__ == '__main__':
    main()
