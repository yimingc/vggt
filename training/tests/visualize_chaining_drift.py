#!/usr/bin/env python3
"""
Visualize VGGT trajectory vs GT for:
  Plot 1: Single window (should look great, ~0.85cm ATE)
  Plot 2: Two windows chained via MST (shows drift accumulation)

Usage:
    python training/tests/visualize_chaining_drift.py \
        --tum_dir /home/yiming/Dev/tum_rgbd \
        --tum_sequence rgbd_dataset_freiburg1_desk \
        --uncertainty_checkpoint checkpoints/best_uncertainty_head.pt
"""

import argparse
import numpy as np
import torch
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model_and_data(args):
    """Load VGGT model and TUM dataset."""
    from vggt.models.vggt import VGGT
    from eval_pgo_uncertainty import load_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    # Load model (uncertainty head not needed for trajectory visualization)
    logger.info("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device)
    model.eval()

    # Load dataset
    seq_filter = [args.tum_sequence] if args.tum_sequence else None
    dataset = load_dataset(args.tum_dir, sequences=seq_filter)

    seq_index = 0
    seq_name = dataset.sequence_list[seq_index]
    seq_len = len(dataset.data_store[seq_name])
    logger.info(f"Sequence: {seq_name}, Length: {seq_len}")

    # Load all GT poses
    gt_poses_list = []
    for i in range(seq_len):
        batch = dataset.get_data(seq_index=seq_index, img_per_seq=1, ids=[i], aspect_ratio=1.0)
        extri = np.array(batch['extrinsics'][0])
        T = np.eye(4)
        T[:3, :] = extri
        gt_poses_list.append(T)
    gt_poses = np.stack(gt_poses_list, axis=0)  # [N, 4, 4]

    return model, dataset, gt_poses, seq_index, seq_len, device, dtype


def run_vggt_window(model, dataset, seq_index, frame_ids, device, dtype):
    """Run VGGT on a set of frames, return predicted poses [S, 4, 4]."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    window_size = len(frame_ids)
    batch = dataset.get_data(seq_index=seq_index, img_per_seq=window_size, ids=frame_ids, aspect_ratio=1.0)

    images = np.stack(batch['images'], axis=0)
    images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    images_tensor = torch.from_numpy(images).to(device).unsqueeze(0)

    with torch.no_grad():
        predictions = model(images_tensor)

    pose_enc = predictions['pose_enc_list'][-1]  # [1, S, 9]
    image_hw = images_tensor.shape[-2:]
    pred_extri, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_hw)

    T_abs = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(window_size, -1, -1).clone()
    T_abs[:, :3, :] = pred_extri[0].float()

    return T_abs.cpu()  # [S, 4, 4]


def extract_camera_centers(poses):
    """Extract camera centers from w2c poses. C = -R^T @ t"""
    if isinstance(poses, torch.Tensor):
        poses = poses.numpy()
    centers = []
    for T in poses:
        R = T[:3, :3]
        t = T[:3, 3]
        C = -R.T @ t
        centers.append(C)
    return np.array(centers)  # [N, 3]


def fit_scale_per_window(pred_poses, gt_poses):
    """Fit GT-based metric scale for a window. Returns scale factor."""
    pred_centers = extract_camera_centers(pred_poses)
    gt_centers = extract_camera_centers(gt_poses)

    # Relative to anchor
    pred_rel = pred_centers[1:] - pred_centers[0:1]
    gt_rel = gt_centers[1:] - gt_centers[0:1]

    gt_norms = np.linalg.norm(gt_rel, axis=-1)
    pred_norms = np.linalg.norm(pred_rel, axis=-1)

    valid = gt_norms > 0.02  # 2cm threshold
    if valid.sum() >= 2:
        gt_travel = gt_norms[valid].sum()
        pred_travel = max(pred_norms[valid].sum(), 1e-6)
        scale = gt_travel / pred_travel
        scale = max(0.1, min(10.0, scale))
    else:
        scale = 1.0

    return scale


def apply_scale(poses, scale):
    """Apply scale to w2c translation: T[:3,3] *= scale."""
    if isinstance(poses, np.ndarray):
        scaled = poses.copy()
    else:
        scaled = poses.clone()
        scaled = scaled.numpy()
    scaled[:, :3, 3] *= scale
    return scaled


def chain_two_windows_centers(pred_w0, pred_w1, gt_w0, gt_w1, overlap_frames):
    """
    Chain two windows by aligning CAMERA CENTERS (not composing 4x4 matrices).

    IMPORTANT: Composing 4x4 w2c matrices via inv(T_0) @ T_i creates relative
    transforms where extracting camera centers via -R^T @ t distorts the shape.
    The distortion is: C_rel = C_world + R_i^T @ t_0, which depends on per-frame
    rotation R_i. Since pred and GT have different rotations, their shapes diverge.

    Instead, we:
    1. Extract camera centers from absolute poses (correct in each window's frame)
    2. Use Umeyama on overlap centers to find Sim3 alignment between windows
    3. Transform W1's centers into W0's frame

    Args:
        pred_w0: [S, 4, 4] scaled pred poses for window 0
        pred_w1: [S, 4, 4] scaled pred poses for window 1
        gt_w0: [S, 4, 4] GT poses for window 0
        gt_w1: [S, 4, 4] GT poses for window 1
        overlap_frames: number of overlapping frames

    Returns:
        pred_centers: [N, 3] chained predicted camera centers
        gt_centers: [N, 3] chained GT camera centers
        frame_ids: list of global frame indices
    """
    from eval_vggt_tum import umeyama_alignment

    S = pred_w0.shape[0]
    stride = S - overlap_frames  # = w1_start

    # Extract camera centers from absolute poses (correct formula)
    pred_c_w0 = extract_camera_centers(pred_w0)  # [S, 3]
    pred_c_w1 = extract_camera_centers(pred_w1)  # [S, 3]
    gt_c_w0 = extract_camera_centers(gt_w0)      # [S, 3]
    gt_c_w1 = extract_camera_centers(gt_w1)      # [S, 3]

    # Overlap: W0 frames [stride, S) correspond to W1 frames [0, overlap_frames)
    pred_overlap_w0 = pred_c_w0[stride:]           # [overlap, 3]
    pred_overlap_w1 = pred_c_w1[:overlap_frames]   # [overlap, 3]
    gt_overlap_w0 = gt_c_w0[stride:]
    gt_overlap_w1 = gt_c_w1[:overlap_frames]

    # Find Sim3 to align W1 centers to W0 centers using overlap
    # Want: W0_overlap ≈ s * R @ W1_overlap + t
    R_pred, t_pred, s_pred = umeyama_alignment(
        pred_overlap_w0, pred_overlap_w1, with_scale=True)
    R_gt, t_gt, s_gt = umeyama_alignment(
        gt_overlap_w0, gt_overlap_w1, with_scale=True)

    # Transform ALL W1 centers into W0's frame
    pred_c_w1_aligned = s_pred * (pred_c_w1 @ R_pred.T) + t_pred
    gt_c_w1_aligned = s_gt * (gt_c_w1 @ R_gt.T) + t_gt

    # Build chained trajectory:
    # - W0 for frames 0..S-1
    # - W1 (aligned) for frames S..S+stride-1 (non-overlapping new frames)
    pred_centers = np.zeros((S + stride, 3)) if stride > 0 else np.zeros((S, 3))
    gt_centers = np.zeros_like(pred_centers)
    frame_ids = []

    # W0 frames
    for i in range(S):
        pred_centers[i] = pred_c_w0[i]
        gt_centers[i] = gt_c_w0[i]
        frame_ids.append(i)

    # W1 new frames (after overlap)
    for i in range(overlap_frames, S):
        idx = S + (i - overlap_frames)  # = stride + i
        if idx < len(pred_centers):
            pred_centers[idx] = pred_c_w1_aligned[i]
            gt_centers[idx] = gt_c_w1_aligned[i]
            frame_ids.append(stride + i)

    return pred_centers[:len(frame_ids)], gt_centers[:len(frame_ids)], frame_ids


def chain_two_windows(pred_w0, pred_w1, gt_w0, gt_w1, overlap_frames):
    """
    Chain two windows using MST-style approach (LEGACY - has shape distortion bug).

    WARNING: This function composes 4x4 w2c matrices via inv(T_0) @ T_i.
    Extracting camera centers via -R^T @ t from composed relative poses
    distorts the trajectory shape. Use chain_two_windows_centers() instead.

    Args:
        pred_w0: [S, 4, 4] scaled pred poses for window 0
        pred_w1: [S, 4, 4] scaled pred poses for window 1
        gt_w0: [S, 4, 4] GT poses for window 0
        gt_w1: [S, 4, 4] GT poses for window 1
        overlap_frames: number of overlapping frames

    Returns:
        chained_poses: dict {global_frame_id: [4, 4]}
        chained_gt: dict {global_frame_id: [4, 4]}
    """
    S = pred_w0.shape[0]

    # Window 0: use directly (it's the anchor)
    chained = {}
    chained_gt = {}

    # Window 0 poses (relative to window 0 frame 0)
    T0_anchor_inv = np.linalg.inv(pred_w0[0])
    gt0_anchor_inv = np.linalg.inv(gt_w0[0])

    for i in range(S):
        chained[i] = T0_anchor_inv @ pred_w0[i]
        chained_gt[i] = gt0_anchor_inv @ gt_w0[i]

    w1_start = S - overlap_frames

    T_overlap_w0 = chained[w1_start]
    T_w1_anchor = pred_w1[0]
    T_w1_anchor_inv = np.linalg.inv(T_w1_anchor)

    alignment = T_overlap_w0 @ T_w1_anchor_inv
    gt_alignment = chained_gt[w1_start] @ np.linalg.inv(gt_w1[0])

    for i in range(S):
        global_idx = w1_start + i
        if global_idx not in chained:
            chained[global_idx] = alignment @ pred_w1[i]
            chained_gt[global_idx] = gt_alignment @ gt_w1[i]

    return chained, chained_gt


def average_se3(transforms):
    """
    Compute the average of a list of SE3 transforms.
    Uses quaternion averaging for rotation and arithmetic mean for translation.

    Args:
        transforms: list of [4, 4] numpy arrays

    Returns:
        [4, 4] numpy array - the averaged SE3
    """
    from scipy.spatial.transform import Rotation

    rotations = [T[:3, :3] for T in transforms]
    translations = [T[:3, 3] for T in transforms]

    # Average translation
    avg_t = np.mean(translations, axis=0)

    # Average rotation via quaternions
    quats = []
    for R in rotations:
        r = Rotation.from_matrix(R)
        q = r.as_quat()  # [x, y, z, w]
        quats.append(q)
    quats = np.array(quats)

    # Ensure quaternions are in the same hemisphere (flip if dot < 0)
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]

    avg_q = np.mean(quats, axis=0)
    avg_q = avg_q / np.linalg.norm(avg_q)  # normalize

    avg_R = Rotation.from_quat(avg_q).as_matrix()

    result = np.eye(4)
    result[:3, :3] = avg_R
    result[:3, 3] = avg_t
    return result


def chain_two_windows_avg_alignment(pred_w0, pred_w1, gt_w0, gt_w1, overlap_frames):
    """
    Chain two windows using the AVERAGE alignment from all overlap frames.

    For each overlap frame k, compute T_align_k = T_w0_global[k] @ inv(T_w1[k]).
    Average all T_align_k to get a robust alignment transform.
    """
    S = pred_w0.shape[0]
    w1_start = S - overlap_frames

    # Window 0 poses relative to frame 0
    T0_anchor_inv = np.linalg.inv(pred_w0[0])
    gt0_anchor_inv = np.linalg.inv(gt_w0[0])

    chained = {}
    chained_gt = {}
    for i in range(S):
        chained[i] = T0_anchor_inv @ pred_w0[i]
        chained_gt[i] = gt0_anchor_inv @ gt_w0[i]

    # Compute alignment transform from each overlap frame
    align_transforms = []
    for k in range(overlap_frames):
        # W0's global pose for overlap frame k
        w0_global_k = chained[w1_start + k]
        # W1's local pose for overlap frame k
        w1_local_k = pred_w1[k]
        # Alignment: T_global = T_align @ T_w1_local
        T_align_k = w0_global_k @ np.linalg.inv(w1_local_k)
        align_transforms.append(T_align_k)

    # Average all alignment transforms
    alignment = average_se3(align_transforms)

    # GT alignment (should be exact, just use frame 0)
    gt_alignment = chained_gt[w1_start] @ np.linalg.inv(gt_w1[0])

    # Add window 1's non-overlapping frames using averaged alignment
    for i in range(S):
        global_idx = w1_start + i
        if global_idx not in chained:
            chained[global_idx] = alignment @ pred_w1[i]
            chained_gt[global_idx] = gt_alignment @ gt_w1[i]

    return chained, chained_gt


def plot_trajectory(ax, centers, label, color, marker='o', markersize=3, linewidth=1.5):
    """Plot a 3D trajectory."""
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
            color=color, label=label, linewidth=linewidth)
    # Mark start and end
    ax.scatter(*centers[0], color=color, marker='^', s=80, zorder=5, edgecolors='black', linewidths=0.5)
    ax.scatter(*centers[-1], color=color, marker='s', s=60, zorder=5, edgecolors='black', linewidths=0.5)


def compute_ate_cm(pred_centers, gt_centers):
    """Compute ATE in cm (after centering)."""
    # Center both
    pred_c = pred_centers - pred_centers.mean(axis=0)
    gt_c = gt_centers - gt_centers.mean(axis=0)
    errors = np.linalg.norm(pred_c - gt_c, axis=-1)
    return np.sqrt(np.mean(errors ** 2)) * 100  # cm


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    parser = argparse.ArgumentParser(description='Visualize chaining drift')
    parser.add_argument('--tum_dir', type=str, required=True)
    parser.add_argument('--tum_sequence', type=str, default=None)
    parser.add_argument('--uncertainty_checkpoint', type=str, default=None,
                        help='Not used for visualization, kept for CLI compatibility')
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--w1_start', type=int, default=None,
                        help='Override W1 start frame (default: computed from overlap)')
    parser.add_argument('--output_dir', type=str, default='./eval_uncertainty')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, dataset, gt_poses, seq_index, seq_len, device, dtype = load_model_and_data(args)

    window_size = args.window_size
    if args.w1_start is not None:
        stride = args.w1_start
        overlap_frames = window_size - stride
    else:
        overlap_frames = int(window_size * args.overlap)
        stride = window_size - overlap_frames

    logger.info(f"Config: window_size={window_size}, stride={stride}, overlap={overlap_frames}")

    # =========================================================================
    # Window 0: frames [0, window_size)
    # =========================================================================
    ids_w0 = list(range(0, window_size))
    logger.info(f"\nWindow 0: frames {ids_w0[0]}-{ids_w0[-1]}")

    pred_w0 = run_vggt_window(model, dataset, seq_index, ids_w0, device, dtype)
    gt_w0 = torch.from_numpy(gt_poses[ids_w0]).float()

    scale_w0 = fit_scale_per_window(pred_w0, gt_w0)
    pred_w0_scaled = apply_scale(pred_w0, scale_w0)
    logger.info(f"  Scale: {scale_w0:.3f}")

    # Extract centers for window 0
    pred_centers_w0 = extract_camera_centers(pred_w0_scaled)
    gt_centers_w0 = extract_camera_centers(gt_w0.numpy())

    # Align pred to GT (Sim3) for clean visualization
    from eval_pgo_uncertainty import _compute_ate_impl
    ate_w0 = _compute_ate_impl(
        pred_w0_scaled[:, :3, :], gt_w0.numpy()[:, :3, :],
        align='sim3', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"  Window 0 ATE: {ate_w0['trans_rmse']*100:.2f} cm, {ate_w0['rot_rmse']:.2f}°")

    # Get aligned centers from the ATE result
    aligned_w0_34 = ate_w0['aligned_poses']  # [S, 3, 4] - already aligned
    aligned_centers_w0 = np.array([-p[:3, :3].T @ p[:3, 3] for p in aligned_w0_34])
    gt_centers_w0_plot = np.array([-p[:3, :3].T @ p[:3, 3] for p in gt_w0.numpy()[:, :3, :]])

    # =========================================================================
    # Window 1: frames [stride, stride + window_size)
    # =========================================================================
    w1_start = stride
    ids_w1 = list(range(w1_start, w1_start + window_size))
    logger.info(f"\nWindow 1: frames {ids_w1[0]}-{ids_w1[-1]}")

    pred_w1 = run_vggt_window(model, dataset, seq_index, ids_w1, device, dtype)
    gt_w1 = torch.from_numpy(gt_poses[ids_w1]).float()

    scale_w1 = fit_scale_per_window(pred_w1, gt_w1)
    pred_w1_scaled = apply_scale(pred_w1, scale_w1)
    logger.info(f"  Scale: {scale_w1:.3f}")

    ate_w1 = _compute_ate_impl(
        pred_w1_scaled[:, :3, :], gt_w1.numpy()[:, :3, :],
        align='sim3', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"  Window 1 ATE: {ate_w1['trans_rmse']*100:.2f} cm, {ate_w1['rot_rmse']:.2f}°")

    # =========================================================================
    # Chain windows 0 + 1 (scale-only correction, current approach)
    # =========================================================================
    logger.info(f"\nChaining windows 0+1 (overlap={overlap_frames} frames, stride={stride})")

    chained_pred, chained_gt = chain_two_windows(
        pred_w0_scaled, pred_w1_scaled, gt_w0.numpy(), gt_w1.numpy(),
        overlap_frames=overlap_frames
    )

    # =========================================================================
    # Chain with AVERAGED alignment from all overlap frames
    # =========================================================================
    logger.info(f"\n--- AVERAGED ALIGNMENT: Using all {overlap_frames} overlap frames ---")

    chained_avg, chained_gt_avg = chain_two_windows_avg_alignment(
        pred_w0_scaled, pred_w1_scaled, gt_w0.numpy(), gt_w1.numpy(),
        overlap_frames=overlap_frames
    )

    sorted_ids_avg = sorted(chained_avg.keys())
    chained_avg_arr = np.array([chained_avg[i] for i in sorted_ids_avg])
    chained_gt_avg_arr = np.array([chained_gt_avg[i] for i in sorted_ids_avg])

    ate_avg = _compute_ate_impl(
        chained_avg_arr[:, :3, :], chained_gt_avg_arr[:, :3, :],
        align='sim3', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"Chained ATE (single-frame align):   {41.37:.2f} cm")  # placeholder, computed later
    logger.info(f"Chained ATE (avg {overlap_frames}-frame align): "
                f"{ate_avg['trans_rmse']*100:.2f} cm, {ate_avg['rot_rmse']:.2f}°")

    # Also test: what does the variance of alignment transforms look like?
    T0_anchor_inv_dbg = np.linalg.inv(pred_w0_scaled[0])
    align_transforms_dbg = []
    for k in range(overlap_frames):
        w0_global_k = T0_anchor_inv_dbg @ pred_w0_scaled[stride + k]
        w1_local_k = pred_w1_scaled[k]
        T_align_k = w0_global_k @ np.linalg.inv(w1_local_k)
        align_transforms_dbg.append(T_align_k)

    # Show how much the per-frame alignment transforms vary
    align_translations = np.array([T[:3, 3] for T in align_transforms_dbg])
    from scipy.spatial.transform import Rotation
    align_angles = []
    R_ref = align_transforms_dbg[0][:3, :3]
    for T in align_transforms_dbg:
        R_diff = R_ref.T @ T[:3, :3]
        angle = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
        align_angles.append(angle)
    align_angles = np.array(align_angles)

    logger.info(f"Alignment transform spread (how noisy is single-frame alignment):")
    logger.info(f"  Translation std: {align_translations.std(axis=0)*100} cm, "
                f"norm_std={np.std(np.linalg.norm(align_translations, axis=1))*100:.2f} cm")
    logger.info(f"  Rotation spread: max={align_angles.max():.2f}°, "
                f"mean={align_angles[1:].mean():.2f}° (relative to frame 0's alignment)")
    logger.info(f"--- END AVERAGED ALIGNMENT ---")

    # =========================================================================
    # Also chain with full Sim3-aligned poses (hypothesis test)
    # =========================================================================
    logger.info(f"\n--- HYPOTHESIS TEST: Chain with Sim3-aligned poses ---")

    # Get Sim3-aligned poses for each window (using GT for alignment)
    # This is what we'd get if we had perfect Sim3 alignment per-window
    aligned_w0_44 = np.zeros((window_size, 4, 4))
    for i in range(window_size):
        aligned_w0_44[i, :3, :] = ate_w0['aligned_poses'][i]
        aligned_w0_44[i, 3, 3] = 1.0

    aligned_w1_44 = np.zeros((window_size, 4, 4))
    aligned_w1_34_arr = ate_w1['aligned_poses']
    for i in range(window_size):
        aligned_w1_44[i, :3, :] = aligned_w1_34_arr[i]
        aligned_w1_44[i, 3, 3] = 1.0

    # --- Method A: Chain with single-frame junction alignment (old approach) ---
    chained_sim3, chained_gt_sim3 = chain_two_windows(
        aligned_w0_44, aligned_w1_44, gt_w0.numpy(), gt_w1.numpy(),
        overlap_frames=overlap_frames
    )
    sorted_ids_sim3 = sorted(chained_sim3.keys())
    chained_sim3_arr = np.array([chained_sim3[i] for i in sorted_ids_sim3])
    chained_gt_sim3_arr = np.array([chained_gt_sim3[i] for i in sorted_ids_sim3])

    ate_chained_sim3 = _compute_ate_impl(
        chained_sim3_arr[:, :3, :], chained_gt_sim3_arr[:, :3, :],
        align='sim3', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"  (A) GT Sim3 per-window → chain via junction frame:  "
                f"{ate_chained_sim3['trans_rmse']*100:.2f} cm, {ate_chained_sim3['rot_rmse']:.2f}°")

    # --- Method B: Direct concatenation (no junction alignment needed) ---
    # After GT Sim3, both windows are in GT world frame already.
    # Just take W0 for frames 0-63, W1 for frames 64-95.
    concat_pred = {}
    concat_gt = {}
    for i in range(window_size):  # W0: frames 0-63
        concat_pred[i] = aligned_w0_44[i]
        concat_gt[i] = gt_poses[i]
    for i in range(overlap_frames, window_size):  # W1: frames 64-95 (skip overlap)
        global_idx = stride + i  # = 32 + i
        concat_pred[global_idx] = aligned_w1_44[i]
        concat_gt[global_idx] = gt_poses[global_idx]

    sorted_concat = sorted(concat_pred.keys())
    concat_pred_arr = np.array([concat_pred[i] for i in sorted_concat])
    concat_gt_arr = np.array([concat_gt[i] for i in sorted_concat])

    # Evaluate WITHOUT alignment (both already in GT frame)
    ate_concat_none = _compute_ate_impl(
        concat_pred_arr[:, :3, :], concat_gt_arr[:, :3, :],
        align='none', gt_convention='w2c', pred_convention='w2c'
    )
    # Also with Sim3 for comparison
    ate_concat_sim3 = _compute_ate_impl(
        concat_pred_arr[:, :3, :], concat_gt_arr[:, :3, :],
        align='sim3', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"  (B) GT Sim3 per-window → direct concat (no align):  "
                f"{ate_concat_none['trans_rmse']*100:.2f} cm, {ate_concat_none['rot_rmse']:.2f}°")
    logger.info(f"  (B) GT Sim3 per-window → direct concat (Sim3 eval): "
                f"{ate_concat_sim3['trans_rmse']*100:.2f} cm, {ate_concat_sim3['rot_rmse']:.2f}°")

    # --- Method C: For overlap region, show W0 vs W1 in GT frame ---
    # Both should be close to GT, but different from each other
    w0_ov_in_gt = aligned_w0_44[stride:]  # W0's frames 32-63, GT-aligned
    w1_ov_in_gt = aligned_w1_44[:overlap_frames]  # W1's frames 32-63, GT-aligned
    gt_ov_direct = gt_poses[stride:window_size]

    w0_ov_c_gt = extract_camera_centers(w0_ov_in_gt)
    w1_ov_c_gt = extract_camera_centers(w1_ov_in_gt)
    gt_ov_c_gt = extract_camera_centers(gt_ov_direct)

    w0_vs_gt = np.linalg.norm(w0_ov_c_gt - gt_ov_c_gt, axis=-1) * 100
    w1_vs_gt = np.linalg.norm(w1_ov_c_gt - gt_ov_c_gt, axis=-1) * 100
    w0_vs_w1 = np.linalg.norm(w0_ov_c_gt - w1_ov_c_gt, axis=-1) * 100

    logger.info(f"\n  Overlap in GT frame:")
    logger.info(f"    W0 vs GT:  mean={w0_vs_gt.mean():.2f} cm, max={w0_vs_gt.max():.2f} cm")
    logger.info(f"    W1 vs GT:  mean={w1_vs_gt.mean():.2f} cm, max={w1_vs_gt.max():.2f} cm")
    logger.info(f"    W0 vs W1:  mean={w0_vs_w1.mean():.2f} cm, max={w0_vs_w1.max():.2f} cm")

    # =========================================================================
    # Test 2: Use ALL overlap frames to compute best-fit Sim3 between windows
    # (no GT needed — purely from overlap correspondences)
    # =========================================================================
    logger.info(f"\n--- TEST 2: Overlap-based Sim3 alignment (no GT) ---")

    # Window 0's predictions for overlap frames (in W0's local frame, relative to W0 frame 0)
    T0_anchor_inv_test = np.linalg.inv(pred_w0_scaled[0])
    w0_overlap_poses = np.array([T0_anchor_inv_test @ pred_w0_scaled[i] for i in range(stride, window_size)])
    w0_ov_centers = extract_camera_centers(w0_overlap_poses)  # [overlap_frames, 3]

    # Window 1's predictions for the same frames (in W1's local frame, relative to W1 frame 0)
    T1_anchor_inv = np.linalg.inv(pred_w1_scaled[0])
    w1_overlap_poses = np.array([T1_anchor_inv @ pred_w1_scaled[i] for i in range(overlap_frames)])
    w1_ov_centers = extract_camera_centers(w1_overlap_poses)  # [overlap_frames, 3]

    # Compute best-fit Sim3: find s, R, t such that w0_ov ≈ s*R*w1_ov + t
    # Using Umeyama algorithm (similar to what _compute_ate_impl does)
    from eval_pgo_uncertainty import _compute_ate_impl as _ate

    # Treat w0 as "GT" and w1 as "pred" to find Sim3: w1 → w0
    # We need poses in [N, 3, 4] format for _compute_ate_impl
    # But Umeyama just needs point correspondences, let me implement directly
    def umeyama_sim3(src, dst):
        """Find Sim3 (s, R, t) such that dst ≈ s*R*src + t. src, dst: [N, 3]"""
        n = src.shape[0]
        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        src_c = src - mu_src
        dst_c = dst - mu_dst

        var_src = np.sum(src_c ** 2) / n

        cov = dst_c.T @ src_c / n  # [3, 3]
        U, D, Vt = np.linalg.svd(cov)

        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1

        R = U @ S @ Vt
        s = np.trace(np.diag(D) @ S) / var_src
        t = mu_dst - s * R @ mu_src

        return s, R, t

    s_ov, R_ov, t_ov = umeyama_sim3(w1_ov_centers, w0_ov_centers)
    logger.info(f"Overlap Sim3: scale={s_ov:.4f}, "
                f"rotation={np.degrees(np.arccos(np.clip((np.trace(R_ov)-1)/2, -1, 1))):.2f}°, "
                f"translation={np.linalg.norm(t_ov)*100:.2f} cm")

    # Apply Sim3 to window 1 and re-chain
    # Transform W1's local poses to W0's local frame
    w1_in_w0_frame = {}
    for i in range(window_size):
        T_w1_local = T1_anchor_inv @ pred_w1_scaled[i]  # W1 local frame (w2c)
        # Apply world-frame Sim3 to w2c pose:
        #   Umeyama: C_w0 = s*R_ov*C_w1 + t_ov  (on camera centers)
        #   For w2c: R_new = R_old @ R_ov^T, t_new = s*t_old - R_new @ t_ov
        R_old = T_w1_local[:3, :3]
        t_old = T_w1_local[:3, 3]
        R_new = R_old @ R_ov.T
        t_new = s_ov * t_old - R_new @ t_ov
        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = t_new
        global_idx = stride + i
        w1_in_w0_frame[global_idx] = T_new

    # Build chained trajectory: W0 frames + aligned W1 frames
    chained_ov = {}
    chained_gt_ov = {}
    for i in range(window_size):
        chained_ov[i] = T0_anchor_inv_test @ pred_w0_scaled[i]
        chained_gt_ov[i] = np.linalg.inv(gt_w0.numpy()[0]) @ gt_w0.numpy()[i]
    for idx, T in w1_in_w0_frame.items():
        if idx not in chained_ov:  # don't overwrite W0
            chained_ov[idx] = T
            chained_gt_ov[idx] = np.linalg.inv(gt_w0.numpy()[0]) @ gt_poses[idx]

    sorted_ov = sorted(chained_ov.keys())
    chained_ov_arr = np.array([chained_ov[i] for i in sorted_ov])
    chained_gt_ov_arr = np.array([chained_gt_ov[i] for i in sorted_ov])

    ate_ov = _ate(chained_ov_arr[:, :3, :], chained_gt_ov_arr[:, :3, :],
                  align='sim3', gt_convention='w2c', pred_convention='w2c')
    logger.info(f"Chained ATE (overlap Sim3 alignment, Sim3-eval): "
                f"{ate_ov['trans_rmse']*100:.2f} cm, {ate_ov['rot_rmse']:.2f}°")

    # Check overlap disagreement after Sim3 alignment
    w1_ov_aligned = np.array([s_ov * R_ov @ w1_ov_centers[i] + t_ov for i in range(overlap_frames)])
    ov_err_after = np.linalg.norm(w0_ov_centers - w1_ov_aligned, axis=-1)
    logger.info(f"Overlap disagreement after Sim3: mean={ov_err_after.mean()*100:.2f} cm, "
                f"max={ov_err_after.max()*100:.2f} cm")

    # =========================================================================
    # Test 3: Understand the 41cm — is it a Sim3 alignment artifact?
    # =========================================================================
    logger.info(f"\n--- TEST 3: Isolate chaining error vs Sim3 artifact ---")

    # GT rotation between frame 0 and frame 32
    R_gt_0 = gt_poses[0][:3, :3]
    R_gt_32 = gt_poses[stride][:3, :3]
    R_gt_rel = R_gt_0.T @ R_gt_32
    gt_rot_angle = np.degrees(np.arccos(np.clip((np.trace(R_gt_rel) - 1) / 2, -1, 1)))
    logger.info(f"GT rotation frame 0→32: {gt_rot_angle:.2f}° (Umeyama found {np.degrees(np.arccos(np.clip((np.trace(R_ov)-1)/2, -1, 1))):.2f}°)")

    # The "true" chaining error: per-frame error in the overlap region
    # (after single-frame alignment at frame 32)
    # This is the ACTUAL disagreement between windows, not the Sim3 artifact
    logger.info(f"\nPer-frame errors in chained trajectory (single-frame alignment):")

    # W0 region: should be exact (these poses come from W0, no chaining)
    w0_pred_local = np.array([np.linalg.inv(pred_w0_scaled[0]) @ pred_w0_scaled[i] for i in range(window_size)])
    w0_gt_local = np.array([np.linalg.inv(gt_w0.numpy()[0]) @ gt_w0.numpy()[i] for i in range(window_size)])

    # W1 new region: chained through junction
    # Already have chained_pred from chain_two_windows
    sorted_ids_all = sorted(chained_pred.keys())
    pred_centers_chained = extract_camera_centers(np.array([chained_pred[i] for i in sorted_ids_all]))
    gt_centers_chained = extract_camera_centers(np.array([chained_gt[i] for i in sorted_ids_all]))

    # Per-frame error WITHOUT any alignment (both rel to frame 0)
    per_frame_raw = np.linalg.norm(pred_centers_chained - gt_centers_chained, axis=-1)
    logger.info(f"  W0 first half  [0-{stride-1}]:   mean={per_frame_raw[:stride].mean()*100:.2f} cm")
    logger.info(f"  W0 second half [{stride}-{window_size-1}]:  mean={per_frame_raw[stride:window_size].mean()*100:.2f} cm")
    if len(per_frame_raw) > window_size:
        logger.info(f"  W1 new frames [{window_size}-{sorted_ids_all[-1]}]: mean={per_frame_raw[window_size:].mean()*100:.2f} cm")

    # What's the trajectory shape error? Align each half separately
    # W0 alone: Sim3 align
    w0_centers_pred = pred_centers_chained[:window_size]
    w0_centers_gt = gt_centers_chained[:window_size]
    # Already know this: 1.27 cm

    # Full trajectory WITH per-segment alignment (no-Sim3 needed, just center)
    # What if we center both trajectories? (subtract mean)
    pred_centered = pred_centers_chained - pred_centers_chained.mean(axis=0)
    gt_centered = gt_centers_chained - gt_centers_chained.mean(axis=0)
    centered_err = np.linalg.norm(pred_centered - gt_centered, axis=-1)
    logger.info(f"\n  Centered (subtract mean): RMSE={np.sqrt(np.mean(centered_err**2))*100:.2f} cm")

    logger.info(f"--- END TEST 3 ---")

    # =========================================================================
    # Test 4: Are overlap relative transforms consistent across windows?
    # If yes → windows differ only by a single global transform (reference frame)
    # If no  → VGGT's internal geometry changes between windows
    # =========================================================================
    logger.info(f"\n--- TEST 4: Relative transform consistency in overlap ---")

    # For overlap frames [32..63], compute relative transform to frame 32 (first overlap frame)
    # Window 0: T_rel_w0(k) = T_w0[32]^{-1} @ T_w0[k],  k=32..63
    # Window 1: T_rel_w1(k) = T_w1[0]^{-1}  @ T_w1[k-32], k=32..63
    # These should be identical if windows differ only by reference frame

    T_w0_32_inv = np.linalg.inv(pred_w0_scaled[stride])  # inv(T_w0[32])
    T_w1_0_inv = np.linalg.inv(pred_w1_scaled[0])         # inv(T_w1[0]) = inv(T_w1's frame 32)

    trans_errors = []
    rot_errors = []
    for k in range(overlap_frames):
        # Relative to first overlap frame
        T_rel_w0 = T_w0_32_inv @ pred_w0_scaled[stride + k]  # W0's relative transform
        T_rel_w1 = T_w1_0_inv @ pred_w1_scaled[k]             # W1's relative transform

        # Translation difference
        C_w0 = -T_rel_w0[:3, :3].T @ T_rel_w0[:3, 3]
        C_w1 = -T_rel_w1[:3, :3].T @ T_rel_w1[:3, 3]
        trans_err = np.linalg.norm(C_w0 - C_w1)
        trans_errors.append(trans_err)

        # Rotation difference
        R_diff = T_rel_w0[:3, :3].T @ T_rel_w1[:3, :3]
        rot_err = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
        rot_errors.append(rot_err)

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    logger.info(f"Relative transforms (frame 32 → frame k), W0 vs W1:")
    logger.info(f"  Translation diff: mean={trans_errors.mean()*100:.3f} cm, "
                f"max={trans_errors.max()*100:.3f} cm, "
                f"median={np.median(trans_errors)*100:.3f} cm")
    logger.info(f"  Rotation diff:    mean={rot_errors.mean():.3f}°, "
                f"max={rot_errors.max():.3f}°, "
                f"median={np.median(rot_errors):.3f}°")

    # Show per-frame breakdown (every 8th frame)
    logger.info(f"\n  Per-frame breakdown (relative to overlap frame 0):")
    logger.info(f"  {'Frame':>6} {'dt':>4} {'Trans diff (cm)':>16} {'Rot diff (°)':>14}")
    for k in range(0, overlap_frames, 4):
        global_frame = stride + k
        logger.info(f"  {global_frame:>6} {k:>4} {trans_errors[k]*100:>16.3f} {rot_errors[k]:>14.3f}")
    # Always show last frame
    k = overlap_frames - 1
    global_frame = stride + k
    logger.info(f"  {global_frame:>6} {k:>4} {trans_errors[k]*100:>16.3f} {rot_errors[k]:>14.3f}")

    # Now repeat with UNSCALED poses to remove scale confound
    # (rotation is already scale-independent, so this only affects translation)
    pred_w0_unscaled = pred_w0.numpy() if isinstance(pred_w0, torch.Tensor) else pred_w0
    pred_w1_unscaled = pred_w1.numpy() if isinstance(pred_w1, torch.Tensor) else pred_w1

    T_w0_32_inv_us = np.linalg.inv(pred_w0_unscaled[stride])
    T_w1_0_inv_us = np.linalg.inv(pred_w1_unscaled[0])

    trans_errors_us = []
    for k in range(overlap_frames):
        T_rel_w0_us = T_w0_32_inv_us @ pred_w0_unscaled[stride + k]
        T_rel_w1_us = T_w1_0_inv_us @ pred_w1_unscaled[k]
        C_w0_us = -T_rel_w0_us[:3, :3].T @ T_rel_w0_us[:3, 3]
        C_w1_us = -T_rel_w1_us[:3, :3].T @ T_rel_w1_us[:3, 3]
        # Normalize to unit length to compare direction (scale-free)
        if np.linalg.norm(C_w0_us) > 1e-6 and np.linalg.norm(C_w1_us) > 1e-6:
            dir_w0 = C_w0_us / np.linalg.norm(C_w0_us)
            dir_w1 = C_w1_us / np.linalg.norm(C_w1_us)
            dir_err = np.degrees(np.arccos(np.clip(np.dot(dir_w0, dir_w1), -1, 1)))
        else:
            dir_err = 0.0
        # Also compare magnitudes (ratio)
        mag_w0 = np.linalg.norm(C_w0_us)
        mag_w1 = np.linalg.norm(C_w1_us)
        mag_ratio = mag_w0 / max(mag_w1, 1e-8)
        trans_errors_us.append({'dir_err': dir_err, 'mag_ratio': mag_ratio,
                                'mag_w0': mag_w0, 'mag_w1': mag_w1})

    dir_errs = np.array([e['dir_err'] for e in trans_errors_us[1:]])  # skip frame 0 (both identity)
    mag_ratios = np.array([e['mag_ratio'] for e in trans_errors_us[1:]])

    logger.info(f"\n  Unscaled analysis (separating scale from geometry):")
    logger.info(f"  Rotation diff (scale-free): mean={rot_errors[1:].mean():.3f}°, max={rot_errors[1:].max():.3f}°")
    logger.info(f"  Direction diff (scale-free): mean={dir_errs.mean():.3f}°, max={dir_errs.max():.3f}°")
    logger.info(f"  Magnitude ratio (W0/W1):     mean={mag_ratios.mean():.4f}, "
                f"std={mag_ratios.std():.4f}, "
                f"expected={scale_w0/scale_w1:.4f}")

    logger.info(f"\n  Per-frame breakdown:")
    logger.info(f"  {'Frame':>6} {'dt':>4} {'Rot diff°':>10} {'Dir diff°':>10} {'Mag ratio':>10} {'|C_w0|':>8} {'|C_w1|':>8}")
    for k in range(0, overlap_frames, 4):
        if k == 0:
            continue
        global_frame = stride + k
        e = trans_errors_us[k]
        logger.info(f"  {global_frame:>6} {k:>4} {rot_errors[k]:>10.3f} {e['dir_err']:>10.3f} "
                    f"{e['mag_ratio']:>10.4f} {e['mag_w0']:>8.4f} {e['mag_w1']:>8.4f}")
    k = overlap_frames - 1
    global_frame = stride + k
    e = trans_errors_us[k]
    logger.info(f"  {global_frame:>6} {k:>4} {rot_errors[k]:>10.3f} {e['dir_err']:>10.3f} "
                f"{e['mag_ratio']:>10.4f} {e['mag_w0']:>8.4f} {e['mag_w1']:>8.4f}")

    # Verdict
    if dir_errs.mean() < 2.0 and rot_errors[1:].mean() < 2.0:
        logger.info(f"\n  ✓ CONSISTENT: Relative geometry matches (direction <2°, rotation <2°)")
        logger.info(f"    Scale ratio explains the translation magnitude difference")
    elif dir_errs.mean() < 5.0:
        logger.info(f"\n  ~ PARTIALLY CONSISTENT: Small geometric differences exist")
    else:
        logger.info(f"\n  ✗ INCONSISTENT: Significant geometric differences between windows")

    logger.info(f"--- END TEST 4 ---")

    # Convert to arrays (sorted by frame index)
    sorted_ids = sorted(chained_pred.keys())
    chained_pred_arr = np.array([chained_pred[i] for i in sorted_ids])  # [N, 4, 4]
    chained_gt_arr = np.array([chained_gt[i] for i in sorted_ids])

    # =========================================================================
    # DEBUG: Verify chaining correctness
    # =========================================================================
    logger.info(f"\n--- CHAINING DIAGNOSTICS ---")

    # 1. Verify GT chaining is exact
    gt_direct = gt_poses[:len(sorted_ids)]  # direct GT poses
    gt_direct_rel = np.array([np.linalg.inv(gt_direct[0]) @ gt_direct[i] for i in range(len(gt_direct))])
    chained_gt_rel = np.array([np.linalg.inv(chained_gt_arr[0]) @ chained_gt_arr[i] for i in range(len(chained_gt_arr))])
    gt_chain_error = np.linalg.norm(
        extract_camera_centers(gt_direct_rel) - extract_camera_centers(chained_gt_rel), axis=-1
    )
    logger.info(f"1. GT chaining error (should be ~0): max={gt_chain_error.max()*100:.4f} cm, mean={gt_chain_error.mean()*100:.4f} cm")

    # 2. Overlap region: compare predictions from two windows for shared frames
    logger.info(f"\n2. Overlap region prediction disagreement (frames {stride}-{window_size-1}):")
    pred_centers_w0_all = extract_camera_centers(pred_w0_scaled)
    pred_centers_w1_all = extract_camera_centers(pred_w1_scaled)

    # Window 0 sees overlap frames at indices [stride, ..., window_size-1]
    # Window 1 sees overlap frames at indices [0, ..., overlap_frames-1]
    # Both in their own coordinate systems, so we need to align first
    # Let's compare in the chained coordinate frame
    T0_anchor_inv = np.linalg.inv(pred_w0_scaled[0])
    w0_overlap_global = np.array([T0_anchor_inv @ pred_w0_scaled[i] for i in range(stride, window_size)])

    # Window 1 in chained frame (using the alignment from chain_two_windows)
    T_overlap_w0 = T0_anchor_inv @ pred_w0_scaled[stride]  # window 0's pose at overlap start
    alignment = T_overlap_w0 @ np.linalg.inv(pred_w1_scaled[0])
    w1_overlap_global = np.array([alignment @ pred_w1_scaled[i] for i in range(overlap_frames)])

    overlap_centers_w0 = extract_camera_centers(w0_overlap_global)
    overlap_centers_w1 = extract_camera_centers(w1_overlap_global)
    overlap_disagreement = np.linalg.norm(overlap_centers_w0 - overlap_centers_w1, axis=-1)
    logger.info(f"   Position disagreement: mean={overlap_disagreement.mean()*100:.2f} cm, "
                f"max={overlap_disagreement.max()*100:.2f} cm, "
                f"min={overlap_disagreement.min()*100:.2f} cm")

    # Rotation disagreement in overlap
    rot_disagree = []
    for i in range(overlap_frames):
        R_w0 = w0_overlap_global[i][:3, :3]
        R_w1 = w1_overlap_global[i][:3, :3]
        R_diff = R_w0.T @ R_w1
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        rot_disagree.append(np.degrees(angle))
    rot_disagree = np.array(rot_disagree)
    logger.info(f"   Rotation disagreement: mean={rot_disagree.mean():.2f}°, "
                f"max={rot_disagree.max():.2f}°, min={rot_disagree.min():.2f}°")

    # 3. Scale impact
    logger.info(f"\n3. Scale analysis:")
    logger.info(f"   W0 scale: {scale_w0:.4f}, W1 scale: {scale_w1:.4f}")
    logger.info(f"   Scale ratio (W1/W0): {scale_w1/scale_w0:.4f} ({(scale_w1/scale_w0 - 1)*100:.1f}% difference)")

    # 4. What does chained pred trajectory look like vs GT?
    chained_pred_centers = extract_camera_centers(chained_pred_arr)
    chained_gt_centers = extract_camera_centers(chained_gt_arr)
    raw_errors = np.linalg.norm(chained_pred_centers - chained_gt_centers, axis=-1)
    logger.info(f"\n4. Raw (no alignment) position error (pred vs GT both relative to frame 0):")
    logger.info(f"   W0 region (frames 0-{stride-1}): mean={raw_errors[:stride].mean()*100:.2f} cm")
    logger.info(f"   Overlap (frames {stride}-{window_size-1}): mean={raw_errors[stride:window_size].mean()*100:.2f} cm")
    if len(raw_errors) > window_size:
        logger.info(f"   W1 new (frames {window_size}-{sorted_ids[-1]}): mean={raw_errors[window_size:].mean()*100:.2f} cm")
    logger.info(f"   Overall: mean={raw_errors.mean()*100:.2f} cm")

    # 4b. Same-frame comparison: use Sim3 per-segment to isolate chaining error
    # W0 segment alone
    w0_pred_34 = chained_pred_arr[:window_size, :3, :]
    w0_gt_34 = chained_gt_arr[:window_size, :3, :]
    ate_w0_in_chain = _compute_ate_impl(w0_pred_34, w0_gt_34, align='sim3',
                                         gt_convention='w2c', pred_convention='w2c')
    logger.info(f"\n4b. W0 segment of chained (Sim3-aligned): {ate_w0_in_chain['trans_rmse']*100:.2f} cm "
                f"(should match standalone: {ate_w0['trans_rmse']*100:.2f} cm)")

    # W1 new segment alone (frames window_size onward)
    if len(sorted_ids) > window_size:
        w1_new_pred_34 = chained_pred_arr[window_size:, :3, :]
        w1_new_gt_34 = chained_gt_arr[window_size:, :3, :]
        ate_w1_new = _compute_ate_impl(w1_new_pred_34, w1_new_gt_34, align='sim3',
                                        gt_convention='w2c', pred_convention='w2c')
        logger.info(f"   W1-new segment of chained (Sim3-aligned): {ate_w1_new['trans_rmse']*100:.2f} cm")

    # 4c. Overlap region: compare chained pred vs chained pred from w1 perspective
    # For overlap frames, we have TWO predictions. Which is closer to GT?
    logger.info(f"\n4c. Overlap frames: W0 pred vs W1 pred vs GT (all Sim3-aligned per-window)")
    # W0's prediction for overlap frames (already Sim3-aligned from ate_w0)
    w0_aligned_overlap = aligned_centers_w0[stride:]  # frames 32-63 from W0 Sim3-aligned
    # W1's prediction for overlap frames (already Sim3-aligned from ate_w1)
    aligned_w1_34 = ate_w1['aligned_poses']
    aligned_centers_w1 = np.array([-p[:3, :3].T @ p[:3, 3] for p in aligned_w1_34])
    w1_aligned_overlap = aligned_centers_w1[:overlap_frames]  # frames 0-31 from W1 = global 32-63

    gt_overlap_from_w0 = gt_centers_w0_plot[stride:]  # GT frames 32-63
    gt_overlap_from_w1 = np.array([-p[:3, :3].T @ p[:3, 3] for p in gt_w1.numpy()[:overlap_frames, :3, :]])

    err_w0_overlap = np.linalg.norm(w0_aligned_overlap - gt_overlap_from_w0, axis=-1)
    err_w1_overlap = np.linalg.norm(w1_aligned_overlap - gt_overlap_from_w1, axis=-1)
    logger.info(f"   W0's error on overlap (Sim3): mean={err_w0_overlap.mean()*100:.2f} cm")
    logger.info(f"   W1's error on overlap (Sim3): mean={err_w1_overlap.mean()*100:.2f} cm")

    # 5. Compute chained ATE with both Sim3 and no alignment
    ate_chained_none = _compute_ate_impl(
        chained_pred_arr[:, :3, :], chained_gt_arr[:, :3, :],
        align='none', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"\n5. Chained ATE (no alignment): {ate_chained_none['trans_rmse']*100:.2f} cm, {ate_chained_none['rot_rmse']:.2f}°")

    # Compute chained ATE with Sim3 alignment
    ate_chained = _compute_ate_impl(
        chained_pred_arr[:, :3, :], chained_gt_arr[:, :3, :],
        align='sim3', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"   Chained ATE (Sim3):         {ate_chained['trans_rmse']*100:.2f} cm, {ate_chained['rot_rmse']:.2f}°")
    logger.info(f"   Sim3 scale: {ate_chained['scale']:.4f}")

    # =========================================================================
    # DIAGNOSTIC TEST SUITE: Why does Sim3 fail on chained trajectories?
    # =========================================================================
    from eval_vggt_tum import umeyama_alignment, extract_camera_positions

    chained_pred_centers = extract_camera_positions(chained_pred_arr[:, :3, :], 'w2c')
    chained_gt_centers = extract_camera_positions(chained_gt_arr[:, :3, :], 'w2c')
    n_total = len(chained_pred_centers)

    logger.info(f"\n{'='*70}")
    logger.info(f"DIAGNOSTIC TEST SUITE: Sim3 alignment on chained trajectories")
    logger.info(f"{'='*70}")
    logger.info(f"Total frames in chained trajectory: {n_total}")
    logger.info(f"W0 contributes frames 0-{window_size-1} ({window_size} frames)")
    logger.info(f"W1 contributes new frames {window_size}-{sorted_ids[-1]} ({n_total - window_size} frames)")

    # TEST A: Per-frame raw position errors (no alignment)
    raw_pos_errors = np.linalg.norm(chained_pred_centers - chained_gt_centers, axis=-1) * 100
    logger.info(f"\n--- TEST A: Raw position errors (no alignment) ---")
    logger.info(f"Mean: {raw_pos_errors.mean():.2f} cm")
    logger.info(f"W0 segment: {raw_pos_errors[:window_size].mean():.2f} cm")
    if n_total > window_size:
        logger.info(f"W1 new segment: {raw_pos_errors[window_size:].mean():.2f} cm")
    for i in [0, 1, window_size//2, window_size-1, min(window_size, n_total-1)]:
        if i < n_total:
            logger.info(f"  Frame {sorted_ids[i]:>3}: pred=({chained_pred_centers[i]*100}) GT=({chained_gt_centers[i]*100}) err={raw_pos_errors[i]:.2f} cm")

    # TEST B: Umeyama on W0 only, then apply to full trajectory
    logger.info(f"\n--- TEST B: W0-only Sim3 applied to full trajectory ---")
    R_w0, t_w0, s_w0 = umeyama_alignment(
        chained_gt_centers[:window_size], chained_pred_centers[:window_size], with_scale=True)
    aligned_w0only = s_w0 * (chained_pred_centers @ R_w0.T) + t_w0
    err_w0only = np.linalg.norm(aligned_w0only - chained_gt_centers, axis=-1) * 100
    logger.info(f"Sim3 from W0 only: scale={s_w0:.4f}")
    logger.info(f"W0 segment (aligned by W0 Sim3): {err_w0only[:window_size].mean():.2f} cm RMSE={np.sqrt(np.mean(err_w0only[:window_size]**2)):.2f} cm")
    if n_total > window_size:
        logger.info(f"W1 new segment (aligned by W0 Sim3): {err_w0only[window_size:].mean():.2f} cm RMSE={np.sqrt(np.mean(err_w0only[window_size:]**2)):.2f} cm")
    logger.info(f"Full trajectory (aligned by W0 Sim3): RMSE={np.sqrt(np.mean(err_w0only**2)):.2f} cm")

    # TEST C: Umeyama on full trajectory (should match ate_chained)
    logger.info(f"\n--- TEST C: Full-trajectory Sim3 alignment ---")
    R_full, t_full, s_full = umeyama_alignment(
        chained_gt_centers, chained_pred_centers, with_scale=True)
    aligned_full = s_full * (chained_pred_centers @ R_full.T) + t_full
    err_full = np.linalg.norm(aligned_full - chained_gt_centers, axis=-1) * 100
    logger.info(f"Sim3 from full trajectory: scale={s_full:.4f}")
    logger.info(f"W0 segment: {err_full[:window_size].mean():.2f} cm")
    if n_total > window_size:
        logger.info(f"W1 new: {err_full[window_size:].mean():.2f} cm")
    logger.info(f"Full trajectory: RMSE={np.sqrt(np.mean(err_full**2)):.2f} cm")
    logger.info(f"(Should match chained ATE Sim3: {ate_chained['trans_rmse']*100:.2f} cm)")

    # TEST D: Compare W0 Sim3 vs Full Sim3
    logger.info(f"\n--- TEST D: Sim3 parameter comparison ---")
    logger.info(f"W0-only Sim3: scale={s_w0:.6f}")
    logger.info(f"Full Sim3:    scale={s_full:.6f}")
    logger.info(f"Scale difference: {abs(s_w0 - s_full):.6f} ({abs(s_w0 - s_full)/s_w0*100:.2f}%)")
    # Rotation difference
    R_diff = R_w0.T @ R_full
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    rot_diff_deg = np.degrees(np.arccos(trace))
    logger.info(f"Rotation difference: {rot_diff_deg:.4f}°")
    logger.info(f"Translation offset: {np.linalg.norm(t_w0 - t_full)*100:.2f} cm")

    # TEST E: GT trajectory extent and shape stats
    logger.info(f"\n--- TEST E: Trajectory shape analysis ---")
    gt_extent = np.max(np.linalg.norm(chained_gt_centers - chained_gt_centers.mean(axis=0), axis=1))
    pred_extent = np.max(np.linalg.norm(chained_pred_centers - chained_pred_centers.mean(axis=0), axis=1))
    logger.info(f"GT extent: {gt_extent*100:.2f} cm")
    logger.info(f"Pred extent: {pred_extent*100:.2f} cm")
    logger.info(f"Extent ratio: {pred_extent/gt_extent:.4f}")
    gt_spread = np.linalg.norm(chained_gt_centers[-1] - chained_gt_centers[0])
    pred_spread = np.linalg.norm(chained_pred_centers[-1] - chained_pred_centers[0])
    logger.info(f"Start-to-end distance: GT={gt_spread*100:.2f} cm, Pred={pred_spread*100:.2f} cm, ratio={pred_spread/gt_spread:.4f}")

    # TEST F: Chain using NO per-window GT scale correction
    logger.info(f"\n--- TEST F: Chain WITHOUT per-window GT scale correction ---")
    # Use raw predictions (no scale correction)
    chained_raw, chained_gt_raw = chain_two_windows(
        pred_w0, pred_w1, gt_w0.numpy(), gt_w1.numpy(),
        overlap_frames=overlap_frames
    )
    sorted_ids_raw = sorted(chained_raw.keys())
    chained_raw_arr = np.array([chained_raw[i] for i in sorted_ids_raw])
    chained_gt_raw_arr = np.array([chained_gt_raw[i] for i in sorted_ids_raw])
    ate_raw_sim3 = _compute_ate_impl(
        chained_raw_arr[:, :3, :], chained_gt_raw_arr[:, :3, :],
        align='sim3', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"No scale correction, Sim3 ATE: {ate_raw_sim3['trans_rmse']*100:.2f} cm (scale={ate_raw_sim3['scale']:.4f})")

    # TEST G: Chain using SAME scale for both windows
    logger.info(f"\n--- TEST G: Chain with UNIFORM scale (average of W0, W1) ---")
    avg_scale = (scale_w0 + scale_w1) / 2
    pred_w0_uniform = apply_scale(pred_w0, avg_scale)
    pred_w1_uniform = apply_scale(pred_w1, avg_scale)
    chained_unif, chained_gt_unif = chain_two_windows(
        pred_w0_uniform, pred_w1_uniform, gt_w0.numpy(), gt_w1.numpy(),
        overlap_frames=overlap_frames
    )
    sorted_ids_unif = sorted(chained_unif.keys())
    chained_unif_arr = np.array([chained_unif[i] for i in sorted_ids_unif])
    chained_gt_unif_arr = np.array([chained_gt_unif[i] for i in sorted_ids_unif])
    ate_unif_sim3 = _compute_ate_impl(
        chained_unif_arr[:, :3, :], chained_gt_unif_arr[:, :3, :],
        align='sim3', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"Uniform scale={avg_scale:.4f}, Sim3 ATE: {ate_unif_sim3['trans_rmse']*100:.2f} cm (scale={ate_unif_sim3['scale']:.4f})")

    # TEST H: Use W0's prediction for ALL frames (duplicate W0 as "W1")
    logger.info(f"\n--- TEST H: Duplicate W0 as W1 (perfect overlap) ---")
    chained_dup, chained_gt_dup = chain_two_windows(
        pred_w0_scaled, pred_w0_scaled, gt_w0.numpy(), gt_w0.numpy(),
        overlap_frames=overlap_frames
    )
    sorted_ids_dup = sorted(chained_dup.keys())
    chained_dup_arr = np.array([chained_dup[i] for i in sorted_ids_dup])
    chained_gt_dup_arr = np.array([chained_gt_dup[i] for i in sorted_ids_dup])
    ate_dup = _compute_ate_impl(
        chained_dup_arr[:, :3, :], chained_gt_dup_arr[:, :3, :],
        align='sim3', gt_convention='w2c', pred_convention='w2c'
    )
    logger.info(f"W0 duplicated as W1, Sim3 ATE: {ate_dup['trans_rmse']*100:.2f} cm")
    logger.info(f"  (Should be ~same as standalone W0: {ate_w0['trans_rmse']*100:.2f} cm)")

    # TEST I: Per-frame errors breakdown for Sim3-aligned chained trajectory
    logger.info(f"\n--- TEST I: Per-frame Sim3-aligned errors (full-trajectory Sim3) ---")
    for i in range(0, n_total, max(1, n_total // 20)):
        marker = " <-- W1 new" if i >= window_size else ""
        logger.info(f"  Frame {sorted_ids[i]:>3}: {err_full[i]:.2f} cm{marker}")
    if n_total - 1 != (n_total - 1) // max(1, n_total // 20) * max(1, n_total // 20):
        logger.info(f"  Frame {sorted_ids[-1]:>3}: {err_full[-1]:.2f} cm <-- last")

    # TEST J: CORRECT chaining using camera centers (no 4x4 composition)
    logger.info(f"\n--- TEST J: Correct chaining via camera centers (no 4x4 distortion) ---")
    pred_cc, gt_cc, cc_frame_ids = chain_two_windows_centers(
        pred_w0_scaled, pred_w1_scaled, gt_w0.numpy(), gt_w1.numpy(),
        overlap_frames=overlap_frames
    )
    logger.info(f"Chained frames: {len(cc_frame_ids)} (W0: {window_size}, W1 new: {len(cc_frame_ids) - window_size})")

    # Sim3 align the chained camera centers to GT centers
    R_cc, t_cc, s_cc = umeyama_alignment(gt_cc, pred_cc, with_scale=True)
    pred_cc_aligned = s_cc * (pred_cc @ R_cc.T) + t_cc
    err_cc = np.linalg.norm(pred_cc_aligned - gt_cc, axis=-1) * 100
    logger.info(f"Sim3 scale: {s_cc:.4f}")
    logger.info(f"W0 segment: RMSE={np.sqrt(np.mean(err_cc[:window_size]**2)):.2f} cm")
    if len(err_cc) > window_size:
        logger.info(f"W1 new:     RMSE={np.sqrt(np.mean(err_cc[window_size:]**2)):.2f} cm")
    rmse_cc = np.sqrt(np.mean(err_cc ** 2))
    logger.info(f"Full trajectory: RMSE={rmse_cc:.2f} cm")
    logger.info(f">>> COMPARE: old chaining (4x4 composition) = {ate_chained['trans_rmse']*100:.2f} cm")
    logger.info(f">>> COMPARE: standalone W0 = {ate_w0['trans_rmse']*100:.2f} cm")

    # Also try: chain centers using single-frame alignment (like original code, but on centers)
    logger.info(f"\n--- TEST K: Chain centers using single-frame alignment at junction ---")
    pred_c_w0_k = extract_camera_centers(pred_w0_scaled)
    pred_c_w1_k = extract_camera_centers(pred_w1_scaled)
    gt_c_w0_k = extract_camera_centers(gt_w0.numpy())
    gt_c_w1_k = extract_camera_centers(gt_w1.numpy())
    stride_k = window_size - overlap_frames

    # Single-frame: align W1's first frame center to W0's corresponding frame center
    # W1's frame 0 should match W0's frame [stride]
    # For a single point, we can only do translation (not rotation/scale)
    offset_pred = pred_c_w0_k[stride_k] - pred_c_w1_k[0]
    offset_gt = gt_c_w0_k[stride_k] - gt_c_w1_k[0]

    pred_centers_k = list(pred_c_w0_k)  # frames 0-63
    gt_centers_k = list(gt_c_w0_k)
    for i in range(overlap_frames, window_size):
        pred_centers_k.append(pred_c_w1_k[i] + offset_pred)
        gt_centers_k.append(gt_c_w1_k[i] + offset_gt)
    pred_centers_k = np.array(pred_centers_k)
    gt_centers_k = np.array(gt_centers_k)

    R_k, t_k, s_k = umeyama_alignment(gt_centers_k, pred_centers_k, with_scale=True)
    pred_cc_aligned_k = s_k * (pred_centers_k @ R_k.T) + t_k
    err_k = np.linalg.norm(pred_cc_aligned_k - gt_centers_k, axis=-1) * 100
    rmse_k = np.sqrt(np.mean(err_k ** 2))
    logger.info(f"Single-frame offset chaining: RMSE={rmse_k:.2f} cm (Sim3 scale={s_k:.4f})")

    logger.info(f"\n{'='*70}")
    logger.info(f"END DIAGNOSTIC TEST SUITE")
    logger.info(f"{'='*70}")
    logger.info(f"--- END DIAGNOSTICS ---\n")

    aligned_chained_34 = ate_chained['aligned_poses']
    aligned_chained_centers = np.array([-p[:3, :3].T @ p[:3, 3] for p in aligned_chained_34])
    chained_gt_centers_plot = np.array([-p[:3, :3].T @ p[:3, 3] for p in chained_gt_arr[:, :3, :]])

    # =========================================================================
    # Plot 1: Single Window
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))

    # 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    plot_trajectory(ax1, gt_centers_w0_plot, 'GT', 'blue')
    plot_trajectory(ax1, aligned_centers_w0, 'VGGT (Sim3-aligned)', 'red')
    ax1.set_title(f'Window 0: 3D Trajectory\nATE: {ate_w0["trans_rmse"]*100:.2f} cm, {ate_w0["rot_rmse"]:.2f}°')
    ax1.legend(fontsize=9)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')

    # XY projection
    ax2 = fig.add_subplot(222)
    ax2.plot(gt_centers_w0_plot[:, 0], gt_centers_w0_plot[:, 1], 'b-', label='GT', linewidth=2)
    ax2.plot(aligned_centers_w0[:, 0], aligned_centers_w0[:, 1], 'r-', label='VGGT', linewidth=2)
    ax2.scatter(gt_centers_w0_plot[0, 0], gt_centers_w0_plot[0, 1], c='blue', marker='^', s=100, zorder=5)
    ax2.scatter(aligned_centers_w0[0, 0], aligned_centers_w0[0, 1], c='red', marker='^', s=100, zorder=5)
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # XZ projection
    ax3 = fig.add_subplot(223)
    ax3.plot(gt_centers_w0_plot[:, 0], gt_centers_w0_plot[:, 2], 'b-', label='GT', linewidth=2)
    ax3.plot(aligned_centers_w0[:, 0], aligned_centers_w0[:, 2], 'r-', label='VGGT', linewidth=2)
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Per-frame error
    ax4 = fig.add_subplot(224)
    per_frame_err = np.linalg.norm(aligned_centers_w0 - gt_centers_w0_plot, axis=-1) * 100  # cm
    ax4.plot(range(len(per_frame_err)), per_frame_err, 'k-', linewidth=1.5)
    ax4.fill_between(range(len(per_frame_err)), 0, per_frame_err, alpha=0.3)
    ax4.set_title(f'Per-Frame Error (RMSE={ate_w0["trans_rmse"]*100:.2f} cm)')
    ax4.set_xlabel('Frame index')
    ax4.set_ylabel('Error (cm)')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Single Window (64 frames) — VGGT is excellent per-window',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path_1 = os.path.join(args.output_dir, 'plot1_single_window.png')
    fig.savefig(out_path_1, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {out_path_1}")
    plt.close(fig)

    # =========================================================================
    # Plot 2: Two Windows Chained
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))

    # 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    plot_trajectory(ax1, chained_gt_centers_plot, 'GT', 'blue')
    plot_trajectory(ax1, aligned_chained_centers, 'VGGT Chained (Sim3)', 'red')

    # Mark window boundary
    boundary_idx = window_size  # where window 1 starts (non-overlap region)
    if boundary_idx < len(aligned_chained_centers):
        ax1.scatter(*aligned_chained_centers[boundary_idx], color='green', marker='D', s=100,
                    zorder=6, edgecolors='black', linewidths=1, label='Window boundary')

    ax1.set_title(f'Windows 0+1 Chained: 3D Trajectory\nATE: {ate_chained["trans_rmse"]*100:.2f} cm, {ate_chained["rot_rmse"]:.2f}°')
    ax1.legend(fontsize=9)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')

    # XY projection
    ax2 = fig.add_subplot(222)
    n_w0 = window_size
    n_overlap = overlap_frames
    # Color-code: window 0 portion, overlap, window 1 new portion
    ax2.plot(chained_gt_centers_plot[:, 0], chained_gt_centers_plot[:, 1],
             'b-', label='GT', linewidth=2)
    # Window 0 region
    ax2.plot(aligned_chained_centers[:n_w0, 0], aligned_chained_centers[:n_w0, 1],
             'r-', label='VGGT (W0)', linewidth=2)
    # Window 1 new region (from overlap point onward)
    ax2.plot(aligned_chained_centers[n_w0-1:, 0], aligned_chained_centers[n_w0-1:, 1],
             color='orange', linestyle='-', label='VGGT (W1, chained)', linewidth=2)
    # Mark overlap region
    overlap_start = stride
    ax2.axvline(x=aligned_chained_centers[overlap_start, 0], color='green', linestyle='--',
                alpha=0.5, label=f'Overlap start (frame {overlap_start})')
    ax2.scatter(aligned_chained_centers[0, 0], aligned_chained_centers[0, 1],
                c='red', marker='^', s=100, zorder=5)
    ax2.set_title('XY Projection')
    ax2.legend(fontsize=8)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # XZ projection
    ax3 = fig.add_subplot(223)
    ax3.plot(chained_gt_centers_plot[:, 0], chained_gt_centers_plot[:, 2],
             'b-', label='GT', linewidth=2)
    ax3.plot(aligned_chained_centers[:n_w0, 0], aligned_chained_centers[:n_w0, 2],
             'r-', label='VGGT (W0)', linewidth=2)
    ax3.plot(aligned_chained_centers[n_w0-1:, 0], aligned_chained_centers[n_w0-1:, 2],
             color='orange', linestyle='-', label='VGGT (W1, chained)', linewidth=2)
    ax3.set_title('XZ Projection')
    ax3.legend(fontsize=8)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Per-frame error with window boundaries
    ax4 = fig.add_subplot(224)
    per_frame_err_chained = np.linalg.norm(aligned_chained_centers - chained_gt_centers_plot, axis=-1) * 100
    ax4.plot(sorted_ids, per_frame_err_chained, 'k-', linewidth=1.5)
    ax4.fill_between(sorted_ids, 0, per_frame_err_chained, alpha=0.3)
    ax4.axvline(x=stride, color='green', linestyle='--', alpha=0.7, label=f'W1 starts (frame {stride})')
    ax4.axvline(x=window_size, color='orange', linestyle='--', alpha=0.7, label=f'W0 ends (frame {window_size})')
    ax4.set_title(f'Per-Frame Error (RMSE={ate_chained["trans_rmse"]*100:.2f} cm)')
    ax4.set_xlabel('Frame index')
    ax4.set_ylabel('Error (cm)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Two Windows Chained ({len(sorted_ids)} frames) — Drift from chaining\n'
                 f'W0 ATE: {ate_w0["trans_rmse"]*100:.2f} cm  |  '
                 f'W1 ATE: {ate_w1["trans_rmse"]*100:.2f} cm  |  '
                 f'Chained ATE: {ate_chained["trans_rmse"]*100:.2f} cm',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path_2 = os.path.join(args.output_dir, 'plot2_two_windows_chained.png')
    fig.savefig(out_path_2, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {out_path_2}")
    plt.close(fig)

    # =========================================================================
    # Plot 3: Both windows Sim3-aligned to GT — the key diagnostic plot
    # Shows: each window matches GT beautifully, but they disagree in the overlap
    # =========================================================================

    # We already have aligned_centers_w0 (W0 Sim3-aligned, frames 0-63)
    # and aligned_centers_w1 (W1 Sim3-aligned, frames 32-95, but in W1's GT frame)
    # We need W1 aligned in the SAME GT frame as W0.

    # W0 Sim3-aligned poses are in gt_w0's frame (GT frames 0-63)
    # W1 Sim3-aligned poses are in gt_w1's frame (GT frames 32-95)
    # Both GT frames share the same global GT coordinate system, so we just need
    # to get GT centers for the full range

    gt_full_centers = extract_camera_centers(gt_poses[:window_size + stride])  # frames 0-95

    # W0 Sim3-aligned centers (frames 0-63 in GT world frame)
    # aligned_centers_w0 is already in GT frame (from Sim3 alignment to gt_w0)

    # W1 Sim3-aligned centers (frames 32-95 in GT world frame)
    # aligned_centers_w1 = already computed above from ate_w1['aligned_poses']
    # But these are aligned to gt_w1 (GT frames 32-95), which IS the same world frame
    aligned_w1_34 = ate_w1['aligned_poses']
    aligned_centers_w1_full = np.array([-p[:3, :3].T @ p[:3, 3] for p in aligned_w1_34])

    # GT for W0 and W1 ranges
    gt_centers_for_w0 = gt_full_centers[:window_size]  # frames 0-63
    gt_centers_for_w1 = gt_full_centers[stride:stride + window_size]  # frames 32-95

    # Per-frame error for each window
    err_w0 = np.linalg.norm(aligned_centers_w0 - gt_centers_for_w0, axis=-1) * 100
    err_w1 = np.linalg.norm(aligned_centers_w1_full - gt_centers_for_w1, axis=-1) * 100

    # Overlap disagreement between W0 and W1 (both Sim3-aligned to GT)
    # W0 overlap: frames 32-63 → aligned_centers_w0[stride:]
    # W1 overlap: frames 32-63 → aligned_centers_w1_full[:overlap_frames]
    w0_overlap_aligned = aligned_centers_w0[stride:]
    w1_overlap_aligned = aligned_centers_w1_full[:overlap_frames]
    overlap_mutual_err = np.linalg.norm(w0_overlap_aligned - w1_overlap_aligned, axis=-1) * 100

    # ---- Figure: 2x3 layout ----
    fig = plt.figure(figsize=(18, 11))

    # --- Top left: 3D trajectory ---
    ax = fig.add_subplot(231, projection='3d')
    # GT full range
    ax.plot(gt_full_centers[:window_size+stride, 0],
            gt_full_centers[:window_size+stride, 1],
            gt_full_centers[:window_size+stride, 2],
            'b-', linewidth=2, label='GT', alpha=0.6)
    # W0 (Sim3-aligned)
    ax.plot(aligned_centers_w0[:, 0], aligned_centers_w0[:, 1], aligned_centers_w0[:, 2],
            'r-', linewidth=2, label=f'W0 [0-63] (ATE {ate_w0["trans_rmse"]*100:.1f}cm)')
    # W1 (Sim3-aligned)
    ax.plot(aligned_centers_w1_full[:, 0], aligned_centers_w1_full[:, 1], aligned_centers_w1_full[:, 2],
            color='#FF8C00', linewidth=2, label=f'W1 [32-95] (ATE {ate_w1["trans_rmse"]*100:.1f}cm)')
    # Mark overlap region on GT
    gt_ov = gt_full_centers[stride:window_size]
    ax.plot(gt_ov[:, 0], gt_ov[:, 1], gt_ov[:, 2],
            'g-', linewidth=4, alpha=0.4, label='Overlap [32-63]')
    # Start/end markers
    ax.scatter(*gt_full_centers[0], c='blue', marker='^', s=100, zorder=5)
    ax.scatter(*aligned_centers_w0[0], c='red', marker='^', s=100, zorder=5)
    ax.scatter(*aligned_centers_w1_full[0], c='#FF8C00', marker='^', s=100, zorder=5)
    ax.set_title('3D: Both Windows vs GT (each Sim3-aligned)', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # --- Top middle: XY projection ---
    ax = fig.add_subplot(232)
    ax.plot(gt_full_centers[:window_size+stride, 0], gt_full_centers[:window_size+stride, 1],
            'b-', linewidth=2, alpha=0.5, label='GT')
    ax.plot(aligned_centers_w0[:, 0], aligned_centers_w0[:, 1],
            'r-', linewidth=2, label='W0 [0-63]')
    ax.plot(aligned_centers_w1_full[:, 0], aligned_centers_w1_full[:, 1],
            color='#FF8C00', linewidth=2, label='W1 [32-95]')
    # Highlight overlap on GT
    ax.plot(gt_ov[:, 0], gt_ov[:, 1], 'g-', linewidth=5, alpha=0.3)
    # Connect corresponding overlap points with thin lines to show disagreement
    for k in range(0, overlap_frames, 4):
        ax.plot([w0_overlap_aligned[k, 0], w1_overlap_aligned[k, 0]],
                [w0_overlap_aligned[k, 1], w1_overlap_aligned[k, 1]],
                'k-', linewidth=0.5, alpha=0.4)
    ax.set_title('XY: Overlap frames connected', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- Top right: XZ projection ---
    ax = fig.add_subplot(233)
    ax.plot(gt_full_centers[:window_size+stride, 0], gt_full_centers[:window_size+stride, 2],
            'b-', linewidth=2, alpha=0.5, label='GT')
    ax.plot(aligned_centers_w0[:, 0], aligned_centers_w0[:, 2],
            'r-', linewidth=2, label='W0 [0-63]')
    ax.plot(aligned_centers_w1_full[:, 0], aligned_centers_w1_full[:, 2],
            color='#FF8C00', linewidth=2, label='W1 [32-95]')
    ax.plot(gt_ov[:, 0], gt_ov[:, 2], 'g-', linewidth=5, alpha=0.3)
    for k in range(0, overlap_frames, 4):
        ax.plot([w0_overlap_aligned[k, 0], w1_overlap_aligned[k, 0]],
                [w0_overlap_aligned[k, 2], w1_overlap_aligned[k, 2]],
                'k-', linewidth=0.5, alpha=0.4)
    ax.set_title('XZ: Overlap frames connected', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- Bottom left: Per-frame error vs GT ---
    ax = fig.add_subplot(234)
    frames_w0 = np.arange(0, window_size)
    frames_w1 = np.arange(stride, stride + window_size)
    ax.plot(frames_w0, err_w0, 'r-', linewidth=1.5, label='W0 error vs GT')
    ax.plot(frames_w1, err_w1, color='#FF8C00', linewidth=1.5, label='W1 error vs GT')
    ax.axvspan(stride, window_size, alpha=0.15, color='green', label='Overlap region')
    ax.set_title('Per-frame error vs GT (each Sim3-aligned)', fontsize=11)
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Error (cm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Bottom middle: Overlap mutual disagreement ---
    ax = fig.add_subplot(235)
    overlap_frames_idx = np.arange(stride, window_size)
    ax.bar(overlap_frames_idx, overlap_mutual_err, color='purple', alpha=0.7, width=0.8)
    ax.axhline(y=np.mean(overlap_mutual_err), color='purple', linestyle='--',
               label=f'Mean: {np.mean(overlap_mutual_err):.2f} cm')
    ax.set_title('W0 vs W1 disagreement in overlap\n(both Sim3-aligned to GT)', fontsize=11)
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Disagreement (cm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Bottom right: Rotation disagreement ---
    ax = fig.add_subplot(236)
    # Compute per-frame rotation disagreement in overlap (both Sim3-aligned)
    rot_disagree_aligned = []
    for k in range(overlap_frames):
        R_w0 = ate_w0['aligned_poses'][stride + k][:3, :3]
        R_w1 = ate_w1['aligned_poses'][k][:3, :3]
        R_diff = R_w0.T @ R_w1
        angle = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
        rot_disagree_aligned.append(angle)
    rot_disagree_aligned = np.array(rot_disagree_aligned)

    ax.bar(overlap_frames_idx, rot_disagree_aligned, color='darkred', alpha=0.7, width=0.8)
    ax.axhline(y=np.mean(rot_disagree_aligned), color='darkred', linestyle='--',
               label=f'Mean: {np.mean(rot_disagree_aligned):.2f}°')
    ax.set_title('Rotation disagreement in overlap\n(both Sim3-aligned to GT)', fontsize=11)
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Disagreement (°)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        'Two Windows Independently Sim3-aligned to GT\n'
        f'Each window is accurate (~1 cm), but they disagree in the overlap by '
        f'{np.mean(overlap_mutual_err):.1f} cm / {np.mean(rot_disagree_aligned):.1f}°',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    out_path_3 = os.path.join(args.output_dir, 'plot3_window_inconsistency.png')
    fig.savefig(out_path_3, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {out_path_3}")
    plt.close(fig)

    # =========================================================================
    # Plot 4: Raw overlap comparison — no GT, no Umeyama
    # Align both windows at frame 32 and compare directly
    # =========================================================================

    # W0: relative poses for overlap frames, anchored at frame 32
    # T_rel_w0(k) = T_w0[32]^{-1} @ T_w0[32+k]
    T_w0_32_inv_plot = np.linalg.inv(pred_w0_scaled[stride])
    w0_overlap_rel = np.array([T_w0_32_inv_plot @ pred_w0_scaled[stride + k]
                               for k in range(overlap_frames)])  # [32, 4, 4]

    # W1: relative poses for overlap frames, anchored at frame 32 (= W1's frame 0)
    # T_rel_w1(k) = T_w1[0]^{-1} @ T_w1[k]
    T_w1_0_inv_plot = np.linalg.inv(pred_w1_scaled[0])
    w1_overlap_rel = np.array([T_w1_0_inv_plot @ pred_w1_scaled[k]
                               for k in range(overlap_frames)])  # [32, 4, 4]

    # GT: relative poses for overlap frames, anchored at frame 32
    gt_32_inv = np.linalg.inv(gt_poses[stride])
    gt_overlap_rel = np.array([gt_32_inv @ gt_poses[stride + k]
                               for k in range(overlap_frames)])  # [32, 4, 4]

    # Extract camera centers (all relative to frame 32 = origin)
    w0_ov_c = extract_camera_centers(w0_overlap_rel)
    w1_ov_c = extract_camera_centers(w1_overlap_rel)
    gt_ov_c = extract_camera_centers(gt_overlap_rel)

    # Per-frame disagreement
    w0w1_diff = np.linalg.norm(w0_ov_c - w1_ov_c, axis=-1) * 100  # cm
    w0gt_diff = np.linalg.norm(w0_ov_c - gt_ov_c, axis=-1) * 100
    w1gt_diff = np.linalg.norm(w1_ov_c - gt_ov_c, axis=-1) * 100

    # Rotation disagreement
    w0w1_rot = []
    w0gt_rot = []
    w1gt_rot = []
    for k in range(overlap_frames):
        def rot_angle(R1, R2):
            R = R1[:3, :3].T @ R2[:3, :3]
            return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        w0w1_rot.append(rot_angle(w0_overlap_rel[k], w1_overlap_rel[k]))
        w0gt_rot.append(rot_angle(w0_overlap_rel[k], gt_overlap_rel[k]))
        w1gt_rot.append(rot_angle(w1_overlap_rel[k], gt_overlap_rel[k]))
    w0w1_rot = np.array(w0w1_rot)
    w0gt_rot = np.array(w0gt_rot)
    w1gt_rot = np.array(w1gt_rot)

    fig = plt.figure(figsize=(18, 12))

    # --- Top left: 3D, all three trajectories relative to frame 32 ---
    ax = fig.add_subplot(231, projection='3d')
    ax.plot(gt_ov_c[:, 0], gt_ov_c[:, 1], gt_ov_c[:, 2],
            'b-', linewidth=2.5, label='GT', zorder=3)
    ax.plot(w0_ov_c[:, 0], w0_ov_c[:, 1], w0_ov_c[:, 2],
            'r-', linewidth=2, label='W0 pred', zorder=2)
    ax.plot(w1_ov_c[:, 0], w1_ov_c[:, 1], w1_ov_c[:, 2],
            color='#FF8C00', linewidth=2, label='W1 pred', zorder=2)
    # Origin marker (frame 32)
    ax.scatter(0, 0, 0, c='green', marker='*', s=200, zorder=5, label='Frame 32 (anchor)')
    # End markers
    for centers, color in [(gt_ov_c, 'blue'), (w0_ov_c, 'red'), (w1_ov_c, '#FF8C00')]:
        ax.scatter(*centers[-1], c=color, marker='s', s=60, zorder=5, edgecolors='black')
    # Connect endpoints to show divergence
    ax.plot([w0_ov_c[-1, 0], w1_ov_c[-1, 0]],
            [w0_ov_c[-1, 1], w1_ov_c[-1, 1]],
            [w0_ov_c[-1, 2], w1_ov_c[-1, 2]],
            'k--', linewidth=2, alpha=0.5, label=f'Endpoint gap: {w0w1_diff[-1]:.1f}cm')
    ax.set_title('3D: Overlap frames relative to frame 32\n(no GT alignment, raw predictions)', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

    # --- Top middle: XY projection ---
    ax = fig.add_subplot(232)
    ax.plot(gt_ov_c[:, 0], gt_ov_c[:, 1], 'b-', linewidth=2.5, label='GT', alpha=0.7)
    ax.plot(w0_ov_c[:, 0], w0_ov_c[:, 1], 'r-', linewidth=2, label='W0')
    ax.plot(w1_ov_c[:, 0], w1_ov_c[:, 1], color='#FF8C00', linewidth=2, label='W1')
    ax.scatter(0, 0, c='green', marker='*', s=200, zorder=5)
    # Connect same frames with thin lines (every 4th)
    for k in range(4, overlap_frames, 4):
        ax.annotate('', xy=(w1_ov_c[k, 0], w1_ov_c[k, 1]),
                    xytext=(w0_ov_c[k, 0], w0_ov_c[k, 1]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))
        ax.text((w0_ov_c[k, 0]+w1_ov_c[k, 0])/2, (w0_ov_c[k, 1]+w1_ov_c[k, 1])/2,
                f'{stride+k}', fontsize=6, ha='center', color='gray')
    ax.set_title('XY: Gray arrows = same frame, different window', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # --- Top right: XZ projection ---
    ax = fig.add_subplot(233)
    ax.plot(gt_ov_c[:, 0], gt_ov_c[:, 2], 'b-', linewidth=2.5, label='GT', alpha=0.7)
    ax.plot(w0_ov_c[:, 0], w0_ov_c[:, 2], 'r-', linewidth=2, label='W0')
    ax.plot(w1_ov_c[:, 0], w1_ov_c[:, 2], color='#FF8C00', linewidth=2, label='W1')
    ax.scatter(0, 0, c='green', marker='*', s=200, zorder=5)
    for k in range(4, overlap_frames, 4):
        ax.annotate('', xy=(w1_ov_c[k, 0], w1_ov_c[k, 2]),
                    xytext=(w0_ov_c[k, 0], w0_ov_c[k, 2]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))
    ax.set_title('XZ: Same frame connections', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # --- Bottom left: Position disagreement ---
    ax = fig.add_subplot(234)
    frames_ov = np.arange(stride, window_size)
    ax.plot(frames_ov, w0w1_diff, 'purple', linewidth=2, label=f'W0↔W1: mean {w0w1_diff[1:].mean():.1f}cm')
    ax.plot(frames_ov, w0gt_diff, 'r--', linewidth=1.5, alpha=0.7, label=f'W0↔GT: mean {w0gt_diff[1:].mean():.1f}cm')
    ax.plot(frames_ov, w1gt_diff, '--', color='#FF8C00', linewidth=1.5, alpha=0.7,
            label=f'W1↔GT: mean {w1gt_diff[1:].mean():.1f}cm')
    ax.set_title('Position disagreement (relative to frame 32)', fontsize=11)
    ax.set_xlabel('Frame index'); ax.set_ylabel('Distance (cm)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    # --- Bottom middle: Rotation disagreement ---
    ax = fig.add_subplot(235)
    ax.plot(frames_ov, w0w1_rot, 'purple', linewidth=2, label=f'W0↔W1: mean {w0w1_rot[1:].mean():.1f}°')
    ax.plot(frames_ov, w0gt_rot, 'r--', linewidth=1.5, alpha=0.7, label=f'W0↔GT: mean {w0gt_rot[1:].mean():.1f}°')
    ax.plot(frames_ov, w1gt_rot, '--', color='#FF8C00', linewidth=1.5, alpha=0.7,
            label=f'W1↔GT: mean {w1gt_rot[1:].mean():.1f}°')
    ax.set_title('Rotation disagreement (relative to frame 32)', fontsize=11)
    ax.set_xlabel('Frame index'); ax.set_ylabel('Angle (°)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    # --- Bottom right: Scale comparison ---
    ax = fig.add_subplot(236)
    # Per-frame distance from anchor (proxy for scale)
    w0_dist = np.linalg.norm(w0_ov_c, axis=-1) * 100  # cm
    w1_dist = np.linalg.norm(w1_ov_c, axis=-1) * 100
    gt_dist = np.linalg.norm(gt_ov_c, axis=-1) * 100
    ax.plot(frames_ov, gt_dist, 'b-', linewidth=2.5, label='GT', alpha=0.7)
    ax.plot(frames_ov, w0_dist, 'r-', linewidth=2, label=f'W0 (scale={scale_w0:.3f})')
    ax.plot(frames_ov, w1_dist, color='#FF8C00', linewidth=2, label=f'W1 (scale={scale_w1:.3f})')
    ax.set_title('Distance from anchor (scale consistency)', fontsize=11)
    ax.set_xlabel('Frame index'); ax.set_ylabel('Distance from frame 32 (cm)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    fig.suptitle(
        'Overlap Region: Raw Predictions Aligned at Frame 32 (No GT Alignment)\n'
        f'W0↔W1 position gap: {w0w1_diff[-1]:.1f} cm at frame 63  |  '
        f'Rotation gap: {w0w1_rot[-1]:.1f}°  |  '
        f'Scale: W0={scale_w0:.3f} vs W1={scale_w1:.3f}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    out_path_4 = os.path.join(args.output_dir, 'plot4_overlap_raw_comparison.png')
    fig.savefig(out_path_4, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {out_path_4}")
    plt.close(fig)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Window 0 ATE:  {ate_w0['trans_rmse']*100:.2f} cm  (per-window, Sim3-aligned)")
    logger.info(f"Window 1 ATE:  {ate_w1['trans_rmse']*100:.2f} cm  (per-window, Sim3-aligned)")
    logger.info(f"Chained ATE:   {ate_chained['trans_rmse']*100:.2f} cm  (2 windows chained)")
    logger.info(f"Degradation:   {ate_chained['trans_rmse'] / ate_w0['trans_rmse']:.1f}x")
    logger.info(f"Scale W0:      {scale_w0:.3f}")
    logger.info(f"Scale W1:      {scale_w1:.3f}")
    logger.info(f"Scale diff:    {abs(scale_w0 - scale_w1):.3f} ({abs(scale_w0 - scale_w1)/scale_w0*100:.1f}%)")
    logger.info(f"\nPlots saved to: {args.output_dir}")

    # =========================================================================
    # Rerun visualization
    # =========================================================================
    try:
        import rerun as rr

        rr_path = os.path.join(args.output_dir, 'chaining_drift.rrd')
        rr.init("chaining_drift", spawn=False)
        rr.save(rr_path)

        # Colors
        BLUE = [30, 100, 255]      # GT
        RED = [220, 40, 40]        # W0
        ORANGE = [255, 140, 0]     # W1
        GREEN = [40, 180, 40]      # GT overlap
        PURPLE = [160, 40, 200]    # Chained
        CYAN = [40, 200, 200]      # GT Sim3 concat

        # --- Full GT trajectory ---
        gt_all_centers = extract_camera_centers(gt_poses[:stride + window_size])
        rr.log("gt/full", rr.LineStrips3D([gt_all_centers], colors=[BLUE], radii=0.002))
        rr.log("gt/full/points", rr.Points3D(gt_all_centers, colors=[BLUE], radii=0.003))
        # Highlight overlap on GT
        gt_overlap_centers = gt_all_centers[stride:window_size]
        rr.log("gt/overlap", rr.LineStrips3D([gt_overlap_centers], colors=[GREEN], radii=0.004))

        # --- W0 Sim3-aligned (frames 0-63) ---
        rr.log("w0_sim3/traj", rr.LineStrips3D([aligned_centers_w0], colors=[RED], radii=0.002))
        rr.log("w0_sim3/points", rr.Points3D(aligned_centers_w0, colors=[RED], radii=0.003))
        # Mark start
        rr.log("w0_sim3/start", rr.Points3D([aligned_centers_w0[0]], colors=[RED], radii=0.008))

        # --- W1 Sim3-aligned (frames 32-95) ---
        rr.log("w1_sim3/traj", rr.LineStrips3D([aligned_centers_w1_full], colors=[ORANGE], radii=0.002))
        rr.log("w1_sim3/points", rr.Points3D(aligned_centers_w1_full, colors=[ORANGE], radii=0.003))
        rr.log("w1_sim3/start", rr.Points3D([aligned_centers_w1_full[0]], colors=[ORANGE], radii=0.008))

        # --- Overlap disagreement lines (W0 vs W1, both Sim3-aligned to GT) ---
        disagree_lines = []
        for k in range(0, overlap_frames, 2):
            disagree_lines.append([w0_overlap_aligned[k], w1_overlap_aligned[k]])
        if disagree_lines:
            rr.log("overlap_disagreement", rr.LineStrips3D(
                disagree_lines, colors=[[200, 200, 200]], radii=0.001))

        # --- GT-Sim3 concatenated trajectory (the 1.26cm result) ---
        concat_centers = extract_camera_centers(concat_pred_arr)
        rr.log("gt_sim3_concat/traj", rr.LineStrips3D([concat_centers], colors=[CYAN], radii=0.002))

        # --- Chained trajectory (single-frame alignment, 41cm result) ---
        rr.log("chained_single_frame/traj", rr.LineStrips3D(
            [aligned_chained_centers], colors=[PURPLE], radii=0.002))

        # --- Raw overlap: both windows relative to frame 32 ---
        rr.log("raw_overlap/gt", rr.LineStrips3D([gt_ov_c], colors=[BLUE], radii=0.002))
        rr.log("raw_overlap/w0", rr.LineStrips3D([w0_ov_c], colors=[RED], radii=0.002))
        rr.log("raw_overlap/w1", rr.LineStrips3D([w1_ov_c], colors=[ORANGE], radii=0.002))
        rr.log("raw_overlap/origin", rr.Points3D([[0, 0, 0]], colors=[GREEN], radii=0.008))
        # Connect same frames
        raw_disagree = []
        for k in range(0, overlap_frames, 2):
            raw_disagree.append([w0_ov_c[k], w1_ov_c[k]])
        if raw_disagree:
            rr.log("raw_overlap/disagreement", rr.LineStrips3D(
                raw_disagree, colors=[[200, 200, 200]], radii=0.001))

        # --- Frame-by-frame annotations using timeline ---
        for k in range(window_size + stride):
            rr.set_time("frame", sequence=k)

            # GT point at this frame
            if k < len(gt_all_centers):
                rr.log("timeline/gt", rr.Points3D([gt_all_centers[k]], colors=[BLUE], radii=0.005))

            # W0 point (frames 0-63)
            if k < window_size:
                rr.log("timeline/w0", rr.Points3D(
                    [aligned_centers_w0[k]], colors=[RED], radii=0.005))

            # W1 point (frames 32-95)
            w1_idx = k - stride
            if 0 <= w1_idx < window_size:
                rr.log("timeline/w1", rr.Points3D(
                    [aligned_centers_w1_full[w1_idx]], colors=[ORANGE], radii=0.005))

            # In overlap, show disagreement line
            if stride <= k < window_size:
                ov_k = k - stride
                rr.log("timeline/disagreement", rr.LineStrips3D(
                    [[aligned_centers_w0[k], aligned_centers_w1_full[ov_k]]],
                    colors=[[200, 200, 200]], radii=0.001))
                err_cm = np.linalg.norm(aligned_centers_w0[k] - aligned_centers_w1_full[ov_k]) * 100
                rr.log("timeline/overlap_error_cm", rr.Scalars([err_cm]))

        logger.info(f"Rerun saved: {rr_path}")
        logger.info(f"  View with: rerun {rr_path}")

    except ImportError:
        logger.warning("rerun-sdk not installed, skipping Rerun visualization")
    except Exception as e:
        logger.warning(f"Rerun visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
