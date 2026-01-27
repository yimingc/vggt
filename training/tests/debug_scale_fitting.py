#!/usr/bin/env python3
"""
Debug script to investigate scale fitting issues.

Compares our window-relative computation with eval script's approach.
"""

import sys
import os
import torch
import numpy as np
import cv2

sys.path.insert(0, '/home/yiming/Dev/vggt')
sys.path.insert(0, '/home/yiming/Dev/vggt/training')

from vggt.models.vggt import VGGT
from vggt.utils.lie_algebra import (
    pose_encoding_to_se3,
    extract_window_relative_poses,
    extract_relative_camera_positions,
)


def extract_camera_positions_buggy(T):
    """BUGGY: Extract camera positions from SE3 - DO NOT USE on relative poses!"""
    R = T.rotation().matrix()
    t = T.translation()
    return -torch.einsum('...ij,...j->...i', R.transpose(-1, -2), t)
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


def extract_camera_centers_numpy(poses, convention='w2c'):
    """Extract camera centers from extrinsic matrices (numpy version from eval script)."""
    positions = []
    for pose in poses:
        R = pose[:3, :3]
        t = pose[:3, 3]
        if convention == 'w2c':
            positions.append(-R.T @ t)
        else:
            positions.append(t)
    return np.array(positions)


def quat_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    # Normalize
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def tum_pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw):
    """Convert TUM pose to OpenCV extrinsic matrix (world-to-camera)."""
    R_cw = quat_to_rotation_matrix(qx, qy, qz, qw)
    t_cw = np.array([tx, ty, tz])
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    extrinsic = np.zeros((3, 4), dtype=np.float32)
    extrinsic[:3, :3] = R_wc
    extrinsic[:3, 3] = t_wc
    return extrinsic


def load_tum_data(tum_dir, num_frames=8):
    """Load TUM RGB-D data directly from rgb.txt and groundtruth.txt."""
    seq_dir = os.path.join(tum_dir, "rgbd_dataset_freiburg1_desk")

    # Load RGB timestamps and paths
    rgb_file = os.path.join(seq_dir, "rgb.txt")
    if not os.path.exists(rgb_file):
        raise FileNotFoundError(f"RGB file not found: {rgb_file}")

    rgb_data = []
    with open(rgb_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                rgb_data.append({
                    'ts': float(parts[0]),
                    'path': os.path.join(seq_dir, parts[1]),
                })

    # Load ground truth
    gt_file = os.path.join(seq_dir, "groundtruth.txt")
    gt_data = {}
    with open(gt_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            ts = float(parts[0])
            gt_data[ts] = [float(x) for x in parts[1:]]

    gt_timestamps = list(gt_data.keys())

    # Match RGB to GT (find closest GT for each RGB)
    frames = []
    for rgb in rgb_data:
        rgb_ts = rgb['ts']
        # Find closest GT
        gt_ts = min(gt_timestamps, key=lambda x: abs(x - rgb_ts))
        if abs(gt_ts - rgb_ts) > 0.02:  # 20ms threshold
            continue
        pose = gt_data[gt_ts]
        extrinsic = tum_pose_to_extrinsic(*pose)
        frames.append({
            'rgb_path': rgb['path'],
            'extrinsic': extrinsic,
        })
        if len(frames) >= num_frames:
            break

    return frames


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # Load model
    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).to(dtype)
    model.eval()

    # Load TUM data directly
    print("\nLoading TUM data...")
    frames = load_tum_data("/home/yiming/Dev/tum_rgbd", num_frames=8)
    print(f"Loaded {len(frames)} frames")

    # Load and preprocess images
    images = []
    extrinsics = []
    for frame in frames:
        img = cv2.imread(frame['rgb_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (518, 518))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        images.append(img)
        extrinsics.append(frame['extrinsic'])

    images = np.stack(images)  # [8, 3, 518, 518]
    extrinsics = np.stack(extrinsics)  # [8, 3, 4]

    # Create intrinsics (approximate for TUM freiburg1)
    fx, fy = 517.3, 516.5
    cx, cy = 318.6, 255.3
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    # Scale for resized image
    intrinsic[0] *= 518 / 640
    intrinsic[1] *= 518 / 480
    intrinsics = np.stack([intrinsic] * 8)  # [8, 3, 3]

    # To tensors
    images_tensor = torch.from_numpy(images).to(device).to(dtype).unsqueeze(0)  # [1, 8, 3, 518, 518]
    gt_extrinsics = torch.from_numpy(extrinsics).unsqueeze(0).to(device)  # [1, 8, 3, 4]
    gt_intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(device)  # [1, 8, 3, 3]

    # Run VGGT
    print("Running VGGT inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(images_tensor)

    pred_pose_enc = predictions['pose_enc']  # [1, 8, 9]

    # Convert GT to pose encoding
    image_hw = images_tensor.shape[-2:]
    gt_pose_enc = extri_intri_to_pose_encoding(
        gt_extrinsics.float(), gt_intrinsics.float(), image_hw, "absT_quaR_FoV"
    )

    print(f"\nGT pose_enc shape: {gt_pose_enc.shape}")
    print(f"Pred pose_enc shape: {pred_pose_enc.shape}")

    # Method 1: OLD BUGGY approach - extract_camera_positions_buggy(T_rel) gives WRONG results
    print("\n" + "="*60)
    print("Method 1: OLD BUGGY (extract_camera_positions on T_rel)")
    print("="*60)

    T_abs_gt = pose_encoding_to_se3(gt_pose_enc.float())
    T_abs_pred = pose_encoding_to_se3(pred_pose_enc.float())
    T_rel_gt = extract_window_relative_poses(T_abs_gt)
    T_rel_pred = extract_window_relative_poses(T_abs_pred)

    # BUGGY: extract_camera_positions on T_rel gives wrong results!
    gt_cam_pos = extract_camera_positions_buggy(T_rel_gt)[0].cpu().numpy()  # [8, 3]
    pred_cam_pos = extract_camera_positions_buggy(T_rel_pred)[0].cpu().numpy()  # [8, 3]

    print("\nGT camera positions (relative to frame 0):")
    for i in range(8):
        print(f"  Frame {i}: [{gt_cam_pos[i, 0]:7.4f}, {gt_cam_pos[i, 1]:7.4f}, {gt_cam_pos[i, 2]:7.4f}]")

    print("\nPred camera positions (relative to frame 0):")
    for i in range(8):
        print(f"  Frame {i}: [{pred_cam_pos[i, 0]:7.4f}, {pred_cam_pos[i, 1]:7.4f}, {pred_cam_pos[i, 2]:7.4f}]")

    # Compute dot product for scale fitting
    gt_pos = gt_cam_pos[1:]  # Skip frame 0
    pred_pos = pred_cam_pos[1:]

    dot_product = np.sum(gt_pos * pred_pos)
    pred_sq = np.sum(pred_pos ** 2)
    scale_raw = dot_product / (pred_sq + 1e-8)

    print(f"\nScale fitting (frames 1-7):")
    print(f"  Dot product: {dot_product:.6f}")
    print(f"  Pred squared: {pred_sq:.6f}")
    print(f"  Scale (raw): {scale_raw:.4f}")
    print(f"  Direction: {'SAME' if scale_raw > 0 else 'OPPOSITE!'}")

    # Method 2: Eval script's approach (absolute poses, then extract centers)
    print("\n" + "="*60)
    print("Method 2: Eval script's approach (absolute camera centers)")
    print("="*60)

    # Get extrinsics from pose encoding
    pred_extri, _ = pose_encoding_to_extri_intri(pred_pose_enc.float(), image_hw)
    pred_extri_np = pred_extri[0].cpu().numpy()  # [8, 3, 4]
    gt_extri_np = gt_extrinsics[0].cpu().numpy()  # [8, 3, 4]

    gt_centers = extract_camera_centers_numpy(gt_extri_np, 'w2c')
    pred_centers = extract_camera_centers_numpy(pred_extri_np, 'w2c')

    print("\nGT camera centers (absolute, world coords):")
    for i in range(8):
        print(f"  Frame {i}: [{gt_centers[i, 0]:7.4f}, {gt_centers[i, 1]:7.4f}, {gt_centers[i, 2]:7.4f}]")

    print("\nPred camera centers (absolute, world coords):")
    for i in range(8):
        print(f"  Frame {i}: [{pred_centers[i, 0]:7.4f}, {pred_centers[i, 1]:7.4f}, {pred_centers[i, 2]:7.4f}]")

    # Make relative to frame 0
    gt_centers_rel = gt_centers - gt_centers[0]
    pred_centers_rel = pred_centers - pred_centers[0]

    print("\nGT camera centers (relative to frame 0):")
    for i in range(8):
        print(f"  Frame {i}: [{gt_centers_rel[i, 0]:7.4f}, {gt_centers_rel[i, 1]:7.4f}, {gt_centers_rel[i, 2]:7.4f}]")

    print("\nPred camera centers (relative to frame 0):")
    for i in range(8):
        print(f"  Frame {i}: [{pred_centers_rel[i, 0]:7.4f}, {pred_centers_rel[i, 1]:7.4f}, {pred_centers_rel[i, 2]:7.4f}]")

    # Scale fitting with eval script's approach
    gt_pos_eval = gt_centers_rel[1:]
    pred_pos_eval = pred_centers_rel[1:]

    dot_product_eval = np.sum(gt_pos_eval * pred_pos_eval)
    pred_sq_eval = np.sum(pred_pos_eval ** 2)
    scale_raw_eval = dot_product_eval / (pred_sq_eval + 1e-8)

    print(f"\nScale fitting (eval script approach):")
    print(f"  Dot product: {dot_product_eval:.6f}")
    print(f"  Pred squared: {pred_sq_eval:.6f}")
    print(f"  Scale (raw): {scale_raw_eval:.4f}")
    print(f"  Direction: {'SAME' if scale_raw_eval > 0 else 'OPPOSITE!'}")

    # Method 3: Our CORRECTED approach using extract_relative_camera_positions
    print("\n" + "="*60)
    print("Method 3: CORRECTED (extract_relative_camera_positions)")
    print("="*60)

    # Reuse T_abs computed earlier
    gt_cam_pos_corrected = extract_relative_camera_positions(T_abs_gt)[0].cpu().numpy()
    pred_cam_pos_corrected = extract_relative_camera_positions(T_abs_pred)[0].cpu().numpy()

    print("\nGT camera positions (CORRECTED, relative to frame 0):")
    for i in range(8):
        print(f"  Frame {i}: [{gt_cam_pos_corrected[i, 0]:7.4f}, {gt_cam_pos_corrected[i, 1]:7.4f}, {gt_cam_pos_corrected[i, 2]:7.4f}]")

    print("\nPred camera positions (CORRECTED, relative to frame 0):")
    for i in range(8):
        print(f"  Frame {i}: [{pred_cam_pos_corrected[i, 0]:7.4f}, {pred_cam_pos_corrected[i, 1]:7.4f}, {pred_cam_pos_corrected[i, 2]:7.4f}]")

    gt_pos_corrected = gt_cam_pos_corrected[1:]
    pred_pos_corrected = pred_cam_pos_corrected[1:]

    dot_product_corrected = np.sum(gt_pos_corrected * pred_pos_corrected)
    pred_sq_corrected = np.sum(pred_pos_corrected ** 2)
    scale_raw_corrected = dot_product_corrected / (pred_sq_corrected + 1e-8)

    print(f"\nScale fitting (CORRECTED approach):")
    print(f"  Dot product: {dot_product_corrected:.6f}")
    print(f"  Pred squared: {pred_sq_corrected:.6f}")
    print(f"  Scale (raw): {scale_raw_corrected:.4f}")
    print(f"  Direction: {'SAME' if scale_raw_corrected > 0 else 'OPPOSITE!'}")

    # Compare the three methods
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    print(f"Method 1 (OLD, buggy): scale: {scale_raw:.4f}")
    print(f"Method 2 (eval script): scale: {scale_raw_eval:.4f}")
    print(f"Method 3 (CORRECTED):   scale: {scale_raw_corrected:.4f}")
    print(f"\nMethod 2 vs Method 3 difference: {abs(scale_raw_eval - scale_raw_corrected):.6f}")
    if abs(scale_raw_eval - scale_raw_corrected) < 0.001:
        print("✓ Method 3 matches Method 2 (eval script) - BUG FIXED!")
    else:
        print("✗ Method 3 differs from Method 2 - still has bug")


if __name__ == "__main__":
    main()
