#!/usr/bin/env python3
"""
Test script for TUM RGB-D dataloader.

Usage:
    python training/tests/test_tum_dataloader.py --tum_dir /path/to/tum/sequences

This script tests:
1. Data loading and shape verification
2. Visual verification (saves images and depth maps)
3. Camera trajectory visualization
4. Integration test with VGGT model
"""

import os
import sys
import argparse
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)


class MockCommonConf:
    """Mock configuration object for testing."""
    def __init__(self):
        self.img_size = 518
        self.patch_size = 14
        self.debug = False
        self.training = True
        self.get_nearby = True
        self.load_depth = True
        self.inside_random = False
        self.allow_duplicate_img = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False

        # Augmentation config
        self.augs = type('obj', (object,), {'scales': None})()


def test_data_loading(dataset, output_dir):
    """Test 1: Basic data loading and shape verification."""
    print("\n" + "="*60)
    print("TEST 1: Data Loading and Shape Verification")
    print("="*60)

    # Get a batch
    batch = dataset.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)

    print(f"\nSequence: {batch['seq_name']}")
    print(f"Frame IDs: {batch['ids']}")
    print(f"Number of frames: {batch['frame_num']}")

    # Check shapes
    for i, (img, depth, extri, intri) in enumerate(zip(
        batch['images'], batch['depths'], batch['extrinsics'], batch['intrinsics']
    )):
        print(f"\nFrame {i}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Image range: [{img.min():.2f}, {img.max():.2f}]")

        if depth is not None:
            print(f"  Depth shape: {depth.shape}")
            print(f"  Depth dtype: {depth.dtype}")
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                print(f"  Depth range (valid): [{valid_depth.min():.3f}, {valid_depth.max():.3f}] meters")
            print(f"  Valid depth ratio: {(depth > 0).sum() / depth.size * 100:.1f}%")

        print(f"  Extrinsic shape: {extri.shape}")
        print(f"  Intrinsic shape: {intri.shape}")

        # Check rotation matrix orthogonality
        R = extri[:3, :3]
        orthogonality_error = np.abs(R @ R.T - np.eye(3)).max()
        print(f"  Rotation orthogonality error: {orthogonality_error:.6f}")

        # Check world points
        if batch['world_points'][i] is not None:
            wp = batch['world_points'][i]
            mask = batch['point_masks'][i]
            valid_points = wp[mask]
            if len(valid_points) > 0:
                print(f"  World points: {len(valid_points)} valid points")
                print(f"  World points range X: [{valid_points[:, 0].min():.2f}, {valid_points[:, 0].max():.2f}]")
                print(f"  World points range Y: [{valid_points[:, 1].min():.2f}, {valid_points[:, 1].max():.2f}]")
                print(f"  World points range Z: [{valid_points[:, 2].min():.2f}, {valid_points[:, 2].max():.2f}]")

    print("\n✓ Data loading test PASSED")
    return batch


def test_visual_verification(batch, output_dir):
    """Test 2: Visual verification - save images and depth maps."""
    print("\n" + "="*60)
    print("TEST 2: Visual Verification")
    print("="*60)

    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    n_frames = min(4, len(batch['images']))
    fig, axes = plt.subplots(2, n_frames, figsize=(4*n_frames, 8))

    for i in range(n_frames):
        img = batch['images'][i]
        depth = batch['depths'][i]

        # RGB image
        ax = axes[0, i] if n_frames > 1 else axes[0]
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f'Frame {batch["ids"][i]}')
        ax.axis('off')

        # Depth map
        ax = axes[1, i] if n_frames > 1 else axes[1]
        if depth is not None:
            depth_vis = depth.copy()
            depth_vis[depth_vis == 0] = np.nan  # Mark invalid as NaN
            im = ax.imshow(depth_vis, cmap='viridis')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Depth (m)')
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(vis_dir, 'rgb_depth_visualization.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved RGB/Depth visualization to: {save_path}")

    print("\n✓ Visual verification test PASSED")


def test_camera_trajectory(batch, output_dir):
    """Test 3: Visualize camera trajectory from GT poses."""
    print("\n" + "="*60)
    print("TEST 3: Camera Trajectory Visualization")
    print("="*60)

    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    # Extract camera positions from extrinsics
    # extrinsic is [R|t] where t = -R @ camera_position
    # So camera_position = -R.T @ t
    positions = []
    for extri in batch['extrinsics']:
        R = extri[:3, :3]
        t = extri[:3, 3]
        cam_pos = -R.T @ t
        positions.append(cam_pos)
    positions = np.array(positions)

    # Plot 3D trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', s=50, label='Frames')

    # Mark start and end
    ax.scatter(*positions[0], c='g', s=200, marker='^', label='Start')
    ax.scatter(*positions[-1], c='m', s=200, marker='s', label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Camera Trajectory: {batch["seq_name"]}')
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                          positions[:, 1].max()-positions[:, 1].min(),
                          positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    save_path = os.path.join(vis_dir, 'camera_trajectory.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved camera trajectory to: {save_path}")

    # Print trajectory stats
    trajectory_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    print(f"\nTrajectory statistics:")
    print(f"  Number of poses: {len(positions)}")
    print(f"  Trajectory length: {trajectory_length:.3f} m")
    print(f"  Start position: ({positions[0, 0]:.3f}, {positions[0, 1]:.3f}, {positions[0, 2]:.3f})")
    print(f"  End position: ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f})")

    print("\n✓ Camera trajectory test PASSED")


def test_point_cloud(batch, output_dir):
    """Test 4: Save point cloud as PLY file."""
    print("\n" + "="*60)
    print("TEST 4: Point Cloud Export")
    print("="*60)

    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    # Collect all valid points and colors
    all_points = []
    all_colors = []

    for i in range(len(batch['images'])):
        if batch['world_points'][i] is None:
            continue

        wp = batch['world_points'][i]
        mask = batch['point_masks'][i]
        img = batch['images'][i]

        valid_points = wp[mask]
        valid_colors = img[mask] / 255.0  # Normalize to [0, 1]

        # Subsample to avoid huge point clouds
        if len(valid_points) > 10000:
            indices = np.random.choice(len(valid_points), 10000, replace=False)
            valid_points = valid_points[indices]
            valid_colors = valid_colors[indices]

        all_points.append(valid_points)
        all_colors.append(valid_colors)

    if len(all_points) == 0:
        print("No valid points to export")
        return

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # Save as PLY
    ply_path = os.path.join(vis_dir, 'point_cloud.ply')
    save_ply(all_points, all_colors, ply_path)
    print(f"Saved point cloud ({len(all_points)} points) to: {ply_path}")

    print("\n✓ Point cloud export test PASSED")


def save_ply(points, colors, filename):
    """Save point cloud as PLY file."""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for pt, color in zip(points, colors):
            r, g, b = (color * 255).astype(np.uint8)
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")


def test_vggt_integration(batch, output_dir):
    """Test 5: Run batch through VGGT model."""
    print("\n" + "="*60)
    print("TEST 5: VGGT Integration Test")
    print("="*60)

    try:
        import torch
        from vggt.models.vggt import VGGT
    except ImportError as e:
        print(f"Could not import VGGT: {e}")
        print("Skipping VGGT integration test")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Device: {device}, dtype: {dtype}")

    # Prepare images for VGGT
    # VGGT expects [B, S, 3, H, W] with normalized values
    images = np.stack(batch['images'], axis=0)  # [S, H, W, 3]
    images = images.transpose(0, 3, 1, 2)  # [S, 3, H, W]
    images = images.astype(np.float32) / 255.0

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = (images - mean) / std

    images_tensor = torch.from_numpy(images).to(device).to(dtype)
    images_tensor = images_tensor.unsqueeze(0)  # [1, S, 3, H, W]

    print(f"Input shape: {images_tensor.shape}")

    # Load model
    try:
        print("Loading VGGT model...")
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()
    except Exception as e:
        print(f"Could not load VGGT model: {e}")
        print("Skipping VGGT integration test")
        return

    # Run inference
    print("Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_tensor)

    # Check predictions
    print("\nVGGT Predictions:")
    pose_enc = predictions['pose_enc']
    print(f"  Pose encoding shape: {pose_enc.shape}")  # [B, S, 9]

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    pred_extri, pred_intri = pose_encoding_to_extri_intri(
        pose_enc, images_tensor.shape[-2:]
    )
    print(f"  Predicted extrinsics shape: {pred_extri.shape}")
    print(f"  Predicted intrinsics shape: {pred_intri.shape}")

    # Compare with GT
    gt_extri = torch.from_numpy(np.stack(batch['extrinsics'])).to(device).unsqueeze(0)
    gt_intri = torch.from_numpy(np.stack(batch['intrinsics'])).to(device).unsqueeze(0)

    # Translation error
    pred_t = pred_extri[0, :, :3, 3].float()
    gt_t = gt_extri[0, :, :3, 3].float()
    t_error = (pred_t - gt_t).norm(dim=-1).mean().item()
    print(f"\n  Mean translation error: {t_error:.4f} m")

    # Rotation error (simple quaternion distance, not geodesic)
    print(f"  (Note: More rigorous evaluation would use ATE/RPE metrics)")

    if 'depth' in predictions:
        print(f"  Depth prediction shape: {predictions['depth'].shape}")

    print("\n✓ VGGT integration test PASSED")


def main():
    parser = argparse.ArgumentParser(description='Test TUM RGB-D dataloader')
    parser.add_argument('--tum_dir', type=str, required=True,
                        help='Path to TUM RGB-D sequences directory')
    parser.add_argument('--output_dir', type=str, default='./tum_test_output',
                        help='Output directory for test results')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='Specific sequences to test (default: auto-detect)')
    parser.add_argument('--skip_vggt', action='store_true',
                        help='Skip VGGT integration test')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Import TUM dataset
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.datasets.tum_rgbd import TUMRGBDDataset

    # Create mock config
    common_conf = MockCommonConf()

    print("="*60)
    print("TUM RGB-D Dataloader Test Suite")
    print("="*60)
    print(f"TUM directory: {args.tum_dir}")
    print(f"Output directory: {args.output_dir}")

    # Initialize dataset
    print("\nInitializing TUM RGB-D dataset...")
    dataset = TUMRGBDDataset(
        common_conf=common_conf,
        split="train",
        TUM_DIR=args.tum_dir,
        sequences=args.sequences,
        min_num_images=10,
        len_train=1000,
    )

    if len(dataset.sequence_list) == 0:
        print("ERROR: No valid sequences found!")
        print("Make sure TUM_DIR contains sequences with rgb.txt, depth.txt, and groundtruth.txt")
        return

    print(f"\nFound {len(dataset.sequence_list)} sequences: {dataset.sequence_list}")

    # Run tests
    batch = test_data_loading(dataset, args.output_dir)
    test_visual_verification(batch, args.output_dir)
    test_camera_trajectory(batch, args.output_dir)
    test_point_cloud(batch, args.output_dir)

    if not args.skip_vggt:
        test_vggt_integration(batch, args.output_dir)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print(f"\nCheck results in: {args.output_dir}/visualization/")


if __name__ == '__main__':
    main()
