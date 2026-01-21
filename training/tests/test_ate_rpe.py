#!/usr/bin/env python3
"""
Unit tests for ATE and RPE computation functions.

Run with:
    cd ~/Dev/vggt && mamba run -n vggt python training/tests/test_ate_rpe.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vggt_tum import compute_ate, compute_rpe, umeyama_alignment, rotation_error


def test_umeyama_identical_points():
    """Alignment of identical point sets should give identity transform."""
    print("Testing Umeyama with identical points...")
    points = np.random.randn(10, 3)
    R, t, s = umeyama_alignment(points, points, with_scale=False)

    assert np.allclose(R, np.eye(3), atol=1e-6), f"R should be identity, got:\n{R}"
    assert np.allclose(t, np.zeros(3), atol=1e-6), f"t should be zero, got: {t}"
    assert np.isclose(s, 1.0, atol=1e-6), f"s should be 1, got: {s}"
    print("  ✓ Passed")


def test_umeyama_with_scale():
    """Test Umeyama alignment with scale estimation."""
    print("Testing Umeyama with scale...")
    np.random.seed(42)
    x = np.random.randn(20, 3)

    # Apply scale only
    scale_true = 2.5
    y = x * scale_true

    R_est, t_est, s_est = umeyama_alignment(x, y, with_scale=True)
    x_reconstructed = s_est * (y @ R_est.T) + t_est

    assert np.allclose(x, x_reconstructed, atol=1e-6), \
        f"Reconstruction error: {np.max(np.abs(x - x_reconstructed))}"
    print(f"  Estimated scale: {s_est:.4f} (expected: {1/scale_true:.4f})")
    print("  ✓ Passed")


def test_ate_identical_poses():
    """ATE of identical poses should be zero."""
    print("Testing ATE with identical poses...")
    n_poses = 10
    poses = np.zeros((n_poses, 3, 4))

    for i in range(n_poses):
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        qx, qy, qz, qw = q
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        t = np.random.randn(3)
        poses[i, :3, :3] = R
        poses[i, :3, 3] = t

    result = compute_ate(poses, poses, align='sim3')

    print(f"  ATE Trans RMSE: {result['trans_rmse']:.2e}")
    print(f"  ATE Rot RMSE: {result['rot_rmse']:.2e}°")
    print(f"  Scale: {result['scale']:.6f}")

    assert result['trans_rmse'] < 1e-6, f"ATE trans should be ~0, got {result['trans_rmse']}"
    assert result['rot_rmse'] < 1e-3, f"ATE rot should be ~0, got {result['rot_rmse']}"
    print("  ✓ Passed")


def test_ate_no_align():
    """ATE without alignment on identical poses should be zero."""
    print("Testing ATE without alignment...")
    n_poses = 10
    poses = np.zeros((n_poses, 3, 4))

    for i in range(n_poses):
        poses[i, :3, :3] = np.eye(3)
        poses[i, :3, 3] = np.array([i * 0.1, 0, 0])

    result = compute_ate(poses, poses, align='none')

    print(f"  ATE Trans RMSE: {result['trans_rmse']:.2e}")
    print(f"  ATE Rot RMSE: {result['rot_rmse']:.2e}°")
    assert result['trans_rmse'] < 1e-10, f"ATE trans should be ~0, got {result['trans_rmse']}"
    assert result['rot_rmse'] < 1e-10, f"ATE rot should be ~0, got {result['rot_rmse']}"
    print("  ✓ Passed")


def test_rpe_identical_poses():
    """RPE of identical poses should be zero."""
    print("Testing RPE with identical poses...")
    n_poses = 10
    poses = np.zeros((n_poses, 3, 4))

    for i in range(n_poses):
        poses[i, :3, :3] = np.eye(3)
        poses[i, :3, 3] = np.array([i * 0.1, i * 0.05, 0])

    rpe_trans, rpe_rot = compute_rpe(poses, poses, delta=1)

    print(f"  RPE Trans: {rpe_trans:.2e}")
    print(f"  RPE Rot: {rpe_rot:.2e}°")

    assert rpe_trans < 1e-10, f"RPE trans should be ~0, got {rpe_trans}"
    assert rpe_rot < 1e-10, f"RPE rot should be ~0, got {rpe_rot}"
    print("  ✓ Passed")


def test_rpe_with_rotation():
    """RPE with rotating poses should still be zero for identical inputs."""
    print("Testing RPE with rotating poses...")
    n_poses = 10
    poses = np.zeros((n_poses, 3, 4))

    for i in range(n_poses):
        angle = i * 0.1
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        poses[i, :3, :3] = R
        poses[i, :3, 3] = np.array([i * 0.1, 0, 0])

    rpe_trans, rpe_rot = compute_rpe(poses, poses, delta=1)

    print(f"  RPE Trans: {rpe_trans:.2e}")
    print(f"  RPE Rot: {rpe_rot:.2e}°")

    assert rpe_trans < 1e-10, f"RPE trans should be ~0, got {rpe_trans}"
    assert rpe_rot < 1e-10, f"RPE rot should be ~0, got {rpe_rot}"
    print("  ✓ Passed")


def test_full_pipeline_gt_vs_gt():
    """Full pipeline test: GT vs GT should give near-zero errors."""
    print("\nTesting full pipeline (GT vs GT)...")
    np.random.seed(123)
    n_poses = 20
    poses = np.zeros((n_poses, 3, 4))

    # Simulate a realistic trajectory
    for i in range(n_poses):
        angle = i * 0.05
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        pos = np.array([
            np.sin(i * 0.1) * 0.5,
            i * 0.02,
            np.cos(i * 0.1) * 0.1
        ])
        poses[i, :3, :3] = R
        poses[i, :3, 3] = -R @ pos

    # Test ATE with Sim3
    result = compute_ate(poses, poses, align='sim3')

    # Test RPE
    rpe_trans, rpe_rot = compute_rpe(poses, poses, delta=1)

    print(f"  ATE Trans RMSE: {result['trans_rmse']:.2e}")
    print(f"  ATE Rot RMSE: {result['rot_rmse']:.2e}°")
    print(f"  RPE Trans: {rpe_trans:.2e}")
    print(f"  RPE Rot: {rpe_rot:.2e}°")
    print(f"  Scale: {result['scale']:.6f}")

    assert result['trans_rmse'] < 1e-6, f"ATE trans should be ~0, got {result['trans_rmse']}"
    assert result['rot_rmse'] < 1e-3, f"ATE rot should be ~0, got {result['rot_rmse']}"
    assert np.isclose(result['scale'], 1.0, atol=1e-3), f"Scale should be ~1, got {result['scale']}"
    assert rpe_trans < 1e-10, f"RPE trans should be ~0, got {rpe_trans}"
    assert rpe_rot < 1e-10, f"RPE rot should be ~0, got {rpe_rot}"
    print("  ✓ Passed")


def test_sim3_alignment_on_positions():
    """Test that Sim3 alignment can recover arbitrary Sim3 transformation on positions."""
    print("\nTesting Sim3 alignment on positions...")
    np.random.seed(456)

    # Create GT camera positions (not full poses, just positions)
    n_points = 20
    gt_positions = np.zeros((n_points, 3))
    for i in range(n_points):
        gt_positions[i] = np.array([i * 0.1, np.sin(i * 0.2) * 0.3, 0.5])

    # Create arbitrary Sim3 transformation
    theta = np.pi / 4  # 45° around Z
    R_sim3 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    t_sim3 = np.array([1.5, -2.0, 0.8])
    s_sim3 = 2.5

    # Apply Sim3 to positions
    pred_positions = s_sim3 * (gt_positions @ R_sim3.T) + t_sim3

    # Use Umeyama to align pred back to gt
    R_align, t_align, s_align = umeyama_alignment(gt_positions, pred_positions, with_scale=True)
    aligned_pred = s_align * (pred_positions @ R_align.T) + t_align

    # Compute error
    errors = np.linalg.norm(aligned_pred - gt_positions, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"  Applied Sim3: scale={s_sim3}, rotation=45°, translation={t_sim3}")
    print(f"  Recovered scale: {s_align:.4f} (expected: {1/s_sim3:.4f})")
    print(f"  RMSE after alignment: {rmse:.2e}")

    assert rmse < 1e-10, f"RMSE should be ~0, got {rmse}"
    assert np.isclose(s_align, 1/s_sim3, atol=1e-6), f"Scale should be {1/s_sim3:.4f}, got {s_align:.4f}"
    print("  ✓ Passed")


def test_sim3_alignment_on_poses():
    """Test Sim3 alignment through the full compute_ate pipeline."""
    print("\nTesting Sim3 alignment on full poses...")
    np.random.seed(789)
    n_poses = 20

    # Create GT poses with identity rotation for simplicity
    gt_poses = np.zeros((n_poses, 3, 4))
    for i in range(n_poses):
        gt_poses[i, :3, :3] = np.eye(3)
        # Camera at position [i*0.1, sin, 0.5], so t = -R @ pos = -pos
        pos = np.array([i * 0.1, np.sin(i * 0.2) * 0.3, 0.5])
        gt_poses[i, :3, 3] = -pos

    # Create pred poses with Sim3-transformed positions
    theta = np.pi / 3  # 60° around Z
    R_sim3 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    t_sim3 = np.array([2.0, -1.0, 0.5])
    s_sim3 = 1.8

    pred_poses = np.zeros((n_poses, 3, 4))
    for i in range(n_poses):
        pred_poses[i, :3, :3] = np.eye(3)  # Keep rotation as identity
        gt_pos = np.array([i * 0.1, np.sin(i * 0.2) * 0.3, 0.5])
        pred_pos = s_sim3 * (R_sim3 @ gt_pos) + t_sim3
        pred_poses[i, :3, 3] = -pred_pos

    # Test Sim3 alignment
    result_sim3 = compute_ate(pred_poses, gt_poses, align='sim3')

    # Test SE3 alignment (should have larger error)
    result_se3 = compute_ate(pred_poses, gt_poses, align='se3')

    print(f"  Applied Sim3: scale={s_sim3}, rotation=60°")
    print(f"  Sim3: ATE trans={result_sim3['trans_rmse']:.2e}, rot={result_sim3['rot_rmse']:.2f}°, scale={result_sim3['scale']:.4f}")
    print(f"  SE3:  ATE trans={result_se3['trans_rmse']:.2e}, rot={result_se3['rot_rmse']:.2f}°")

    assert result_sim3['trans_rmse'] < 1e-10, f"Sim3 ATE trans should be ~0, got {result_sim3['trans_rmse']}"
    assert np.isclose(result_sim3['scale'], 1/s_sim3, atol=1e-6), f"Scale should be {1/s_sim3:.4f}, got {result_sim3['scale']:.4f}"
    print("  ✓ Passed")


def test_sim3_alignment_full_transform():
    """Test Sim3 alignment when BOTH position and rotation are transformed.

    This simulates the case where a SLAM system outputs poses in a different
    coordinate frame (related by Sim3). After alignment, both translation
    and rotation errors should be ~0.
    """
    print("\nTesting Sim3 alignment with full pose transform...")
    np.random.seed(790)
    n_poses = 20

    # Create GT poses with varying rotations
    gt_poses = np.zeros((n_poses, 3, 4))
    for i in range(n_poses):
        # Varying rotation around Z axis
        angle = i * 0.1
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        pos = np.array([i * 0.1, np.sin(i * 0.2) * 0.3, 0.5])
        gt_poses[i, :3, :3] = R
        gt_poses[i, :3, 3] = -R @ pos  # w2c: t = -R @ pos

    # Sim3 transformation parameters
    theta = np.pi / 3  # 60° around Z
    R_sim3 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    t_sim3 = np.array([2.0, -1.0, 0.5])
    s_sim3 = 1.8

    # Create pred poses by transforming GT poses with Sim3
    # For w2c poses: if world transforms by Sim3, then:
    # - New position: p_new = s * R_sim3 @ p_old + t_sim3
    # - New rotation: R_wc_new = R_wc_old @ R_sim3.T
    pred_poses = np.zeros((n_poses, 3, 4))
    for i in range(n_poses):
        R_gt = gt_poses[i, :3, :3]
        t_gt = gt_poses[i, :3, 3]
        # Get GT camera position
        pos_gt = -R_gt.T @ t_gt
        # Transform position by Sim3
        pos_pred = s_sim3 * (R_sim3 @ pos_gt) + t_sim3
        # Transform rotation: R_pred = R_gt @ R_sim3.T
        R_pred = R_gt @ R_sim3.T
        pred_poses[i, :3, :3] = R_pred
        pred_poses[i, :3, 3] = -R_pred @ pos_pred  # w2c: t = -R @ pos

    # Test Sim3 alignment - both trans and rot should be ~0
    result_sim3 = compute_ate(pred_poses, gt_poses, align='sim3')

    print(f"  Applied Sim3: scale={s_sim3}, rotation=60°")
    print(f"  Recovered scale: {result_sim3['scale']:.4f} (expected: {1/s_sim3:.4f})")
    print(f"  ATE Trans: {result_sim3['trans_rmse']:.2e}")
    print(f"  ATE Rot: {result_sim3['rot_rmse']:.2e}°")

    assert result_sim3['trans_rmse'] < 1e-6, f"Sim3 ATE trans should be ~0, got {result_sim3['trans_rmse']}"
    assert result_sim3['rot_rmse'] < 1e-3, f"Sim3 ATE rot should be ~0, got {result_sim3['rot_rmse']}"
    assert np.isclose(result_sim3['scale'], 1/s_sim3, atol=1e-3), f"Scale should be {1/s_sim3:.4f}, got {result_sim3['scale']:.4f}"
    print("  ✓ Passed")


def test_tum_gt_vs_gt():
    """
    End-to-end test using real TUM dataset.
    Load GT poses from TUM, compute ATE(GT, GT), should be ~0.
    This tests the entire pipeline including pose loading and conversion.
    """
    print("\nTesting TUM GT vs GT (end-to-end pipeline)...")

    tum_dir = os.path.expanduser("~/Dev/tum_rgbd")
    if not os.path.exists(tum_dir):
        print("  ⚠ TUM data not found, skipping test")
        return

    # Import TUM dataset loader
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    try:
        from data.datasets.tum_rgbd import TUMRGBDDataset

        # Create mock config
        class MockConf:
            img_size = 518
            patch_size = 14
            debug = False
            training = False
            get_nearby = False
            load_depth = True  # Must be True for process_one_image to work
            inside_random = False
            allow_duplicate_img = False
            landscape_check = False
            rescale = True
            rescale_aug = False
            augs = type('obj', (object,), {'scales': None})()

        dataset = TUMRGBDDataset(
            common_conf=MockConf(),
            split="test",
            TUM_DIR=tum_dir,
            min_num_images=10,
        )

        # Get data with uniform sampling
        seq_name = dataset.sequence_list[0]
        seq_len = len(dataset.data_store[seq_name])
        n_frames = min(20, seq_len)
        ids = np.linspace(0, seq_len - 1, n_frames, dtype=int).tolist()

        batch = dataset.get_data(seq_index=0, img_per_seq=n_frames, ids=ids, aspect_ratio=1.0)
        gt_poses = np.stack(batch['extrinsics'], axis=0)

        print(f"  Loaded {len(gt_poses)} GT poses from {seq_name}")

        # Test: GT vs GT should give zero error
        result = compute_ate(gt_poses, gt_poses, align='sim3')
        rpe_trans, rpe_rot = compute_rpe(gt_poses, gt_poses, delta=1)

        print(f"  GT vs GT results:")
        print(f"    ATE Trans RMSE: {result['trans_rmse']:.2e}")
        print(f"    ATE Rot RMSE: {result['rot_rmse']:.2e}°")
        print(f"    RPE Trans: {rpe_trans:.2e}")
        print(f"    RPE Rot: {rpe_rot:.2e}°")
        print(f"    Scale: {result['scale']:.6f}")

        # Tolerances account for numerical precision (image resizing, coordinate transforms)
        assert result['trans_rmse'] < 1e-6, f"GT vs GT ATE trans should be ~0, got {result['trans_rmse']}"
        assert result['rot_rmse'] < 0.1, f"GT vs GT ATE rot should be ~0, got {result['rot_rmse']}"
        assert rpe_trans < 1e-6, f"GT vs GT RPE trans should be ~0, got {rpe_trans}"
        assert rpe_rot < 0.1, f"GT vs GT RPE rot should be ~0, got {rpe_rot}"
        assert np.isclose(result['scale'], 1.0, atol=1e-4), f"Scale should be 1.0, got {result['scale']}"

        print("  ✓ Passed")

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        raise


def create_realistic_trajectory(n_poses=50, seed=42):
    """Create a realistic camera trajectory (circular motion with forward movement)."""
    np.random.seed(seed)
    poses = np.zeros((n_poses, 3, 4))

    for i in range(n_poses):
        t = i / n_poses
        # Circular path with some forward motion
        pos = np.array([
            np.sin(t * 2 * np.pi) * 1.0,  # X: circular
            t * 2.0,                        # Y: forward
            np.cos(t * 2 * np.pi) * 0.5,   # Z: circular (smaller)
        ])
        # Camera looks roughly forward (along Y) with some rotation
        angle_y = t * 2 * np.pi * 0.3  # Yaw
        angle_x = np.sin(t * np.pi) * 0.2  # Pitch
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        R = Ry @ Rx
        poses[i, :3, :3] = R
        poses[i, :3, 3] = -R @ pos  # w2c convention: t = -R @ pos

    return poses


def add_translation_noise(poses, noise_std=0.05):
    """Add Gaussian noise to translation."""
    noisy = poses.copy()
    n = len(poses)
    noise = np.random.randn(n, 3) * noise_std
    noisy[:, :3, 3] += noise
    return noisy


def add_rotation_noise(poses, noise_std_deg=2.0):
    """Add small rotation noise (axis-angle)."""
    noisy = poses.copy()
    noise_std_rad = np.deg2rad(noise_std_deg)

    for i in range(len(poses)):
        # Random axis-angle noise
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        angle = np.random.randn() * noise_std_rad

        # Rodrigues formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        noisy[i, :3, :3] = R_noise @ poses[i, :3, :3]

    return noisy


def add_scale_error(poses, scale=1.5):
    """Apply global scale to translations (simulating scale ambiguity)."""
    scaled = poses.copy()
    scaled[:, :3, 3] *= scale
    return scaled


def test_disturbed_translation():
    """Test ATE/RPE with translation noise."""
    print("\nTesting with translation noise...")
    np.random.seed(100)

    gt_poses = create_realistic_trajectory(n_poses=30)
    noise_std = 0.03  # 3cm noise

    pred_poses = add_translation_noise(gt_poses, noise_std=noise_std)

    # ATE should be approximately the noise level
    result_sim3 = compute_ate(pred_poses, gt_poses, align='sim3')
    result_se3 = compute_ate(pred_poses, gt_poses, align='se3')
    scale = result_sim3['scale']

    # RPE - use aligned poses for Sim3 comparison
    rpe_trans_raw, rpe_rot_raw = compute_rpe(pred_poses, gt_poses, delta=1)
    rpe_trans_sim3, rpe_rot_sim3 = compute_rpe(result_sim3['aligned_poses'], gt_poses, delta=1)

    print(f"  Noise std: {noise_std*100:.1f} cm")
    print(f"  ATE Trans (SE3):  {result_se3['trans_rmse']*100:.2f} cm")
    print(f"  ATE Trans (Sim3): {result_sim3['trans_rmse']*100:.2f} cm, scale={scale:.3f}")
    print(f"  ATE Rot (Sim3): {result_sim3['rot_rmse']:.2f}°")
    print(f"  RPE Trans (raw): {rpe_trans_raw*100:.2f} cm")
    print(f"  RPE Trans (Sim3): {rpe_trans_sim3*100:.2f} cm")
    print(f"  RPE Rot: {rpe_rot_raw:.2f}°")

    # ATE should be on the order of noise_std (within factor of 3)
    assert result_sim3['trans_rmse'] < noise_std * 3, f"ATE too large: {result_sim3['trans_rmse']}"
    assert result_sim3['trans_rmse'] > noise_std * 0.3, f"ATE suspiciously small: {result_sim3['trans_rmse']}"
    # Scale should be close to 1 (no scale error added)
    assert np.isclose(scale, 1.0, atol=0.1), f"Scale should be ~1, got {scale}"
    # RPE rotation should be near zero (we only added translation noise)
    assert rpe_rot_raw < 1.0, f"RPE rot should be small: {rpe_rot_raw}"

    print("  ✓ Passed")


def test_disturbed_rotation():
    """Test ATE/RPE with rotation noise."""
    print("\nTesting with rotation noise...")
    np.random.seed(101)

    gt_poses = create_realistic_trajectory(n_poses=30)
    noise_std_deg = 3.0  # 3 degree noise

    pred_poses = add_rotation_noise(gt_poses, noise_std_deg=noise_std_deg)

    # ATE
    result_sim3 = compute_ate(pred_poses, gt_poses, align='sim3')
    scale = result_sim3['scale']

    # RPE - use aligned poses for Sim3 comparison
    rpe_trans_raw, rpe_rot_raw = compute_rpe(pred_poses, gt_poses, delta=1)
    rpe_trans_sim3, rpe_rot_sim3 = compute_rpe(result_sim3['aligned_poses'], gt_poses, delta=1)

    print(f"  Rotation noise std: {noise_std_deg:.1f}°")
    print(f"  ATE Trans (Sim3): {result_sim3['trans_rmse']*100:.2f} cm, scale={scale:.3f}")
    print(f"  ATE Rot (Sim3): {result_sim3['rot_rmse']:.2f}°")
    print(f"  RPE Trans (Sim3): {rpe_trans_sim3*100:.2f} cm")
    print(f"  RPE Rot: {rpe_rot_raw:.2f}°")

    # ATE rotation should be on the order of noise_std_deg
    assert result_sim3['rot_rmse'] < noise_std_deg * 3, f"ATE rot too large: {result_sim3['rot_rmse']}"
    assert result_sim3['rot_rmse'] > noise_std_deg * 0.2, f"ATE rot suspiciously small: {result_sim3['rot_rmse']}"
    # RPE rotation should also be on the order of noise_std_deg
    assert rpe_rot_raw < noise_std_deg * 3, f"RPE rot too large: {rpe_rot_raw}"

    print("  ✓ Passed")


def test_disturbed_scale():
    """Test ATE/RPE with scale error (simulating monocular scale ambiguity)."""
    print("\nTesting with scale error...")
    np.random.seed(102)

    gt_poses = create_realistic_trajectory(n_poses=30)
    true_scale = 2.5  # Predictions are 2.5x larger

    pred_poses = add_scale_error(gt_poses, scale=true_scale)

    # Without scale alignment, ATE should be large
    result_none = compute_ate(pred_poses, gt_poses, align='none')

    # With Sim3, should recover scale and get near-zero ATE
    result_sim3 = compute_ate(pred_poses, gt_poses, align='sim3')
    recovered_scale = result_sim3['scale']

    # RPE without alignment should be large (scale mismatch)
    rpe_trans_raw, rpe_rot_raw = compute_rpe(pred_poses, gt_poses, delta=1)
    # RPE with Sim3 aligned poses should be near zero
    rpe_trans_sim3, rpe_rot_sim3 = compute_rpe(result_sim3['aligned_poses'], gt_poses, delta=1)

    print(f"  True scale: {true_scale}")
    print(f"  Recovered scale: {recovered_scale:.4f} (expected: {1/true_scale:.4f})")
    print(f"  ATE Trans (no align): {result_none['trans_rmse']*100:.2f} cm")
    print(f"  ATE Trans (Sim3): {result_sim3['trans_rmse']*100:.2f} cm")
    print(f"  ATE Rot (Sim3): {result_sim3['rot_rmse']:.2f}°")
    print(f"  RPE Trans (no align): {rpe_trans_raw*100:.2f} cm")
    print(f"  RPE Trans (Sim3): {rpe_trans_sim3*100:.2f} cm")
    print(f"  RPE Rot: {rpe_rot_raw:.2f}°")

    # Sim3 should recover the scale
    assert np.isclose(recovered_scale, 1/true_scale, rtol=0.01), \
        f"Scale should be {1/true_scale:.4f}, got {recovered_scale:.4f}"
    # ATE with Sim3 should be near zero
    assert result_sim3['trans_rmse'] < 1e-6, f"Sim3 ATE trans should be ~0, got {result_sim3['trans_rmse']}"
    assert result_sim3['rot_rmse'] < 1e-3, f"Sim3 ATE rot should be ~0, got {result_sim3['rot_rmse']}"
    # RPE with scale correction should be near zero
    assert rpe_trans_sim3 < 1e-6, f"RPE with scale should be ~0, got {rpe_trans_sim3}"
    # RPE rotation should be zero (no rotation error)
    assert rpe_rot_raw < 1e-6, f"RPE rot should be ~0, got {rpe_rot_raw}"

    print("  ✓ Passed")


def test_disturbed_combined():
    """Test with combined noise: translation + rotation + scale (realistic scenario)."""
    print("\nTesting with combined noise (realistic scenario)...")
    np.random.seed(103)

    gt_poses = create_realistic_trajectory(n_poses=50)

    # Apply combined disturbances
    pred_poses = gt_poses.copy()
    pred_poses = add_translation_noise(pred_poses, noise_std=0.02)  # 2cm
    pred_poses = add_rotation_noise(pred_poses, noise_std_deg=1.5)  # 1.5°
    pred_poses = add_scale_error(pred_poses, scale=1.8)             # 1.8x scale

    # Compute metrics
    result_se3 = compute_ate(pred_poses, gt_poses, align='se3')
    result_sim3 = compute_ate(pred_poses, gt_poses, align='sim3')
    scale = result_sim3['scale']

    rpe_trans_raw, rpe_rot_raw = compute_rpe(pred_poses, gt_poses, delta=1)
    rpe_trans_sim3, rpe_rot_sim3 = compute_rpe(result_sim3['aligned_poses'], gt_poses, delta=1)

    print(f"  Applied: trans_noise=2cm, rot_noise=1.5°, scale=1.8x")
    print(f"  Recovered scale: {scale:.4f} (expected: {1/1.8:.4f})")
    print(f"  ATE Trans (SE3):  {result_se3['trans_rmse']*100:.2f} cm")
    print(f"  ATE Trans (Sim3): {result_sim3['trans_rmse']*100:.2f} cm")
    print(f"  ATE Rot (SE3):  {result_se3['rot_rmse']:.2f}°")
    print(f"  ATE Rot (Sim3): {result_sim3['rot_rmse']:.2f}°")
    print(f"  RPE Trans (raw):  {rpe_trans_raw*100:.2f} cm")
    print(f"  RPE Trans (Sim3): {rpe_trans_sim3*100:.2f} cm")
    print(f"  RPE Rot (raw):  {rpe_rot_raw:.2f}°")
    print(f"  RPE Rot (Sim3): {rpe_rot_sim3:.2f}°")

    # Scale should be approximately recovered
    assert np.isclose(scale, 1/1.8, rtol=0.1), f"Scale recovery failed: {scale}"
    # Sim3 ATE should be smaller than SE3 (since we have scale error)
    assert result_sim3['trans_rmse'] < result_se3['trans_rmse'], "Sim3 should be better than SE3 with scale error"
    # RPE with scale should be better than raw
    assert rpe_trans_sim3 < rpe_trans_raw, "RPE with scale should be better"
    # Rotation errors should be similar (scale doesn't affect rotation)
    assert np.isclose(rpe_rot_raw, rpe_rot_sim3, atol=0.1), "RPE rot should be same with/without scale"

    print("  ✓ Passed")


def test_rpe_with_scale_alignment():
    """Verify that RPE with Sim3 aligned poses correctly handles scale."""
    print("\nTesting RPE with scale alignment...")
    np.random.seed(104)

    gt_poses = create_realistic_trajectory(n_poses=20)

    # Create predictions with only scale difference
    scale_true = 3.0
    pred_poses = add_scale_error(gt_poses, scale=scale_true)

    # Use Sim3 alignment to get aligned poses
    ate_result = compute_ate(pred_poses, gt_poses, align='sim3')

    # RPE without alignment (should be large due to scale mismatch)
    rpe_trans_raw, rpe_rot_raw = compute_rpe(pred_poses, gt_poses, delta=1)

    # RPE with Sim3 aligned poses (should be ~0)
    rpe_trans_aligned, rpe_rot_aligned = compute_rpe(ate_result['aligned_poses'], gt_poses, delta=1)

    print(f"  Scale applied: {scale_true}x")
    print(f"  Recovered scale: {ate_result['scale']:.4f} (expected: {1/scale_true:.4f})")
    print(f"  RPE Trans (no alignment): {rpe_trans_raw*100:.2f} cm")
    print(f"  RPE Trans (Sim3 aligned): {rpe_trans_aligned*100:.2f} cm")
    print(f"  RPE Rot (no alignment): {rpe_rot_raw:.4f}°")
    print(f"  RPE Rot (Sim3 aligned): {rpe_rot_aligned:.4f}°")

    # With Sim3 aligned poses, RPE trans should be ~0
    assert rpe_trans_aligned < 1e-6, f"Aligned RPE trans should be ~0, got {rpe_trans_aligned}"
    # Without alignment, RPE trans should be large
    assert rpe_trans_raw > 0.01, f"Raw RPE trans should be large, got {rpe_trans_raw}"
    # Rotation should be the same (scale doesn't affect rotation)
    assert np.isclose(rpe_rot_raw, rpe_rot_aligned, atol=1e-6), \
        f"Rotation should be same: {rpe_rot_raw} vs {rpe_rot_aligned}"

    print("  ✓ Passed")


def test_rpe_with_rotation_alignment():
    """Test RPE with Sim3 aligned poses when world frames differ by rotation.

    This tests the case where predictions are in a different world frame
    (rotated relative to GT). With properly aligned poses, RPE should be ~0.
    """
    print("\nTesting RPE with rotation alignment...")
    np.random.seed(105)

    gt_poses = create_realistic_trajectory(n_poses=30)

    # Create predictions in a rotated world frame
    # For w2c poses with world rotated by R_world: R_pred = R_gt @ R_world.T
    theta = np.pi / 4  # 45° rotation around Z
    R_world = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    pred_poses = np.zeros_like(gt_poses)
    for i in range(len(gt_poses)):
        R_gt = gt_poses[i, :3, :3]
        t_gt = gt_poses[i, :3, 3]
        # Get GT camera position
        pos_gt = -R_gt.T @ t_gt
        # Transform position to different world frame: pos_pred = R_world @ pos_gt
        pos_pred = R_world @ pos_gt
        # For w2c: R_pred = R_gt @ R_world.T
        R_pred = R_gt @ R_world.T
        pred_poses[i, :3, :3] = R_pred
        pred_poses[i, :3, 3] = -R_pred @ pos_pred

    # Use compute_ate to get aligned poses
    ate_result = compute_ate(pred_poses, gt_poses, align='sim3')
    R_align = ate_result['R_align']
    scale = ate_result['scale']

    print(f"  World frame rotation: {theta * 180 / np.pi:.0f}°")
    print(f"  R_align rotation angle: {rotation_error(R_align, np.eye(3)):.1f}°")
    print(f"  Scale: {scale:.4f}")

    # RPE without alignment
    rpe_trans_raw, rpe_rot_raw = compute_rpe(pred_poses, gt_poses, delta=1)

    # RPE with Sim3 aligned poses
    rpe_trans_aligned, rpe_rot_aligned = compute_rpe(ate_result['aligned_poses'], gt_poses, delta=1)

    print(f"  RPE Rot (no alignment): {rpe_rot_raw:.2f}°")
    print(f"  RPE Rot (Sim3 aligned): {rpe_rot_aligned:.2f}°")
    print(f"  RPE Trans (no alignment): {rpe_trans_raw*100:.2f} cm")
    print(f"  RPE Trans (Sim3 aligned): {rpe_trans_aligned*100:.2f} cm")

    # With Sim3 aligned poses, RPE should be ~0 (perfect alignment of world frames)
    assert rpe_rot_aligned < 0.1, f"Aligned RPE rot should be ~0, got {rpe_rot_aligned}"
    assert rpe_trans_aligned < 1e-6, f"Aligned RPE trans should be ~0, got {rpe_trans_aligned}"

    print("  ✓ Passed")


def main():
    print("=" * 60)
    print("ATE/RPE Unit Tests")
    print("=" * 60)

    tests = [
        test_umeyama_identical_points,
        test_umeyama_with_scale,
        test_ate_identical_poses,
        test_ate_no_align,
        test_rpe_identical_poses,
        test_rpe_with_rotation,
        test_full_pipeline_gt_vs_gt,
        test_sim3_alignment_on_positions,
        test_sim3_alignment_on_poses,
        test_sim3_alignment_full_transform,
        test_tum_gt_vs_gt,  # End-to-end test with real TUM data
        # Disturbed trajectory tests
        test_disturbed_translation,
        test_disturbed_rotation,
        test_disturbed_scale,
        test_disturbed_combined,
        test_rpe_with_scale_alignment,
        test_rpe_with_rotation_alignment,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
