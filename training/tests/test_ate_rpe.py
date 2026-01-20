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
from eval_vggt_tum import compute_ate, compute_rpe, umeyama_alignment


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

    ate_rmse, ate_mean, aligned_pred, gt_pos, scale = compute_ate(poses, poses, align='sim3')

    print(f"  ATE RMSE: {ate_rmse:.2e}")
    print(f"  ATE Mean: {ate_mean:.2e}")
    print(f"  Scale: {scale:.6f}")

    assert ate_rmse < 1e-6, f"ATE RMSE should be ~0, got {ate_rmse}"
    assert ate_mean < 1e-6, f"ATE mean should be ~0, got {ate_mean}"
    print("  ✓ Passed")


def test_ate_no_align():
    """ATE without alignment on identical poses should be zero."""
    print("Testing ATE without alignment...")
    n_poses = 10
    poses = np.zeros((n_poses, 3, 4))

    for i in range(n_poses):
        poses[i, :3, :3] = np.eye(3)
        poses[i, :3, 3] = np.array([i * 0.1, 0, 0])

    ate_rmse, ate_mean, _, _, _ = compute_ate(poses, poses, align='none')

    print(f"  ATE RMSE: {ate_rmse:.2e}")
    assert ate_rmse < 1e-10, f"ATE RMSE should be ~0, got {ate_rmse}"
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
    ate_rmse, ate_mean, _, _, scale = compute_ate(poses, poses, align='sim3')

    # Test RPE
    rpe_trans, rpe_rot = compute_rpe(poses, poses, delta=1)

    print(f"  ATE RMSE: {ate_rmse:.2e}")
    print(f"  ATE Mean: {ate_mean:.2e}")
    print(f"  RPE Trans: {rpe_trans:.2e}")
    print(f"  RPE Rot: {rpe_rot:.2e}°")
    print(f"  Scale: {scale:.6f}")

    assert ate_rmse < 1e-6, f"ATE RMSE should be ~0, got {ate_rmse}"
    assert np.isclose(scale, 1.0, atol=1e-3), f"Scale should be ~1, got {scale}"
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
    ate_rmse_sim3, _, _, _, scale = compute_ate(pred_poses, gt_poses, align='sim3')

    # Test SE3 alignment (should have larger error)
    ate_rmse_se3, _, _, _, _ = compute_ate(pred_poses, gt_poses, align='se3')

    print(f"  Applied Sim3: scale={s_sim3}, rotation=60°")
    print(f"  Sim3 alignment: ATE RMSE={ate_rmse_sim3:.2e}, scale={scale:.4f}")
    print(f"  SE3 alignment:  ATE RMSE={ate_rmse_se3:.2e}")

    assert ate_rmse_sim3 < 1e-10, f"Sim3 ATE should be ~0, got {ate_rmse_sim3}"
    assert np.isclose(scale, 1/s_sim3, atol=1e-6), f"Scale should be {1/s_sim3:.4f}, got {scale:.4f}"
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
        ate_rmse, ate_mean, _, _, scale = compute_ate(gt_poses, gt_poses, align='sim3')
        rpe_trans, rpe_rot = compute_rpe(gt_poses, gt_poses, delta=1)

        print(f"  GT vs GT results:")
        print(f"    ATE RMSE: {ate_rmse:.2e}")
        print(f"    ATE Mean: {ate_mean:.2e}")
        print(f"    RPE Trans: {rpe_trans:.2e}")
        print(f"    RPE Rot: {rpe_rot:.2e}°")
        print(f"    Scale: {scale:.6f}")

        # Tolerances account for numerical precision (image resizing, coordinate transforms)
        assert ate_rmse < 1e-6, f"GT vs GT ATE should be ~0, got {ate_rmse}"
        assert rpe_trans < 1e-6, f"GT vs GT RPE trans should be ~0, got {rpe_trans}"
        assert rpe_rot < 0.01, f"GT vs GT RPE rot should be ~0, got {rpe_rot}"
        assert np.isclose(scale, 1.0, atol=1e-4), f"Scale should be 1.0, got {scale}"

        print("  ✓ Passed")

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        raise


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
        test_tum_gt_vs_gt,  # End-to-end test with real TUM data
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
