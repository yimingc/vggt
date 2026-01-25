"""
Unit tests for SE(3) Lie algebra utilities and convention verification.

These tests verify:
1. PyPose conventions match VGGT's pose encoding conventions
2. Log/Exp roundtrip works correctly
3. Scale fitting is correct
4. Window-relative poses have frame 0 as identity

IMPORTANT: Run these tests BEFORE training to verify conventions!
"""

import torch
import torch.nn.functional as F
import pypose as pp
import pytest
import numpy as np

from vggt.utils.lie_algebra import (
    pose_encoding_to_se3,
    compute_window_relative_poses,
    compute_window_scale_batched,
    compute_se3_residual,
    reconstruct_scaled_se3,
    split_se3_tangent,
    concat_se3_tangent,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class TestDimensionConvention:
    """Test dimension convention functions."""

    def test_split_se3_tangent(self):
        """Verify split_se3_tangent correctly splits [v, w] vectors."""
        # Test on 1D tensor
        test_1d = torch.arange(6).float()
        v, w = split_se3_tangent(test_1d)
        assert v.tolist() == [0, 1, 2], f"Expected [0, 1, 2], got {v.tolist()}"
        assert w.tolist() == [3, 4, 5], f"Expected [3, 4, 5], got {w.tolist()}"

        # Test on batched tensor [..., 6]
        B, S = 2, 4
        test_batched = torch.randn(B, S, 6)

        trans, rot = split_se3_tangent(test_batched)

        assert trans.shape == (B, S, 3), f"Expected shape {(B, S, 3)}, got {trans.shape}"
        assert rot.shape == (B, S, 3), f"Expected shape {(B, S, 3)}, got {rot.shape}"

        # Verify they extract the correct parts
        assert torch.equal(trans, test_batched[..., :3])
        assert torch.equal(rot, test_batched[..., 3:])

        print("split_se3_tangent works correctly")

    def test_concat_se3_tangent(self):
        """Verify concat_se3_tangent is inverse of split_se3_tangent."""
        B, S = 2, 4
        original = torch.randn(B, S, 6)

        v, w = split_se3_tangent(original)
        reconstructed = concat_se3_tangent(v, w)

        assert torch.equal(original, reconstructed), "concat should be inverse of split"

        print("concat_se3_tangent works correctly")

    def test_split_concat_roundtrip(self):
        """Verify split -> concat is identity."""
        test_cases = [
            torch.randn(6),           # 1D
            torch.randn(10, 6),       # 2D
            torch.randn(2, 4, 6),     # 3D (batched)
            torch.randn(2, 4, 8, 6),  # 4D
        ]

        for original in test_cases:
            v, w = split_se3_tangent(original)
            reconstructed = concat_se3_tangent(v, w)
            assert torch.equal(original, reconstructed), \
                f"Roundtrip failed for shape {original.shape}"

        print("split/concat roundtrip verified for all shapes")


class TestPyPoseConventions:
    """Test that PyPose conventions match expected behavior."""

    def test_so3_quaternion_order(self):
        """Verify SO3.tensor() returns [qx, qy, qz, qw]."""
        q_input = torch.tensor([[0.1, 0.2, 0.3, 0.9]])  # [qx, qy, qz, qw]
        q_input = F.normalize(q_input, dim=-1)

        T = pp.SE3(torch.cat([torch.zeros(1, 3), q_input], dim=-1))
        q_output = T.rotation().tensor()

        assert torch.allclose(q_input, q_output, atol=1e-6), \
            f"SO3.tensor() quaternion order mismatch! Input: {q_input}, Output: {q_output}"
        print("SO3.tensor() returns [qx, qy, qz, qw] as expected")

    def test_log_exp_roundtrip(self):
        """Verify Log(Exp(xi)) = xi and Exp(Log(T)) = T."""
        # Test with random se(3) vectors
        torch.manual_seed(42)
        xi = torch.randn(5, 6) * 0.5  # small perturbations

        # Exp then Log
        T = pp.se3(xi).Exp()
        xi_recovered = T.Log()
        assert torch.allclose(xi, xi_recovered, atol=1e-5), \
            f"Log(Exp(xi)) != xi. Max diff: {(xi - xi_recovered).abs().max()}"

        # Test with random SE3
        q = F.normalize(torch.randn(5, 4), dim=-1)
        t = torch.randn(5, 3)
        T = pp.SE3(torch.cat([t, q], dim=-1))

        # Log then Exp
        xi = T.Log()
        T_recovered = pp.se3(xi).Exp()

        # Compare matrices since SE3 representation may differ
        assert torch.allclose(T.matrix(), T_recovered.matrix(), atol=1e-5), \
            f"Exp(Log(T)) != T. Max diff: {(T.matrix() - T_recovered.matrix()).abs().max()}"

        print("Log/Exp roundtrip verified")

    def test_se3_composition(self):
        """Verify SE3 composition: (T1 @ T2).matrix() = T1.matrix() @ T2.matrix()."""
        torch.manual_seed(42)

        q1 = F.normalize(torch.randn(3, 4), dim=-1)
        t1 = torch.randn(3, 3)
        T1 = pp.SE3(torch.cat([t1, q1], dim=-1))

        q2 = F.normalize(torch.randn(3, 4), dim=-1)
        t2 = torch.randn(3, 3)
        T2 = pp.SE3(torch.cat([t2, q2], dim=-1))

        T12_pypose = (T1 @ T2).matrix()
        T12_manual = T1.matrix() @ T2.matrix()

        assert torch.allclose(T12_pypose, T12_manual, atol=1e-5), \
            f"SE3 composition mismatch. Max diff: {(T12_pypose - T12_manual).abs().max()}"

        print("SE3 composition verified")


class TestPoseEncodingConversion:
    """Test pose encoding to SE3 conversion matches VGGT's official conversion."""

    def test_pose_encoding_to_se3_matches_vggt(self):
        """Verify pose_encoding_to_se3 matches pose_encoding_to_extri_intri.

        Uses synthetic pose_enc that mimics real model output.
        """
        torch.manual_seed(42)
        B, S = 2, 4
        image_hw = (256, 256)

        # Create synthetic pose_enc: [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]
        t = torch.randn(B, S, 3) * 0.5
        q = F.normalize(torch.randn(B, S, 4), dim=-1)
        # Force qw >= 0 (hemisphere canonicalization)
        q = torch.where(q[..., 3:4] < 0, -q, q)
        fov = torch.ones(B, S, 2) * 1.0  # ~57 degrees

        pose_enc = torch.cat([t, q, fov], dim=-1)  # [B, S, 9]

        # Method 1: VGGT's official conversion
        extri_vggt, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_hw)
        # extri_vggt: [B, S, 3, 4] - world-to-camera

        # Method 2: Our PyPose conversion
        T = pose_encoding_to_se3(pose_enc)
        T_matrix = T.matrix()  # [B, S, 4, 4]

        # Extract R, t from both
        R_vggt = extri_vggt[..., :3, :3]  # [B, S, 3, 3]
        t_vggt = extri_vggt[..., :3, 3]   # [B, S, 3]

        R_pypose = T_matrix[..., :3, :3]
        t_pypose = T_matrix[..., :3, 3]

        # Compare R and t separately
        assert torch.allclose(R_vggt, R_pypose, atol=1e-5), \
            f"Rotation mismatch! Max diff: {(R_vggt - R_pypose).abs().max()}"
        assert torch.allclose(t_vggt, t_pypose, atol=1e-5), \
            f"Translation mismatch! Max diff: {(t_vggt - t_pypose).abs().max()}"

        # Cross-check with camera center formula
        # If w2c: C = -R^T @ t
        C_vggt = -torch.einsum('...ij,...j->...i', R_vggt.transpose(-1, -2), t_vggt)
        C_pypose = -torch.einsum('...ij,...j->...i', R_pypose.transpose(-1, -2), t_pypose)
        assert torch.allclose(C_vggt, C_pypose, atol=1e-5), \
            "Camera center mismatch!"

        print("Convention verified: R, t, and camera center all match")


class TestWindowRelativePoses:
    """Test window-relative pose computation."""

    def test_frame0_is_identity(self):
        """Verify T_rel[0] = Identity for all batches."""
        torch.manual_seed(42)
        B, S = 3, 5

        # Create synthetic pose_enc
        t = torch.randn(B, S, 3)
        q = F.normalize(torch.randn(B, S, 4), dim=-1)
        q = torch.where(q[..., 3:4] < 0, -q, q)
        fov = torch.ones(B, S, 2)
        pose_enc = torch.cat([t, q, fov], dim=-1)

        T_rel = compute_window_relative_poses(pose_enc)

        # Frame 0 should be identity
        T_rel_0 = T_rel[:, 0]  # [B]
        identity = pp.identity_SE3(B, device=pose_enc.device, dtype=pose_enc.dtype)

        assert torch.allclose(T_rel_0.matrix(), identity.matrix(), atol=1e-5), \
            f"T_rel[0] is not identity! Max diff: {(T_rel_0.matrix() - identity.matrix()).abs().max()}"

        print("Window-relative frame 0 is identity")

    def test_window_relative_consistency(self):
        """Verify T_rel_i = T_0^{-1} @ T_i manually."""
        torch.manual_seed(42)
        B, S = 2, 4

        t = torch.randn(B, S, 3)
        q = F.normalize(torch.randn(B, S, 4), dim=-1)
        q = torch.where(q[..., 3:4] < 0, -q, q)
        fov = torch.ones(B, S, 2)
        pose_enc = torch.cat([t, q, fov], dim=-1)

        T_rel = compute_window_relative_poses(pose_enc)

        # Manual computation
        T_abs = pose_encoding_to_se3(pose_enc)
        T_0 = T_abs[:, 0:1]
        T_rel_manual = T_0.Inv() @ T_abs

        assert torch.allclose(T_rel.matrix(), T_rel_manual.matrix(), atol=1e-5), \
            f"Window relative mismatch! Max diff: {(T_rel.matrix() - T_rel_manual.matrix()).abs().max()}"

        print("Window-relative computation verified")


class TestScaleFitting:
    """Test per-window scale fitting."""

    def test_scale_with_known_factor(self):
        """Verify scale fitting recovers known scale factor."""
        torch.manual_seed(42)
        B, S = 3, 6
        true_scale = torch.tensor([0.5, 2.0, 1.5])  # [B]

        # Create GT translations
        gt_t_rel = torch.randn(B, S, 3)
        gt_t_rel[:, 0] = 0  # Frame 0 is identity

        # Create pred translations as scaled GT + small noise
        pred_t_rel = gt_t_rel / true_scale[:, None, None]

        # Recover scale
        recovered_scale = compute_window_scale_batched(pred_t_rel, gt_t_rel, detach=False)

        assert torch.allclose(recovered_scale, true_scale, rtol=0.1), \
            f"Scale recovery failed! True: {true_scale}, Recovered: {recovered_scale}"

        print(f"Scale fitting verified. True: {true_scale.tolist()}, Recovered: {recovered_scale.tolist()}")

    def test_scale_detach_prevents_gradient(self):
        """Verify scale.detach() prevents gradient flow."""
        B, S = 2, 4

        gt_t_rel = torch.randn(B, S, 3)
        gt_t_rel[:, 0] = 0
        pred_t_rel = torch.randn(B, S, 3, requires_grad=True)
        pred_t_rel.data[:, 0] = 0

        # With detach=True
        scale_detached = compute_window_scale_batched(pred_t_rel, gt_t_rel, detach=True)
        assert not scale_detached.requires_grad, "Detached scale should not require grad"

        # With detach=False
        scale_attached = compute_window_scale_batched(pred_t_rel, gt_t_rel, detach=False)
        assert scale_attached.requires_grad, "Non-detached scale should require grad"

        print("Scale detach behavior verified")

    def test_scale_fallback_on_static_sequence(self):
        """Verify scale=1.0 fallback when sequence is static."""
        B, S = 2, 4

        # All translations are zero (static)
        gt_t_rel = torch.zeros(B, S, 3)
        pred_t_rel = torch.zeros(B, S, 3)

        scale = compute_window_scale_batched(pred_t_rel, gt_t_rel)

        assert torch.allclose(scale, torch.ones(B)), \
            f"Static sequence should have scale=1.0, got {scale}"

        print("Scale fallback for static sequence verified")


class TestSE3Residual:
    """Test SE(3) residual computation."""

    def test_residual_is_zero_for_matching_poses(self):
        """Verify residual is zero when pred == gt."""
        torch.manual_seed(42)
        B, S = 2, 4

        t = torch.randn(B, S, 3)
        q = F.normalize(torch.randn(B, S, 4), dim=-1)
        q = torch.where(q[..., 3:4] < 0, -q, q)
        fov = torch.ones(B, S, 2)
        pose_enc = torch.cat([t, q, fov], dim=-1)

        T_rel = compute_window_relative_poses(pose_enc)

        residual = compute_se3_residual(T_rel, T_rel)

        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5), \
            f"Residual should be zero for matching poses! Max: {residual.abs().max()}"

        print("Residual for matching poses is zero")

    def test_residual_dimension_convention(self):
        """Verify residual dimension convention: v=trans, w=rot (PyPose order)."""
        torch.manual_seed(42)

        # Create two SE3 that differ only in rotation
        # Use batched format [B, S, 7]
        B, S = 1, 1
        t = torch.zeros(B, S, 3)
        q_gt = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])  # [B, S, 4] identity rotation
        q_pred = F.normalize(torch.tensor([[[0.3, 0.0, 0.0, 1.0]]]), dim=-1)  # larger x rotation

        # Create SE3 directly with [B, S, 7] format
        T_gt = pp.SE3(torch.cat([t, q_gt], dim=-1))  # [B, S, 7] -> SE3 of shape [B, S]
        T_pred = pp.SE3(torch.cat([t, q_pred], dim=-1))

        residual = compute_se3_residual(T_pred, T_gt)  # [B, S, 6]

        # Use split_se3_tangent for consistency
        v, w = split_se3_tangent(residual)
        trans_residual = v.abs().max()
        rot_residual = w.abs().max()

        assert rot_residual > 0.01, f"Rotation residual should be non-zero: {rot_residual}"
        assert trans_residual < 1e-5, f"Translation residual should be zero: {trans_residual}"

        print(f"Dimension convention verified: trans_residual={trans_residual:.6f}, rot_residual={rot_residual:.4f}")


class TestReconstructScaledSE3:
    """Test SE3 reconstruction with scaled translation."""

    def test_reconstruction_preserves_rotation(self):
        """Verify rotation is preserved after scaling translation."""
        torch.manual_seed(42)
        B, S = 2, 4

        t = torch.randn(B, S, 3)
        q = F.normalize(torch.randn(B, S, 4), dim=-1)
        q = torch.where(q[..., 3:4] < 0, -q, q)

        T = pp.SE3(torch.cat([t, q], dim=-1))
        scale = torch.tensor([2.0, 0.5])

        T_scaled = reconstruct_scaled_se3(T, scale)

        # Rotation should be the same
        R_orig = T.rotation().tensor()
        R_scaled = T_scaled.rotation().tensor()

        # Handle sign ambiguity
        for b in range(B):
            for s in range(S):
                q_orig = R_orig[b, s]
                q_scaled = R_scaled[b, s]
                # Check if they're the same or opposite
                same = torch.allclose(q_orig, q_scaled, atol=1e-5)
                opposite = torch.allclose(q_orig, -q_scaled, atol=1e-5)
                assert same or opposite, f"Rotation changed at [{b}, {s}]"

        print("Reconstruction preserves rotation")

    def test_reconstruction_scales_translation(self):
        """Verify translation is correctly scaled."""
        torch.manual_seed(42)
        B, S = 2, 4

        t = torch.randn(B, S, 3)
        q = F.normalize(torch.randn(B, S, 4), dim=-1)
        q = torch.where(q[..., 3:4] < 0, -q, q)

        T = pp.SE3(torch.cat([t, q], dim=-1))
        scale = torch.tensor([2.0, 0.5])

        T_scaled = reconstruct_scaled_se3(T, scale)

        t_orig = T.translation()
        t_scaled = T_scaled.translation()

        expected_t = scale[:, None, None] * t_orig

        assert torch.allclose(t_scaled, expected_t, atol=1e-5), \
            f"Translation scaling failed! Max diff: {(t_scaled - expected_t).abs().max()}"

        print("Reconstruction correctly scales translation")


class TestResidualSanityCheck:
    """Sanity checks for residual magnitudes."""

    def test_residual_magnitude_reasonable(self):
        """Check residual magnitudes are in reasonable range for small perturbations."""
        torch.manual_seed(42)
        B, S = 5, 10

        # Create GT poses
        t_gt = torch.randn(B, S, 3) * 0.5  # small translations
        q_gt = F.normalize(torch.randn(B, S, 4), dim=-1)
        q_gt = torch.where(q_gt[..., 3:4] < 0, -q_gt, q_gt)
        fov = torch.ones(B, S, 2)
        gt_pose_enc = torch.cat([t_gt, q_gt, fov], dim=-1)

        # Create pred poses with small perturbations
        t_pred = t_gt + torch.randn(B, S, 3) * 0.1  # ~10cm noise
        q_noise = F.normalize(torch.randn(B, S, 4) * 0.1 + q_gt, dim=-1)
        q_pred = torch.where(q_noise[..., 3:4] < 0, -q_noise, q_noise)
        pred_pose_enc = torch.cat([t_pred, q_pred, fov], dim=-1)

        # Compute window-relative
        T_rel_gt = compute_window_relative_poses(gt_pose_enc)
        T_rel_pred = compute_window_relative_poses(pred_pose_enc)

        # Scale fitting
        gt_t_rel = T_rel_gt.translation()
        pred_t_rel = T_rel_pred.translation()
        scale = compute_window_scale_batched(pred_t_rel, gt_t_rel)

        # Reconstruct scaled
        T_rel_pred_scaled = reconstruct_scaled_se3(T_rel_pred, scale)

        # Compute residual
        residual = compute_se3_residual(T_rel_pred_scaled, T_rel_gt)

        # Skip frame 0
        residual = residual[:, 1:]

        # Use split_se3_tangent for dimension convention
        v, w = split_se3_tangent(residual)
        trans_norm = v.norm(dim=-1)
        rot_norm = w.norm(dim=-1)

        print(f"Residual statistics:")
        print(f"  trans_norm: mean={trans_norm.mean():.3f} m, p90={trans_norm.quantile(0.9):.3f} m")
        print(f"  rot_norm: mean={rot_norm.mean():.3f} rad, p90={rot_norm.quantile(0.9):.3f} rad")

        # Sanity checks
        assert rot_norm.mean() < 1.0, f"Mean rotation residual too large: {rot_norm.mean():.3f} rad"
        assert trans_norm.mean() < 1.0, f"Mean translation residual too large: {trans_norm.mean():.3f} m"

        print("Residual magnitudes are reasonable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
