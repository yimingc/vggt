#!/usr/bin/env python3
"""
Phase 4: Post-Training Uncertainty Calibration Evaluation.

Evaluates uncertainty calibration on TUM RGB-D sequences.

A well-calibrated uncertainty should satisfy:
- d² = Σ λ_k r_k² follows χ²(n) distribution
- d²_rot follows χ²(3), d²_trans follows χ²(3)
- Mean d² ≈ 3 for rot/trans separately

CAVEAT: Residuals are NOT i.i.d. (sliding windows, temporal correlation).
KS p-values will often be overly pessimistic. Treat KS as QUALITATIVE,
not pass/fail. The mean and p95 matching χ² is more important.

Usage:
    python training/tests/eval_uncertainty_calibration.py \
        --checkpoint ./checkpoints/best.pt \
        --tum_dir /path/to/tum_rgbd \
        --output_dir ./eval_output
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockCommonConf:
    """Mock configuration for dataset."""
    def __init__(self):
        self.img_size = 518
        self.patch_size = 14
        self.debug = False
        self.training = False  # Evaluation mode
        self.get_nearby = True
        self.load_depth = True
        self.inside_random = False
        self.allow_duplicate_img = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.augs = type('obj', (object,), {'scales': None})()


def load_checkpoint(model, checkpoint_path):
    """Load uncertainty head weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_state_dict = model.state_dict()
    for name, param in checkpoint['uncertainty_head_state_dict'].items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model")

    print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
    print(f"  Training d²_rot: {checkpoint['d2_rot_mean']:.2f}")
    print(f"  Training d²_trans: {checkpoint['d2_trans_mean']:.2f}")

    return checkpoint


def compute_residuals_and_uncertainty(model, dataset, device, dtype, num_windows=100, num_frames=8):
    """
    Run model on dataset and collect residuals and uncertainties.

    Returns:
        dict with keys:
            - residuals: [N, 6] array (trans first: vx,vy,vz, rot: wx,wy,wz)
            - log_var: [N, 6] array
            - d2_rot: [N] array
            - d2_trans: [N] array
    """
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding
    from vggt.utils.lie_algebra import (
        pose_encoding_to_se3,
        extract_window_relative_poses,
        compute_window_scale_batched,
        compute_se3_residual,
        reconstruct_scaled_se3,
        split_se3_tangent,
        extract_relative_camera_positions,
    )

    all_residuals = []
    all_log_var = []
    all_d2_rot = []
    all_d2_trans = []

    model.eval()

    with torch.no_grad():
        for i in range(num_windows):
            seq_idx = i % len(dataset.sequence_list)
            batch = dataset.get_data(seq_index=seq_idx, img_per_seq=num_frames, ids=None, aspect_ratio=1.0)

            # Prepare images
            images = np.stack(batch['images'], axis=0)
            images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            images_tensor = torch.from_numpy(images).to(device).to(dtype).unsqueeze(0)

            # Prepare GT
            gt_extrinsics = torch.from_numpy(np.stack(batch['extrinsics'], axis=0)).to(device).float().unsqueeze(0)
            gt_intrinsics = torch.from_numpy(np.stack(batch['intrinsics'], axis=0)).to(device).float().unsqueeze(0)

            # Forward pass
            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = model(images_tensor)

            # Get predictions (last stage)
            pred_pose_enc = predictions['pose_enc_list'][-1]  # [B, S, 9]
            log_var_raw = predictions['pose_log_var_list'][-1]  # [B, S, 6]

            # Convert GT to pose encoding
            image_hw = images_tensor.shape[-2:]
            gt_pose_enc = extri_intri_to_pose_encoding(
                gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type="absT_quaR_FoV"
            )

            # Compute window-relative poses
            T_cam_world_gt = pose_encoding_to_se3(gt_pose_enc)
            T_cam0_cami_gt = extract_window_relative_poses(T_cam_world_gt)
            gt_cam_pos_rel = extract_relative_camera_positions(T_cam_world_gt)

            T_cam_world_pred = pose_encoding_to_se3(pred_pose_enc)
            T_cam0_cami_pred = extract_window_relative_poses(T_cam_world_pred)
            pred_cam_pos_rel = extract_relative_camera_positions(T_cam_world_pred)

            # Fit scale
            scale = compute_window_scale_batched(
                pred_cam_pos_rel, gt_cam_pos_rel, detach=True, min_translation=0.02
            )

            # Apply scale and compute residual
            T_cam0_cami_pred_scaled = reconstruct_scaled_se3(T_cam0_cami_pred, scale)
            residual = compute_se3_residual(T_cam0_cami_pred_scaled, T_cam0_cami_gt)  # [B, S, 6]

            # Skip frame 0
            residual = residual[:, 1:]  # [B, S-1, 6]
            log_var = log_var_raw[:, 1:]  # [B, S-1, 6]

            # Clamp log_var
            log_var = log_var.clamp(min=-20.0, max=20.0)

            # Compute d² = r² * exp(-log_var)
            residual_sq = residual ** 2
            rsq_trans, rsq_rot = split_se3_tangent(residual_sq)
            lv_trans, lv_rot = split_se3_tangent(log_var)

            d2_rot = (rsq_rot * torch.exp(-lv_rot)).sum(dim=-1)  # [B, S-1]
            d2_trans = (rsq_trans * torch.exp(-lv_trans)).sum(dim=-1)  # [B, S-1]

            # Collect data
            all_residuals.append(residual.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())
            all_d2_rot.append(d2_rot.cpu().numpy().flatten())
            all_d2_trans.append(d2_trans.cpu().numpy().flatten())

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{num_windows} windows")

    return {
        'residuals': np.concatenate(all_residuals, axis=1).reshape(-1, 6),
        'log_var': np.concatenate(all_log_var, axis=1).reshape(-1, 6),
        'd2_rot': np.concatenate(all_d2_rot),
        'd2_trans': np.concatenate(all_d2_trans),
    }


def compute_baseline_comparison(data):
    """
    Compare trained NLL vs baselines:
    1. Weak baseline: σ=1 (arbitrary, for sanity check only)
    2. Strong baseline: Homoscedastic MLE (best constant σ per dimension)

    The MLE baseline is the fair comparison - it represents the best possible
    constant uncertainty. If trained NLL < MLE baseline NLL, the heteroscedastic
    model has learned useful per-sample uncertainty.
    """
    if data['residuals'] is None:
        print("\nBaseline comparison: residuals not available")
        return {}

    residuals = data['residuals']  # [N, 6]
    log_var_trained = data['log_var']  # [N, 6]

    residual_sq = residuals ** 2

    # Trained model NLL: 0.5 * (r² * exp(-log_var) + log_var)
    nll_trained = 0.5 * (residual_sq * np.exp(-log_var_trained) + log_var_trained)

    # ===== Baseline 0: σ=1 (weak, for sanity only) =====
    # NLL = 0.5 * (r² * 1 + 0) = 0.5 * r²
    nll_baseline_unit = 0.5 * residual_sq

    # ===== Baseline 1: Homoscedastic MLE (strong, fair comparison) =====
    # MLE for constant σ: σ_k² = E[r_k²]
    # This is the BEST possible constant σ for this data
    sigma_sq_mle = residual_sq.mean(axis=0)  # [6] - one per dimension
    log_var_mle = np.log(sigma_sq_mle + 1e-12)  # [6]
    sigma_mle = np.sqrt(sigma_sq_mle)

    # NLL with MLE constant: 0.5 * (r² / σ² + log(σ²)) = 0.5 * (r²/σ² + log_var)
    # Since σ² = E[r²], we have E[r²/σ²] = 1, so E[NLL] ≈ 0.5 * (1 + log_var)
    nll_baseline_mle = 0.5 * (residual_sq / sigma_sq_mle + log_var_mle)

    # d² values
    d2_trained = (residual_sq * np.exp(-log_var_trained)).sum(axis=-1)
    d2_mle = (residual_sq / sigma_sq_mle).sum(axis=-1)  # Should be ~6 by construction

    print(f"\n{'='*60}")
    print("BASELINE COMPARISON")
    print(f"{'='*60}")

    print(f"\n1. Trained Model (heteroscedastic):")
    print(f"   NLL mean: {nll_trained.mean():.4f}")
    print(f"   d² mean:  {d2_trained.mean():.2f} (target: 6)")

    print(f"\n2. Homoscedastic MLE Baseline (best constant σ):")
    print(f"   σ_mle per dim: [{', '.join([f'{s:.4f}' for s in sigma_mle])}]")
    print(f"   log_var_mle:   [{', '.join([f'{lv:.2f}' for lv in log_var_mle])}]")
    print(f"   NLL mean: {nll_baseline_mle.mean():.4f}")
    print(f"   d² mean:  {d2_mle.mean():.2f} (should be ~6 by MLE construction)")

    print(f"\n3. Unit Baseline (σ=1, sanity check only):")
    print(f"   NLL mean: {nll_baseline_unit.mean():.4f}")

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    nll_improvement_vs_mle = nll_baseline_mle.mean() - nll_trained.mean()
    print(f"\n  Trained vs MLE baseline:")
    print(f"    NLL improvement: {nll_improvement_vs_mle:.4f}")
    if nll_improvement_vs_mle > 0:
        print(f"    ✓ Heteroscedastic model is BETTER than best constant σ")
        print(f"      (learned useful per-sample uncertainty)")
    else:
        print(f"    ✗ Constant σ is as good or better")
        print(f"      (heteroscedastic model may not be learning useful signal)")

    return {
        'nll_trained': nll_trained.mean(),
        'nll_baseline_mle': nll_baseline_mle.mean(),
        'nll_baseline_unit': nll_baseline_unit.mean(),
        'd2_trained': d2_trained.mean(),
        'd2_mle': d2_mle.mean(),
        'sigma_mle': sigma_mle,
        'log_var_mle': log_var_mle,
        'nll_improvement_vs_mle': nll_improvement_vs_mle,
    }


def evaluate_calibration(data):
    """Compute calibration statistics."""
    d2_rot = data['d2_rot']
    d2_trans = data['d2_trans']
    d2_total = d2_rot + d2_trans

    # Chi-square reference values
    chi2_3 = stats.chi2(df=3)
    chi2_6 = stats.chi2(df=6)

    print(f"\n{'='*60}")
    print("CALIBRATION STATISTICS")
    print(f"{'='*60}")
    print(f"\n  d²_rot:   mean={d2_rot.mean():.2f} (expect ~3), p95={np.percentile(d2_rot, 95):.2f}")
    print(f"  d²_trans: mean={d2_trans.mean():.2f} (expect ~3), p95={np.percentile(d2_trans, 95):.2f}")
    print(f"  d²_total: mean={d2_total.mean():.2f} (expect ~6), p95={np.percentile(d2_total, 95):.2f}")

    print(f"\n  Reference χ²(3): mean=3.0, p95={chi2_3.ppf(0.95):.2f}")
    print(f"  Reference χ²(6): mean=6.0, p95={chi2_6.ppf(0.95):.2f}")

    # Kolmogorov-Smirnov test
    ks_rot = stats.kstest(d2_rot, 'chi2', args=(3,))
    ks_trans = stats.kstest(d2_trans, 'chi2', args=(3,))
    ks_total = stats.kstest(d2_total, 'chi2', args=(6,))

    print(f"\n  KS test (QUALITATIVE - samples are correlated, p-values pessimistic):")
    print(f"    d²_rot vs χ²(3):   statistic={ks_rot.statistic:.3f}, p={ks_rot.pvalue:.4f}")
    print(f"    d²_trans vs χ²(3): statistic={ks_trans.statistic:.3f}, p={ks_trans.pvalue:.4f}")
    print(f"    d²_total vs χ²(6): statistic={ks_total.statistic:.3f}, p={ks_total.pvalue:.4f}")
    print(f"\n  NOTE: Mean and p95 matching reference is MORE important than p-value.")

    # Calibration assessment
    rot_calibrated = 1.5 < d2_rot.mean() < 6.0
    trans_calibrated = 1.5 < d2_trans.mean() < 6.0

    print(f"\n  Assessment:")
    print(f"    Rotation:    {'✓ Reasonably calibrated' if rot_calibrated else '✗ Miscalibrated'} (d²={d2_rot.mean():.2f})")
    print(f"    Translation: {'✓ Reasonably calibrated' if trans_calibrated else '✗ Miscalibrated'} (d²={d2_trans.mean():.2f})")

    return {
        'd2_rot_mean': d2_rot.mean(),
        'd2_trans_mean': d2_trans.mean(),
        'd2_total_mean': d2_total.mean(),
        'd2_rot_p95': np.percentile(d2_rot, 95),
        'd2_trans_p95': np.percentile(d2_trans, 95),
        'ks_rot_pvalue': ks_rot.pvalue,
        'ks_trans_pvalue': ks_trans.pvalue,
    }


def plot_uncertainty_vs_error(data, output_dir):
    """Plot predicted sigma vs actual |residual|."""
    import matplotlib.pyplot as plt

    if data['residuals'] is None:
        print("Cannot plot: residuals not available")
        return

    residuals = data['residuals']  # [N, 6]
    log_var = data['log_var']  # [N, 6]

    # sigma = exp(0.5 * log_var)
    sigma = np.exp(0.5 * log_var)
    actual_error = np.abs(residuals)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # PyPose se(3) Log returns [vx, vy, vz, wx, wy, wz] - trans first, rot second
    labels = ['vx (trans)', 'vy (trans)', 'vz (trans)', 'wx (rot)', 'wy (rot)', 'wz (rot)']

    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        # Subsample for visualization
        n_points = min(5000, len(sigma))
        idx = np.random.choice(len(sigma), n_points, replace=False)

        ax.scatter(sigma[idx, i], actual_error[idx, i], alpha=0.3, s=1)
        max_val = max(np.percentile(sigma[:, i], 99), np.percentile(actual_error[:, i], 99))
        ax.plot([0, max_val], [0, max_val], 'r--', label='y=x (ideal)', linewidth=2)
        ax.set_xlabel(f'Predicted σ')
        ax.set_ylabel(f'Actual |r|')
        ax.set_title(f'{label}: corr={np.corrcoef(sigma[:, i], actual_error[:, i])[0,1]:.3f}')
        ax.legend()
        ax.set_xlim(0, max_val * 1.1)
        ax.set_ylim(0, max_val * 1.1)

    plt.suptitle('Uncertainty vs Error Correlation', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'uncertainty_vs_error.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved uncertainty vs error scatter plot to {output_path}")


def compute_coverage_statistics(data):
    """
    Compute coverage statistics for calibration assessment.

    For well-calibrated Gaussian uncertainty:
    - z = r / σ should be ~ N(0, 1)
    - P(|z| < 1) ≈ 0.6827 (1-sigma)
    - P(|z| < 2) ≈ 0.9545 (2-sigma)
    - P(|z| < 3) ≈ 0.9973 (3-sigma)

    Returns dict with coverage statistics.
    """
    if data['residuals'] is None:
        print("\nCoverage statistics: residuals not available")
        return {}

    residuals = data['residuals']  # [N, 6]
    log_var = data['log_var']  # [N, 6]

    sigma = np.exp(0.5 * log_var)
    z = residuals / sigma  # Normalized residuals (should be ~N(0,1))

    # Expected coverage for standard normal
    expected_coverage = {
        1: 0.6827,  # P(|z| < 1)
        2: 0.9545,  # P(|z| < 2)
        3: 0.9973,  # P(|z| < 3)
    }

    print(f"\n{'='*60}")
    print("COVERAGE STATISTICS (Calibration Test)")
    print(f"{'='*60}")
    print("\nFor well-calibrated N(0,σ²), normalized residual z=r/σ ~ N(0,1)")
    print("\n  Expected coverage:")
    for k, v in expected_coverage.items():
        print(f"    P(|z| < {k}) = {v:.4f}")

    # Compute per-dimension coverage
    labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    coverage_results = {}

    print("\n  Empirical coverage per dimension:")
    print(f"    {'Dim':<6} {'P(|z|<1)':<12} {'P(|z|<2)':<12} {'P(|z|<3)':<12}")
    print(f"    {'-'*6} {'-'*12} {'-'*12} {'-'*12}")

    for i, label in enumerate(labels):
        z_i = z[:, i]
        cov_1 = (np.abs(z_i) < 1).mean()
        cov_2 = (np.abs(z_i) < 2).mean()
        cov_3 = (np.abs(z_i) < 3).mean()
        coverage_results[label] = {'1sigma': cov_1, '2sigma': cov_2, '3sigma': cov_3}
        print(f"    {label:<6} {cov_1:<12.4f} {cov_2:<12.4f} {cov_3:<12.4f}")

    # Aggregate (all dimensions pooled)
    z_all = z.flatten()
    cov_1_all = (np.abs(z_all) < 1).mean()
    cov_2_all = (np.abs(z_all) < 2).mean()
    cov_3_all = (np.abs(z_all) < 3).mean()

    print(f"    {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
    print(f"    {'ALL':<6} {cov_1_all:<12.4f} {cov_2_all:<12.4f} {cov_3_all:<12.4f}")
    print(f"    {'Target':<6} {0.6827:<12.4f} {0.9545:<12.4f} {0.9973:<12.4f}")

    # Calibration assessment
    print(f"\n  Calibration Assessment:")
    if abs(cov_1_all - 0.6827) < 0.05 and abs(cov_2_all - 0.9545) < 0.03:
        print(f"    ✓ WELL CALIBRATED (coverage close to expected)")
    elif cov_1_all < 0.6827:
        print(f"    ~ OVERCONFIDENT (coverage < expected, σ too small)")
        print(f"      1σ coverage: {cov_1_all:.3f} < 0.683 expected")
    else:
        print(f"    ~ UNDERCONFIDENT (coverage > expected, σ too large)")
        print(f"      1σ coverage: {cov_1_all:.3f} > 0.683 expected")

    return {
        'coverage_1sigma': cov_1_all,
        'coverage_2sigma': cov_2_all,
        'coverage_3sigma': cov_3_all,
        'per_dim': coverage_results,
        'z': z,  # Store for plotting
    }


def plot_coverage_reliability(data, coverage_data, output_dir):
    """
    Plot quantile-coverage reliability diagram.

    For d² ~ χ²(df), we check: empirical P(d² ≤ t_p) vs theoretical p
    where t_p = χ²_df^{-1}(p) is the p-th quantile.

    This is more statistically rigorous than σ vs |r| binned plots.
    """
    import matplotlib.pyplot as plt

    d2_rot = data['d2_rot']
    d2_trans = data['d2_trans']
    d2_total = d2_rot + d2_trans

    # Quantiles to check
    p_values = np.linspace(0.1, 0.9, 9)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, d2, df, title in [
        (axes[0], d2_rot, 3, 'd²_rot vs χ²(3)'),
        (axes[1], d2_trans, 3, 'd²_trans vs χ²(3)'),
        (axes[2], d2_total, 6, 'd²_total vs χ²(6)'),
    ]:
        # Theoretical quantiles
        t_p = stats.chi2.ppf(p_values, df=df)

        # Empirical coverage at each quantile
        empirical_p = np.array([(d2 <= t).mean() for t in t_p])

        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal (y=x)')
        ax.plot(p_values, empirical_p, 'bo-', markersize=8, linewidth=2, label='Empirical')

        # Fill region for ±5% tolerance
        ax.fill_between([0, 1], [0-0.05, 1-0.05], [0+0.05, 1+0.05],
                        alpha=0.2, color='green', label='±5% tolerance')

        ax.set_xlabel('Theoretical quantile p')
        ax.set_ylabel('Empirical P(d² ≤ χ²_p)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Quantile-Coverage Reliability (d² vs χ² distribution)', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'coverage_reliability.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved coverage reliability diagram to {output_path}")


def plot_normalized_residual_histogram(coverage_data, output_dir):
    """
    Plot histogram of normalized residuals z = r/σ vs N(0,1).
    """
    import matplotlib.pyplot as plt

    z = coverage_data.get('z')
    if z is None:
        print("Cannot plot: normalized residuals not available")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    labels = ['vx (trans)', 'vy (trans)', 'vz (trans)', 'wx (rot)', 'wy (rot)', 'wz (rot)']

    x = np.linspace(-4, 4, 100)
    standard_normal = stats.norm.pdf(x)

    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        z_i = z[:, i]

        # Clip for visualization (rare outliers)
        z_clipped = np.clip(z_i, -5, 5)

        ax.hist(z_clipped, bins=50, density=True, alpha=0.7,
                label=f'z (μ={z_i.mean():.2f}, σ={z_i.std():.2f})')
        ax.plot(x, standard_normal, 'r-', linewidth=2, label='N(0,1)')
        ax.set_xlabel('z = r / σ')
        ax.set_ylabel('Density')
        ax.set_title(f'{label}')
        ax.legend()
        ax.set_xlim(-4, 4)

    plt.suptitle('Normalized Residuals z=r/σ vs Standard Normal', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'normalized_residual_histogram.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved normalized residual histogram to {output_path}")


def plot_reliability_diagram(data, output_dir, n_bins=10):
    """
    Reliability diagram: bin by predicted σ, plot mean |residual| per bin.

    NOTE: For Gaussian r ~ N(0, σ²), E[|r|] = σ * sqrt(2/π) ≈ 0.798σ
    So the "ideal" line is y = 0.798x, NOT y = x.
    """
    import matplotlib.pyplot as plt

    if data['residuals'] is None:
        print("Cannot plot: residuals not available")
        return

    residuals = data['residuals']
    log_var = data['log_var']

    sigma = np.exp(0.5 * log_var)
    actual_error = np.abs(residuals)

    # E[|r|] = σ * sqrt(2/π) for Gaussian
    GAUSSIAN_FACTOR = np.sqrt(2 / np.pi)  # ≈ 0.798

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    labels = ['vx (trans)', 'vy (trans)', 'vz (trans)', 'wx (rot)', 'wy (rot)', 'wz (rot)']

    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        sigma_i = sigma[:, i]
        error_i = actual_error[:, i]

        # Use quantile bins for balanced counts
        bin_edges = np.percentile(sigma_i, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        mean_errors = []
        std_errors = []

        for j in range(n_bins):
            mask = (sigma_i >= bin_edges[j]) & (sigma_i < bin_edges[j+1])
            if mask.sum() > 10:
                bin_centers.append(sigma_i[mask].mean())
                mean_errors.append(error_i[mask].mean())
                std_errors.append(error_i[mask].std() / np.sqrt(mask.sum()))

        if bin_centers:
            bin_centers = np.array(bin_centers)
            mean_errors = np.array(mean_errors)
            std_errors = np.array(std_errors)

            ax.errorbar(bin_centers, mean_errors, yerr=std_errors, fmt='o-', capsize=3, markersize=6)
            max_val = max(bin_centers.max(), mean_errors.max()) * 1.1

            # Plot CORRECT ideal line: y = 0.798 * x (not y = x)
            x_line = np.linspace(0, max_val, 100)
            ax.plot(x_line, GAUSSIAN_FACTOR * x_line, 'r--',
                    label=f'y={GAUSSIAN_FACTOR:.2f}x (Gaussian ideal)', linewidth=2)
            ax.plot(x_line, x_line, 'k:', alpha=0.5, label='y=x (reference)', linewidth=1)
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)

        ax.set_xlabel(f'Predicted σ (binned)')
        ax.set_ylabel(f'Mean |r|')
        ax.set_title(f'{label}')
        ax.legend(fontsize=8)

    plt.suptitle('Reliability Diagram\n(For Gaussian: E[|r|] = 0.798σ, not σ)', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'reliability_diagram.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved reliability diagram to {output_path}")


def compute_whitened_covariance(data):
    """
    Compute covariance matrix of whitened (normalized) residuals.

    If calibrated AND diagonal assumption is correct:
    - Cov(z) should be close to identity matrix
    - Diagonal elements ≈ 1
    - Off-diagonal elements ≈ 0

    Large off-diagonal elements indicate coupling between dimensions
    that the diagonal uncertainty cannot capture (motivates full covariance).

    Returns dict with covariance matrix and diagnostics.
    """
    if data['residuals'] is None:
        print("\nWhitened covariance: residuals not available")
        return {}

    residuals = data['residuals']  # [N, 6]
    log_var = data['log_var']  # [N, 6]

    sigma = np.exp(0.5 * log_var)
    z = residuals / sigma  # Whitened residuals

    # Compute empirical covariance
    cov_z = np.cov(z, rowvar=False)  # [6, 6]

    # Extract diagnostics
    diag = np.diag(cov_z)
    off_diag = cov_z - np.diag(diag)
    max_off_diag = np.abs(off_diag).max()
    mean_off_diag = np.abs(off_diag[np.triu_indices(6, k=1)]).mean()

    labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']

    print(f"\n{'='*60}")
    print("WHITENED RESIDUAL COVARIANCE (Diagonal Assumption Check)")
    print(f"{'='*60}")
    print("\nIf calibrated & diagonal assumption valid: Cov(z) ≈ Identity")
    print("\nEmpirical Cov(z) diagonal elements (expect ~1.0):")
    for i, label in enumerate(labels):
        status = "✓" if 0.7 < diag[i] < 1.3 else "~" if 0.5 < diag[i] < 1.5 else "✗"
        print(f"  {label}: {diag[i]:.3f} {status}")

    print(f"\nOff-diagonal statistics:")
    print(f"  Max |off-diag|:  {max_off_diag:.3f} (ideal: ~0)")
    print(f"  Mean |off-diag|: {mean_off_diag:.3f} (ideal: ~0)")

    # Check for significant coupling
    print(f"\nDiagonal assumption assessment:")
    if max_off_diag < 0.2:
        print(f"  ✓ Diagonal assumption reasonable (max off-diag < 0.2)")
    elif max_off_diag < 0.4:
        print(f"  ~ Mild coupling detected (consider full covariance in future)")
        # Find largest coupling
        idx = np.unravel_index(np.abs(off_diag).argmax(), off_diag.shape)
        print(f"    Largest coupling: {labels[idx[0]]}-{labels[idx[1]]} = {cov_z[idx]:.3f}")
    else:
        print(f"  ✗ Significant coupling (diagonal assumption limiting)")
        idx = np.unravel_index(np.abs(off_diag).argmax(), off_diag.shape)
        print(f"    Largest coupling: {labels[idx[0]]}-{labels[idx[1]]} = {cov_z[idx]:.3f}")
        print(f"    Recommendation: Consider full 6x6 covariance prediction")

    return {
        'cov_z': cov_z,
        'diag': diag,
        'max_off_diag': max_off_diag,
        'mean_off_diag': mean_off_diag,
        'labels': labels,
    }


def plot_whitened_covariance(cov_data, output_dir):
    """Plot heatmap of whitened residual covariance matrix."""
    import matplotlib.pyplot as plt

    cov_z = cov_data.get('cov_z')
    if cov_z is None:
        print("Cannot plot: covariance not available")
        return

    labels = cov_data['labels']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Full covariance matrix
    ax = axes[0]
    im = ax.imshow(cov_z, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title('Cov(z) - Whitened Residual Covariance\n(Ideal: Identity matrix)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Add text annotations
    for i in range(6):
        for j in range(6):
            color = 'white' if abs(cov_z[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{cov_z[i, j]:.2f}', ha='center', va='center', color=color, fontsize=9)

    # Plot 2: Deviation from identity
    ax = axes[1]
    deviation = cov_z - np.eye(6)
    im = ax.imshow(deviation, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title('Cov(z) - Identity\n(Ideal: all zeros)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Add text annotations
    for i in range(6):
        for j in range(6):
            color = 'white' if abs(deviation[i, j]) > 0.25 else 'black'
            ax.text(j, i, f'{deviation[i, j]:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.suptitle('Diagonal Assumption Diagnostic: Is Cov(z) ≈ Identity?', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'whitened_covariance.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved whitened covariance plot to {output_path}")


def plot_d2_histogram(data, output_dir):
    """Plot histogram of d² values vs χ² distribution."""
    import matplotlib.pyplot as plt

    d2_rot = data['d2_rot']
    d2_trans = data['d2_trans']
    d2_total = d2_rot + d2_trans

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # d²_rot vs χ²(3)
    ax = axes[0]
    ax.hist(d2_rot, bins=50, density=True, alpha=0.7, label=f'd²_rot (mean={d2_rot.mean():.2f})')
    x = np.linspace(0, 15, 100)
    ax.plot(x, stats.chi2.pdf(x, df=3), 'r-', linewidth=2, label='χ²(3)')
    ax.set_xlabel('d²_rot')
    ax.set_ylabel('Density')
    ax.set_title('Rotation Calibration')
    ax.legend()
    ax.set_xlim(0, 15)

    # d²_trans vs χ²(3)
    ax = axes[1]
    ax.hist(d2_trans, bins=50, density=True, alpha=0.7, label=f'd²_trans (mean={d2_trans.mean():.2f})')
    ax.plot(x, stats.chi2.pdf(x, df=3), 'r-', linewidth=2, label='χ²(3)')
    ax.set_xlabel('d²_trans')
    ax.set_ylabel('Density')
    ax.set_title('Translation Calibration')
    ax.legend()
    ax.set_xlim(0, 15)

    # d²_total vs χ²(6)
    ax = axes[2]
    ax.hist(d2_total, bins=50, density=True, alpha=0.7, label=f'd²_total (mean={d2_total.mean():.2f})')
    x = np.linspace(0, 25, 100)
    ax.plot(x, stats.chi2.pdf(x, df=6), 'r-', linewidth=2, label='χ²(6)')
    ax.set_xlabel('d²_total')
    ax.set_ylabel('Density')
    ax.set_title('Total Calibration')
    ax.legend()
    ax.set_xlim(0, 25)

    plt.suptitle('d² Distribution vs χ² Reference', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'd2_histogram.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved d² histogram to {output_path}")


def run_evaluation(args):
    """Run full calibration evaluation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"\n{'='*60}")
    print("PHASE 4: UNCERTAINTY CALIBRATION EVALUATION")
    print(f"{'='*60}")
    print(f"\nDevice: {device}, dtype: {dtype}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"TUM directory: {args.tum_dir}")
    print(f"Number of windows: {args.num_windows}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("\nLoading TUM dataset...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.datasets.tum_rgbd import TUMRGBDDataset

    common_conf = MockCommonConf()
    dataset = TUMRGBDDataset(
        common_conf=common_conf,
        split="train",
        TUM_DIR=args.tum_dir,
        min_num_images=args.num_frames,
    )
    print(f"Loaded {len(dataset.sequence_list)} sequences")

    # Load model
    print("\nLoading VGGT model...")
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint)

    # Collect residuals and uncertainties
    print(f"\nRunning inference on {args.num_windows} windows...")
    data = compute_residuals_and_uncertainty(
        model, dataset, device, dtype,
        num_windows=args.num_windows,
        num_frames=args.num_frames
    )

    # Baseline comparison (now includes MLE baseline)
    baseline_results = compute_baseline_comparison(data)

    # Calibration statistics (d² vs χ²)
    calibration_results = evaluate_calibration(data)

    # Coverage statistics (z = r/σ vs N(0,1))
    coverage_results = compute_coverage_statistics(data)

    # Whitened covariance (diagonal assumption check)
    cov_results = compute_whitened_covariance(data)

    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING DIAGNOSTIC PLOTS")
    print(f"{'='*60}")

    plot_d2_histogram(data, args.output_dir)
    plot_uncertainty_vs_error(data, args.output_dir)
    plot_reliability_diagram(data, args.output_dir)
    plot_coverage_reliability(data, coverage_results, args.output_dir)
    plot_normalized_residual_histogram(coverage_results, args.output_dir)
    plot_whitened_covariance(cov_results, args.output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - d2_histogram.png              (d² distribution vs χ²)")
    print(f"  - coverage_reliability.png      (quantile-coverage diagram)")
    print(f"  - normalized_residual_histogram.png (z=r/σ vs N(0,1))")
    print(f"  - whitened_covariance.png       (Cov(z) heatmap)")
    print(f"  - uncertainty_vs_error.png      (σ vs |r| scatter)")
    print(f"  - reliability_diagram.png       (binned σ vs mean|r|)")

    # Final assessment
    print(f"\n{'='*60}")
    print("FINAL ASSESSMENT")
    print(f"{'='*60}")

    d2_rot = calibration_results['d2_rot_mean']
    d2_trans = calibration_results['d2_trans_mean']
    cov_1sigma = coverage_results.get('coverage_1sigma', 0)
    nll_improvement = baseline_results.get('nll_improvement_vs_mle', 0)
    max_off_diag = cov_results.get('max_off_diag', 0)

    print("\n  1. Calibration (d² vs χ²):")
    if 2.0 < d2_rot < 5.0 and 2.0 < d2_trans < 5.0:
        print(f"     ✓ WELL CALIBRATED")
    elif 1.0 < d2_rot < 8.0 and 1.0 < d2_trans < 8.0:
        print(f"     ~ REASONABLY CALIBRATED")
    else:
        print(f"     ✗ MISCALIBRATED")
    print(f"       d²_rot = {d2_rot:.2f}, d²_trans = {d2_trans:.2f} (target: 3)")

    print("\n  2. Coverage (z=r/σ vs N(0,1)):")
    if abs(cov_1sigma - 0.6827) < 0.05:
        print(f"     ✓ WELL CALIBRATED")
    elif abs(cov_1sigma - 0.6827) < 0.10:
        print(f"     ~ REASONABLY CALIBRATED")
    else:
        print(f"     ✗ MISCALIBRATED")
    print(f"       1σ coverage = {cov_1sigma:.3f} (target: 0.683)")

    print("\n  3. Heteroscedastic vs Homoscedastic:")
    if nll_improvement > 0.01:
        print(f"     ✓ Heteroscedastic BETTER than MLE baseline")
    elif nll_improvement > 0:
        print(f"     ~ Marginal improvement over MLE baseline")
    else:
        print(f"     ✗ MLE baseline is as good or better")
    print(f"       NLL improvement = {nll_improvement:.4f}")

    print("\n  4. Diagonal Assumption:")
    if max_off_diag < 0.2:
        print(f"     ✓ Diagonal assumption VALID (max off-diag = {max_off_diag:.3f})")
    elif max_off_diag < 0.4:
        print(f"     ~ Mild coupling detected (max off-diag = {max_off_diag:.3f})")
    else:
        print(f"     ✗ Significant coupling (max off-diag = {max_off_diag:.3f})")
        print(f"       Consider full covariance prediction")

    # Overall verdict
    print(f"\n{'='*60}")
    print("OVERALL VERDICT")
    print(f"{'='*60}")
    issues = []
    if not (2.0 < d2_rot < 5.0 and 2.0 < d2_trans < 5.0):
        issues.append("d² calibration")
    if abs(cov_1sigma - 0.6827) >= 0.10:
        issues.append("coverage calibration")
    if nll_improvement <= 0:
        issues.append("no improvement over MLE baseline")

    if len(issues) == 0:
        print("\n  ✓ UNCERTAINTY HEAD IS WELL CALIBRATED")
        print("    Ready for downstream use (PGO, fusion, etc.)")
    elif len(issues) <= 1:
        print(f"\n  ~ UNCERTAINTY HEAD IS REASONABLY CALIBRATED")
        print(f"    Minor issues: {', '.join(issues)}")
    else:
        print(f"\n  ✗ UNCERTAINTY HEAD NEEDS IMPROVEMENT")
        print(f"    Issues: {', '.join(issues)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate uncertainty calibration')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--tum_dir', type=str,
                        default='/home/yiming/Dev/tum_rgbd',
                        help='Path to TUM RGB-D root directory')
    parser.add_argument('--num_windows', type=int, default=100,
                        help='Number of windows to evaluate')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames per window')
    parser.add_argument('--output_dir', type=str, default='./eval_uncertainty',
                        help='Output directory for plots')
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()
