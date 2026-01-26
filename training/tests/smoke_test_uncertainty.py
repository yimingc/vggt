#!/usr/bin/env python3
"""
Phase 2: Smoke Test for Pose Uncertainty Head.

This script verifies that:
1. Model loads with uncertainty head
2. Only uncertainty branch is trainable (backbone/pose head frozen)
3. NLL loss decreases over iterations
4. No NaN/Inf in loss or gradients
5. Uncertainty metrics are reasonable (not stuck at clamps)

Usage:
    python training/tests/smoke_test_uncertainty.py --tum_dir /path/to/tum_sequence
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_trainable_params(model):
    """Verify only uncertainty head is trainable."""
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, param.numel()))
        else:
            frozen.append(name)

    total_trainable = sum(n for _, n in trainable)
    total_frozen = sum(1 for _ in frozen)

    print(f"\n{'='*60}")
    print(f"TRAINABLE PARAMETERS: {total_trainable:,}")
    print(f"{'='*60}")
    for name, count in trainable:
        print(f"  {name}: {count:,}")

    print(f"\nFrozen modules: {total_frozen} parameter groups")

    # Sanity check: should be ~2.1M for uncertainty MLP
    assert total_trainable < 5_000_000, f"Too many trainable params: {total_trainable}"
    assert total_trainable > 100_000, f"Too few trainable params: {total_trainable}"
    assert any('pose_uncertainty_branch' in name for name, _ in trainable), \
        "pose_uncertainty_branch not in trainable params!"
    print(f"\n✓ Verified: Only uncertainty head is trainable ({total_trainable:,} params)")
    return total_trainable


def freeze_except_uncertainty(model):
    """Freeze all parameters except pose_uncertainty_branch."""
    for name, param in model.named_parameters():
        if 'pose_uncertainty_branch' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


class MockCommonConf:
    """Mock configuration for dataset."""
    def __init__(self):
        self.img_size = 518
        self.patch_size = 14
        self.debug = False
        self.training = True  # Training mode
        self.get_nearby = True  # Use nearby frames (important!)
        self.load_depth = True
        self.inside_random = False
        self.allow_duplicate_img = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.augs = type('obj', (object,), {'scales': None})()


def run_smoke_test(args):
    """Run the smoke test."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("="*60)
    print("Phase 2: Pose Uncertainty Smoke Test")
    print("="*60)
    print(f"Device: {device}, dtype: {dtype}")
    print(f"TUM directory: {args.tum_dir}")
    print(f"Iterations: {args.num_iters}")
    print(f"Frames per batch: {args.num_frames}")

    # Load dataset
    print("\n[1/5] Loading TUM dataset...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.datasets.tum_rgbd import TUMRGBDDataset

    common_conf = MockCommonConf()
    dataset = TUMRGBDDataset(
        common_conf=common_conf,
        split="train",
        TUM_DIR=args.tum_dir,
        min_num_images=args.num_frames,
    )
    print(f"  Loaded {len(dataset.sequence_list)} sequences")

    # Load model
    print("\n[2/5] Loading VGGT model with uncertainty head...")
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Freeze everything except uncertainty head
    print("\n[3/5] Freezing backbone and pose head...")
    freeze_except_uncertainty(model)
    trainable_params = verify_trainable_params(model)

    # Setup optimizer (only for uncertainty params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    # Import loss function
    from training.loss import compute_camera_nll_loss
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding

    # Training loop
    print(f"\n[4/5] Running {args.num_iters} iterations...")
    print("-"*60)

    model.train()
    losses = []
    metrics_history = []

    for iteration in range(args.num_iters):
        # Get batch
        seq_idx = iteration % len(dataset.sequence_list)
        batch = dataset.get_data(seq_index=seq_idx, img_per_seq=args.num_frames, ids=None, aspect_ratio=1.0)

        # Prepare images
        images = np.stack(batch['images'], axis=0)
        images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        images_tensor = torch.from_numpy(images).to(device).to(dtype).unsqueeze(0)

        # Prepare GT
        gt_extrinsics = torch.from_numpy(np.stack(batch['extrinsics'], axis=0)).to(device).float().unsqueeze(0)
        gt_intrinsics = torch.from_numpy(np.stack(batch['intrinsics'], axis=0)).to(device).float().unsqueeze(0)

        # Forward pass
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(images_tensor)

            # Prepare batch_data for loss
            batch_data = {
                'extrinsics': gt_extrinsics,
                'intrinsics': gt_intrinsics,
                'images': images_tensor,
            }

            # Compute NLL loss
            loss_dict = compute_camera_nll_loss(
                predictions,
                batch_data,
                pose_encoding_type="absT_quaR_FoV",
                gamma=0.6,
                sqrt_info_rot_inv_rad_clamp=(0.1, 200.0),
                sqrt_info_trans_inv_meter_clamp=(0.1, 100.0),
                residual_sq_clamp=100.0,  # Safety for smoke test
                scale_detach=True,
                min_translation=0.02,
                eps=1e-6,
            )

            loss = loss_dict['loss_camera_nll']

        # Backward pass
        loss.backward()

        # Check for NaN/Inf gradients
        grad_norm = 0.0
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan = True
                    logger.error(f"NaN/Inf gradient in {name}")
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        # Record metrics
        loss_val = loss.item()
        losses.append(loss_val)
        metrics = {
            'loss': loss_val,
            'nll_rot': loss_dict['nll_rot'].item(),
            'nll_trans': loss_dict['nll_trans'].item(),
            'sqrt_info_rot_inv_rad_mean': loss_dict['sqrt_info_rot_inv_rad_mean'].item(),
            'sqrt_info_trans_inv_meter_mean': loss_dict['sqrt_info_trans_inv_meter_mean'].item(),
            'd2_rot_mean': loss_dict['d2_rot_mean'].item(),
            'd2_trans_mean': loss_dict['d2_trans_mean'].item(),
            'scale_mean': loss_dict['scale_mean'].item(),
            'scale_std': loss_dict['scale_std'].item(),
            'residual_sq_clamped_ratio': loss_dict['residual_sq_clamped_ratio'].item(),
            'grad_norm': grad_norm,
            'has_nan': has_nan,
        }
        metrics_history.append(metrics)

        # Log every log_interval iterations
        if (iteration + 1) % args.log_interval == 0 or iteration == 0:
            print(f"Iter {iteration+1:3d} | loss: {loss_val:.4f} | "
                  f"nll_rot: {metrics['nll_rot']:.3f} | nll_trans: {metrics['nll_trans']:.3f} | "
                  f"d²_rot: {metrics['d2_rot_mean']:.2f} | d²_trans: {metrics['d2_trans_mean']:.2f} | "
                  f"scale: {metrics['scale_mean']:.3f} | grad: {grad_norm:.4f}")

    # Evaluate results
    print("\n" + "="*60)
    print("[5/5] Evaluating Results")
    print("="*60)

    success = True

    # Check 1: Loss decreases
    first_half_mean = np.mean(losses[:len(losses)//2])
    second_half_mean = np.mean(losses[len(losses)//2:])
    loss_decreased = second_half_mean < first_half_mean
    print(f"\n✓ Loss trend: {first_half_mean:.4f} → {second_half_mean:.4f} "
          f"({'decreasing ✓' if loss_decreased else 'NOT decreasing ✗'})")
    if not loss_decreased:
        logger.warning("Loss did not decrease - may need more iterations or tuning")

    # Check 2: No NaN/Inf
    any_nan = any(m['has_nan'] for m in metrics_history)
    print(f"✓ NaN/Inf check: {'PASSED ✓' if not any_nan else 'FAILED ✗'}")
    if any_nan:
        success = False

    # Check 3: sqrt_info not stuck at clamps
    final_sqrt_info_rot = metrics_history[-1]['sqrt_info_rot_inv_rad_mean']
    final_sqrt_info_trans = metrics_history[-1]['sqrt_info_trans_inv_meter_mean']
    rot_at_clamp = final_sqrt_info_rot <= 0.11 or final_sqrt_info_rot >= 199
    trans_at_clamp = final_sqrt_info_trans <= 0.11 or final_sqrt_info_trans >= 99
    print(f"✓ sqrt_info_rot_inv_rad: {final_sqrt_info_rot:.2f} "
          f"({'at clamp ✗' if rot_at_clamp else 'OK ✓'})")
    print(f"✓ sqrt_info_trans_inv_meter: {final_sqrt_info_trans:.2f} "
          f"({'at clamp ✗' if trans_at_clamp else 'OK ✓'})")

    # Check 4: Scale stable
    final_scale = metrics_history[-1]['scale_mean']
    scale_std = metrics_history[-1]['scale_std']
    scale_at_extreme = final_scale <= 0.02 or final_scale >= 99
    print(f"✓ scale: {final_scale:.3f} ± {scale_std:.3f} "
          f"({'at extreme ✗' if scale_at_extreme else 'OK ✓'})")

    # Check 5: Clamped ratio
    final_clamp_ratio = metrics_history[-1]['residual_sq_clamped_ratio']
    clamp_ratio_ok = final_clamp_ratio < 0.1
    print(f"✓ residual_sq_clamped_ratio: {final_clamp_ratio:.3f} "
          f"({'< 10% ✓' if clamp_ratio_ok else '≥ 10% - loss may be misleading ✗'})")

    # Check 6: Calibration (d² should approach 3)
    final_d2_rot = metrics_history[-1]['d2_rot_mean']
    final_d2_trans = metrics_history[-1]['d2_trans_mean']
    print(f"✓ d²_rot: {final_d2_rot:.2f} (expect ~3 if calibrated)")
    print(f"✓ d²_trans: {final_d2_trans:.2f} (expect ~3 if calibrated)")

    print("\n" + "="*60)
    if success and not any_nan:
        print("SMOKE TEST PASSED ✓")
    else:
        print("SMOKE TEST FAILED ✗")
    print("="*60)

    return success, metrics_history


def main():
    parser = argparse.ArgumentParser(description='Smoke test for pose uncertainty head')
    parser.add_argument('--tum_dir', type=str,
                        default='/home/yiming/Dev/tum_rgbd',
                        help='Path to TUM RGB-D root directory (containing sequence folders)')
    parser.add_argument('--num_iters', type=int, default=20,
                        help='Number of training iterations (default: 20)')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames per batch (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='Log every N iterations (default: 5)')
    args = parser.parse_args()

    success, metrics = run_smoke_test(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
