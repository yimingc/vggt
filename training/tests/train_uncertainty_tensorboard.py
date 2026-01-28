#!/usr/bin/env python3
"""
Phase 3: Training with TensorBoard and WandB Monitoring.

Train the pose uncertainty head with logging for visualization.

Usage:
    # TensorBoard only:
    python training/tests/train_uncertainty_tensorboard.py --tum_dir /path/to/tum --num_iters 500
    tensorboard --logdir ./runs

    # With WandB:
    python training/tests/train_uncertainty_tensorboard.py --tum_dir /path/to/tum --num_iters 500 --wandb
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def freeze_except_uncertainty(model):
    """Freeze all parameters except pose_log_var_branch."""
    for name, param in model.named_parameters():
        if 'pose_log_var_branch' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def verify_trainable_params(model):
    """Verify only uncertainty head is trainable."""
    trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, param.numel()))

    total_trainable = sum(n for _, n in trainable)
    print(f"\nTrainable parameters: {total_trainable:,}")
    for name, count in trainable:
        print(f"  {name}: {count:,}")
    return total_trainable


class MockCommonConf:
    """Mock configuration for dataset."""
    def __init__(self):
        self.img_size = 518
        self.patch_size = 14
        self.debug = False
        self.training = True
        self.get_nearby = True  # Use nearby frames
        self.load_depth = True
        self.inside_random = False
        self.allow_duplicate_img = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.augs = type('obj', (object,), {'scales': None})()


def run_training(args):
    """Run training with TensorBoard and optional WandB logging."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Setup TensorBoard
    run_name = f"uncertainty_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir)
    print(f"\n{'='*60}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Run: tensorboard --logdir {args.log_dir}")
    print(f"{'='*60}")

    # Setup WandB
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: WandB requested but not installed. Install with: pip install wandb")
    if use_wandb:
        wandb.init(
            project="vggt-uncertainty",
            name=run_name,
            config={
                "num_iters": args.num_iters,
                "num_frames": args.num_frames,
                "lr": args.lr,
                "clamp_residual": args.clamp_residual,
            }
        )
        print(f"WandB run: {wandb.run.url}")

    print(f"\nDevice: {device}, dtype: {dtype}")
    print(f"Iterations: {args.num_iters}")
    print(f"Frames per batch: {args.num_frames}")

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

    # Freeze everything except uncertainty head
    print("Freezing backbone and pose head...")
    freeze_except_uncertainty(model)
    verify_trainable_params(model)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    # Import loss function
    from training.loss import compute_camera_nll_loss

    # Training loop
    print(f"\nStarting training for {args.num_iters} iterations...")
    print("-"*60)

    model.train()

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

            batch_data = {
                'extrinsics': gt_extrinsics,
                'intrinsics': gt_intrinsics,
                'images': images_tensor,
            }

            loss_dict = compute_camera_nll_loss(
                predictions,
                batch_data,
                pose_encoding_type="absT_quaR_FoV",
                gamma=0.6,
                sqrt_info_rot_inv_rad_clamp=(0.1, 200.0),
                sqrt_info_trans_inv_meter_clamp=(0.1, 100.0),
                residual_sq_clamp=100.0 if args.clamp_residual else None,
                scale_detach=True,
                min_translation=0.02,
                eps=1e-6,
            )

            loss = loss_dict['pose_uncertainty_nll']

        # Backward pass
        loss.backward()

        # Compute gradient norm for log-variance head
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if 'pose_log_var_branch' in name and param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        # Log to TensorBoard
        step = iteration + 1

        # Loss metrics
        writer.add_scalar('loss/pose_uncertainty_nll', loss_dict['pose_uncertainty_nll'].item(), step)
        writer.add_scalar('loss/rot_uncertainty_nll', loss_dict['rot_uncertainty_nll'].item(), step)
        writer.add_scalar('loss/trans_uncertainty_nll', loss_dict['trans_uncertainty_nll'].item(), step)

        # Uncertainty metrics (sigma = exp(0.5 * log_var))
        writer.add_scalar('uncertainty/sigma_rot_mean', loss_dict['sigma_rot_mean'].item(), step)
        writer.add_scalar('uncertainty/sigma_trans_mean', loss_dict['sigma_trans_mean'].item(), step)

        # Calibration metrics
        writer.add_scalar('calibration/d2_rot_mean', loss_dict['d2_rot_mean'].item(), step)
        writer.add_scalar('calibration/d2_trans_mean', loss_dict['d2_trans_mean'].item(), step)

        # Scale fitting metrics
        writer.add_scalar('scale/mean', loss_dict['scale_mean'].item(), step)
        writer.add_scalar('scale/std', loss_dict['scale_std'].item() if isinstance(loss_dict['scale_std'], torch.Tensor) else loss_dict['scale_std'], step)
        writer.add_scalar('scale/valid_count_mean', loss_dict['scale_valid_count_mean'].item(), step)
        writer.add_scalar('scale/raw', loss_dict['scale_raw'].item(), step)

        # Debug metrics
        writer.add_scalar('debug/residual_sq_clamped_ratio', loss_dict['residual_sq_clamped_ratio'].item(), step)
        writer.add_scalar('debug/gt_trans_norm_mean', loss_dict['gt_trans_norm_mean'].item(), step)
        writer.add_scalar('debug/pred_trans_norm_raw_mean', loss_dict['pred_trans_norm_raw_mean'].item(), step)
        writer.add_scalar('debug/pred_trans_norm_scaled_mean', loss_dict['pred_trans_norm_scaled_mean'].item(), step)

        # Diagnostic: sigma percentiles (σ = exp(0.5 * log_var))
        writer.add_scalar('diagnostic/sigma_rot_p10', loss_dict['sigma_rot_p10'].item(), step)
        writer.add_scalar('diagnostic/sigma_rot_p50', loss_dict['sigma_rot_p50'].item(), step)
        writer.add_scalar('diagnostic/sigma_rot_p90', loss_dict['sigma_rot_p90'].item(), step)
        writer.add_scalar('diagnostic/sigma_trans_p10', loss_dict['sigma_trans_p10'].item(), step)
        writer.add_scalar('diagnostic/sigma_trans_p50', loss_dict['sigma_trans_p50'].item(), step)
        writer.add_scalar('diagnostic/sigma_trans_p90', loss_dict['sigma_trans_p90'].item(), step)

        # Diagnostic: log_var clamp hit rate (should be 0 with loose clamp)
        writer.add_scalar('diagnostic/log_var_rot_at_min', loss_dict['log_var_rot_at_min'].item(), step)
        writer.add_scalar('diagnostic/log_var_rot_at_max', loss_dict['log_var_rot_at_max'].item(), step)
        writer.add_scalar('diagnostic/log_var_trans_at_min', loss_dict['log_var_trans_at_min'].item(), step)
        writer.add_scalar('diagnostic/log_var_trans_at_max', loss_dict['log_var_trans_at_max'].item(), step)

        # Diagnostic: residual distribution
        writer.add_scalar('diagnostic/residual_rot_p90', loss_dict['residual_rot_p90'].item(), step)
        writer.add_scalar('diagnostic/residual_trans_p90', loss_dict['residual_trans_p90'].item(), step)

        # Gradient norm
        writer.add_scalar('grad/uncertainty_head_norm', grad_norm, step)

        # Log to WandB
        if use_wandb:
            wandb.log({
                "loss/pose_uncertainty_nll": loss_dict['pose_uncertainty_nll'].item(),
                "loss/rot_uncertainty_nll": loss_dict['rot_uncertainty_nll'].item(),
                "loss/trans_uncertainty_nll": loss_dict['trans_uncertainty_nll'].item(),
                "uncertainty/sigma_rot_mean": loss_dict['sigma_rot_mean'].item(),
                "uncertainty/sigma_trans_mean": loss_dict['sigma_trans_mean'].item(),
                "calibration/d2_rot_mean": loss_dict['d2_rot_mean'].item(),
                "calibration/d2_trans_mean": loss_dict['d2_trans_mean'].item(),
                "scale/mean": loss_dict['scale_mean'].item(),
                "scale/std": loss_dict['scale_std'].item() if isinstance(loss_dict['scale_std'], torch.Tensor) else loss_dict['scale_std'],
                "scale/valid_count_mean": loss_dict['scale_valid_count_mean'].item(),
                "scale/raw": loss_dict['scale_raw'].item(),
                "debug/residual_sq_clamped_ratio": loss_dict['residual_sq_clamped_ratio'].item(),
                "debug/gt_trans_norm_mean": loss_dict['gt_trans_norm_mean'].item(),
                "debug/pred_trans_norm_raw_mean": loss_dict['pred_trans_norm_raw_mean'].item(),
                "debug/pred_trans_norm_scaled_mean": loss_dict['pred_trans_norm_scaled_mean'].item(),
                "grad/uncertainty_head_norm": grad_norm,
                # Diagnostic: sigma percentiles
                "diagnostic/sigma_rot_p10": loss_dict['sigma_rot_p10'].item(),
                "diagnostic/sigma_rot_p50": loss_dict['sigma_rot_p50'].item(),
                "diagnostic/sigma_rot_p90": loss_dict['sigma_rot_p90'].item(),
                "diagnostic/sigma_trans_p10": loss_dict['sigma_trans_p10'].item(),
                "diagnostic/sigma_trans_p50": loss_dict['sigma_trans_p50'].item(),
                "diagnostic/sigma_trans_p90": loss_dict['sigma_trans_p90'].item(),
                # Diagnostic: log_var clamp hit rate
                "diagnostic/log_var_rot_at_min": loss_dict['log_var_rot_at_min'].item(),
                "diagnostic/log_var_rot_at_max": loss_dict['log_var_rot_at_max'].item(),
                "diagnostic/log_var_trans_at_min": loss_dict['log_var_trans_at_min'].item(),
                "diagnostic/log_var_trans_at_max": loss_dict['log_var_trans_at_max'].item(),
                # Diagnostic: residual distribution
                "diagnostic/residual_rot_p90": loss_dict['residual_rot_p90'].item(),
                "diagnostic/residual_trans_p90": loss_dict['residual_trans_p90'].item(),
            }, step=step)

        # Console output
        if step % args.log_interval == 0 or step == 1:
            print(f"Iter {step:4d} | loss: {loss.item():.4f} | "
                  f"rot_nll: {loss_dict['rot_uncertainty_nll'].item():.3f} | "
                  f"trans_nll: {loss_dict['trans_uncertainty_nll'].item():.3f} | "
                  f"d²_rot: {loss_dict['d2_rot_mean'].item():.2f} | "
                  f"d²_trans: {loss_dict['d2_trans_mean'].item():.2f} | "
                  f"scale: {loss_dict['scale_mean'].item():.3f}")

    writer.close()
    if use_wandb:
        wandb.finish()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"Run: tensorboard --logdir {args.log_dir}")
    if use_wandb:
        print(f"WandB run: https://wandb.ai")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Train uncertainty head with TensorBoard')
    parser.add_argument('--tum_dir', type=str,
                        default='/home/yiming/Dev/tum_rgbd',
                        help='Path to TUM RGB-D root directory')
    parser.add_argument('--num_iters', type=int, default=500,
                        help='Number of training iterations')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames per batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log to console every N iterations')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='TensorBoard log directory')
    parser.add_argument('--clamp_residual', action='store_true',
                        help='Clamp residual_sq (for stability)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
