#!/usr/bin/env python3
"""
Phase 3: Training with TensorBoard and WandB Monitoring.

Train the pose uncertainty head with logging for visualization.

Supports augmented data sampling:
- Original: varied spacing (ids=None with get_nearby=True)
- Consecutive: fixed window sizes (8, 16, 32, 64) with 50% overlap

Usage:
    # TensorBoard only:
    python training/tests/train_uncertainty_tensorboard.py --tum_dir /path/to/tum --num_iters 500
    tensorboard --logdir ./runs

    # With WandB:
    python training/tests/train_uncertainty_tensorboard.py --tum_dir /path/to/tum --num_iters 500 --wandb

    # With checkpoint saving:
    python training/tests/train_uncertainty_tensorboard.py --tum_dir /path/to/tum --num_iters 2000 \\
        --checkpoint_dir ./checkpoints --save_interval 500 --wandb

    # With augmented consecutive window sampling (recommended):
    python training/tests/train_uncertainty_tensorboard.py --tum_dir /path/to/tum --num_iters 5000 \\
        --augment_consecutive --checkpoint_dir ./checkpoints --wandb

    # With specific sequences only:
    python training/tests/train_uncertainty_tensorboard.py --tum_dir /path/to/tum --num_iters 5000 \\
        --tum_sequences rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_room rgbd_dataset_freiburg2_xyz
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


def save_checkpoint(model, optimizer, iteration, loss_dict, checkpoint_dir, filename):
    """Save checkpoint with only uncertainty head weights.

    Only saves the pose_log_var_branch parameters (~8MB) instead of full model (~4GB).
    """
    # Extract only uncertainty head state dict
    uncertainty_state_dict = {
        name: param.cpu() for name, param in model.named_parameters()
        if 'pose_log_var_branch' in name
    }

    checkpoint = {
        'iteration': iteration,
        'uncertainty_head_state_dict': uncertainty_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_dict['pose_uncertainty_nll'].item(),
        'd2_rot_mean': loss_dict['d2_rot_mean'].item(),
        'd2_trans_mean': loss_dict['d2_trans_mean'].item(),
        'sigma_rot_mean': loss_dict['sigma_rot_mean'].item(),
        'sigma_trans_mean': loss_dict['sigma_trans_mean'].item(),
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, path)
    return path


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint and restore uncertainty head weights.

    Returns:
        iteration: The iteration number from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load uncertainty head weights
    model_state_dict = model.state_dict()
    for name, param in checkpoint['uncertainty_head_state_dict'].items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model")

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
    print(f"  d²_rot: {checkpoint['d2_rot_mean']:.2f}, d²_trans: {checkpoint['d2_trans_mean']:.2f}")

    return checkpoint['iteration']


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
    def __init__(self, get_nearby=True):
        self.img_size = 518
        self.patch_size = 14
        self.debug = False
        self.training = True
        self.get_nearby = get_nearby  # Use nearby frames for varied spacing
        self.load_depth = True
        self.inside_random = False
        self.allow_duplicate_img = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.augs = type('obj', (object,), {'scales': None})()


class AugmentedDataSampler:
    """
    Sampler that mixes varied-spacing and consecutive window sampling.

    Sampling strategies:
    1. Varied spacing (original): ids=None with get_nearby=True
       - Samples frames with varying temporal spacing
       - Good for learning covariance with different baselines

    2. Consecutive windows: fixed frame ranges with 50% overlap
       - Window sizes: 8, 16, 32, 64 frames
       - Overlap: 50% between adjacent windows
       - Good for learning uncertainty at different scales
    """

    def __init__(self, dataset, window_sizes=[8, 16, 32, 64],
                 consecutive_ratio=0.5, seed=42):
        """
        Args:
            dataset: TUMRGBDDataset instance
            window_sizes: List of window sizes for consecutive sampling
            consecutive_ratio: Probability of using consecutive sampling (0-1)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.window_sizes = window_sizes
        self.consecutive_ratio = consecutive_ratio
        self.rng = np.random.RandomState(seed)

        # Pre-compute sequence lengths for consecutive sampling
        self.seq_lengths = []
        for seq_idx in range(len(dataset.sequence_list)):
            seq_name = dataset.sequence_list[seq_idx]
            # data_store[seq_name] is a list of frame dicts
            seq_len = len(dataset.data_store[seq_name])
            self.seq_lengths.append(seq_len)

        # Build index of valid (seq_idx, window_start, window_size) tuples
        self.consecutive_windows = []
        for window_size in window_sizes:
            for seq_idx, seq_len in enumerate(self.seq_lengths):
                if seq_len < window_size:
                    continue
                # 50% overlap: step = window_size // 2
                step = window_size // 2
                for start in range(0, seq_len - window_size + 1, step):
                    self.consecutive_windows.append((seq_idx, start, window_size))

        logger.info(f"AugmentedDataSampler initialized:")
        logger.info(f"  - Window sizes: {window_sizes}")
        logger.info(f"  - Consecutive ratio: {consecutive_ratio}")
        logger.info(f"  - Total consecutive windows: {len(self.consecutive_windows)}")
        logger.info(f"    - By size: {dict((ws, sum(1 for w in self.consecutive_windows if w[2]==ws)) for ws in window_sizes)}")

    def sample(self, iteration):
        """
        Sample a batch for the given iteration.

        Returns:
            dict with keys:
                - 'seq_idx': sequence index
                - 'ids': frame indices (list or None)
                - 'num_frames': number of frames
                - 'sampling_type': 'varied' or 'consecutive'
                - 'window_size': window size (for consecutive)
        """
        use_consecutive = self.rng.rand() < self.consecutive_ratio

        if use_consecutive and len(self.consecutive_windows) > 0:
            # Sample a random consecutive window
            idx = self.rng.randint(len(self.consecutive_windows))
            seq_idx, start, window_size = self.consecutive_windows[idx]
            ids = list(range(start, start + window_size))
            return {
                'seq_idx': seq_idx,
                'ids': ids,
                'num_frames': window_size,
                'sampling_type': 'consecutive',
                'window_size': window_size,
            }
        else:
            # Use varied spacing (original method with ids=None)
            seq_idx = iteration % len(self.dataset.sequence_list)
            # Default to 8 frames for varied spacing
            num_frames = 8
            return {
                'seq_idx': seq_idx,
                'ids': None,  # Let dataset handle sampling
                'num_frames': num_frames,
                'sampling_type': 'varied',
                'window_size': num_frames,
            }


def run_training(args):
    """Run training with TensorBoard and optional WandB logging."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Setup TensorBoard
    aug_suffix = "_aug" if args.augment_consecutive else ""
    run_name = f"uncertainty{aug_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                "augment_consecutive": args.augment_consecutive,
                "consecutive_ratio": args.consecutive_ratio,
                "window_sizes": args.window_sizes,
                "tum_sequences": args.tum_sequences,
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

    # For consecutive sampling, we need get_nearby=False to use exact frame IDs
    # For varied spacing, we need get_nearby=True
    # Create both configs and switch as needed
    common_conf_nearby = MockCommonConf(get_nearby=True)
    common_conf_exact = MockCommonConf(get_nearby=False)

    # Determine minimum frames needed
    if args.augment_consecutive:
        min_frames = max(args.window_sizes)  # Need at least max window size
    else:
        min_frames = args.num_frames

    dataset_nearby = TUMRGBDDataset(
        common_conf=common_conf_nearby,
        split="train",
        TUM_DIR=args.tum_dir,
        sequences=args.tum_sequences,
        min_num_images=min_frames,
    )
    dataset_exact = TUMRGBDDataset(
        common_conf=common_conf_exact,
        split="train",
        TUM_DIR=args.tum_dir,
        sequences=args.tum_sequences,
        min_num_images=min_frames,
    )
    print(f"Loaded {len(dataset_nearby.sequence_list)} sequences:")
    for seq_name in dataset_nearby.sequence_list:
        print(f"  - {seq_name}")

    # Setup augmented sampler if requested
    sampler = None
    if args.augment_consecutive:
        sampler = AugmentedDataSampler(
            dataset=dataset_nearby,
            window_sizes=args.window_sizes,
            consecutive_ratio=args.consecutive_ratio,
            seed=42,
        )

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

    # Checkpoint tracking
    start_iteration = 0
    best_calibration_error = float('inf')  # |d²_mean - 3|
    checkpoint_enabled = args.checkpoint_dir is not None

    if checkpoint_enabled:
        print(f"\nCheckpoint saving enabled:")
        print(f"  Directory: {args.checkpoint_dir}")
        print(f"  Save interval: {args.save_interval}")

    # Resume from checkpoint if specified
    if args.resume:
        start_iteration = load_checkpoint(model, optimizer, args.resume)
        print(f"Resuming from iteration {start_iteration}")

    # Training loop
    print(f"\nStarting training for {args.num_iters} iterations...")
    print("-"*60)

    model.train()

    # Track sampling statistics
    sampling_stats = {'varied': 0, 'consecutive': 0}
    window_size_stats = {ws: 0 for ws in args.window_sizes}

    for iteration in range(start_iteration, args.num_iters):
        # Get batch using augmented sampler or original method
        if sampler is not None:
            sample_info = sampler.sample(iteration)
            seq_idx = sample_info['seq_idx']
            ids = sample_info['ids']
            num_frames = sample_info['num_frames']
            sampling_type = sample_info['sampling_type']

            # Use appropriate dataset based on sampling type
            if sampling_type == 'consecutive':
                dataset = dataset_exact
            else:
                dataset = dataset_nearby

            # Track statistics
            sampling_stats[sampling_type] += 1
            if sampling_type == 'consecutive':
                window_size_stats[sample_info['window_size']] += 1
        else:
            # Original method: varied spacing
            seq_idx = iteration % len(dataset_nearby.sequence_list)
            ids = None
            num_frames = args.num_frames
            sampling_type = 'varied'
            dataset = dataset_nearby
            sampling_stats['varied'] += 1

        batch = dataset.get_data(seq_index=seq_idx, img_per_seq=num_frames, ids=ids, aspect_ratio=1.0)

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
                loss_type=args.loss_type,
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
            sample_info_str = f"[{sampling_type[:4]}:{num_frames}f]" if sampler else ""
            print(f"Iter {step:4d} {sample_info_str} | loss: {loss.item():.4f} | "
                  f"rot_nll: {loss_dict['rot_uncertainty_nll'].item():.3f} | "
                  f"trans_nll: {loss_dict['trans_uncertainty_nll'].item():.3f} | "
                  f"d²_rot: {loss_dict['d2_rot_mean'].item():.2f} | "
                  f"d²_trans: {loss_dict['d2_trans_mean'].item():.2f} | "
                  f"scale: {loss_dict['scale_mean'].item():.3f}")

        # Log sampling statistics periodically
        if sampler is not None and step % (args.log_interval * 10) == 0:
            writer.add_scalar('sampling/varied_ratio', sampling_stats['varied'] / max(step, 1), step)
            writer.add_scalar('sampling/consecutive_ratio', sampling_stats['consecutive'] / max(step, 1), step)
            for ws in args.window_sizes:
                writer.add_scalar(f'sampling/window_{ws}', window_size_stats[ws], step)

        # Checkpoint saving
        if checkpoint_enabled:
            # Compute calibration error: |d²_rot - 3| + |d²_trans - 3|
            # Perfect calibration means d² ≈ 3 for χ²(3)
            d2_rot = loss_dict['d2_rot_mean'].item()
            d2_trans = loss_dict['d2_trans_mean'].item()
            calibration_error = abs(d2_rot - 3.0) + abs(d2_trans - 3.0)

            # Save best checkpoint (based on calibration)
            if calibration_error < best_calibration_error:
                best_calibration_error = calibration_error
                path = save_checkpoint(
                    model, optimizer, step, loss_dict,
                    args.checkpoint_dir, "best.pt"
                )
                print(f"  → New best checkpoint saved (calibration_error={calibration_error:.2f})")

            # Save periodic checkpoint
            if step % args.save_interval == 0:
                path = save_checkpoint(
                    model, optimizer, step, loss_dict,
                    args.checkpoint_dir, f"iter_{step:06d}.pt"
                )
                print(f"  → Checkpoint saved: {path}")

    # Save final checkpoint
    if checkpoint_enabled:
        path = save_checkpoint(
            model, optimizer, args.num_iters, loss_dict,
            args.checkpoint_dir, "final.pt"
        )
        print(f"\nFinal checkpoint saved: {path}")

    writer.close()
    if use_wandb:
        wandb.finish()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"Run: tensorboard --logdir {args.log_dir}")
    if use_wandb:
        print(f"WandB run: https://wandb.ai")
    if checkpoint_enabled:
        print(f"Checkpoints saved to: {args.checkpoint_dir}")
        print(f"  - best.pt (calibration_error={best_calibration_error:.2f})")
        print(f"  - final.pt")
    if sampler is not None:
        total = sampling_stats['varied'] + sampling_stats['consecutive']
        print(f"\nSampling statistics:")
        print(f"  - Varied spacing: {sampling_stats['varied']} ({100*sampling_stats['varied']/total:.1f}%)")
        print(f"  - Consecutive: {sampling_stats['consecutive']} ({100*sampling_stats['consecutive']/total:.1f}%)")
        for ws in args.window_sizes:
            if window_size_stats[ws] > 0:
                print(f"    - Window {ws}: {window_size_stats[ws]}")
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
    parser.add_argument('--loss_type', type=str, default='gaussian',
                        choices=['gaussian', 'laplace'],
                        help='NLL loss type: gaussian (default) or laplace (heavier tails)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoints (disabled if not set)')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    # Augmented sampling arguments
    parser.add_argument('--augment_consecutive', action='store_true',
                        help='Enable augmented consecutive window sampling')
    parser.add_argument('--consecutive_ratio', type=float, default=0.5,
                        help='Ratio of consecutive vs varied sampling (0-1)')
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[8, 16, 32, 64],
                        help='Window sizes for consecutive sampling')
    # Multi-sequence support
    parser.add_argument('--tum_sequences', type=str, nargs='+', default=None,
                        help='List of TUM sequence names to train on (e.g. rgbd_dataset_freiburg1_desk '
                             'rgbd_dataset_freiburg1_room). If not set, uses all sequences found in tum_dir.')
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
