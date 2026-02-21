#!/usr/bin/env python3
"""
Diagnostic: Train uncertainty head on dt=1 and dt=9 windows, evaluate separately.

Tests whether the head can learn different σ for different geometric regimes.
If d² ≈ 3 for both → head works, training variety was the issue.
If d² still overconfident for dt=9 → architecture limitation.

Usage:
    export HF_HUB_OFFLINE=1
    python training/tests/diag_dt_calibration.py --tum_dir /path/to/tum \
        --tum_sequence rgbd_dataset_freiburg1_desk --num_iters 3000
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockCommonConf:
    def __init__(self):
        self.img_size = 518
        self.patch_size = 14
        self.debug = False
        self.training = True
        self.get_nearby = False  # exact frame IDs
        self.load_depth = True
        self.inside_random = False
        self.allow_duplicate_img = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.augs = type('obj', (object,), {'scales': None})()


def make_strided_window(seq_len, num_frames, stride, rng):
    """Make a window of num_frames with given stride.

    Returns list of frame ids, or None if sequence too short.
    """
    span = (num_frames - 1) * stride + 1
    if span > seq_len:
        return None
    start = rng.randint(0, seq_len - span + 1)
    return [start + i * stride for i in range(num_frames)]


def evaluate_regime(model, dataset, seq_idx, regime_name, stride, num_frames,
                    num_eval, rng, device, dtype):
    """Evaluate d² on a specific stride regime."""
    from training.loss import compute_camera_nll_loss

    seq_name = dataset.sequence_list[seq_idx]
    seq_len = len(dataset.data_store[seq_name])

    d2_trans_all = []
    d2_rot_all = []
    sigma_trans_all = []
    residual_trans_all = []

    model.eval()
    with torch.no_grad():
        for i in range(num_eval):
            ids = make_strided_window(seq_len, num_frames, stride, rng)
            if ids is None:
                continue

            batch = dataset.get_data(seq_index=seq_idx, img_per_seq=num_frames,
                                     ids=ids, aspect_ratio=1.0)

            images = np.stack(batch['images'], axis=0)
            images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            images_tensor = torch.from_numpy(images).to(device).to(dtype).unsqueeze(0)

            gt_extrinsics = torch.from_numpy(
                np.stack(batch['extrinsics'], axis=0)).to(device).float().unsqueeze(0)
            gt_intrinsics = torch.from_numpy(
                np.stack(batch['intrinsics'], axis=0)).to(device).float().unsqueeze(0)

            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = model(images_tensor)
                batch_data = {
                    'extrinsics': gt_extrinsics,
                    'intrinsics': gt_intrinsics,
                    'images': images_tensor,
                }
                loss_dict = compute_camera_nll_loss(
                    predictions, batch_data,
                    pose_encoding_type="absT_quaR_FoV",
                    gamma=0.6,
                    sqrt_info_rot_inv_rad_clamp=(0.1, 200.0),
                    sqrt_info_trans_inv_meter_clamp=(0.1, 100.0),
                    scale_detach=True,
                    min_translation=0.02,
                    eps=1e-6,
                    loss_type='gaussian',
                )

            d2_trans_all.append(loss_dict['d2_trans_mean'].item())
            d2_rot_all.append(loss_dict['d2_rot_mean'].item())
            sigma_trans_all.append(loss_dict['sigma_trans_mean'].item())
            # Approximate residual from d² and σ: r² ≈ d² * σ²
            r_approx = (loss_dict['d2_trans_mean'].item() *
                        loss_dict['sigma_trans_mean'].item() ** 2) ** 0.5
            residual_trans_all.append(r_approx)

    model.train()

    d2_t = np.array(d2_trans_all)
    d2_r = np.array(d2_rot_all)
    sig_t = np.array(sigma_trans_all)
    res_t = np.array(residual_trans_all)

    logger.info(f"  {regime_name} (stride={stride}, n={len(d2_t)}):")
    logger.info(f"    d²_trans: mean={d2_t.mean():.2f}, median={np.median(d2_t):.2f}, "
                f"p95={np.percentile(d2_t, 95):.2f}  (target=3.0)")
    logger.info(f"    d²_rot:  mean={d2_r.mean():.2f}, median={np.median(d2_r):.2f}")
    logger.info(f"    σ_trans:  mean={sig_t.mean()*100:.2f}cm, "
                f"median={np.median(sig_t)*100:.2f}cm")
    logger.info(f"    r_trans:  mean={res_t.mean()*100:.2f}cm")

    return {
        'd2_trans_mean': d2_t.mean(),
        'd2_rot_mean': d2_r.mean(),
        'sigma_trans_mean': sig_t.mean(),
        'residual_trans_mean': res_t.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnostic: dt=1 vs dt=9 calibration')
    parser.add_argument('--tum_dir', type=str, default='/home/yiming/Dev/tum_rgbd')
    parser.add_argument('--tum_sequence', type=str, default='rgbd_dataset_freiburg1_desk')
    parser.add_argument('--num_iters', type=int, default=3000)
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Frames per window')
    parser.add_argument('--stride_a', type=int, default=1,
                        help='Stride for regime A (default: 1 = consecutive)')
    parser.add_argument('--stride_b', type=int, default=9,
                        help='Stride for regime B (default: 9 = spread out)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_eval', type=int, default=50,
                        help='Number of eval windows per regime')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Save checkpoints here')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

    # Load dataset
    logger.info("Loading dataset...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.datasets.tum_rgbd import TUMRGBDDataset

    dataset = TUMRGBDDataset(
        common_conf=MockCommonConf(),
        split="train",
        TUM_DIR=args.tum_dir,
        sequences=[args.tum_sequence],
        min_num_images=args.num_frames,
    )
    seq_name = dataset.sequence_list[0]
    seq_len = len(dataset.data_store[seq_name])
    logger.info(f"Sequence: {seq_name}, {seq_len} frames")

    # Verify both strides fit
    span_a = (args.num_frames - 1) * args.stride_a + 1
    span_b = (args.num_frames - 1) * args.stride_b + 1
    logger.info(f"Regime A: stride={args.stride_a}, span={span_a} frames")
    logger.info(f"Regime B: stride={args.stride_b}, span={span_b} frames")
    assert span_b <= seq_len, f"Stride B too large: need {span_b} frames, have {seq_len}"

    # Load model
    logger.info("Loading VGGT model...")
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Freeze except uncertainty head
    for name, param in model.named_parameters():
        param.requires_grad = 'pose_log_var_branch' in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    from training.loss import compute_camera_nll_loss

    rng = np.random.RandomState(42)

    # =========================================================================
    # Evaluate BEFORE training (baseline)
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"BEFORE TRAINING (random init)")
    logger.info(f"{'='*60}")
    eval_rng = np.random.RandomState(99)
    evaluate_regime(model, dataset, 0, "Regime A", args.stride_a,
                    args.num_frames, args.num_eval, eval_rng, device, dtype)
    eval_rng = np.random.RandomState(99)
    evaluate_regime(model, dataset, 0, "Regime B", args.stride_b,
                    args.num_frames, args.num_eval, eval_rng, device, dtype)

    # =========================================================================
    # Training: 50/50 mix of stride_a and stride_b
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"Training for {args.num_iters} iterations (50/50 stride {args.stride_a} / {args.stride_b})")
    logger.info(f"{'='*60}")

    model.train()
    stats = {'stride_a': 0, 'stride_b': 0}
    best_cal_error = float('inf')

    for iteration in range(args.num_iters):
        # 50/50 split
        use_b = rng.rand() < 0.5
        stride = args.stride_b if use_b else args.stride_a
        regime = 'stride_b' if use_b else 'stride_a'
        stats[regime] += 1

        ids = make_strided_window(seq_len, args.num_frames, stride, rng)
        if ids is None:
            ids = make_strided_window(seq_len, args.num_frames, args.stride_a, rng)
            stats['stride_a'] += 1
            stats['stride_b'] -= 1

        batch = dataset.get_data(seq_index=0, img_per_seq=args.num_frames,
                                 ids=ids, aspect_ratio=1.0)

        images = np.stack(batch['images'], axis=0)
        images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        images_tensor = torch.from_numpy(images).to(device).to(dtype).unsqueeze(0)

        gt_extrinsics = torch.from_numpy(
            np.stack(batch['extrinsics'], axis=0)).to(device).float().unsqueeze(0)
        gt_intrinsics = torch.from_numpy(
            np.stack(batch['intrinsics'], axis=0)).to(device).float().unsqueeze(0)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(images_tensor)
            batch_data = {
                'extrinsics': gt_extrinsics,
                'intrinsics': gt_intrinsics,
                'images': images_tensor,
            }
            loss_dict = compute_camera_nll_loss(
                predictions, batch_data,
                pose_encoding_type="absT_quaR_FoV",
                gamma=0.6,
                sqrt_info_rot_inv_rad_clamp=(0.1, 200.0),
                sqrt_info_trans_inv_meter_clamp=(0.1, 100.0),
                scale_detach=True,
                min_translation=0.02,
                eps=1e-6,
                loss_type='gaussian',
            )
            loss = loss_dict['pose_uncertainty_nll']

        loss.backward()
        optimizer.step()

        step = iteration + 1
        if step % 100 == 0 or step == 1:
            tag = 'A' if not use_b else 'B'
            logger.info(f"Iter {step:4d} [{tag}:s{stride}] | loss: {loss.item():.4f} | "
                        f"d²_rot: {loss_dict['d2_rot_mean'].item():.2f} | "
                        f"d²_trans: {loss_dict['d2_trans_mean'].item():.2f} | "
                        f"σ_trans: {loss_dict['sigma_trans_mean'].item()*100:.2f}cm | "
                        f"scale: {loss_dict['scale_mean'].item():.3f}")

        # Save best checkpoint
        if args.checkpoint_dir:
            d2_rot = loss_dict['d2_rot_mean'].item()
            d2_trans = loss_dict['d2_trans_mean'].item()
            cal_error = abs(d2_rot - 3.0) + abs(d2_trans - 3.0)
            if cal_error < best_cal_error:
                best_cal_error = cal_error
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                ckpt = {
                    'iteration': step,
                    'uncertainty_head_state_dict': {
                        n: p.cpu() for n, p in model.named_parameters()
                        if 'pose_log_var_branch' in n
                    },
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'd2_rot_mean': d2_rot,
                    'd2_trans_mean': d2_trans,
                    'sigma_rot_mean': loss_dict['sigma_rot_mean'].item(),
                    'sigma_trans_mean': loss_dict['sigma_trans_mean'].item(),
                }
                torch.save(ckpt, os.path.join(args.checkpoint_dir, 'best.pt'))

    logger.info(f"\nTraining stats: A={stats['stride_a']}, B={stats['stride_b']}")

    # =========================================================================
    # Evaluate AFTER training
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"AFTER TRAINING ({args.num_iters} iters)")
    logger.info(f"{'='*60}")
    eval_rng = np.random.RandomState(99)
    result_a = evaluate_regime(model, dataset, 0, "Regime A", args.stride_a,
                               args.num_frames, args.num_eval, eval_rng, device, dtype)
    eval_rng = np.random.RandomState(99)
    result_b = evaluate_regime(model, dataset, 0, "Regime B", args.stride_b,
                               args.num_frames, args.num_eval, eval_rng, device, dtype)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"DIAGNOSTIC SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Regime A (stride={args.stride_a}): d²_trans={result_a['d2_trans_mean']:.2f}, "
                f"σ_trans={result_a['sigma_trans_mean']*100:.2f}cm, "
                f"r_trans={result_a['residual_trans_mean']*100:.2f}cm")
    logger.info(f"  Regime B (stride={args.stride_b}): d²_trans={result_b['d2_trans_mean']:.2f}, "
                f"σ_trans={result_b['sigma_trans_mean']*100:.2f}cm, "
                f"r_trans={result_b['residual_trans_mean']*100:.2f}cm")
    logger.info(f"")

    sigma_ratio = result_b['sigma_trans_mean'] / max(result_a['sigma_trans_mean'], 1e-8)
    residual_ratio = result_b['residual_trans_mean'] / max(result_a['residual_trans_mean'], 1e-8)
    logger.info(f"  σ ratio (B/A): {sigma_ratio:.2f}")
    logger.info(f"  r ratio (B/A): {residual_ratio:.2f}")
    logger.info(f"  (If head works, σ ratio ≈ r ratio)")
    logger.info(f"")

    both_calibrated = (abs(result_a['d2_trans_mean'] - 3.0) < 3.0 and
                       abs(result_b['d2_trans_mean'] - 3.0) < 3.0)
    if both_calibrated:
        logger.info(f"  ✓ PASS: Both regimes near-calibrated (d² ≈ 3)")
        logger.info(f"    → Head CAN differentiate, training variety was the issue")
    else:
        logger.info(f"  ✗ FAIL: At least one regime miscalibrated")
        if result_b['d2_trans_mean'] > result_a['d2_trans_mean'] * 1.5:
            logger.info(f"    → Head is overconfident for large-baseline regime")
            logger.info(f"    → Likely architecture limitation (insufficient σ dynamic range)")
        else:
            logger.info(f"    → Both regimes similarly miscalibrated")
            logger.info(f"    → May need more training or different hyperparameters")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
