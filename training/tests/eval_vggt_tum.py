#!/usr/bin/env python3
"""
Evaluate VGGT pose predictions on TUM RGB-D dataset.

Computes:
- ATE (Absolute Trajectory Error) after Umeyama alignment
- RPE (Relative Pose Error) for translation and rotation

Usage:
    python training/tests/eval_vggt_tum.py --tum_dir /path/to/tum --num_frames 8
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least-squares solution for the Umeyama alignment.
    Aligns y to x: finds R, t, s such that x ≈ s * R @ y + t

    For row vectors (N, 3), the aligned result is: s * (y @ R.T) + t

    Args:
        x: (N, 3) reference/target points
        y: (N, 3) source points to align
        with_scale: whether to estimate scale

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scale factor
    """
    assert x.shape == y.shape
    n, dim = x.shape

    # Center the points
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Compute covariance: H = Y.T @ X (for Procrustes: find R s.t. X ≈ Y @ R.T)
    # This is equivalent to sum_i (y_i.T @ x_i) where y_i, x_i are column vectors
    H = y_centered.T @ x_centered

    # SVD: H = U @ S @ V.T
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Rotation: R = V @ U.T (standard Procrustes solution)
    # Handle reflection case
    d = np.ones(dim)
    if np.linalg.det(V @ U.T) < 0:
        d[dim - 1] = -1

    R = V @ np.diag(d) @ U.T

    # Compute scale
    if with_scale:
        var_y = np.sum(y_centered ** 2)
        s = np.sum(S * d) / var_y
    else:
        s = 1.0

    # Compute translation: x = s * R @ y + t => t = x_mean - s * R @ y_mean
    t = x_mean - s * R @ y_mean

    return R, t, s


def extract_camera_positions(poses, convention='w2c'):
    """Extract camera positions from extrinsic matrices.

    Args:
        poses: (N, 3, 4) pose matrices
        convention: 'w2c' - World-to-Camera: P_cam = R @ P_world + t
                           Camera position = -R^T @ t
                    'c2w' - Camera-to-World: P_world = R @ P_cam + t
                           Camera position = t (translation is camera center)
    """
    positions = []
    for i in range(len(poses)):
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        if convention == 'w2c':
            positions.append(-R.T @ t)
        else:  # c2w
            positions.append(t)
    return np.array(positions)


def compute_ate(pred_poses, gt_poses, align='sim3',
                gt_convention='w2c', pred_convention='w2c'):
    """
    Compute Absolute Trajectory Error.

    Args:
        pred_poses: (N, 3, 4) predicted poses
        gt_poses: (N, 3, 4) ground truth poses
        align: 'sim3' for Sim3 alignment (with scale),
               'se3' for SE3 alignment (no scale),
               'none' for no alignment
        gt_convention: 'w2c' or 'c2w' for ground truth poses
        pred_convention: 'w2c' or 'c2w' for predicted poses

    Returns:
        ate_rmse: RMSE of position errors
        ate_mean: mean position error
        aligned_pred: aligned predicted positions
        gt_positions: ground truth positions
        scale: scale factor (1.0 if not using Sim3)
    """
    pred_positions = extract_camera_positions(pred_poses, pred_convention)
    gt_positions = extract_camera_positions(gt_poses, gt_convention)

    if align == 'sim3':
        # Sim3 alignment (with scale)
        R_align, t_align, s_align = umeyama_alignment(gt_positions, pred_positions, with_scale=True)
        aligned_pred = s_align * (pred_positions @ R_align.T) + t_align
    elif align == 'se3':
        # SE3 alignment (no scale)
        R_align, t_align, s_align = umeyama_alignment(gt_positions, pred_positions, with_scale=False)
        aligned_pred = (pred_positions @ R_align.T) + t_align
        s_align = 1.0
    else:
        aligned_pred = pred_positions
        s_align = 1.0

    # Compute errors
    errors = np.linalg.norm(aligned_pred - gt_positions, axis=1)
    ate_rmse = np.sqrt(np.mean(errors ** 2))
    ate_mean = np.mean(errors)

    return ate_rmse, ate_mean, aligned_pred, gt_positions, s_align


def rotation_error(R1, R2):
    """Compute rotation error in degrees."""
    R_diff = R1.T @ R2
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    angle = np.arccos(trace) * 180 / np.pi
    return angle


def compute_rpe(pred_poses, gt_poses, delta=1):
    """
    Compute Relative Pose Error.

    Args:
        pred_poses: (N, 3, 4) predicted poses
        gt_poses: (N, 3, 4) ground truth poses
        delta: frame interval for computing relative poses

    Returns:
        rpe_trans: mean relative translation error
        rpe_rot: mean relative rotation error (degrees)
    """
    trans_errors = []
    rot_errors = []

    for i in range(len(pred_poses) - delta):
        # Predicted relative pose
        R1_pred = pred_poses[i, :3, :3]
        t1_pred = pred_poses[i, :3, 3]
        R2_pred = pred_poses[i + delta, :3, :3]
        t2_pred = pred_poses[i + delta, :3, 3]

        R_rel_pred = R1_pred.T @ R2_pred
        t_rel_pred = R1_pred.T @ (t2_pred - t1_pred)

        # GT relative pose
        R1_gt = gt_poses[i, :3, :3]
        t1_gt = gt_poses[i, :3, 3]
        R2_gt = gt_poses[i + delta, :3, :3]
        t2_gt = gt_poses[i + delta, :3, 3]

        R_rel_gt = R1_gt.T @ R2_gt
        t_rel_gt = R1_gt.T @ (t2_gt - t1_gt)

        # Errors
        trans_errors.append(np.linalg.norm(t_rel_pred - t_rel_gt))
        rot_errors.append(rotation_error(R_rel_pred, R_rel_gt))

    return np.mean(trans_errors), np.mean(rot_errors)


class MockCommonConf:
    """Mock configuration for dataset."""
    def __init__(self):
        self.img_size = 518
        self.patch_size = 14
        self.debug = False
        self.training = False
        self.get_nearby = True
        self.load_depth = True
        self.inside_random = False
        self.allow_duplicate_img = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.augs = type('obj', (object,), {'scales': None})()


def evaluate_sequence(dataset, seq_index, num_frames, model, device, dtype,
                       sampling='uniform', start_frame=0, end_frame=None,
                       dryrun=False, return_vggt_predictions=False):
    """Evaluate VGGT on a single sequence.

    Args:
        sampling: 'uniform' - evenly spread within [start_frame, end_frame]
                  'consecutive' - consecutive frames starting from start_frame
                  'random' - random frames within [start_frame, end_frame]
        start_frame: starting frame index (default: 0)
        end_frame: ending frame index, inclusive (default: last frame)
        dryrun: if True, use GT as prediction (skip VGGT inference) to verify pipeline
        return_vggt_predictions: if True, also return full VGGT predictions dict for viser
    """
    # Get sequence name and length
    seq_name = dataset.sequence_list[seq_index]
    seq_len = len(dataset.data_store[seq_name])

    # Set default end_frame to last frame
    if end_frame is None:
        end_frame = seq_len - 1

    # Clamp to valid range
    start_frame = max(0, min(start_frame, seq_len - 1))
    end_frame = max(start_frame, min(end_frame, seq_len - 1))
    frame_range = end_frame - start_frame + 1

    # Determine frame IDs based on sampling strategy
    if sampling == 'uniform':
        # Evenly spread num_frames within [start_frame, end_frame]
        ids = np.linspace(start_frame, end_frame, num_frames, dtype=int).tolist()
    elif sampling == 'consecutive':
        # Consecutive frames starting from start_frame
        actual_num = min(num_frames, frame_range)
        ids = list(range(start_frame, start_frame + actual_num))
        if actual_num < num_frames:
            print(f"  Warning: Only {actual_num} frames available in range [{start_frame}, {end_frame}]")
    else:  # random
        # Random frames within [start_frame, end_frame]
        available = list(range(start_frame, end_frame + 1))
        actual_num = min(num_frames, len(available))
        ids = sorted(np.random.choice(available, actual_num, replace=False).tolist())

    # Get data with specified IDs
    batch = dataset.get_data(seq_index=seq_index, img_per_seq=len(ids), ids=ids, aspect_ratio=1.0)

    print(f"\nSequence: {batch['seq_name']} (total {seq_len} frames)")
    print(f"Frame range: [{start_frame}, {end_frame}], Sampling: {sampling}, Frames: {len(ids)}")
    print(f"IDs: {ids[:5]}{'...' + str(ids[-1]) if len(ids) > 5 else ''}")
    if dryrun:
        print("  [DRYRUN MODE] Using GT as prediction")

    # Get GT poses (our loader converts TUM to w2c convention)
    gt_poses = np.stack(batch['extrinsics'], axis=0)

    vggt_predictions = None  # Will be set if return_vggt_predictions=True

    if dryrun:
        # Use GT as prediction to verify pipeline
        pred_poses = gt_poses.copy()
    else:
        # Prepare images for VGGT
        # Note: VGGT model does normalization internally (aggregator.py:200-201)
        # So we only convert to [0, 1] range, do NOT apply mean/std normalization here
        images = np.stack(batch['images'], axis=0)
        images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        images_tensor = torch.from_numpy(images).to(device).to(dtype).unsqueeze(0)

        # Run VGGT
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = model(images_tensor)

        # Get predicted poses
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        pred_extri, pred_intri = pose_encoding_to_extri_intri(
            predictions['pose_enc'], images_tensor.shape[-2:]
        )
        pred_poses = pred_extri[0].float().cpu().numpy()

        # Store full predictions for viser visualization if requested
        if return_vggt_predictions:
            predictions["extrinsic"] = pred_extri
            predictions["intrinsic"] = pred_intri
            # Convert predictions to numpy for viser
            vggt_predictions = {}
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    # Convert to float32 first (bfloat16 can't be converted to numpy directly)
                    vggt_predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)
                else:
                    vggt_predictions[key] = predictions[key]
            # Add un-normalized images (0-1 range) for viser
            vggt_predictions["images"] = images

    # Both GT (from TUM loader) and VGGT predictions use w2c (world-to-camera) convention
    # Reference: vggt/utils/pose_enc.py - "representing camera from world transformation"
    pose_convention = 'w2c'

    # Compute ATE with Sim3 alignment
    ate_rmse_sim3, _, aligned_pred_sim3, gt_pos, scale = compute_ate(
        pred_poses, gt_poses, align='sim3', gt_convention=pose_convention, pred_convention=pose_convention)

    # Compute SE3 (no scale)
    ate_rmse_se3, ate_mean_se3, _, _, _ = compute_ate(
        pred_poses, gt_poses, align='se3', gt_convention=pose_convention, pred_convention=pose_convention)
    ate_mean_sim3 = ate_rmse_sim3  # For now, use RMSE as mean approximation

    # RPE (relative pose error)
    rpe_trans, rpe_rot = compute_rpe(pred_poses, gt_poses, delta=1)

    print(f"\n  Results with SE3 alignment (no scale):")
    print(f"    ATE RMSE: {ate_rmse_se3 * 100:.2f} cm")

    print(f"\n  Results with Sim3 alignment (scale={scale:.3f}):")
    print(f"    ATE RMSE: {ate_rmse_sim3 * 100:.2f} cm")

    print(f"\n  Relative Pose Error (δ=1):")
    print(f"    RPE Trans: {rpe_trans * 100:.2f} cm")
    print(f"    RPE Rot: {rpe_rot:.2f}°")

    return {
        'ate_rmse_se3': ate_rmse_se3,
        'ate_mean_se3': ate_mean_se3,
        'ate_rmse_sim3': ate_rmse_sim3,
        'ate_mean_sim3': ate_mean_sim3,
        'rpe_trans': rpe_trans,
        'pose_convention': pose_convention,
        'rpe_rot': rpe_rot,
        'scale': scale,
        'pred_positions': aligned_pred_sim3,
        'gt_positions': gt_pos,
        'vggt_predictions': vggt_predictions,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate VGGT on TUM RGB-D')
    parser.add_argument('--tum_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./tum_eval_output')
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'consecutive'],
                        help='Frame sampling strategy within [start_frame, end_frame]: uniform (default), random, or consecutive')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Starting frame index (default: 0)')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='Ending frame index, inclusive (default: last frame)')
    parser.add_argument('--dryrun', action='store_true',
                        help='Dry run: use GT as prediction to verify evaluation pipeline (skip VGGT inference)')
    parser.add_argument('--viser', action='store_true', default=True,
                        help='Enable viser 3D visualization (default: enabled)')
    parser.add_argument('--no-viser', dest='viser', action='store_false',
                        help='Disable viser 3D visualization')
    parser.add_argument('--viser_port', type=int, default=8080,
                        help='Port for viser server (default: 8080)')
    parser.add_argument('--viser_conf_threshold', type=float, default=25.0,
                        help='Initial confidence threshold percentage for viser (default: 25.0)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("="*60)
    print("VGGT Evaluation on TUM RGB-D")
    print("="*60)
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Frames per trial: {args.num_frames}")
    end_frame_str = str(args.end_frame) if args.end_frame is not None else "last"
    print(f"Frame range: [{args.start_frame}, {end_frame_str}]")
    print(f"Sampling strategy: {args.sampling}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Viser visualization: {'enabled (port ' + str(args.viser_port) + ')' if args.viser else 'disabled'}")
    if args.dryrun:
        print(f"*** DRYRUN MODE: Using GT as prediction (skipping VGGT) ***")

    # Load dataset
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.datasets.tum_rgbd import TUMRGBDDataset

    common_conf = MockCommonConf()
    dataset = TUMRGBDDataset(
        common_conf=common_conf,
        split="test",
        TUM_DIR=args.tum_dir,
        min_num_images=args.num_frames,
    )

    # Load VGGT model (skip in dryrun mode)
    if args.dryrun:
        print("\n[DRYRUN] Skipping VGGT model loading")
        model = None
    else:
        print("\nLoading VGGT model...")
        from vggt.models.vggt import VGGT
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()

    # Run evaluation
    all_results = []
    for trial in range(args.num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{args.num_trials}")
        print("="*60)

        # Request VGGT predictions for viser on the last trial
        return_vggt = args.viser and not args.dryrun and (trial == args.num_trials - 1)
        results = evaluate_sequence(dataset, 0, args.num_frames, model, device, dtype,
                                     sampling=args.sampling, start_frame=args.start_frame,
                                     end_frame=args.end_frame, dryrun=args.dryrun,
                                     return_vggt_predictions=return_vggt)
        all_results.append(results)

    # Summary
    print("\n" + "="*60)
    if args.dryrun:
        print("SUMMARY [DRYRUN - GT vs GT]")
    else:
        print("SUMMARY")
    print("="*60)

    ate_rmses_se3 = [r['ate_rmse_se3'] for r in all_results]
    ate_means_se3 = [r['ate_mean_se3'] for r in all_results]
    ate_rmses_sim3 = [r['ate_rmse_sim3'] for r in all_results]
    ate_means_sim3 = [r['ate_mean_sim3'] for r in all_results]
    rpe_trans = [r['rpe_trans'] for r in all_results]
    rpe_rots = [r['rpe_rot'] for r in all_results]
    scales = [r['scale'] for r in all_results]

    print(f"\n[SE3 Alignment (no scale)]")
    print(f"  ATE RMSE: {np.mean(ate_rmses_se3)*100:.2f} ± {np.std(ate_rmses_se3)*100:.2f} cm")
    print(f"  ATE Mean: {np.mean(ate_means_se3)*100:.2f} ± {np.std(ate_means_se3)*100:.2f} cm")

    print(f"\n[Sim3 Alignment (with scale)]")
    print(f"  ATE RMSE: {np.mean(ate_rmses_sim3)*100:.2f} ± {np.std(ate_rmses_sim3)*100:.2f} cm")
    print(f"  ATE Mean: {np.mean(ate_means_sim3)*100:.2f} ± {np.std(ate_means_sim3)*100:.2f} cm")
    print(f"  Scale factor: {np.mean(scales):.3f} ± {np.std(scales):.3f}")

    print(f"\n[Relative Pose Error]")
    print(f"  RPE Trans: {np.mean(rpe_trans)*100:.2f} ± {np.std(rpe_trans)*100:.2f} cm")
    print(f"  RPE Rot: {np.mean(rpe_rots):.2f} ± {np.std(rpe_rots):.2f}°")

    # Save trajectory plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot last trial
    pred_pos = all_results[-1]['pred_positions']
    gt_pos = all_results[-1]['gt_positions']

    pred_label = 'GT (dryrun)' if args.dryrun else 'VGGT (aligned)'
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'g-', linewidth=2, label='GT')
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', linewidth=2, label=pred_label)

    ax.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], c='g', s=100, marker='^')
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], c='r', s=100, marker='^')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    title_prefix = '[DRYRUN] GT vs GT' if args.dryrun else 'VGGT vs GT'
    ax.set_title(f'{title_prefix} Trajectory\nATE (Sim3): {all_results[-1]["ate_rmse_sim3"]*100:.1f} cm')

    plt.savefig(os.path.join(args.output_dir, 'trajectory_comparison.png'), dpi=150)
    plt.close()
    print(f"\nSaved trajectory plot to: {args.output_dir}/trajectory_comparison.png")

    # Viser 3D visualization
    if args.viser and not args.dryrun:
        vggt_predictions = all_results[-1].get('vggt_predictions')
        if vggt_predictions is not None:
            print(f"\n{'='*60}")
            print("Starting Viser 3D Visualization")
            print("="*60)
            print(f"Open http://localhost:{args.viser_port} in your browser")

            # Import viser_wrapper from demo_viser
            from demo_viser import viser_wrapper

            viser_wrapper(
                vggt_predictions,
                port=args.viser_port,
                init_conf_threshold=args.viser_conf_threshold,
                use_point_map=False,
                background_mode=False,
                mask_sky=False,
                image_folder=None,
            )
        else:
            print("\n[Warning] No VGGT predictions available for viser visualization")
    elif args.viser and args.dryrun:
        print("\n[Note] Viser visualization is not available in dryrun mode")


if __name__ == '__main__':
    main()
