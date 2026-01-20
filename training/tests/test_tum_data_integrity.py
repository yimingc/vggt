#!/usr/bin/env python3
"""
TUM RGB-D Data Integrity Test.

Verifies that all frames in the dataset have:
- Valid RGB images
- Valid depth maps
- Valid GT poses (orthogonal rotation matrices)
- Proper timestamp associations

Usage:
    python training/tests/test_tum_data_integrity.py --tum_dir /path/to/tum
"""

import os
import sys
import argparse
import logging

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)


def check_rotation_matrix(R, tol=1e-4):
    """Check if R is a valid rotation matrix."""
    # Check orthogonality: R @ R.T = I
    ortho_error = np.abs(R @ R.T - np.eye(3)).max()
    # Check determinant = 1
    det_error = abs(np.linalg.det(R) - 1.0)
    return ortho_error < tol and det_error < tol, ortho_error, det_error


def test_data_integrity(tum_dir, sequences=None, batch_size=50):
    """
    Test data integrity for all frames in TUM dataset.

    Args:
        tum_dir: Path to TUM RGB-D sequences
        sequences: List of sequence names (None = auto-detect)
        batch_size: Number of frames to load at once
    """
    from data.datasets.tum_rgbd import (
        TUMRGBDDataset, parse_tum_list_file, associate_timestamps,
        tum_pose_to_extrinsic, TUM_DEPTH_FACTOR
    )

    print("="*70)
    print("TUM RGB-D Data Integrity Test")
    print("="*70)
    print(f"TUM directory: {tum_dir}")

    # Auto-detect sequences
    if sequences is None:
        sequences = []
        for name in os.listdir(tum_dir):
            seq_path = os.path.join(tum_dir, name)
            if os.path.isdir(seq_path):
                if (os.path.exists(os.path.join(seq_path, 'rgb.txt')) and
                    os.path.exists(os.path.join(seq_path, 'depth.txt')) and
                    os.path.exists(os.path.join(seq_path, 'groundtruth.txt'))):
                    sequences.append(name)
        sequences = sorted(sequences)

    print(f"Found {len(sequences)} sequences: {sequences}")

    total_stats = {
        'total_frames': 0,
        'valid_frames': 0,
        'missing_rgb': 0,
        'missing_depth': 0,
        'invalid_depth': 0,
        'missing_pose': 0,
        'invalid_pose': 0,
        'depth_stats': [],
        'pose_errors': [],
    }

    for seq_name in sequences:
        print(f"\n{'='*70}")
        print(f"Sequence: {seq_name}")
        print("="*70)

        seq_path = os.path.join(tum_dir, seq_name)

        # Parse files
        rgb_list = parse_tum_list_file(os.path.join(seq_path, 'rgb.txt'))
        depth_list = parse_tum_list_file(os.path.join(seq_path, 'depth.txt'))
        gt_list = parse_tum_list_file(os.path.join(seq_path, 'groundtruth.txt'))

        print(f"  RGB entries: {len(rgb_list)}")
        print(f"  Depth entries: {len(depth_list)}")
        print(f"  GT entries: {len(gt_list)}")

        # Associate timestamps
        rgb_depth_matches = associate_timestamps(rgb_list, depth_list, max_diff=0.02)
        print(f"  RGB-Depth associations: {len(rgb_depth_matches)}")

        # Build GT lookup
        gt_dict = {}
        for entry in gt_list:
            gt_dict[entry[0]] = entry[1:]
        gt_timestamps = np.array(list(gt_dict.keys()))

        # Test each frame
        seq_stats = {
            'total': len(rgb_depth_matches),
            'valid': 0,
            'missing_rgb': 0,
            'missing_depth': 0,
            'invalid_depth': 0,
            'missing_pose': 0,
            'invalid_pose': 0,
            'depth_min': [],
            'depth_max': [],
            'depth_valid_ratio': [],
            'ortho_errors': [],
        }

        print(f"\n  Testing {len(rgb_depth_matches)} frames...")

        for i, ((rgb_ts, rgb_file), (_, depth_file)) in enumerate(rgb_depth_matches):
            frame_valid = True

            # Check RGB
            rgb_path = os.path.join(seq_path, rgb_file)
            if not os.path.exists(rgb_path):
                seq_stats['missing_rgb'] += 1
                frame_valid = False
            else:
                rgb = cv2.imread(rgb_path)
                if rgb is None:
                    seq_stats['missing_rgb'] += 1
                    frame_valid = False

            # Check Depth
            depth_path = os.path.join(seq_path, depth_file)
            if not os.path.exists(depth_path):
                seq_stats['missing_depth'] += 1
                frame_valid = False
            else:
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    seq_stats['missing_depth'] += 1
                    frame_valid = False
                else:
                    depth_m = depth.astype(np.float32) / TUM_DEPTH_FACTOR
                    valid_mask = depth_m > 0
                    valid_ratio = valid_mask.sum() / depth_m.size

                    if valid_ratio < 0.1:  # Less than 10% valid depth
                        seq_stats['invalid_depth'] += 1
                        frame_valid = False
                    else:
                        seq_stats['depth_min'].append(depth_m[valid_mask].min())
                        seq_stats['depth_max'].append(depth_m[valid_mask].max())
                        seq_stats['depth_valid_ratio'].append(valid_ratio)

            # Check GT pose
            diffs = np.abs(gt_timestamps - rgb_ts)
            min_idx = np.argmin(diffs)

            if diffs[min_idx] > 0.02:  # No pose within 20ms
                seq_stats['missing_pose'] += 1
                frame_valid = False
            else:
                gt_ts = gt_timestamps[min_idx]
                pose_data = gt_dict[gt_ts]
                tx, ty, tz = float(pose_data[0]), float(pose_data[1]), float(pose_data[2])
                qx, qy, qz, qw = float(pose_data[3]), float(pose_data[4]), float(pose_data[5]), float(pose_data[6])

                extri = tum_pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw)
                R = extri[:3, :3]

                is_valid, ortho_err, det_err = check_rotation_matrix(R)
                if not is_valid:
                    seq_stats['invalid_pose'] += 1
                    frame_valid = False
                else:
                    seq_stats['ortho_errors'].append(ortho_err)

            if frame_valid:
                seq_stats['valid'] += 1

            # Progress
            if (i + 1) % 100 == 0 or i == len(rgb_depth_matches) - 1:
                print(f"    Processed {i+1}/{len(rgb_depth_matches)} frames", end='\r')

        print()  # Newline after progress

        # Print sequence summary
        print(f"\n  Sequence Summary:")
        print(f"    Total frames: {seq_stats['total']}")
        print(f"    Valid frames: {seq_stats['valid']} ({100*seq_stats['valid']/seq_stats['total']:.1f}%)")
        print(f"    Missing RGB: {seq_stats['missing_rgb']}")
        print(f"    Missing Depth: {seq_stats['missing_depth']}")
        print(f"    Invalid Depth (<10% valid): {seq_stats['invalid_depth']}")
        print(f"    Missing Pose: {seq_stats['missing_pose']}")
        print(f"    Invalid Pose: {seq_stats['invalid_pose']}")

        if seq_stats['depth_min']:
            print(f"\n  Depth Statistics (valid frames):")
            print(f"    Min depth: {np.min(seq_stats['depth_min']):.3f} m")
            print(f"    Max depth: {np.max(seq_stats['depth_max']):.3f} m")
            print(f"    Valid ratio: {np.mean(seq_stats['depth_valid_ratio'])*100:.1f}% ± {np.std(seq_stats['depth_valid_ratio'])*100:.1f}%")

        if seq_stats['ortho_errors']:
            print(f"\n  Pose Statistics (valid frames):")
            print(f"    Max orthogonality error: {np.max(seq_stats['ortho_errors']):.2e}")
            print(f"    Mean orthogonality error: {np.mean(seq_stats['ortho_errors']):.2e}")

        # Accumulate total stats
        total_stats['total_frames'] += seq_stats['total']
        total_stats['valid_frames'] += seq_stats['valid']
        total_stats['missing_rgb'] += seq_stats['missing_rgb']
        total_stats['missing_depth'] += seq_stats['missing_depth']
        total_stats['invalid_depth'] += seq_stats['invalid_depth']
        total_stats['missing_pose'] += seq_stats['missing_pose']
        total_stats['invalid_pose'] += seq_stats['invalid_pose']
        total_stats['depth_stats'].extend(zip(seq_stats['depth_min'], seq_stats['depth_max'], seq_stats['depth_valid_ratio']))
        total_stats['pose_errors'].extend(seq_stats['ortho_errors'])

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total frames across all sequences: {total_stats['total_frames']}")
    print(f"Valid frames: {total_stats['valid_frames']} ({100*total_stats['valid_frames']/total_stats['total_frames']:.1f}%)")
    print(f"\nIssues found:")
    print(f"  Missing RGB: {total_stats['missing_rgb']}")
    print(f"  Missing Depth: {total_stats['missing_depth']}")
    print(f"  Invalid Depth: {total_stats['invalid_depth']}")
    print(f"  Missing Pose: {total_stats['missing_pose']}")
    print(f"  Invalid Pose: {total_stats['invalid_pose']}")

    if total_stats['depth_stats']:
        depth_mins, depth_maxs, depth_ratios = zip(*total_stats['depth_stats'])
        print(f"\nOverall Depth Statistics:")
        print(f"  Depth range: [{np.min(depth_mins):.3f}, {np.max(depth_maxs):.3f}] m")
        print(f"  Valid depth ratio: {np.mean(depth_ratios)*100:.1f}% ± {np.std(depth_ratios)*100:.1f}%")

    if total_stats['pose_errors']:
        print(f"\nOverall Pose Statistics:")
        print(f"  Max orthogonality error: {np.max(total_stats['pose_errors']):.2e}")

    # Test loading through dataloader
    print("\n" + "="*70)
    print("DATALOADER BATCH TEST")
    print("="*70)

    class MockConf:
        img_size = 518
        patch_size = 14
        debug = False
        training = False
        get_nearby = False  # Sequential loading
        load_depth = True
        inside_random = False
        allow_duplicate_img = False
        landscape_check = False
        rescale = True
        rescale_aug = False
        class augs:
            scales = None

    dataset = TUMRGBDDataset(
        common_conf=MockConf(),
        TUM_DIR=tum_dir,
        sequences=sequences,
        min_num_images=10,
    )

    # Test loading batches sequentially
    seq_name = dataset.sequence_list[0]
    seq_len = len(dataset.data_store[seq_name])

    print(f"\nTesting sequential batch loading for '{seq_name}' ({seq_len} frames)")
    print(f"Batch size: {batch_size}")

    loaded_count = 0
    failed_count = 0

    for start_idx in range(0, seq_len, batch_size):
        end_idx = min(start_idx + batch_size, seq_len)
        ids = list(range(start_idx, end_idx))

        try:
            batch = dataset.get_data(seq_index=0, img_per_seq=len(ids), ids=ids, aspect_ratio=1.0)
            loaded_count += batch['frame_num']
            print(f"  Batch [{start_idx}:{end_idx}]: Loaded {batch['frame_num']} frames ✓", end='\r')
        except Exception as e:
            failed_count += len(ids)
            print(f"  Batch [{start_idx}:{end_idx}]: FAILED - {e}")

    print(f"\n\nDataloader Results:")
    print(f"  Successfully loaded: {loaded_count} frames")
    print(f"  Failed: {failed_count} frames")

    if failed_count == 0 and total_stats['valid_frames'] == total_stats['total_frames']:
        print("\n" + "="*70)
        print("✓ ALL DATA INTEGRITY CHECKS PASSED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠ SOME DATA INTEGRITY ISSUES FOUND")
        print("="*70)

    return total_stats


def main():
    parser = argparse.ArgumentParser(description='TUM RGB-D Data Integrity Test')
    parser.add_argument('--tum_dir', type=str, required=True,
                        help='Path to TUM RGB-D sequences directory')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='Specific sequences to test')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for dataloader test')
    args = parser.parse_args()

    test_data_integrity(args.tum_dir, args.sequences, args.batch_size)


if __name__ == '__main__':
    main()
