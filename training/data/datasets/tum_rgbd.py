# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TUM RGB-D Dataset Loader for VGGT training.

TUM RGB-D dataset format:
- rgb/: RGB images (timestamp.png)
- depth/: Depth images (timestamp.png), 16-bit PNG, depth factor = 5000 (divide to get meters)
- rgb.txt: timestamp filename
- depth.txt: timestamp filename
- groundtruth.txt: timestamp tx ty tz qx qy qz qw

Reference: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
"""

import os
import os.path as osp
import logging
import random

import cv2
import numpy as np

from data.dataset_util import read_image_cv2, threshold_depth_map
from data.base_dataset import BaseDataset


# TUM RGB-D depth factor: depth_in_meters = depth_png_value / 5000
TUM_DEPTH_FACTOR = 5000.0

# TUM RGB-D camera intrinsics (for freiburg1, freiburg2, freiburg3 sequences)
# These are approximate; for precise work, use calibration files
TUM_INTRINSICS = {
    "freiburg1": {
        "fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3
    },
    "freiburg2": {
        "fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7
    },
    "freiburg3": {
        "fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6
    },
}

# Default intrinsics (freiburg3)
DEFAULT_INTRINSICS = TUM_INTRINSICS["freiburg3"]


def parse_tum_list_file(filepath):
    """
    Parse TUM-style list file (rgb.txt, depth.txt, groundtruth.txt).

    Returns:
        list of tuples: [(timestamp, data...), ...]
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            timestamp = float(parts[0])
            data.append((timestamp, *parts[1:]))
    return data


def associate_timestamps(first_list, second_list, max_diff=0.02):
    """
    Associate two lists of timestamps.

    Args:
        first_list: list of (timestamp, data...)
        second_list: list of (timestamp, data...)
        max_diff: maximum time difference for association (seconds)

    Returns:
        list of tuples: [(first_entry, second_entry), ...]
    """
    matches = []
    second_timestamps = np.array([x[0] for x in second_list])

    for first_entry in first_list:
        t1 = first_entry[0]
        diffs = np.abs(second_timestamps - t1)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < max_diff:
            matches.append((first_entry, second_list[min_idx]))

    return matches


def quat_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    TUM uses Hamilton convention with scalar last.
    """
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # Rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def tum_pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw):
    """
    Convert TUM pose to OpenCV extrinsic matrix (world-to-camera).

    TUM groundtruth gives the camera pose in world frame:
    - (tx, ty, tz): camera position in world coordinates
    - (qx, qy, qz, qw): quaternion representing camera-to-world rotation
      i.e., P_world = R_cw @ P_cam + t_cw

    OpenCV extrinsic convention:
    - P_cam = R_wc @ P_world + t_wc
    - where R_wc = R_cw.T and t_wc = -R_wc @ t_cw

    Returns:
        np.ndarray: 3x4 extrinsic matrix [R|t] in OpenCV convention
    """
    # TUM quaternion gives camera-to-world rotation
    R_cw = quat_to_rotation_matrix(qx, qy, qz, qw)  # camera to world
    t_cw = np.array([tx, ty, tz])  # camera position in world

    # Convert to world-to-camera (OpenCV convention)
    R_wc = R_cw.T  # transpose to get world-to-camera
    t_wc = -R_wc @ t_cw

    extrinsic = np.zeros((3, 4), dtype=np.float32)
    extrinsic[:3, :3] = R_wc
    extrinsic[:3, 3] = t_wc

    return extrinsic


class TUMRGBDDataset(BaseDataset):
    """
    TUM RGB-D Dataset for VGGT training.

    Supports multiple TUM sequences in a single dataset instance.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        TUM_DIR: str = None,
        sequences: list = None,
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        max_time_diff: float = 0.02,
        expand_ratio: float = 2.0,
    ):
        """
        Initialize TUM RGB-D Dataset.

        Args:
            common_conf: Configuration object with common settings.
            split: Dataset split ('train' or 'test').
            TUM_DIR: Root directory containing TUM sequences.
            sequences: List of sequence names to use. If None, auto-detect.
            min_num_images: Minimum number of images per sequence.
            len_train: Virtual length of training dataset.
            len_test: Virtual length of test dataset.
            max_time_diff: Maximum timestamp difference for association (seconds).
            expand_ratio: Ratio for nearby frame sampling.
        """
        super().__init__(common_conf=common_conf)

        if TUM_DIR is None:
            raise ValueError("TUM_DIR must be specified.")

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.TUM_DIR = TUM_DIR
        self.max_time_diff = max_time_diff
        self.expand_ratio = expand_ratio
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        else:
            self.len_train = len_test

        # Auto-detect sequences if not provided
        if sequences is None:
            sequences = self._auto_detect_sequences()

        # Load all sequences
        self.data_store = {}
        self.sequence_list = []
        total_frame_num = 0

        for seq_name in sequences:
            seq_data = self._load_sequence(seq_name)
            if seq_data is not None and len(seq_data) >= min_num_images:
                self.data_store[seq_name] = seq_data
                self.sequence_list.append(seq_name)
                total_frame_num += len(seq_data)
                logging.info(f"Loaded TUM sequence: {seq_name} with {len(seq_data)} frames")

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: TUM RGB-D sequences: {self.sequence_list_len}")
        logging.info(f"{status}: TUM RGB-D total frames: {total_frame_num}")
        logging.info(f"{status}: TUM RGB-D dataset length: {len(self)}")

    def _auto_detect_sequences(self):
        """Auto-detect TUM sequences in TUM_DIR."""
        sequences = []
        for name in os.listdir(self.TUM_DIR):
            seq_path = osp.join(self.TUM_DIR, name)
            if osp.isdir(seq_path):
                # Check if it looks like a TUM sequence
                if (osp.exists(osp.join(seq_path, 'rgb.txt')) and
                    osp.exists(osp.join(seq_path, 'depth.txt')) and
                    osp.exists(osp.join(seq_path, 'groundtruth.txt'))):
                    sequences.append(name)
        return sorted(sequences)

    def _get_intrinsics_for_sequence(self, seq_name):
        """Get camera intrinsics based on sequence name."""
        seq_lower = seq_name.lower()
        if 'freiburg1' in seq_lower or 'fr1' in seq_lower:
            params = TUM_INTRINSICS["freiburg1"]
        elif 'freiburg2' in seq_lower or 'fr2' in seq_lower:
            params = TUM_INTRINSICS["freiburg2"]
        elif 'freiburg3' in seq_lower or 'fr3' in seq_lower:
            params = TUM_INTRINSICS["freiburg3"]
        else:
            params = DEFAULT_INTRINSICS

        intrinsic = np.array([
            [params['fx'], 0, params['cx']],
            [0, params['fy'], params['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        return intrinsic

    def _load_sequence(self, seq_name):
        """
        Load a single TUM sequence.

        Returns:
            list of dicts with keys: rgb_path, depth_path, timestamp, extrinsic, intrinsic
        """
        seq_path = osp.join(self.TUM_DIR, seq_name)

        rgb_file = osp.join(seq_path, 'rgb.txt')
        depth_file = osp.join(seq_path, 'depth.txt')
        gt_file = osp.join(seq_path, 'groundtruth.txt')

        if not all(osp.exists(f) for f in [rgb_file, depth_file, gt_file]):
            logging.warning(f"Missing files for sequence: {seq_name}")
            return None

        # Parse files
        rgb_list = parse_tum_list_file(rgb_file)
        depth_list = parse_tum_list_file(depth_file)
        gt_list = parse_tum_list_file(gt_file)

        # Associate RGB with depth
        rgb_depth_matches = associate_timestamps(rgb_list, depth_list, self.max_time_diff)

        # Convert GT to dict for quick lookup
        gt_dict = {}
        for entry in gt_list:
            gt_dict[entry[0]] = entry[1:]  # timestamp -> (tx, ty, tz, qx, qy, qz, qw)

        # Get intrinsics for this sequence
        intrinsic = self._get_intrinsics_for_sequence(seq_name)

        # Build frame list
        frames = []
        for (rgb_ts, rgb_file), (_, depth_file) in rgb_depth_matches:
            # Find closest GT pose
            gt_timestamps = np.array(list(gt_dict.keys()))
            diffs = np.abs(gt_timestamps - rgb_ts)
            min_idx = np.argmin(diffs)

            if diffs[min_idx] > self.max_time_diff:
                continue  # No close enough GT pose

            gt_ts = gt_timestamps[min_idx]
            pose_data = gt_dict[gt_ts]
            tx, ty, tz = float(pose_data[0]), float(pose_data[1]), float(pose_data[2])
            qx, qy, qz, qw = float(pose_data[3]), float(pose_data[4]), float(pose_data[5]), float(pose_data[6])

            extrinsic = tum_pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw)

            frames.append({
                'rgb_path': osp.join(seq_path, rgb_file),
                'depth_path': osp.join(seq_path, depth_file),
                'timestamp': rgb_ts,
                'extrinsic': extrinsic,
                'intrinsic': intrinsic.copy(),
            })

        return frames

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index: Index of the sequence to retrieve.
            img_per_seq: Number of images per sequence.
            seq_name: Name of the sequence.
            ids: Specific frame IDs to retrieve.
            aspect_ratio: Aspect ratio for image processing.

        Returns:
            dict: Batch of data including images, depths, poses, etc.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]
        seq_len = len(metadata)

        if ids is None:
            if self.get_nearby and img_per_seq > 1:
                # Sample nearby frames
                # get_nearby_ids expects ids list with desired length
                start_idx = random.randint(0, seq_len - 1)
                ids = [start_idx] * img_per_seq  # Dummy list with desired length
                ids = self.get_nearby_ids(
                    ids, seq_len, expand_ratio=self.expand_ratio
                )
                ids = ids[:img_per_seq]
            else:
                ids = np.random.choice(
                    seq_len, img_per_seq, replace=self.allow_duplicate_img
                )

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        timestamps = []
        original_sizes = []

        for idx in ids:
            frame = metadata[idx]

            # Load RGB image
            image = read_image_cv2(frame['rgb_path'])
            if image is None:
                logging.warning(f"Failed to load image: {frame['rgb_path']}")
                continue

            # Load depth map
            if self.load_depth:
                depth_map = self._load_tum_depth(frame['depth_path'])
                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            extri_opencv = frame['extrinsic'].copy()
            intri_opencv = frame['intrinsic'].copy()

            # Process image (resize, crop, etc.)
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=frame['rgb_path'],
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            timestamps.append(frame['timestamp'])
            original_sizes.append(original_size)

        batch = {
            "seq_name": "tum_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "timestamps": timestamps,
            "original_sizes": original_sizes,
        }
        return batch

    def _load_tum_depth(self, depth_path):
        """
        Load TUM depth map.

        TUM depth is stored as 16-bit PNG with depth factor 5000.
        depth_in_meters = png_value / 5000
        """
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            logging.warning(f"Failed to load depth: {depth_path}")
            return np.zeros((480, 640), dtype=np.float32)

        # Convert to meters (0 in TUM means invalid/no reading, stays as 0)
        depth = depth.astype(np.float32) / TUM_DEPTH_FACTOR

        return depth
