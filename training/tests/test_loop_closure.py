"""
Unit tests for loop closure detection and window construction.

These tests verify:
1. find_loop_closure_candidates correctly identifies revisit pairs
2. Camera attitude filtering rejects backwards-facing cameras
3. construct_loop_closure_windows builds valid mixed windows
4. Window deduplication works correctly
5. Edge cases: no candidates, boundary frames, overlapping regions

Run:
    pytest training/tests/test_loop_closure.py -v
"""

import numpy as np
import pytest

# Add project root to path
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'training'))

from training.tests.eval_pgo_uncertainty import (
    find_loop_closure_candidates,
    construct_loop_closure_windows,
)


# =============================================================================
# Helpers to build synthetic GT poses
# =============================================================================

def make_w2c_pose(R, C):
    """Build a 4x4 w2c pose from rotation R and camera center C.

    w2c convention: t = -R @ C
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return T


def rotation_z(angle_deg):
    """Rotation matrix around z-axis."""
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0, 0, 1],
    ])


def rotation_y(angle_deg):
    """Rotation matrix around y-axis."""
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), 0, np.sin(a)],
        [0, 1, 0],
        [-np.sin(a), 0, np.cos(a)],
    ])


def build_loop_trajectory(n_frames=200, loop_at=150, revisit_offset=0.1):
    """Build a trajectory that revisits its start after loop_at frames.

    Frames 0..loop_at-1: move in a circle (radius 2m)
    Frames loop_at..n_frames-1: return near the start

    The camera always faces inward (toward the circle center), so revisit
    frames have similar orientation to early frames.

    Args:
        n_frames: Total number of frames
        loop_at: Frame at which the camera returns near the start
        revisit_offset: Spatial offset in meters for the revisit

    Returns:
        gt_poses: [N, 4, 4] w2c poses
    """
    poses = []
    for i in range(n_frames):
        if i < loop_at:
            # Move along a circle
            angle = 2 * np.pi * i / loop_at
            C = np.array([2 * np.cos(angle), 2 * np.sin(angle), 0.0])
            # Camera faces inward (toward origin)
            R = rotation_z(np.degrees(angle) + 180)
        else:
            # Revisit: near the start with small offset
            frac = (i - loop_at) / max(n_frames - loop_at - 1, 1)
            angle = 2 * np.pi * frac * 0.1  # Small arc near start
            C = np.array([
                2 * np.cos(angle) + revisit_offset,
                2 * np.sin(angle) + revisit_offset,
                0.0
            ])
            R = rotation_z(np.degrees(angle) + 180)
        poses.append(make_w2c_pose(R, C))

    return np.stack(poses, axis=0)


# =============================================================================
# Tests for find_loop_closure_candidates
# =============================================================================

class TestFindLoopClosureCandidates:
    """Tests for find_loop_closure_candidates."""

    def test_basic_detection(self):
        """Candidates are found when frames revisit the same location."""
        gt_poses = build_loop_trajectory(n_frames=200, loop_at=150,
                                          revisit_offset=0.05)
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=0.5,
            max_rotation_deg=60.0
        )
        assert len(candidates) > 0, "Should find at least one LC candidate"
        # All candidates should have temporal gap >= 100
        for fi, fj, dist, rot_deg in candidates:
            assert fj - fi >= 100, f"Temporal gap {fj - fi} < 100"
            assert dist < 0.5, f"Distance {dist} >= 0.5m"
            assert rot_deg < 60.0, f"Rotation {rot_deg} >= 60°"

    def test_sorted_by_distance(self):
        """Candidates are returned sorted by increasing distance."""
        gt_poses = build_loop_trajectory(n_frames=200, loop_at=150,
                                          revisit_offset=0.05)
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=0.5
        )
        if len(candidates) > 1:
            distances = [c[2] for c in candidates]
            assert distances == sorted(distances), "Candidates not sorted by distance"

    def test_temporal_gap_filter(self):
        """No candidates returned when min_temporal_gap exceeds sequence length."""
        gt_poses = build_loop_trajectory(n_frames=200, loop_at=150)
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=500, spatial_threshold=1.0
        )
        assert len(candidates) == 0, "Should find no candidates with huge temporal gap"

    def test_spatial_threshold_filter(self):
        """No candidates when spatial threshold is very small."""
        gt_poses = build_loop_trajectory(n_frames=200, loop_at=150,
                                          revisit_offset=0.5)
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=0.001
        )
        assert len(candidates) == 0, (
            "Should find no candidates with very tight spatial threshold"
        )

    def test_attitude_filter_rejects_opposing_cameras(self):
        """Cameras facing opposite directions are rejected by attitude check."""
        # N=200 with boundary at 100, min_temporal_gap=100 ensures all
        # surviving pairs cross the rotation boundary (no within-group pairs
        # can have gap >= 100 since each group has exactly 100 frames).
        N = 200
        poses = []
        for i in range(N):
            C = np.array([0.0, 0.0, 0.0])  # All at origin (spatially identical)
            if i < 100:
                R = np.eye(3)  # Facing forward
            else:
                R = rotation_y(180)  # Facing backward (180° rotation)
            poses.append(make_w2c_pose(R, C))
        gt_poses = np.stack(poses, axis=0)

        # With default 60° threshold, 180° rotation should be rejected
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=1.0,
            max_rotation_deg=60.0
        )
        assert len(candidates) == 0, (
            f"Should reject 180° rotated cameras, but found {len(candidates)} candidates"
        )

    def test_attitude_filter_accepts_similar_cameras(self):
        """Cameras with small rotation difference pass attitude check."""
        # Same structure as reject test: N=200, boundary at 100, gap=100
        # ensures all surviving pairs cross the boundary.
        N = 200
        poses = []
        for i in range(N):
            C = np.array([0.0, 0.0, 0.0])  # All at origin
            # Small rotation offset: 10° for revisit frames
            if i < 100:
                R = np.eye(3)
            else:
                R = rotation_y(10)  # Only 10° offset
            poses.append(make_w2c_pose(R, C))
        gt_poses = np.stack(poses, axis=0)

        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=1.0,
            max_rotation_deg=60.0
        )
        assert len(candidates) > 0, (
            "Should accept cameras with only 10° rotation difference"
        )
        for _, _, _, rot_deg in candidates:
            assert rot_deg < 60.0
            assert abs(rot_deg - 10.0) < 1.0, f"Expected ~10° rotation, got {rot_deg}°"

    def test_returns_four_tuple(self):
        """Each candidate is a 4-tuple (frame_i, frame_j, distance, rot_deg)."""
        gt_poses = build_loop_trajectory(n_frames=200, loop_at=150,
                                          revisit_offset=0.05)
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=0.5
        )
        for c in candidates:
            assert len(c) == 4, f"Expected 4-tuple, got {len(c)}-tuple"
            fi, fj, dist, rot_deg = c
            assert isinstance(fi, (int, np.integer))
            assert isinstance(fj, (int, np.integer))
            assert isinstance(dist, float)
            assert isinstance(rot_deg, float)
            assert fi < fj, f"Expected fi < fj, got {fi} >= {fj}"

    def test_no_self_pairs(self):
        """No candidate should pair a frame with itself."""
        gt_poses = build_loop_trajectory(n_frames=200, loop_at=150)
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=1, spatial_threshold=10.0,
            max_rotation_deg=180.0
        )
        for fi, fj, _, _ in candidates:
            assert fi != fj, "Should not pair a frame with itself"

    def test_empty_input(self):
        """Empty pose array returns empty candidates."""
        gt_poses = np.zeros((0, 4, 4))
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=10, spatial_threshold=1.0
        )
        assert len(candidates) == 0

    def test_straight_line_no_revisit(self):
        """A straight-line trajectory has no loop closures."""
        N = 300
        poses = []
        for i in range(N):
            C = np.array([i * 0.01, 0.0, 0.0])  # Move 1cm per frame
            R = np.eye(3)
            poses.append(make_w2c_pose(R, C))
        gt_poses = np.stack(poses, axis=0)

        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=0.3
        )
        assert len(candidates) == 0, (
            "Straight-line trajectory should have no loop closures"
        )


# =============================================================================
# Tests for construct_loop_closure_windows
# =============================================================================

class TestConstructLoopClosureWindows:
    """Tests for construct_loop_closure_windows."""

    def _make_candidates(self, pairs, dist=0.1, rot_deg=5.0):
        """Helper to build candidate list from frame pairs."""
        return [(fi, fj, dist, rot_deg) for fi, fj in pairs]

    def test_basic_construction(self):
        """Windows are constructed from candidate pairs."""
        candidates = self._make_candidates([(50, 200)])
        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=16, max_lc_windows=10
        )
        assert len(windows) == 1
        assert len(windows[0]) <= 16
        # Should contain frames near 50 and near 200
        frames = set(windows[0])
        has_near_50 = any(abs(f - 50) < 10 for f in frames)
        has_near_200 = any(abs(f - 200) < 10 for f in frames)
        assert has_near_50, "Window should have frames near first candidate"
        assert has_near_200, "Window should have frames near second candidate"

    def test_window_size_limit(self):
        """Window does not exceed window_size."""
        candidates = self._make_candidates([(50, 400)])
        for ws in [8, 16, 32]:
            windows = construct_loop_closure_windows(
                candidates, seq_len=500, window_size=ws
            )
            for w in windows:
                assert len(w) <= ws, f"Window has {len(w)} frames, exceeds {ws}"

    def test_frames_sorted(self):
        """Frames within each window are sorted ascending."""
        candidates = self._make_candidates([(50, 250), (100, 350)])
        windows = construct_loop_closure_windows(
            candidates, seq_len=500, window_size=16
        )
        for w in windows:
            assert w == sorted(w), f"Window frames not sorted: {w}"

    def test_frames_within_bounds(self):
        """All frame indices are within [0, seq_len)."""
        # Test near boundaries
        candidates = self._make_candidates([(2, 298)])
        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=16
        )
        for w in windows:
            for f in w:
                assert 0 <= f < 300, f"Frame {f} out of bounds [0, 300)"

    def test_no_duplicate_frames(self):
        """No duplicate frame indices within a window."""
        candidates = self._make_candidates([(50, 200)])
        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=16
        )
        for w in windows:
            assert len(w) == len(set(w)), f"Duplicate frames in window: {w}"

    def test_max_lc_windows_respected(self):
        """Number of windows does not exceed max_lc_windows."""
        # Many candidates, limited windows
        pairs = [(i, i + 150) for i in range(0, 100, 10)]
        candidates = self._make_candidates(pairs)
        for max_lc in [1, 3, 5]:
            windows = construct_loop_closure_windows(
                candidates, seq_len=500, window_size=16,
                max_lc_windows=max_lc
            )
            assert len(windows) <= max_lc, (
                f"Got {len(windows)} windows, expected <= {max_lc}"
            )

    def test_deduplication(self):
        """Nearly identical candidates produce only one window."""
        # All candidates are essentially the same pair
        candidates = self._make_candidates([
            (50, 200),
            (51, 201),
            (52, 202),
            (49, 199),
        ])
        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=16,
            overlap_threshold=0.5
        )
        # These should mostly overlap → at most 1-2 windows
        assert len(windows) <= 2, (
            f"Nearly identical candidates produced {len(windows)} windows, expected <=2"
        )

    def test_distinct_pairs_not_deduplicated(self):
        """Well-separated candidates produce distinct windows."""
        candidates = self._make_candidates([
            (50, 200),
            (300, 450),
        ])
        windows = construct_loop_closure_windows(
            candidates, seq_len=500, window_size=16,
            overlap_threshold=0.5
        )
        assert len(windows) == 2, (
            f"Distinct pairs should produce 2 windows, got {len(windows)}"
        )

    def test_empty_candidates(self):
        """Empty candidates list returns empty windows."""
        windows = construct_loop_closure_windows(
            [], seq_len=300, window_size=16
        )
        assert len(windows) == 0

    def test_boundary_frame_0(self):
        """Candidate near frame 0 doesn't produce negative indices."""
        candidates = self._make_candidates([(1, 200)])
        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=16
        )
        for w in windows:
            assert all(f >= 0 for f in w), f"Negative frame index in {w}"

    def test_boundary_frame_end(self):
        """Candidate near the end doesn't exceed seq_len."""
        candidates = self._make_candidates([(50, 298)])
        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=16
        )
        for w in windows:
            assert all(f < 300 for f in w), f"Frame >= seq_len in {w}"

    def test_window_contains_both_regions(self):
        """Each window contains frames from both the first-visit and revisit."""
        candidates = self._make_candidates([(50, 400)])
        windows = construct_loop_closure_windows(
            candidates, seq_len=500, window_size=16
        )
        assert len(windows) == 1
        w = windows[0]
        # The temporal gap between the two regions should be visible
        # (i.e., there should be a large jump somewhere in the sorted frames)
        gaps = [w[i+1] - w[i] for i in range(len(w) - 1)]
        max_gap = max(gaps)
        assert max_gap > 100, (
            f"Expected a large temporal gap in window, max gap was {max_gap}"
        )


# =============================================================================
# Integration test: candidates → windows pipeline
# =============================================================================

class TestLoopClosurePipeline:
    """Integration tests for the full LC detection → window pipeline."""

    def test_end_to_end_with_loop_trajectory(self):
        """Full pipeline: loop trajectory → candidates → windows."""
        gt_poses = build_loop_trajectory(n_frames=300, loop_at=200,
                                          revisit_offset=0.05)
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=0.5,
            max_rotation_deg=60.0
        )
        assert len(candidates) > 0, "Should find LC candidates on loop trajectory"

        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=16, max_lc_windows=5
        )
        assert len(windows) > 0, "Should construct at least one LC window"

        # Verify all windows are valid
        for w in windows:
            assert len(w) <= 16
            assert len(w) == len(set(w))  # No duplicates
            assert w == sorted(w)  # Sorted
            assert all(0 <= f < 300 for f in w)  # In bounds

    def test_no_lc_on_straight_trajectory(self):
        """Straight trajectory: no candidates → no windows."""
        N = 300
        poses = []
        for i in range(N):
            C = np.array([i * 0.01, 0.0, 0.0])
            R = np.eye(3)
            poses.append(make_w2c_pose(R, C))
        gt_poses = np.stack(poses, axis=0)

        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=0.3
        )
        assert len(candidates) == 0

        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=16
        )
        assert len(windows) == 0

    def test_circular_trajectory_many_candidates(self):
        """Circular trajectory: many frames are spatially close to opposite side."""
        N = 400
        poses = []
        radius = 1.0
        for i in range(N):
            angle = 2 * np.pi * i / N
            C = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            # Camera faces center
            R = rotation_z(np.degrees(angle) + 180)
            poses.append(make_w2c_pose(R, C))
        gt_poses = np.stack(poses, axis=0)

        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=0.5,
            max_rotation_deg=90.0
        )
        # On a circle, frames ~N/2 apart are on opposite sides (distance=2*radius)
        # Frames near start and near end are spatially close
        # Should find candidates between frames ~0 and ~350-399
        if len(candidates) > 0:
            windows = construct_loop_closure_windows(
                candidates, seq_len=N, window_size=16, max_lc_windows=5
            )
            for w in windows:
                assert len(w) <= 16
                assert w == sorted(w)

    def test_stationary_camera_all_pairs(self):
        """Stationary camera: all temporally-distant pairs are candidates."""
        N = 250
        poses = []
        for i in range(N):
            C = np.array([0.0, 0.0, 0.0])
            R = np.eye(3)
            poses.append(make_w2c_pose(R, C))
        gt_poses = np.stack(poses, axis=0)

        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=100, spatial_threshold=1.0,
            max_rotation_deg=90.0
        )
        # Stationary: every pair (i, j) with j-i >= 100 should be a candidate
        expected = sum(1 for i in range(N) for j in range(i + 100, N))
        assert len(candidates) == expected, (
            f"Stationary camera: expected {expected} candidates, got {len(candidates)}"
        )


# =============================================================================
# Edge cases and robustness
# =============================================================================

class TestEdgeCases:
    """Edge cases for loop closure functions."""

    def test_single_frame(self):
        """Single frame: no candidates possible."""
        gt_poses = np.eye(4).reshape(1, 4, 4)
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=1, spatial_threshold=10.0
        )
        assert len(candidates) == 0

    def test_two_frames(self):
        """Two frames: only possible if gap >= min_temporal_gap."""
        poses = [make_w2c_pose(np.eye(3), np.zeros(3)) for _ in range(2)]
        gt_poses = np.stack(poses, axis=0)

        # min_temporal_gap=1: should find candidate (0, 1)
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=1, spatial_threshold=10.0,
            max_rotation_deg=180.0
        )
        assert len(candidates) == 1
        assert candidates[0][0] == 0
        assert candidates[0][1] == 1

        # min_temporal_gap=2: no candidate
        candidates = find_loop_closure_candidates(
            gt_poses, min_temporal_gap=2, spatial_threshold=10.0
        )
        assert len(candidates) == 0

    def test_window_size_larger_than_region(self):
        """Window size larger than available frames in one region still works."""
        candidates = [(5, 295)]
        candidates = [(5, 295, 0.1, 5.0)]
        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=64
        )
        # Should still produce a valid window
        if len(windows) > 0:
            w = windows[0]
            assert len(w) <= 64
            assert all(0 <= f < 300 for f in w)
            assert w == sorted(w)

    def test_overlapping_regions_merge(self):
        """When first-visit and revisit regions overlap, frames are deduplicated."""
        # Candidate pair where regions are adjacent
        candidates = [(100, 108, 0.1, 5.0)]  # Only 8 frames apart
        windows = construct_loop_closure_windows(
            candidates, seq_len=300, window_size=16
        )
        if len(windows) > 0:
            w = windows[0]
            assert len(w) == len(set(w)), "Duplicate frames in overlapping window"

    def test_rotation_angle_correctness(self):
        """Verify rotation angle computation is correct for known rotations."""
        N = 250
        test_angles = [0, 15, 30, 45, 90, 120, 179]
        for target_deg in test_angles:
            poses = []
            for i in range(N):
                C = np.zeros(3)
                if i < 120:
                    R = np.eye(3)
                else:
                    R = rotation_y(target_deg)
                poses.append(make_w2c_pose(R, C))
            gt_poses = np.stack(poses, axis=0)

            # Use very permissive thresholds to find all candidates
            candidates = find_loop_closure_candidates(
                gt_poses, min_temporal_gap=100, spatial_threshold=10.0,
                max_rotation_deg=180.0
            )
            if len(candidates) > 0:
                # Check that reported rotation is close to the actual angle
                # Only check pairs where one frame is < 120 and other >= 120
                cross_pairs = [(fi, fj, d, r) for fi, fj, d, r in candidates
                               if fi < 120 and fj >= 120]
                if cross_pairs:
                    rot_deg = cross_pairs[0][3]
                    assert abs(rot_deg - target_deg) < 1.0, (
                        f"Expected ~{target_deg}°, got {rot_deg:.1f}°"
                    )
