#!/usr/bin/env python3
"""
Phase 5.9: PGO Evaluation - Prove Uncertainty Value in Optimization.

Evaluates whether learned uncertainty improves pose graph optimization.

Comparison:
- Uniform weights: Λ = I
- Homoscedastic MLE: Λ = diag(exp(-log_var_mle)) with global constant
- Heteroscedastic: Λ = diag(exp(-log_var(x))) per-sample

Usage:
    python training/tests/eval_pgo_uncertainty.py \
        --tum_dir /path/to/tum \
        --tum_sequence rgbd_dataset_freiburg1_desk \
        --uncertainty_checkpoint ./checkpoints/best.pt \
        --window_size 64 \
        --overlap 0.5 \
        --output_dir ./eval_pgo
"""

import os
import sys
import argparse
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from scipy import stats

# Add project root and training dir to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
training_dir = os.path.join(project_root, 'training')
sys.path.insert(0, project_root)
sys.path.insert(0, training_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import tested evaluation functions from eval_vggt_tum
from training.tests.eval_vggt_tum import (
    compute_ate as _compute_ate_impl,
    compute_rpe as _compute_rpe_impl,
    extract_camera_positions,
    umeyama_alignment,
)


# =============================================================================
# Sanity Tests (Run First!)
# =============================================================================

def test_theseus_residual_order():
    """
    Sanity Test A: Verify Theseus Between residual dimension order.

    PyPose uses [trans, rot] = [vx,vy,vz, wx,wy,wz].
    Theseus might use a different order.

    Returns:
        str: "trans_rot" or "rot_trans" or "unknown"
    """
    try:
        import theseus as th
    except ImportError:
        logger.error("Theseus not installed. Run: pip install theseus-ai")
        return "unknown"

    # Note: Theseus SE3 expects [1, 3, 4] tensor (3x4 matrix, not 4x4)
    identity_34 = torch.eye(3, 4).unsqueeze(0)  # [1, 3, 4]
    X_i = th.SE3(tensor=identity_34.clone())
    X_j = th.SE3(tensor=identity_34.clone())

    # Test 1: Pure translation [1, 0, 0]
    Z_trans = identity_34.clone()
    Z_trans[0, 0, 3] = 1.0  # tx = 1
    measurement_trans = th.SE3(tensor=Z_trans)
    cost_trans = th.eb.Between(X_i, X_j, measurement_trans, th.ScaleCostWeight(1.0))
    r_trans = cost_trans.error().squeeze()

    # Test 2: Pure rotation (10° around z-axis)
    angle = np.radians(10)
    R_z = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    Z_rot = identity_34.clone()
    Z_rot[0, :3, :3] = R_z
    measurement_rot = th.SE3(tensor=Z_rot)
    cost_rot = th.eb.Between(X_i, X_j, measurement_rot, th.ScaleCostWeight(1.0))
    r_rot = cost_rot.error().squeeze()

    print("\n" + "=" * 60)
    print("Sanity Test A: Theseus Residual Dimension Order")
    print("=" * 60)
    print(f"Pure translation residual: {r_trans.tolist()}")
    print(f"  [:3] norm: {r_trans[:3].norm():.4f}")
    print(f"  [3:] norm: {r_trans[3:].norm():.4f}")
    print(f"Pure rotation residual: {r_rot.tolist()}")
    print(f"  [:3] norm: {r_rot[:3].norm():.4f}")
    print(f"  [3:] norm: {r_rot[3:].norm():.4f}")

    trans_in_first3 = r_trans[:3].norm() > r_trans[3:].norm()
    rot_in_last3 = r_rot[3:].norm() > r_rot[:3].norm()

    if trans_in_first3 and rot_in_last3:
        print("\n✓ Theseus uses [trans, rot] order - matches PyPose")
        return "trans_rot"
    elif not trans_in_first3 and not rot_in_last3:
        print("\n✗ Theseus uses [rot, trans] order")
        print("  → MUST swap: log_var_theseus = torch.cat([log_var[3:], log_var[:3]], dim=-1)")
        return "rot_trans"
    else:
        print(f"\n? Inconsistent results - investigate manually!")
        return "unknown"


def test_weight_semantics():
    """
    Sanity Test B: Verify whether DiagonalCostWeight expects √Λ or Λ.

    Returns:
        str: "lambda" or "sqrt_lambda" or "unknown"
    """
    try:
        import theseus as th
    except ImportError:
        logger.error("Theseus not installed. Run: pip install theseus-ai")
        return "unknown"

    # Theseus SE3 expects [1, 3, 4] tensor
    identity_34 = torch.eye(3, 4).unsqueeze(0)
    X_i = th.SE3(tensor=identity_34.clone())
    X_j = th.SE3(tensor=identity_34.clone())

    Z = identity_34.clone()
    Z[0, 0, 3] = 0.1  # Small translation
    measurement = th.SE3(tensor=Z)

    # Intended information matrix: Λ = diag([4, 4, 4, 1, 1, 1])
    # We want the objective to be 0.5 * r^T Λ r
    lambda_diag = torch.tensor([[4.0, 4.0, 4.0, 1.0, 1.0, 1.0]])
    sqrt_lambda_diag = torch.sqrt(lambda_diag)  # [2, 2, 2, 1, 1, 1]

    # Get raw residual first (with unit weight)
    unit_weight = th.ScaleCostWeight(1.0)
    cost_unit = th.eb.Between(X_i, X_j, measurement, unit_weight)
    r_raw = cost_unit.error().squeeze()  # Unweighted residual

    # Test 1: Pass Λ to DiagonalCostWeight
    weight_lambda = th.DiagonalCostWeight(lambda_diag)
    cost_lambda = th.eb.Between(X_i, X_j, measurement, weight_lambda)
    obj_lambda = th.Objective()
    obj_lambda.add(cost_lambda)
    obj_val_lambda = obj_lambda.error_metric().sum().item()

    # Test 2: Pass √Λ to DiagonalCostWeight
    weight_sqrt = th.DiagonalCostWeight(sqrt_lambda_diag)
    cost_sqrt = th.eb.Between(X_i, X_j, measurement, weight_sqrt)
    obj_sqrt = th.Objective()
    obj_sqrt.add(cost_sqrt)
    obj_val_sqrt = obj_sqrt.error_metric().sum().item()

    # Expected objective: 0.5 * r^T Λ r
    expected = 0.5 * (r_raw ** 2 * lambda_diag.squeeze()).sum().item()

    print("\n" + "=" * 60)
    print("Sanity Test B: DiagonalCostWeight Semantics")
    print("=" * 60)
    print(f"Raw residual r: {r_raw.tolist()}")
    print(f"Λ = {lambda_diag.squeeze().tolist()}")
    print(f"√Λ = {sqrt_lambda_diag.squeeze().tolist()}")
    print(f"\nExpected 0.5 * r^T Λ r = {expected:.6f}")
    print(f"Theseus with w=Λ:        {obj_val_lambda:.6f}")
    print(f"Theseus with w=√Λ:       {obj_val_sqrt:.6f}")

    err_lambda = abs(obj_val_lambda - expected) / (expected + 1e-12)
    err_sqrt = abs(obj_val_sqrt - expected) / (expected + 1e-12)

    if err_sqrt < 0.01:
        print("\n✓ DiagonalCostWeight expects √Λ (sqrt information)")
        print("  → To achieve 0.5 * r^T Λ r, pass √Λ to DiagonalCostWeight")
        print("  → Use: weight = DiagonalCostWeight(torch.exp(-0.5 * log_var))")
        return "sqrt_lambda"
    elif err_lambda < 0.01:
        print("\n✓ DiagonalCostWeight expects Λ (information matrix)")
        print("  → To achieve 0.5 * r^T Λ r, pass Λ to DiagonalCostWeight")
        print("  → Use: weight = DiagonalCostWeight(torch.exp(-log_var))")
        return "lambda"
    else:
        print(f"\n? Neither matches expected (err_Λ={err_lambda:.2%}, err_√Λ={err_sqrt:.2%})")
        return "unknown"


def run_sanity_tests():
    """Run all sanity tests and return configuration."""
    order = test_theseus_residual_order()
    weight_type = test_weight_semantics()

    print("\n" + "=" * 60)
    print("Sanity Test Summary")
    print("=" * 60)
    print(f"THESEUS_ORDER = '{order}'")
    print(f"THESEUS_WEIGHT = '{weight_type}'")

    if order == "unknown" or weight_type == "unknown":
        print("\n⚠️ WARNING: Some sanity tests failed. Review results before proceeding.")
        return None

    return {"order": order, "weight_type": weight_type}


# =============================================================================
# Data Loading
# =============================================================================

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


def load_model_and_checkpoint(checkpoint_path, device, dtype):
    """Load VGGT model with uncertainty head."""
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device)  # Don't convert to bfloat16 - model handles dtype internally
    model.eval()

    # Load uncertainty head weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = model.state_dict()

        loaded_count = 0
        for name, param in checkpoint['uncertainty_head_state_dict'].items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
                loaded_count += 1

        logger.info(f"Loaded {loaded_count} uncertainty head parameters from {checkpoint_path}")
        logger.info(f"  Checkpoint iteration: {checkpoint['iteration']}")
        logger.info(f"  Checkpoint d²_rot: {checkpoint['d2_rot_mean']:.3f}")
        logger.info(f"  Checkpoint d²_trans: {checkpoint['d2_trans_mean']:.3f}")

        # Return MLE log_var if available
        mle_log_var = checkpoint.get('log_var_mle', None)
    else:
        mle_log_var = None

    return model, mle_log_var


def load_dataset(tum_dir, sequences=None):
    """Load TUM dataset.

    Args:
        tum_dir: Root directory containing TUM sequences.
        sequences: Optional list of sequence names to load. If None, auto-detect all.
    """
    from data.datasets.tum_rgbd import TUMRGBDDataset

    common_conf = MockCommonConf()
    dataset = TUMRGBDDataset(
        common_conf=common_conf,
        split='train',
        TUM_DIR=tum_dir,
        sequences=sequences,
    )

    return dataset


# =============================================================================
# Window Sampling & Edge Generation
# =============================================================================

def generate_windows(seq_len, window_size, overlap=0.5):
    """
    Generate overlapping window indices.

    Args:
        seq_len: Total sequence length
        window_size: Number of frames per window
        overlap: Overlap ratio (0.5 = 50% overlap)

    Returns:
        List of (start_idx, end_idx) tuples
    """
    stride = int(window_size * (1 - overlap))
    stride = max(1, stride)

    windows = []
    start = 0
    while start + window_size <= seq_len:
        windows.append((start, start + window_size))
        start += stride

    # Handle last incomplete window if needed
    if start < seq_len and len(windows) > 0:
        last_start = seq_len - window_size
        if last_start > windows[-1][0]:
            windows.append((last_start, seq_len))

    return windows


def generate_edges_for_window(
    model, dataset, seq_index, window_start, window_size,
    gt_poses, device, dtype, theseus_config, max_dt=None, global_scale=None,
    training_style_sampling=False
):
    """
    Generate STAR edges (anchor→i) for a single window with GT-based scale normalization.

    STAR EDGES vs CONSECUTIVE EDGES:
    - Consecutive (i-1→i): Creates only dt=1 pairs, essentially an odometry chain
    - Star (anchor→i): Creates dt=1..S-1 pairs, enabling loop closures with overlapping windows

    IMPORTANT: This matches the TRAINING semantic - uncertainty head was trained on
    star constraints (T_rel_i = T_0^{-1} @ T_i), not consecutive constraints.

    Each window's predicted poses are scaled using GT translations before
    creating edges. This ensures all windows share a consistent scale when chaining
    via MST, preventing trajectory shape distortion.

    Args:
        model: VGGT model
        dataset: TUM dataset
        seq_index: Sequence index
        window_start: Starting frame index (this frame becomes the anchor)
        window_size: Window size
        gt_poses: GT poses for the entire sequence [N, 4, 4]
        device: torch device
        dtype: torch dtype
        theseus_config: {"order": ..., "weight_type": ...}

    Returns:
        List of edge dicts with 'from', 'to', 'measurement', 'information'
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # Get frames - either consecutive (PGO-style) or training-style (ids=None)
    if training_style_sampling:
        # Use ids=None to let dataset sample frames with varied spacing (same as training)
        batch = dataset.get_data(seq_index=seq_index, img_per_seq=window_size, ids=None, aspect_ratio=1.0)
        # Get the actual frame indices that were sampled
        sampled_ids = list(batch['ids'])
    else:
        # Consecutive frames (original PGO-style)
        sampled_ids = list(range(window_start, window_start + window_size))
        batch = dataset.get_data(seq_index=seq_index, img_per_seq=window_size, ids=sampled_ids, aspect_ratio=1.0)

    # Prepare images (model handles dtype internally via autocast)
    images = np.stack(batch['images'], axis=0)
    images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    images_tensor = torch.from_numpy(images).to(device).unsqueeze(0)

    # Run VGGT
    with torch.no_grad():
        predictions = model(images_tensor)

    # Get pose encoding and log_var
    pose_enc = predictions['pose_enc_list'][-1]  # [1, S, 9]
    log_var = predictions['pose_log_var_list'][-1]  # [1, S, 6]

    # Convert pose encoding to SE3 matrices
    image_hw = images_tensor.shape[-2:]  # (H, W)
    pred_extri, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_hw)  # [1, S, 3, 4], intrinsics

    # Make 4x4 matrices (use float32 for matrix operations)
    T_abs = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(window_size, -1, -1).clone()
    T_abs[:, :3, :] = pred_extri[0].float()  # [S, 4, 4]

    # Get GT poses for the sampled frames (use sampled_ids, not window_start range)
    gt_window = torch.from_numpy(gt_poses[sampled_ids]).float().to(device)  # [S, 4, 4]

    # Extract camera centers from absolute poses (w2c: C = -R.T @ t)
    pred_centers = torch.stack([-T[:3, :3].T @ T[:3, 3] for T in T_abs])  # [S, 3]
    gt_centers = torch.stack([-T[:3, :3].T @ T[:3, 3] for T in gt_window])  # [S, 3]

    # Make relative to anchor (frame 0)
    pred_c_rel = pred_centers - pred_centers[0:1]  # [S, 3]
    gt_c_rel = gt_centers - gt_centers[0:1]  # [S, 3]

    # Fit scale: use global_scale if provided, otherwise compute per-window
    pred_c = pred_c_rel[1:]  # [S-1, 3]
    gt_c = gt_c_rel[1:]       # [S-1, 3]

    # Compute window-specific scale for logging (and use if no global_scale)
    gt_norm = gt_c.norm(dim=-1)
    pred_norm = pred_c.norm(dim=-1)
    valid = gt_norm > 0.02  # 2cm threshold

    if valid.sum() >= 2:
        gt_travel = gt_norm[valid].sum().item()
        pred_travel = pred_norm[valid].sum().item()
        window_scale = gt_travel / max(pred_travel, 1e-6)
        window_scale = max(0.1, min(10.0, window_scale))
    else:
        window_scale = 1.0
        gt_travel = gt_norm.sum().item()
        pred_travel = pred_norm.sum().item()

    # Use global_scale if provided, otherwise use window_scale
    if global_scale is not None:
        scale = global_scale
        logger.info(f"  Window {window_start}: using global_scale={scale:.3f} (window_scale would be {window_scale:.3f})")
    else:
        scale = window_scale
        logger.info(f"  Scale debug: gt_travel={gt_travel:.4f}m, pred_travel={pred_travel:.4f}m, raw_scale={window_scale:.4f}")

    # Apply scale to camera centers, then convert back to w2c translation
    # For w2c: t = -R @ C, so scaled_t = -R @ (scale * C) = scale * t
    # Actually for w2c: C = -R.T @ t, so t = -R @ C
    # If we scale C to sC, then new_t = -R @ (sC) = s * (-R @ C) = s * t
    # So we can just scale t directly!
    T_abs_scaled = T_abs.clone()
    T_abs_scaled[:, :3, 3] = T_abs[:, :3, 3] * scale

    # Debug print
    sigma_trans = torch.exp(0.5 * log_var[0, :, :3]).mean().item()
    gt_travel = gt_c.norm(dim=-1).sum().item()
    pred_travel = pred_c.norm(dim=-1).sum().item()
    if training_style_sampling:
        frame_span = sampled_ids[-1] - sampled_ids[0]
        logger.info(f"Window (training_style): frames={sampled_ids[0]}-{sampled_ids[-1]} (span={frame_span}), scale={scale:.3f}, gt_travel={gt_travel:.3f}m, σ_trans={sigma_trans:.4f}m")
    else:
        logger.info(f"Window {window_start}: scale={scale:.3f}, gt_travel={gt_travel:.3f}m, pred_travel={pred_travel:.3f}m, σ_trans={sigma_trans:.4f}m")

    # Create edges (STAR constraints: anchor → i)
    # This matches training semantic: uncertainty was trained on T_rel = T_0^{-1} @ T_i
    # Using scaled poses ensures consistent scale across windows
    edges = []

    # Use sampled_ids for global frame indices (handles both consecutive and training-style)
    anchor_global = sampled_ids[0]  # Anchor is first frame of window
    T_anchor_inv = torch.inverse(T_abs_scaled[0])  # Inverse of anchor pose

    for i in range(1, window_size):
        # Filter by max_dt if specified (use actual frame distance from sampled_ids)
        dt = abs(sampled_ids[i] - sampled_ids[0])  # actual frame distance
        if max_dt is not None and dt > max_dt:
            continue

        target_global = sampled_ids[i]  # Use actual sampled frame id

        # Star relative pose: Z = T_anchor^{-1} @ T_i (anchor to target)
        Z_star = T_anchor_inv @ T_abs_scaled[i]  # [4, 4]

        # Use per-frame uncertainty directly (SEMANTICALLY CORRECT)
        # The uncertainty head was trained for Log(T_rel_gt^{-1} @ T_rel_pred)
        # where T_rel = T_0^{-1} @ T_i, so log_var[i] is the uncertainty for frame i
        log_var_i = log_var[0, i]  # [6]

        # Adjust based on sanity test results
        if theseus_config["weight_type"] == "lambda":
            info = torch.exp(-log_var_i).cpu()  # [6] = Λ
        else:  # sqrt_lambda
            info = torch.exp(-0.5 * log_var_i).cpu()  # [6] = √Λ

        # Dimension order adjustment
        if theseus_config["order"] == "rot_trans":
            info = torch.cat([info[3:], info[:3]])

        edges.append({
            'from': anchor_global,
            'to': target_global,
            'measurement': Z_star.cpu(),
            'information': info,
        })

    return edges, scale  # Also return the scale used/computed


# =============================================================================
# Graph Diagnostics
# =============================================================================

def print_graph_diagnostics(edges):
    """
    Print graph structure diagnostics: dt histogram and cycle rank.

    For PGO to improve poses, we need:
    - dt not just =1 (should have edges with dt=1..S-1 from star edges)
    - cycle_rank > 0 (graph has loops, not just a tree/chain)

    cycle_rank = m - n + c, where m=edges, n=nodes, c=connected components
    """
    from collections import Counter

    # Collect all unique pairs and their dt
    pair_counts = Counter()
    dt_counts = Counter()

    for e in edges:
        u, v = e['from'], e['to']
        pair = (min(u, v), max(u, v))
        pair_counts[pair] += 1
        dt = abs(v - u)
        dt_counts[dt] += 1

    num_edges = len(edges)
    num_unique_pairs = len(pair_counts)
    nodes = set()
    for e in edges:
        nodes.add(e['from'])
        nodes.add(e['to'])
    num_nodes = len(nodes)

    # Simple cycle_rank (assuming 1 connected component)
    # For graph with loops: m > n-1 (a tree has exactly n-1 edges)
    cycle_rank = num_unique_pairs - num_nodes + 1  # Assuming 1 component

    logger.info("\n" + "=" * 60)
    logger.info("Graph Structure Diagnostics")
    logger.info("=" * 60)
    logger.info(f"  Nodes: {num_nodes}")
    logger.info(f"  Total edges: {num_edges}")
    logger.info(f"  Unique node pairs: {num_unique_pairs}")
    logger.info(f"  Cycle rank: {cycle_rank} (>0 means graph has loops)")

    # dt histogram
    logger.info(f"\n  dt (frame distance) histogram:")
    for dt in sorted(dt_counts.keys()):
        count = dt_counts[dt]
        pct = 100.0 * count / num_edges
        bar = '#' * min(40, int(pct / 2.5))
        logger.info(f"    dt={dt:2d}: {count:4d} ({pct:5.1f}%) {bar}")

    # Warnings
    if cycle_rank <= 0:
        logger.warning("  ⚠️ WARNING: cycle_rank <= 0 (graph is tree/chain, PGO cannot correct drift)")
    if len(dt_counts) == 1 and 1 in dt_counts:
        logger.warning("  ⚠️ WARNING: Only dt=1 edges (consecutive), no loop closures possible")

    logger.info("=" * 60 + "\n")


# =============================================================================
# MST Initialization
# =============================================================================

def initialize_poses_mst(edges):
    """
    Initialize global poses using MST traversal.

    IMPORTANT: MST weights are based on dt (frame distance), NOT predicted uncertainty.
    This ensures fair baseline comparison - init doesn't "leak" predicted info.

    Handles duplicate edges by keeping the one with smallest dt.

    Returns:
        Dict {node_id: SE3 tensor [4, 4]}
    """
    import networkx as nx

    # Step 1: Keep only best edge per pair (prefer smaller dt = shorter baseline)
    best_edges = {}
    for e in edges:
        u, v = e['from'], e['to']
        key = (min(u, v), max(u, v))
        dt = abs(v - u)  # Frame distance
        weight = float(dt)  # Prefer smaller dt for more stable init

        if key not in best_edges or weight < best_edges[key]['weight']:
            best_edges[key] = {
                'weight': weight,
                'src': u,
                'dst': v,
                'Z_fwd': e['measurement'],
                'Z_inv': torch.inverse(e['measurement']),
            }

    logger.info(f"MST init: {len(edges)} edges → {len(best_edges)} unique pairs")

    # Step 2: Build graph
    G = nx.Graph()
    for (u, v), data in best_edges.items():
        G.add_edge(u, v, **data)

    # Step 3: Check connectivity and find starting node
    all_nodes = set(G.nodes())
    if not all_nodes:
        logger.error("No nodes in graph!")
        return {}

    # Use the smallest node id as the root (for consistency)
    root_node = min(all_nodes)

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        logger.warning(f"Graph has {len(components)} connected components!")
        # Use the largest connected component
        largest_comp = max(components, key=len)
        logger.info(f"Using largest component with {len(largest_comp)} nodes")
        G = G.subgraph(largest_comp).copy()
        root_node = min(largest_comp)

    # Step 4: Compute MST
    mst = nx.minimum_spanning_tree(G)

    # Step 5: BFS to initialize poses (starting from root_node)
    poses = {root_node: torch.eye(4)}

    for parent, child in nx.bfs_edges(mst, source=root_node):
        edge_data = mst.edges[parent, child]

        if parent == edge_data['src'] and child == edge_data['dst']:
            poses[child] = poses[parent] @ edge_data['Z_fwd']
        else:
            poses[child] = poses[parent] @ edge_data['Z_inv']

    return poses


# =============================================================================
# PGO with Theseus
# =============================================================================

def to_34(mat_44):
    """Convert 4x4 matrix to 3x4 for Theseus SE3."""
    if isinstance(mat_44, torch.Tensor):
        return mat_44[:3, :]
    else:
        return mat_44[:3, :]


def to_44(mat_34):
    """Convert 3x4 matrix back to 4x4."""
    if isinstance(mat_34, torch.Tensor):
        result = torch.eye(4, dtype=mat_34.dtype, device=mat_34.device)
        result[:3, :] = mat_34
    else:
        result = np.eye(4, dtype=mat_34.dtype)
        result[:3, :] = mat_34
    return result


def run_pgo_theseus(edges, initial_poses, weight_mode='predicted', log_var_mle=None,
                    theseus_config=None, use_robust=False, oracle_info=None):
    """
    Run PGO using Theseus.

    Args:
        edges: List of edge dicts
        initial_poses: Dict {node_id: [4, 4] tensor}
        weight_mode: 'uniform', 'mle', 'predicted', 'oracle_isotropic', or 'oracle_binned'
        log_var_mle: [6] tensor, required if weight_mode='mle'
        theseus_config: {"order": ..., "weight_type": ...}
        use_robust: Whether to use Huber robust kernel
        oracle_info: Dict with oracle weights, required for oracle modes:
            - 'oracle_weights': [N, 6] precomputed weights (sqrt_lambda or lambda based on theseus_config)

    Returns:
        optimized_poses: Dict {node_id: [4, 4] tensor}
        info: Optimization info
    """
    import theseus as th

    # Print edge statistics
    from collections import Counter
    pair_counts = Counter((e['from'], e['to']) for e in edges)
    logger.info(f"PGO: {len(edges)} edges, {len(pair_counts)} unique pairs")
    max_count = max(pair_counts.values())
    if max_count > 10:
        logger.warning(f"Some pairs have {max_count} edges!")

    # Create optimization variables (Theseus SE3 uses 3x4 matrices)
    node_ids = sorted(initial_poses.keys())
    root_node = node_ids[0]  # Use first node as gauge fix
    poses = {}
    for i in node_ids:
        pose_34 = to_34(initial_poses[i]).unsqueeze(0).float()
        poses[i] = th.SE3(tensor=pose_34, name=f"pose_{i}")

    # Fix root pose (gauge fix) using a strong prior
    # th.eb.Local computes: error = Log(inv(target) @ var)
    identity_34 = torch.eye(3, 4).unsqueeze(0)
    prior_target = th.SE3(tensor=identity_34, name=f"pose_{root_node}_prior")
    prior_weight = th.DiagonalCostWeight(torch.ones(1, 6) * 1e8)  # Very high weight
    prior_cost = th.eb.Local(poses[root_node], prior_target, prior_weight, name="gauge_fix")
    objective = th.Objective()
    objective.add(prior_cost)

    # Prepare MLE weight if needed
    if weight_mode == 'mle' and log_var_mle is not None:
        if theseus_config["weight_type"] == "lambda":
            lambda_mle = torch.exp(-log_var_mle)
        else:
            lambda_mle = torch.exp(-0.5 * log_var_mle)

        if theseus_config["order"] == "rot_trans":
            lambda_mle = torch.cat([lambda_mle[3:], lambda_mle[:3]])

        lambda_mle = lambda_mle.unsqueeze(0)

    # Create cost functions (add to objective that already has gauge prior)
    # Filter edges to only include those where both nodes are in poses (handles disconnected graphs)
    valid_nodes = set(node_ids)
    edges_used = 0
    for idx, e in enumerate(edges):
        if e['from'] not in valid_nodes or e['to'] not in valid_nodes:
            continue  # Skip edges to nodes not in the graph
        edges_used += 1

        # Convert 4x4 measurement to 3x4 for Theseus
        meas_34 = to_34(e['measurement']).unsqueeze(0).float()
        measurement = th.SE3(tensor=meas_34, name=f"Z_{idx}")

        if weight_mode == 'uniform':
            weight = th.DiagonalCostWeight(torch.ones(1, 6))
        elif weight_mode == 'mle':
            weight = th.DiagonalCostWeight(lambda_mle)
        elif weight_mode in ['oracle_isotropic', 'oracle_binned']:
            # Oracle weighting: use precomputed weights from oracle_info
            # Weights are already in correct format (sqrt_lambda or lambda based on theseus_config)
            oracle_weight = oracle_info['oracle_weights'][idx]  # [6] numpy array
            weight = th.DiagonalCostWeight(torch.tensor(oracle_weight, dtype=torch.float32).unsqueeze(0))
        else:  # 'predicted'
            weight = th.DiagonalCostWeight(e['information'].unsqueeze(0))

        cost = th.eb.Between(
            poses[e['from']],
            poses[e['to']],
            measurement,
            weight,
        )

        if use_robust:
            # Wrap with Huber robust kernel
            cost = th.RobustCostFunction(cost, th.HuberLoss, {"threshold": 1.0})

        objective.add(cost)

    # Build optimizer (use dense solver - sparse cholmod not available)
    linear_solver_cls = th.CholeskyDenseSolver
    logger.debug("Using CholeskyDenseSolver")

    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=50,
        step_size=1.0,
        linear_solver_cls=linear_solver_cls,
    )

    layer = th.TheseusLayer(optimizer)

    # Initial values (exclude the fixed root node)
    input_tensors = {f"pose_{i}": poses[i].tensor for i in node_ids if i != root_node}

    # Optimize
    with torch.no_grad():
        solution, info = layer.forward(input_tensors)

    # Get initial objective from optimization info (if available)
    # Theseus stores error history in info
    init_obj = None
    if hasattr(info, 'err_history') and info.err_history is not None and len(info.err_history) > 0:
        init_obj = info.err_history[0].item() if hasattr(info.err_history[0], 'item') else info.err_history[0]

    # Extract optimized poses (convert 3x4 back to 4x4)
    optimized_poses = {root_node: to_44(poses[root_node].tensor.squeeze(0))}
    for i in node_ids:
        if i != root_node:
            optimized_poses[i] = to_44(solution[f"pose_{i}"].squeeze(0))

    # Get final objective value
    final_obj = objective.error_metric().sum().item()

    # If we couldn't get init_obj from history, estimate from final (not ideal)
    if init_obj is None:
        init_obj = final_obj  # Fallback: assume no change (conservative)
        logger.debug("Could not determine initial objective from optimization info")

    return optimized_poses, {'objective': final_obj, 'init_objective': init_obj, 'info': info}


# =============================================================================
# Evaluation
# =============================================================================

def poses_dict_to_array(poses_dict):
    """Convert poses dict to numpy array."""
    max_id = max(poses_dict.keys())
    poses = np.zeros((max_id + 1, 4, 4))
    for i, pose in poses_dict.items():
        if isinstance(pose, torch.Tensor):
            poses[i] = pose.numpy()
        else:
            poses[i] = pose
    return poses


def se3_alignment(pred_poses, gt_poses):
    """
    Align predicted poses to GT using SE3 Procrustes (rotation + translation, no scale).

    Uses camera centers for Procrustes alignment.

    Args:
        pred_poses: [N, 4, 4] predicted poses (w2c or c2w, assumed same as GT)
        gt_poses: [N, 4, 4] ground truth poses

    Returns:
        aligned_poses: [N, 4, 4]
    """
    # Extract camera centers: C = -R^T @ t (for w2c) or just t (for c2w)
    # Try both conventions and pick the one with lower error

    # Assume poses are [R | t] where t is translation
    # For w2c: camera center = -R.T @ t
    # For c2w: camera center = t

    # First try: assume w2c (camera center = -R.T @ t)
    pred_centers_w2c = np.array([-pose[:3, :3].T @ pose[:3, 3] for pose in pred_poses])
    gt_centers_w2c = np.array([-pose[:3, :3].T @ pose[:3, 3] for pose in gt_poses])

    # Second try: assume c2w (camera center = t directly)
    pred_centers_c2w = np.array([pose[:3, 3] for pose in pred_poses])
    gt_centers_c2w = np.array([pose[:3, 3] for pose in gt_poses])

    # Procrustes alignment (R @ pred + t = gt)
    def procrustes_align(pred_pts, gt_pts):
        # Center the points
        pred_mean = pred_pts.mean(axis=0)
        gt_mean = gt_pts.mean(axis=0)
        pred_centered = pred_pts - pred_mean
        gt_centered = gt_pts - gt_mean

        # SVD for optimal rotation
        H = pred_centered.T @ gt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = gt_mean - R @ pred_mean
        return R, t

    # Try w2c alignment
    R_w2c, t_w2c = procrustes_align(pred_centers_w2c, gt_centers_w2c)
    aligned_w2c = R_w2c @ pred_centers_w2c.T + t_w2c[:, None]
    err_w2c = np.linalg.norm(aligned_w2c.T - gt_centers_w2c)

    # Try c2w alignment
    R_c2w, t_c2w = procrustes_align(pred_centers_c2w, gt_centers_c2w)
    aligned_c2w = R_c2w @ pred_centers_c2w.T + t_c2w[:, None]
    err_c2w = np.linalg.norm(aligned_c2w.T - gt_centers_c2w)

    # Use the better alignment
    if err_w2c < err_c2w:
        R_align, t_align = R_w2c, t_w2c
        logger.debug(f"Using w2c alignment (err={err_w2c:.4f} < {err_c2w:.4f})")
    else:
        R_align, t_align = R_c2w, t_c2w
        logger.debug(f"Using c2w alignment (err={err_c2w:.4f} < {err_w2c:.4f})")

    # Apply alignment to poses
    aligned = np.zeros_like(pred_poses)
    for i in range(len(pred_poses)):
        aligned[i, :3, :3] = R_align @ pred_poses[i, :3, :3]
        aligned[i, :3, 3] = R_align @ pred_poses[i, :3, 3] + t_align
        aligned[i, 3, 3] = 1.0

    return aligned


def compute_ate(pred_poses, gt_poses):
    """Compute Absolute Trajectory Error (translation and rotation)."""
    # Translation error
    trans_errors = []
    rot_errors = []

    for i in range(len(pred_poses)):
        # Translation error
        t_pred = pred_poses[i, :3, 3]
        t_gt = gt_poses[i, :3, 3]
        trans_errors.append(np.linalg.norm(t_pred - t_gt))

        # Rotation error
        R_pred = pred_poses[i, :3, :3]
        R_gt = gt_poses[i, :3, :3]
        R_err = R_pred @ R_gt.T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle))

    return np.mean(trans_errors), np.mean(rot_errors)


def compute_rpe(pred_poses, gt_poses, delta=1):
    """Compute Relative Pose Error."""
    trans_errors = []
    rot_errors = []

    for i in range(len(pred_poses) - delta):
        # Predicted relative pose
        T_pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + delta]

        # GT relative pose
        T_gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]

        # Error
        T_err = np.linalg.inv(T_pred_rel) @ T_gt_rel
        trans_errors.append(np.linalg.norm(T_err[:3, 3]))

        R_err = T_err[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle))

    return np.mean(trans_errors), np.mean(rot_errors)


def compute_gt_residuals_theseus(edges, gt_poses):
    """
    Compute GT residuals using Theseus Between cost.error() for coordinate consistency.

    This ensures the residual is computed in the exact same way as during PGO optimization,
    avoiding any coordinate system mismatches.

    Args:
        edges: List of edge dicts with 'from', 'to', 'measurement'
        gt_poses: [N, 4, 4] numpy array of GT poses

    Returns:
        residuals_6d: [num_edges, 6] numpy array, Theseus residual order [trans, rot]
        dt_list: [num_edges] numpy array of frame distances
    """
    import theseus as th

    residuals_6d = []
    dt_list = []

    for e in edges:
        i, j = e['from'], e['to']

        # GT poses as Theseus SE3 (3x4 format)
        gt_i_34 = torch.from_numpy(gt_poses[i, :3, :]).unsqueeze(0).float()
        gt_j_34 = torch.from_numpy(gt_poses[j, :3, :]).unsqueeze(0).float()
        T_gt_i = th.SE3(tensor=gt_i_34)
        T_gt_j = th.SE3(tensor=gt_j_34)

        # Measurement (predicted relative pose)
        meas_34 = e['measurement'][:3, :].unsqueeze(0).float()
        Z_pred = th.SE3(tensor=meas_34)

        # Compute residual using Theseus Between
        # Between computes: error = Log(Z^{-1} @ T_i^{-1} @ T_j)
        unit_weight = th.ScaleCostWeight(1.0)
        cost = th.eb.Between(T_gt_i, T_gt_j, Z_pred, unit_weight)
        r = cost.error().squeeze().numpy()  # [6] in Theseus order [trans, rot]

        residuals_6d.append(r)
        dt_list.append(abs(j - i))

    return np.array(residuals_6d), np.array(dt_list)


def compute_oracle_weights_binned(residuals_6d, dt_list, theseus_config):
    """
    Compute Oracle-B weights: binned covariance by dt.

    This represents "if we knew the true noise model as a function of dt".

    For each dt bin:
        σ_t²(dt) = mean(r_t²)  # empirical variance of translation residual
        σ_R²(dt) = mean(r_R²)  # empirical variance of rotation residual
        λ_t = 1/σ_t², λ_R = 1/σ_R²

    Args:
        residuals_6d: [N, 6] numpy array of GT residuals [trans, rot]
        dt_list: [N] numpy array of frame distances
        theseus_config: {"order": ..., "weight_type": ...}

    Returns:
        weights: [N, 6] numpy array of weights (sqrt_lambda if theseus expects sqrt)
        bin_stats: dict with per-bin statistics for logging
    """
    # Define dt bins
    dt_bins = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)]

    # Compute per-bin statistics
    bin_stats = {}
    dt_to_bin = {}

    for dt_min, dt_max in dt_bins:
        mask = (dt_list >= dt_min) & (dt_list <= dt_max)
        if mask.sum() == 0:
            continue

        r_trans = residuals_6d[mask, :3]  # [n, 3]
        r_rot = residuals_6d[mask, 3:]    # [n, 3]

        # Empirical variance (per-dimension, then average)
        # Or use norm² for isotropic
        sigma_t_sq = np.mean(r_trans ** 2)  # scalar
        sigma_R_sq = np.mean(r_rot ** 2)    # scalar

        # Add floor to prevent extreme weights
        sigma_t_sq = max(sigma_t_sq, 1e-4)  # ~1cm floor
        sigma_R_sq = max(sigma_R_sq, 1e-4)  # ~0.01 rad floor

        lambda_t = 1.0 / sigma_t_sq
        lambda_R = 1.0 / sigma_R_sq

        bin_stats[(dt_min, dt_max)] = {
            'n': mask.sum(),
            'sigma_t': np.sqrt(sigma_t_sq),
            'sigma_R': np.sqrt(sigma_R_sq),
            'lambda_t': lambda_t,
            'lambda_R': lambda_R,
        }

        # Map dt values in this bin
        for dt in range(dt_min, dt_max + 1):
            dt_to_bin[dt] = (dt_min, dt_max)

    # Assign weights to each edge
    weights = np.zeros((len(dt_list), 6))
    for idx, dt in enumerate(dt_list):
        # Find the bin for this dt
        bin_key = dt_to_bin.get(dt)
        if bin_key is None:
            # dt not in any bin, use global statistics
            sigma_t_sq = np.mean(residuals_6d[:, :3] ** 2)
            sigma_R_sq = np.mean(residuals_6d[:, 3:] ** 2)
            lambda_t = 1.0 / max(sigma_t_sq, 1e-4)
            lambda_R = 1.0 / max(sigma_R_sq, 1e-4)
        else:
            lambda_t = bin_stats[bin_key]['lambda_t']
            lambda_R = bin_stats[bin_key]['lambda_R']

        # Isotropic: same weight for all 3 trans dims, same for all 3 rot dims
        # Theseus order: [trans, rot]
        weights[idx, :3] = lambda_t
        weights[idx, 3:] = lambda_R

    # Convert to sqrt_lambda if needed
    if theseus_config["weight_type"] == "sqrt_lambda":
        weights = np.sqrt(weights)

    return weights, bin_stats


def compute_oracle_weights_isotropic(residuals_6d, dt_list, theseus_config):
    """
    Compute Oracle weights with trans/rot isotropic + σ_floor.

    Instead of per-dimension λ = 1/r², use:
        r_t = ||r[:3]||, r_R = ||r[3:]||
        λ_t = 1/(r_t² + σ_floor_t²)
        λ_R = 1/(r_R² + σ_floor_R²)

    σ_floor uses median of residuals to avoid extreme weights.

    Args:
        residuals_6d: [N, 6] numpy array of GT residuals [trans, rot]
        dt_list: [N] numpy array of frame distances
        theseus_config: {"order": ..., "weight_type": ...}

    Returns:
        weights: [N, 6] numpy array of weights (sqrt_lambda if theseus expects sqrt)
    """
    # Compute norms
    r_t_norms = np.linalg.norm(residuals_6d[:, :3], axis=1)  # [N]
    r_R_norms = np.linalg.norm(residuals_6d[:, 3:], axis=1)  # [N]

    # Floor using median (avoids extreme weights while being data-driven)
    sigma_floor_t = np.median(r_t_norms)
    sigma_floor_R = np.median(r_R_norms)

    # Ensure minimum floor
    sigma_floor_t = max(sigma_floor_t, 0.01)  # 1cm minimum
    sigma_floor_R = max(sigma_floor_R, 0.01)  # ~0.6° minimum

    logger.info(f"Oracle isotropic: σ_floor_t={sigma_floor_t*100:.1f}cm, σ_floor_R={np.degrees(sigma_floor_R):.1f}°")

    # Compute isotropic weights
    lambda_t = 1.0 / (r_t_norms ** 2 + sigma_floor_t ** 2)  # [N]
    lambda_R = 1.0 / (r_R_norms ** 2 + sigma_floor_R ** 2)  # [N]

    # Build 6D weight (isotropic per group)
    weights = np.zeros((len(dt_list), 6))
    weights[:, :3] = lambda_t[:, np.newaxis]  # Same for all 3 trans dims
    weights[:, 3:] = lambda_R[:, np.newaxis]  # Same for all 3 rot dims

    # Convert to sqrt_lambda if needed
    if theseus_config["weight_type"] == "sqrt_lambda":
        weights = np.sqrt(weights)

    # Log weight statistics
    logger.info(f"Oracle isotropic weights: λ_t range=[{lambda_t.min():.1f}, {lambda_t.max():.1f}], "
                f"λ_R range=[{lambda_R.min():.1f}, {lambda_R.max():.1f}]")

    return weights


def compute_pre_opt_correlation(edges, gt_poses, theseus_config):
    """
    Compute correlation between GT residual and predicted sigma.

    Uses Theseus cost.error() for GT residuals to ensure coordinate consistency.

    Also computes d² calibration statistics bucketed by dt (frame distance).

    d² = Σ_k (r_k² * λ_k) should be ~6 for 6-DoF SE(3) if calibrated.
    For separate rot/trans: d²_rot ~ 3, d²_trans ~ 3.

    This validates that uncertainty predicts edge quality.
    """
    # Compute GT residuals using Theseus for coordinate consistency
    gt_residuals_6d, dt_list = compute_gt_residuals_theseus(edges, gt_poses)

    # Extract trans/rot norms
    gt_residuals_trans = np.linalg.norm(gt_residuals_6d[:, :3], axis=1)  # [N]
    gt_residuals_rot = np.linalg.norm(gt_residuals_6d[:, 3:], axis=1)    # [N]

    # Extract predicted information from edges
    pred_sigmas = []
    pred_info_6d = []

    for e in edges:
        # Predicted information (in Theseus order: [trans, rot])
        info = e['information'].numpy()  # Already in Theseus order
        # Convert sqrt_lambda to lambda for d² computation
        if theseus_config["weight_type"] == "sqrt_lambda":
            lambda_6d = info ** 2
            sigma = 1.0 / (info + 1e-12)
        else:
            lambda_6d = info
            sigma = 1.0 / np.sqrt(info + 1e-12)

        sigma_trans = sigma[:3].mean()  # Translation sigma

        pred_sigmas.append(sigma_trans)
        pred_info_6d.append(lambda_6d)

    # Convert to numpy arrays
    pred_sigmas = np.array(pred_sigmas)
    pred_info_6d = np.array(pred_info_6d)  # [N, 6]

    # Spearman correlation
    corr, pval = stats.spearmanr(gt_residuals_trans, pred_sigmas)

    logger.info(f"\nPre-opt Edge Diagnostic:")
    logger.info(f"  Spearman(|r_gt|, σ_pred) = {corr:.3f} (p={pval:.4f})")
    logger.info(f"  Expected: positive correlation")
    logger.info(f"")
    logger.info(f"  GT residual (trans): mean={gt_residuals_trans.mean()*100:.2f}cm, "
                f"p50={np.percentile(gt_residuals_trans, 50)*100:.2f}cm, "
                f"p90={np.percentile(gt_residuals_trans, 90)*100:.2f}cm, "
                f"max={gt_residuals_trans.max()*100:.2f}cm")
    logger.info(f"  GT residual (rot): mean={np.degrees(gt_residuals_rot.mean()):.2f}°, "
                f"p50={np.degrees(np.percentile(gt_residuals_rot, 50)):.2f}°, "
                f"p90={np.degrees(np.percentile(gt_residuals_rot, 90)):.2f}°, "
                f"max={np.degrees(gt_residuals_rot.max()):.2f}°")
    logger.info(f"  Pred σ_trans: mean={pred_sigmas.mean()*100:.2f}cm, "
                f"p50={np.percentile(pred_sigmas, 50)*100:.2f}cm, "
                f"p90={np.percentile(pred_sigmas, 90)*100:.2f}cm, "
                f"max={pred_sigmas.max()*100:.2f}cm")

    # Compute predicted information (√λ) statistics
    all_info = [e['information'] for e in edges]
    info_trans = np.array([i[:3].mean().item() for i in all_info])  # Translation √λ
    info_rot = np.array([i[3:].mean().item() for i in all_info])    # Rotation √λ
    logger.info(f"  Pred √λ_trans: mean={info_trans.mean():.1f}, "
                f"p50={np.percentile(info_trans, 50):.1f}, "
                f"p90={np.percentile(info_trans, 90):.1f}, "
                f"max={info_trans.max():.1f}")
    logger.info(f"  Pred √λ_rot: mean={info_rot.mean():.1f}, "
                f"p50={np.percentile(info_rot, 50):.1f}, "
                f"p90={np.percentile(info_rot, 90):.1f}, "
                f"max={info_rot.max():.1f}")

    # ==========================================================================
    # d² Calibration Verification
    # ==========================================================================
    # d² = Σ_k (r_k² * λ_k) should be ~6 for 6-DoF SE(3) if calibrated
    # For separate rot/trans: d²_trans ~ 3, d²_rot ~ 3
    logger.info(f"")
    logger.info(f"  d² Calibration Verification (expect ~6 for 6-DoF, ~3 for trans/rot each):")

    d2_full = (gt_residuals_6d ** 2 * pred_info_6d).sum(axis=1)  # [N]
    d2_trans = (gt_residuals_6d[:, :3] ** 2 * pred_info_6d[:, :3]).sum(axis=1)  # [N]
    d2_rot = (gt_residuals_6d[:, 3:] ** 2 * pred_info_6d[:, 3:]).sum(axis=1)  # [N]

    logger.info(f"    Overall: d²_full  mean={d2_full.mean():.2f} (expect ~6), "
                f"p50={np.percentile(d2_full, 50):.2f}, "
                f"p95={np.percentile(d2_full, 95):.2f}")
    logger.info(f"    Trans:   d²_trans mean={d2_trans.mean():.2f} (expect ~3), "
                f"p50={np.percentile(d2_trans, 50):.2f}, "
                f"p95={np.percentile(d2_trans, 95):.2f}")
    logger.info(f"    Rot:     d²_rot   mean={d2_rot.mean():.2f} (expect ~3), "
                f"p50={np.percentile(d2_rot, 50):.2f}, "
                f"p95={np.percentile(d2_rot, 95):.2f}")

    # Breakdown by dt (frame distance)
    logger.info(f"")
    logger.info(f"  Breakdown by dt (frame distance):")
    dt_ranges = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64)]
    for dt_min, dt_max in dt_ranges:
        mask = (dt_list >= dt_min) & (dt_list <= dt_max)
        if mask.sum() > 0:
            logger.info(f"    dt={dt_min:2d}-{dt_max:2d}: n={mask.sum():4d}, "
                       f"r_trans={gt_residuals_trans[mask].mean()*100:5.1f}cm, "
                       f"r_rot={np.degrees(gt_residuals_rot[mask].mean()):5.2f}°, "
                       f"σ_trans={pred_sigmas[mask].mean()*100:5.2f}cm, "
                       f"d²={d2_full[mask].mean():5.1f} (p95={np.percentile(d2_full[mask], 95):5.1f})")

    return corr, pval, {
        'gt_residuals_trans': gt_residuals_trans,
        'gt_residuals_rot': gt_residuals_rot,
        'gt_residuals_6d': gt_residuals_6d,
        'pred_info_6d': pred_info_6d,
        'pred_sigmas': pred_sigmas,
        'dt_list': dt_list,
        'd2_full': d2_full,
        'd2_trans': d2_trans,
        'd2_rot': d2_rot,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PGO Evaluation for Uncertainty')
    parser.add_argument('--tum_dir', type=str, required=True, help='Path to TUM root directory')
    parser.add_argument('--tum_sequence', type=str, default=None,
                        help='Name of the specific TUM sequence to evaluate (e.g. rgbd_dataset_freiburg1_desk). '
                             'If not set, evaluates the first auto-detected sequence.')
    parser.add_argument('--uncertainty_checkpoint', type=str, required=True, help='Uncertainty head checkpoint')
    parser.add_argument('--window_size', type=int, default=64, help='Window size (default: 64 frames)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--output_dir', type=str, default='./eval_pgo', help='Output directory')
    parser.add_argument('--robust', action='store_true', help='Use Huber robust kernel')
    parser.add_argument('--sanity_only', action='store_true', help='Only run sanity tests')
    parser.add_argument('--max_windows', type=int, default=None, help='Maximum number of windows (for debugging)')
    parser.add_argument('--single_window_vggt_only', action='store_true',
                        help='Evaluate single window with VGGT only (no PGO, for quick debugging)')
    parser.add_argument('--max_dt', type=int, default=None,
                        help='Max frame distance (dt) for edges. Limits to short baselines where uncertainty is more accurate.')
    parser.add_argument('--global_scale', action='store_true',
                        help='Use a single global scale (from first window) for all windows instead of per-window scales.')
    parser.add_argument('--training_style', action='store_true',
                        help='Use training-style frame sampling (ids=None with get_nearby=True, varied spacing) instead of consecutive frames.')
    parser.add_argument('--oracle', action='store_true',
                        help='[DEPRECATED] Old oracle mode with per-dim 1/r². Use --oracle_isotropic or --oracle_binned instead.')
    parser.add_argument('--oracle_isotropic', action='store_true',
                        help='Oracle with trans/rot isotropic weights + σ_floor (median). '
                             'λ_t = 1/(||r_t||² + σ_floor²), avoids extreme per-dim weights.')
    parser.add_argument('--oracle_binned', action='store_true',
                        help='Oracle with binned covariance by dt. '
                             'Each dt bin uses empirical σ²(dt) = E[r²]. Best "upper bound" estimate.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run sanity tests
    theseus_config = run_sanity_tests()

    if args.sanity_only:
        return

    if theseus_config is None:
        logger.error("Sanity tests failed. Fix issues before proceeding.")
        return

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    # Load model and data
    logger.info("\nLoading model and data...")
    model, log_var_mle = load_model_and_checkpoint(args.uncertainty_checkpoint, device, dtype)

    # If a specific sequence is requested, pass it as a list to filter
    seq_filter = [args.tum_sequence] if args.tum_sequence else None
    dataset = load_dataset(args.tum_dir, sequences=seq_filter)

    if len(dataset.sequence_list) == 0:
        logger.error(f"No sequences found in {args.tum_dir}" +
                      (f" matching '{args.tum_sequence}'" if args.tum_sequence else ""))
        return

    seq_index = 0
    seq_name = dataset.sequence_list[seq_index]
    seq_len = len(dataset.data_store[seq_name])

    logger.info(f"Sequence: {seq_name}, Length: {seq_len}")

    # Get GT poses
    gt_poses_list = []
    for i in range(seq_len):
        batch = dataset.get_data(seq_index=seq_index, img_per_seq=1, ids=[i], aspect_ratio=1.0)
        extri = np.array(batch['extrinsics'][0])
        T = np.eye(4)
        T[:3, :] = extri
        gt_poses_list.append(T)
    gt_poses = np.stack(gt_poses_list, axis=0)  # [N, 4, 4]

    # Generate windows
    windows = generate_windows(seq_len, args.window_size, args.overlap)
    if args.max_windows is not None:
        windows = windows[:args.max_windows]
    logger.info(f"Generated {len(windows)} windows with {args.overlap*100:.0f}% overlap")

    # Single window mode: just evaluate VGGT on one window, no PGO needed
    if args.single_window_vggt_only:
        logger.info("\n[Single Window Mode - Direct VGGT evaluation, no PGO]")
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        w_start, w_end = windows[0]
        ids = list(range(w_start, w_start + args.window_size))

        batch = dataset.get_data(seq_index=seq_index, img_per_seq=args.window_size, ids=ids, aspect_ratio=1.0)
        images = np.stack(batch['images'], axis=0)
        images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        images_tensor = torch.from_numpy(images).to(device).unsqueeze(0)

        with torch.no_grad():
            predictions = model(images_tensor)

        pose_enc = predictions['pose_enc_list'][-1]
        image_hw = images_tensor.shape[-2:]
        pred_extri, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_hw)
        pred_poses_34 = pred_extri[0].float().cpu().numpy()  # [S, 3, 4]

        gt_window = gt_poses[w_start:w_start + args.window_size]  # [S, 4, 4]
        gt_poses_34 = gt_window[:, :3, :]  # [S, 3, 4]

        # Evaluate using tested implementation
        ate_result = _compute_ate_impl(pred_poses_34, gt_poses_34, align='sim3',
                                        gt_convention='w2c', pred_convention='w2c')
        rpe_result = _compute_rpe_impl(ate_result['aligned_poses'], gt_poses_34, delta=1)

        print("\n" + "=" * 60)
        print(f"Single Window Evaluation (frames {w_start}-{w_start + args.window_size - 1})")
        print("=" * 60)
        print(f"  ATE Trans RMSE: {ate_result['trans_rmse']*100:.2f} cm (p50={ate_result['trans_p50']*100:.2f}, p90={ate_result['trans_p90']*100:.2f}, p99={ate_result['trans_p99']*100:.2f})")
        print(f"  ATE Rot RMSE:   {ate_result['rot_rmse']:.2f}° (p50={ate_result['rot_p50']:.2f}, p90={ate_result['rot_p90']:.2f}, p99={ate_result['rot_p99']:.2f})")
        print(f"  RPE Trans RMSE: {rpe_result['trans_rmse']*100:.2f} cm (p50={rpe_result['trans_p50']*100:.2f}, p90={rpe_result['trans_p90']*100:.2f}, p99={rpe_result['trans_p99']*100:.2f})")
        print(f"  RPE Rot RMSE:   {rpe_result['rot_rmse']:.2f}° (p50={rpe_result['rot_p50']:.2f}, p90={rpe_result['rot_p90']:.2f}, p99={rpe_result['rot_p99']:.2f})")
        print(f"  Sim3 Scale: {ate_result['scale']:.3f}")
        return

    # Generate edges (with GT-based scale normalization)
    scale_mode = "global" if args.global_scale else "per-window"
    dt_info = f", max_dt={args.max_dt}" if args.max_dt is not None else ""
    sampling_info = ", training_style" if args.training_style else ", consecutive"
    logger.info(f"Generating edges (scale={scale_mode}{dt_info}{sampling_info})...")

    all_edges = []
    global_scale_value = None

    for w_idx, (w_start, _) in enumerate(windows):
        edges, window_scale = generate_edges_for_window(
            model, dataset, seq_index, w_start, args.window_size,
            gt_poses, device, dtype, theseus_config,
            max_dt=args.max_dt,
            global_scale=global_scale_value if args.global_scale else None,
            training_style_sampling=args.training_style
        )
        all_edges.extend(edges)

        # For global scale mode, capture scale from first window
        if args.global_scale and w_idx == 0:
            global_scale_value = window_scale
            logger.info(f"Using global scale from first window: {global_scale_value:.3f}")

    logger.info(f"Total edges: {len(all_edges)}")

    # Graph structure diagnostics
    print_graph_diagnostics(all_edges)

    # Pre-opt diagnostic (returns correlation stats)
    corr, pval, diag_info = compute_pre_opt_correlation(all_edges, gt_poses, theseus_config)

    # Initialize poses with MST (uses dt-based weights, not predicted uncertainty)
    logger.info("\nInitializing poses with MST...")
    initial_poses = initialize_poses_mst(all_edges)

    # Compute oracle weights if needed (using Theseus residuals for coordinate consistency)
    oracle_info = {}
    if args.oracle_isotropic or args.oracle_binned:
        logger.info("\nComputing oracle weights using Theseus residuals...")
        gt_residuals_6d, dt_list_oracle = compute_gt_residuals_theseus(all_edges, gt_poses)
        oracle_info['gt_residuals_6d'] = gt_residuals_6d
        oracle_info['dt_list'] = dt_list_oracle

        if args.oracle_isotropic:
            oracle_weights_iso = compute_oracle_weights_isotropic(
                gt_residuals_6d, dt_list_oracle, theseus_config)
            oracle_info['oracle_weights_isotropic'] = oracle_weights_iso

        if args.oracle_binned:
            oracle_weights_binned, bin_stats = compute_oracle_weights_binned(
                gt_residuals_6d, dt_list_oracle, theseus_config)
            oracle_info['oracle_weights_binned'] = oracle_weights_binned
            # Log binned statistics
            logger.info("\nOracle binned covariance statistics:")
            for (dt_min, dt_max), stats in sorted(bin_stats.items()):
                logger.info(f"  dt={dt_min:2d}-{dt_max:2d}: n={stats['n']:4d}, "
                           f"σ_t={stats['sigma_t']*100:5.1f}cm, "
                           f"σ_R={np.degrees(stats['sigma_R']):5.1f}°, "
                           f"λ_t={stats['lambda_t']:8.1f}, λ_R={stats['lambda_R']:8.1f}")

    # Run PGO with different weight modes
    results = {}

    # Determine which modes to run
    weight_modes = ['uniform', 'predicted']
    if args.oracle_isotropic:
        weight_modes.append('oracle_isotropic')
        logger.info("Oracle isotropic mode enabled: trans/rot isotropic + σ_floor")
    if args.oracle_binned:
        weight_modes.append('oracle_binned')
        logger.info("Oracle binned mode enabled: per-dt-bin empirical covariance")

    for mode in weight_modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running PGO with {mode} weights...")
        logger.info(f"{'='*60}")

        # Prepare oracle_info for this mode
        mode_oracle_info = None
        if mode == 'oracle_isotropic':
            mode_oracle_info = {'oracle_weights': oracle_info['oracle_weights_isotropic']}
        elif mode == 'oracle_binned':
            mode_oracle_info = {'oracle_weights': oracle_info['oracle_weights_binned']}

        optimized_poses, pgo_info = run_pgo_theseus(
            all_edges, initial_poses,
            weight_mode=mode,
            log_var_mle=log_var_mle,
            theseus_config=theseus_config,
            use_robust=args.robust,
            oracle_info=mode_oracle_info,
        )

        # Convert to array
        opt_array = poses_dict_to_array(optimized_poses)

        # Debug: print trajectory statistics
        if mode == 'uniform':
            # Get the GT poses for the frames we optimized
            num_opt_frames = len(optimized_poses)
            gt_subset = gt_poses[:num_opt_frames]

            # Try both conventions
            # w2c: C = -R.T @ t
            gt_centers_w2c = np.array([-p[:3, :3].T @ p[:3, 3] for p in gt_subset])
            pred_centers_w2c = np.array([-p[:3, :3].T @ p[:3, 3] for p in opt_array[:num_opt_frames]])
            # c2w: C = t directly
            gt_centers_c2w = np.array([p[:3, 3] for p in gt_subset])
            pred_centers_c2w = np.array([p[:3, 3] for p in opt_array[:num_opt_frames]])

            logger.info(f"\nTrajectory Stats (before alignment):")
            logger.info(f"  GT (w2c) span: x=[{gt_centers_w2c[:,0].min():.2f}, {gt_centers_w2c[:,0].max():.2f}], "
                       f"y=[{gt_centers_w2c[:,1].min():.2f}, {gt_centers_w2c[:,1].max():.2f}], "
                       f"z=[{gt_centers_w2c[:,2].min():.2f}, {gt_centers_w2c[:,2].max():.2f}]")
            logger.info(f"  Pred (w2c) span: x=[{pred_centers_w2c[:,0].min():.2f}, {pred_centers_w2c[:,0].max():.2f}], "
                       f"y=[{pred_centers_w2c[:,1].min():.2f}, {pred_centers_w2c[:,1].max():.2f}], "
                       f"z=[{pred_centers_w2c[:,2].min():.2f}, {pred_centers_w2c[:,2].max():.2f}]")
            logger.info(f"  GT (c2w) span: x=[{gt_centers_c2w[:,0].min():.2f}, {gt_centers_c2w[:,0].max():.2f}], "
                       f"y=[{gt_centers_c2w[:,1].min():.2f}, {gt_centers_c2w[:,1].max():.2f}], "
                       f"z=[{gt_centers_c2w[:,2].min():.2f}, {gt_centers_c2w[:,2].max():.2f}]")
            logger.info(f"  Pred (c2w) span: x=[{pred_centers_c2w[:,0].min():.2f}, {pred_centers_c2w[:,0].max():.2f}], "
                       f"y=[{pred_centers_c2w[:,1].min():.2f}, {pred_centers_c2w[:,1].max():.2f}], "
                       f"z=[{pred_centers_c2w[:,2].min():.2f}, {pred_centers_c2w[:,2].max():.2f}]")
            logger.info(f"  GT (w2c) travel: {np.sum(np.linalg.norm(np.diff(gt_centers_w2c, axis=0), axis=1)):.2f} m")
            logger.info(f"  Pred (w2c) travel: {np.sum(np.linalg.norm(np.diff(pred_centers_w2c, axis=0), axis=1)):.2f} m")

        # Evaluate using tested implementations (expects [N, 3, 4] poses, w2c convention)
        # Note: opt_array may cover only a subset of frames (when using --max_windows)
        num_opt_frames = len(optimized_poses)
        pred_poses_34 = opt_array[:num_opt_frames, :3, :]  # [N, 3, 4]
        gt_poses_34 = gt_poses[:num_opt_frames, :3, :]     # [N, 3, 4]

        # Use Sim3 alignment from tested implementation
        ate_result = _compute_ate_impl(pred_poses_34, gt_poses_34, align='sim3',
                                        gt_convention='w2c', pred_convention='w2c')
        rpe_result = _compute_rpe_impl(ate_result['aligned_poses'], gt_poses_34, delta=1)

        results[mode] = {
            'ate_trans': ate_result['trans_rmse'],
            'ate_rot': ate_result['rot_rmse'],
            'rpe_trans': rpe_result['trans_rmse'],
            'rpe_rot': rpe_result['rot_rmse'],
            'objective': pgo_info['objective'],
            'init_objective': pgo_info['init_objective'],
            'scale': ate_result['scale'],
        }

        logger.info(f"  ATE Trans: {ate_result['trans_rmse']*100:.2f} cm")
        logger.info(f"  ATE Rot:   {ate_result['rot_rmse']:.2f}°")
        logger.info(f"  RPE Trans: {rpe_result['trans_rmse']*100:.2f} cm")
        logger.info(f"  RPE Rot:   {rpe_result['rot_rmse']:.2f}°")
        logger.info(f"  Sim3 Scale: {ate_result['scale']:.3f}")
        logger.info(f"  Objective: {pgo_info['objective']:.4f}")

    # Also evaluate init poses using tested implementation
    num_init_frames = len(initial_poses)
    init_array = poses_dict_to_array(initial_poses)
    init_poses_34 = init_array[:num_init_frames, :3, :]  # [N, 3, 4]
    init_gt_34 = gt_poses[:num_init_frames, :3, :]  # [N, 3, 4]
    init_ate_result = _compute_ate_impl(init_poses_34, init_gt_34, align='sim3',
                                         gt_convention='w2c', pred_convention='w2c')
    init_rpe_result = _compute_rpe_impl(init_ate_result['aligned_poses'],
                                         init_gt_34, delta=1)
    init_ate_trans = init_ate_result['trans_rmse']
    init_ate_rot = init_ate_result['rot_rmse']
    init_rpe_trans = init_rpe_result['trans_rmse']
    init_rpe_rot = init_rpe_result['rot_rmse']

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'ATE Trans':>12} {'ATE Rot':>10} {'RPE Trans':>12} {'RPE Rot':>10} {'Objective':>12}")
    print("-" * 80)

    # Before PGO row
    print(f"{'Before PGO (init)':<20} {init_ate_trans*100:>10.2f} cm {init_ate_rot:>8.2f}° {init_rpe_trans*100:>10.2f} cm {init_rpe_rot:>8.2f}° {'N/A':>12}")

    for mode, r in results.items():
        print(f"{'PGO + ' + mode:<20} {r['ate_trans']*100:>10.2f} cm {r['ate_rot']:>8.2f}° {r['rpe_trans']*100:>10.2f} cm {r['rpe_rot']:>8.2f}° {r['objective']:>12.4f}")

    print("=" * 80)

    # Check success criteria
    if 'predicted' in results and 'uniform' in results:
        if results['predicted']['ate_trans'] < results['uniform']['ate_trans']:
            print("\n✓ SUCCESS: PGO + Predicted < PGO + Uniform (ATE)")
        else:
            print("\n✗ FAIL: PGO + Predicted >= PGO + Uniform (ATE)")


if __name__ == '__main__':
    main()
