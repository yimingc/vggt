# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

        # Log-variance branch: outputs 6-dim log(σ²) for uncertainty
        # Convention: 3 translation + 3 rotation (matching PyPose se(3) order)
        # log_var=0 means σ=1 (reasonable default), NLL uses: 0.5 * (r²/σ² + log(σ²))
        self.pose_log_var_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=6,  # 3 trans + 3 rot (use split_se3_tangent in loss)
            drop=0
        )

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> tuple:
        """
        Forward pass to predict camera parameters and pose uncertainty.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            tuple: (pred_pose_enc_list, pred_log_var_list)
                - pred_pose_enc_list: List of predicted camera encodings (post-activation) from each iteration.
                - pred_log_var_list: List of log-variance values (6-dim) from each iteration.
                  log_var = log(σ²), where σ is the uncertainty standard deviation.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        # Explicitly cast to float32 for layer_norm (tokens may be bfloat16 from autocast)
        pose_tokens = pose_tokens.float()
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc_list, pred_log_var_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list, pred_log_var_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> tuple:
        """
        Iteratively refine camera pose predictions and uncertainty estimates.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            tuple: (pred_pose_enc_list, pred_log_var_list)
                - pred_pose_enc_list: List of activated camera encodings from each iteration.
                - pred_log_var_list: List of log-variance values (6-dim) from each iteration.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []
        pred_log_var_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            trunk_output = self.trunk_norm(pose_tokens_modulated)

            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(trunk_output)

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

            # Compute log-variance for uncertainty: log(σ²)
            # log_var=0 means σ=1, NLL uses: 0.5 * (r²*exp(-log_var) + log_var)
            pred_log_var = self.pose_log_var_branch(trunk_output)
            pred_log_var_list.append(pred_log_var)

        return pred_pose_enc_list, pred_log_var_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
