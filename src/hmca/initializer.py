"""Trainable direct initializer operating on preprocessed history–map tensors."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._encoder import HistoryMapConflictAlignment


class DirectInitializer(nn.Module):
    """The full three-branch initializer used before geometric refinement."""

    def __init__(self):
        super().__init__()
        self.core = HistoryMapConflictAlignment()
        # Retain the archived tensor layout. Direct regression bypasses severity.
        self.core.severity_head.requires_grad_(False)

    def forward(
        self,
        *,
        history_positions,
        history_angles,
        history_padding_mask,
        all_history_positions,
        all_history_angles,
        all_history_padding_mask,
        lane_positions,
        lane_padding_mask,
    ):
        h = self.core
        lane_geometry = h._prepare_map(lane_positions, lane_padding_mask)
        focal, focal_stats = h._focal_branch(
            history_positions, history_angles, history_padding_mask, *lane_geometry
        )
        agents, agent_stats = h._agent_branch(
            all_history_positions, all_history_angles, all_history_padding_mask, *lane_geometry
        )
        stats = h.summary_mlp(torch.cat((focal_stats, agent_stats), dim=-1))
        features = h.fusion(torch.cat((focal, agents, stats), dim=-1))
        raw = h.output(features)
        return torch.cat((
            raw[:, :2].tanh() * h.max_translation,
            raw[:, 2:].tanh() * h.max_yaw,
        ), dim=-1)


def direct_pose_loss(predicted, target):
    """Smooth L1 on translation and yaw scaled by two degrees."""
    yaw_scale = math.radians(2)
    return (
        F.smooth_l1_loss(predicted[:, :2], target[:, :2])
        + F.smooth_l1_loss(predicted[:, 2] / yaw_scale, target[:, 2] / yaw_scale)
    )
