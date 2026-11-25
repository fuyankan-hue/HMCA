import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_angle(x):
    return torch.atan2(
        torch.sin(x),
        torch.cos(x),
    )


def inverse_map_rigid_conflict(
    lane_positions,
    lane_centers,
    lane_angles,
    delta,
):
    translation = delta[:, :2]
    yaw = delta[:, 2]

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    inv_rot = torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s,  c], dim=-1),
        ],
        dim=-2,
    )

    lane_positions = (
        lane_positions
        - translation[:, None, None, :]
    )

    lane_centers = (
        lane_centers
        - translation[:, None, :]
    )

    lane_positions = torch.einsum(
        "bmld,bdk->bmlk",
        lane_positions,
        inv_rot,
    )

    lane_centers = torch.einsum(
        "bmd,bdk->bmk",
        lane_centers,
        inv_rot,
    )

    lane_angles = (
        lane_angles
        - yaw[:, None]
    )

    return (
        lane_positions,
        lane_centers,
        lane_angles,
    )


class HistoryMapConflictAlignment(nn.Module):
    """
    HMCA v2.1

    Branch A:
        focal historical trajectory -> lane geometry

    Branch B:
        all currently visible actors -> lane geometry

    Branch B provides scene-level geometric consensus.

    Output:
        severity in [0,1]
        base_delta = maximum-severity SE(2) correction
        final_delta = severity * base_delta
    """

    def __init__(
        self,
        hidden_dim=96,
        num_heads=4,
        temporal_depth=2,
        agent_depth=2,
        knn=4,
        knn_temperature=2.0,
        max_translation=3.0,
        max_yaw_deg=3.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.knn = int(knn)
        self.knn_temperature = float(
            knn_temperature
        )

        self.max_translation = float(
            max_translation
        )

        self.max_yaw = math.radians(
            float(max_yaw_deg)
        )

        # =====================================================
        # FOCAL TRAJECTORY BRANCH
        # =====================================================

        # x,y
        # dx,dy
        # nearest distance
        # longitudinal
        # lateral
        # sin/cos heading residual
        # time
        # valid
        #
        # = 11

        self.geometry_mlp = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
        )

        temporal_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=
                    hidden_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.temporal_encoder = (
            nn.TransformerEncoder(
                temporal_layer,
                num_layers=temporal_depth,
            )
        )

        self.temporal_score = nn.Linear(
            hidden_dim,
            1,
        )

        # =====================================================
        # MULTI-AGENT CONSENSUS BRANCH
        # =====================================================

        self.agent_geometry_mlp = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
        )

        agent_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=
                    hidden_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.agent_encoder = (
            nn.TransformerEncoder(
                agent_layer,
                num_layers=agent_depth,
            )
        )

        self.agent_score = nn.Linear(
            hidden_dim,
            1,
        )

        # =====================================================
        # EXPLICIT STATISTICS
        #
        # focal statistics = 6
        # multi-agent statistics = 8
        #
        # total = 14
        # =====================================================

        self.summary_mlp = nn.Sequential(
            nn.Linear(14, 48),
            nn.GELU(),
            nn.Linear(48, 48),
            nn.GELU(),
        )

        # temporal 96
        # agent    96
        # stats    48
        # total   240

        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + 48,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                hidden_dim,
            ),
            nn.GELU(),
        )

        # Maximum-severity correction.
        self.output = nn.Linear(
            hidden_dim,
            3,
        )

        # Continuous conflict severity.
        self.severity_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

        self.reset_output()

    def reset_output(self):

        # Start as exact identity correction.
        nn.init.zeros_(
            self.output.weight
        )

        nn.init.zeros_(
            self.output.bias
        )

        # Start severity around 0.1.
        last = self.severity_head[-1]

        nn.init.zeros_(
            last.weight
        )

        nn.init.constant_(
            last.bias,
            -2.197224577,
        )

    @staticmethod
    def _lane_point_tangents(
        lane_positions,
        lane_padding_mask,
    ):

        B, M, L, _ = (
            lane_positions.shape
        )

        tangent = torch.zeros_like(
            lane_positions
        )

        if L == 1:
            tangent[..., 0] = 1.0
            return tangent

        tangent[:, :, 0] = (
            lane_positions[:, :, 1]
            - lane_positions[:, :, 0]
        )

        tangent[:, :, -1] = (
            lane_positions[:, :, -1]
            - lane_positions[:, :, -2]
        )

        if L > 2:
            tangent[:, :, 1:-1] = (
                lane_positions[:, :, 2:]
                - lane_positions[:, :, :-2]
            )

        tangent = F.normalize(
            tangent,
            dim=-1,
            eps=1e-6,
        )

        tangent = tangent.masked_fill(
            lane_padding_mask.unsqueeze(-1),
            0.0,
        )

        return tangent

    def _prepare_map(
        self,
        lane_positions,
        lane_padding_mask,
    ):

        B, M, L, _ = (
            lane_positions.shape
        )

        tangent = (
            self._lane_point_tangents(
                lane_positions,
                lane_padding_mask,
            )
        )

        points = lane_positions.reshape(
            B,
            M * L,
            2,
        )

        tangent = tangent.reshape(
            B,
            M * L,
            2,
        )

        invalid = lane_padding_mask.reshape(
            B,
            M * L,
        )

        all_invalid = invalid.all(
            dim=1
        )

        if all_invalid.any():

            points = points.clone()
            tangent = tangent.clone()
            invalid = invalid.clone()

            invalid[
                all_invalid,
                0,
            ] = False

            points[
                all_invalid,
                0,
            ] = 0.0

            tangent[
                all_invalid,
                0,
                0,
            ] = 1.0

        return (
            points,
            tangent,
            invalid,
        )

    def _soft_match(
        self,
        query_points,
        map_points,
        map_tangents,
        map_invalid,
    ):

        B, Q, _ = (
            query_points.shape
        )

        distances = torch.cdist(
            query_points,
            map_points,
        )

        distances = distances.masked_fill(
            map_invalid[:, None, :],
            1e6,
        )

        k = min(
            self.knn,
            map_points.shape[1],
        )

        knn_dist, knn_idx = torch.topk(
            distances,
            k=k,
            dim=-1,
            largest=False,
        )

        batch_index = torch.arange(
            B,
            device=query_points.device,
        )[:, None, None]

        batch_index = batch_index.expand(
            B,
            Q,
            k,
        )

        knn_points = map_points[
            batch_index,
            knn_idx,
        ]

        knn_tangent = map_tangents[
            batch_index,
            knn_idx,
        ]

        weight = torch.softmax(
            -knn_dist
            / self.knn_temperature,
            dim=-1,
        )

        matched_point = (
            weight.unsqueeze(-1)
            * knn_points
        ).sum(dim=2)

        matched_tangent = (
            weight.unsqueeze(-1)
            * knn_tangent
        ).sum(dim=2)

        matched_tangent = F.normalize(
            matched_tangent,
            dim=-1,
            eps=1e-6,
        )

        return (
            matched_point,
            matched_tangent,
        )

    def _geometry(
        self,
        query_points,
        query_angles,
        map_points,
        map_tangents,
        map_invalid,
    ):

        matched_point, matched_tangent = (
            self._soft_match(
                query_points,
                map_points,
                map_tangents,
                map_invalid,
            )
        )

        offset = (
            query_points
            - matched_point
        )

        distance = torch.norm(
            offset,
            dim=-1,
        )

        normal = torch.stack(
            [
                -matched_tangent[..., 1],
                matched_tangent[..., 0],
            ],
            dim=-1,
        )

        longitudinal = (
            offset
            * matched_tangent
        ).sum(dim=-1)

        lateral = (
            offset
            * normal
        ).sum(dim=-1)

        lane_heading = torch.atan2(
            matched_tangent[..., 1],
            matched_tangent[..., 0],
        )

        heading_error = wrap_angle(
            query_angles
            - lane_heading
        )

        return {
            "offset": offset,
            "distance": distance,
            "longitudinal": longitudinal,
            "lateral": lateral,
            "heading_error": heading_error,
        }

    def _focal_branch(
        self,
        history_positions,
        history_angles,
        history_padding_mask,
        map_points,
        map_tangents,
        map_invalid,
    ):

        B, T, _ = (
            history_positions.shape
        )

        valid = (
            ~history_padding_mask
        )

        valid_float = (
            valid.float()
        )

        geo = self._geometry(
            history_positions,
            history_angles,
            map_points,
            map_tangents,
            map_invalid,
        )

        time = torch.linspace(
            -1.0,
            0.0,
            T,
            device=history_positions.device,
            dtype=history_positions.dtype,
        )

        time = (
            time
            .view(1, T)
            .expand(B, T)
        )

        features = torch.cat(
            [
                history_positions / 50.0,
                geo["offset"] / 5.0,
                geo["distance"].unsqueeze(-1)
                    / 5.0,
                geo["longitudinal"].unsqueeze(-1)
                    / 5.0,
                geo["lateral"].unsqueeze(-1)
                    / 5.0,
                torch.sin(
                    geo["heading_error"]
                ).unsqueeze(-1),
                torch.cos(
                    geo["heading_error"]
                ).unsqueeze(-1),
                time.unsqueeze(-1),
                valid_float.unsqueeze(-1),
            ],
            dim=-1,
        )

        features = features.masked_fill(
            history_padding_mask.unsqueeze(-1),
            0.0,
        )

        token = self.geometry_mlp(
            features
        )

        token = self.temporal_encoder(
            token,
            src_key_padding_mask=
                history_padding_mask,
        )

        score = self.temporal_score(
            token
        ).squeeze(-1)

        score = score.masked_fill(
            history_padding_mask,
            -1e4,
        )

        weight = torch.softmax(
            score,
            dim=-1,
        )

        summary = (
            weight.unsqueeze(-1)
            * token
        ).sum(dim=1)

        denominator = (
            valid_float
            .sum(dim=1)
            .clamp_min(1.0)
        )

        def valid_mean(x):
            return (
                (
                    x
                    * valid_float
                ).sum(dim=1)
                / denominator
            )

        mean_dist = valid_mean(
            geo["distance"]
        )

        max_dist = (
            geo["distance"]
            .masked_fill(
                history_padding_mask,
                -1e6,
            )
            .max(dim=1)
            .values
        )

        mean_abs_lat = valid_mean(
            geo["lateral"].abs()
        )

        mean_abs_lon = valid_mean(
            geo["longitudinal"].abs()
        )

        mean_abs_heading = valid_mean(
            geo["heading_error"].abs()
        )

        current_dist = (
            geo["distance"][:, -1]
        )

        stats = torch.stack(
            [
                mean_dist / 5.0,
                max_dist / 5.0,
                mean_abs_lat / 5.0,
                mean_abs_lon / 5.0,
                mean_abs_heading
                    / math.radians(10.0),
                current_dist / 5.0,
            ],
            dim=-1,
        )

        return (
            summary,
            stats,
        )

    def _agent_branch(
        self,
        all_history_positions,
        all_history_angles,
        all_history_padding_mask,
        map_points,
        map_tangents,
        map_invalid,
    ):

        # EMP actors are selected at t=49.
        current_pos = (
            all_history_positions[:, :, -1]
        )

        current_angle = (
            all_history_angles[:, :, -1]
        )

        valid = (
            ~all_history_padding_mask[:, :, -1]
        )

        # Focal actor should always be valid, but keep
        # a safety fallback.
        all_invalid = (
            ~valid
        ).all(dim=1)

        if all_invalid.any():
            valid = valid.clone()
            valid[
                all_invalid,
                0,
            ] = True

        valid_float = valid.float()

        geo = self._geometry(
            current_pos,
            current_angle,
            map_points,
            map_tangents,
            map_invalid,
        )

        radial_distance = torch.norm(
            current_pos,
            dim=-1,
        )

        features = torch.cat(
            [
                current_pos / 50.0,
                geo["offset"] / 5.0,
                geo["distance"].unsqueeze(-1)
                    / 5.0,
                geo["longitudinal"].unsqueeze(-1)
                    / 5.0,
                geo["lateral"].unsqueeze(-1)
                    / 5.0,
                torch.sin(
                    geo["heading_error"]
                ).unsqueeze(-1),
                torch.cos(
                    geo["heading_error"]
                ).unsqueeze(-1),
                radial_distance.unsqueeze(-1)
                    / 50.0,
                valid_float.unsqueeze(-1),
            ],
            dim=-1,
        )

        features = features.masked_fill(
            (~valid).unsqueeze(-1),
            0.0,
        )

        token = self.agent_geometry_mlp(
            features
        )

        token = self.agent_encoder(
            token,
            src_key_padding_mask=
                ~valid,
        )

        score = self.agent_score(
            token
        ).squeeze(-1)

        score = score.masked_fill(
            ~valid,
            -1e4,
        )

        weight = torch.softmax(
            score,
            dim=-1,
        )

        summary = (
            weight.unsqueeze(-1)
            * token
        ).sum(dim=1)

        denominator = (
            valid_float.sum(dim=1)
            .clamp_min(1.0)
        )

        mean_offset = (
            geo["offset"]
            * valid_float.unsqueeze(-1)
        ).sum(dim=1)

        mean_offset = (
            mean_offset
            / denominator.unsqueeze(-1)
        )

        centered = (
            geo["offset"]
            - mean_offset.unsqueeze(1)
        )

        var_offset = (
            centered.square()
            * valid_float.unsqueeze(-1)
        ).sum(dim=1)

        var_offset = (
            var_offset
            / denominator.unsqueeze(-1)
        )

        std_offset = torch.sqrt(
            var_offset + 1e-6
        )

        mean_dist = (
            geo["distance"]
            * valid_float
        ).sum(dim=1) / denominator

        centered_dist = (
            geo["distance"]
            - mean_dist.unsqueeze(1)
        )

        var_dist = (
            centered_dist.square()
            * valid_float
        ).sum(dim=1) / denominator

        std_dist = torch.sqrt(
            var_dist + 1e-6
        )

        mean_sin_heading = (
            torch.sin(
                geo["heading_error"]
            )
            * valid_float
        ).sum(dim=1) / denominator

        mean_cos_heading = (
            torch.cos(
                geo["heading_error"]
            )
            * valid_float
        ).sum(dim=1) / denominator

        stats = torch.cat(
            [
                mean_offset / 5.0,
                std_offset / 5.0,
                (mean_dist / 5.0).unsqueeze(-1),
                (std_dist / 5.0).unsqueeze(-1),
                mean_sin_heading.unsqueeze(-1),
                mean_cos_heading.unsqueeze(-1),
            ],
            dim=-1,
        )

        return (
            summary,
            stats,
        )

    def forward(
        self,
        history_positions,
        history_angles,
        history_padding_mask,
        all_history_positions,
        all_history_angles,
        all_history_padding_mask,
        lane_positions,
        lane_padding_mask,
        lane_centers=None,
        lane_angles=None,
        lane_key_padding_mask=None,
    ):

        (
            map_points,
            map_tangents,
            map_invalid,
        ) = self._prepare_map(
            lane_positions,
            lane_padding_mask,
        )

        focal_summary, focal_stats = (
            self._focal_branch(
                history_positions,
                history_angles,
                history_padding_mask,
                map_points,
                map_tangents,
                map_invalid,
            )
        )

        agent_summary, agent_stats = (
            self._agent_branch(
                all_history_positions,
                all_history_angles,
                all_history_padding_mask,
                map_points,
                map_tangents,
                map_invalid,
            )
        )

        stats = torch.cat(
            [
                focal_stats,
                agent_stats,
            ],
            dim=-1,
        )

        stats_feature = (
            self.summary_mlp(
                stats
            )
        )

        fused = self.fusion(
            torch.cat(
                [
                    focal_summary,
                    agent_summary,
                    stats_feature,
                ],
                dim=-1,
            )
        )

        raw = self.output(
            fused
        )

        base_translation = (
            torch.tanh(
                raw[:, :2]
            )
            * self.max_translation
        )

        base_yaw = (
            torch.tanh(
                raw[:, 2:3]
            )
            * self.max_yaw
        )

        base_delta = torch.cat(
            [
                base_translation,
                base_yaw,
            ],
            dim=-1,
        )

        severity_logit = (
            self.severity_head(
                fused
            ).squeeze(-1)
        )

        severity = torch.sigmoid(
            severity_logit
        )

        final_delta = (
            base_delta
            * severity.unsqueeze(-1)
        )

        return {
            "raw_delta":
                base_delta,

            "severity_logit":
                severity_logit,

            "severity":
                severity,

            # Compatibility aliases.
            "conflict_logit":
                severity_logit,

            "conflict_prob":
                severity,

            "gated_delta":
                final_delta,

            "geometry_statistics":
                stats,
        }
