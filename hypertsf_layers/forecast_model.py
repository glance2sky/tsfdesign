"""End-to-end hyperbolic graph forecaster."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .front_layers import (
    DualGraphHyperbolicLayer,
    ManifoldEmbedding,
    RevIN,
)


class LowRankTemporalProjection(nn.Module):
    """Factorized projection from history positions to future positions."""

    def __init__(
        self,
        input_length: int,
        output_length: int,
        feature_dim: int,
        rank: int | None,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.feature_dim = feature_dim
        if rank is None:
            self.rank = None
            self.full_projection = nn.Linear(input_length, output_length)
            self.register_parameter("input_basis", None)
            self.register_parameter("output_basis", None)
            self.register_parameter("output_anchor", None)
        else:
            if rank <= 0:
                raise ValueError("rank must be positive")
            rank = min(rank, input_length, output_length)
            self.rank = rank
            self.register_module("full_projection", None)
            self.input_basis = nn.Parameter(torch.empty(rank, input_length))
            self.output_basis = nn.Parameter(torch.empty(output_length, rank))
            self.output_anchor = nn.Parameter(torch.zeros(output_length, 1))
            nn.init.normal_(
                self.input_basis,
                mean=0.0,
                std=input_length**-0.5,
            )
            nn.init.normal_(
                self.output_basis,
                mean=0.0,
                std=rank**-0.5,
            )
            nn.init.constant_(self.output_anchor, 0.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, time, features]")
        if values.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, got {values.size(1)}"
            )
        if self.rank is None:
            return self.full_projection(values.transpose(1, 2)).transpose(1, 2)
        basis = self.input_basis / self.input_basis.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        coefficients = torch.einsum("rl,blh->brh", basis, values)
        projected = torch.einsum(
            "pr,brh->bph", self.output_basis, coefficients
        )
        anchor = self.output_anchor.view(1, self.output_length, 1)
        return projected + anchor * values.mean(dim=1, keepdim=True)


class MultiScaleTemporalProjection(nn.Module):
    """Fine-to-coarse temporal projection with low-frequency corrections.

    Coarse branches are output-zero initialized, while their gates start at
    one. This preserves the original fine branch at initialization without
    blocking gradients to the coarse projection parameters.
    """

    def __init__(
        self,
        input_length: int,
        output_length: int,
        feature_dim: int,
        rank: int | None,
        factors: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if not factors or factors[0] != 1:
            raise ValueError("factors must start with 1 for the fine branch")
        if any(factor <= 0 for factor in factors):
            raise ValueError("all temporal scale factors must be positive")
        if len(set(factors)) != len(factors):
            raise ValueError("temporal scale factors must be unique")

        self.input_length = input_length
        self.output_length = output_length
        self.feature_dim = feature_dim
        self.factors = factors
        self.base_projection = LowRankTemporalProjection(
            input_length=input_length,
            output_length=output_length,
            feature_dim=feature_dim,
            rank=rank,
        )
        self.coarse_projections = nn.ModuleList()
        for factor in factors[1:]:
            pooled_length = max(1, (input_length + factor - 1) // factor)
            projection = LowRankTemporalProjection(
                input_length=pooled_length,
                output_length=output_length,
                feature_dim=feature_dim,
                rank=rank,
            )
            if projection.rank is None:
                nn.init.zeros_(projection.full_projection.weight)
                nn.init.zeros_(projection.full_projection.bias)
            else:
                nn.init.zeros_(projection.output_basis)
                nn.init.zeros_(projection.output_anchor)
            self.coarse_projections.append(projection)
        self.coarse_norms = nn.ModuleList(
            [nn.LayerNorm(feature_dim) for _ in factors[1:]]
        )
        self.coarse_gate_raw = nn.Parameter(torch.zeros(len(factors) - 1))

    @property
    def rank(self) -> int | None:
        """Expose the fine-branch rank for backwards-compatible inspection."""
        return self.base_projection.rank

    def forward(
        self,
        values: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, time, features]")
        if values.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, got {values.size(1)}"
            )

        projected = self.base_projection(values)
        values_time_first = values.transpose(1, 2)
        gates = torch.exp(0.25 * torch.tanh(self.coarse_gate_raw))
        contributions = []
        for gate, projection, norm, factor in zip(
            gates, self.coarse_projections, self.coarse_norms, self.factors[1:]
        ):
            pooled_length = max(1, (self.input_length + factor - 1) // factor)
            pooled = F.adaptive_avg_pool1d(
                values_time_first, pooled_length
            ).transpose(1, 2)
            correction = gate * norm(projection(pooled))
            projected = projected + correction
            contributions.append(correction.detach().abs().mean())
        if return_diagnostics:
            return projected, {
                "scale_gates": gates.detach(),
                "scale_contributions": torch.stack(contributions),
            }
        return projected

    def gate_values(self) -> torch.Tensor:
        return torch.exp(0.25 * torch.tanh(self.coarse_gate_raw))


class LocalTrendResidual(nn.Module):
    """Low-variance local level and slope extrapolation."""

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        output_dim: int,
        trend_window: int | None = None,
    ) -> None:
        super().__init__()
        if trend_window is None:
            trend_window = min(input_length, 24)
        if trend_window < 2 or trend_window > input_length:
            raise ValueError(
                "trend_window must be in [2, input_length]"
            )
        self.input_length = input_length
        self.pred_length = pred_length
        self.output_dim = output_dim
        self.trend_window = trend_window
        self.level_weight = nn.Parameter(torch.ones(1, 1, output_dim))
        self.slope_weight = nn.Parameter(torch.full((1, 1, output_dim), 0.1))
        self.trend_scale_raw = nn.Parameter(torch.zeros(()))
        positions = torch.arange(
            1, pred_length + 1, dtype=torch.float32
        ).view(1, pred_length, 1)
        self.register_buffer(
            "future_positions",
            positions / float(trend_window - 1),
        )

    def forward(
        self,
        normalized_input: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_input.ndim != 3:
            raise ValueError(
                "normalized_input must have shape [batch, time, variables]"
            )
        if normalized_input.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, "
                f"got {normalized_input.size(1)}"
            )
        recent = normalized_input[:, -self.trend_window :]
        level = recent[:, -1:, :]
        slope = (recent[:, -1:, :] - recent[:, :1, :]) / float(
            self.trend_window - 1
        )
        trend = (
            self.level_weight * level
            + self.slope_weight * slope * self.future_positions
        )
        return torch.tanh(self.trend_scale_raw) * trend.index_select(
            -1, target_indices
        )


class AdaptivePathFusion(nn.Module):
    """Variable- and horizon-conditioned fusion of direct and residual paths.

    The path coefficients are ``2 * sigmoid(logits)`` and the final layer is
    initialized to zero. Consequently both coefficients start at one and the
    new head is functionally identical to ``direct + residual`` at
    initialization. The conditioning features describe local level, scale,
    range, recent volatility, and recent change for each target variable.
    """

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_length <= 0 or pred_length <= 0:
            raise ValueError("input_length and pred_length must be positive")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_length = input_length
        self.pred_length = pred_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stats_projection = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.horizon_embedding = nn.Parameter(
            torch.empty(1, pred_length, hidden_dim)
        )
        self.path_projection = nn.Linear(hidden_dim, 2)
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.path_projection.weight)
        nn.init.zeros_(self.path_projection.bias)

    def forward(
        self,
        direct: torch.Tensor,
        residual: torch.Tensor,
        normalized_input: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if direct.ndim != 3 or residual.ndim != 3:
            raise ValueError(
                "direct and residual must have shape [batch, horizon, variables]"
            )
        if direct.shape != residual.shape:
            raise ValueError("direct and residual must have the same shape")
        if normalized_input.ndim != 3:
            raise ValueError(
                "normalized_input must have shape [batch, time, variables]"
            )
        if normalized_input.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, "
                f"got {normalized_input.size(1)}"
            )
        if direct.size(1) != self.pred_length:
            raise ValueError(
                f"Expected prediction length {self.pred_length}, "
                f"got {direct.size(1)}"
            )

        recent_length = min(24, self.input_length)
        recent = normalized_input[:, -recent_length:]
        mean = normalized_input.mean(dim=1)
        scale = normalized_input.std(dim=1, unbiased=False)
        value_range = (
            normalized_input.amax(dim=1) - normalized_input.amin(dim=1)
        )
        recent_scale = recent.std(dim=1, unbiased=False)
        recent_change = recent[:, -1] - recent[:, 0]
        stats = torch.stack(
            (mean, scale, value_range, recent_scale, recent_change),
            dim=-1,
        )
        variable_features = self.stats_projection(stats)
        variable_features = variable_features.index_select(
            1, target_indices.to(normalized_input.device)
        )
        fusion_features = (
            variable_features.unsqueeze(1)
            + self.horizon_embedding.unsqueeze(2)
        )
        logits = self.path_projection(fusion_features)
        # Keep the total path mass equal to the original two additive paths.
        # At zero logits this gives [1, 1], while training can trade mass
        # between direct and residual branches without amplifying both.
        weights = 2.0 * torch.softmax(logits, dim=-1)
        fused = (
            weights[..., 0] * direct
            + weights[..., 1] * residual
        )
        return {
            "prediction": fused,
            "weights": weights,
            "direct_weight_mean": weights[..., 0].detach().mean(),
            "residual_weight_mean": weights[..., 1].detach().mean(),
            "path_weight_std": weights.detach().std(unbiased=False),
            "adaptive_correction_abs_mean": (
                fused - direct - residual
            ).detach().abs().mean(),
        }


class PathAmplitudeCalibration(nn.Module):
    """Conditionally calibrate direct and residual path amplitudes.

    The v4 fusion layer can trade path coefficients, but the two paths can
    still have very different output magnitudes. This module learns a small
    variable-conditional rescaling before fusion. All deltas start at zero,
    so the initial scale of both paths is exactly one.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim
        self.condition = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scale_projection = nn.Linear(hidden_dim, 2)
        self.path_scale_raw = nn.Parameter(torch.zeros(2, output_dim))
        nn.init.zeros_(self.scale_projection.weight)
        nn.init.zeros_(self.scale_projection.bias)

    def forward(
        self,
        direct: torch.Tensor,
        residual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if direct.ndim != 3 or residual.ndim != 3:
            raise ValueError(
                "direct and residual must have shape [batch, horizon, variables]"
            )
        if direct.shape != residual.shape:
            raise ValueError("direct and residual must have the same shape")
        if direct.size(-1) != self.output_dim:
            raise ValueError(
                f"Expected {self.output_dim} output variables, got {direct.size(-1)}"
            )

        direct_abs = direct.detach().abs().mean(dim=1)
        residual_abs = residual.detach().abs().mean(dim=1)
        ratio = residual_abs / direct_abs.clamp_min(1e-5)
        features = torch.stack(
            (
                torch.log1p(direct_abs),
                torch.log1p(residual_abs),
                torch.log1p(ratio),
            ),
            dim=-1,
        )
        hidden = self.condition(features)
        dynamic_delta = self.scale_projection(hidden)
        static_delta = self.path_scale_raw.transpose(0, 1).unsqueeze(0)
        scales = torch.exp(0.5 * torch.tanh(dynamic_delta + static_delta))
        calibrated_direct = direct * scales[..., 0].unsqueeze(1)
        calibrated_residual = residual * scales[..., 1].unsqueeze(1)
        return {
            "direct": calibrated_direct,
            "residual": calibrated_residual,
            "scales": scales,
            "direct_scale_mean": scales[..., 0].detach().mean(),
            "residual_scale_mean": scales[..., 1].detach().mean(),
            "scale_std": scales.detach().std(unbiased=False),
            "calibration_correction_abs_mean": (
                calibrated_direct
                + calibrated_residual
                - direct
                - residual
            ).detach().abs().mean(),
        }


class DirectForecastHead(nn.Module):
    """Structured direct multi-horizon head.

    The historical-to-future map is factorized into shared temporal bases.
    This regularizes long-horizon extrapolation while retaining a local trend
    residual for variables whose recent level and slope remain predictive.
    """

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        hidden_dim: int,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        use_linear_residual: bool = True,
        temporal_rank: int | None = None,
        trend_window: int | None = None,
        use_multiscale_projection: bool = False,
        multiscale_factors: tuple[int, ...] = (1, 2, 4),
        use_adaptive_path_fusion: bool = False,
        use_path_amplitude_calibration: bool = False,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_linear_residual = use_linear_residual
        self.temporal_rank = (
            None
            if temporal_rank is None
            else min(temporal_rank, input_length, pred_length)
        )

        self.use_multiscale_projection = use_multiscale_projection
        self.use_adaptive_path_fusion = use_adaptive_path_fusion
        self.use_path_amplitude_calibration = use_path_amplitude_calibration
        if use_multiscale_projection:
            self.horizon_projection = MultiScaleTemporalProjection(
                input_length=input_length,
                output_length=pred_length,
                feature_dim=hidden_dim,
                rank=self.temporal_rank,
                factors=multiscale_factors,
            )
        else:
            self.horizon_projection = LowRankTemporalProjection(
                input_length=input_length,
                output_length=pred_length,
                feature_dim=hidden_dim,
                rank=self.temporal_rank,
            )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        if use_multiscale_projection:
            self.multiscale_factors = multiscale_factors
        else:
            self.multiscale_factors = (1,)

        if use_linear_residual:
            residual_rank = (
                None
                if self.temporal_rank is None
                else min(self.temporal_rank, input_length, pred_length)
            )
            self.linear_residual = LowRankTemporalProjection(
                input_length=input_length,
                output_length=pred_length,
                feature_dim=input_dim,
                rank=residual_rank,
            )
            self.trend_residual = LocalTrendResidual(
                input_length=input_length,
                pred_length=pred_length,
                output_dim=output_dim,
                trend_window=trend_window,
            )
            if use_adaptive_path_fusion:
                self.path_fusion = AdaptivePathFusion(
                    input_length=input_length,
                    pred_length=pred_length,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=max(8, min(hidden_dim, 64)),
                    dropout=dropout,
                )
            else:
                self.register_module("path_fusion", None)
            if use_path_amplitude_calibration:
                self.path_calibration = PathAmplitudeCalibration(
                    output_dim=output_dim,
                    hidden_dim=max(8, min(hidden_dim, 64)),
                    dropout=dropout,
                )
            else:
                self.register_module("path_calibration", None)
        else:
            self.register_module("linear_residual", None)
            self.register_module("trend_residual", None)
            self.register_module("path_fusion", None)
            self.register_module("path_calibration", None)

    def forward(
        self,
        temporal_context: torch.Tensor,
        normalized_input: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if temporal_context.ndim != 3:
            raise ValueError(
                "temporal_context must have shape [batch, time, hidden_dim]"
            )
        if normalized_input.ndim != 3:
            raise ValueError(
                "normalized_input must have shape [batch, time, variables]"
            )
        if temporal_context.size(1) != self.input_length:
            raise ValueError(
                f"Expected temporal length {self.input_length}, "
                f"got {temporal_context.size(1)}"
            )
        if normalized_input.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, "
                f"got {normalized_input.size(1)}"
            )

        if isinstance(self.horizon_projection, MultiScaleTemporalProjection):
            hidden, scale_diagnostics = self.horizon_projection(
                temporal_context,
                return_diagnostics=True,
            )
        else:
            hidden = self.horizon_projection(temporal_context)
            scale_diagnostics = None
        hidden = self.context_dropout(self.context_norm(hidden))
        direct = self.output_projection(hidden)

        if self.linear_residual is None:
            residual = torch.zeros_like(direct)
            trend_scale = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
        else:
            residual_all = self.linear_residual(normalized_input)
            residual = residual_all.index_select(-1, target_indices)
            residual = residual + self.trend_residual(
                normalized_input, target_indices
            )
            trend_scale = torch.tanh(self.trend_residual.trend_scale_raw)

        if self.path_calibration is None:
            calibrated_direct = direct
            calibrated_residual = residual
            direct_scale_mean = torch.ones(
                (), device=direct.device, dtype=direct.dtype
            )
            residual_scale_mean = torch.ones(
                (), device=direct.device, dtype=direct.dtype
            )
            calibration_scale_std = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
            calibration_correction_abs_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
        else:
            calibration = self.path_calibration(direct, residual)
            calibrated_direct = calibration["direct"]
            calibrated_residual = calibration["residual"]
            direct_scale_mean = calibration["direct_scale_mean"]
            residual_scale_mean = calibration["residual_scale_mean"]
            calibration_scale_std = calibration["scale_std"]
            calibration_correction_abs_mean = calibration[
                "calibration_correction_abs_mean"
            ]

        if self.path_fusion is None:
            normalized_prediction = calibrated_direct + calibrated_residual
            path_weights = torch.ones(
                direct.size(0),
                direct.size(1),
                direct.size(2),
                2,
                device=direct.device,
                dtype=direct.dtype,
            )
            direct_weight_mean = torch.ones(
                (), device=direct.device, dtype=direct.dtype
            )
            residual_weight_mean = torch.ones(
                (), device=direct.device, dtype=direct.dtype
            )
            path_weight_std = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
            adaptive_correction_abs_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
        else:
            fusion = self.path_fusion(
                calibrated_direct,
                calibrated_residual,
                normalized_input,
                target_indices,
            )
            normalized_prediction = fusion["prediction"]
            path_weights = fusion["weights"]
            direct_weight_mean = fusion["direct_weight_mean"]
            residual_weight_mean = fusion["residual_weight_mean"]
            path_weight_std = fusion["path_weight_std"]
            adaptive_correction_abs_mean = fusion[
                "adaptive_correction_abs_mean"
            ]

        if isinstance(self.horizon_projection, MultiScaleTemporalProjection):
            scale_gates = self.horizon_projection.gate_values()
            scale_gate_mean = scale_gates.detach().mean()
            scale_gate_std = scale_gates.detach().std(unbiased=False)
            scale_contribution_mean = scale_diagnostics[
                "scale_contributions"
            ].mean()
            scale_contribution_std = scale_diagnostics[
                "scale_contributions"
            ].std(unbiased=False)
        else:
            scale_gate_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
            scale_gate_std = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
            scale_contribution_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
            scale_contribution_std = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )

        return {
            "direct": direct,
            "residual": residual,
            "trend_scale": trend_scale,
            "scale_gate_mean": scale_gate_mean,
            "scale_gate_std": scale_gate_std,
            "scale_contribution_mean": scale_contribution_mean,
            "scale_contribution_std": scale_contribution_std,
            "direct_weight_mean": direct_weight_mean,
            "residual_weight_mean": residual_weight_mean,
            "path_weight_std": path_weight_std,
            "path_weights": path_weights,
            "adaptive_correction_abs_mean": adaptive_correction_abs_mean,
            "direct_scale_mean": direct_scale_mean,
            "residual_scale_mean": residual_scale_mean,
            "calibration_scale_std": calibration_scale_std,
            "calibration_correction_abs_mean": calibration_correction_abs_mean,
            "direct_abs_mean": direct.detach().abs().mean(),
            "residual_abs_mean": residual.detach().abs().mean(),
            "residual_to_direct_ratio": (
                residual.detach().abs().mean()
                / direct.detach().abs().mean().clamp_min(1e-6)
            ),
            "calibrated_direct_abs_mean": calibrated_direct.detach().abs().mean(),
            "calibrated_residual_abs_mean": calibrated_residual.detach().abs().mean(),
            "calibrated_residual_to_direct_ratio": (
                calibrated_residual.detach().abs().mean()
                / calibrated_direct.detach().abs().mean().clamp_min(1e-6)
            ),
            "normalized_prediction": normalized_prediction,
        }


class HyperbolicTSF(nn.Module):
    """HAO-style dual-graph hyperbolic time-series forecaster.

    The model keeps the graph encoder independent from the forecasting head:

    ``input -> RevIN -> manifold embedding -> dual graph encoder -> direct head``.
    """

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        num_variables: int,
        output_dim: int | None = None,
        target_indices: list[int] | tuple[int, ...] | torch.Tensor | None = None,
        tangent_dim: int = 8,
        hidden_dim: int = 16,
        manifold: str = "poincare",
        trainable_curvature: bool = True,
        init_curvature: float = 1.0,
        dropout: float = 0.0,
        use_revin: bool = True,
        revin_affine: bool = True,
        revin_subtract_last: bool = False,
        use_linear_residual: bool = True,
        spatial_rank: int | None = None,
        temporal_rank: int | None = None,
        trend_window: int | None = None,
        hgcn_residual_init: float | None = None,
        use_time_identity: bool = False,
        use_multiscale_projection: bool = False,
        multiscale_factors: tuple[int, ...] = (1, 2, 4),
        use_adaptive_path_fusion: bool = False,
        use_path_amplitude_calibration: bool = False,
    ) -> None:
        super().__init__()
        if input_length <= 0 or pred_length <= 0:
            raise ValueError("input_length and pred_length must be positive")
        if num_variables <= 0:
            raise ValueError("num_variables must be positive")
        if output_dim is None:
            output_dim = num_variables
        if output_dim <= 0 or output_dim > num_variables:
            raise ValueError(
                "output_dim must be positive and no greater than num_variables"
            )

        if target_indices is None:
            target_indices = list(range(output_dim))
        target_indices = torch.as_tensor(target_indices, dtype=torch.long)
        if target_indices.ndim != 1 or target_indices.numel() != output_dim:
            raise ValueError(
                "target_indices must contain exactly output_dim indices"
            )
        if torch.any(target_indices < 0) or torch.any(target_indices >= num_variables):
            raise ValueError("target_indices contains an invalid variable index")
        if torch.unique(target_indices).numel() != target_indices.numel():
            raise ValueError("target_indices must not contain duplicates")

        self.input_length = input_length
        self.pred_length = pred_length
        self.num_variables = num_variables
        self.output_dim = output_dim
        self.use_revin = use_revin
        self.register_buffer("target_indices", target_indices)

        self.revin = RevIN(
            num_variables=num_variables,
            affine=revin_affine,
            subtract_last=revin_subtract_last,
        )
        self.embedding = ManifoldEmbedding(
            num_variables=num_variables,
            tangent_dim=tangent_dim,
            input_length=input_length if use_time_identity else None,
            manifold=manifold,
            trainable_curvature=trainable_curvature,
            init_curvature=init_curvature,
            dropout=dropout,
        )
        self.graph_encoder = DualGraphHyperbolicLayer(
            input_length=input_length,
            num_variables=num_variables,
            tangent_dim=tangent_dim,
            hidden_dim=hidden_dim,
            manifold=manifold,
            trainable_curvature=trainable_curvature,
            init_curvature=init_curvature,
            dropout=dropout,
            spatial_rank=spatial_rank,
            hgcn_residual_init=hgcn_residual_init,
        )
        self.forecast_head = DirectForecastHead(
            input_length=input_length,
            pred_length=pred_length,
            hidden_dim=hidden_dim,
            input_dim=num_variables,
            output_dim=output_dim,
            dropout=dropout,
            use_linear_residual=use_linear_residual,
            temporal_rank=temporal_rank,
            trend_window=trend_window,
            use_multiscale_projection=use_multiscale_projection,
            multiscale_factors=multiscale_factors,
            use_adaptive_path_fusion=use_adaptive_path_fusion,
            use_path_amplitude_calibration=use_path_amplitude_calibration,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, variables]")
        if x.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, got {x.size(1)}"
            )
        if x.size(2) != self.num_variables:
            raise ValueError(
                f"Expected {self.num_variables} variables, got {x.size(2)}"
            )

        if self.use_revin:
            revin_state = self.revin(x)
            normalized_input = revin_state["x"]
        else:
            normalized_input = x
            revin_state = None

        embedding = self.embedding(normalized_input)
        encoded = self.graph_encoder(embedding)
        head_output = self.forecast_head(
            encoded["temporal_context"],
            normalized_input,
            self.target_indices,
        )
        prediction_normalized = head_output["normalized_prediction"]

        if self.use_revin:
            prediction = self.revin.denormalize(
                prediction_normalized,
                revin_state,
                variable_indices=self.target_indices,
            )
        else:
            prediction = prediction_normalized

        if not return_aux:
            return prediction
        return {
            "prediction": prediction,
            "prediction_normalized": prediction_normalized,
            "normalized_input": normalized_input,
            "revin_state": revin_state,
            "embedding": embedding,
            "encoder": encoded,
            "head": head_output,
        }
