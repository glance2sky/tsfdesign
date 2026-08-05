"""End-to-end hyperbolic graph forecaster."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

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

        self.horizon_projection = LowRankTemporalProjection(
            input_length=input_length,
            output_length=pred_length,
            feature_dim=hidden_dim,
            rank=self.temporal_rank,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

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
        else:
            self.register_module("linear_residual", None)
            self.register_module("trend_residual", None)

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

        hidden = self.horizon_projection(temporal_context)
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

        return {
            "direct": direct,
            "residual": residual,
            "trend_scale": trend_scale,
            "direct_abs_mean": direct.detach().abs().mean(),
            "residual_abs_mean": residual.detach().abs().mean(),
            "residual_to_direct_ratio": (
                residual.detach().abs().mean()
                / direct.detach().abs().mean().clamp_min(1e-6)
            ),
            "normalized_prediction": direct + residual,
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
