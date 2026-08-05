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


class DirectForecastHead(nn.Module):
    """Direct multi-horizon head with a trend-preserving linear residual."""

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        hidden_dim: int,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        use_linear_residual: bool = True,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_linear_residual = use_linear_residual

        self.horizon_projection = nn.Linear(input_length, pred_length)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        if use_linear_residual:
            self.linear_residual = nn.Linear(input_length, pred_length)
        else:
            self.register_module("linear_residual", None)

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

        hidden = self.horizon_projection(temporal_context.transpose(1, 2))
        hidden = hidden.transpose(1, 2)
        hidden = self.context_dropout(self.context_norm(hidden))
        direct = self.output_projection(hidden)

        if self.linear_residual is None:
            residual = torch.zeros_like(direct)
        else:
            residual_all = self.linear_residual(normalized_input.transpose(1, 2))
            residual_all = residual_all.transpose(1, 2)
            residual = residual_all.index_select(-1, target_indices)

        return {
            "direct": direct,
            "residual": residual,
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
        )
        self.forecast_head = DirectForecastHead(
            input_length=input_length,
            pred_length=pred_length,
            hidden_dim=hidden_dim,
            input_dim=num_variables,
            output_dim=output_dim,
            dropout=dropout,
            use_linear_residual=use_linear_residual,
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
