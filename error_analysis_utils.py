"""Shared helpers for leakage-free HyperbolicTSF error analysis."""

from __future__ import annotations

import torch

from hypertsf_layers import HyperbolicTSF


def decompose_head_forecasts(
    model: HyperbolicTSF,
    direct: torch.Tensor,
    residual: torch.Tensor,
    revin_state: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Return valid branch forecasts in the target coordinate system.

    ``direct`` and ``residual`` are produced in the RevIN-normalized
    coordinate system.  Because RevIN denormalization includes a location
    term, independently denormalized branches cannot be added together.
    This helper exposes the shared base and the two additive contributions:

        prediction = base + direct_contribution + residual_contribution

    ``direct_forecast`` and ``residual_forecast`` are the predictions obtained
    when only the corresponding branch is retained, including the shared
    RevIN base.
    """
    if direct.ndim != 3 or residual.ndim != 3:
        raise ValueError("direct and residual must have shape [batch, time, variables]")
    if direct.shape != residual.shape:
        raise ValueError("direct and residual must have the same shape")

    normalized_prediction = direct + residual
    if revin_state is None:
        base = torch.zeros_like(direct)
        direct_forecast = direct
        residual_forecast = residual
        prediction = normalized_prediction
    else:
        target_indices = model.target_indices.to(direct.device)
        zeros = torch.zeros_like(direct)
        base = model.revin.denormalize(
            zeros,
            revin_state,
            variable_indices=target_indices,
        )
        direct_forecast = model.revin.denormalize(
            direct,
            revin_state,
            variable_indices=target_indices,
        )
        residual_forecast = model.revin.denormalize(
            residual,
            revin_state,
            variable_indices=target_indices,
        )
        prediction = model.revin.denormalize(
            normalized_prediction,
            revin_state,
            variable_indices=target_indices,
        )

    direct_contribution = direct_forecast - base
    residual_contribution = residual_forecast - base
    reconstructed_prediction = (
        base + direct_contribution + residual_contribution
    )

    return {
        "base_forecast": base,
        "direct_forecast": direct_forecast,
        "residual_forecast": residual_forecast,
        "direct_contribution": direct_contribution,
        "residual_contribution": residual_contribution,
        "prediction": prediction,
        "reconstructed_prediction": reconstructed_prediction,
    }
