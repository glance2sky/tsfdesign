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
        level = recent[:, -1:, :].index_select(
            -1, target_indices.to(normalized_input.device)
        )
        slope = (
            recent[:, -1:, :] - recent[:, :1, :]
        ).index_select(
            -1, target_indices.to(normalized_input.device)
        ) / float(self.trend_window - 1)
        trend = (
            self.level_weight * level
            + self.slope_weight * slope * self.future_positions
        )
        return torch.tanh(self.trend_scale_raw) * trend


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


class OutputMultiScaleResidual(nn.Module):
    """Predict low-frequency corrections directly in output space.

    Each branch pools the normalized history to a coarser resolution and
    projects it to the forecast horizon. The projections start at zero, so
    enabling the module preserves the v4d function at initialization while
    keeping gradients available to the branch parameters.
    """

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        input_dim: int,
        output_dim: int,
        rank: int | None,
        factors: tuple[int, ...] = (2, 4, 8),
    ) -> None:
        super().__init__()
        if input_length <= 0 or pred_length <= 0:
            raise ValueError("input_length and pred_length must be positive")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if not factors or any(factor <= 1 for factor in factors):
            raise ValueError("output multiscale factors must be greater than 1")
        if len(set(factors)) != len(factors):
            raise ValueError("output multiscale factors must be unique")

        self.input_length = input_length
        self.pred_length = pred_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.factors = factors
        self.projections = nn.ModuleList()
        for factor in factors:
            pooled_length = max(1, (input_length + factor - 1) // factor)
            projection = LowRankTemporalProjection(
                input_length=pooled_length,
                output_length=pred_length,
                feature_dim=input_dim,
                rank=rank,
            )
            if projection.rank is None:
                nn.init.zeros_(projection.full_projection.weight)
                nn.init.zeros_(projection.full_projection.bias)
            else:
                nn.init.zeros_(projection.output_basis)
                nn.init.zeros_(projection.output_anchor)
            self.projections.append(projection)
        self.gate_raw = nn.Parameter(torch.zeros(len(factors)))

    def forward(
        self,
        normalized_input: torch.Tensor,
        target_indices: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if normalized_input.ndim != 3:
            raise ValueError(
                "normalized_input must have shape [batch, time, variables]"
            )
        if normalized_input.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, "
                f"got {normalized_input.size(1)}"
            )

        input_time_first = normalized_input.transpose(1, 2)
        gates = torch.exp(0.25 * torch.tanh(self.gate_raw))
        corrections = []
        for gate, projection, factor in zip(
            gates, self.projections, self.factors
        ):
            pooled_length = max(1, (self.input_length + factor - 1) // factor)
            pooled = F.adaptive_avg_pool1d(
                input_time_first, pooled_length
            ).transpose(1, 2)
            correction = gate * projection(pooled)
            corrections.append(correction)

        total = torch.stack(corrections, dim=0).sum(dim=0)
        target_corrections = total.index_select(
            -1, target_indices.to(normalized_input.device)
        )
        if not return_diagnostics:
            return target_corrections
        return target_corrections, {
            "gates": gates.detach(),
            "branch_contributions": torch.stack(
                [value.detach().abs().mean() for value in corrections]
            ),
        }

    def gate_values(self) -> torch.Tensor:
        return torch.exp(0.25 * torch.tanh(self.gate_raw))


class FrequencyResidual(nn.Module):
    """Learn a periodic residual from low-frequency Fourier components.

    The FFT features are deterministic and computed from the normalized
    history. A zero-initialized projection learns how much each harmonic
    should contribute, so the optional branch starts as an exact zero
    correction without blocking gradients to its learnable parameters.
    """

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        input_dim: int,
        output_dim: int,
        num_harmonics: int = 8,
    ) -> None:
        super().__init__()
        if input_length < 4:
            raise ValueError("input_length must be at least 4")
        if input_dim <= 0 or pred_length <= 0 or output_dim <= 0:
            raise ValueError(
                "input_dim, pred_length, and output_dim must be positive"
            )
        max_harmonics = input_length // 2
        if num_harmonics <= 0:
            raise ValueError("num_harmonics must be positive")
        self.input_length = input_length
        self.pred_length = pred_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = min(num_harmonics, max_harmonics)
        self.harmonic_bins = tuple(range(1, self.num_harmonics + 1))
        self.harmonic_projection = nn.Parameter(
            torch.zeros(input_dim, self.num_harmonics)
        )
        self.variable_scale_raw = nn.Parameter(torch.zeros(input_dim))
        self.gate_raw = nn.Parameter(torch.zeros(()))

        frequencies = torch.tensor(
            self.harmonic_bins, dtype=torch.float32
        ).view(1, 1, -1)
        future_positions = torch.arange(
            input_length,
            input_length + pred_length,
            dtype=torch.float32,
        ).view(1, pred_length, 1)
        phase = 2.0 * torch.pi * frequencies * future_positions / float(
            input_length
        )
        self.register_buffer("future_cos", torch.cos(phase))
        self.register_buffer("future_sin", torch.sin(phase))

    def forward(
        self,
        normalized_input: torch.Tensor,
        target_indices: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if normalized_input.ndim != 3:
            raise ValueError(
                "normalized_input must have shape [batch, time, variables]"
            )
        if normalized_input.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, "
                f"got {normalized_input.size(1)}"
            )

        centered = normalized_input - normalized_input.mean(
            dim=1, keepdim=True
        )
        spectrum = torch.fft.rfft(centered, dim=1)
        selected = spectrum[:, list(self.harmonic_bins), :]
        real = selected.real.permute(0, 2, 1)
        imag = selected.imag.permute(0, 2, 1)
        harmonic_features = (
            real.unsqueeze(1) * self.future_cos.unsqueeze(2)
            - imag.unsqueeze(1) * self.future_sin.unsqueeze(2)
        )
        harmonic_features = harmonic_features / float(self.input_length)
        correction = torch.einsum(
            "bpch,ch->bpc",
            harmonic_features,
            self.harmonic_projection,
        )
        correction = correction * torch.exp(
            0.5 * torch.tanh(self.variable_scale_raw)
        ).view(1, 1, -1)
        correction = torch.exp(0.25 * torch.tanh(self.gate_raw)) * correction
        correction = correction.index_select(
            -1, target_indices.to(normalized_input.device)
        )
        if not return_diagnostics:
            return correction
        return correction, {
            "gate": torch.exp(0.25 * torch.tanh(self.gate_raw)).detach(),
            "harmonic_contribution": correction.detach().abs().mean(),
        }


class TrendDifferenceResidual(nn.Module):
    """Bounded correction from short-vs-long level differences.

    This branch models an observed regime shift rather than learning another
    unrestricted history-to-future projection. Its fixed profiles decay over
    the forecast horizon, and all learnable amplitudes start at zero.
    """

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        output_dim: int,
        windows: tuple[int, ...] = (12, 24, 48, 96),
        max_amplitude: float = 0.25,
    ) -> None:
        super().__init__()
        if input_length < 4 or pred_length <= 0 or output_dim <= 0:
            raise ValueError(
                "input_length must be at least 4 and pred_length/output_dim "
                "must be positive"
            )
        if not windows:
            raise ValueError("windows must not be empty")
        if any(window < 2 for window in windows):
            raise ValueError("trend windows must be at least 2")
        if len(set(windows)) != len(windows):
            raise ValueError("trend windows must be unique")
        if max_amplitude <= 0:
            raise ValueError("max_amplitude must be positive")

        self.input_length = input_length
        self.pred_length = pred_length
        self.output_dim = output_dim
        self.windows = windows
        self.max_amplitude = max_amplitude
        self.amplitude_raw = nn.Parameter(
            torch.zeros(len(windows), output_dim)
        )

        positions = torch.arange(
            1, pred_length + 1, dtype=torch.float32
        ).view(1, pred_length, 1)
        window_tensor = torch.as_tensor(windows, dtype=torch.float32).view(
            -1, 1, 1
        )
        profiles = torch.exp(-positions / window_tensor)
        profiles = profiles / profiles[:, :1].clamp_min(1e-6)
        self.register_buffer("profiles", profiles)

    def forward(
        self,
        normalized_input: torch.Tensor,
        target_indices: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if normalized_input.ndim != 3:
            raise ValueError(
                "normalized_input must have shape [batch, time, variables]"
            )
        if normalized_input.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, "
                f"got {normalized_input.size(1)}"
            )

        context_mean = normalized_input.mean(dim=1)
        differences = torch.stack(
            [
                normalized_input[:, -window:].mean(dim=1) - context_mean
                for window in self.windows
            ],
            dim=1,
        )
        differences = differences.index_select(
            -1, target_indices.to(normalized_input.device)
        )
        amplitudes = self.max_amplitude * torch.tanh(self.amplitude_raw)
        correction = torch.zeros(
            normalized_input.size(0),
            self.pred_length,
            self.output_dim,
            device=normalized_input.device,
            dtype=normalized_input.dtype,
        )
        for index in range(len(self.windows)):
            correction = correction + (
                differences[:, index].unsqueeze(1)
                * self.profiles[index].to(normalized_input.dtype)
                * amplitudes[index].view(1, 1, -1)
            )
        correction = correction / float(len(self.windows))
        correction = correction.index_select(
            -1, torch.arange(
                self.output_dim,
                device=normalized_input.device,
            )
        )
        if not return_diagnostics:
            return correction
        return correction, {
            "amplitude_abs_mean": amplitudes.detach().abs().mean(),
            "branch_contribution": correction.detach().abs().mean(),
            "difference_abs_mean": differences.detach().abs().mean(),
        }


class ExplicitPeriodicResidual(nn.Module):
    """Bounded residual on explicitly specified periodic bases.

    The basis period is independent of ``input_length``. This avoids tying
    the extrapolation frequency to FFT bins that may not match the known
    sampling period.
    """

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        output_dim: int,
        periods: tuple[int, ...] = (12, 24, 48),
        max_amplitude: float = 0.25,
    ) -> None:
        super().__init__()
        if input_length < 4 or pred_length <= 0 or output_dim <= 0:
            raise ValueError(
                "input_length must be at least 4 and pred_length/output_dim "
                "must be positive"
            )
        if not periods:
            raise ValueError("periods must not be empty")
        if any(period < 2 for period in periods):
            raise ValueError("periods must be at least 2")
        if len(set(periods)) != len(periods):
            raise ValueError("periods must be unique")
        if max_amplitude <= 0:
            raise ValueError("max_amplitude must be positive")

        self.input_length = input_length
        self.pred_length = pred_length
        self.output_dim = output_dim
        self.periods = periods
        self.max_amplitude = max_amplitude
        self.amplitude_raw = nn.Parameter(
            torch.zeros(len(periods), output_dim, 2)
        )

        history_positions = torch.arange(
            input_length, dtype=torch.float32
        ).view(1, input_length)
        future_positions = torch.arange(
            input_length,
            input_length + pred_length,
            dtype=torch.float32,
        ).view(1, pred_length)
        period_tensor = torch.as_tensor(periods, dtype=torch.float32).view(
            -1, 1
        )
        history_phase = 2.0 * torch.pi * history_positions / period_tensor
        future_phase = 2.0 * torch.pi * future_positions / period_tensor
        self.register_buffer("history_cos", torch.cos(history_phase))
        self.register_buffer("history_sin", torch.sin(history_phase))
        self.register_buffer("future_cos", torch.cos(future_phase))
        self.register_buffer("future_sin", torch.sin(future_phase))

    def forward(
        self,
        normalized_input: torch.Tensor,
        target_indices: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if normalized_input.ndim != 3:
            raise ValueError(
                "normalized_input must have shape [batch, time, variables]"
            )
        if normalized_input.size(1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, "
                f"got {normalized_input.size(1)}"
            )

        centered = normalized_input - normalized_input.mean(
            dim=1, keepdim=True
        )
        centered = centered.index_select(
            -1, target_indices.to(normalized_input.device)
        )
        cosine_coeff = torch.einsum(
            "blv,pl->bpv", centered, self.history_cos
        ) / float(self.input_length)
        sine_coeff = torch.einsum(
            "blv,pl->bpv", centered, self.history_sin
        ) / float(self.input_length)
        coefficients = torch.stack((cosine_coeff, sine_coeff), dim=-1)
        future_basis = torch.stack(
            (self.future_cos, self.future_sin), dim=-1
        )
        amplitudes = self.max_amplitude * torch.tanh(self.amplitude_raw)
        correction = torch.einsum(
            "bpvc,phc,pvc->bhv",
            coefficients,
            future_basis,
            amplitudes,
        )
        correction = correction / float(len(self.periods))
        if not return_diagnostics:
            return correction
        return correction, {
            "amplitude_abs_mean": amplitudes.detach().abs().mean(),
            "basis_contribution": correction.detach().abs().mean(),
            "coefficient_abs_mean": coefficients.detach().abs().mean(),
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
        use_output_multiscale_residual: bool = False,
        output_multiscale_factors: tuple[int, ...] = (2, 4, 8),
        use_frequency_residual: bool = False,
        frequency_harmonics: int = 8,
        use_trend_difference_residual: bool = False,
        trend_difference_windows: tuple[int, ...] = (12, 24, 48, 96),
        trend_difference_max_amplitude: float = 0.25,
        use_explicit_periodic_residual: bool = False,
        explicit_periods: tuple[int, ...] = (12, 24, 48),
        explicit_periodic_max_amplitude: float = 0.25,
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
        self.use_output_multiscale_residual = use_output_multiscale_residual
        self.use_frequency_residual = use_frequency_residual
        self.use_trend_difference_residual = use_trend_difference_residual
        self.use_explicit_periodic_residual = use_explicit_periodic_residual
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

        if use_output_multiscale_residual:
            self.output_multiscale_residual = OutputMultiScaleResidual(
                input_length=input_length,
                pred_length=pred_length,
                input_dim=input_dim,
                output_dim=output_dim,
                rank=self.temporal_rank,
                factors=output_multiscale_factors,
            )
        else:
            self.register_module("output_multiscale_residual", None)
        if use_frequency_residual:
            self.frequency_residual = FrequencyResidual(
                input_length=input_length,
                pred_length=pred_length,
                input_dim=input_dim,
                output_dim=output_dim,
                num_harmonics=frequency_harmonics,
            )
        else:
            self.register_module("frequency_residual", None)
        if use_trend_difference_residual:
            self.trend_difference_residual = TrendDifferenceResidual(
                input_length=input_length,
                pred_length=pred_length,
                output_dim=output_dim,
                windows=trend_difference_windows,
                max_amplitude=trend_difference_max_amplitude,
            )
        else:
            self.register_module("trend_difference_residual", None)
        if use_explicit_periodic_residual:
            self.explicit_periodic_residual = ExplicitPeriodicResidual(
                input_length=input_length,
                pred_length=pred_length,
                output_dim=output_dim,
                periods=explicit_periods,
                max_amplitude=explicit_periodic_max_amplitude,
            )
        else:
            self.register_module("explicit_periodic_residual", None)

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

        if self.output_multiscale_residual is None:
            output_multiscale_correction = torch.zeros_like(normalized_prediction)
            output_multiscale_gate_mean = torch.ones(
                (), device=direct.device, dtype=direct.dtype
            )
            output_multiscale_contribution_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
        else:
            (
                output_multiscale_correction,
                output_multiscale_diagnostics,
            ) = self.output_multiscale_residual(
                normalized_input,
                target_indices,
                return_diagnostics=True,
            )
            output_multiscale_gate_mean = (
                output_multiscale_diagnostics["gates"].mean()
            )
            output_multiscale_contribution_mean = (
                output_multiscale_diagnostics["branch_contributions"].mean()
            )

        if self.frequency_residual is None:
            frequency_correction = torch.zeros_like(normalized_prediction)
            frequency_gate = torch.ones(
                (), device=direct.device, dtype=direct.dtype
            )
            frequency_contribution_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
        else:
            frequency_correction, frequency_diagnostics = self.frequency_residual(
                normalized_input,
                target_indices,
                return_diagnostics=True,
            )
            frequency_gate = frequency_diagnostics["gate"]
            frequency_contribution_mean = frequency_diagnostics[
                "harmonic_contribution"
            ]

        output_residual = (
            output_multiscale_correction + frequency_correction
        )
        if self.trend_difference_residual is None:
            trend_difference_correction = torch.zeros_like(
                normalized_prediction
            )
            trend_difference_amplitude_abs_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
            trend_difference_contribution_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
        else:
            (
                trend_difference_correction,
                trend_difference_diagnostics,
            ) = self.trend_difference_residual(
                normalized_input,
                target_indices,
                return_diagnostics=True,
            )
            trend_difference_amplitude_abs_mean = (
                trend_difference_diagnostics["amplitude_abs_mean"]
            )
            trend_difference_contribution_mean = (
                trend_difference_diagnostics["branch_contribution"]
            )

        if self.explicit_periodic_residual is None:
            explicit_periodic_correction = torch.zeros_like(
                normalized_prediction
            )
            explicit_periodic_amplitude_abs_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
            explicit_periodic_contribution_mean = torch.zeros(
                (), device=direct.device, dtype=direct.dtype
            )
        else:
            (
                explicit_periodic_correction,
                explicit_periodic_diagnostics,
            ) = self.explicit_periodic_residual(
                normalized_input,
                target_indices,
                return_diagnostics=True,
            )
            explicit_periodic_amplitude_abs_mean = (
                explicit_periodic_diagnostics["amplitude_abs_mean"]
            )
            explicit_periodic_contribution_mean = (
                explicit_periodic_diagnostics["basis_contribution"]
            )

        output_residual = (
            output_residual
            + trend_difference_correction
            + explicit_periodic_correction
        )
        normalized_prediction = normalized_prediction + output_residual

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
            "output_multiscale_gate_mean": output_multiscale_gate_mean,
            "output_multiscale_contribution_mean": (
                output_multiscale_contribution_mean
            ),
            "frequency_gate": frequency_gate,
            "frequency_contribution_mean": frequency_contribution_mean,
            "trend_difference_amplitude_abs_mean": (
                trend_difference_amplitude_abs_mean
            ),
            "trend_difference_contribution_mean": (
                trend_difference_contribution_mean
            ),
            "explicit_periodic_amplitude_abs_mean": (
                explicit_periodic_amplitude_abs_mean
            ),
            "explicit_periodic_contribution_mean": (
                explicit_periodic_contribution_mean
            ),
            "output_residual_abs_mean": output_residual.detach().abs().mean(),
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
        use_output_multiscale_residual: bool = False,
        output_multiscale_factors: tuple[int, ...] = (2, 4, 8),
        use_frequency_residual: bool = False,
        frequency_harmonics: int = 8,
        use_trend_difference_residual: bool = False,
        trend_difference_windows: tuple[int, ...] = (12, 24, 48, 96),
        trend_difference_max_amplitude: float = 0.25,
        use_explicit_periodic_residual: bool = False,
        explicit_periods: tuple[int, ...] = (12, 24, 48),
        explicit_periodic_max_amplitude: float = 0.25,
        use_variable_hierarchy: bool = False,
        variable_hierarchy_groups: int = 3,
        use_temporal_hierarchy: bool = False,
        temporal_hierarchy_factors: tuple[int, ...] = (2, 4, 8),
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
            use_variable_hierarchy=use_variable_hierarchy,
            variable_hierarchy_groups=variable_hierarchy_groups,
            use_temporal_hierarchy=use_temporal_hierarchy,
            temporal_hierarchy_factors=temporal_hierarchy_factors,
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
            use_output_multiscale_residual=use_output_multiscale_residual,
            output_multiscale_factors=output_multiscale_factors,
            use_frequency_residual=use_frequency_residual,
            frequency_harmonics=frequency_harmonics,
            use_trend_difference_residual=use_trend_difference_residual,
            trend_difference_windows=trend_difference_windows,
            trend_difference_max_amplitude=trend_difference_max_amplitude,
            use_explicit_periodic_residual=use_explicit_periodic_residual,
            explicit_periods=explicit_periods,
            explicit_periodic_max_amplitude=explicit_periodic_max_amplitude,
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
