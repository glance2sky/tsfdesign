"""Numerically stable origin-based maps for Euclidean and hyperbolic spaces."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ManifoldSpace(nn.Module):
    """Origin-based Euclidean, Poincare-ball, or Lorentz-space operations.

    The public curvature parameter is positive. For the Lorentz model it is
    interpreted as sectional curvature, so the hyperboloid radius-squared is
    ``K = 1 / curvature``.
    """

    def __init__(
        self,
        name: str = "poincare",
        trainable_curvature: bool = True,
        init_curvature: float = 1.0,
    ) -> None:
        super().__init__()
        name = name.lower()
        aliases = {
            "euclidean": "euclidean",
            "e": "euclidean",
            "poincare": "poincare",
            "poincareball": "poincare",
            "p": "poincare",
            "lorentz": "lorentz",
            "lorentzian": "lorentz",
            "hyperboloid": "lorentz",
            "h": "lorentz",
        }
        if name not in aliases:
            raise ValueError(f"Unsupported manifold: {name}")
        if init_curvature <= 0:
            raise ValueError("init_curvature must be positive")
        self.name = aliases[name]
        self.trainable_curvature = trainable_curvature
        if self.name == "euclidean":
            self.register_buffer("_euclidean_curvature", torch.ones(()))
        else:
            raw = torch.log(torch.expm1(torch.tensor(float(init_curvature))))
            self.raw_curvature = nn.Parameter(
                raw, requires_grad=trainable_curvature
            )

    @property
    def curvature(self) -> torch.Tensor:
        if self.name == "euclidean":
            return self._euclidean_curvature
        return F.softplus(self.raw_curvature).clamp_min(1e-4)

    @property
    def is_lorentz(self) -> bool:
        return self.name == "lorentz"

    def manifold_dim(self, tangent_dim: int) -> int:
        return tangent_dim + 1 if self.is_lorentz else tangent_dim

    def expmap0(self, tangent: torch.Tensor) -> torch.Tensor:
        """Map origin tangent vectors to the selected manifold."""
        if self.name == "euclidean":
            return tangent
        if self.name == "poincare":
            return self._poincare_expmap0(tangent)
        return self._lorentz_expmap0(tangent)

    def logmap0(self, point: torch.Tensor) -> torch.Tensor:
        """Map manifold points back to the origin tangent space."""
        if self.name == "euclidean":
            return point
        if self.name == "poincare":
            return self._poincare_logmap0(point)
        return self._lorentz_logmap0(point)

    def project(self, point: torch.Tensor) -> torch.Tensor:
        if self.name == "euclidean":
            return point
        if self.name == "poincare":
            c = self.curvature
            eps = 1e-5 if point.dtype == torch.float64 else 4e-3
            norm = point.norm(dim=-1, keepdim=True).clamp_min(1e-15)
            max_norm = (1.0 - eps) / torch.sqrt(c)
            projected = point / norm * max_norm
            return torch.where(norm > max_norm, projected, point)
        return self._lorentz_project(point)

    def pairwise_sqdist(self, points: torch.Tensor) -> torch.Tensor:
        """Return pairwise squared geodesic distances over the penultimate axis."""
        if points.ndim < 2:
            raise ValueError("points must have at least two dimensions")
        if self.name == "euclidean":
            return torch.cdist(points, points, p=2).pow(2)
        if self.name == "poincare":
            left = points.unsqueeze(-2)
            right = points.unsqueeze(-3)
            delta = self._mobius_add(-left, right, c=self.curvature)
            norm = delta.norm(dim=-1).clamp_min(1e-15)
            c = self.curvature
            eps = 1e-5 if points.dtype == torch.float64 else 4e-3
            argument = (torch.sqrt(c) * norm).clamp(
                min=0.0, max=1.0 - eps
            )
            distance = 2.0 / torch.sqrt(c) * torch.atanh(argument)
            return distance.pow(2)
        left = points.unsqueeze(-2)
        right = points.unsqueeze(-3)
        dot = self._minkowski_dot(left, right)
        radius_sq = 1.0 / self.curvature
        argument = (-dot / radius_sq).clamp_min(1.0 + 1e-6)
        distance = torch.sqrt(radius_sq) * torch.acosh(argument)
        return distance.pow(2)

    def mobius_matvec(
        self,
        weight: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a manifold-native linear map to manifold points.

        This is the origin-based Mobius matrix-vector operation used by HNN
        layers. The matrix acts on the tangent coordinates, while the returned
        value is mapped back to the manifold.
        """
        if weight.ndim != 2:
            raise ValueError("weight must have shape [out_dim, in_dim]")
        tangent = self.logmap0(point)
        tangent = torch.matmul(tangent, weight.transpose(-1, -2))
        return self.expmap0(tangent)

    def mobius_add(
        self,
        point: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        """Add two manifold points at the origin."""
        if self.name == "euclidean":
            return point + other
        if self.name == "poincare":
            return self._mobius_add(point, other, c=self.curvature)
        return self.expmap0(self.logmap0(point) + self.logmap0(other))

    def manifold_bias(
        self,
        point: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Add a trainable tangent bias and return a manifold point."""
        bias = bias.reshape(*([1] * (point.ndim - 1)), -1)
        return self.mobius_add(point, self.expmap0(bias))

    def _poincare_expmap0(self, tangent: torch.Tensor) -> torch.Tensor:
        c = self.curvature
        sqrt_c = torch.sqrt(c)
        norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        return torch.tanh(sqrt_c * norm) * tangent / (sqrt_c * norm)

    def _poincare_logmap0(self, point: torch.Tensor) -> torch.Tensor:
        c = self.curvature
        sqrt_c = torch.sqrt(c)
        point = self.project(point)
        norm = point.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        argument = (sqrt_c * norm).clamp(
            min=0.0, max=1.0 - (1e-5 if point.dtype == torch.float64 else 4e-3)
        )
        return torch.atanh(argument) * point / (sqrt_c * norm)

    def _lorentz_expmap0(self, tangent: torch.Tensor) -> torch.Tensor:
        curvature = self.curvature
        radius_sq = 1.0 / curvature
        radius = torch.sqrt(radius_sq)
        norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        theta = norm / radius
        time = radius * torch.cosh(theta)
        spatial = radius * torch.sinh(theta) * tangent / norm
        return self._lorentz_project(torch.cat([time, spatial], dim=-1))

    def _lorentz_logmap0(self, point: torch.Tensor) -> torch.Tensor:
        point = self._lorentz_project(point)
        curvature = self.curvature
        radius = torch.rsqrt(curvature)
        spatial = point[..., 1:]
        spatial_norm = spatial.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        theta = torch.acosh((point[..., :1] / radius).clamp_min(1.0 + 1e-6))
        return radius * theta * spatial / spatial_norm

    def _lorentz_project(self, point: torch.Tensor) -> torch.Tensor:
        radius_sq = 1.0 / self.curvature
        spatial = point[..., 1:]
        time = torch.sqrt(
            (radius_sq + spatial.pow(2).sum(dim=-1, keepdim=True)).clamp_min(1e-8)
        )
        return torch.cat([time, spatial], dim=-1)

    @staticmethod
    def _mobius_add(
        x: torch.Tensor, y: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        if c is None:
            raise ValueError("curvature is required")
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + c.pow(2) * x2 * y2
        return numerator / denominator.clamp_min(1e-15)

    def _minkowski_dot(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return (x * y).sum(dim=-1) - 2 * x[..., 0] * y[..., 0]
