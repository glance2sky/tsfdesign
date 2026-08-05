"""First three layers of the HAO-style hyperbolic TSF front-end."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .manifolds import ManifoldSpace


class RevIN(nn.Module):
    """Layer 1: reversible instance normalization over the time axis."""

    def __init__(
        self,
        num_variables: int,
        affine: bool = True,
        subtract_last: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_variables <= 0:
            raise ValueError("num_variables must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.num_variables = num_variables
        self.affine = affine
        self.subtract_last = subtract_last
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_variables))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_variables))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_series(x, self.num_variables)
        location = (
            x[:, -1:, :] if self.subtract_last else x.mean(dim=1, keepdim=True)
        ).detach()
        scale = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False).detach() + self.eps
        )
        normalized = (x - location) / scale
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return {"x": normalized, "location": location, "scale": scale}

    def denormalize(
        self,
        values: torch.Tensor,
        state: dict[str, torch.Tensor],
        variable_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, time, variables]")
        if variable_indices is None:
            if values.size(-1) != self.num_variables:
                raise ValueError(
                    f"Expected {self.num_variables} variables, got {values.size(-1)}"
                )
            weight = self.weight
            bias = self.bias
            location = state["location"]
            scale = state["scale"]
        else:
            indices = torch.as_tensor(
                variable_indices, device=values.device, dtype=torch.long
            )
            if indices.ndim != 1 or indices.numel() != values.size(-1):
                raise ValueError(
                    "variable_indices must contain one index per value variable"
                )
            if indices.numel() == 0 or torch.any(indices < 0):
                raise ValueError("variable_indices must contain valid variables")
            if torch.any(indices >= self.num_variables):
                raise ValueError("variable_indices contains an out-of-range index")
            weight = None if self.weight is None else self.weight.index_select(-1, indices)
            bias = None if self.bias is None else self.bias.index_select(-1, indices)
            location = state["location"].index_select(-1, indices)
            scale = state["scale"].index_select(-1, indices)
        if self.affine:
            denominator = torch.where(
                weight.abs() < self.eps,
                torch.full_like(weight, self.eps),
                weight,
            )
            values = (values - bias) / denominator
        return values * scale + location


class ManifoldEmbedding(nn.Module):
    """Layer 2: preserve variables while mapping values to a manifold."""

    def __init__(
        self,
        num_variables: int,
        tangent_dim: int,
        manifold: str = "poincare",
        trainable_curvature: bool = True,
        init_curvature: float = 1.0,
        dropout: float = 0.0,
        input_length: int | None = None,
    ) -> None:
        super().__init__()
        if num_variables <= 0 or tangent_dim <= 0:
            raise ValueError("num_variables and tangent_dim must be positive")
        if input_length is not None and input_length <= 0:
            raise ValueError("input_length must be positive when provided")
        self.num_variables = num_variables
        self.tangent_dim = tangent_dim
        self.input_length = input_length
        self.space = ManifoldSpace(
            manifold,
            trainable_curvature=trainable_curvature,
            init_curvature=init_curvature,
        )
        self.value_projection = nn.Linear(1, tangent_dim)
        self.variable_identity = nn.Parameter(
            torch.empty(1, 1, num_variables, tangent_dim)
        )
        if input_length is None:
            self.register_parameter("time_identity", None)
        else:
            self.time_identity = nn.Parameter(
                torch.empty(1, input_length, 1, tangent_dim)
            )
        self.norm = nn.LayerNorm(tangent_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.variable_identity, mean=0.0, std=0.02)
        if self.time_identity is not None:
            nn.init.normal_(self.time_identity, mean=0.0, std=0.02)

    @property
    def manifold_dim(self) -> int:
        return self.space.manifold_dim(self.tangent_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_series(x, self.num_variables)
        tangent = self.value_projection(x.unsqueeze(-1))
        tangent = tangent + self.variable_identity
        if self.time_identity is not None:
            if x.size(1) != self.input_length:
                raise ValueError(
                    f"Expected input length {self.input_length}, got {x.size(1)}"
                )
            tangent = tangent + self.time_identity
        tangent = self.norm(tangent)
        tangent = self.dropout(tangent)
        return {
            "tangent": tangent,
            "manifold": self.space.expmap0(tangent),
            "curvature": self.space.curvature,
        }


class _HyperbolicNeuralProjection(nn.Module):
    """HNN-style manifold linear map, bias, and hyperbolic activation."""

    def __init__(
        self,
        space: ManifoldSpace,
        in_features: int,
        out_features: int,
        dropout: float,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        self.space = space
        if rank is None:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.register_parameter("left_weight", None)
            self.register_parameter("right_weight", None)
        else:
            if rank <= 0:
                raise ValueError("rank must be positive")
            rank = min(rank, in_features, out_features)
            self.register_parameter("weight", None)
            self.left_weight = nn.Parameter(torch.empty(out_features, rank))
            self.right_weight = nn.Parameter(torch.empty(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        if rank is None:
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.xavier_uniform_(self.left_weight)
            nn.init.xavier_uniform_(self.right_weight)

    def forward(self, tangent: torch.Tensor) -> torch.Tensor:
        point = self.space.expmap0(tangent)
        if self.weight is None:
            weight = self.left_weight @ self.right_weight
        else:
            weight = self.weight
        point = self.space.mobius_matvec(weight, point)
        point = self.space.manifold_bias(point, self.bias)
        activation = torch.tanh(self.space.logmap0(point))
        activation = self.dropout(activation)
        return self.space.expmap0(activation)


class _HyperbolicGraphConvolution(nn.Module):
    """HGCN block with manifold-native linear and graph aggregation."""

    def __init__(
        self,
        space: ManifoldSpace,
        hidden_dim: int,
        dropout: float,
        residual_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.space = space
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        if not 0.0 <= residual_init <= 1.0:
            raise ValueError("residual_init must be in [0, 1]")
        residual_logit = torch.logit(torch.tensor(residual_init).clamp(1e-4, 1 - 1e-4))
        self.residual_logit = nn.Parameter(residual_logit)
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        points: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        input_tangent = self.space.logmap0(points)
        points = self.space.mobius_matvec(self.weight, points)
        tangent = self.space.logmap0(points)
        tangent = torch.bmm(adjacency, tangent)
        tangent = self.dropout(tangent)
        tangent = torch.tanh(tangent)
        residual_weight = torch.sigmoid(self.residual_logit)
        tangent = (
            (1.0 - residual_weight) * tangent
            + residual_weight * input_tangent
        )
        points = self.space.expmap0(tangent)
        return self.space.manifold_bias(points, self.bias)


class DualGraphHyperbolicLayer(nn.Module):
    """Layer 3: parallel temporal and variable HNN/HGCN branches.

    The input is ``[B, L, C, D]`` from layer 2. The variable branch treats
    each variable as a node with feature size ``L * D``. The temporal branch
    treats each time point as a node with feature size ``C * D``. Each branch
    independently learns a prior graph, applies HNN and HGCN operations, and
    refines the graph from pairwise manifold distances.
    """

    def __init__(
        self,
        input_length: int,
        num_variables: int,
        tangent_dim: int,
        hidden_dim: int = 16,
        manifold: str = "poincare",
        trainable_curvature: bool = True,
        init_curvature: float = 1.0,
        dropout: float = 0.0,
        spatial_rank: int | None = None,
        hgcn_residual_init: float = 0.5,
    ) -> None:
        super().__init__()
        if input_length <= 0 or num_variables <= 0:
            raise ValueError("input_length and num_variables must be positive")
        if tangent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("tangent_dim and hidden_dim must be positive")

        self.input_length = input_length
        self.num_variables = num_variables
        self.tangent_dim = tangent_dim
        self.hidden_dim = hidden_dim
        self.spatial_rank = spatial_rank
        self.spatial_space = ManifoldSpace(
            manifold,
            trainable_curvature=trainable_curvature,
            init_curvature=init_curvature,
        )
        self.temporal_space = ManifoldSpace(
            manifold,
            trainable_curvature=trainable_curvature,
            init_curvature=init_curvature,
        )

        spatial_input_dim = input_length * tangent_dim
        temporal_input_dim = num_variables * tangent_dim
        self.spatial_hnn = _HyperbolicNeuralProjection(
            self.spatial_space,
            spatial_input_dim,
            hidden_dim,
            dropout,
            rank=spatial_rank,
        )
        self.temporal_hnn = _HyperbolicNeuralProjection(
            self.temporal_space, temporal_input_dim, hidden_dim, dropout
        )
        self.spatial_hgcn = _HyperbolicGraphConvolution(
            self.spatial_space,
            hidden_dim,
            dropout,
            residual_init=hgcn_residual_init,
        )
        self.temporal_hgcn = _HyperbolicGraphConvolution(
            self.temporal_space,
            hidden_dim,
            dropout,
            residual_init=hgcn_residual_init,
        )

        self.spatial_graph_logits = nn.Parameter(
            torch.zeros(num_variables, num_variables)
        )
        self.temporal_graph_logits = nn.Parameter(
            torch.zeros(input_length, input_length)
        )
        self.spatial_graph_mix_logit = nn.Parameter(torch.tensor(-1.0))
        self.temporal_graph_mix_logit = nn.Parameter(torch.tensor(-1.0))
        self.spatial_temperature_logit = nn.Parameter(torch.tensor(0.0))
        self.temporal_temperature_logit = nn.Parameter(torch.tensor(0.0))
        self.spatial_output_norm = nn.LayerNorm(hidden_dim)
        self.temporal_output_norm = nn.LayerNorm(hidden_dim)
        self.variable_context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        layer2_output: dict[str, torch.Tensor] | torch.Tensor,
    ) -> dict[str, Any]:
        tangent = (
            layer2_output["tangent"]
            if isinstance(layer2_output, dict)
            else layer2_output
        )
        if tangent.ndim != 4:
            raise ValueError(
                "layer2 tangent must have shape [batch, time, variables, tangent_dim]"
            )
        batch, length, variables, dimension = tangent.shape
        expected = (self.input_length, self.num_variables, self.tangent_dim)
        if (length, variables, dimension) != expected:
            raise ValueError(
                "Expected layer2 shape "
                f"[B, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(tangent.shape)}"
            )

        # Two parallel views, matching HAO's spatial/temporal decomposition.
        spatial_input = tangent.permute(0, 2, 1, 3).reshape(
            batch, variables, length * dimension
        )
        temporal_input = tangent.reshape(batch, length, variables * dimension)

        spatial_prior = self._learned_graph(
            self.spatial_graph_logits, batch, tangent
        )
        temporal_prior = self._learned_graph(
            self.temporal_graph_logits, batch, tangent
        )

        spatial_points = self.spatial_hnn(spatial_input)
        temporal_points = self.temporal_hnn(temporal_input)
        spatial_hidden = self.spatial_hgcn(spatial_points, spatial_prior)
        temporal_hidden = self.temporal_hgcn(temporal_points, temporal_prior)

        spatial_distance = self.spatial_space.pairwise_sqdist(spatial_hidden)
        temporal_distance = self.temporal_space.pairwise_sqdist(temporal_hidden)
        spatial_dynamic = self._distance_graph(
            spatial_distance, self.spatial_temperature_logit
        )
        temporal_dynamic = self._distance_graph(
            temporal_distance, self.temporal_temperature_logit
        )

        spatial_mix = torch.sigmoid(self.spatial_graph_mix_logit)
        temporal_mix = torch.sigmoid(self.temporal_graph_mix_logit)
        spatial_graph = self._blend_graph(
            spatial_prior, spatial_dynamic, spatial_mix
        )
        temporal_graph = self._blend_graph(
            temporal_prior, temporal_dynamic, temporal_mix
        )

        # Re-aggregate with the learned graph so the dynamic graph affects the
        # representation used by the prediction head, not only an auxiliary
        # diagnostic output.
        spatial_hidden = self.spatial_hgcn(spatial_hidden, spatial_graph)
        temporal_hidden = self.temporal_hgcn(temporal_hidden, temporal_graph)

        spatial_tangent = self.spatial_space.logmap0(spatial_hidden)
        temporal_tangent = self.temporal_space.logmap0(temporal_hidden)
        spatial_tangent = self.spatial_output_norm(spatial_tangent)
        temporal_tangent = self.temporal_output_norm(temporal_tangent)
        interaction = torch.einsum(
            "bch,blh->bcl", spatial_tangent, temporal_tangent
        )
        variable_weights = F.softmax(
            interaction.transpose(1, 2), dim=-1
        )
        variable_context = torch.bmm(variable_weights, spatial_tangent)
        variable_context = self.variable_context_projection(variable_context)
        fusion_input = torch.cat((temporal_tangent, variable_context), dim=-1)
        fusion_gate = torch.sigmoid(self.fusion_gate(fusion_input))
        temporal_context = temporal_tangent + fusion_gate * variable_context

        return {
            "spatial_nodes": self.spatial_space.expmap0(spatial_tangent),
            "temporal_nodes": self.temporal_space.expmap0(temporal_tangent),
            "spatial_tangent": spatial_tangent,
            "temporal_tangent": temporal_tangent,
            "temporal_context": temporal_context,
            "interaction": interaction,
            "variable_weights": variable_weights,
            "variable_context": variable_context,
            "fusion_gate": fusion_gate,
            "spatial_graph": spatial_graph,
            "temporal_graph": temporal_graph,
            "spatial_prior": spatial_prior,
            "temporal_prior": temporal_prior,
            "spatial_dynamic": spatial_dynamic,
            "temporal_dynamic": temporal_dynamic,
            "spatial_graph_mix": spatial_mix,
            "temporal_graph_mix": temporal_mix,
            "spatial_graph_entropy": self._normalized_graph_entropy(spatial_graph),
            "temporal_graph_entropy": self._normalized_graph_entropy(
                temporal_graph
            ),
            "spatial_prior_dynamic_gap": (
                spatial_prior - spatial_dynamic
            ).abs().mean(),
            "temporal_prior_dynamic_gap": (
                temporal_prior - temporal_dynamic
            ).abs().mean(),
            "spatial_tangent_norm": spatial_tangent.norm(dim=-1).mean(),
            "temporal_tangent_norm": temporal_tangent.norm(dim=-1).mean(),
            "variable_weight_entropy": self._normalized_graph_entropy(
                variable_weights
            ),
            "fusion_gate_mean": fusion_gate.mean(),
            "fusion_gate_std": fusion_gate.std(unbiased=False),
            "spatial_distance": spatial_distance,
            "temporal_distance": temporal_distance,
            "spatial_curvature": self.spatial_space.curvature,
            "temporal_curvature": self.temporal_space.curvature,
        }

    def _learned_graph(
        self,
        logits: torch.Tensor,
        batch: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        symmetric_logits = 0.5 * (logits + logits.transpose(-1, -2))
        graph = F.softmax(symmetric_logits, dim=-1)
        identity = torch.eye(
            logits.size(0), device=reference.device, dtype=reference.dtype
        )
        graph = 0.5 * (graph + identity)
        graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return graph.unsqueeze(0).expand(batch, -1, -1)

    def _distance_graph(
        self,
        distance: torch.Tensor,
        temperature_logit: torch.Tensor,
    ) -> torch.Tensor:
        temperature = F.softplus(temperature_logit) + 1e-4
        graph = F.softmax(-distance / temperature, dim=-1)
        identity = torch.eye(
            distance.size(-1),
            device=distance.device,
            dtype=distance.dtype,
        ).unsqueeze(0)
        graph = 0.5 * (graph + identity)
        return graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _blend_graph(
        prior: torch.Tensor,
        dynamic: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        graph = (1.0 - mix) * prior + mix * dynamic
        return graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _normalized_graph_entropy(graph: torch.Tensor) -> torch.Tensor:
        entropy = -(
            graph.clamp_min(1e-8) * graph.clamp_min(1e-8).log()
        ).sum(dim=-1)
        maximum = torch.log(
            torch.tensor(
                graph.size(-1),
                device=graph.device,
                dtype=graph.dtype,
            )
        ).clamp_min(1e-8)
        return (entropy / maximum).mean()

def _check_series(x: torch.Tensor, num_variables: int) -> None:
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, time, variables]")
    if x.size(-1) != num_variables:
        raise ValueError(
            f"Expected {num_variables} variables, got {x.size(-1)}"
        )
