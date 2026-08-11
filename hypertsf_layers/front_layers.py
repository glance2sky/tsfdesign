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
        use_patch_tokens: bool = False,
        patch_lengths: tuple[int, ...] = (8, 16, 32),
        patch_strides: tuple[int, ...] | None = None,
        patch_hidden_dim: int | None = None,
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
        if use_patch_tokens:
            if input_length is None:
                raise ValueError(
                    "input_length is required when use_patch_tokens=True"
                )
            self.patch_encoder = HyperbolicPatchTokenEncoder(
                self.space,
                input_length=input_length,
                num_variables=num_variables,
                tangent_dim=tangent_dim,
                hidden_dim=(
                    tangent_dim
                    if patch_hidden_dim is None
                    else patch_hidden_dim
                ),
                patch_lengths=patch_lengths,
                patch_strides=patch_strides,
                dropout=dropout,
            )
        else:
            self.register_module("patch_encoder", None)
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
        if self.patch_encoder is None:
            patch_diagnostics = {
                "patch_scale_gate_mean": tangent.new_zeros(()),
                "patch_scale_gate_std": tangent.new_zeros(()),
                "patch_local_contribution": tangent.new_zeros(()),
                "patch_token_contribution": tangent.new_zeros(()),
                "patch_correction_abs_mean": tangent.new_zeros(()),
                "patch_token_entropy": tangent.new_zeros(()),
            }
        else:
            patch_points, patch_diagnostics = self.patch_encoder(x)
            patch_tangent = self.space.logmap0(patch_points)
            tangent = tangent + patch_tangent
        return {
            "tangent": tangent,
            "manifold": self.space.expmap0(tangent),
            "curvature": self.space.curvature,
            **patch_diagnostics,
        }


class HyperbolicPatchTokenEncoder(nn.Module):
    """Multi-scale patch and variable-token encoder for the manifold front-end.

    Patch extraction is channel-independent. Patch summaries are then treated
    as variable tokens and mixed across variables with self-attention, in the
    spirit of inverted time-series Transformers. The resulting local patch
    signal and variable-token signal are projected to a tangent correction and
    fused with the point-wise embedding using Mobius addition.
    """

    def __init__(
        self,
        space: ManifoldSpace,
        input_length: int,
        num_variables: int,
        tangent_dim: int,
        hidden_dim: int,
        patch_lengths: tuple[int, ...] = (8, 16, 32),
        patch_strides: tuple[int, ...] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_length <= 0 or num_variables <= 0:
            raise ValueError(
                "input_length and num_variables must be positive"
            )
        if tangent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("tangent_dim and hidden_dim must be positive")
        if not patch_lengths:
            raise ValueError("patch_lengths must not be empty")
        if any(
            patch_length < 2 or patch_length > input_length
            for patch_length in patch_lengths
        ):
            raise ValueError(
                "patch lengths must be in [2, input_length]"
            )
        if len(set(patch_lengths)) != len(patch_lengths):
            raise ValueError("patch lengths must be unique")
        if patch_strides is None:
            patch_strides = tuple(
                max(1, patch_length // 2)
                for patch_length in patch_lengths
            )
        if len(patch_strides) != len(patch_lengths):
            raise ValueError(
                "patch_strides must contain one stride per patch length"
            )
        if any(stride <= 0 for stride in patch_strides):
            raise ValueError("patch strides must be positive")

        self.space = space
        self.input_length = input_length
        self.num_variables = num_variables
        self.tangent_dim = tangent_dim
        self.hidden_dim = hidden_dim
        self.patch_lengths = patch_lengths
        self.patch_strides = patch_strides

        self.patch_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(patch_length, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
                for patch_length in patch_lengths
            ]
        )
        num_heads = self._attention_heads(hidden_dim)
        self.variable_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.variable_norm = nn.LayerNorm(hidden_dim)
        self.attended_token_projection = nn.Linear(hidden_dim, tangent_dim)
        self.patch_to_tangent = nn.ModuleList(
            [nn.Linear(hidden_dim, tangent_dim) for _ in patch_lengths]
        )
        self.token_to_tangent = nn.Linear(
            hidden_dim * len(patch_lengths),
            tangent_dim,
        )
        self.scale_gate_raw = nn.Parameter(
            torch.zeros(len(patch_lengths))
        )
        self.token_gate_raw = nn.Parameter(torch.zeros(()))
        for projection in self.patch_to_tangent:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.zeros_(self.token_to_tangent.weight)
        nn.init.zeros_(self.token_to_tangent.bias)
        nn.init.zeros_(self.attended_token_projection.weight)
        nn.init.zeros_(self.attended_token_projection.bias)

    def forward(
        self,
        normalized_input: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if normalized_input.ndim != 3:
            raise ValueError(
                "normalized_input must have shape [batch, time, variables]"
            )
        batch, length, variables = normalized_input.shape
        if (length, variables) != (
            self.input_length,
            self.num_variables,
        ):
            raise ValueError(
                "Expected input shape "
                f"[B, {self.input_length}, {self.num_variables}], "
                f"got {tuple(normalized_input.shape)}"
            )

        input_channel_first = normalized_input.transpose(1, 2)
        scale_gates = torch.exp(
            0.25 * torch.tanh(self.scale_gate_raw)
        )
        reconstructed = torch.zeros(
            batch,
            variables,
            length,
            self.tangent_dim,
            device=normalized_input.device,
            dtype=normalized_input.dtype,
        )
        reconstruction_weight = torch.zeros(
            1,
            1,
            length,
            1,
            device=normalized_input.device,
            dtype=normalized_input.dtype,
        )
        token_features = []
        local_contributions = []

        for scale_index, (
            patch_length,
            stride,
            projection,
            tangent_projection,
        ) in enumerate(
            zip(
                self.patch_lengths,
                self.patch_strides,
                self.patch_projections,
                self.patch_to_tangent,
            )
        ):
            patches = input_channel_first.unfold(
                dimension=2,
                size=patch_length,
                step=stride,
            )
            # [B, C, num_patches, patch_length] -> [B, C, num_patches, H]
            features = projection(patches)
            token_features.append(features.mean(dim=2))
            patch_tangent = tangent_projection(features)
            starts = range(
                0,
                self.input_length - patch_length + 1,
                stride,
            )
            for patch_index, start in enumerate(starts):
                indices = torch.arange(
                    start,
                    start + patch_length,
                    device=normalized_input.device,
                )
                source = patch_tangent[:, :, patch_index, :]
                source = source.unsqueeze(2).expand(
                    -1,
                    -1,
                    patch_length,
                    -1,
                )
                reconstructed = reconstructed.index_add(
                    2,
                    indices,
                    scale_gates[scale_index] * source,
                )
                reconstruction_weight = reconstruction_weight.index_add(
                    2,
                    indices,
                    torch.ones_like(
                        reconstruction_weight[:, :, :patch_length, :]
                    ),
                )
            local_contributions.append(
                patch_tangent.detach().abs().mean()
            )

        token_input = torch.cat(token_features, dim=-1)
        # Compress the multi-scale token to the shared attention width.
        token_chunks = token_input.chunk(len(self.patch_lengths), dim=-1)
        token_input = torch.stack(token_chunks, dim=0).mean(dim=0)
        attended_tokens, _ = self.variable_attention(
            token_input,
            token_input,
            token_input,
            need_weights=False,
        )
        attended_tokens = self.variable_norm(
            token_input + attended_tokens
        )
        token_correction = self.token_to_tangent(
            torch.cat(token_features, dim=-1)
        )
        attended_correction = self.attended_token_projection(
            attended_tokens
        )
        token_correction = token_correction + attended_correction
        token_correction = (
            torch.exp(0.25 * torch.tanh(self.token_gate_raw))
            * token_correction
        )
        token_correction = token_correction.unsqueeze(2).expand(
            -1,
            -1,
            length,
            -1,
        )
        reconstructed = reconstructed / reconstruction_weight.clamp_min(1.0)
        correction = reconstructed + token_correction
        correction = correction / float(len(self.patch_lengths))
        correction = correction.permute(0, 2, 1, 3)
        patch_points = self.space.expmap0(correction)
        return patch_points, {
            "patch_scale_gate_mean": scale_gates.detach().mean(),
            "patch_scale_gate_std": scale_gates.detach().std(
                unbiased=False
            ),
            "patch_local_contribution": torch.stack(
                local_contributions
            ).mean(),
            "patch_token_contribution": token_correction.detach().abs().mean(),
            "patch_correction_abs_mean": correction.detach().abs().mean(),
            "patch_token_entropy": self._normalized_entropy(
                torch.softmax(
                    attended_tokens.detach().norm(dim=-1),
                    dim=-1,
                )
            ),
        }

    @staticmethod
    def _attention_heads(hidden_dim: int) -> int:
        for candidate in (8, 4, 2, 1):
            if hidden_dim % candidate == 0:
                return candidate
        return 1

    @staticmethod
    def _normalized_entropy(values: torch.Tensor) -> torch.Tensor:
        entropy = -(
            values.clamp_min(1e-8) * values.clamp_min(1e-8).log()
        ).sum(dim=-1)
        maximum = torch.log(
            torch.tensor(
                values.size(-1),
                device=values.device,
                dtype=values.dtype,
            )
        ).clamp_min(1e-8)
        return (entropy / maximum).mean()


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
        residual_init: float | None = None,
    ) -> None:
        super().__init__()
        self.space = space
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        if residual_init is None:
            self.register_parameter("residual_logit", None)
        else:
            if not 0.0 <= residual_init <= 1.0:
                raise ValueError("residual_init must be in [0, 1]")
            residual_logit = torch.logit(
                torch.tensor(residual_init).clamp(1e-4, 1 - 1e-4)
            )
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
        if self.residual_logit is not None:
            residual_weight = torch.sigmoid(self.residual_logit)
            tangent = (
                (1.0 - residual_weight) * tangent
                + residual_weight * input_tangent
            )
        points = self.space.expmap0(tangent)
        return self.space.manifold_bias(points, self.bias)


class HyperbolicVariableHierarchy(nn.Module):
    """Learn a variable -> group -> global hierarchy on the manifold.

    Variable nodes are softly assigned to latent groups. Group representations
    are hyperbolic barycenter approximations in the origin tangent space,
    updated by a group-level HGCN, and then propagated back to the leaves via
    Mobius addition. The residual mix starts at zero, so the optional module
    preserves the previous variable branch at initialization while keeping a
    useful gradient path.
    """

    def __init__(
        self,
        space: ManifoldSpace,
        num_variables: int,
        hidden_dim: int,
        num_groups: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_variables <= 0 or hidden_dim <= 0 or num_groups <= 0:
            raise ValueError(
                "num_variables, hidden_dim, and num_groups must be positive"
            )
        if num_groups > num_variables:
            raise ValueError(
                "num_groups must not exceed num_variables"
            )
        self.space = space
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups

        self.assignment_logits = nn.Parameter(
            torch.empty(num_variables, num_groups)
        )
        self.assignment_projection = nn.Linear(hidden_dim, num_groups)
        self.group_graph_logits = nn.Parameter(
            torch.zeros(num_groups, num_groups)
        )
        self.group_temperature_logit = nn.Parameter(torch.zeros(()))
        self.group_graph_mix_logit = nn.Parameter(torch.tensor(-1.0))
        self.group_hgcn = _HyperbolicGraphConvolution(
            space,
            hidden_dim,
            dropout,
        )
        self.global_projection = _HyperbolicNeuralProjection(
            space,
            hidden_dim,
            hidden_dim,
            dropout,
        )
        self.child_projection = _HyperbolicNeuralProjection(
            space,
            hidden_dim,
            hidden_dim,
            dropout,
        )
        self.hierarchy_mix_raw = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.assignment_logits, mean=0.0, std=0.02)
        nn.init.zeros_(self.assignment_projection.weight)
        nn.init.zeros_(self.assignment_projection.bias)
        # The child path starts at the manifold origin. This preserves the
        # previous leaf representation while keeping gradients to the new
        # hierarchy branch non-zero from the first update.
        nn.init.zeros_(self.child_projection.weight)
        nn.init.zeros_(self.child_projection.bias)

    def forward(
        self,
        leaf_points: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if leaf_points.ndim != 3:
            raise ValueError(
                "leaf_points must have shape [batch, variables, hidden_dim]"
            )
        batch, variables, hidden_dim = leaf_points.shape
        if (variables, hidden_dim) != (
            self.num_variables,
            self.hidden_dim,
        ):
            raise ValueError(
                "Expected leaf shape "
                f"[B, {self.num_variables}, {self.hidden_dim}], "
                f"got {tuple(leaf_points.shape)}"
            )

        leaf_tangent = self.space.logmap0(leaf_points)
        assignment_logits = (
            self.assignment_logits.unsqueeze(0)
            + self.assignment_projection(leaf_tangent)
        )
        assignment = F.softmax(assignment_logits, dim=-1)
        assignment_mass = assignment.sum(dim=1).clamp_min(1e-6)
        group_tangent = torch.einsum(
            "bcg,bch->bgh", assignment, leaf_tangent
        )
        group_tangent = group_tangent / assignment_mass.unsqueeze(-1)
        group_points = self.space.expmap0(group_tangent)

        group_prior = self._learned_graph(
            self.group_graph_logits,
            batch,
            group_points,
        )
        group_hidden = self.group_hgcn(group_points, group_prior)
        group_distance = self.space.pairwise_sqdist(group_hidden)
        group_dynamic = self._distance_graph(
            group_distance,
            self.group_temperature_logit,
        )
        group_mix = torch.sigmoid(self.group_graph_mix_logit)
        group_graph = (
            (1.0 - group_mix) * group_prior + group_mix * group_dynamic
        )
        group_graph = group_graph / group_graph.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        group_hidden = self.group_hgcn(group_hidden, group_graph)
        group_tangent = self.space.logmap0(group_hidden)

        global_tangent = group_tangent.mean(dim=1)
        global_message = self.global_projection(global_tangent)
        global_tangent = self.space.logmap0(global_message)
        group_message = group_tangent + global_tangent.unsqueeze(1)
        leaf_group_message = torch.einsum(
            "bcg,bgh->bch", assignment, group_message
        )
        child_message = self.child_projection(leaf_group_message)
        propagated_points = self.space.mobius_add(
            leaf_points,
            child_message,
        )
        hierarchy_mix = torch.exp(
            0.25 * torch.tanh(self.hierarchy_mix_raw)
        )
        propagated_tangent = self.space.logmap0(propagated_points)
        updated_tangent = (
            leaf_tangent
            + hierarchy_mix * (propagated_tangent - leaf_tangent)
        )
        updated_points = self.space.expmap0(updated_tangent)

        return updated_points, {
            "assignment": assignment,
            "assignment_entropy": self._normalized_entropy(assignment),
            "group_graph": group_graph,
            "group_graph_entropy": self._normalized_entropy(group_graph),
            "group_graph_mix": group_mix,
            "hierarchy_mix": hierarchy_mix,
            "hierarchy_contribution": (
                updated_tangent - leaf_tangent
            ).detach().abs().mean(),
            "leaf_tangent_norm": leaf_tangent.detach().norm(dim=-1).mean(),
            "group_tangent_norm": group_tangent.detach().norm(dim=-1).mean(),
            "global_tangent_norm": global_tangent.detach().norm(dim=-1).mean(),
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
            logits.size(0),
            device=reference.device,
            dtype=reference.dtype,
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
    def _normalized_entropy(values: torch.Tensor) -> torch.Tensor:
        entropy = -(
            values.clamp_min(1e-8) * values.clamp_min(1e-8).log()
        ).sum(dim=-1)
        maximum = torch.log(
            torch.tensor(
                values.size(-1),
                device=values.device,
                dtype=values.dtype,
            )
        ).clamp_min(1e-8)
        return (entropy / maximum).mean()


class HyperbolicTemporalHierarchy(nn.Module):
    """Fixed multi-resolution temporal hierarchy on a hyperbolic manifold.

    Unlike the variable hierarchy, temporal membership is determined by
    position and therefore does not suffer from assignment symmetry. Each
    level pools contiguous time nodes, performs manifold graph propagation,
    adds a global context, and broadcasts a bounded correction back to the
    fine time nodes.
    """

    def __init__(
        self,
        space: ManifoldSpace,
        input_length: int,
        hidden_dim: int,
        factors: tuple[int, ...] = (2, 4, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_length <= 0 or hidden_dim <= 0:
            raise ValueError("input_length and hidden_dim must be positive")
        if not factors or any(factor <= 1 for factor in factors):
            raise ValueError("temporal hierarchy factors must be greater than 1")
        if len(set(factors)) != len(factors):
            raise ValueError("temporal hierarchy factors must be unique")
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.factors = factors
        self.space = space
        self.levels = nn.ModuleList()
        self.graph_logits = nn.ParameterList()
        self.temperature_logits = nn.ParameterList()
        self.graph_mix_logits = nn.ParameterList()
        self.assignments = []

        for index, factor in enumerate(factors):
            group_count = (input_length + factor - 1) // factor
            assignment = torch.zeros(group_count, input_length)
            for time_index in range(input_length):
                group_index = min(time_index // factor, group_count - 1)
                assignment[group_index, time_index] = 1.0
            assignment = assignment / assignment.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            self.register_buffer(f"assignment_{index}", assignment)
            self.assignments.append(f"assignment_{index}")
            self.levels.append(
                nn.ModuleDict(
                    {
                        "hgcn": _HyperbolicGraphConvolution(
                            space,
                            hidden_dim,
                            dropout,
                        ),
                        "projection": _HyperbolicNeuralProjection(
                            space,
                            hidden_dim,
                            hidden_dim,
                            dropout,
                        ),
                        "child_projection": _HyperbolicNeuralProjection(
                            space,
                            hidden_dim,
                            hidden_dim,
                            dropout,
                        ),
                    }
                )
            )
            self.graph_logits.append(
                nn.Parameter(torch.zeros(group_count, group_count))
            )
            self.temperature_logits.append(nn.Parameter(torch.zeros(())))
            self.graph_mix_logits.append(
                nn.Parameter(torch.tensor(-1.0))
            )
            nn.init.zeros_(self.levels[-1]["child_projection"].weight)
            nn.init.zeros_(self.levels[-1]["child_projection"].bias)

        self.global_projection = _HyperbolicNeuralProjection(
            space,
            hidden_dim,
            hidden_dim,
            dropout,
        )
        nn.init.zeros_(self.global_projection.weight)
        nn.init.zeros_(self.global_projection.bias)
        self.hierarchy_mix_raw = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        temporal_points: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if temporal_points.ndim != 3:
            raise ValueError(
                "temporal_points must have shape [batch, time, hidden_dim]"
            )
        batch, length, hidden_dim = temporal_points.shape
        if (length, hidden_dim) != (self.input_length, self.hidden_dim):
            raise ValueError(
                "Expected temporal shape "
                f"[B, {self.input_length}, {self.hidden_dim}], "
                f"got {tuple(temporal_points.shape)}"
            )

        fine_tangent = self.space.logmap0(temporal_points)
        corrections = []
        graph_entropies = []
        graph_mixes = []
        level_contributions = []
        current_global = fine_tangent.mean(dim=1)

        for level_index, level in enumerate(self.levels):
            assignment = getattr(self, self.assignments[level_index])
            assignment = assignment.to(
                device=temporal_points.device,
                dtype=temporal_points.dtype,
            )
            group_tangent = torch.einsum(
                "gl,blh->bgh", assignment, fine_tangent
            )
            group_points = self.space.expmap0(group_tangent)
            prior = self._learned_graph(
                self.graph_logits[level_index],
                batch,
                group_points,
            )
            group_hidden = level["hgcn"](group_points, prior)
            distance = self.space.pairwise_sqdist(group_hidden)
            dynamic = self._distance_graph(
                distance,
                self.temperature_logits[level_index],
            )
            mix = torch.sigmoid(self.graph_mix_logits[level_index])
            graph = (1.0 - mix) * prior + mix * dynamic
            graph = graph / graph.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            group_hidden = level["hgcn"](group_hidden, graph)
            group_tangent = self.space.logmap0(group_hidden)
            current_global = 0.5 * (
                current_global + group_tangent.mean(dim=1)
            )
            global_message = self.global_projection(current_global)
            global_tangent = self.space.logmap0(global_message)
            group_message = level["projection"](group_tangent)
            group_message = self.space.logmap0(group_message)
            group_message = group_message + global_tangent.unsqueeze(1)
            broadcast = torch.einsum(
                "lg,bgh->blh",
                assignment.transpose(0, 1),
                group_message,
            )
            child_message = level["child_projection"](
                self.space.expmap0(broadcast)
            )
            child_tangent = self.space.logmap0(child_message)
            corrections.append(child_tangent)
            level_contributions.append(child_tangent.detach().abs().mean())
            graph_entropies.append(self._normalized_entropy(graph))
            graph_mixes.append(mix.detach())

        hierarchy_correction = torch.stack(corrections, dim=0).mean(dim=0)
        hierarchy_mix = torch.exp(
            0.25 * torch.tanh(self.hierarchy_mix_raw)
        )
        updated_tangent = fine_tangent + hierarchy_mix * hierarchy_correction
        updated_points = self.space.expmap0(updated_tangent)
        return updated_points, {
            "temporal_hierarchy_mix": hierarchy_mix,
            "temporal_hierarchy_contribution": (
                hierarchy_correction.detach().abs().mean()
            ),
            "temporal_level_contribution": torch.stack(level_contributions),
            "temporal_level_graph_entropy": torch.stack(graph_entropies),
            "temporal_level_graph_mix": torch.stack(graph_mixes),
            "temporal_hierarchy_fine_norm": fine_tangent.detach().norm(
                dim=-1
            ).mean(),
            "temporal_hierarchy_global_norm": current_global.detach().norm(
                dim=-1
            ).mean(),
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
            logits.size(0),
            device=reference.device,
            dtype=reference.dtype,
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
    def _normalized_entropy(values: torch.Tensor) -> torch.Tensor:
        entropy = -(
            values.clamp_min(1e-8) * values.clamp_min(1e-8).log()
        ).sum(dim=-1)
        maximum = torch.log(
            torch.tensor(
                values.size(-1),
                device=values.device,
                dtype=values.dtype,
            )
        ).clamp_min(1e-8)
        return (entropy / maximum).mean()


class RecursiveHyperbolicTemporalHierarchy(nn.Module):
    """Recursive parent-child temporal hierarchy with shared hyperbolic blocks.

    ``factors`` are per-level coarsening ratios. For example, ``(2, 2, 2)``
    maps 96 fine nodes to 48, then 24, then 12 coarse nodes. The upward pass
    uses one shared HGCN and a local sparse graph at every level. The
    top-down pass broadcasts information through the fixed parent-child maps.
    The downward projection is zero initialized, so the module is an exact
    identity at initialization while its projection parameters still receive
    gradients.
    """

    def __init__(
        self,
        space: ManifoldSpace,
        input_length: int,
        hidden_dim: int,
        factors: tuple[int, ...] = (2, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_length <= 0 or hidden_dim <= 0:
            raise ValueError("input_length and hidden_dim must be positive")
        if not factors or any(factor <= 1 for factor in factors):
            raise ValueError(
                "recursive temporal factors must be greater than 1"
            )
        self.space = space
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.factors = factors

        self.assignments = []
        self.local_adjacencies = []
        child_length = input_length
        for index, factor in enumerate(factors):
            parent_length = (child_length + factor - 1) // factor
            assignment = torch.zeros(parent_length, child_length)
            for child_index in range(child_length):
                parent_index = min(child_index // factor, parent_length - 1)
                assignment[parent_index, child_index] = 1.0
            assignment = assignment / assignment.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            adjacency = self._local_adjacency(parent_length)
            assignment_name = f"assignment_{index}"
            adjacency_name = f"local_adjacency_{index}"
            self.register_buffer(assignment_name, assignment)
            self.register_buffer(adjacency_name, adjacency)
            self.assignments.append(assignment_name)
            self.local_adjacencies.append(adjacency_name)
            child_length = parent_length

        self.shared_hgcn = _HyperbolicGraphConvolution(
            space,
            hidden_dim,
            dropout,
        )
        self.shared_down_projection = _HyperbolicNeuralProjection(
            space,
            hidden_dim,
            hidden_dim,
            dropout,
        )
        nn.init.zeros_(self.shared_down_projection.weight)
        nn.init.zeros_(self.shared_down_projection.bias)
        self.global_projection = _HyperbolicNeuralProjection(
            space,
            hidden_dim,
            hidden_dim,
            dropout,
        )
        self.temperature_logit = nn.Parameter(torch.zeros(()))
        self.graph_mix_logit = nn.Parameter(torch.tensor(-1.0))
        self.hierarchy_mix_raw = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        temporal_points: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if temporal_points.ndim != 3:
            raise ValueError(
                "temporal_points must have shape [batch, time, hidden_dim]"
            )
        batch, length, hidden_dim = temporal_points.shape
        if (length, hidden_dim) != (self.input_length, self.hidden_dim):
            raise ValueError(
                "Expected temporal shape "
                f"[B, {self.input_length}, {self.hidden_dim}], "
                f"got {tuple(temporal_points.shape)}"
            )

        level_points = [temporal_points]
        level_contributions = []
        graph_entropies = []
        graph_mix_values = []
        current_points = temporal_points

        for assignment_name, adjacency_name in zip(
            self.assignments,
            self.local_adjacencies,
        ):
            assignment = getattr(self, assignment_name).to(
                device=temporal_points.device,
                dtype=temporal_points.dtype,
            )
            local_adjacency = getattr(self, adjacency_name).to(
                device=temporal_points.device,
                dtype=temporal_points.dtype,
            )
            current_tangent = self.space.logmap0(current_points)
            pooled_tangent = torch.einsum(
                "pc,bch->bph",
                assignment,
                current_tangent,
            )
            pooled_points = self.space.expmap0(pooled_tangent)
            distance = self.space.pairwise_sqdist(pooled_points)
            dynamic = self._distance_graph(
                distance,
                self.temperature_logit,
            )
            mix = torch.sigmoid(self.graph_mix_logit)
            graph = (1.0 - mix) * local_adjacency.unsqueeze(0) + (
                mix * dynamic
            )
            graph = graph / graph.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            current_points = self.shared_hgcn(pooled_points, graph)
            level_points.append(current_points)
            graph_entropies.append(self._normalized_entropy(graph))
            graph_mix_values.append(mix.detach())
            level_contributions.append(
                self.space.logmap0(current_points).detach().abs().mean()
            )

        top_tangent = self.space.logmap0(level_points[-1])
        global_tangent = top_tangent.mean(dim=1)
        global_message = self.global_projection(global_tangent)
        global_tangent = self.space.logmap0(global_message)
        down_points = self.space.expmap0(
            top_tangent + global_tangent.unsqueeze(1)
        )

        for level_index in range(len(self.assignments) - 1, -1, -1):
            assignment = getattr(
                self,
                self.assignments[level_index],
            ).to(
                device=temporal_points.device,
                dtype=temporal_points.dtype,
            )
            child_base = level_points[level_index]
            parent_tangent = self.space.logmap0(down_points)
            broadcast = torch.einsum(
                "pc,bph->bch",
                assignment,
                parent_tangent,
            )
            message = self.shared_down_projection(
                self.space.expmap0(broadcast)
            )
            child_tangent = self.space.logmap0(child_base) + self.space.logmap0(
                message
            )
            down_points = self.space.expmap0(child_tangent)

        fine_tangent = self.space.logmap0(temporal_points)
        updated_tangent = self.space.logmap0(down_points)
        correction = updated_tangent - fine_tangent
        hierarchy_mix = torch.exp(
            0.25 * torch.tanh(self.hierarchy_mix_raw)
        )
        updated_points = self.space.expmap0(
            fine_tangent + hierarchy_mix * correction
        )
        level_contributions_tensor = torch.stack(level_contributions)
        graph_entropies_tensor = torch.stack(graph_entropies)
        graph_mix_tensor = torch.stack(graph_mix_values)
        return updated_points, {
            "recursive_temporal_hierarchy_mix": hierarchy_mix,
            "recursive_temporal_hierarchy_contribution": (
                correction.detach().abs().mean()
            ),
            "recursive_temporal_level_contribution": (
                level_contributions_tensor
            ),
            "recursive_temporal_level_contribution_mean": (
                level_contributions_tensor.mean()
            ),
            "recursive_temporal_level_contribution_std": (
                level_contributions_tensor.std(unbiased=False)
            ),
            "recursive_temporal_level_graph_entropy": graph_entropies_tensor,
            "recursive_temporal_level_graph_entropy_mean": (
                graph_entropies_tensor.mean()
            ),
            "recursive_temporal_level_graph_entropy_std": (
                graph_entropies_tensor.std(unbiased=False)
            ),
            "recursive_temporal_level_graph_mix": graph_mix_tensor,
            "recursive_temporal_level_graph_mix_mean": graph_mix_tensor.mean(),
            "recursive_temporal_hierarchy_fine_norm": (
                fine_tangent.detach().norm(dim=-1).mean()
            ),
            "recursive_temporal_hierarchy_global_norm": (
                global_tangent.detach().norm(dim=-1).mean()
            ),
            "recursive_temporal_hierarchy_depth": torch.tensor(
                len(self.factors),
                device=temporal_points.device,
                dtype=temporal_points.dtype,
            ),
        }

    @staticmethod
    def _local_adjacency(size: int) -> torch.Tensor:
        adjacency = torch.eye(size)
        if size > 1:
            adjacency += torch.diag(torch.ones(size - 1), diagonal=1)
            adjacency += torch.diag(torch.ones(size - 1), diagonal=-1)
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)

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
    def _normalized_entropy(values: torch.Tensor) -> torch.Tensor:
        entropy = -(
            values.clamp_min(1e-8) * values.clamp_min(1e-8).log()
        ).sum(dim=-1)
        maximum = torch.log(
            torch.tensor(
                values.size(-1),
                device=values.device,
                dtype=values.dtype,
            )
        ).clamp_min(1e-8)
        return (entropy / maximum).mean()


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
        hgcn_residual_init: float | None = None,
        use_variable_hierarchy: bool = False,
        variable_hierarchy_groups: int = 3,
        use_temporal_hierarchy: bool = False,
        temporal_hierarchy_factors: tuple[int, ...] = (2, 4, 8),
        use_recursive_temporal_hierarchy: bool = False,
        recursive_temporal_factors: tuple[int, ...] = (2, 2, 2),
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
        if use_variable_hierarchy:
            self.variable_hierarchy = HyperbolicVariableHierarchy(
                self.spatial_space,
                num_variables=num_variables,
                hidden_dim=hidden_dim,
                num_groups=variable_hierarchy_groups,
                dropout=dropout,
            )
        else:
            self.register_module("variable_hierarchy", None)
        if use_temporal_hierarchy:
            self.temporal_hierarchy = HyperbolicTemporalHierarchy(
                self.temporal_space,
                input_length=input_length,
                hidden_dim=hidden_dim,
                factors=temporal_hierarchy_factors,
                dropout=dropout,
            )
        else:
            self.register_module("temporal_hierarchy", None)
        if use_recursive_temporal_hierarchy:
            self.recursive_temporal_hierarchy = (
                RecursiveHyperbolicTemporalHierarchy(
                    self.temporal_space,
                    input_length=input_length,
                    hidden_dim=hidden_dim,
                    factors=recursive_temporal_factors,
                    dropout=dropout,
                )
            )
        else:
            self.register_module("recursive_temporal_hierarchy", None)

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

        if self.variable_hierarchy is None:
            hierarchy_diagnostics = {
                "assignment_entropy": spatial_graph.new_zeros(()),
                "group_graph_entropy": spatial_graph.new_zeros(()),
                "group_graph_mix": spatial_graph.new_zeros(()),
                "hierarchy_mix": spatial_graph.new_zeros(()),
                "hierarchy_contribution": spatial_graph.new_zeros(()),
                "leaf_tangent_norm": spatial_graph.new_zeros(()),
                "group_tangent_norm": spatial_graph.new_zeros(()),
                "global_tangent_norm": spatial_graph.new_zeros(()),
            }
        else:
            spatial_hidden, hierarchy_diagnostics = self.variable_hierarchy(
                spatial_hidden
            )

        if self.temporal_hierarchy is None:
            temporal_hierarchy_diagnostics = {
                "temporal_hierarchy_mix": temporal_graph.new_zeros(()),
                "temporal_hierarchy_contribution": temporal_graph.new_zeros(
                    ()
                ),
                "temporal_level_contribution": temporal_graph.new_zeros(0),
                "temporal_level_graph_entropy": temporal_graph.new_zeros(0),
                "temporal_level_graph_mix": temporal_graph.new_zeros(0),
                "temporal_hierarchy_fine_norm": temporal_graph.new_zeros(()),
                "temporal_hierarchy_global_norm": temporal_graph.new_zeros(
                    ()
                ),
            }
        else:
            temporal_hidden, temporal_hierarchy_diagnostics = (
                self.temporal_hierarchy(temporal_hidden)
            )

        if self.recursive_temporal_hierarchy is None:
            recursive_temporal_hierarchy_diagnostics = {
                "recursive_temporal_hierarchy_mix": temporal_graph.new_zeros(
                    ()
                ),
                "recursive_temporal_hierarchy_contribution": (
                    temporal_graph.new_zeros(())
                ),
                "recursive_temporal_level_contribution_mean": (
                    temporal_graph.new_zeros(())
                ),
                "recursive_temporal_level_contribution_std": (
                    temporal_graph.new_zeros(())
                ),
                "recursive_temporal_level_graph_entropy_mean": (
                    temporal_graph.new_zeros(())
                ),
                "recursive_temporal_level_graph_entropy_std": (
                    temporal_graph.new_zeros(())
                ),
                "recursive_temporal_level_graph_mix_mean": (
                    temporal_graph.new_zeros(())
                ),
                "recursive_temporal_hierarchy_fine_norm": (
                    temporal_graph.new_zeros(())
                ),
                "recursive_temporal_hierarchy_global_norm": (
                    temporal_graph.new_zeros(())
                ),
                "recursive_temporal_hierarchy_depth": (
                    temporal_graph.new_zeros(())
                ),
            }
        else:
            (
                temporal_hidden,
                recursive_temporal_hierarchy_diagnostics,
            ) = self.recursive_temporal_hierarchy(temporal_hidden)

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
            "variable_hierarchy_assignment": (
                self.variable_hierarchy is not None
            ),
            **hierarchy_diagnostics,
            **temporal_hierarchy_diagnostics,
            **recursive_temporal_hierarchy_diagnostics,
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
