import pytest
import torch

from hypertsf_layers import (
    DualGraphHyperbolicLayer,
    ManifoldEmbedding,
    RevIN,
)
from hypertsf_layers.manifolds import ManifoldSpace


def test_revin_normalizes_and_denormalizes_window() -> None:
    layer = RevIN(num_variables=3, affine=True)
    x = torch.randn(4, 12, 3) * torch.tensor([1.0, 3.0, 10.0]) + torch.tensor(
        [5.0, -2.0, 20.0]
    )
    state = layer(x)
    normalized = state["x"]
    assert normalized.shape == x.shape
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(4, 3), atol=1e-5)
    restored = layer.denormalize(normalized, state)
    assert torch.allclose(restored, x, atol=1e-5)


@pytest.mark.parametrize(
    "manifold,ambient_offset",
    [("euclidean", 0), ("poincare", 0), ("lorentz", 1)],
)
def test_manifold_embedding_preserves_variable_axis(
    manifold: str,
    ambient_offset: int,
) -> None:
    layer = ManifoldEmbedding(
        num_variables=3,
        tangent_dim=6,
        manifold=manifold,
        trainable_curvature=True,
    )
    output = layer(torch.randn(2, 16, 3))
    assert output["tangent"].shape == (2, 16, 3, 6)
    assert output["manifold"].shape == (2, 16, 3, 6 + ambient_offset)
    assert torch.isfinite(output["manifold"]).all()


def test_dual_graph_layer_builds_parallel_graphs_and_backpropagates() -> None:
    embedding = ManifoldEmbedding(
        num_variables=3,
        tangent_dim=4,
        manifold="poincare",
    )
    graph_layer = DualGraphHyperbolicLayer(
        input_length=16,
        num_variables=3,
        tangent_dim=4,
        hidden_dim=8,
        manifold="poincare",
    )
    x = torch.randn(2, 16, 3, requires_grad=True)
    output = graph_layer(embedding(x))

    assert output["spatial_graph"].shape == (2, 3, 3)
    assert output["temporal_graph"].shape == (2, 16, 16)
    assert output["spatial_tangent"].shape == (2, 3, 8)
    assert output["temporal_tangent"].shape == (2, 16, 8)
    assert output["interaction"].shape == (2, 3, 16)
    assert output["temporal_context"].shape == (2, 16, 8)
    assert torch.allclose(
        output["spatial_graph"].sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-5,
    )
    assert torch.allclose(
        output["temporal_graph"].sum(dim=-1),
        torch.ones(2, 16),
        atol=1e-5,
    )

    output["temporal_context"].square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_dual_graph_layer_rejects_wrong_window_shape() -> None:
    layer = DualGraphHyperbolicLayer(
        input_length=16,
        num_variables=3,
        tangent_dim=4,
    )
    with pytest.raises(ValueError, match="Expected layer2 shape"):
        layer(torch.randn(2, 15, 3, 4))


def test_poincare_pairwise_distance_is_finite_and_symmetric() -> None:
    space = ManifoldSpace("poincare")
    points = space.expmap0(torch.randn(2, 5, 4) * 0.1)
    distance = space.pairwise_sqdist(points)
    assert distance.shape == (2, 5, 5)
    assert torch.isfinite(distance).all()
    assert torch.allclose(distance, distance.transpose(-1, -2), atol=1e-5)
