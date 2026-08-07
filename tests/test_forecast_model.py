import pytest
import torch

from hypertsf_layers import HyperbolicTSF, RevIN


def test_revin_denormalizes_selected_variables() -> None:
    layer = RevIN(num_variables=3, affine=True)
    x = torch.randn(2, 8, 3)
    state = layer(x)
    selected = state["x"][..., [2]]
    restored = layer.denormalize(selected, state, variable_indices=[2])
    assert restored.shape == (2, 8, 1)
    assert torch.allclose(restored, x[..., [2]], atol=1e-5)


@pytest.mark.parametrize("manifold", ["euclidean", "poincare", "lorentz"])
def test_hyperbolic_tsf_returns_future_forecast(manifold: str) -> None:
    model = HyperbolicTSF(
        input_length=12,
        pred_length=5,
        num_variables=3,
        output_dim=1,
        target_indices=[2],
        tangent_dim=4,
        hidden_dim=8,
        manifold=manifold,
        use_linear_residual=True,
    )
    x = torch.randn(2, 12, 3, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 5, 1)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_hyperbolic_tsf_returns_auxiliary_graph_outputs() -> None:
    model = HyperbolicTSF(
        input_length=10,
        pred_length=4,
        num_variables=3,
        tangent_dim=3,
        hidden_dim=6,
        manifold="poincare",
    )
    output = model(torch.randn(2, 10, 3), return_aux=True)
    assert output["prediction"].shape == (2, 4, 3)
    assert output["encoder"]["variable_weights"].shape == (2, 10, 3)
    assert output["encoder"]["variable_context"].shape == (2, 10, 6)
    assert output["head"]["direct"].shape == (2, 4, 3)
    for key in (
        "spatial_graph_entropy",
        "temporal_graph_entropy",
        "variable_weight_entropy",
        "fusion_gate_mean",
        "fusion_gate_std",
        "spatial_graph_mix",
        "temporal_graph_mix",
        "spatial_tangent_norm",
        "temporal_tangent_norm",
    ):
        assert torch.isfinite(output["encoder"][key])
    assert 0.0 <= output["encoder"]["spatial_graph_entropy"] <= 1.0
    assert 0.0 <= output["encoder"]["temporal_graph_entropy"] <= 1.0
    assert 0.0 <= output["encoder"]["variable_weight_entropy"] <= 1.0
    assert 0.0 <= output["encoder"]["fusion_gate_mean"] <= 1.0


def test_structured_head_scales_to_long_horizon() -> None:
    model = HyperbolicTSF(
        input_length=96,
        pred_length=720,
        num_variables=7,
        tangent_dim=8,
        hidden_dim=16,
        manifold="poincare",
        temporal_rank=16,
        spatial_rank=8,
    )
    output = model(torch.randn(2, 96, 7))
    assert output.shape == (2, 720, 7)
    assert torch.isfinite(output).all()
    assert model.forecast_head.horizon_projection.rank == 16


def test_default_configuration_preserves_v1_capacity_and_disables_new_paths() -> None:
    model = HyperbolicTSF(
        input_length=96,
        pred_length=720,
        num_variables=7,
        tangent_dim=32,
        hidden_dim=64,
        manifold="poincare",
        use_linear_residual=True,
    )
    assert model.embedding.time_identity is None
    assert model.graph_encoder.spatial_hnn.weight is not None
    assert model.graph_encoder.spatial_hgcn.residual_logit is None
    assert model.forecast_head.horizon_projection.rank is None
    assert model.forecast_head.linear_residual.rank is None
    assert model.forecast_head.trend_residual.trend_scale_raw.item() == 0.0

    output = model(torch.randn(2, 96, 7), return_aux=True)
    assert output["head"]["trend_scale"].item() == 0.0
    assert torch.isfinite(output["head"]["residual_to_direct_ratio"])
    assert torch.allclose(
        output["head"]["normalized_prediction"],
        output["head"]["direct"] + output["head"]["residual"],
    )


def test_multiscale_and_adaptive_head_preserves_shapes_and_diagnostics() -> None:
    model = HyperbolicTSF(
        input_length=24,
        pred_length=12,
        num_variables=4,
        tangent_dim=4,
        hidden_dim=8,
        manifold="poincare",
        use_linear_residual=True,
        use_multiscale_projection=True,
        multiscale_factors=(1, 2, 4),
        use_adaptive_path_fusion=True,
    )
    x = torch.randn(2, 24, 4, requires_grad=True)
    output = model(x, return_aux=True)
    head = output["head"]

    assert output["prediction"].shape == (2, 12, 4)
    assert head["path_weights"].shape == (2, 12, 4, 2)
    assert head["scale_gate_mean"].item() == 0.0
    assert torch.allclose(
        head["path_weights"],
        torch.ones_like(head["path_weights"]),
        atol=1e-6,
    )
    assert torch.allclose(
        head["normalized_prediction"],
        head["direct"] + head["residual"],
        atol=1e-6,
    )
    output["prediction"].square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
