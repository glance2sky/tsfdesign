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
    assert head["scale_gate_mean"].item() == 1.0
    assert head["scale_contribution_mean"].item() == 0.0
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


def test_v5_path_calibration_preserves_initial_function_and_has_gradients() -> None:
    model = HyperbolicTSF(
        input_length=24,
        pred_length=12,
        num_variables=4,
        tangent_dim=4,
        hidden_dim=8,
        manifold="poincare",
        use_linear_residual=True,
        use_multiscale_projection=True,
        use_adaptive_path_fusion=True,
        use_path_amplitude_calibration=True,
    )
    x = torch.randn(2, 24, 4, requires_grad=True)
    output = model(x, return_aux=True)
    head = output["head"]

    assert torch.allclose(
        head["normalized_prediction"],
        head["direct"] + head["residual"],
        atol=1e-6,
    )
    assert head["scale_gate_mean"].item() == 1.0
    assert head["scale_contribution_mean"].item() == 0.0
    assert head["direct_scale_mean"].item() == 1.0
    assert head["residual_scale_mean"].item() == 1.0

    output["prediction"].square().mean().backward()
    projection = model.forecast_head.horizon_projection
    for branch in projection.coarse_projections:
        branch_grad = sum(
            parameter.grad.abs().sum()
            for parameter in branch.parameters()
            if parameter.grad is not None
        )
        assert branch_grad.item() > 0.0
    calibration_grad = model.forecast_head.path_calibration.path_scale_raw.grad
    assert calibration_grad is not None
    assert calibration_grad.abs().sum().item() > 0.0


def test_v6_output_residuals_are_zero_starting_and_differentiable() -> None:
    model = HyperbolicTSF(
        input_length=24,
        pred_length=12,
        num_variables=4,
        tangent_dim=4,
        hidden_dim=8,
        manifold="poincare",
        use_linear_residual=True,
        use_adaptive_path_fusion=True,
        use_path_amplitude_calibration=True,
        use_output_multiscale_residual=True,
        use_frequency_residual=True,
        frequency_harmonics=4,
    )
    x = torch.randn(2, 24, 4, requires_grad=True)
    output = model(x, return_aux=True)
    head = output["head"]

    assert output["prediction"].shape == (2, 12, 4)
    assert torch.isfinite(output["prediction"]).all()
    assert head["output_residual_abs_mean"].item() == 0.0
    assert head["output_multiscale_gate_mean"].item() == 1.0
    assert head["frequency_gate"].item() == 1.0

    output["prediction"].square().mean().backward()
    output_ms = model.forecast_head.output_multiscale_residual
    for branch in output_ms.projections:
        branch_grad = sum(
            parameter.grad.abs().sum()
            for parameter in branch.parameters()
            if parameter.grad is not None
        )
        assert branch_grad.item() > 0.0
    frequency_grad = (
        model.forecast_head.frequency_residual.harmonic_projection.grad
    )
    assert frequency_grad is not None
    assert frequency_grad.abs().sum().item() > 0.0


def test_v7_residuals_are_zero_starting_for_full_and_subset_targets() -> None:
    model = HyperbolicTSF(
        input_length=24,
        pred_length=12,
        num_variables=4,
        output_dim=2,
        target_indices=[2, 0],
        tangent_dim=4,
        hidden_dim=8,
        manifold="poincare",
        use_linear_residual=True,
        use_adaptive_path_fusion=True,
        use_path_amplitude_calibration=True,
        use_trend_difference_residual=True,
        trend_difference_windows=(6, 12, 24),
        use_explicit_periodic_residual=True,
        explicit_periods=(6, 12, 24),
    )
    x = torch.randn(2, 24, 4, requires_grad=True)
    output = model(x, return_aux=True)
    head = output["head"]

    assert output["prediction"].shape == (2, 12, 2)
    assert torch.isfinite(output["prediction"]).all()
    assert head["trend_difference_amplitude_abs_mean"].item() == 0.0
    assert head["explicit_periodic_amplitude_abs_mean"].item() == 0.0
    assert head["output_residual_abs_mean"].item() == 0.0
    assert torch.allclose(
        head["normalized_prediction"],
        head["direct"] + head["residual"],
        atol=1e-6,
    )

    output["prediction"].square().mean().backward()
    trend_grad = (
        model.forecast_head.trend_difference_residual.amplitude_raw.grad
    )
    periodic_grad = (
        model.forecast_head.explicit_periodic_residual.amplitude_raw.grad
    )
    assert trend_grad is not None
    assert periodic_grad is not None
    assert trend_grad.abs().sum().item() > 0.0
    assert periodic_grad.abs().sum().item() > 0.0


def test_v7_residual_modules_support_long_horizon_and_short_lookback() -> None:
    model = HyperbolicTSF(
        input_length=96,
        pred_length=720,
        num_variables=7,
        tangent_dim=8,
        hidden_dim=16,
        manifold="poincare",
        use_linear_residual=True,
        use_trend_difference_residual=True,
        use_explicit_periodic_residual=True,
    )
    output = model(torch.randn(2, 96, 7), return_aux=True)
    assert output["prediction"].shape == (2, 720, 7)
    assert torch.isfinite(output["prediction"]).all()


def test_v8_variable_hierarchy_is_identity_but_trainable_at_initialization() -> None:
    model = HyperbolicTSF(
        input_length=24,
        pred_length=12,
        num_variables=4,
        tangent_dim=4,
        hidden_dim=8,
        manifold="poincare",
        use_linear_residual=True,
        use_adaptive_path_fusion=True,
        use_path_amplitude_calibration=True,
        use_variable_hierarchy=True,
        variable_hierarchy_groups=2,
    )
    x = torch.randn(2, 24, 4, requires_grad=True)
    output = model(x, return_aux=True)
    encoder = output["encoder"]

    assert output["prediction"].shape == (2, 12, 4)
    assert torch.isfinite(output["prediction"]).all()
    assert encoder["hierarchy_mix"].item() == 1.0
    assert encoder["hierarchy_contribution"].item() == 0.0
    assert encoder["assignment"].shape == (2, 4, 2)
    assert encoder["group_graph"].shape == (2, 2, 2)

    output["prediction"].square().mean().backward()
    hierarchy = model.graph_encoder.variable_hierarchy
    child_grad = hierarchy.child_projection.weight.grad
    assert child_grad is not None
    assert child_grad.abs().sum().item() > 0.0

    with torch.no_grad():
        hierarchy.child_projection.weight.normal_(mean=0.0, std=0.01)
    model.zero_grad()
    output = model(x.detach(), return_aux=True)
    output["prediction"].square().mean().backward()
    gate_grad = hierarchy.hierarchy_mix_raw.grad
    assert gate_grad is not None
    assert gate_grad.abs().sum().item() > 0.0


def test_v8_temporal_hierarchy_supports_long_horizon_and_has_gradients() -> None:
    model = HyperbolicTSF(
        input_length=96,
        pred_length=720,
        num_variables=7,
        tangent_dim=8,
        hidden_dim=16,
        manifold="poincare",
        use_linear_residual=True,
        use_adaptive_path_fusion=True,
        use_path_amplitude_calibration=True,
        use_temporal_hierarchy=True,
        temporal_hierarchy_factors=(2, 4, 8),
    )
    x = torch.randn(2, 96, 7, requires_grad=True)
    output = model(x, return_aux=True)
    encoder = output["encoder"]

    assert output["prediction"].shape == (2, 720, 7)
    assert torch.isfinite(output["prediction"]).all()
    assert encoder["temporal_hierarchy_mix"].item() == 1.0
    assert encoder["temporal_hierarchy_contribution"].item() == 0.0
    assert encoder["temporal_level_contribution"].shape == (3,)
    assert encoder["temporal_level_graph_entropy"].shape == (3,)
    assert encoder["temporal_level_graph_mix"].shape == (3,)

    output["prediction"].square().mean().backward()
    hierarchy = model.graph_encoder.temporal_hierarchy
    child_grad = hierarchy.levels[0]["child_projection"].weight.grad
    assert child_grad is not None
    assert child_grad.abs().sum().item() > 0.0


def test_v8c_recursive_temporal_hierarchy_is_identity_and_recursive() -> None:
    model = HyperbolicTSF(
        input_length=96,
        pred_length=720,
        num_variables=7,
        tangent_dim=8,
        hidden_dim=16,
        manifold="poincare",
        use_linear_residual=True,
        use_adaptive_path_fusion=True,
        use_path_amplitude_calibration=True,
        use_recursive_temporal_hierarchy=True,
        recursive_temporal_factors=(2, 2, 2),
    )
    x = torch.randn(2, 96, 7, requires_grad=True)
    output = model(x, return_aux=True)
    encoder = output["encoder"]
    hierarchy = model.graph_encoder.recursive_temporal_hierarchy

    assert output["prediction"].shape == (2, 720, 7)
    assert torch.isfinite(output["prediction"]).all()
    assert encoder["recursive_temporal_hierarchy_depth"].item() == 3.0
    assert encoder["recursive_temporal_hierarchy_mix"].item() == 1.0
    assert encoder["recursive_temporal_hierarchy_contribution"].item() < 1e-6
    assert [
        tuple(getattr(hierarchy, name).shape)
        for name in hierarchy.assignments
    ] == [(48, 96), (24, 48), (12, 24)]
    assert [
        tuple(getattr(hierarchy, name).shape)
        for name in hierarchy.local_adjacencies
    ] == [(48, 48), (24, 24), (12, 12)]

    output["prediction"].square().mean().backward()
    down_grad = hierarchy.shared_down_projection.weight.grad
    assert down_grad is not None
    assert down_grad.abs().sum().item() > 0.0
