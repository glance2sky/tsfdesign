import torch

from error_analysis_utils import decompose_head_forecasts
from hypertsf_layers import HyperbolicTSF


def test_branch_decomposition_reconstructs_revin_prediction() -> None:
    model = HyperbolicTSF(
        input_length=12,
        pred_length=5,
        num_variables=3,
        output_dim=1,
        target_indices=[2],
        tangent_dim=4,
        hidden_dim=8,
        manifold="poincare",
        use_revin=True,
        use_linear_residual=True,
    )
    x = torch.randn(2, 12, 3)
    output = model(x, return_aux=True)

    components = decompose_head_forecasts(
        model,
        output["head"]["direct"],
        output["head"]["residual"],
        output["revin_state"],
    )

    assert torch.allclose(
        components["reconstructed_prediction"],
        output["prediction"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        components["direct_forecast"],
        components["base_forecast"] + components["direct_contribution"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        components["residual_forecast"],
        components["base_forecast"] + components["residual_contribution"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_branch_decomposition_without_revin_is_identity() -> None:
    model = HyperbolicTSF(
        input_length=10,
        pred_length=4,
        num_variables=2,
        tangent_dim=3,
        hidden_dim=6,
        manifold="euclidean",
        use_revin=False,
        use_linear_residual=True,
    )
    direct = torch.randn(2, 4, 2)
    residual = torch.randn(2, 4, 2)

    components = decompose_head_forecasts(
        model,
        direct,
        residual,
        revin_state=None,
    )

    assert torch.equal(components["base_forecast"], torch.zeros_like(direct))
    assert torch.equal(components["direct_forecast"], direct)
    assert torch.equal(components["residual_forecast"], residual)
    assert torch.equal(
        components["reconstructed_prediction"],
        direct + residual,
    )
