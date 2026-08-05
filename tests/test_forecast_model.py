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
