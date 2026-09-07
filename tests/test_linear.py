import numpy as np
from nnfs.layers import Linear

def compute_linear_objective(
        layer: Linear,
        x: np.ndarray,
        grad_out: np.ndarray,
) -> float:

    output = layer.forward(x)
    return float(np.sum(output * grad_out))



def finite_difference_parameter_gradient(
        layer: Linear,
        x: np.ndarray,
        grad_out: np.ndarray,
        parameter_name: str, 
        epsilon: float = 1e-5,
) -> np.ndarray:
    parameter = layer._parameters[parameter_name]
    numerical_gradient = np.zeros_like(parameter)

    iterator = np.nditer(parameter, flags=["multi_index"], op_flags=["readwrite"])

    while not iterator.finished:
        index = iterator.multi_index
        original_value = parameter[index]

        parameter[index] = original_value + epsilon
        loss_plus = compute_linear_objective(layer, x, grad_out)

        parameter[index] = original_value - epsilon
        loss_minus = compute_linear_objective(layer, x, grad_out)

        numerical_gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)
        iterator.iternext()

    return numerical_gradient

def finite_difference_input_gradient(
        layer: Linear,
        x: np.ndarray,
        grad_out: np.ndarray,
        epsilon: float = 1e-5,
) -> np.ndarray:

    numerical_gradient = np.zeros_like(x)

    iterator = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not iterator.finished:
        index = iterator.multi_index
        original_value = x[index]

        x[index] = original_value + epsilon
        loss_plus = compute_linear_objective(layer, x, grad_out)

        x[index] = original_value - epsilon
        loss_minus = compute_linear_objective(layer, x, grad_out)

        numerical_gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)
        iterator.iternext()

    return numerical_gradient

def test_linear_forward_shape() -> None:
    rng = np.random.default_rng(67)

    batch_size = 4
    in_features = 3
    out_features = 2

    x = rng.normal(size=(batch_size, in_features))
    layer = Linear(in_features=in_features, out_features=out_features, rng=rng)

    output = layer.forward(x)

    assert output.shape == (batch_size, out_features)

def test_linear_backward_shapes() -> None:
    rng = np.random.default_rng(67)

    batch_size = 4
    in_features = 3
    out_features = 2

    x = rng.normal(size=(batch_size, in_features))
    grad_out = rng.normal(size=(batch_size, out_features))

    layer = Linear(in_features=in_features, out_features=out_features, rng=rng)

    layer.forward(x)
    grad_input = layer.backward(grad_out)

    assert layer._grads["weight"].shape == layer.weight.shape
    assert layer._grads["bias"].shape == layer.bias.shape
    assert grad_input.shape == x.shape

def test_linear_weight_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(67)

    x = rng.normal(size=(4, 3))
    grad_out = rng.normal(size=(4, 2))

    layer = Linear(in_features=3, out_features=2, rng=rng)

    layer.forward(x)
    layer.backward(grad_out)

    numerical_weight_gradient = finite_difference_parameter_gradient(
        layer=layer,
        x=x,
        grad_out=grad_out,
        parameter_name="weight",
    )

    assert np.allclose(
        layer._grads["weight"],
        numerical_weight_gradient,
        rtol=1e-5,
        atol=1e-7,
    )

def test_linear_bias_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(67)

    x = rng.normal(size=(4, 3))
    grad_out = rng.normal(size=(4, 2))

    layer = Linear(in_features=3, out_features=2, rng=rng)

    layer.forward(x)
    layer.backward(grad_out)

    numerical_bias_gradient = finite_difference_parameter_gradient(
        layer=layer,
        x=x,
        grad_out=grad_out,
        parameter_name="bias",
    )

    assert np.allclose(
        layer._grads["bias"],
        numerical_bias_gradient,
        rtol=1e-5,
        atol=1e-7,
    )

def test_linear_input_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(67)

    x = rng.normal(size=(4, 3))
    grad_out = rng.normal(size=(4, 2))

    layer = Linear(in_features=3, out_features=2, rng=rng)

    layer.forward(x)
    analytical_input_gradient = layer.backward(grad_out)

    numerical_input_gradient = finite_difference_input_gradient(
        layer=layer,
        x=x,
        grad_out=grad_out,
    )

    assert np.allclose(
        analytical_input_gradient,
        numerical_input_gradient,
        rtol=1e-5,
        atol=1e-7,
    )