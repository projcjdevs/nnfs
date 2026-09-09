import numpy as np

from nnfs.activations import ReLU

EPSILON = 1e-5


def compute_relu_objective(
    activation: ReLU,
    x: np.ndarray,
    grad_out: np.ndarray,
) -> float:
    """
    Computes a scalar objective for gradient checking.

    objective = sum(ReLU(x) * grad_out)

    This makes the upstream gradient with respect to the ReLU output
    exactly equal to grad_out.
    """
    output = activation.forward(x)
    return float(np.sum(output * grad_out))


def finite_difference_input_gradient(
    activation: ReLU,
    x: np.ndarray,
    grad_out: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """
    Numerically approximates the gradient with respect to the input x.

    Formula:

        df/dx ≈ (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
    """
    numerical_gradient = np.zeros_like(x)

    iterator = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not iterator.finished:
        index = iterator.multi_index
        original_value = x[index]

        x[index] = original_value + epsilon
        loss_plus = compute_relu_objective(activation, x, grad_out)

        x[index] = original_value - epsilon
        loss_minus = compute_relu_objective(activation, x, grad_out)

        numerical_gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

        x[index] = original_value
        iterator.iternext()

    return numerical_gradient


def test_relu_forward_values() -> None:
    x = np.array(
        [
            [1.0, -1.0],
            [0.0, 2.5],
        ],
        dtype=np.float64,
    )

    activation = ReLU()
    output = activation.forward(x)

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.5],
        ],
        dtype=np.float64,
    )

    assert np.allclose(output, expected)


def test_relu_backward_mask() -> None:
    x = np.array(
        [
            [1.2, -0.7, 0.5],
            [-1.5, 2.0, -0.25],
        ],
        dtype=np.float64,
    )

    grad_out = np.array(
        [
            [0.8, -1.1, 0.3],
            [-0.4, 1.7, 0.9],
        ],
        dtype=np.float64,
    )

    activation = ReLU()
    activation.forward(x)
    grad_input = activation.backward(grad_out)

    expected = grad_out * (x > 0)

    assert np.allclose(grad_input, expected)


def test_relu_backward_shape() -> None:
    x = np.array(
        [
            [1.2, -0.7, 0.5],
            [-1.5, 2.0, -0.25],
        ],
        dtype=np.float64,
    )

    grad_out = np.array(
        [
            [0.8, -1.1, 0.3],
            [-0.4, 1.7, 0.9],
        ],
        dtype=np.float64,
    )

    activation = ReLU()
    activation.forward(x)
    grad_input = activation.backward(grad_out)

    assert grad_input.shape == x.shape


def test_relu_input_gradient_matches_finite_difference() -> None:
    x = np.array(
        [
            [1.2, -0.7, 0.5],
            [-1.5, 2.0, -0.25],
        ],
        dtype=np.float64,
    )

    grad_out = np.array(
        [
            [0.8, -1.1, 0.3],
            [-0.4, 1.7, 0.9],
        ],
        dtype=np.float64,
    )

    activation = ReLU()

    activation.forward(x)
    analytical_gradient = activation.backward(grad_out)

    numerical_gradient = finite_difference_input_gradient(
        activation=activation,
        x=x,
        grad_out=grad_out,
    )

    assert np.allclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-5,
        atol=1e-7,
    )