import cupy as np
from activations import ReLU, Tanh, Sigmoid
import pytest


# Fixtures to create instances of each activation class
@pytest.fixture
def relu_activation():
    return ReLU()


@pytest.fixture
def tanh_activation():
    return Tanh()


@pytest.fixture
def sigmoid_activation():
    return Sigmoid()


# Test the ReLU activation function and its derivative
def test_relu(relu_activation):
    # Test forward pass
    input_value = np.array([-1.0, 0.0, 1.0])
    expected_output = np.array([0.0, 0.0, 1.0])
    assert np.array_equal(relu_activation.forward(input_value), expected_output)

    # Test backward pass (derivative)
    derivative_input = np.array([-1.0, 0.0, 1.0])
    expected_derivative = np.array([0.0, 0.0, 1.0])
    assert np.array_equal(
        relu_activation.backward(derivative_input, 1), expected_derivative
    )


def test_tanh(tanh_activation):
    # Test forward pass
    input_value = np.array([0.0])
    expected_output = np.tanh(input_value)
    assert np.allclose(tanh_activation.forward(input_value), expected_output)

    # Test backward pass (derivative)
    derivative_input = np.array([0.0])  # At 0, the derivative of tanh is 1
    expected_derivative = 1 - np.tanh(derivative_input) ** 2
    # Since backward method expects the gradient of the loss w.r.t the output, we can use 1.0 for simplicity
    output_gradient = np.array([1.0])
    assert np.allclose(
        tanh_activation.backward(output_gradient, 1), expected_derivative
    )


def test_sigmoid(sigmoid_activation):
    # Test forward pass
    input_value = np.array([0.0])
    expected_output = 1 / (1 + np.exp(-input_value))
    assert np.allclose(sigmoid_activation.forward(input_value), expected_output)

    # Test backward pass (derivative)
    # The output_gradient is the gradient with respect to the output, which is sigmoid_output
    # For the sigmoid function, the derivative is f'(x) = f(x) * (1 - f(x))
    # Since backward method expects the gradient of the loss w.r.t the output, we can use 1.0 for simplicity
    output_gradient = np.array([1.0])
    sigmoid_output = sigmoid_activation.forward(input_value)
    expected_derivative = sigmoid_output * (1 - sigmoid_output)
    assert np.allclose(
        sigmoid_activation.backward(output_gradient, 1), expected_derivative
    )
