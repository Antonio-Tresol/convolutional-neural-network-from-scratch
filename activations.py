import cupy as np
from layer import Layer
from activation import Activation


class Tanh(Activation):
    def __init__(self):
        """
        Initializes the Tanh activation function.

        The Tanh activation function is defined as the hyperbolic tangent function.

        Args:
            None

        Returns:
            None
        """

        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        """
        Initializes the Sigmoid activation function.

        The Sigmoid activation function is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))

        The derivative of the Sigmoid activation function is calculated as:
        sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))
        """

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Softmax(Layer):
    def forward(self, input):
        """
        Applies the softmax activation function to the input.

        Parameters:
        input (numpy.ndarray): The input array.

        Returns:
        numpy.ndarray: The output array after applying the softmax activation function.
        """
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Computes the gradient of the softmax activation function.

        Parameters:
        output_gradient (numpy.ndarray): The gradient of the loss function with respect to the output.
        learning_rate (float): The learning rate for updating the weights.

        Returns:
        numpy.ndarray: The gradient of the loss function with respect to the input.
        """
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
