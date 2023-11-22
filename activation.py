import cupy as np
from layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        """
        Initializes an Activation layer.

        Args:
            activation: The activation function.
            activation_prime: The derivative of the activation function.
        """
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        """
        Performs the forward pass of the Activation layer.

        Args:
            input: The input to the layer.

        Returns:
            The output of the activation function applied to the input.
        """
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        """
        Performs the backward pass of the Activation layer.

        Args:
            output_gradient: The gradient of the output.
            learning_rate: The learning rate.

        Returns:
            The gradient of the input.
        """
        return np.multiply(output_gradient, self.activation_prime(self.input))
