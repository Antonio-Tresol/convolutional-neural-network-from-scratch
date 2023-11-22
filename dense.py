import cupy as np
from layer import Layer


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """
        Initializes a Dense layer with given input size and output size.

        Args:
            input_size (int): The size of the input to the layer.
            output_size (int): The size of the output from the layer.
        """
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        """
        Performs forward propagation for the Dense layer.

        Args:
            input (ndarray): The input to the layer.

        Returns:
            ndarray: The output from the layer.
        """
        self.input = input
        return np.dot(self.weights, self.input)

    def backward(self, output_gradient, learning_rate):
        """
        Performs backward propagation for the Dense layer.

        Args:
            output_gradient (ndarray): The gradient of the loss with respect to the output of the layer.
            learning_rate (float): The learning rate for updating the weights and biases.

        Returns:
            ndarray: The gradient of the loss with respect to the input of the layer.
        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
