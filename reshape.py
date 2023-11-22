import cupy as np
from layer import Layer


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        """
        Initializes a Reshape layer with the given input and output shapes.

        Args:
            input_shape (tuple): The shape of the input tensor.
            output_shape (tuple): The desired shape of the output tensor.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        """
        Performs the forward pass of the Reshape layer.

        Args:
            input (ndarray): The input tensor.

        Returns:
            ndarray: The reshaped output tensor.
        """
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        """
        Performs the backward pass of the Reshape layer.

        Args:
            output_gradient (ndarray): The gradient of the loss with respect to the output tensor.
            learning_rate (float): The learning rate for the backward pass.

        Returns:
            ndarray: The gradient of the loss with respect to the input tensor.
        """
        return np.reshape(output_gradient, self.input_shape)
