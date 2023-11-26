import cupy as np
from cupyx.scipy import signal
from layer import Layer
import pickle as pkl


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        """
        Initializes a Convolutional layer.

        Args:
            input_shape (tuple): The shape of the input tensor (depth, height, width).
            kernel_size (int): The size of the convolutional kernel.
            depth (int): The number of output channels.

        """
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (
            depth,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        """
        Performs forward propagation for the Convolutional layer.

        Args:
            input (ndarray): The input tensor.

        Returns:
            ndarray: The output tensor.

        """
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid"
                )
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Performs backward propagation for the Convolutional layer.

        Args:
            output_gradient (ndarray): The gradient of the loss with respect to the output tensor.
            learning_rate (float): The learning rate.

        Returns:
            ndarray: The gradient of the loss with respect to the input tensor.

        """
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], "valid"
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], "full"
                )

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

    def save(self, file_path):
        """
        Saves the Convolutional layer to a file.

        Args:
            file_path (str): The path to the file.

        """
        with open(file_path, "wb") as file:
            state = {
                "kernels": self.kernels,
                "biases": self.biases,
                "input_shape": self.input_shape,
                "depth": self.depth,
            }
            pkl.dump(state, file)

    def load(self, file_path):
        """
        Loads the Convolutional layer from a file.

        Args:
            file_path (str): The path to the file.

        """
        with open(file_path, "rb") as file:
            state = pkl.load(file)
            self.kernels = state["kernels"]
            self.biases = state["biases"]
            self.input_shape = state["input_shape"]
            self.depth = state["depth"]
