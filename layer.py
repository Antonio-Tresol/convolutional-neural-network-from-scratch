class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """
        Performs the forward pass of the layer.

        Args:
            input: The input data.

        Returns:
            The output of the layer.
        """
        pass

    def backward(self, output_gradient, learning_rate):
        """
        Performs the backward pass of the layer.

        Args:
            output_gradient: The gradient of the loss with respect to the layer's output.
            learning_rate: The learning rate for updating the layer's parameters.
        """
        pass

    def save(self, file_path):
        """
        Saves the layer's parameters to the given folder path.

        Args:
            file_path (str): The path to the folder where the parameters should be saved.
        """
        pass

    def load(self, file_path):
        """
        Loads the layer's parameters from the given folder path.

        Args:
            file_path (str): The path to the folder where the parameters should be loaded from.
        """
        pass
