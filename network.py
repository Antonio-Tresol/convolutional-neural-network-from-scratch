import os
import cupy as np


def predict(network, input):
    """
    Predicts the output of the neural network given an input.

    Args:
        network (list): The list of layers in the neural network.
        input: The input to the neural network.

    Returns:
        The output of the neural network.
    """
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(
    network,
    loss,
    loss_prime,
    x_train,
    y_train,
    epochs=1000,
    learning_rate=0.01,
    verbose=True,
    error_data=None,
):
    """
    Trains a neural network using backpropagation.

    Args:
        network (list): List of layers in the neural network.
        loss (function): Loss function used to calculate the error.
        loss_prime (function): Derivative of the loss function.
        x_train (list): List of input training data.
        y_train (list): List of target training data.
        epochs (int, optional): Number of training epochs. Defaults to 1000.
        learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.01.
        verbose (bool, optional): Whether to print training progress. Defaults to True.
    """
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)
            # error
            error += loss(y, output)
            error_data.append(error)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")


def train_with_batch(
    x_train,
    y_train,
    network,
    error_data,
    loss,
    loss_prime,
    epochs=1,
    learning_rate=0.1,
    verbose=False,
):
    """
    Trains the neural network using batch training.

    Args:
        x_train (numpy.ndarray): Input data for training.
        y_train (numpy.ndarray): Target data for training.
        network (NeuralNetwork): The neural network model.
        error_data (dict): Dictionary to store the error data during training.
        loss (function): Loss function used to calculate the error.
        loss_prime (function): Derivative of the loss function.
        epochs (int, optional): Number of training epochs. Defaults to 1.
        learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.1.
        verbose (bool, optional): Whether to print training progress. Defaults to False.
    """
    train(
        network,
        loss,
        loss_prime,
        x_train,
        y_train,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=verbose,
        error_data=error_data,
    )


def save(network, file_path="network/layer"):
    """
    Saves the parameters of the network to a file.

    Args:
        network (list): The list of layers in the neural network.
        file_path (str): The path to the file where the parameters should be saved.
    """
    folder_path = file_path.split("/")[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, layer in enumerate(network):
        layer.save(f"{file_path}-{i}")
        print(f"Layer {i} saved")


def load(network, file_path="network/layer"):
    """
    Loads the parameters of the network from a file.

    Args:
        network (list): The list of layers in the neural network.
        file_path (str): The path to the file where the parameters should be loaded from.
    """
    for i, layer in enumerate(network):
        layer.load(f"{file_path}-{i}")
        print(f"Layer {i} loaded")
