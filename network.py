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
