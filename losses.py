import cupy as np


def mse(y_true, y_pred):
    """
    Calculates the mean squared error between the true values and the predicted values.

    Parameters:
    - y_true (cupy.ndarray): The true values.
    - y_pred (cupy.ndarray): The predicted values.

    Returns:
    - float: The mean squared error.
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    """
    Calculates the derivative of the mean squared error with respect to the predicted values.

    Parameters:
    - y_true (cupy.ndarray): The true values.
    - y_pred (cupy.ndarray): The predicted values.

    Returns:
    - cupy.ndarray: The derivative of the mean squared error.
    """
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred):
    """
    Calculates the binary cross-entropy loss between the true values and the predicted values.

    Parameters:
    - y_true (cupy.ndarray): The true values.
    - y_pred (cupy.ndarray): The predicted values.

    Returns:
    - float: The binary cross-entropy loss.
    """
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    """
    Calculates the derivative of the binary cross-entropy loss with respect to the predicted values.

    Parameters:
    - y_true (cupy.ndarray): The true values.
    - y_pred (cupy.ndarray): The predicted values.

    Returns:
    - cupy.ndarray: The derivative of the binary cross-entropy loss.
    """
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def categorical_cross_entropy(y_true, y_pred):
    """
    Calculates the categorical cross-entropy loss between the true values and the predicted values.

    Parameters:
    - y_true (cupy.ndarray): The true values.
    - y_pred (cupy.ndarray): The predicted values.

    Returns:
    - float: The categorical cross-entropy loss.
    """
    # Clip y_pred to prevent numerical instability (log(0) issues)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # Calculate categorical cross-entropy
    result = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return result


def categorical_cross_entropy_prime(y_true, y_pred):
    """
    Calculates the derivative of the categorical cross-entropy loss with respect to the predicted values.

    Parameters:
    - y_true (cupy.ndarray): The true values.
    - y_pred (cupy.ndarray): The predicted values.

    Returns:
    - cupy.ndarray: The derivative of the categorical cross-entropy loss.
    """
    y_true = np.reshape(y_true, y_pred.shape)
    result = y_pred - y_true
    return result
