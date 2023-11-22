import cupy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def categorical_cross_entropy(y_true, y_pred):
    # Clip y_pred to prevent numerical instability (log(0) issues)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # Calculate categorical cross-entropy
    result = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return result


def categorical_cross_entropy_prime(y_true, y_pred):
    y_true = np.reshape(y_true, y_pred.shape)
    result = y_pred - y_true
    return result
