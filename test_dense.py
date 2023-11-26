import cupy as np
from dense import Dense
import pytest
import os
import tempfile


@pytest.fixture
def create_dense_layer():
    # This fixture creates a Dense layer to be used in different tests
    input_size = 10
    output_size = 5
    dense_layer = Dense(input_size, output_size)
    return dense_layer


def test_save_method(create_dense_layer, tmp_path):
    # tmp_path is a pytest fixture that provides a temporary directory unique to the test invocation
    file_path = tmp_path / "dense_layer_state.pkl"

    # Use the fixture to get a Dense layer
    dense_layer = create_dense_layer
    dense_layer.save(file_path)

    # Check if the file is created
    assert os.path.isfile(file_path), "The save method did not create a file."


def test_load_method(create_dense_layer, tmp_path):
    dense_layer = create_dense_layer

    # Save the current state of the layer
    file_path = tmp_path / "dense_layer_state.pkl"
    dense_layer.save(file_path)

    # Create a new Dense layer to load the state into
    new_dense_layer = Dense(dense_layer.weights.shape[1], dense_layer.weights.shape[0])
    new_dense_layer.load(file_path)

    # Check if the loaded weights and biases match the original
    assert np.array_equal(
        dense_layer.weights, new_dense_layer.weights
    ), "Weights do not match after load."
    assert np.array_equal(
        dense_layer.bias, new_dense_layer.bias
    ), "Biases do not match after load."
