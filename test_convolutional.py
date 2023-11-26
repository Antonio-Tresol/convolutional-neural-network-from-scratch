import cupy as np
from convolutional import Convolutional
import pytest
import os
import tempfile


@pytest.fixture
def create_conv_layer():
    # This fixture creates a Convolutional layer to be used in different tests
    input_shape = (3, 256, 256)
    kernel_size = 3
    depth = 12
    conv_layer = Convolutional(input_shape, kernel_size, depth)
    return conv_layer


def test_save_method(create_conv_layer, tmp_path):
    # tmp_path is a pytest fixture that provides a temporary directory unique to the test invocation
    file_path = tmp_path / "conv_layer_state.pkl"

    # Use the fixture to get a Convolutional layer
    conv_layer = create_conv_layer
    conv_layer.save(file_path)

    # Check if the file is created
    assert os.path.isfile(file_path), "The save method did not create a file."


def test_load_method(create_conv_layer, tmp_path):
    # Use the fixture to get a Convolutional layer
    conv_layer = create_conv_layer

    # Save the current state of the layer
    file_path = tmp_path / "conv_layer_state.pkl"
    conv_layer.save(file_path)

    # Create a new Convolutional layer to load the state into
    new_conv_layer = Convolutional(
        conv_layer.input_shape, conv_layer.kernels_shape[2], conv_layer.depth
    )
    new_conv_layer.load(file_path)

    # Check if the loaded state matches the original
    assert np.array_equal(
        conv_layer.kernels, new_conv_layer.kernels
    ), "Kernels do not match after load."
    assert np.array_equal(
        conv_layer.biases, new_conv_layer.biases
    ), "Biases do not match after load."
    assert (
        conv_layer.input_shape == new_conv_layer.input_shape
    ), "Input shapes do not match after load."
    assert conv_layer.depth == new_conv_layer.depth, "Depth does not match after load."
