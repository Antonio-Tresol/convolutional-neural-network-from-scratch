import cupy as np
import network as nn
import pytest
import os


def clean_up():
    os.remove("test_network/layer-0")
    os.remove("test_network/layer-1")
    os.remove("test_network/layer-2")
    os.rmdir("test_network")


@pytest.fixture
def test_save_and_load_conv_method():
    # arrange
    network = nn.get_test_convolutional_network()
    folder_path = "test_network"
    file_path = folder_path + "/layer"
    for layer in network:
        layer.kernels = np.random.randn(*layer.kernels.shape)
        layer.biases = np.random.randn(*layer.biases.shape)
    # act
    nn.save(network=network, file_path=file_path)

    new_network = nn.get_test_convolutional_network()
    nn.load(network=new_network, file_path=file_path)

    # assert
    assert os.path.isdir(folder_path), "The save method did not create a folder."
    for i, layer in enumerate(network):
        assert os.path.isfile(
            file_path + "-" + str(i)
        ), f"The save method did not create a file for layer {i}."
        assert np.array_equal(
            layer.kernels, new_network[i].kernels
        ), f"Weights do not match after load for layer {i}."
        assert np.array_equal(
            layer.biases, new_network[i].biases
        ), f"Biases do not match after load for layer {i}."
    clean_up()


def test_save_and_load_dense_method():
    # arrange
    network = nn.get_test_dense_network()
    folder_path = "test_network"
    file_path = folder_path + "/layer"
    for layer in network:
        layer.weights = np.random.randn(*layer.weights.shape)
        layer.bias = np.random.randn(*layer.bias.shape)

    # act
    nn.save(network=network, file_path=file_path)
    new_network = nn.get_test_dense_network()
    nn.load(network=new_network, file_path=file_path)

    # assert
    assert os.path.isdir(folder_path), "The save method did not create a folder."
    for i, layer in enumerate(network):
        assert os.path.isfile(
            file_path + "-" + str(i)
        ), f"The save method did not create a file for layer {i}."
        assert np.array_equal(
            layer.weights, new_network[i].weights
        ), f"Weights do not match after load for layer {i}."
        assert np.array_equal(
            layer.bias, new_network[i].bias
        ), f"Biases do not match after load for layer {i}."
    clean_up()


def test_get_get_test_dense_network_method():
    # act
    network = nn.get_test_dense_network()
    different_network = nn.get_test_dense_network()
    # assert
    assert len(network) == len(different_network), "The networks are not the same size."
    for i, layer in enumerate(network):
        assert not np.array_equal(
            layer.weights, different_network[i].weights
        ), f"Weights match for layer {i}."
        assert np.array_equal(
            layer.bias, different_network[i].bias
        ), f"Bias do not match for layer {i}."
