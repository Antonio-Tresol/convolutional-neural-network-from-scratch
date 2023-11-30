from logging import exception
import cupy as np
from sklearn.model_selection import train_test_split
import gc

from activations import Sigmoid, Softmax, ReLU
from convolutional import Convolutional
from dense import Dense
from losses import categorical_cross_entropy, categorical_cross_entropy_prime
from network import (
    train,
    predict,
    train_with_batch,
    save,
    load,
    get_small_size_network,
    get_medium_size_network,
)
from reshape import Reshape
import datahandler as dh
import time


def main():
    # Get the training data
    directory = input(
        "Enter the path to data directory with one piece images to classify: "
    )

    unique_labels = []

    try:
        unique_labels = dh.get_labels_in_dir(directory)
    except exception as e:
        print("Error: ", e)
        exit(1)

    one_hot_encoded = dh.get_one_hot_encoding(unique_labels)

    X, Y = dh.get_images_and_labels(directory, unique_labels, one_hot_encoded)

    if X is None or Y is None:
        print("No images found in the directory. Please enter a valid directory path.")
        exit(1)

    # Convert X and y to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Normalize the image data to the range [0, 1]
    X = X.astype("float32") / 255.0

    print("Data normalized.")

    # Create the network

    network = get_medium_size_network()

    # determine if the network should be trained from scratch or loaded from a saved model

    try:
        load(network)
    except exception as e:
        print("Error loading network. : ", e)
        exit(1)

    # Test and display predictions
    errors = []

    for x, y in zip(X, Y):
        prediction = predict(network, x)
        indices = int(np.argmax(prediction))
        predicted_label = unique_labels[indices]
        true_label = unique_labels[int(np.argmax(y))]
        print(f"Prediction: {predicted_label}, Actual: {true_label}")
        if predicted_label != true_label:
            errors.append(0)
        else:
            errors.append(1)

    # save the errors on to the historical errors file
    history_errors = dh.load_classification_error_data()
    history_errors = [*history_errors, *errors]
    dh.save_classification_error_data(history_errors)

    errors = np.array(errors)

    print(f"Accuracy on data: {errors.sum() / errors.size * 100}%")


if __name__ == "__main__":
    main()
