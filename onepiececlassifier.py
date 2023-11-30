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
    directory = input("Enter the path to the training data directory: ")

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

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42
    )
    # Delete X and Y to free up memory
    del X, Y
    gc.collect()

    x_train_split, y_train_split = dh.split_data_into_batches(x_train, y_train, 20)

    print("Data split into training and test sets.")
    print(f"Training set size: {x_train.shape[0]}")
    print(f"Test set size: {x_test.shape[0]}")

    # Create the network

    network = get_medium_size_network()

    # determine if the network should be trained from scratch or loaded from a saved model
    load_data = input("Load saved model? (y/n): ").lower()
    if load_data == "y":
        try:
            load(network)
        except exception as e:
            print("Error loading network. : ", e)
            print("Training new network.")
        print("Network loaded.")
    else:
        print("Training new network.")

    start = time.time()
    error_data = dh.load_classification_error_data("error_data.csv")

    # Train the network with batches
    learning_rate = 0.1
    total_batches = len(x_train_split)

    for i in reversed(range(total_batches)):
        print(f"Training batch {i + 1} of {total_batches}")
        train_with_batch(
            x_train_split[i],
            y_train_split[i],
            network,
            error_data,
            categorical_cross_entropy,
            categorical_cross_entropy_prime,
            epochs=3,
            learning_rate=learning_rate,
            verbose=True,
        )
        # Delete the last batch to free memory
        del x_train_split[i]
        del y_train_split[i]
        gc.collect()
        learning_rate *= 0.99

    end = time.time()
    print(f"Time taken in training: {end - start} seconds")
    # save the error data
    dh.save_classification_error_data(error_data, "error_data.csv")

    # Test and display predictions
    errors = dh.load_classification_error_data()

    for x, y in zip(x_test, y_test):
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
    dh.save_classification_error_data(errors)

    errors = np.array(errors)

    print(f"Accuracy on test data: {errors.sum() / errors.size * 100}%")

    print(f"Time taken: {end - start} seconds")

    save(network)


if __name__ == "__main__":
    main()
