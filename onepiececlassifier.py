from logging import exception
import cupy as np
from sklearn.model_selection import train_test_split

from activations import Sigmoid, Softmax
from convolutional import Convolutional
from dense import Dense
from losses import categorical_cross_entropy, categorical_cross_entropy_prime
from network import train, predict
from reshape import Reshape
import datahandler as dh
import time


def main():
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
        X, Y, test_size=0.1, random_state=42
    )

    print("Data split into training and test sets.")
    print(f"Training set size: {x_train.shape[0]}")
    print(f"Test set size: {x_test.shape[0]}")

    network = [
        Convolutional(input_shape=(3, 256, 256), kernel_size=3, depth=5),
        Sigmoid(),
        Convolutional(input_shape=(5, 254, 254), kernel_size=3, depth=5),
        Sigmoid(),
        Convolutional(input_shape=(5, 252, 252), kernel_size=3, depth=5),
        Sigmoid(),
        Reshape(input_shape=(5, 250, 250), output_shape=(5 * 250 * 250, 1)),
        Dense(input_size=5 * 250 * 250, output_size=128),
        Sigmoid(),
        Dense(input_size=128, output_size=64),
        Sigmoid(),
        Dense(input_size=64, output_size=18),
        Sigmoid(),
        Dense(input_size=18, output_size=18),
        Softmax(),
    ]

    start = time.time()
    error_data = []
    # Train the network
    train(
        network,
        categorical_cross_entropy,
        categorical_cross_entropy_prime,
        x_train,
        y_train,
        epochs=10,
        learning_rate=0.1,
        verbose=True,
        error_data=error_data,
    )

    end = time.time()
    print(f"Time taken in training: {end - start} seconds")
    # Test and display predictions
    errors = []

    for x, y in zip(x_test, y_test):
        prediction = predict(network, x)
        indices = int(np.argmax(prediction))
        predicted_label = unique_labels[indices]
        true_label = unique_labels[int(np.argmax(y))]
        # print(f"Prediction: {predicted_label}, Actual: {true_label}")
        if predicted_label != true_label:
            errors.append(0)
        else:
            errors.append(1)

    errors = np.array(errors)

    print(f"Accuracy on test data: {errors.sum() / errors.size * 100}%")
    np.savetxt("errors.csv", errors, delimiter=",")
    error_data = np.array(error_data)
    np.savetxt("error_data.csv", error_data, delimiter=",")

    print(f"Time taken: {end - start} seconds")


if __name__ == "__main__":
    main()
