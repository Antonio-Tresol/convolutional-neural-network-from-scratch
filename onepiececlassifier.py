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


start = time.time()

# Replace this with the path to your 'training_data' directory
directory = r"C:\Users\anton\Downloads\Neural-Network-master\Data_resized_data_size_256x256\ConvolutionalNetwork\training_data"

unique_labels = []

try:
    unique_labels = dh.get_labels_in_dir(directory)
except FileNotFoundError:
    print("Directory not found. Please enter a valid directory path.")
    exit(1)


one_hot_encoded = dh.get_one_hot_encoding(unique_labels)

X, Y = dh.get_images_and_labels(directory, unique_labels, one_hot_encoded)

# Convert X and y to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Normalize the image data to the range [0, 1]
X = X.astype("float32") / 255.0

print("Data normalized.")

# Use one-hot encoded labels for splitting
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42
)

print("Data split into training and test sets.")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Modify network architecture if needed
network = [
    Convolutional(input_shape=(3, 256, 256), kernel_size=3, depth=5),
    # output_shape=(5, 254, 254)
    Sigmoid(),
    Convolutional(input_shape=(5, 254, 254), kernel_size=3, depth=5),
    # output_shape=(5, 252, 252)
    Sigmoid(),
    Convolutional(input_shape=(5, 252, 252), kernel_size=3, depth=5),
    # output_shape=(5, 250, 250)
    Sigmoid(),
    Reshape(input_shape=(5, 250, 250), output_shape=(5 * 250 * 250, 1)),
    # output_shape=(5, 250, 250)
    Dense(input_size=5 * 250 * 250, output_size=128),
    Sigmoid(),
    Dense(input_size=128, output_size=64),
    Sigmoid(),
    Dense(input_size=64, output_size=18),
    Sigmoid(),
    Dense(input_size=18, output_size=18),
    Softmax(),
]
error_data = []
# Train the network
train(
    network,
    categorical_cross_entropy,
    categorical_cross_entropy_prime,
    X_train,
    Y_train,
    epochs=1,
    learning_rate=0.01,
    verbose=True,
    error_data=error_data,
)

end = time.time()
print(f"Time taken: {end - start} seconds")
# Test and display predictions

for x, y in zip(X_test, Y_test):
    prediction = predict(network, x)
    indices = int(np.argmax(prediction))
    predicted_label = unique_labels[indices]
    true_label = unique_labels[int(np.argmax(y))]
    print(f"Prediction: {predicted_label}, Actual: {true_label}")

# Convert the lists to NumPy arrays and save them into CSV files# Assuming error_data is your list of errors
error_data = np.array(error_data)
np.savetxt("error_data.csv", error_data, delimiter=",")

print(f"Time taken: {end - start} seconds")
