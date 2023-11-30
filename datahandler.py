from logging import exception
import os

import cupy as np
from PIL import Image


def get_labels_in_dir(directory):
    """
    Get the list of unique labels in the given directory.

    Args:
        directory (str): The directory path.

    Returns:
        list: A list of unique labels found in the directory.
    """
    unique_labels = []
    for label in os.listdir(directory):
        unique_labels.append(label)
    return unique_labels


def get_one_hot_encoding(unique_labels):
    """
    Returns the one-hot encoding for the given unique labels.

    Parameters:
    unique_labels (list): A list of unique labels.

    Returns:
    numpy.ndarray: The one-hot encoding matrix.
    """
    return np.identity(len(unique_labels))


def get_y_from_label(label, one_hot_encoded, unique_labels):
    """
    Get the one-hot encoded representation of a label.

    Parameters:
    label (str): The label to convert to one-hot encoded representation.
    one_hot_encoded (list): The list of one-hot encoded labels.
    unique_labels (list): The list of unique labels.

    Returns:
    list: The one-hot encoded representation of the label.
    """
    label_index = unique_labels.index(label)
    return one_hot_encoded[label_index]


def get_images_and_labels(directory, unique_labels, one_hot_encoded):
    """
    Load images from a directory and their corresponding labels.

    Args:
        directory (str): The directory path containing the images.
        unique_labels (list): A list of unique labels for the images.
        one_hot_encoded (bool): Flag indicating whether the labels should be one-hot encoded.

    Returns:
        tuple: A tuple containing two lists: X (images) and Y (labels).
    """
    X = []
    Y = []

    print("Loading images...")

    # Iterate over each directory in 'training_data'
    for label in os.listdir(directory):
        subdirectory = os.path.join(directory, label)
        if os.path.isdir(subdirectory):
            for image_filename in os.listdir(subdirectory):
                image_path = os.path.join(subdirectory, image_filename)
                # Open the image file
                image = Image.open(image_path)
                # Convert the image to a numpy array and append it to X
                shape = np.transpose(np.array(image), (2, 0, 1))
                X.append(shape)
                # Append the label of this image to y
                Y.append(get_y_from_label(label, one_hot_encoded, unique_labels))
    return X, Y


def split_data_into_batches(training_data, training_data_output, batch_size=32):
    """
    Split the data into batches.

    Args:
        training_data (list): The input data.
        training_data_output (list): The expected outputs for the input data.
        batch_size (int): The size of each batch.

    Returns:
        list: A list of tuples containing the input and target data for each batch.
    """
    training_data_output_split = np.array_split(training_data_output, batch_size)
    training_data_split = np.array_split(training_data, batch_size)
    return training_data_split, training_data_output_split


def save_classification_error_data(
    error_data, file_path="classification_error_data.csv"
):
    """
    Saves the error data to a CSV file.

    Args:
        error_data (list): The list of errors.
        file_path (str): The path to the file where the error data should be saved.
    """
    error_data = np.array(error_data)
    np.savetxt(file_path, error_data, delimiter=",")


def load_classification_error_data(file_path="classification_error_data.csv"):
    """
    Loads the error data from a CSV file.

    Args:
        file_path (str): The path to the file where the error data is stored.

    Returns:
        list: The list of errors.
    """
    try:
        error_data = np.loadtxt(file_path, delimiter=",").tolist()
    except Exception as e:
        print("Error loading historical error data: ", e)
        error_data = []
    return error_data
