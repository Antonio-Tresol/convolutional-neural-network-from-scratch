import os

import cupy as np
from PIL import Image


# Initialize lists to hold the image data and labels
def get_labels_in_dir(directory):
    unique_labels = []
    for label in os.listdir(directory):
        unique_labels.append(label)
    return unique_labels


# prepare the one-hot encoding for the labels.
def get_one_hot_encoding(unique_labels):
    return np.identity(len(unique_labels))


def get_y_from_label(label, one_hot_encoded, unique_labels):
    label_index = unique_labels.index(label)
    return one_hot_encoded[label_index]


def get_images_and_labels(directory, unique_labels, one_hot_encoded):
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
