import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    # Load the error data from the CSV file
    file_path = "error_data.csv"
    folder_path = "network-error-data"
    file_path = os.path.join(folder_path, file_path)
    error_data = np.loadtxt(file_path, delimiter=",")

    # Plot the error data
    plt.plot(error_data)
    plt.title("Error Plot")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()

    # pie char for error
    labels = "misclassified", "correctly classified"
    file_path = "classification_error_data.csv"
    file_path = os.path.join(folder_path, file_path)
    classification_error_data = np.loadtxt(file_path, delimiter=",")
    # misses are stored as 1, correct as 0
    misses = np.count_nonzero(classification_error_data)
    correct = len(classification_error_data) - misses
    sizes = [correct, misses]
    explode = (0.1, 0)
    miss_and_correct_figure, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True)
    ax1.axis("equal")
    plt.title("Missed and Correctly classified")
    plt.show()


if __name__ == "__main__":
    main()
