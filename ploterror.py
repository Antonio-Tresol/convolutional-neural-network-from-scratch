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


if __name__ == "__main__":
    main()
