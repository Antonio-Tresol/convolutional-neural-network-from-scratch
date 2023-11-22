import numpy as np
import matplotlib.pyplot as plt

# Load the error data from the CSV file
error_data = np.loadtxt("error_data.csv", delimiter=",")

# Plot the error data
plt.plot(error_data)
plt.title("Error Plot")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()
