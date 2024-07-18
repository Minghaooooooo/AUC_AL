import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(40)

# Parameters for the true distribution
true_mu = 5
true_sigma = 2

# Generate the dataset
data = np.random.normal(loc=true_mu, scale=true_sigma, size=200)

# Plot the histogram of the dataset
plt.hist(data, bins=20, edgecolor='k', alpha=0.7)
plt.title('Histogram of the Generated Dataset')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

