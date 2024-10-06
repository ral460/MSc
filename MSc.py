import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# Number of time steps
T = 100

# Number of variables
n_vars = 10

# Initialize list to store time-varying precision matrices
precision_matrices = []

# Generate a time-varying precision matrix for each time step
for t in range(T):
    precision_matrix = np.eye(n_vars) * 10
    for i in range(n_vars - 1):
        precision_matrix[i, i + 1] = precision_matrix[i + 1, i] = -2 * (1 + 0.1 * np.sin(2 * np.pi * t / T))
    precision_matrices.append(precision_matrix)

# Number of samples to generate at each time step
num_samples = 1

# Initialize list to store generated samples
samples = []

# Generate samples for each time step
for t in range(T):
    # Invert the precision matrix to get the covariance matrix
    cov_matrix = inv(precision_matrices[t])

    # Cholesky decomposition to get the lower triangular matrix
    L = np.linalg.cholesky(cov_matrix)

    # Generate standard normal random variables
    z = np.random.normal(size=(num_samples, n_vars))

    # Generate the samples with the specified covariance matrix
    sample = z @ L.T
    samples.append(sample)

# Convert samples list to a numpy array for easier plotting
samples = np.vstack(samples)

# Plot all time series
plt.figure(figsize=(12, 8))
for i in range(n_vars):
    plt.plot(samples[:, i], label=f'Series {i+1}')
plt.title('Generated Time Series from Time-Varying Multivariate Normal Distribution')
plt.xlabel('Time Index')
plt.ylabel('Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()