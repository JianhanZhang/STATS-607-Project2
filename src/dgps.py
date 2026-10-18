import numpy as np
import pandas as pd

def generate_data(num_samples, alpha_2, beta_2, delta):
    """
    Generate the variables X, Y, Z, W, and U.

    Parameters:
    num_samples (int): Number of samples to generate.
    alpha_2 (float): The alpha_2 parameter in the data-generating mechanism.
    beta_2 (float): The beta_2 parameter in the data-generating mechanism.
    delta (float): The delta parameter in the data-generating mechanism.

    Returns:
    X, Y, Z, W, U (np.ndarray): Generated variables, each having shape (num_samples,).
    """
    # Generate the unobserved latent variable
    noises = np.random.normal(0, 1, size=(num_samples, 5))
    U = noises[:, 0]

    # Generate the observed variables
    X = 0.5 + 0.5 * U + alpha_2 * U**2 + noises[:, 1]
    Y = -1 + 0.5 * U + beta_2 * U**2 + delta * X + noises[:, 2]
    Z = 0.5 + 1.5 * U + noises[:, 3]
    W = 1 + U + noises[:, 4]

    return X, Y, Z, W, U
