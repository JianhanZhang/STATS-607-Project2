import numpy as np

def compute_rejection_rate(p_values, alpha=0.05):
    """
    Compute proportion of simulation replications in which the null hypothesis is rejected.

    Parameters:
    p_values (list): A list of p-values from each simulation replication.
    alpha (float): The significance level for the hypothesis test.

    Returns:
    float: The proportion of simulation replications in which the null hypothesis is rejected.
    """
    return np.mean(np.array(p_values) < alpha)