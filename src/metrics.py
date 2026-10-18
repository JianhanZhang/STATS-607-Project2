import numpy as np

def compute_rejection_rate(results):
    """
    Compute proportion of simulation replications in which the null hypothesis is rejected.

    Parameters:
    results (list): A list of boolean values indicating whether the null hypothesis is rejected in each simulation replication.

    Returns:
    float: The proportion of simulationreplications in which the null hypothesis is rejected.
    """
    return np.mean(results)