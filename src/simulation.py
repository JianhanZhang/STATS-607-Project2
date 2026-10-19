import numpy as np
import pandas as pd
from src.dgps import generate_data
from src.methods import ClassicalTetradTest, GeneralizedTetradTestGMM, GeneralizedTetradTestPSMD
from src.metrics import compute_rejection_rate
from tqdm import tqdm

def run_one_simulation(num_samples, alpha_2, beta_2, delta, seed):
    """
    Run one simulation replication.

    Parameters:
    num_samples (int): The sample size.
    alpha_2 (float): The alpha_2 parameter in the data-generating mechanism.
    beta_2 (float): The beta_2 parameter in the data-generating mechanism.
    delta (float): The delta parameter in the data-generating mechanism.
    seed (int): The random seed.

    Returns:
    dict: A dictionary containing the resulting p-values.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Generate data
    X, Y, Z, W, U = generate_data(num_samples, alpha_2, beta_2, delta)

    # Initialize the tetrad tests
    classical_tetrad = ClassicalTetradTest(X, Y, Z, W)
    generalized_tetrad_gmm = GeneralizedTetradTestGMM(X, Y, Z, W)
    generalized_tetrad_psmd = GeneralizedTetradTestPSMD(X, Y, Z, W)

    # Compute the p-values for each test
    p_value_classical = classical_tetrad.compute_p_value()
    p_value_generalized_gmm = generalized_tetrad_gmm.compute_p_value()
    p_value_generalized_psmd = generalized_tetrad_psmd.compute_p_value()

    # Return the results as a dictionary
    return {
        'classical': p_value_classical, 
        'generalized_gmm': p_value_generalized_gmm, 
        'generalized_psmd': p_value_generalized_psmd
    }

def run_simulations(num_replications, num_samples_list, parameters_list, alpha, start_seed, raw_path=None, agg_path=None):
    """
    Run multiple simulation replications.

    Parameters:
    num_replications (int): The number of simulation replications to run.
    num_samples_list (list): A list of sample sizes to be tested.
    parameters_list (list): A list containing the lists of parameters for the data-generating mechanism for all the settings. 
        It should look like: [[alpha_2_1, beta_2_1, delta_1], [alpha_2_2, beta_2_2, delta_2], ...].
    alpha (float): The significance level for the hypothesis tests.
    start_seed (int): The random seed for the first replication.
    raw_path (str): The path where the raw results will be saved as a .csv file. If None, the results will not be saved to a file.
    agg_path (str): The path where the aggregated results will be saved as a .csv file. If None, the results will not be saved to a file.

    Returns:
    pd.DataFrame: Two DataFrames, one containing the raw simulation results and another containing the aggregated rejection rates.
    """

    raw_results = {
        'num_replications': [],
        'num_samples': [],
        'alpha_2': [],
        'beta_2': [],
        'delta': [],
        'p_value_classical': [],
        'p_value_generalized_gmm': [],
        'p_value_generalized_psmd': [],
        'rejection_decision_classical': [],
        'rejection_decision_generalized_gmm': [],
        'rejection_decision_generalized_psmd': [], 
        'replication_id': []
    }
    for num_samples in tqdm(num_samples_list, desc="Sample Sizes"):
        for parameters in tqdm(parameters_list, desc="Data-generating Mechanisms"):
            # Run the tests and record the p-values for each simulation replication
            for i in tqdm(range(num_replications), desc="Simulation Replications"):
                current_seed = start_seed + i
                result = run_one_simulation(num_samples=num_samples, 
                                            alpha_2=parameters[0], 
                                            beta_2=parameters[1], 
                                            delta=parameters[2], 
                                            seed=current_seed)
                # Store the raw results
                raw_results['num_replications'].append(num_replications)
                raw_results['num_samples'].append(num_samples)
                raw_results['alpha_2'].append(parameters[0])
                raw_results['beta_2'].append(parameters[1])
                raw_results['delta'].append(parameters[2])
                raw_results['p_value_classical'].append(result['classical'])
                raw_results['p_value_generalized_gmm'].append(result['generalized_gmm'])
                raw_results['p_value_generalized_psmd'].append(result['generalized_psmd'])
                raw_results['rejection_decision_classical'].append(result['classical'] < alpha)
                raw_results['rejection_decision_generalized_gmm'].append(result['generalized_gmm'] < alpha)
                raw_results['rejection_decision_generalized_psmd'].append(result['generalized_psmd'] < alpha)
                raw_results['replication_id'].append(i + 1)

    # Save and return the results
    df_raw = pd.DataFrame(raw_results)
    if raw_path is not None:
        df_raw.to_csv(raw_path, index=False)
    return df_raw



# ---------- MAIN SIMULATION DRIVER ----------
if __name__ == "__main__":
    # Experiment 1: Reproduce Table 2 in the generalized tetrad test paper
    parameters_list = [
        [0, 0, 0], # Setting I
        [0.1, 0.2, 0], # Setting II(a)
        [0.3, 0.4, 0], # Setting II(b)
        [0, 0, 0.15], # Setting III(a)
        [0, 0, 0.3] # Setting III(b)
    ]
    run_simulations(num_replications=1000, num_samples_list=[500, 1000], parameters_list=parameters_list, alpha=0.05, start_seed=1, raw_path='results/raw/raw_results_table2.csv', agg_path='results/raw/results_table2.csv')

    # Experiment 2: Fix beta_2 = delta = 0, and vary alpha_2
    parameters_list = []
    for alpha_2 in np.arange(0, 1.0, 0.1):
        parameters_list.append([alpha_2, 0, 0])
    run_simulations(num_replications=100, num_samples_list=[1000], parameters_list=parameters_list, alpha=0.05, start_seed=1, raw_path='results/raw/raw_results_alpha2.csv', agg_path='results/raw/results_alpha2.csv')

    # Experiment 3: Fix alpha_2 = delta = 0, and vary beta_2
    parameters_list = []
    for beta_2 in np.arange(0, 1.0, 0.1):
        parameters_list.append([0, beta_2, 0])
    run_simulations(num_replications=100, num_samples_list=[1000], parameters_list=parameters_list, alpha=0.05, start_seed=1, raw_path='results/raw/raw_results_beta2.csv', agg_path='results/raw/results_beta2.csv')

    # Experiment 4: Fix alpha_2 = beta_2 = 0, and vary delta
    parameters_list = []
    for delta in np.arange(0, 1.0, 0.1):
        parameters_list.append([0, 0, delta])
    run_simulations(num_replications=100, num_samples_list=[1000], parameters_list=parameters_list, alpha=0.05, start_seed=1, raw_path='results/raw/raw_results_delta.csv', agg_path='results/raw/results_delta.csv')
