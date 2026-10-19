
import pandas as pd
from pathlib import Path
from src.metrics import compute_rejection_rate


def aggregate_results(input_path, output_path, alpha=0.05):
    '''
    Aggregate raw simulation results into empirical rejection rates.

    Parameters:
    input_path (str): The path to the .csv file containing the raw simulation results.
    output_path (str): The path where the aggregated results will be saved.
    alpha (float): Significance level used to calculate rejection rates.

    Returns:
    pd.DataFrame: A DataFrame containing the empirical rejection
        rates for each method under each simulation setting.
    '''

    # Load raw simulation results
    df = pd.read_csv(input_path)

    # Aggregate p-values into empirical rejection rates
    aggregated = df.groupby(
        ["num_samples", "alpha_2", "beta_2", "delta"], as_index=False
    ).agg(
        num_replications=(
            "p_value_classical", "size"
        ),
        rejection_rate_classical=(
            "p_value_classical",
            lambda x: compute_rejection_rate(x, alpha)
        ),
        rejection_rate_generalized_gmm=(
            "p_value_generalized_gmm",
            lambda x: compute_rejection_rate(x, alpha)
        ),
        rejection_rate_generalized_psmd=(
            "p_value_generalized_psmd",
            lambda x: compute_rejection_rate(x, alpha)
        )
    )

    # Save the aggregated results
    aggregated.to_csv(output_path, index=False)

    return aggregated


if __name__ == "__main__":
    aggregate_results(
        input_path=f"results/raw/raw_results_table2.csv",
        output_path=f"results/raw/results_table2.csv",
        alpha=0.05
    )
    aggregate_results(
        input_path=f"results/raw/raw_results_alpha2.csv",
        output_path=f"results/raw/results_alpha2.csv",
        alpha=0.05
    )
    aggregate_results(
        input_path=f"results/raw/raw_results_beta2.csv",
        output_path=f"results/raw/results_beta2.csv",
        alpha=0.05
    )
    aggregate_results(
        input_path=f"results/raw/raw_results_delta.csv",
        output_path=f"results/raw/results_delta.csv",
        alpha=0.05
    )