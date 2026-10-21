'''Implement functions for producing visualizations.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def make_diagnostic_histogram(input_path, output_path):
    """
    Create diagnostic histograms of p-values under Setting I with sample size 1000.

    Parameters:
    input_path (str): The path to the .csv file containing the raw simulation results.
    output_path (str): The path where the resulting plot will be saved.
    """

    df = pd.read_csv(input_path)

    # Use Setting I with sample size 1000
    df = df[
        (df["num_samples"] == 1000)
        & np.isclose(df["alpha_2"], 0)
        & np.isclose(df["beta_2"], 0)
        & np.isclose(df["delta"], 0)
    ]

    if df.empty:
        raise ValueError("No observations for Setting I, n=1000.")

    methods = {
        "CT": "p_value_classical",
        "GT1": "p_value_generalized_gmm",
        "GT2": "p_value_generalized_psmd"
    }

    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4), sharey=True
    )

    # Set bins
    bins = np.linspace(0, 1, 21)

    # Expected count per bin under Uniform(0, 1)
    expected_count = len(df) / 20

    for ax, (method, column) in zip(axes, methods.items()):
        p_values = df[column].to_numpy()

        if not np.all(
            np.isfinite(p_values)
            & (p_values >= 0)
            & (p_values <= 1)
        ):
            raise ValueError(f"Invalid p-values for {method}.")

        ax.hist(
            p_values,
            bins=bins,
            edgecolor="white",
            alpha=0.85
        )

        # Reference for uniform p-values
        ax.axhline(
            expected_count,
            linestyle="--",
            color="gray",
            label="Expected count when p-values are uniformly distributed"
        )

        # Significance threshold
        ax.axvline(
            0.05,
            linestyle=":",
            color="red",
            label="Significance level"
        )

        # Empirical Type I error
        rejection_rate = np.mean(p_values < 0.05)

        ax.set_title(method)
        ax.set_xlabel("P-value")
        ax.set_xlim(0, 1)
        ax.text(
            0.97, 0.95,
            f"Rejection rate: {rejection_rate:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10
        )

    axes[0].set_ylabel("Frequency")

    fig.suptitle(
        "Distribution of P-values Under the Null (Setting I, n=1000)"
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    Path(output_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def make_publication_ready_line_plot(input_path_alpha2, input_path_beta2, input_path_delta, output_path):
    '''
    Make publication-ready line plots that demonstrate how the rejection rate changes with each of alpha_2, beta_2, and delta.

    Parameters:
    input_path_alpha2 (str): The path to the input .csv file that contains the experiment results with varying alpha_2.
    input_path_beta2 (str): The path to the input .csv file that contains the experiment results with varying beta_2.
    input_path_delta (str): The path to the input .csv file that contains the experiment results with varying delta.
    output_path (str): The path where the resulting plot will be saved.
    '''

    # Load data
    df_alpha2 = pd.read_csv(input_path_alpha2)
    df_beta2 = pd.read_csv(input_path_beta2)
    df_delta = pd.read_csv(input_path_delta)

    # Create 3 panels in one row
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 6), sharey=True
    )

    methods = {
        "CT": "rejection_rate_classical",
        "GT1": "rejection_rate_generalized_gmm",
        "GT2": "rejection_rate_generalized_psmd"
    }

    experiments = [
        (df_alpha2, r"alpha_2", r"(a) Fix $\beta_2=\delta=0$, and vary $\alpha_2$"),
        (df_beta2, r"beta_2", r"(b) Fix $\alpha_2=\delta=0$, and vary $\beta_2$"),
        (df_delta, r"delta", r"(c) Fix $\alpha_2=\beta_2=0$, and vary $\delta$")
    ]

    markers = {
        "CT": "o",
        "GT1": "s",
        "GT2": "^"
    }
    linestyles = {
        "CT": "-",
        "GT1": "--",
        "GT2": ":"
    }

    for ax, (df, parameter, title) in zip(axes, experiments):
        ax.set_title(title)

    for ax, (df, parameter, title) in zip(axes, experiments):
        df = df.sort_values(parameter)

        for method, column in methods.items():
            x = df[parameter].to_numpy()
            p_hat = df[column].to_numpy()
            mcse_column = column.replace(
                "rejection_rate_", "mcse_"
            )
            mcse = df[mcse_column].to_numpy()

            # Approximate 95% Monte Carlo confidence interval
            lower = np.maximum(0, p_hat - 1.96 * mcse)
            upper = np.minimum(1, p_hat + 1.96 * mcse)

            # Plot rejection rates
            line, = ax.plot(
                x,
                p_hat,
                marker=markers[method],
                linestyle=linestyles[method],
                linewidth=2,
                label=method
            )

            # Add 95% Monte Carlo confidence bands
            ax.fill_between(
                x,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.15
            )

        ax.set_title(title)
        ax.set_xlabel(f"$\{parameter.replace('_', '_{') + '}' if '_' in parameter else parameter}$")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    # Add title
    fig.suptitle(
        "Empirical Rejection Rates Across Simulation Settings",
        fontsize=16
    )

    # Reference lines for Type I error
    for i, ax in enumerate(axes[:2]):
        ax.axhline(
            0.05,
            linestyle="--",
            color="gray",
            label="Nominal significance level (0.05)" if i == 0 else None
        )

    # Shared y-axis label
    axes[0].set_ylabel("Empirical Rejection Rate")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.02)
    )
    fig.text(
        0.5, 0.01,
        "Shaded bands represent approximate 95% Monte Carlo confidence intervals.",
        ha="center",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    # Save before displaying
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )



if __name__ == "__main__":
    # Make diagnostic plot
    make_diagnostic_histogram(
        input_path='results/raw/raw_results_table2.csv', 
        output_path='results/figures/plot_p_value_distribution.png'
    )

    # Make publication-ready plot: line plot showing how the rejection rate varies with alpha_2, beta_2, and delta, respectively
    make_publication_ready_line_plot(
        input_path_alpha2='results/raw/results_alpha2.csv', 
        input_path_beta2='results/raw/results_beta2.csv', 
        input_path_delta='results/raw/results_delta.csv', 
        output_path='results/figures/plot_rejection_rate_by_parameter_value.png'
    )