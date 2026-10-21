# STATS-607-Project2

## Project Description

This repository reproduces the simulation study presented in Section 5 of the paper "A generalized tetrad constraint for testing conditional independence given a latent variable" [2]. The full paper can be found [here](https://arxiv.org/abs/2504.14173), and details about simulation objectives and settings implemented in this repository can be found in `ADEMP.md`.

The tetrad constraint is commonly used to test whether a collection of four variables are mutually independent conditional on an shared unobserved latent common cause. However, the classical tetrad test, proposed by [1], is only guaranteed to work when the underlying structural equation model is linear. The generalized tetrad constraint paper proposes a generalization of the classical tetrad test that is able to test the conditional independence in possibly nonlinear structural equation models.

## Setup Instructions

This project was developed using Python 3.14.7.

To set up the project, first clone the repository and move into the project directory.

```bash
git clone https://github.com/JianhanZhang/STATS-607-Project2.git
cd STATS-607-Project2
```

Once in the project directory, run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Testing

The following command runs the unit tests 

```bash
make test
```

## Running the Analysis

The following command runs the simulation and generates raw simulation results

```bash
make simulate
```

Running the above command should generate the following files:

- `results/raw/raw_results_table2.csv`
- `results/raw/raw_results_alpha2.csv`
- `results/raw/raw_results_beta2.csv`
- `results/raw/raw_results_delta.csv`

The following command aggregates the raw simulation results into summary tables

```bash
make analyze
```

Running the above command should generate the following files:

- `results/raw/results_table2.csv`
- `results/raw/results_alpha2.csv`
- `results/raw/results_beta2.csv`
- `results/raw/results_delta.csv`

The following command produces visualizations on the empirical distribution of p-values and on the 
empirical rejection rates under different data-generating parameters

```bash
make figures
```

Running the above command should generate the following files:

- `results/figures/plot_p_value_distribution.png`
- `results/figures/plot_rejection_rate_by_parameter_value.png`

Alternatively, the following command runs the simulation, produces raw results, aggregates the raw results, and 
makes the visualizations together. It generates all of the files mentioned above in one command.

```bash
make all
```

(Optionally) the following command removes previously generated result files and visualizations

```bash
make clean
```

## Estimated Runtime

An estimated runtime of the full simulation is 14887 seconds (approximately 4 hours). Please note that the actual runtime should depend on the specific computing environment.

## Summary of Key Findings

Under relatively large samples, across the settings tested in the simulation, the generalized tetrad test with PSMD successfully controls the type I error rate under both linear and nonlinear data-generating mechanisms when the null is true and has power close to 1 under linear data-generating mechanisms when the null is false, while the classical tetrad test is sometimes unreliable when the data-generating mechanism is nonlinear. Moreover, the generalized tetrad test with PSMD appears to consistently maintain a low Type I error rate as we increase the degree of deviation from linearity.

## Declaration of AI Use:

Generative AI is involved in the workflow of this project. Its main contributions include:

1. I used AI to help me understand the technical details of the paper, remind me of useful Python syntax and packages, and debug my code.
2. I used AI to write the `GeneralizedTetradTestPSMD` class because I am not yet very familiar with PSMD.
3. I discussed with AI the implementation of some nuanced technical components of `ClassicalTetradTest` and `GeneralizedTetradTestGMM`, such as the implementation of $S_n$ in the generalized tetrad test statistic. I checked the code to make sure that it properly reproduces the original paper.
4. AI helped draft the plotting functions and raw result aggregation functions with general ideas supplied by me. I then modified the draft code to make sure that they are correct and that their outputs are readable and include all the needed components.
5. AI helped with creating some docstrings. Where AI assistance is used, I read and modified the corresponding docstrings to make sure that the descriptions are clear and accurate.

## References

[1] Bollen, K. A. (1990). Outlier screening and a distribution-free test for vanishing tetrads. *Sociological Methods & Research*, 19(1), 80-92.

[2] Ying, N., Zhang, P., Luo, S., & Miao, W. (2025). A generalized tetrad constraint for testing conditional independence given a latent variable. *arXiv preprint arXiv:2504.14173*.