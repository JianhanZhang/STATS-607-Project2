# ADEMP Summary

## Aims

This simulation study aims to evaluate the empirical performance of the generalized tetrad test, a newly proposed method that tests for the independence of a quartet of variables conditional on an unobserved latent variable. 

## Data-generating mechanisms

Let $U$ be the unobserved latent variable, and $X,Y,Z,W$ be the variables whose conditional independence we want to test. The data are generated according to 

$$
\begin{aligned}
X &= \alpha_0 + \alpha_1 U + \alpha_2 U^2 + \epsilon_1, \\
Y &= \beta_0 + \beta_1 U + \beta_2 U^2 + \delta X + \epsilon_2, \\
Z &= \gamma_0 + \gamma_1 U + \epsilon_3, \\
W &= \eta_0 + \eta_1 U + \epsilon_4, \\
(U, \epsilon_1, \epsilon_2, \epsilon_3, \epsilon_4)^T &\sim N(0, I_5), \\
(\alpha_0, \beta_0, \gamma_0, \eta_0) &= (0.5, -1, 0.5, 1), \\
(\alpha_1, \beta_1, \gamma_1, \eta_1) &= (0.5, 0.5, 1.5, 1).
\end{aligned}
$$

The parameters $\alpha_2$, $\beta_2$, and $\delta$ are varied in the simulation study. In particular, $\alpha_2$ controls the degree of deviation from linearity in the generation of $X$, $\beta_2$ controls the degree of deviation from linearity in the generation of $Y$, and $\delta$ controls the dependence of $Y$ on $X$ conditional on $U$. The data-generating mechanism is perfectly linear if and only if $\alpha_2=\beta_2=0$. $X,Y,Z,W$ are independent conditional on $U$ if and only if $\delta=0$.

## Estimands

We aim to estimate the probability of rejecting the null hypothesis under each data-generating mechanism. This rejection probability corresponds to the Type I error rate when the null hypothesis is true, and corresponds to the statistical power when the null hypothesis is false. 

## Methods

We compare three methods:

- $GT_1$: The proposed generalized tetrad test, with the bridge functions estimated by generalized method of moments.
- $GT_2$: The proposed generalized tetrad test, with the bridge functions estimated by penalized sieve minimum distance.
- $CT$: The classical tetrad test, which is designed for only linear data-generating mechanisms.

## Performance measures

The performance of the methods are evaluated based on the empirical rejection probabilities at a significance level of 0.05. In particular, the empirical rejection probability is calculated as the proportion of simulation replications in which the null hypothesis is rejected.

## Simulation design matrix

The simulation study consists of 4 experiments, which are summarized below.

### Experiment 1

This experiment reproduces Table 2 in the original generalized tetrad paper.

| Setting | $\alpha_2$ | $\beta_2$ | $\delta$ | $H_0$ |
|---|---|---|---|---|
| I | 0 | 0 | 0 | True |
| II.a | 0.1 | 0.2 | 0 | True |
| II.b | 0.3 | 0.4 | 0 | True |
| III.a | 0 | 0 | 0.15 | False |
| III.b | 0 | 0 | 0.3 | False |

For each setting, we consider sample sizes $n=500$ and
$n=1000$, with 1,000 replications per sample size.
The significance level is fixed at 0.05. The result will be summarized in a table.

### Experiment 2

This experiment investigates how the methods' performance vary with $\alpha_2$.

| Setting | $\alpha_2$ | $\beta_2$ | $\delta$ | $H_0$ |
|---|---|---|---|---|
| I | 0 | 0 | 0 | True |
| II | 0.1 | 0 | 0 | True |
| III | 0.2 | 0 | 0 | True |
| IV | 0.3 | 0 | 0 | True |
| V | 0.4 | 0 | 0 | True |
| VI | 0.5 | 0 | 0 | True |
| VII | 0.6 | 0 | 0 | True |
| VIII | 0.7 | 0 | 0 | True |
| IX | 0.8 | 0 | 0 | True |
| X | 0.9 | 0 | 0 | True |

For each setting, we consider sample size $n=1000$, with 100 replications.
The significance level is fixed at 0.05. The result will be summarized as a line plot.

### Experiment 3

This experiment investigates how the methods' performance vary with $\beta_2$.

| Setting | $\alpha_2$ | $\beta_2$ | $\delta$ | $H_0$ |
|---|---|---|---|---|
| I | 0 | 0 | 0 | True |
| II | 0 | 0.1 | 0 | True |
| III | 0 | 0.2 | 0 | True |
| IV | 0 | 0.3 | 0 | True |
| V | 0 | 0.4 | 0 | True |
| VI | 0 | 0.5 | 0 | True |
| VII | 0 | 0.6 | 0 | True |
| VIII | 0 | 0.7 | 0 | True |
| IX | 0 | 0.8 | 0 | True |
| X | 0 | 0.9 | 0 | True |

For each setting, we consider sample size $n=1000$, with 100 replications.
The significance level is fixed at 0.05. The result will be summarized as a line plot.

### Experiment 4

This experiment investigates how the methods' performance vary with $\delta$.

| Setting | $\alpha_2$ | $\beta_2$ | $\delta$ | $H_0$ |
|---|---|---|---|---|
| I | 0 | 0 | 0 | True |
| II | 0 | 0 | 0.1 | False |
| III | 0 | 0 | 0.2 | False |
| IV | 0 | 0 | 0.3 | False |
| V | 0 | 0 | 0.4 | False |
| VI | 0 | 0 | 0.5 | False |
| VII | 0 | 0 | 0.6 | False |
| VIII | 0 | 0 | 0.7 | False |
| IX | 0 | 0 | 0.8 | False |
| X | 0 | 0 | 0.9 | False |

For each setting, we consider sample size $n=1000$, with 100 replications.
The significance level is fixed at 0.05. The result will be summarized as a line plot.
