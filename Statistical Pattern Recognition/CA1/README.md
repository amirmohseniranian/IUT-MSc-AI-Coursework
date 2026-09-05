# Statistical Pattern Recognition: Bayes Minimum Error and Bayes Minimum Risk Classification

## Overview

This project implements a reproducible statistical pattern recognition pipeline based on Bayesian classification. It focuses on Bayes Minimum Error (BME) and Bayes Minimum Risk (BMR) decision rules for Gaussian class-conditional models, including parameter estimation, posterior computation, risk minimization, leave-one-out evaluation, and experimental analysis.

The source implementation is a corrected version of a statistical pattern recognition assignment. It has two main parts:

1. A two-class Gaussian Bayes classifier experiment based on textbook parameters.
2. A multiclass Gaussian classification framework evaluated on Iris, a chemical-analysis dataset equivalent, and a synthetic Normal dataset.

The implementation addresses several correctness issues in the original submission, including Gaussian rather than uniform sampling, the correct interpretation of covariance, separate class-wise evaluation, fixed random seeds, log-domain discriminants, and reproducible artifact generation.

---

# Scientific Background

## Bayesian Classification

Given an observation vector $x$ and class label $\omega_i$, Bayesian classification models the posterior probability as:

$$
P(\omega_i \mid x)
=
\frac{
p(x \mid \omega_i)P(\omega_i)
}{
\sum_{j=1}^{C} p(x \mid \omega_j)P(\omega_j)
}.
$$

where:

* $p(x \mid \omega_i)$ is the class-conditional probability density.
* $P(\omega_i)$ is the prior probability of class $\omega_i$.
* $C$ is the number of classes.

Under the minimum-error rule, each sample is assigned to the class with the highest posterior probability.

---

## Bayes Minimum Error (BME)

For equal misclassification costs, the BME decision rule selects the class that maximizes the discriminant function:

$$
g_i(x)=\log p(x\mid\omega_i)+\log P(\omega_i).
$$

The predicted class is:

$$
\hat{\omega}=\arg\max_i g_i(x).
$$

The implementation uses logarithmic discriminants instead of raw probability densities to improve numerical stability.

---

## Bayes Minimum Risk (BMR)

When classification errors have different costs, the optimal decision minimizes the conditional risk:

$$
R(\alpha_i\mid x)
=
\sum_{j=1}^{C}
\lambda_{ij}P(\omega_j\mid x)
$$

where:

* $\lambda_{ij}$ is the loss associated with deciding class $i$ when the true class is $j$.
* $R(\alpha_i\mid x)$ is the conditional risk of decision $\alpha_i$.

The implementation computes posterior probabilities and conditional risks, then selects the decision with the minimum expected loss.

---

# Project Structure

```
.
├── README.md
├── statistical_pattern_recognition_assignment.py
└── statistical_pattern_recognition_assignment1_outputs/
    ├── CSV evaluation files
    ├── generated datasets
    ├── plots
    └── reproducibility_manifest.json
```

---

# Methodology

## Part I — Two-Class Gaussian Experiment

The first experiment evaluates BME and BMR classification for two Gaussian classes.

Configuration:

* Number of classes: 2
* Samples per class: 100
* Priors:

$$
P(\omega_1)=P(\omega_2)=0.5
$$

* Covariance:

$$
\Sigma=0.2I
$$

* First class mean:

$$
\mu_1=[1,1]^T
$$

* Second class means:

$$
\mu_2=[1.5,1.5]^T
$$

and

$$
\mu_2=[3,3]^T
$$

The loss matrix used for BMR is:

$$
\Lambda=
\begin{bmatrix}
0 & 1\\
0.5 & 0
\end{bmatrix}
$$

These parameters are defined directly in the implementation.

The experiment evaluates:

* empirical error rate
* class-wise error rate
* accuracy
* number of misclassified samples
* theoretical Gaussian Bayes error

---

# Part II — Multiclass Gaussian Classification

The second part implements a general Gaussian BME classifier.

## Maximum-Likelihood Parameter Estimation

For each class, the mean vector is estimated as:

$$
\hat{\mu}_i
=
\frac{1}{n_i}
\sum_{k=1}^{n_i}x_k
$$

The covariance matrix uses the maximum-likelihood estimator:

$$
\hat{\Sigma}_i
=
\frac{1}{n_i}
\sum_{k=1}^{n_i}
(x_k-\hat{\mu}_i)(x_k-\hat{\mu}_i)^T
$$

The implementation intentionally uses divisor $n$ rather than $n-1$, matching maximum-likelihood Gaussian estimation.

---

# Datasets and Experiments

## Iris Dataset

Configuration:

* Samples: 150
* Classes: 3
* Features: 4
* Evaluation: 150-fold leave-one-out classification

The feature order is:

1. Petal Length
2. Sepal Length
3. Sepal Width
4. Petal Width

The implementation uses the standard Iris dataset and performs leave-one-out Bayesian classification.

---

## Chemical Analysis Dataset Equivalent

The original Liquid dataset was unavailable. The implementation uses the standard Wine dataset as a documented equivalent because it matches:

* 178 samples
* 3 classes
* class distribution 59 / 71 / 48
* chemical-analysis measurements

Six original chemical features are retained:

* Alcohol
* Malic Acid
* Ash
* Alcalinity of Ash
* Magnesium
* Total Phenols

This substitution is explicitly documented in the source implementation.

---

## Normal Gaussian Dataset

The Normal experiment creates deterministic training and testing datasets.

Configuration:

Training:

* 1000 samples
* 500 samples per class

Testing:

* 1000 samples
* 500 samples per class

Gaussian parameters:

$$
\mu_1=[0,0]^T
$$

$$
\mu_2=[4,0]^T
$$

$$
\Sigma=
\begin{bmatrix}
2 & 0\\
0 & 2
\end{bmatrix}
$$

The datasets are generated using fixed random seeds for reproducibility.

---

# Implementation Details

## Main Components

### Gaussian Probability Model

The function `log_gaussian_pdf` computes the logarithm of the multivariate Gaussian density.

It evaluates:

$$
\log p(x|\omega_i)
$$

using matrix operations and covariance inversion.

---

## Posterior Computation

Posterior probabilities are obtained from log-likelihoods and priors:

$$
P(\omega_i|x)
$$

with numerical stabilization through shifted exponentiation.

---

## Evaluation Metrics

The project reports:

* overall error rate
* class-specific error rates
* accuracy
* confusion matrices
* misclassification details

The evaluation functions compute both global and class-level performance statistics.

---

# Installation

Recommended environment:

* Python 3.10+
* NumPy
* Pandas
* Matplotlib
* SciPy
* Scikit-learn

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

---

# Usage

Run:

```bash
python statistical_pattern_recognition_assignment.py
```

The program creates an output directory:

```
statistical_pattern_recognition_assignment1_outputs/
```

containing:

* generated datasets
* CSV performance reports
* confusion matrices
* misclassification records
* figures
* reproducibility metadata

The output directory and random seed are configured in the source code.

---

# Reproducibility

The experiment uses a fixed random seed:

$$
\text{seed}=20260822
$$

The implementation stores:

* Python version
* NumPy version
* Pandas version
* experiment parameters
* covariance matrices
* priors
* loss matrices
* dataset configurations

in a reproducibility manifest.

---
