# Statistical Pattern Recognition — Computer Assignment #3

## Abstract

This repository contains a complete, deterministic Python implementation of a Statistical Pattern Recognition assignment covering binary Perceptron learning, least-squares (LS) classification, and multicategory logistic discrimination. It uses the four-class, two-dimensional training table specified by the assignment and evaluates the requested class pairs and multicategory decompositions without replacing the assignment data with an external dataset.

The core algorithms are implemented directly with NumPy rather than a high-level machine-learning estimator. This exposes the optimization and decision rules: zero-initialized Perceptron learning with a bias term, Moore–Penrose pseudoinverse LS classification, and unregularized batch-gradient-descent logistic discrimination. The workflow also exports tabular results, decision-region plots, ambiguity maps, and a ZIP archive for reproducibility.

> **Important numerical note.** Unregularized logistic maximum-likelihood training may not reach a finite stationary solution when the data are linearly separable. The implementation does not treat the iteration limit as convergence; it records both the iteration count and a `converged` flag.

## Research Scope

The implementation covers the following experiments:

| Component             | Experiments                                      | Main output                                                     |
| --------------------- | ------------------------------------------------ | --------------------------------------------------------------- |
| Perceptron            | `omega1` vs. `omega2`; `omega3` vs. `omega2`     | Convergence epochs and separating weights                       |
| Least squares         | `omega1` vs. `omega2`; `omega3` vs. `omega2`     | Linear classifier and training accuracy                         |
| Logistic, one-vs-rest | `omega_i` vs. not `omega_i`, for `i = 1, ..., 4` | Four binary models, training diagnostics, multiclass prediction |
| Logistic, one-vs-one  | All six unordered class pairs                    | Six pairwise models, majority-vote prediction                   |
| Ambiguity analysis    | One-vs-rest and one-vs-one decision grids        | CSV regions and PNG visualizations                              |

The implementation extends an earlier Perceptron-only version to the full assignment while retaining the assignment dataset and the named methods.

## Dataset

The training set contains four classes, each with ten two-dimensional samples. The features are named `x1` and `x2`, and the class labels are `omega1`, `omega2`, `omega3`, and `omega4`.

The script constructs the dataset directly in Python, stores it in a `pandas.DataFrame`, and writes an exact CSV copy to `assignment3_outputs/assignment3_dataset.csv`.

### Dataset schema

| Column   | Type    | Meaning                                  |
| -------- | ------- | ---------------------------------------- |
| `class`  | string  | Class label                              |
| `sample` | integer | Within-class sample index, starting at 1 |
| `x1`     | float   | First feature                            |
| `x2`     | float   | Second feature                           |

A two-dimensional scatter plot of the four classes is also exported as `assignment3_outputs/dataset_overview.png`.

## Methodology

### 1. Feature augmentation

Every linear classifier uses an explicit bias coordinate. For a sample $\mathbf{x} = [x_1, x_2]^T$, the implementation constructs

$$
\tilde{\mathbf{x}} = [x_1, x_2, 1]^T.
$$

The weight vector is therefore $\mathbf{w} = [w_1, w_2, b]^T$, and the linear discriminant is $g(\mathbf{x}) = \mathbf{w}^T\tilde{\mathbf{x}}$.

### 2. Perceptron

For each requested binary class pair, the first class is encoded as $+1$ and the second as $-1$. Training starts from the zero vector and performs sample-wise updates whenever the current pattern violates the margin condition used by the implementation:

$$
y_i(\mathbf{w}^T\tilde{\mathbf{x}}_i) \le 0.
$$

The update is

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta y_i\tilde{\mathbf{x}}_i,
$$

where $\eta$ is the learning rate. The default value in the source is $\eta = 1.0$.

Training stops after a complete epoch with zero mistakes or after the maximum of 1000 epochs. The code returns the final weight vector, the number of epochs, the per-epoch mistake counts, and a Boolean convergence flag.

The script plots the separating boundary

$$
w_1x_1 + w_2x_2 + b = 0.
$$

### 3. Least-squares classification

For a binary pair, targets are encoded as $+1$ and $-1$. With an augmented design matrix $\tilde{X}$, the classifier uses the minimum-norm linear least-squares solution

$$
\mathbf{w} = \tilde{X}^{+}\mathbf{y},
$$

where $\tilde{X}^{+}$ is the Moore–Penrose pseudoinverse computed by `numpy.linalg.pinv`.

Predictions use the sign of the linear score. A score greater than or equal to zero is assigned to the positive class.

### 4. Binary logistic discrimination

The logistic model maps a linear score to a value in $(0,1)$ using the sigmoid function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

Before evaluating the exponential, the source clips the sigmoid input to $[-60, 60]$ to reduce overflow risk for large-magnitude scores.

For binary targets $y_i \in \{0,1\}$, the batch loss is the mean binary cross-entropy:

$$
L(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right],
$$

where

$$
p_i = \sigma(\tilde{\mathbf{x}}_i^T\mathbf{w}).
$$

The gradient used by the implementation is

$$
\nabla L(\mathbf{w}) = \frac{\tilde{X}^T(\mathbf{p}-\mathbf{y})}{n}.
$$

The parameter update is batch gradient descent:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha\nabla L(\mathbf{w}_t),
$$

The default learning rate is $\alpha = 0.25$, the maximum number of iterations is `10000`, and the stopping tolerance is $10^{-8}$ on the Euclidean norm of the proposed parameter change. The implementation stops when

$$
\lVert \mathbf{w}_{t+1}-\mathbf{w}_t \rVert_2 \le 10^{-8}.
$$

The arguments of the logarithms in the loss are clipped to $[10^{-12}, 1-10^{-12}]$ before `log` is evaluated.

### 5. One-vs-rest logistic discrimination

Four independent binary logistic models are trained:

* `omega1` vs. not `omega1`
* `omega2` vs. not `omega2`
* `omega3` vs. not `omega3`
* `omega4` vs. not `omega4`

At prediction time, the implementation computes the sigmoid score from each one-vs-rest classifier and selects the class with the largest score:

$$
\hat{c}(\mathbf{x}) = \arg\max_{c \in \{\omega_1,\omega_2,\omega_3,\omega_4\}} p_c(\mathbf{x}).
$$

These four sigmoid outputs are independent one-vs-rest scores; the code does **not** renormalize them into a probability distribution that sums to one.

### 6. One-vs-one logistic discrimination

The four classes produce six unordered binary pairs. The class ordering is deterministic, and the first class in lexicographic order is encoded as the positive class for each pair.

For a test point, each pair votes for one class using a 0.5 probability threshold. The predicted class is the class with the largest vote count. A point is also marked as ambiguous when more than one class shares the maximum vote count.

Because `numpy.argmax` returns the first maximum, a tied vote still produces a deterministic predicted class. The separate `tied` flag records that the point is categorically ambiguous.

## Reproducibility and Determinism

The source sets the NumPy random seed to `42`. None of the core training loops introduces stochastic minibatching or randomized initialization; model weights are initialized to zero. The Perceptron experiment also preserves the dataset order in the constructed `DataFrame`, so its epoch count is tied to this exact sample presentation order.

A smaller Perceptron epoch count for one class pair does not imply that the corresponding class separation is universally easier under every possible sample ordering. The reported comparison applies specifically to the implemented initialization, ordering, learning rate, and dataset.

## Implementation Structure

The code is organized as a single executable Python script with the following main functions:

| Function               | Role                                                        |
| ---------------------- | ----------------------------------------------------------- |
| `add_bias`             | Appends the constant bias coordinate                        |
| `get_binary_pair`      | Extracts a requested class pair and creates $\pm1$ labels   |
| `perceptron_train`     | Trains the Perceptron and records convergence history       |
| `plot_binary_boundary` | Visualizes binary data and a learned linear boundary        |
| `least_squares_train`  | Computes the pseudoinverse LS solution and accuracy         |
| `sigmoid`              | Stable sigmoid evaluation with input clipping               |
| `logistic_train`       | Batch gradient descent for unregularized logistic loss      |
| `train_one_vs_rest`    | Fits the four one-vs-rest logistic models                   |
| `predict_one_vs_rest`  | Scores samples using the OVR models                         |
| `train_one_vs_one`     | Fits all six pairwise logistic models                       |
| `pairwise_probability` | Returns the requested orientation of a pairwise probability |
| `predict_one_vs_one`   | Aggregates pairwise decisions by majority voting            |

### Core dependencies

The source uses:

* Python standard library: `math`, `shutil`, `sys`, `zipfile`, `itertools`, `pathlib`
* NumPy
* pandas
* Matplotlib
* IPython when available for richer display in notebook environments
* `google.colab.files` only when the script is executed inside Google Colab

The implementation does not require scikit-learn, PyTorch, TensorFlow, or another high-level machine-learning framework.

## Installation

A minimal environment can be prepared with:

```bash
python -m pip install numpy pandas matplotlib
```

The script is pure Python apart from these runtime dependencies. `IPython` is optional; when `IPython.display` is unavailable, the source falls back to `print`.

For reproducibility, use a recent Python 3 environment and record the exact package versions printed by the script at execution time. The source does not pin dependency versions.

## Usage

Run the script from the repository root:

```bash
python "assignment3_solution(2).py"
```

The program creates `assignment3_outputs/` and generates the assignment dataset, numerical result tables, plots, ambiguity-region CSV files, and an artifact manifest. It then creates `assignment3_outputs.zip` in the working directory.

When the script runs inside Google Colab, the final cell attempts to trigger a download of the ZIP archive. Outside Colab, the archive remains on disk.

### Expected output artifacts

| Artifact                                                         | Description                                   |
| ---------------------------------------------------------------- | --------------------------------------------- |
| `assignment3_outputs/assignment3_dataset.csv`                    | Exact assignment dataset in tabular form      |
| `assignment3_outputs/dataset_overview.png`                       | Four-class training-data visualization        |
| `assignment3_outputs/perceptron_results.csv`                     | Perceptron convergence and final weights      |
| `assignment3_outputs/perceptron_omega1_vs_omega2.png`            | Perceptron boundary for `omega1` vs. `omega2` |
| `assignment3_outputs/perceptron_omega3_vs_omega2.png`            | Perceptron boundary for `omega3` vs. `omega2` |
| `assignment3_outputs/least_squares_results.csv`                  | LS weights and training accuracy              |
| `assignment3_outputs/logistic_one_vs_rest_results.csv`           | OVR optimizer status and training accuracy    |
| `assignment3_outputs/logistic_one_vs_rest_predictions.csv`       | OVR sample-level predictions and scores       |
| `assignment3_outputs/logistic_one_vs_rest_regions.png`           | OVR decision-region visualization             |
| `assignment3_outputs/logistic_one_vs_rest_ambiguous_regions.csv` | OVR ambiguity grid points                     |
| `assignment3_outputs/logistic_one_vs_one_results.csv`            | OVO pairwise optimizer status and accuracy    |
| `assignment3_outputs/logistic_one_vs_one_predictions.csv`        | OVO sample-level predictions and vote counts  |
| `assignment3_outputs/logistic_one_vs_one_regions.png`            | OVO decision-region visualization             |
| `assignment3_outputs/logistic_one_vs_one_ambiguous_regions.csv`  | OVO vote-tie grid points                      |
| `assignment3_outputs/experiment_summary.csv`                     | Consolidated experiment metrics               |
| `assignment3_outputs/artifact_manifest.csv`                      | Manifest of generated files                   |
| `assignment3_outputs.zip`                                        | ZIP archive containing generated artifacts    |

> The script copies a source file into the output directory only when it finds a file named `assignment3_solution.py` beside the script or at `/content/assignment3_solution.py`. Therefore, if the repository contains only the uploaded filename `assignment3_solution(2).py`, the run remains reproducible, but the automatic source-copy step is not activated unless the file is also provided under the expected name.

## Results

The following values were obtained by executing the supplied source as provided, using its fixed initialization and sample order.

### Perceptron

| Class pair            | Converged | Epochs |
| --------------------- | --------: | -----: |
| `omega1` vs. `omega2` |       Yes |      9 |
| `omega3` vs. `omega2` |       Yes |      5 |

The final weight vectors were:

| Class pair            | $w_1$ | $w_2$ | Bias $b$ |
| --------------------- | ----: | ----: | -------: |
| `omega1` vs. `omega2` | -10.2 |  11.3 |     13.0 |
| `omega3` vs. `omega2` |  -5.5 |   6.4 |      5.0 |

### Least squares

| Class pair            | Training accuracy |
| --------------------- | ----------------: |
| `omega1` vs. `omega2` |             95.0% |
| `omega3` vs. `omega2` |             95.0% |

### Logistic one-vs-rest

| Classifier                | Training accuracy | Iterations | Converged | Final loss |
| ------------------------- | ----------------: | ---------: | --------: | ---------: |
| `omega1` vs. not `omega1` |             75.0% |        444 |       Yes |   0.489403 |
| `omega2` vs. not `omega2` |             92.5% |      4,157 |       Yes |   0.215195 |
| `omega3` vs. not `omega3` |             92.5% |      1,245 |       Yes |   0.313237 |
| `omega4` vs. not `omega4` |            100.0% |     10,000 |        No |   0.017750 |

The overall one-vs-rest multiclass training accuracy is **90.0%**.

### Logistic one-vs-one

| Pair                  | Training accuracy | Iterations | Converged | Final loss |
| --------------------- | ----------------: | ---------: | --------: | ---------: |
| `omega1` vs. `omega2` |            100.0% |     10,000 |        No |   0.005305 |
| `omega1` vs. `omega3` |             90.0% |        962 |       Yes |   0.325618 |
| `omega1` vs. `omega4` |            100.0% |     10,000 |        No |   0.029952 |
| `omega2` vs. `omega3` |            100.0% |     10,000 |        No |   0.002469 |
| `omega2` vs. `omega4` |            100.0% |     10,000 |        No |   0.006054 |
| `omega3` vs. `omega4` |            100.0% |     10,000 |        No |   0.007358 |

The mean pairwise training accuracy is **98.33%**. The reported one-vs-one multiclass training accuracy after majority voting is **95.0%**.

### Interpretation of the logistic convergence behavior

The `converged=False` rows do not indicate failed code execution. They mean that the update-step stopping criterion was not reached within the 10,000-iteration budget. Several of these models still reach 100% training accuracy and very small losses. This is consistent with the behavior of unregularized logistic models on strongly or perfectly separable data: the optimizer can continue increasing the weight magnitude and improving the likelihood without reaching a finite parameter vector that satisfies a conventional finite-optimum criterion.

The implementation preserves this distinction rather than labeling every maximum-iteration run as converged.

## Ambiguity Analysis

### One-vs-rest ambiguity rule

The code evaluates a $500 \times 500$ grid over an expanded feature-space rectangle. A grid point is flagged as potentially ambiguous when either of two conditions holds:

$$
\max_c p_c < 0.60
$$

or

$$
\max_c p_c - \operatorname{secondmax}_c p_c < 0.10.
$$

The first rule identifies locations where even the strongest independent OVR score is relatively weak; the second identifies locations where the best and second-best scores are close.

### One-vs-one ambiguity rule

For the pairwise system, ambiguity is defined categorically rather than probabilistically. A point is ambiguous when two or more classes receive the same maximum number of votes. The script saves these locations to `logistic_one_vs_one_ambiguous_regions.csv` and shows them on the OVO decision-region plot.


