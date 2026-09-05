# Deep Learning — Computer Assignment 1

> A from-scratch Python reconstruction of a two-part introductory deep-learning assignment: a three-class linear perceptron and a two-layer nonlinear network for a moon-shaped dataset.

## Abstract

This project reconstructs the supplied MATLAB assignment in Python, preserving its experimental structure while replacing several implementation problems with mathematically consistent learning procedures. The work addresses two supervised classification tasks on small two-dimensional datasets.

**Task 1** uses the first 70 samples for training and the final 30 samples for testing. The target labels are reconstructed from the documented coordinate rules to form three classes: the lower half-plane, the upper-left quadrant, and the upper-right quadrant. A single-layer multiclass perceptron is trained from scratch with NumPy, using one linear score per class and winner-takes-all prediction.

**Task 2** uses the same 70/30 split on a binary moon-shaped dataset. Because a single linear decision surface cannot represent the curved class boundary, the implementation uses a two-layer network with a `tanh` hidden layer and a sigmoid output. Forward propagation, binary cross-entropy, backpropagation, parameter updates, accuracy, and confusion matrices are implemented directly with NumPy rather than with a ready-made machine-learning or neural-network estimator.

The repository is designed as an auditable, reproducible scientific-computing project: initialization is seeded, Task 2 standardization is fitted only on the training subset, held-out samples are evaluated after training, and result artifacts are written automatically to `results/`.

## Research Scope and Provenance

The source file identifies the project as a reconstruction of a supplied MATLAB assignment and records the reconstruction decisions made during the audit. In particular, the original assignment materials contained inconsistent target handling for Task 1 and an invalid evaluation path for Task 2; the Python implementation therefore uses a genuine multiclass perceptron for the three-class problem and a genuine two-layer nonlinear model for the moon-shaped problem.

The source code also states that the assignment restricts the solution from using ready-made classification or neural-network toolboxes. The implementation respects that restriction by coding the learning rules and evaluation metrics directly.

## Scientific Background

### Linear classification and the perceptron

A single-layer perceptron computes a linear score for each class. For an input vector $x \in \mathbb{R}^{d}$, class $k$ has parameters $w_k \in \mathbb{R}^{d}$ and bias $b_k \in \mathbb{R}$. The score is

$$
s_k(x) = w_k^{\top}x + b_k.
$$

The predicted class is the index of the largest score:

$$
\hat{y} = \operatorname*{arg\,max}_{k \in \{1,\ldots,K\}} s_k(x).
$$

The implementation uses $K=3$. During training, labels use the assignment representation $1,2,3$, while NumPy arrays are indexed internally as $0,1,2$.

For a misclassified training example with true class $t$ and predicted class $p$, the implemented update is

$$
w_t \leftarrow w_t + x, \qquad b_t \leftarrow b_t + 1,
$$

$$
w_p \leftarrow w_p - x, \qquad b_p \leftarrow b_p - 1.
$$

This corresponds to a unit learning rate. The update increases the score of the correct class and decreases the score of the predicted incorrect class. The algorithm is appropriate for linearly separable class geometry, which is the regime targeted by Task 1.

### Nonlinear representation learning

The moon-shaped dataset is inherently nonlinear. A single affine decision surface cannot capture a curved class boundary in the original two-dimensional input space. The second task therefore uses the architecture

`2-D input → tanh hidden layer → sigmoid output`

with 12 hidden units.

For input matrix $X$, first-layer parameters $W_1,b_1$, second-layer parameters $W_2,b_2$, the forward pass is

$$
z_1 = XW_1 + b_1,
$$

$$
a_1 = \tanh(z_1),
$$

$$
z_2 = a_1W_2 + b_2,
$$

$$
p = \sigma(z_2) = \frac{1}{1+e^{-z_2}}.
$$

Here $p$ is the model's estimated probability for the positive class. The implementation clips the sigmoid input to the interval $[-50,50]$ before evaluating the exponential for numerical stability.

### Binary cross-entropy

The training objective for Task 2 is the mean binary cross-entropy:

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right].
$$

Before evaluating the logarithms, the implementation clips predicted probabilities to $[10^{-12}, 1-10^{-12}]$ to avoid numerical issues.

### Manual backpropagation

Because the output layer uses a sigmoid together with binary cross-entropy, the derivative with respect to the output pre-activation simplifies to

$$
dz_2 = p - y.
$$

The implemented gradients are

$$
dW_2 = \frac{a_1^{\top}dz_2}{n}, \qquad db_2 = \operatorname{mean}(dz_2),
$$

$$
dz_1 = (dz_2W_2^{\top}) \odot (1-a_1^2),
$$

$$
dW_1 = \frac{X^{\top}dz_1}{n}, \qquad db_1 = \operatorname{mean}(dz_1).
$$

The parameters are then updated by gradient descent:

$$
W \leftarrow W - \eta\,dW, \qquad b \leftarrow b - \eta\,db,
$$

where the Task 2 learning rate is $\eta=0.03$.

## Methodology

### End-to-end pipeline

1. Set a reproducible NumPy seed (`42`).
2. Create `datasets/` and `results/` directories when they do not already exist.
3. Load the four required CSV datasets.
4. Validate shapes, finiteness, and label domains.
5. Reconstruct Task 1 three-class labels from the documented coordinate rules.
6. Use the first 70 samples for training and the final 30 samples for testing.
7. Train the Task 1 multiclass perceptron.
8. Compute Task 1 predictions, accuracy, and a 3 × 3 confusion matrix.
9. Standardize Task 2 features using training-set mean and standard deviation only.
10. Train the Task 2 two-layer network for 5,000 epochs with manual backpropagation.
11. Compute Task 2 probabilities, class predictions, accuracy, and a 2 × 2 confusion matrix.
12. Save figures, tables, model parameters, training history, a reproducibility manifest, and a ZIP archive.

### Dataset validation

The implementation expects exactly 100 samples with two numerical features for each task:

| Dataset         | Expected shape | Target representation                                    |              Split |
| --------------- | -------------: | -------------------------------------------------------- | -----------------: |
| `testdataX.csv` |     `(100, 2)` | Reconstructed 3-class labels                             | 70 train / 30 test |
| `testdataY.csv` |       `(100,)` | Supplied binary labels retained for consistency checking |            70 / 30 |
| `moondataX.csv` |     `(100, 2)` | Binary                                                   | 70 train / 30 test |
| `moondataY.csv` |       `(100,)` | Binary labels `0` and `1`                                |            70 / 30 |

The supplied `testdataY.csv` is verified to contain only binary labels. Task 1 nevertheless uses the three-class labels reconstructed from the coordinate rules because that is the representation specified by the prepared assignment code.

### Task 1 target construction

The function `build_three_class_targets` assigns:

| Condition               | Class |
| ----------------------- | ----: |
| $x_2 < 0$               |     1 |
| $x_2 > 0$ and $x_1 < 0$ |     2 |
| $x_2 > 0$ and $x_1 > 0$ |     3 |

Samples lying exactly on either coordinate axis are rejected because the supplied implementation requires an explicit tie rule for such cases.

### Task 2 preprocessing

Task 2 uses training-only standardization:

$$
z = \frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.
$$

The same training statistics are then reused for the held-out test data. If a training feature has zero standard deviation, its denominator is replaced by $1.0$ rather than performing a division by zero.

This separation is important for reproducibility and evaluation integrity: the held-out 30 samples do not influence the fitted preprocessing statistics.

## Mathematical Details

The implementation is a direct numerical realization of the following equations. The notation is chosen to match the array operations in the source code.

### Task 1: Multiclass perceptron

For $K=3$ classes and an input $x \in \mathbb{R}^{d}$, the class scores are

$$
s_k(x) = w_k^{\top}x + b_k.
$$

Prediction uses the maximum score:

$$
\hat{y} = \operatorname*{arg\,max}_{k \in \{1,\ldots,K\}} s_k(x).
$$

If the sample is misclassified, with true class index $t$ and predicted class index $p$, the code performs

$$
w_t \leftarrow w_t + x, \qquad b_t \leftarrow b_t + 1,
$$

$$
w_p \leftarrow w_p - x, \qquad b_p \leftarrow b_p - 1.
$$

The source implements these equations with a unit learning rate and updates only when $p \neq t$.

### Task 2: Two-layer network

With $X$ as the batch of standardized inputs, the hidden representation and output probability are

$$
z_1 = XW_1 + b_1, \qquad a_1 = \tanh(z_1),
$$

$$
z_2 = a_1W_2 + b_2, \qquad p = \sigma(z_2) = \frac{1}{1+e^{-z_2}}.
$$

The objective is mean binary cross-entropy:

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right].
$$

The implemented gradient equations are

$$
dz_2 = p-y,
$$

$$
dW_2 = \frac{a_1^{\top}dz_2}{n}, \qquad db_2 = \operatorname{mean}(dz_2),
$$

$$
dz_1 = (dz_2W_2^{\top}) \odot (1-a_1^2),
$$

$$
dW_1 = \frac{X^{\top}dz_1}{n}, \qquad db_1 = \operatorname{mean}(dz_1).
$$

Each parameter is updated by

$$
\theta \leftarrow \theta - \eta\nabla_{\theta}\mathcal{L},
$$

with $\eta=0.03$ in the supplied training call. These expressions correspond directly to the matrix multiplications and elementwise operations in `train_two_layer_network`.

### Task 2: Training-only standardization

For each feature, the source computes statistics from the 70 training samples and applies them to both subsets:

$$
z = \frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.
$$

A zero training standard deviation is replaced by $1.0$ in the implementation.

## Implementation Details

### Core functions

| Function                      | Role                                                                        |
| ----------------------------- | --------------------------------------------------------------------------- |
| `build_three_class_targets`   | Reconstructs the Task 1 three-class target from the coordinate rules.       |
| `multiclass_perceptron_train` | Initializes and trains the three-class linear perceptron.                   |
| `confusion_matrix`            | Builds a confusion matrix without a machine-learning library.               |
| `plot_linear_boundary`        | Computes and plots each learned Task 1 linear boundary.                     |
| `sigmoid`                     | Computes a numerically stabilized sigmoid activation.                       |
| `binary_cross_entropy`        | Computes mean binary cross-entropy with probability clipping.               |
| `train_two_layer_network`     | Trains the tanh/sigmoid network using explicit forward and backward passes. |
| `predict_two_layer_network`   | Produces probabilities and thresholded class predictions for Task 2.        |

### Initialization and optimization

Task 1 uses a NumPy random generator seeded with `42` and initializes weights and biases from a Gaussian distribution with mean `0` and standard deviation `0.1`.

Task 2 uses the same seed and initializes both trainable weight matrices from a Gaussian distribution with mean `0` and standard deviation `0.5`; hidden and output biases start at zero.

Task 1 trains for at most 1,000 epochs and stops early when an epoch finishes with zero classification errors. Task 2 performs 5,000 full-batch gradient-descent epochs with a learning rate of `0.03` and records loss and training accuracy at epoch 1, every 100 epochs, and the final epoch.

### Dependencies

The source imports the following Python packages:

| Package    | Purpose in the source                                                                                                       |
| ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| NumPy      | Arrays, initialization, linear algebra, activations, gradients, metrics, and training.                                      |
| pandas     | CSV loading and tabular export of results.                                                                                  |
| Matplotlib | Dataset, boundary, loss, and accuracy visualization.                                                                        |
| SciPy      | MATLAB-file loading compatibility via `scipy.io.loadmat`; the execution path documented here reads the converted CSV files. |
| nbformat   | Available as part of the environment setup for notebook-related project compatibility.                                      |

Standard-library modules used include `os`, `pathlib`, `json`, `math`, `importlib`, `subprocess`, `sys`, and `zipfile`.

## Repository Layout

The implementation assumes the following working structure:

```text
.
├── assignment_solution(7).py
├── datasets/
│   ├── testdataX.csv
│   ├── testdataY.csv
│   ├── moondataX.csv
│   └── moondataY.csv
└── results/
    └── generated automatically at runtime
```

If the CSV files are not present and the code is running inside Google Colab, the source provides an upload fallback. Outside Colab, missing datasets raise a `FileNotFoundError` instructing the user to place the four CSV files in `./datasets/`.

## Installation

A standard Python environment with Python 3 is sufficient. The source installs any missing imported packages with `pip` when necessary.

For a clean environment, the equivalent explicit installation command is:

```bash
python -m pip install numpy pandas matplotlib scipy nbformat
```

No scikit-learn, TensorFlow, PyTorch, or other ready-made classifier/neural-network estimator is required by the implementation.

## Usage

### Local execution

From the repository root, run:

```bash
python "assignment_solution(7).py"
```

Before execution, place the four required CSV files in `./datasets/` unless the script is being run in Google Colab, where the built-in upload fallback can be used.

### Google Colab execution

Upload the Python source to a Colab runtime and execute it as Python code. When the four CSV datasets are missing, the source invokes the Colab file-upload interface and writes uploaded files into `datasets/`.

The code creates the `results/` directory automatically and writes the generated artifacts described below.

## Results and Outputs

The source does **not** hard-code numerical accuracies, confusion-matrix values, or training outcomes into the documentation. These values are produced by executing the program on the supplied datasets and are saved as generated artifacts.

The following files are created under `results/`:

| File                        | Content                                                                      |
| --------------------------- | ---------------------------------------------------------------------------- |
| `q1_confusion_matrix.csv`   | Task 1 3 × 3 confusion matrix.                                               |
| `q1_classification.png`     | Task 1 training/test visualization with learned linear boundaries.           |
| `q2_confusion_matrix.csv`   | Task 2 2 × 2 confusion matrix.                                               |
| `q2_model_parameters.npz`   | Learned Task 2 weights, biases, and training standardization statistics.     |
| `q2_training_history.csv`   | Recorded Task 2 epochs, loss, and training accuracy.                         |
| `q2_training_loss.png`      | Task 2 binary cross-entropy history.                                         |
| `q2_training_accuracy.png`  | Task 2 training-accuracy history.                                            |
| `q2_decision_boundary.png`  | Task 2 nonlinear decision surface over the complete moon dataset.            |
| `final_results_summary.csv` | Consolidated training/test sample counts, accuracies, and iteration counts.  |
| `run_manifest.json`         | Seed, hyperparameters, metrics, and generated filenames for reproducibility. |

The complete contents of `results/` are also packed into `results.zip`. In Google Colab, the source attempts a one-click download of this archive.

## Interpretation

### Task 1 — Three-class linear classification

The reconstructed Task 1 target geometry is defined by axis-aligned half-plane and quadrant conditions. The model itself is a standard linear multiclass scorer: each class receives an independent affine score and the largest score determines the prediction. The implementation therefore tests whether the supplied three-class geometry can be represented adequately by a single-layer linear model.

The final 30 samples are not used for parameter updates. They are reserved for the final evaluation, and the resulting predictions are used to construct the confusion matrix and test accuracy.

### Task 2 — Nonlinear moon-shaped classification

The moon-shaped dataset provides a direct contrast with Task 1. The model introduces a hidden nonlinear transformation through `tanh`, allowing the final classifier to represent a curved decision region in the original input space. The output probability is thresholded at `0.5` to obtain binary class predictions.

The code also visualizes the learned probability field and the $p=0.5$ contour, providing a geometric view of the trained classifier rather than relying only on scalar accuracy.

## Limitations

1. **Small sample size.** Each task contains only 100 samples, with 70 used for training and 30 for testing. The resulting test metrics should therefore be interpreted as evaluation on this specific assignment split, not as a statistically robust estimate of population-level generalization.
2. **Fixed split.** The project follows the assignment's deterministic first-70/last-30 split instead of using cross-validation or repeated random splits.
3. **Task 1 label discrepancy in the supplied materials.** The supplied binary target file is not the same representation as the three-class target constructed by the prepared assignment code. The reconstruction documents and preserves this discrepancy rather than silently replacing the source data.
4. **No hyperparameter search.** Task 2 uses a fixed hidden width of 12, learning rate of `0.03`, and 5,000 epochs. These values are implementation parameters, not the result of a reported model-selection study.
5. **Full-batch optimization.** Task 2 uses all 70 training samples in every gradient step. No mini-batching, momentum, adaptive optimizer, weight decay, or other regularization is implemented.
6. **Limited diagnostics.** The project records training loss and training accuracy for Task 2, but it does not maintain a validation set or perform systematic calibration, uncertainty estimation, or statistical confidence analysis.

## Academic Relevance

Although the datasets are small and the project is instructional in scope, the implementation demonstrates several research-relevant practices: explicit mathematical modeling, controlled preprocessing, reproducible initialization, transparent optimization, manual gradient derivation, separation of training and held-out evaluation, and artifact-level experiment logging.

The project provides a compact demonstration of the progression from a linear classifier to a nonlinear neural model. It exposes the computational steps that are normally hidden behind high-level machine-learning libraries and makes the connection between the mathematical formulation and executable NumPy operations explicit.

## Implementation Audit Notes

The reconstruction makes the following scientifically material corrections relative to the issues identified in the source audit:

| Issue in the supplied implementation path                               | Reconstruction in this project                                                                              |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Task 1 mixed binary labels with a three-class geometric definition      | Three classes are reconstructed directly from the documented coordinate rules.                              |
| Task 1 independently trained binary-style units and averaged parameters | A multiclass perceptron maintains one score vector per class and applies a correct winner-takes-all update. |
| Task 1 evaluation mixed incompatible label representations              | A dedicated 3 × 3 confusion matrix is computed from the reconstructed three-class target.                   |
| Task 2 treated threshold units as independent linear classifiers        | A two-layer nonlinear network is trained with an explicit hidden representation.                            |
| Task 2 evaluation replaced predictions with random labels               | Predictions are generated deterministically from the trained network.                                       |
| Random initialization was not explicitly controlled                     | NumPy initialization is seeded with `42` and recorded in the run manifest.                                  |

This audit trail is intentionally kept visible because reproducibility depends not only on the final code, but also on understanding which design choices came from the supplied assignment and which were introduced to make the learning procedure mathematically coherent.

## License and Dataset Note

No project license or dataset redistribution license is declared in the supplied source. Before publishing the repository publicly, add the appropriate license and verify that the four dataset files may legally be redistributed with the code.

