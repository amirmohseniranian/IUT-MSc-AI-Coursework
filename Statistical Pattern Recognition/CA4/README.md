# Statistical Pattern Recognition — Computer Assignment #4

## Audited MLP with Online Backpropagation

This repository contains a reproducible, from-scratch implementation of a small multilayer perceptron (MLP) for a two-class statistical pattern-recognition problem. It builds the network explicitly, performs forward propagation, applies online backpropagation sample by sample, tracks training mean-squared error, evaluates the trained classifier on a held-out test set, and exports both machine-readable and visual artifacts.

The implementation follows the assignment specification encoded in the source file:

**2 input features + bias → 4 sigmoid hidden units + bias → 2 sigmoid output units**

Training uses the supplied initial weights, the supplied sample order without shuffling, a learning rate of `0.2`, and exactly `500` epochs.

> **Scope and scientific honesty.** This repository documents and reproduces the supplied assignment implementation. The dataset is an instructor-provided workbook embedded directly in the source as a Base64-encoded payload; it is not identified in the source as a public benchmark dataset. Therefore, no external dataset name, benchmark comparison, or novelty claim is assigned to the problem.

---

## Overview

The project covers the complete computational pipeline for a small feed-forward neural classifier:

1. Resolve the assignment workbook from an optional URL, a local file, or the self-contained embedded copy.
2. Load the training set, test set, and initial weight matrices.
3. Validate the workbook structure and assignment-specific assumptions.
4. Perform an explicit numerical forward pass using sigmoid activation functions.
5. Train with online backpropagation, updating weights after every training sample.
6. Record the full training-error trajectory for all `500` epochs.
7. Classify test samples using the larger of the two output activations.
8. Compute accuracy, misclassification count, and a 2 × 2 confusion matrix.
9. Export trained weights, predictions, error history, performance summaries, and figures.
10. Package the generated artifacts into a ZIP archive.

Model training does not rely on a neural-network framework such as PyTorch, TensorFlow, Keras, or scikit-learn. The forward pass and learning rule are implemented directly with NumPy.

---

## Problem Definition

The assignment is a supervised binary classification problem with two scalar input features, $x_1$ and $x_2$, and two output nodes representing classes 1 and 2.

The workbook contains:

* **Training set:** 1,000 samples
* **Testing set:** 1,000 samples
* **Classes:** 1 and 2
* **Training input features:** `x1`, `x2`
* **Training targets:** `Target 1`, `Target 2`
* **Feature range enforced by the implementation:** $[0.2, 0.8]$
* **Target vectors:** $(0.95, 0.05)$ for class 1 and $(0.05, 0.95)$ for class 2
* **Training order:** alternating class 1 / class 2
* **Initial input-to-hidden weights:** 3 × 4
* **Initial hidden-to-output weights:** 5 × 2

The target encoding is intentionally not one-hot with exact values 0 and 1. Instead, the assignment uses the two soft target vectors

$$
\mathbf{t}^{(1)} = (0.95, 0.05)
$$

and

$$
\mathbf{t}^{(2)} = (0.05, 0.95).
$$

This target encoding matches the sigmoid output layer and the values stored in the workbook.

---

## Scientific Background

### Multilayer Perceptron

The model is a fully connected feed-forward neural network with one hidden layer. Bias terms are represented explicitly as additional constant input nodes equal to `+1`.

For an input sample,

$$
\mathbf{x} =
\begin{bmatrix}
x_1 & x_2 & 1
\end{bmatrix},
$$

where the final component is the input-layer bias.

The input-to-hidden weight matrix is

$$
W \in \mathbb{R}^{3 \times 4},
$$

and the hidden-to-output matrix is

$$
V \in \mathbb{R}^{5 \times 2}.
$$

All four hidden units and both output units use sigmoid activations.

### Sigmoid Activation

The source implements the logistic sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

Its derivative is

$$
\sigma'(z) = \sigma(z)\left(1-\sigma(z)\right).
$$

The implementation uses an overflow-safe numerical form of the sigmoid rather than evaluating $e^{-z}$ blindly for every value.

---

## Network Architecture

The network can be summarized as follows:

```text
                 Input layer
        ┌─────────────────────────┐
        │ x1                      │
        │ x2                      │
        │ +1 bias                 │
        └─────────────┬───────────┘
                      │
                3 × 4 weights
                      │
                      ▼
        ┌─────────────────────────┐
        │ Hidden 1 — sigmoid      │
        │ Hidden 2 — sigmoid      │
        │ Hidden 3 — sigmoid      │
        │ Hidden 4 — sigmoid      │
        │ +1 bias                 │
        └─────────────┬───────────┘
                      │
                5 × 2 weights
                      │
                      ▼
        ┌─────────────────────────┐
        │ Output 1 — sigmoid      │
        │ Output 2 — sigmoid      │
        └─────────────────────────┘
```

Thus, the dimensional flow is

$$
3 \rightarrow 4 \rightarrow 2,
$$

where the `3` consists of two real input features plus one bias component, and the `5` entering the final layer consists of four hidden activations plus one hidden-layer bias component.

### Parameter Count

The total number of trainable scalar weights is

$$
(3\times4) + (5\times2) = 12 + 10 = 22.
$$

The exported `weight_matrices.csv` therefore contains **22 learned weights**.

---

## Mathematical Formulation

### Forward Propagation

For one sample, append the input bias:

$$
\tilde{\mathbf{x}} =
\begin{bmatrix}
x_1 & x_2 & 1
\end{bmatrix}.
$$

The hidden pre-activations are

$$
\mathbf{a}^{(h)} = \tilde{\mathbf{x}}W.
$$

Applying the sigmoid element-wise gives

$$
\mathbf{h} = \sigma\left(\mathbf{a}^{(h)}\right).
$$

A second bias component is then appended:

$$
\tilde{\mathbf{h}} =
\begin{bmatrix}
h_1 & h_2 & h_3 & h_4 & 1
\end{bmatrix}.
$$

The output pre-activations are

$$
\mathbf{a}^{(o)} = \tilde{\mathbf{h}}V,
$$

and the two output activations are

$$
\mathbf{y} = \sigma\left(\mathbf{a}^{(o)}\right).
$$

The source implements these operations directly with NumPy matrix products.

### Online Backpropagation

For a training target $\mathbf{t}$, the output error is

$$
\mathbf{e}^{(o)} = \mathbf{t} - \mathbf{y}.
$$

The output-layer delta is

$$
\boldsymbol{\delta}^{(o)} =
\mathbf{e}^{(o)}
\odot
\mathbf{y}
\odot
(1-\mathbf{y}),
$$

where $\odot$ denotes element-wise multiplication.

For each hidden unit, the error signal is propagated backward through the current hidden-to-output weights:

$$
\mathbf{e}^{(h)} =
\boldsymbol{\delta}^{(o)}
\left(V_{\text{non-bias}}\right)^{\mathsf T}.
$$

The hidden-layer delta is then

$$
\boldsymbol{\delta}^{(h)} =
\mathbf{e}^{(h)}
\odot
\mathbf{h}
\odot
(1-\mathbf{h}).
$$

### Weight Updates

The implementation uses an **online** update: weights are changed immediately after each training sample.

For the hidden-to-output weights,

$$
V_{\text{non-bias}}
\leftarrow
V_{\text{non-bias}}
+
\eta\,
\mathbf{h}^{\mathsf T}
\boldsymbol{\delta}^{(o)},
$$

and the output bias weights are updated by

$$
\mathbf{v}_{\text{bias}}^{(o)}
\leftarrow
\mathbf{v}_{\text{bias}}^{(o)}
+
\eta\,\boldsymbol{\delta}^{(o)}.
$$

For the input-to-hidden weights,

$$
W_{\text{non-bias}}
\leftarrow
W_{\text{non-bias}}
+
\eta\,
\mathbf{x}^{\mathsf T}
\boldsymbol{\delta}^{(h)},
$$

with the hidden-layer bias update

$$
\mathbf{w}_{\text{bias}}^{(h)}
\leftarrow
\mathbf{w}_{\text{bias}}^{(h)}
+
\eta\,\boldsymbol{\delta}^{(h)}.
$$

Here,

$$
\eta = 0.2.
$$

The order of these operations matters. The hidden-layer error is computed with the hidden-to-output weights **before** those weights are updated for the current sample. This is the sample-wise backpropagation order used by the source.

### Training Objective Tracked by the Implementation

After each epoch, the code evaluates all training samples with the updated weights and stores

$$
\operatorname{MSE} =
\frac{1}{2N}
\sum_{n=1}^{N}
\sum_{k=1}^{2}
\left(t_{nk}-y_{nk}\right)^2,
$$

where $N=1000$ and the factor $2$ accounts for the two scalar output coordinates per sample.

The source computes this as the mean of the squared differences over the complete `1000 × 2` target/output array.

### Classification Rule

For each test sample, the two sigmoid outputs are compared directly. The predicted class is

$$
\hat{c} = 1 + \operatorname*{arg\,max}_{k\in\{0,1\}} y_k.
$$

Therefore:

* the larger output node determines the predicted class;
* there is no additional probability calibration step;
* there is no manually selected decision threshold beyond the two-way `argmax`.

---

## Data Handling and Validation

The workbook is the authoritative assignment data source. To keep the project self-contained, the exact workbook bytes are embedded in the Python source as a Base64-encoded payload.

The loading function supports three resolution paths:

1. **Optional remote source:** if `ASSIGNMENT_DATA_URL` is defined, the script attempts to download the workbook from that URL.
2. **Existing local workbook:** if the expected workbook already exists, it is used directly.
3. **Embedded fallback:** if neither source is available, the embedded workbook is reconstructed locally.

The loader requires these worksheet names:

```text
Training Set
Testing Set
Initial Weights
```

The validation stage then checks the assignment-specific conditions:

| Validation              | Expected condition                            |
| ----------------------- | --------------------------------------------- |
| Training shape          | 1,000 × 5                                     |
| Testing shape           | 1,000 × 3                                     |
| Training columns        | `Class #`, `x1`, `x2`, `Target 1`, `Target 2` |
| Testing columns         | `Class #`, `x1`, `x2`                         |
| Training classes        | Exactly `{1, 2}`                              |
| Testing classes         | Exactly `{1, 2}`                              |
| Training order          | Alternating class 1 / class 2                 |
| Feature range           | $0.2 \leq x_i \leq 0.8$                       |
| Target vectors          | $(0.95,0.05)$ and $(0.05,0.95)$               |
| Input-to-hidden matrix  | 3 × 4                                         |
| Hidden-to-output matrix | 5 × 2                                         |

These checks are intentionally strict. An incorrectly formatted or unrelated workbook should fail rather than silently changing the experiment.

---

## Training Procedure

The training loop follows the supplied sample order and is deterministic.

### Configuration

| Parameter           |                                Value |
| ------------------- | -----------------------------------: |
| Input features      |                                    2 |
| Input bias          |                                    1 |
| Hidden units        |                                    4 |
| Hidden activation   |                              Sigmoid |
| Hidden bias         |                                    1 |
| Output units        |                                    2 |
| Output activation   |                              Sigmoid |
| Learning rate       |                                  0.2 |
| Epochs              |                                  500 |
| Update rule         | Online / sample-wise backpropagation |
| Shuffling           |                             Disabled |
| Regularization      |                                 None |
| Validation split    |                                 None |
| Optimizer framework | None; updates implemented explicitly |

Because the code neither shuffles samples nor draws random numbers, there is no training seed to configure. With the same workbook and software stack, the training trajectory is deterministic.

---

## Implementation Details

The implementation favors transparency over a high-level framework.

### Core components

| Function                   | Role                                               |
| -------------------------- | -------------------------------------------------- |
| `_embedded_dataset_bytes`  | Decode the bundled workbook payload                |
| `prepare_dataset`          | Resolve URL, local workbook, or embedded fallback  |
| `load_assignment_data`     | Load data and initial weights from Excel           |
| `validate_assignment_data` | Enforce assignment-specific structural assumptions |
| `sigmoid`                  | Numerically stable sigmoid activation              |
| `forward_pass`             | Forward computation for one sample                 |
| `forward_dataset`          | Vectorized inference over a complete dataset       |
| `train_mlp`                | Online backpropagation for exactly 500 epochs      |
| `predict`                  | Test-time class prediction and raw outputs         |
| `save_trained_weights`     | Export weights in an assignment-like layout        |
| `save_weight_matrices`     | Export the 22 learned weights in tidy tabular form |
| `create_package`           | Collect artifacts into a ZIP archive               |

### Numerical stability

The sigmoid function uses separate branches for non-negative and negative inputs. This avoids directly forming very large values of $e^{-x}$ when $x$ is strongly negative.

This is more numerically stable than evaluating the naive expression

$$
\frac{1}{1+e^{-x}}
$$

evaluated without range protection.

### Training / inference separation

The code uses two distinct execution paths:

* `forward_pass(...)` processes a single sample and is used inside training.
* `forward_dataset(...)` evaluates complete arrays without changing the weights.

This separation keeps the online training rule explicit while providing a vectorized path for evaluation and epoch-level MSE calculation.

---

## Results

Running the supplied source with its bundled assignment workbook produced the following results.

### Final Performance

| Metric                |           Result |
| --------------------- | ---------------: |
| Test samples          |            1,000 |
| True Class 1 samples  |              500 |
| True Class 2 samples  |              500 |
| Correct predictions   |              932 |
| Misclassified samples |               68 |
| Test accuracy         |       **93.20%** |
| Final training MSE    | **0.0452958638** |

### Confusion Matrix

Rows correspond to the actual class and columns to the predicted class.

|                    | Predicted Class 1 | Predicted Class 2 |
| ------------------ | ----------------: | ----------------: |
| **Actual Class 1** |               465 |                35 |
| **Actual Class 2** |                33 |               467 |

The matrix gives the following error counts:

* **Class 1:** 465 correct, 35 errors
* **Class 2:** 467 correct, 33 errors
* **Total:** 932 correct out of 1,000

The class-wise error counts are close: 35 errors for class 1 and 33 for class 2.

### Training Error

The first eight epoch-level MSE values generated by the implementation are:

| Epoch |      MSE |
| ----: | -------: |
|     1 | 0.200715 |
|     2 | 0.195029 |
|     3 | 0.177652 |
|     4 | 0.139594 |
|     5 | 0.099729 |
|     6 | 0.077303 |
|     7 | 0.066335 |
|     8 | 0.060503 |

At the end of training, the MSE is approximately `0.045296`.

The final ten recorded values show continued but very small improvement:

| Epoch |      MSE |
| ----: | -------: |
|   491 | 0.045303 |
|   492 | 0.045302 |
|   493 | 0.045301 |
|   494 | 0.045300 |
|   495 | 0.045300 |
|   496 | 0.045299 |
|   497 | 0.045298 |
|   498 | 0.045297 |
|   499 | 0.045297 |
|   500 | 0.045296 |

Under the supplied training protocol, the model is close to a stable solution by the later epochs.

### Generated Visualizations

After execution, the program produces two diagnostic figures.

#### Confusion Matrix

![Confusion matrix](assignment_4_results/confusion_matrix.png)

#### Training Error Curve

![Training error across 500 epochs](assignment_4_results/training_error_curve.png)

The Python source generates these images directly; they are not manually edited.

---

## Misclassified Samples

The program exports every test sample for which the predicted class differs from the ground-truth class.

The exported file contains:

* test sample number
* true class
* predicted class
* `x1`
* `x2`
* output node 1 activation
* output node 2 activation

For example, the first misclassified test sample is:

| Field           |    Value |
| --------------- | -------: |
| Sample          |       11 |
| True class      |        1 |
| Predicted class |        2 |
| `x1`            | 0.514235 |
| `x2`            | 0.511510 |
| Output 1        | 0.495861 |
| Output 2        | 0.504042 |

This sample lies close to the decision boundary because the two output activations are nearly equal.

The complete set of 68 errors is available in `assignment_4_results/misclassified_samples.csv`.

---

## Learned Weights

The trained network contains 22 scalar parameters.

### Input → Hidden

| Source  | Target   |     Weight |
| ------- | -------- | ---------: |
| Input 1 | Hidden 1 |  10.312665 |
| Input 1 | Hidden 2 | -22.983529 |
| Input 1 | Hidden 3 |  -1.690799 |
| Input 1 | Hidden 4 |  -6.082619 |
| Input 2 | Hidden 1 |  -0.688900 |
| Input 2 | Hidden 2 |   1.472401 |
| Input 2 | Hidden 3 |  -0.673926 |
| Input 2 | Hidden 4 |  -0.906648 |
| Bias    | Hidden 1 |  -5.163270 |
| Bias    | Hidden 2 |  10.876937 |
| Bias    | Hidden 3 |  -1.138529 |
| Bias    | Hidden 4 |   0.889838 |

### Hidden → Output

| Source   | Target   |    Weight |
| -------- | -------- | --------: |
| Hidden 1 | Output 1 | -1.952405 |
| Hidden 1 | Output 2 |  2.066747 |
| Hidden 2 | Output 1 |  5.403473 |
| Hidden 2 | Output 2 | -5.329131 |
| Hidden 3 | Output 1 |  0.549356 |
| Hidden 3 | Output 2 | -0.615908 |
| Hidden 4 | Output 1 |  2.466650 |
| Hidden 4 | Output 2 | -2.606184 |
| Bias     | Output 1 | -1.793857 |
| Bias     | Output 2 |  1.723230 |

The values above are the learned parameters produced by the execution described in this README; they were not re-estimated or hand-selected for documentation.

---

## Output Artifacts

Running the script creates an `assignment_4_results/` directory containing the generated artifacts.

```text
assignment_4_results/
├── normal_assignment_4_dataset.xlsx
├── trained_weights.csv
├── weight_matrices.csv
├── confusion_matrix.csv
├── confusion_matrix.png
├── misclassified_samples.csv
├── model_performance_results.csv
├── training_error_history.csv
├── test_predictions.csv
├── training_error_curve.png
└── assignment_4_results.zip
```

### Artifact descriptions

| File                               | Purpose                                                     |
| ---------------------------------- | ----------------------------------------------------------- |
| `normal_assignment_4_dataset.xlsx` | Reconstructed self-contained assignment workbook            |
| `trained_weights.csv`              | Trained parameters in an assignment-style layout            |
| `weight_matrices.csv`              | All 22 learned weights in tidy long format                  |
| `confusion_matrix.csv`             | Numeric 2 × 2 confusion matrix                              |
| `confusion_matrix.png`             | Rendered confusion-matrix figure                            |
| `misclassified_samples.csv`        | Full list of 68 misclassified test samples                  |
| `model_performance_results.csv`    | Summary metrics                                             |
| `training_error_history.csv`       | MSE for each of the 500 epochs                              |
| `test_predictions.csv`             | Predictions and raw output activations for all test samples |
| `training_error_curve.png`         | Training MSE diagnostic plot                                |
| `assignment_4_results.zip`         | Compressed collection of generated artifacts                |

---

## Project Structure

A typical repository layout is:

```text
.
├── amirmohsen_sharifi_hw4_mlp_bp(1).py
├── README.md
└── assignment_4_results/
    ├── normal_assignment_4_dataset.xlsx
    ├── trained_weights.csv
    ├── weight_matrices.csv
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    ├── misclassified_samples.csv
    ├── model_performance_results.csv
    ├── training_error_history.csv
    ├── test_predictions.csv
    ├── training_error_curve.png
    └── assignment_4_results.zip
```

The source is a Python export of a Colab-oriented notebook and contains the experiment logic in executable form.

---

## Installation

The project requires Python 3 with the following packages:

* NumPy
* pandas
* Matplotlib
* openpyxl

The source also checks for missing dependencies and installs them through `pip` when needed.

For a clean environment, it is preferable to install the dependencies explicitly:

```bash
python -m pip install numpy pandas matplotlib openpyxl
```

A virtual environment is recommended for local reproducible execution:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the required packages:

```bash
python -m pip install numpy pandas matplotlib openpyxl
```

---

## Usage

From the repository root:

```bash
python "amirmohsen_sharifi_hw4_mlp_bp(1).py"
```

The script then:

1. prepare the assignment workbook;
2. validate the data and initial weights;
3. execute a diagnostic forward pass;
4. train the MLP for 500 epochs;
5. evaluate the 1,000 test samples;
6. write CSV outputs and figures;
7. create `assignment_4_results.zip`.

### Optional dataset URL override

The script recognizes the environment variable `ASSIGNMENT_DATA_URL`.

Linux/macOS:

```bash
export ASSIGNMENT_DATA_URL="https://example.invalid/path/to/workbook.xlsx"
python "amirmohsen_sharifi_hw4_mlp_bp(1).py"
```

PowerShell:

```powershell
$env:ASSIGNMENT_DATA_URL="https://example.invalid/path/to/workbook.xlsx"
python "amirmohsen_sharifi_hw4_mlp_bp(1).py"
```

The URL is optional. If retrieval fails, the script falls back to the embedded assignment workbook.

---

## Reproducibility

The implementation is deterministic under a fixed software environment and fixed assignment workbook.

### Deterministic factors

* No random number generation is used for training.
* The supplied initial weights are fixed.
* The training order is preserved exactly.
* Samples are not shuffled.
* The learning rate is fixed at `0.2`.
* The number of epochs is fixed at `500`.
* The update equations are explicitly implemented with NumPy.
* The assignment workbook is embedded in the source.

### Reproducibility procedure

A reproducible run consists of:

```text
1. Start from a clean Python environment.
2. Install the four required packages.
3. Run the provided source file.
4. Preserve the generated assignment_4_results/ directory.
5. Compare the generated CSV metrics and confusion matrix with the values documented here.
```

Because floating-point arithmetic can vary slightly across software stacks, the most useful reproducibility criterion is agreement of the generated artifacts and reported metrics to an appropriate numerical precision, rather than bit-for-bit identity of every serialized floating-point value.

---

## Computational Complexity

Let:

* $N$ be the number of training samples,
* $E$ be the number of epochs,
* $H=4$ be the number of hidden units,
* $I=2$ be the number of input features,
* $O=2$ be the number of output units.

For one training sample, the forward and backward passes involve constant-size matrix/vector operations. More generally, the work per sample scales approximately with

$$
O(IH + HO + H + O).
$$

Across the complete training process, the dominant cost scales as

$$
O\!\left(E N (IH + HO)\right).
$$

For this assignment, the network is small, so the computation is lightweight on ordinary CPU hardware.

Memory usage is also modest. Aside from the dataset and stored training history, the model itself contains only 22 trainable scalar weights.

---

## Evaluation Methodology

The implementation evaluates the trained model on the separate `Testing Set` worksheet. Test labels are not used during training.

The main reported metric is classification accuracy:

$$
\operatorname{Accuracy} =
\frac{\text{number of correct predictions}}
{\text{number of test samples}}.
$$

For the reported run,

$$
\operatorname{Accuracy} =
\frac{932}{1000}
= 0.932 = 93.20\%.
$$

The confusion matrix provides a more detailed view of the errors for the two classes.

The training MSE is a diagnostic of the regression-style target fit at the two sigmoid output nodes; it is not itself the classification accuracy metric.

---

## Interpretation of the Results

The run reaches **93.20% test accuracy** and ends with a training MSE of approximately **0.045296**.

The confusion matrix is relatively symmetric:

* class 1 has 35 misclassifications;
* class 2 has 33 misclassifications.

The late-epoch MSE values change only in the fifth or sixth decimal place, so the training trajectory has largely flattened by epoch 500 under the specified learning rate and update rule.

These observations apply only to the supplied experiment. They should not be generalized to MLPs, sigmoid networks, or other datasets without additional controlled experiments.

---

