# Persian Handwritten Digit Recognition with a Fully Connected Neural Network

## Overview

This project implements a reproducible deep learning pipeline for recognizing handwritten Persian digits from the **HODA** dataset using a fully connected neural network (FNN). The implementation is contained in `assignment_solution.py` and is organized as a complete Google Colab workflow.

The pipeline:

1. Loads the required HODA training and test samples from CSV files.
2. Reconstructs the original variable-sized grayscale images.
3. Resizes every image to $20 \times 20$ pixels using anti-aliasing.
4. Scales pixel intensities to $[0,1]$ and flattens each image into a 400-dimensional feature vector.
5. Creates a reproducible, stratified 85/15 validation split for optimizer selection.
6. Standardizes features using statistics computed only from the fitting partition.
7. Compares Adam, RMSprop, and Nesterov momentum SGD at two learning rates each.
8. Selects the configuration with the highest observed validation accuracy.
9. Trains a final $400 \rightarrow 256 \rightarrow 128 \rightarrow 10$ fully connected network with early stopping and learning-rate reduction.
10. Evaluates the trained model on the untouched 2,000-image test set.
11. Produces predictions, misclassification visualizations, a confusion matrix, a classification report, and the probability vector for the first test image.
12. Exports the trained model and all generated artifacts to `results/`.

The implementation intentionally **does not hard-code experimental results**. Accuracy, loss, misclassification counts, and optimizer-search outcomes are computed when the pipeline is executed with the dataset.

> **Source scope.** The uploaded source reconstructs the supplied assignment as a Colab-based Python solution. The source notes that the original archive contained `CA2.pdf`, `Data_hoda_full.mat`, and `HW2_starter.ipynb`, with no MATLAB `.m` files available. Accordingly, the implementation treats the assignment requirements and starter implementation as its project specification.

## Motivation

Handwritten digit recognition is a compact benchmark for studying the interaction between image preprocessing, representation learning, optimization, and statistical evaluation. In this project, the model is deliberately restricted to fully connected layers rather than convolutional layers, making the effect of the input representation and dense architecture easier to inspect.

The task is ten-class classification. Each input image is mapped to one digit label in ${0,1,\ldots,9}$.

The central experimental questions are:

* How effectively can a compact fully connected architecture classify resized HODA images?
* How does optimization change with different optimizer and learning-rate choices?
* Which classes are most difficult for the resulting model?
* What information is contained in the model's predicted probability distribution for an individual test image?

## Scientific Background

### HODA image representation

The dataset contains variable-sized grayscale images. The implementation reconstructs each image from the CSV representation before resizing.

For an original image $I$ with height $H$ and width $W$, the CSV representation stores a row-major pixel vector with exactly $H W$ values. The implementation verifies this contract before reshaping the vector back to its original geometry.

Each reconstructed image is then resized to a fixed spatial resolution:

$$
I_{\mathrm{20\times20}} = \operatorname{Resize}(I, 20,20).
$$

The resize operation uses anti-aliasing and preserves the original numerical intensity range before the explicit scaling step.

Pixel intensities are subsequently normalized to the interval $[0,1]$:

$$
x' = \frac{x}{255}.
$$

After resizing, the $20 \times 20$ image is flattened:

$$
\mathbf{x} \in \mathbb{R}^{400}.
$$

This flattened representation is the input to the dense network.

### Train-only feature standardization

For final modeling, the source computes a mean and standard deviation for each of the 400 input features using only the fitting partition:

$$
\mu_j = \frac{1}{N_{\mathrm{fit}}}
\sum_{i=1}^{N_{\mathrm{fit}}} x_{ij},
$$

$$
\sigma_j =
\sqrt{
\frac{1}{N_{\mathrm{fit}}}
\sum_{i=1}^{N_{\mathrm{fit}}}
(x_{ij}-\mu_j)^2
}.
$$

A feature whose standard deviation is smaller than $10^{-6}$ is assigned a standard deviation of $1$ to avoid unstable division.

Standardization is then applied to fitting, validation, full-training, and test matrices with the same training-derived statistics:

$$
z_{ij} = \frac{x_{ij}-\mu_j}{\sigma_j}.
$$

This separation prevents test observations from influencing the feature-scaling parameters during preprocessing.

## Methodology

### Dataset contract

The implementation expects the following two files:

* `datasets/hoda_train.csv`
* `datasets/hoda_test.csv`

Each file must contain the following columns:

| Column      | Meaning                                      |
| ----------- | -------------------------------------------- |
| `sample_id` | Sample identifier                            |
| `split`     | Dataset split identifier (`train` or `test`) |
| `label`     | Digit class in ${0,\ldots,9}$                |
| `height`    | Original image height                        |
| `width`     | Original image width                         |
| `pixels`    | Space-separated row-major pixel vector       |

The source validates the dataset contract before training. It requires exactly:

* 10,000 training rows;
* 2,000 test rows;
* labels between 0 and 9;
* positive image dimensions;
* non-missing pixel vectors.

### Reproducible validation split

The 10,000 training examples are partitioned with a stratified 85/15 split:

$$
\mathcal{D}_{\mathrm{train}} =
\mathcal{D}_{\mathrm{fit}}
\cup
\mathcal{D}_{\mathrm{val}},
$$

with

$$
\frac{|\mathcal{D}_{\mathrm{val}}|}{|\mathcal{D}_{\mathrm{train}}|}
=0.15.
$$

The split uses `random_state=42` and preserves class proportions through `stratify=y_train`.

The optimizer search uses only the fitting and validation partitions. The test set remains untouched until final evaluation.

### Network architecture

The main model is a fully connected classifier:

| Stage          | Operation               | Output |
| -------------- | ----------------------- | -----: |
| Input          | Flattened resized image |    400 |
| Hidden layer 1 | Dense                   |    256 |
| Normalization  | Batch Normalization     |    256 |
| Activation     | ReLU                    |    256 |
| Regularization | Dropout, rate 0.25      |    256 |
| Hidden layer 2 | Dense                   |    128 |
| Normalization  | Batch Normalization     |    128 |
| Activation     | ReLU                    |    128 |
| Regularization | Dropout, rate 0.20      |    128 |
| Output         | Dense + Softmax         |     10 |

The first dense transformation is

$$
\mathbf{h}_1
=
\operatorname{ReLU}
\left(
\operatorname{BN}
\left(
W_1\mathbf{x}+\mathbf{b}_1
\right)
\right).
$$

The second hidden representation is

$$
\mathbf{h}_2
=
\operatorname{ReLU}
\left(
\operatorname{BN}
\left(
W_2\mathbf{h}_1+\mathbf{b}_2
\right)
\right).
$$

Dropout is applied after each hidden activation during training. The output logits are mapped to class probabilities with softmax:

$$
P(y=k\mid\mathbf{x})
=
\frac{\exp(z_k)}
{\sum_{j=0}^{9}\exp(z_j)},
\qquad k\in\{0,\ldots,9\}.
$$

The predicted class is the maximum-probability class:

$$
\hat{y}
=
\operatorname*{arg\,max}_{k\in\{0,\ldots,9\}}
P(y=k\mid\mathbf{x}).
$$

The network contains **137,610 trainable parameters** in total, excluding the non-trainable moving statistics maintained by Batch Normalization.

### Optimization objective

The model is compiled with sparse categorical cross-entropy loss and accuracy.

For a single example with true class $y$ and predicted probability $p_y$, the loss is

$$
\mathcal{L}
=
-\log p_y.
$$

For a batch of $B$ observations, the implementation optimizes the mean loss:

$$
\mathcal{L}_{\mathrm{batch}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log p_{i,y_i}.
$$

### Optimizer comparison

The implementation evaluates six controlled configurations:

| Optimizer                  |    Learning rate |
| -------------------------- | ---------------: |
| Adam                       |        $10^{-3}$ |
| Adam                       | $3\times10^{-4}$ |
| RMSprop                    |        $10^{-3}$ |
| RMSprop                    | $3\times10^{-4}$ |
| SGD with Nesterov momentum |        $10^{-2}$ |
| SGD with Nesterov momentum |        $10^{-3}$ |

For every search run, the architecture, random seed, batch size, training duration, fitting partition, and validation partition are held constant.

The SGD implementation uses momentum $0.9$ and Nesterov acceleration.

Each candidate is trained for up to 12 epochs. Candidates are ranked primarily by maximum validation accuracy, with minimum validation loss as the secondary criterion.

### Final training protocol

The best optimizer and learning rate from the search are selected automatically.

The final model is re-created from scratch and trained with:

| Hyperparameter          | Value      |
| ----------------------- | ---------- |
| Input dimension         | 400        |
| Hidden units            | 256, 128   |
| Output classes          | 10         |
| Batch size              | 256        |
| Maximum epochs          | 25         |
| Dropout                 | 0.25, 0.20 |
| Seed                    | 42         |
| Early-stopping patience | 5 epochs   |
| LR-reduction patience   | 2 epochs   |
| LR-reduction factor     | 0.5        |
| Minimum learning rate   | $10^{-5}$  |

The final `fit` call uses all 10,000 training examples and reserves 15% as Keras's internal validation subset for callback monitoring:

```python
final_model.fit(
    X_train_std,
    y_train,
    validation_split=0.15,
    epochs=FINAL_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)
```

Early stopping monitors validation loss, waits five epochs without improvement, and restores the best weights. `ReduceLROnPlateau` halves the learning rate after two epochs without improvement, subject to a minimum of $10^{-5}$.

## Mathematical Formulation

Let the standardized input vector be $\mathbf{x}\in\mathbb{R}^{400}$.

The first hidden layer has parameters

$$
W_1\in\mathbb{R}^{256\times400},
\qquad
\mathbf{b}_1\in\mathbb{R}^{256}.
$$

After the first affine transformation and Batch Normalization, ReLU produces

$$
\mathbf{h}_1
=
\operatorname{ReLU}
\left(
\operatorname{BN}
\left(
W_1\mathbf{x}+\mathbf{b}_1
\right)
\right).
$$

The second hidden layer uses

$$
W_2\in\mathbb{R}^{128\times256},
\qquad
\mathbf{b}_2\in\mathbb{R}^{128},
$$

giving

$$
\mathbf{h}_2
=
\operatorname{ReLU}
\left(
\operatorname{BN}
\left(
W_2\mathbf{h}_1+\mathbf{b}_2
\right)
\right).
$$

The output layer uses

$$
W_3\in\mathbb{R}^{10\times128},
\qquad
\mathbf{b}_3\in\mathbb{R}^{10},
$$

with logits

$$
\mathbf{z}
=
W_3\mathbf{h}_2+\mathbf{b}_3.
$$

The class-probability vector is

$$
\mathbf{p}
=
\operatorname{softmax}(\mathbf{z}),
$$

where

$$
p_k
=
\frac{\exp(z_k)}
{\sum_{j=0}^{9}\exp(z_j)}.
$$

The classifier output is

$$
\hat{y}
=
\operatorname*{arg\,max}_k p_k.
$$

For the observed label $y$, sparse categorical cross-entropy is

$$
\mathcal{L}(\mathbf{x},y)
=
-\log p_y.
$$

During training, dropout randomly masks hidden activations according to the configured dropout rates. During inference, Keras disables dropout and uses the learned Batch Normalization inference statistics.

## Hyperparameters and Experimental Controls

The main experimental controls in the source are:

| Category                   | Setting                                |
| -------------------------- | -------------------------------------- |
| Random seed                | 42                                     |
| Training samples           | 10,000                                 |
| Test samples               | 2,000                                  |
| Input resolution           | $20\times20$                           |
| Flattened input size       | 400                                    |
| Number of classes          | 10                                     |
| Batch size                 | 256                                    |
| Search epochs              | 12                                     |
| Final maximum epochs       | 25                                     |
| Optimizers                 | Adam, RMSprop, SGD + Nesterov          |
| Search learning rates      | $10^{-3}$, $3\times10^{-4}$, $10^{-2}$ |
| First hidden layer         | 256 units                              |
| Second hidden layer        | 128 units                              |
| Dropout rates              | 0.25 and 0.20                          |
| Loss                       | Sparse categorical cross-entropy       |
| Primary search criterion   | Maximum validation accuracy            |
| Secondary search criterion | Minimum validation loss                |

The code seeds Python's `random` module, NumPy, and TensorFlow with the same seed.

Reproducibility is improved by explicitly clearing the Keras session before each search model and reseeding TensorFlow and NumPy for each trial.

## Implementation Details

### Environment setup

The source checks whether TensorFlow is already importable. If not, it installs a compatible TensorFlow release in the range:

```text
tensorflow>=2.15,<2.21
```

The remaining core dependencies imported by the source are:

* NumPy
* pandas
* Matplotlib
* scikit-learn
* scikit-image
* TensorFlow / Keras

Google Colab is supported explicitly. When TensorFlow detects a GPU, the notebook prints the available physical GPU devices.

### Data decoding

CSV pixel vectors are reconstructed with `numpy.fromstring`. The implementation checks that the number of decoded pixels equals `height × width` for every sample before reshaping.

This matters because the CSV files preserve variable-sized source images rather than forcing all original images to the same dimensions before preprocessing.

### Image resizing

The resizing function uses:

```python
resize(
    image,
    (20, 20),
    preserve_range=True,
    anti_aliasing=True,
)
```

The resulting images are converted to `float32` and then divided by 255.

### Feature construction

The resized image tensor has shape:

```text
(n_samples, 20, 20)
```

It is flattened into:

```text
(n_samples, 400)
```

before standardization and model input.

### Diagnostics

The implementation automatically creates the following diagnostics:

1. Resized training-image samples.
2. Optimizer and learning-rate comparison.
3. Training and validation classification-error curves.
4. Example misclassified test images with confidence.
5. Test confusion matrix.
6. First test image visualization.
7. First test image probability table.
8. Per-class classification report.

## Project Structure

A typical execution environment contains the following:

```text
.
├── assignment_solution.py
├── datasets/
│   ├── hoda_train.csv
│   └── hoda_test.csv
└── results/
    ├── 01_resized_samples.png
    ├── 02_optimizer_learning_rate_comparison.png
    ├── 03_training_error_curve.png
    ├── 04_misclassified_examples.png
    ├── 05_confusion_matrix.png
    ├── 06_first_test_image.png
    ├── classification_report.csv
    ├── first_test_image_probabilities.csv
    ├── final_summary.json
    ├── hoda_fully_connected_model.keras
    ├── optimizer_learning_rate_comparison.csv
    └── run_manifest.json
```

When executed in Google Colab, the project root is `/content`. Outside Colab, the current working directory is used as the project root.

## Installation

### Recommended Python environment

A Python environment with the dependencies imported by the source is required.

```bash
python -m pip install "tensorflow>=2.15,<2.21" numpy pandas matplotlib scikit-learn scikit-image
```

The script can also install TensorFlow automatically if it is missing.

### Google Colab

The source is designed to run in Google Colab. Open the notebook or adapt the Python source to a Colab cell structure, then place the two required CSV files in `datasets/` or upload them when prompted.

## Usage

### 1. Prepare the dataset

Create:

```text
datasets/hoda_train.csv
datasets/hoda_test.csv
```

The files must satisfy the schema and row-count checks described above.

### 2. Run the pipeline

Execute:

```bash
python assignment_solution.py
```

or run the corresponding cells in Google Colab.

### 3. Inspect the generated results

After execution, inspect the files under `results/` and the printed final summary.

The pipeline automatically saves the trained Keras model, optimizer-search table, classification report, probability vector, diagnostic figures, and JSON manifests.

### 4. Recreate the exported archive

The source also creates:

```text
assignment_results.zip
```

from the contents of `results/`. In Google Colab, the archive is offered for download automatically.

## Evaluation

The final test evaluation uses Keras `model.evaluate` on the full 2,000-image test set:

```python
test_loss, test_accuracy = final_model.evaluate(
    X_test_std,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=0,
)
```

The implementation also predicts the full test set with:

```python
test_probabilities = final_model.predict(
    X_test_std,
    batch_size=BATCH_SIZE,
    verbose=0,
)
```

Predicted classes are obtained with `argmax` over the ten output probabilities.

### Metrics

The source reports the following:

* test loss;
* test accuracy;
* number of misclassified test images;
* confusion matrix;
* per-class precision;
* per-class recall;
* per-class F1-score;
* support;
* the first test image's ten-class probability vector.

Classification error rate is defined as:

$$
\operatorname{error}
=
1-\operatorname{accuracy}.
$$

The training-error plot therefore shows

$$
e_t = 1-a_t,
$$

where $a_t$ is the training accuracy at epoch $t$.

## Results

The repository does not embed fixed numerical results in the source. This is intentional: the optimizer winner, validation accuracy, training accuracy, test accuracy, test loss, misclassification count, and first-test-image prediction are all computed at runtime.

The runtime summary contains:

| Field                               | Generated at execution time |
| ----------------------------------- | --------------------------- |
| Best optimizer / learning rate      | Yes                         |
| Best validation accuracy            | Yes                         |
| Best observed training accuracy     | Yes                         |
| Test accuracy                       | Yes                         |
| Test loss                           | Yes                         |
| Number of misclassified test images | Yes                         |
| First test image: true digit        | Yes                         |
| First test image: predicted digit   | Yes                         |
| First test image: correctness       | Yes                         |

This design avoids presenting fabricated or outdated experimental numbers.

## Error Analysis

Misclassified test examples are displayed with:

* the true digit;
* the predicted digit;
* the predicted-class confidence.

The source also computes a confusion matrix and a scikit-learn classification report to make class-specific error patterns observable.

The implementation identifies several plausible contributors to error:

* visual similarity between digit classes;
* variation in handwriting style and stroke width;
* loss of fine spatial detail caused by resizing to $20\times20$;
* the limited ability of a purely fully connected architecture to exploit local spatial structure explicitly.

These are hypotheses for interpreting the observed errors, not universal causal claims. Stronger conclusions should be based on the confusion matrix and displayed misclassified examples.

## First-Test-Image Probability Analysis

The source explicitly inspects the first test image.

For that sample, it records:

* the ten-dimensional softmax probability vector;
* the predicted digit;
* the true digit;
* whether the prediction is correct.

The probability table has the form:

| Digit |   Probability |
| ----: | ------------: |
|     0 | Runtime value |
|     1 | Runtime value |
|     2 | Runtime value |
|     3 | Runtime value |
|     4 | Runtime value |
|     5 | Runtime value |
|     6 | Runtime value |
|     7 | Runtime value |
|     8 | Runtime value |
|     9 | Runtime value |

The exact probabilities are omitted from this static README because they are produced only when the model is run.

## Reproducibility

The source makes several reproducibility choices:

### Deterministic seeds

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

The same seed is reapplied during optimizer-search trials and final model construction.

### Controlled search

Every optimizer trial uses the same:

* network architecture;
* seed;
* batch size;
* number of search epochs;
* fitting partition;
* validation partition.

Only optimizer type and learning rate vary across the search grid.

### Leakage control

Feature-standardization statistics are estimated from the fitting partition rather than from the complete dataset.

The test set is not used during optimizer selection.

### Runtime manifests

The source exports:

* `run_manifest.json`, containing the dataset and architecture configuration plus the TensorFlow version;
* `final_summary.json`, containing the principal numerical outcomes of the executed run.

### Model serialization

The trained Keras model is saved as:

```text
results/hoda_fully_connected_model.keras
```

This allows later loading and inference without retraining, provided the software environment is compatible.

## Limitations

This implementation has several methodological limitations that should be kept explicit.

### Fully connected architecture

The model receives flattened pixels rather than exploiting convolutional locality. Consequently, the representation does not encode translation-aware local structure in the way a CNN would.

### Fixed 20×20 representation

All original images are projected to $20\times20$. Resizing simplifies the model input contract but can remove fine-grained stroke information.

### Small hyperparameter search

Only six optimizer/learning-rate configurations are compared. The source does not perform a broad search over architecture, regularization, batch size, or data augmentation.

### No data augmentation

The implementation does not introduce geometric or photometric augmentation.

### Validation protocol

The optimizer search uses a stratified scikit-learn validation split, whereas the final training call uses Keras's `validation_split=0.15` on the full training matrix for callback monitoring. These are separate validation mechanisms and should not be conflated.

### Reproducibility is practical, not absolute

Random seeds are explicitly set, but exact bit-for-bit determinism can still depend on the TensorFlow version, hardware, GPU kernels, and execution environment.

### No hard-coded benchmark claim

Because the source computes results dynamically and the uploaded `.py` file does not include a completed numerical run, this README makes no claim about a specific final accuracy.

## Future Work

The current implementation provides a baseline for further experimentation.

Natural extensions include:

* replacing the FNN with a convolutional neural network while preserving the same train/test protocol;
* introducing controlled data augmentation;
* expanding hyperparameter search beyond optimizer and learning rate;
* comparing alternative normalization strategies;
* calibrating softmax confidence;
* performing repeated runs across multiple random seeds;
* reporting confidence intervals for evaluation metrics;
* analyzing class-pair confusion quantitatively;
* comparing parameter count, training time, and accuracy across architectures;
* evaluating whether a higher-resolution representation improves recognition without excessive computational cost.

These extensions are recommendations for future experiments, not results established by the current implementation.

## Files Generated at Runtime

The source writes the following artifacts:

| File                                        | Purpose                                                     |
| ------------------------------------------- | ----------------------------------------------------------- |
| `01_resized_samples.png`                    | Visual sanity check of resized images                       |
| `02_optimizer_learning_rate_comparison.png` | Validation-accuracy comparison across search configurations |
| `03_training_error_curve.png`               | Training and validation classification-error curves         |
| `04_misclassified_examples.png`             | Visual examples of incorrect test predictions               |
| `05_confusion_matrix.png`                   | Test-set confusion matrix                                   |
| `06_first_test_image.png`                   | First test image                                            |
| `classification_report.csv`                 | Per-class classification metrics                            |
| `first_test_image_probabilities.csv`        | Ten-class probability vector for the first test image       |
| `final_summary.json`                        | Runtime summary of the experiment                           |
| `hoda_fully_connected_model.keras`          | Serialized final Keras model                                |
| `optimizer_learning_rate_comparison.csv`    | Raw optimizer-search results                                |
| `run_manifest.json`                         | Experiment configuration and TensorFlow version             |

## Research and Engineering Notes

### Why validation accuracy is used for model selection

The search is designed as a controlled comparison of optimization configurations. Validation accuracy is used as the primary selection criterion, with validation loss used to break ties.

The test set is held back until the selected configuration has been finalized.

### Why standardization is performed before the dense network

A dense network is sensitive to input scale because optimization operates through weighted linear combinations and nonlinear transformations. Feature standardization centers each input dimension and scales it using statistics estimated from the fitting partition.

### Why Batch Normalization and Dropout are both used

Batch Normalization and Dropout address different aspects of training. Batch Normalization normalizes layer activations, while Dropout randomly removes a fraction of activations during training as a regularization mechanism.

The source uses both mechanisms rather than relying on either one alone.

### Why the model returns probabilities

The softmax output provides a probability distribution over the ten digit classes. This supports not only hard classification through `argmax`, but also confidence inspection and error analysis.

