# Computer Assignment 3 — Flower Species Classification with a Regularized CNN

A reproducible TensorFlow/Keras implementation of a five-class flower image classifier built with a custom convolutional neural network (CNN). The project includes systematic optimizer and learning-rate comparison, regularization, detailed classification metrics, inference visualization, and automated result packaging.

> **Academic context.** This repository documents *Computer Assignment 3 (CA3)* from a Deep Learning course. The implementation is written as an executable Python workflow and is structured to expose the complete experimental pipeline rather than only the final classifier.

## Overview

The project addresses five-class image classification for the flower categories:

- `daisy`
- `dandelion`
- `roses`
- `sunflowers`
- `tulips`

The implementation performs the following sequence:

1. Downloads and validates the TensorFlow flower image dataset.
2. Creates an 80/20 training/validation split with a fixed random seed.
3. Builds cached and prefetched `tf.data` pipelines.
4. Visualizes sample training images.
5. Constructs a regularized CNN with on-the-fly augmentation, normalization, batch normalization, L2 regularization, dropout, and global average pooling.
6. Compares four optimizers across three learning rates using a fixed 15-epoch grid search.
7. Selects the configuration with the highest observed validation accuracy.
8. Retrains the selected configuration for up to 25 epochs with checkpointing, adaptive learning-rate reduction, and early stopping.
9. Evaluates the selected model on both training and validation sets using accuracy, macro-averaged precision/recall/F1, weighted precision/recall/F1, and confusion matrices.
10. Demonstrates inference on four sample images and records predicted classes and confidence scores.
11. Exports tables, high-resolution plots, model artifacts, a text summary, and a ZIP archive.

The source implementation is a single executable script, `CA3(1).py`, and contains 854 lines of Python source.

## Research and Engineering Objectives

The implementation covers the assignment objectives stated in the source code: systematic hyperparameter selection, multiple measures to reduce overfitting, training-accuracy reporting, multi-class precision/recall/F1 evaluation, convergence visualization, optimizer and learning-rate comparison, custom-image inference, and automated artifact generation.

From a machine-learning perspective, the project examines the interaction between architectural inductive bias, regularization, and optimization in a small multi-class image-classification problem.

## Dataset

The script downloads the `flower_photos` dataset from the TensorFlow example-image storage endpoint and extracts it into a local cache under `ca3_assignment_outputs/dataset_cache`. The dataset is then interpreted as a directory tree whose immediate subdirectories represent the five target classes.

The dataset pipeline is constructed with `tf.keras.utils.image_dataset_from_directory` using:

| Setting | Value |
|---|---:|
| Input height | 180 |
| Input width | 180 |
| Channels | 3 (RGB) |
| Batch size | 32 |
| Training fraction | 80% |
| Validation fraction | 20% |
| Split seed | 123 |
| Training shuffle | Yes |
| Validation shuffle | No |

These settings are taken directly from the implementation.

### Important evaluation note

The code creates training and validation partitions, but it does **not** create a separate held-out test set. The same validation partition is used during optimizer/learning-rate selection, early stopping, model checkpoint selection, and final validation reporting. Consequently, the validation metrics should be interpreted as model-selection metrics rather than as an unbiased estimate from an untouched test set. This is an important methodological limitation when presenting the project as research evidence.

## Reproducibility Configuration

The workflow uses a single global seed:

```python
RANDOM_SEED = 123
```

The seed is propagated to Python's `random` module, NumPy, TensorFlow, the dataset splitting process, augmentation layers, and dropout. The script also sets `PYTHONHASHSEED`.

The implementation reports the TensorFlow version at runtime and checks for available physical GPUs before training.

> **Determinism caveat.** Setting these seeds improves repeatability, but exact bit-for-bit reproducibility can still depend on TensorFlow version, hardware, GPU kernels, and execution environment. The source does not explicitly enable TensorFlow's global deterministic-operation mode.

## Scientific Background

### Multi-class image classification

Let an input image be represented by $x$ and let the target space contain $C=5$ classes. The network produces a vector of class logits $z \in \mathbb{R}^{C}$, which is transformed into a probability distribution by the softmax function.

$$
P(y=i \mid x) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}.
$$

The implementation uses a five-unit output layer with `softmax`, so the output probabilities sum to one and the predicted class is the index of the largest probability.

### Input normalization

Images are represented initially on the usual 8-bit RGB scale. The network applies a `Rescaling(1/255)` layer, giving the normalized input

$$
x' = \frac{x}{255}.
$$

This transformation is implemented inside the model rather than as a separate preprocessing script.

### Sparse categorical cross-entropy

Because labels are stored as integer class indices rather than one-hot vectors, the compiled loss is sparse categorical cross-entropy. For a sample with true class $y$, the cross-entropy term is

$$
\mathcal{L}_{\mathrm{CE}} = -\log P(y \mid x).
$$

The source configures `SparseCategoricalCrossentropy(from_logits=False)`, which is consistent with the final softmax output.

### L2 regularization

The convolutional and dense feature layers use an L2 kernel regularizer with coefficient $\lambda=10^{-4}$. Conceptually, a regularized objective can be written as

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda \sum_k \lVert W_k \rVert_2^2,
$$

where $W_k$ denotes a regularized trainable kernel. In Keras, these regularization losses are added to the model's training loss. The source applies the regularizer to all four convolutional kernels and to the dense feature layer.

## Methodology

The experiment can be viewed as a two-stage optimization procedure.

### Stage 1 — architecture and optimization search

The model architecture is held fixed while the script evaluates four optimizers:

- Adam
- SGD with momentum and Nesterov acceleration
- RMSprop
- Adagrad

Each optimizer is evaluated at three learning rates:

$$
\eta \in \{10^{-2},10^{-3},10^{-4}\}.
$$

This produces $4\times3=12$ training configurations. Every configuration is trained for 15 epochs using the same training and validation pipelines. The configuration with the largest **peak validation accuracy over its 15 epochs** is selected.

### Stage 2 — full training of the selected configuration

The selected optimizer and learning rate are used to instantiate a fresh model. It is then trained for up to 25 epochs with three callbacks:

- `ModelCheckpoint`: monitors `val_accuracy`, uses `mode="max"`, saves only the best checkpoint.
- `ReduceLROnPlateau`: monitors `val_loss`, multiplies the learning rate by `0.5` after 4 stagnant epochs, and never reduces it below `1e-6`.
- `EarlyStopping`: monitors `val_accuracy`, uses patience `10`, and restores the best weights.

The exact settings are implemented in the training callback list.

## Model Architecture

The classifier is a custom CNN with a sequential layer structure, implemented through the Keras Functional API.

### Architecture summary

```text
Input: 180 × 180 × 3
  │
  ├── Random horizontal flip
  ├── Random rotation (0.15)
  ├── Random zoom (0.15)
  ├── Random contrast (0.10)
  ├── Rescaling: x / 255
  │
  ├── Conv2D: 32 filters, 3×3, same padding, L2
  ├── BatchNormalization
  ├── ReLU
  └── MaxPooling2D: 2×2
       ↓ 90 × 90 × 32
  ├── Conv2D: 64 filters, 3×3, same padding, L2
  ├── BatchNormalization
  ├── ReLU
  └── MaxPooling2D: 2×2
       ↓ 45 × 45 × 64
  ├── Conv2D: 128 filters, 3×3, same padding, L2
  ├── BatchNormalization
  ├── ReLU
  └── MaxPooling2D: 2×2
       ↓ 22 × 22 × 128
  ├── Conv2D: 256 filters, 3×3, same padding, L2
  ├── BatchNormalization
  ├── ReLU
  └── MaxPooling2D: 2×2
       ↓ 11 × 11 × 256
  ├── GlobalAveragePooling2D
  ├── Dense: 256 units, L2
  ├── BatchNormalization
  ├── ReLU
  ├── Dropout: 0.30
  └── Dense: 5 units, Softmax
```

The layer sequence and hyperparameters are implemented explicitly in `build_regularized_cnn`.

### Architectural rationale

#### Data augmentation

Augmentation is performed dynamically inside the model using:

- random horizontal flipping,
- random rotation with parameter `0.15`,
- random zoom with parameter `0.15`, and
- random contrast with parameter `0.1`.

This introduces input variability during training without permanently modifying the stored dataset.

The exact geometric magnitude represented by Keras's `RandomRotation(0.15)` and `RandomZoom(0.15)` is delegated to TensorFlow/Keras layer semantics; this README therefore intentionally does not reinterpret those parameters as degrees or percentages.

#### Convolution and pooling

Each convolution uses a $3\times3$ kernel with `same` padding. The number of filters increases from 32 to 64, 128, and 256, while $2\times2$ max pooling reduces spatial resolution after each block.

This pattern increases representational capacity while progressively reducing spatial resolution.

#### Batch normalization

Batch normalization follows every convolution and also the dense feature layer. The source describes this as a way to help stabilize training.

#### Global average pooling

Instead of flattening the final feature maps into a very large vector, the implementation uses `GlobalAveragePooling2D`. For channel $k$, the operation aggregates the spatial activations and can be viewed conceptually as

$$
g_k = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} a_{h,w,k}.
$$

This reduces the dimensionality of the classification head and avoids the large increase in parameters that direct flattening would introduce.

#### Dropout

A dropout rate of 0.3 is applied after the 256-unit dense feature layer.

## Optimization Study

The implementation supports five optimizer choices at the model-construction level:

| Optimizer key | TensorFlow/Keras optimizer | Additional configuration |
|---|---|---|
| `adam` | `Adam` | Learning rate only |
| `sgd` / `sgd_momentum` | `SGD` | Momentum `0.9`, Nesterov enabled |
| `rmsprop` | `RMSprop` | Learning rate only |
| `adagrad` | `Adagrad` | Learning rate only |
| `adamw` | `AdamW` | Learning rate plus `weight_decay=l2_reg` |

Only the first four entries are included in the 12-run comparison grid. `AdamW` is supported by the model factory but is not benchmarked in the automated search.

The selection criterion is explicitly **peak validation accuracy**, not final validation loss or final validation accuracy:

$$
A_{\mathrm{peak}}(o,\eta) = \max_{t=1,\ldots,15} A_{\mathrm{val}}(o,\eta,t).
$$

$$
(\hat{o},\hat{\eta}) = \mathrm{argmax}_{o,\eta} A_{\mathrm{peak}}(o,\eta).
$$

This distinction is important when interpreting the generated comparison table.

## Regularization Strategy

The implementation combines several independent anti-overfitting mechanisms:

1. **Data augmentation:** horizontal flip, rotation, zoom, and contrast perturbation.
2. **Batch normalization:** inserted after every convolution and after the dense feature layer.
3. **L2 kernel regularization:** coefficient $10^{-4}$ on convolutional and dense feature kernels.
4. **Dropout:** rate $0.30$ on the dense representation.
5. **Early stopping:** validation-accuracy based, with best-weight restoration.
6. **Adaptive learning-rate reduction:** validation-loss based `ReduceLROnPlateau`.

These mechanisms are visible directly in the model and training code.

## Evaluation Methodology

### Accuracy

Accuracy is the fraction of correctly classified examples:

$$
\mathrm{Accuracy}
=
\frac{1}{N}
\sum_{n=1}^{N}
\mathbf{1}(\hat{y}_n=y_n).
$$

The project reports accuracy from direct comparisons between predicted and ground-truth class indices.

### Precision, recall, and F1-score

For a class $c$, precision, recall, and F1-score are defined as

$$
\mathrm{Precision}_c = \frac{TP_c}{TP_c+FP_c},
$$

$$
\mathrm{Recall}_c = \frac{TP_c}{TP_c+FN_c},
$$

and

$$
F1_c = \frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}{\mathrm{Precision}_c+\mathrm{Recall}_c}.
$$

The implementation computes both **macro averages** and **weighted averages** using scikit-learn's `precision_recall_fscore_support`, with `zero_division=0`.

For $C$ classes, the macro average gives each class equal weight:

$$
M_{\mathrm{macro}} = \frac{1}{C}\sum_{c=1}^{C} M_c.
$$

The weighted version accounts for class support:

$$
M_{\mathrm{weighted}} = \sum_{c=1}^{C}\frac{n_c}{N}M_c.
$$

where $n_c$ is the number of true samples belonging to class $c$.

### Confusion matrices

The code computes one confusion matrix for the training set and one for the validation set, with true labels on the vertical axis and predicted labels on the horizontal axis.

A confusion matrix is useful for identifying systematic class confusions that aggregate accuracy can hide.

## Data Pipeline and Performance

The raw dataset objects are wrapped with TensorFlow's `cache`, training-set `shuffle(1000)`, and `prefetch(AUTOTUNE)` operations. Validation is cached and prefetched but not shuffled.

This pipeline improves input throughput while preserving deterministic label ordering for validation evaluation under the configured seed and pipeline structure.

## Inference Procedure

The inference demonstration selects up to four images from the class directories, falls back to a recursive search if necessary, resizes each image to $180\times180$, converts it to an array, adds a batch dimension, and obtains a softmax prediction from the trained model.

The predicted class is

$$
\hat{y} = \operatorname*{arg\,max}_{i\in\{1,\ldots,5\}} P(y=i\mid x),
$$

and the reported confidence is the corresponding maximum softmax probability expressed as a percentage.

> **Interpretation caution.** A softmax probability is a model score, not automatically a calibrated probability of correctness. The source does not perform calibration analysis.

The inference results are saved to `sample_inference_results.csv`, and the corresponding visualization is saved as a high-resolution PNG.

## Outputs and Artifact Generation

All generated artifacts are placed under `ca3_assignment_outputs/` unless explicitly noted otherwise. The script creates the following directories:

```text
ca3_assignment_outputs/
├── flower_photos.tgz
├── final_assignment_report.txt
├── dataset_cache/
├── plots/
│   ├── sample_dataset_images.png
│   ├── optimizer_lr_comparison.png
│   ├── training_and_error_curves.png
│   ├── confusion_matrices_train_val.png
│   └── sample_inference_predictions.png
├── tables/
│   ├── optimizer_lr_comparison.csv
│   ├── overall_metrics_summary.csv
│   ├── classification_report_train.csv
│   ├── classification_report_val.csv
│   └── sample_inference_results.csv
└── saved_models/
    └── best_flower_cnn.keras
```

The exact file creation logic is implemented in the source.

In addition, the script creates a root-level archive:

```text
flower_classification_ca3_results.zip
```

The archive is generated with `ZIP_DEFLATED` compression and intentionally excludes the internal `dataset_cache` directory.

## Project Structure

```text
.
├── CA3(1).py
├── README.md
├── ca3_assignment_outputs/
│   ├── plots/
│   ├── tables/
│   ├── saved_models/
│   ├── dataset_cache/
│   ├── flower_photos.tgz
│   └── final_assignment_report.txt
└── flower_classification_ca3_results.zip
```

The last entries are generated after execution; they do not need to exist before the first run.

## Installation

### Python environment

The source imports the following third-party Python packages:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

Standard-library modules used by the script include `os`, `sys`, `shutil`, `zipfile`, `pathlib`, and `random`.

A minimal installation can therefore be prepared with:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

Because the source does not pin exact package versions, strict environment reproduction should use the TensorFlow/Python versions recorded in your execution environment and, for archival reproducibility, a separately maintained environment specification such as `requirements.txt` or `environment.yml`.

## Usage

Run the script from the repository root:

```bash
python "CA3(1).py"
```

The script then runs the complete workflow automatically:

```text
Dataset download
      ↓
Train/validation split
      ↓
Data pipeline preparation
      ↓
Sample visualization
      ↓
12-run optimizer/LR comparison
      ↓
Best configuration selection
      ↓
Full training with callbacks
      ↓
Training/validation curves
      ↓
Metric computation
      ↓
Confusion matrices
      ↓
Sample inference
      ↓
Text report generation
      ↓
ZIP packaging
```

No command-line arguments are defined in the source; experiment settings are controlled by module-level constants and the arguments of `build_regularized_cnn`.

## Configuration

The primary global configuration is:

| Parameter | Value |
|---|---:|
| `RANDOM_SEED` | `123` |
| `IMG_HEIGHT` | `180` |
| `IMG_WIDTH` | `180` |
| `CHANNELS` | `3` |
| `BATCH_SIZE` | `32` |
| `NUM_CLASSES` | `5` |
| `DEFAULT_EPOCHS` | `25` |
| `EXP_EPOCHS` | `15` |
| Default dropout | `0.3` |
| Default L2 coefficient | `1e-4` |
| Default learning rate | `1e-3` |
| SGD momentum | `0.9` |

These values correspond to the source definitions and model-construction defaults.

The optimizer search space is:

| Dimension | Values |
|---|---|
| Optimizer | Adam, SGD+Momentum/Nesterov, RMSprop, Adagrad |
| Learning rate | `0.01`, `0.001`, `0.0001` |
| Comparison epochs | `15` |
| Full-training maximum | `25` |

The comparison grid is created directly from these candidate lists.

## Results

The source computes results at runtime and writes them to CSV and text artifacts. The uploaded source file therefore does **not** provide fixed numerical experiment outputs that can be reproduced in this README without executing the experiment. To avoid fabricating empirical evidence, this document intentionally does not claim specific accuracy, precision, recall, F1, or optimizer-ranking values.

After a successful run, the main quantitative artifacts are:

- `ca3_assignment_outputs/tables/optimizer_lr_comparison.csv`
- `ca3_assignment_outputs/tables/overall_metrics_summary.csv`
- `ca3_assignment_outputs/tables/classification_report_train.csv`
- `ca3_assignment_outputs/tables/classification_report_val.csv`
- `ca3_assignment_outputs/tables/sample_inference_results.csv`
- `ca3_assignment_outputs/final_assignment_report.txt`

The script selects the best configuration using the maximum validation accuracy observed during the 15-epoch comparison stage, then reinitializes and retrains the model with that configuration.

### Interpreting the generated metrics

The source reports both final and peak validation performance during the optimizer study. These quantities should not be treated as interchangeable:

- **Final validation accuracy:** validation accuracy at the final recorded comparison epoch.
- **Peak validation accuracy:** maximum validation accuracy across all comparison epochs.
- **Final model validation accuracy:** accuracy recomputed from the selected checkpoint on the validation dataset.
- **Macro F1:** treats all flower classes equally.
- **Weighted F1:** weights classes according to the number of true samples.

This distinction is encoded in the generated comparison records and overall summary table.

## Reproducibility Checklist

For a reproducible run, record at least:

- Python version
- TensorFlow version
- NumPy version
- pandas version
- scikit-learn version
- Matplotlib version
- Seaborn version
- operating system
- CPU/GPU model
- CUDA/cuDNN versions when GPU execution is used
- the exact source revision
- the generated optimizer comparison CSV
- the generated checkpoint

The script already records the TensorFlow version and whether a GPU is detected.

## Limitations and Methodological Considerations

### No independent test set

The largest evaluation limitation is the absence of an untouched test partition. Because validation performance drives both hyperparameter selection and checkpoint selection, the final validation score can be optimistic relative to performance on genuinely unseen data.

### Single split

The experiment uses one fixed 80/20 split rather than repeated stratified cross-validation or multiple independent random seeds. This limits the ability to quantify variance and confidence intervals across data partitions.

### Hyperparameter search scope

The search varies only optimizer choice and learning rate. The architecture, augmentation configuration, dropout rate, L2 coefficient, batch size, and input resolution are not systematically optimized by the grid search.

### Search criterion

The winning configuration is chosen by peak validation accuracy across the 15 comparison epochs. Peak-based selection can favor transient performance and does not directly optimize calibration, macro F1, or final validation loss.

### No external benchmark

The source does not compare the custom CNN with a pretrained architecture such as ResNet, EfficientNet, or MobileNet. Consequently, no claim of state-of-the-art performance is justified from this experiment alone.

### No statistical uncertainty analysis

The implementation reports point estimates but does not compute confidence intervals, bootstrap intervals, repeated-run variance, or formal significance tests.

### Confidence is not calibration

The inference visualization labels the maximum softmax score as confidence, but the project does not evaluate calibration metrics such as expected calibration error. The reported percentage should therefore be treated as a model output score rather than a calibrated reliability estimate.

### Internal validation leakage in sample inference

The demonstration code selects images directly from the dataset directory and does not maintain a separate test-image repository. Some displayed samples may therefore belong to the same underlying pool used for training or validation. The inference demonstration is illustrative rather than an independent generalization benchmark.

### Package-version non-pinning

The source does not provide a lockfile or pinned dependency specification. Long-term reproducibility therefore requires recording the execution environment externally.

## Future Work

Several extensions could strengthen the project as a research-oriented image-classification study:

1. Introduce a third, untouched test partition and reserve it strictly for final reporting.
2. Repeat the experiment across multiple random seeds and report mean, standard deviation, and confidence intervals.
3. Replace the single optimizer/LR grid with a structured experiment over augmentation strength, dropout, L2 coefficient, batch size, and architectural depth.
4. Add class-wise support counts and normalized confusion matrices to improve diagnostic interpretation.
5. Compare the custom CNN with transfer-learning baselines such as ResNet, EfficientNet, or MobileNet under identical data splits.
6. Evaluate calibration using reliability diagrams and expected calibration error.
7. Add explicit experiment metadata capture, including package versions, hardware information, and configuration serialization.
8. Separate hyperparameter selection from final evaluation through nested validation or a dedicated test set.
9. Add model-interpretability analysis, for example Grad-CAM, to investigate which image regions influence predictions.
10. Pin the software environment to make the artifact easier to reproduce years after publication.

## Implementation Notes and Code Audit Findings

The following details are particularly relevant when reading the source as a research artifact:

- `build_regularized_cnn` appears twice in the source. The earlier occurrence is a duplicated documentation scaffold; the later definition is the executable implementation used by Python.
- The model factory supports `adamw`, but `adamw` is absent from the automated optimizer comparison list.
- The optimizer study uses a fixed 15-epoch budget for comparability, while the final selected model may train for fewer than 25 epochs because of early stopping.
- The model-selection checkpoint is monitored by validation accuracy, whereas learning-rate reduction is driven by validation loss. These are deliberately different control signals.
- The source uses `cache()` for both training and validation datasets. This can increase memory consumption for larger datasets, but is reasonable for a small image corpus if sufficient host memory is available.
- The ZIP packaging stage excludes the internal dataset cache to keep the archive smaller while preserving generated experimental artifacts.

## Scientific Interpretation

The central methodological idea is not simply “train a CNN,” but to expose the full experimental loop:

$$
\text{data}
\rightarrow
\text{augmentation and normalization}
\rightarrow
\text{CNN representation learning}
\rightarrow
\text{optimizer/LR comparison}
\rightarrow
\text{model selection}
\rightarrow
\text{regularized retraining}
\rightarrow
\text{multi-metric evaluation}.
$$

This structure is useful for an academic portfolio because it shows that a machine-learning result depends on the full experimental protocol, not only on network topology.

At the same time, the evaluation protocol should be interpreted carefully: because the validation set is repeatedly consulted during search and checkpoint selection, the experiment demonstrates a controlled development workflow rather than a fully independent generalization study.

## References

### Project source

- Source implementation: `CA3(1).py` in this repository.

### Dataset

- TensorFlow example-image endpoint used by the script: <https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz>

### GitHub mathematical rendering

- GitHub Docs — *Writing mathematical expressions*: <https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions>

GitHub documents support LaTeX-formatted mathematics in Markdown through MathJax and explicitly document `$...$` for inline expressions and `$$...$$` for display mathematics.

### Core software libraries

- TensorFlow / Keras: <https://www.tensorflow.org/>
- scikit-learn: <https://scikit-learn.org/>
- NumPy: <https://numpy.org/>
- pandas: <https://pandas.pydata.org/>
- Matplotlib: <https://matplotlib.org/>
- Seaborn: <https://seaborn.pydata.org/>

## License and Attribution

No explicit software license is declared in the provided source. This README therefore does not assign a license that is not present in the project.

The source identifies the work as a Deep Learning course computer assignment and attributes the script header to an “Academic Machine Learning Engineer & Reproducibility Team.”

## Citation

If this project is referenced academically, cite the repository version and the original course assignment context according to the repository metadata used for publication. Because the provided source does not contain a formal bibliographic citation or DOI, this README does not invent one.
