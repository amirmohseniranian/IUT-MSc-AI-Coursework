# Combined Classical Machine Learning Analysis

A reproducible Google Colab workflow for analyzing a tabular Excel dataset with classical supervised-learning methods. The implementation combines regression, binary classification, decision-boundary visualization, and a manual matrix-based derivation in a single executable script.

The repository is built around the supplied Python source and documents the implemented workflow without introducing unimplemented experiments or unsupported performance claims.

## Overview

The program expects an Excel workbook named `mltak1Data.xlsx` (or, as a fallback, the single uploaded workbook in the Colab session). It loads the data, constructs predictor variables, and executes four analytical components:

| Component | Method | Main purpose | Primary outputs |
|---|---|---|---|
| 1 | Linear Regression | Predict the numeric `total` variable from the remaining feature columns | Feature coefficients and intercept |
| 2 | Ridge Classifier | Binary classification of `pass` | Accuracy, precision, confusion matrix |
| 2 | SGD Classifier | Binary classification of `pass` | Accuracy, precision, confusion matrix |
| 2 | Logistic Regression | Binary classification of `pass` | Accuracy, precision, confusion matrix |
| 3 | Logistic Regression on `midterm` and `final` | Visualize a two-dimensional classification boundary | Scatter plot and fitted parameters |
| 4 | Manual matrix-based least-squares model | Reproduce a two-dimensional linear score model from matrix operations | Weight vector, matrix `W`, and decision-boundary plot |

The workflow creates separate directories for figures, tables, and text results, then packages the generated artifacts together with the input workbook in a ZIP archive.

## Motivation

The project provides a compact study of several classical machine-learning techniques applied to the same tabular dataset. It also connects scikit-learn estimators with an explicit matrix calculation, making the relationship between a learned linear model and its underlying linear-algebra formulation easier to inspect.

The project is structured for Google Colab: the input workbook is uploaded interactively, results are written to the Colab filesystem, and the final archive is downloaded automatically.

## Scientific Background

### Linear Regression

For observations represented by a feature matrix $X$ and a numeric target vector $y$, ordinary linear regression models the conditional mean of the target as a linear function of the predictors:

$$
\hat{y} = Xw + b.
$$

The implementation fits `sklearn.linear_model.LinearRegression` using `df_2` as the feature matrix and `df['total']` as the response variable. The source reports each fitted feature coefficient and the intercept.

### Binary Classification

The `pass` column is converted into a binary target for the classifier comparison:

$$
 t_i =
 \begin{cases}
 0, & \text{if the original value is `no`},\\
 1, & \text{otherwise}.
 \end{cases}
$$

The resulting target vector is used with three scikit-learn classifiers: `RidgeClassifier`, `SGDClassifier`, and `LogisticRegression`.

For a binary linear classifier, a decision score can be expressed generally as

$$
 s(x) = w^\top x + b.
$$

The corresponding decision boundary is the set of points for which the score equals zero:

$$
 w^\top x + b = 0.
$$

### Logistic Regression

Logistic regression models the probability of the positive class through the logistic function:

$$
 P(y=1\mid x) = \sigma(w^\top x+b),
$$

where

$$
\sigma(z) = \frac{1}{1+e^{-z}}.
$$

The source uses the default `LogisticRegression` estimator and evaluates it on the same data used for training.

### Linear Decision Boundary in Two Dimensions

When the predictor vector contains two variables, such as `midterm` and `final`, a linear boundary has the form

$$
 w_0 + w_1 x_1 + w_2 x_2 = 0.
$$

Solving for $x_2$ gives

$$
 x_2 = -\frac{w_0}{w_2} - \frac{w_1}{w_2}x_1,
$$

provided $w_2 \neq 0$.

### Manual Matrix-Based Calculation

The final analytical component explicitly constructs an augmented design matrix whose first column contains ones. With

$$
X =
\begin{bmatrix}
1 & x_{11} & x_{12}\\
1 & x_{21} & x_{22}\\
\vdots & \vdots & \vdots\\
1 & x_{n1} & x_{n2}
\end{bmatrix},
$$

the code computes

$$
X^{+}_{\text{normal}} = (X^\top X)^{-1}X^\top,
$$

and then multiplies this matrix by a two-column one-hot target matrix $T$:

$$
W = (X^\top X)^{-1}X^\top T.
$$

The implementation then forms a three-parameter difference vector from the two columns of $W$:

$$
\begin{aligned}
w_0 &= W_{0,2} - W_{0,1},\\
w_1 &= W_{1,2} - W_{1,1},\\
w_2 &= W_{2,2} - W_{2,1}.
\end{aligned}
$$

The resulting boundary is plotted as

$$
 x_2 = -\frac{w_0}{w_2} - \frac{w_1}{w_2}x_1.
$$

This procedure is a manual least-squares construction with one-hot targets. It is not an implementation of maximum-likelihood logistic regression.

## Methodology

### Data Loading and Feature Selection

The workflow reads the Excel workbook with `pandas.read_excel`. The columns `total`, `pass`, and `name` are removed from the general feature matrix for Parts 1 and 2:

```python
df_2 = df.drop(['total', 'pass', 'name'], axis=1)
```

The numeric response for linear regression is `total`:

```python
yout = df['total']
```

For the binary classifiers, `pass` is converted to a numerical target using the rule implemented in the source code. The classifier feature matrix remains `df_2`.

The two-dimensional analyses explicitly use:

```python
X2 = df[['midterm', 'final']]
```

The workbook therefore needs to contain at least `total`, `pass`, `name`, `midterm`, and `final`, along with any additional predictor columns used by Parts 1 and 2.

### Part 1 — Linear Regression

The first component does the following:

1. Reads the Excel workbook.
2. Removes `total`, `pass`, and `name` from the predictor matrix.
3. Uses `total` as the continuous target.
4. Fits `LinearRegression()`.
5. Prints each feature coefficient and the intercept.
6. Exports the coefficient table to CSV and Excel.

The model has the standard form

$$
\hat{y} = b + \sum_{j=1}^{p} w_j x_j.
$$

The implementation does not perform a train/test split, feature scaling, cross-validation, regularization, or explicit residual analysis.

### Part 2 — Classifier Comparison

The source fits three binary classifiers to the same feature matrix and target:

- `RidgeClassifier`
- `SGDClassifier`
- `LogisticRegression`

For each estimator, the code reports:

- accuracy,
- precision,
- the confusion matrix,
- the fitted intercept.

The accuracy metric is

$$
\mathrm{Accuracy} = \frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}.
$$

The precision metric is

$$
\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},
$$

whenever the denominator is non-zero.

A confusion matrix records the counts of predicted classes against the corresponding true classes.

#### Important Evaluation Note

The source fits and evaluates each classifier on the same dataset:

```python
model.fit(df_2, t)
y = model.predict(df_2)
```

Consequently, the reported accuracy and precision are in-sample training metrics, not estimates of out-of-sample generalization performance. They should not be described as held-out test performance.

### Part 3 — Logistic Regression Decision-Boundary Visualization

This component fits logistic regression using only `midterm` and `final`, plots the observed points, and saves the resulting decision-boundary figure.

The class labels are retained as the original strings (`yes` and `no`) in this component, while the earlier classifier comparison explicitly converts them to `0` and `1`.

The fitted parameters are exported as:

- intercept,
- coefficient for `midterm`,
- coefficient for `final`.

#### Implementation-Specific Boundary Issue

The source fits a logistic-regression model whose score is of the form

$$
 b_0 + b_1 x_1 + b_2 x_2.
$$

Its zero-score boundary should therefore satisfy

$$
 x_2 = -\frac{b_0}{b_2} - \frac{b_1}{b_2}x_1.
$$

However, the plotting expression in the supplied source instead uses the first coefficient as the denominator. In mathematical form, the implemented plotting expression is equivalent to

$$
 x_2 = -\frac{b_0}{b_1} - \frac{b_2}{b_1}x_1.
$$

This is generally not the logistic-regression decision boundary. The discrepancy is documented here rather than silently changing the supplied implementation. The exported fitted parameters remain available for reproducing or correcting the visualization in a later revision.

### Part 4 — Manual Matrix-Based Classifier

The final component shows the explicit linear-algebra operations behind a least-squares model.

The feature vectors are built from `midterm` and `final`, and a leading constant column is inserted so the intercept is part of the design matrix. The code then computes an explicit matrix inverse:

```python
X_inverse_mul = np.linalg.inv(np.matmul(X2_trans, X2))
X_new = np.matmul(X_inverse_mul, X2_trans)
W = np.matmul(X_new, t)
```

The resulting weight differences define a linear score and a two-dimensional boundary. This is useful for teaching because the algebraic operations are exposed directly rather than hidden inside a machine-learning estimator.

## Implementation

### Software Stack

The supplied script installs and imports the following main packages:

| Package | Role |
|---|---|
| `pandas` | Excel loading, tabular data manipulation, and result export |
| `numpy` | Matrix and numerical operations in the manual derivation |
| `scikit-learn` | Linear regression, Ridge classification, SGD classification, logistic regression, and evaluation metrics |
| `matplotlib` | Scatter plots and decision-boundary figures |
| `seaborn` | Confusion-matrix heatmaps |
| `openpyxl` | Excel workbook I/O used by pandas and direct import |
| `google.colab` | File upload and download in the Colab environment |

The installation command in the source is:

```bash
pip -q install openpyxl seaborn scikit-learn pandas matplotlib numpy
```

The source does not pin package versions or record the exact Colab runtime version. Exact numerical reproducibility across future environments therefore cannot be guaranteed from the script alone.

### Project Execution Flow

```text
Excel workbook upload
        |
        v
Data loading with pandas
        |
        +---------------------------+
        |                           |
        v                           v
Linear regression            Binary classifiers
        |                    Ridge / SGD / Logistic
        |                           |
        v                           v
Coefficient exports       Accuracy / precision / CM
        |                           |
        +-------------+-------------+
                      |
                      v
         Logistic boundary analysis
                      |
                      v
         Manual matrix-based analysis
                      |
                      v
            Tables + figures + ZIP
```

## Project Structure

The Python source creates the following runtime structure under `/content/mltak_outputs`:

```text
mltak_outputs/
├── figures/
│   ├── 02_ridge_classifier_confusion_matrix.png
│   ├── 03_sgd_classifier_confusion_matrix.png
│   ├── 04_logistic_regression_confusion_matrix.png
│   ├── 06_logistic_regression_decision_boundary.png
│   └── 07_manual_matrix_decision_boundary.png
├── tables/
│   ├── 01_linear_regression_coefficients.csv
│   ├── 01_linear_regression_coefficients.xlsx
│   ├── 02_ridge_classifier_confusion_matrix.csv
│   ├── 03_sgd_classifier_confusion_matrix.csv
│   ├── 04_logistic_regression_confusion_matrix.csv
│   ├── 05_classifier_summary.csv
│   ├── 05_classifier_summary.xlsx
│   ├── 06_logistic_decision_boundary_parameters.csv
│   ├── 07_manual_matrix_weights.csv
│   └── 07_manual_matrix_W.csv
└── text_results/
    └── generated_files.txt
```

The script also copies the input Excel workbook into the output directory when possible and creates a ZIP archive named `mltak_all_results.zip` in `/content`.

## Installation and Environment

### Recommended Environment

The script was written for Google Colab and uses Colab-specific file APIs:

```python
from google.colab import files
```

and therefore should be executed in a Google Colab notebook or another environment that provides compatible `google.colab` functionality.

### Input File

Upload an Excel workbook named:

```text
mltak1Data.xlsx
```

The upload logic uses that exact filename when it is present. If exactly one file is uploaded under another name, the script uses that file as a fallback.

The script requires the following semantic columns:

| Column | Required for | Meaning inferred from source |
|---|---|---|
| `total` | Part 1 | Continuous regression target |
| `pass` | Parts 2–4 | Binary outcome encoded using `yes` / `no` |
| `name` | Parts 1–2 | Identifier excluded from the general predictor matrix |
| `midterm` | Parts 3–4 | First two-dimensional predictor |
| `final` | Parts 3–4 | Second two-dimensional predictor |

Any additional columns remaining after the exclusions may be used as predictors in Parts 1 and 2.

## Usage

### Google Colab

1. Open a new Google Colab notebook.
2. Upload the Python source or paste its cells into the notebook.
3. Run the setup cell.
4. When prompted, upload `mltak1Data.xlsx`.
5. Execute the remaining cells in order.
6. Inspect the generated tables and figures under `/content/mltak_outputs`.
7. The final cell creates `mltak_all_results.zip` and calls `files.download(ZIP_PATH)` to initiate the Colab download.

### Script Execution

The supplied source is a Python script representation of a Colab workflow. It contains shell commands beginning with `!`, imports from `google.colab`, and automatic file-download logic; therefore, it is not intended as a generic local command-line program without adaptation.

## Configuration

The source does not expose a dedicated configuration object. The principal configurable elements are hard-coded in the script:

| Setting | Source value |
|---|---|
| Expected input filename | `mltak1Data.xlsx` |
| Output directory | `/content/mltak_outputs` |
| Figure directory | `/content/mltak_outputs/figures` |
| Table directory | `/content/mltak_outputs/tables` |
| Text-result directory | `/content/mltak_outputs/text_results` |
| ZIP archive | `/content/mltak_all_results.zip` |
| Figure resolution | 300 DPI |

The scikit-learn estimators are instantiated with their library defaults. No explicit random seed is set for `SGDClassifier`.

## Evaluation

The evaluation procedure is transparent but limited.

### Metrics

For each of the three classifiers in Part 2, the source reports accuracy and precision and stores the confusion matrix as a separate CSV file. These metrics are computed on the same observations used for training.

### Reproducibility of Numerical Results

Exact numerical outputs may vary across environments for several reasons:

1. dependency versions are not pinned;
2. the Colab runtime is not pinned;
3. `SGDClassifier()` is instantiated without an explicit `random_state`;
4. no preprocessing or normalization pipeline is versioned separately;
5. the source depends on the exact contents and ordering of the uploaded workbook.

For stronger reproducibility, a future revision should record the Python version, package versions, random seed, and dataset checksum and should separate training and evaluation data.

## Results

The source code does not contain fixed numerical results; it generates them dynamically from the uploaded workbook. Accordingly, this README does not report numerical accuracy, precision, coefficients, or confusion-matrix counts that cannot be established without executing the supplied program on the actual input dataset.

After execution, the generated artifacts provide the empirical results needed to assess the fitted models.

### Generated Visualizations

The workflow produces five PNG figures:

1. Ridge classifier confusion matrix.
2. SGD classifier confusion matrix.
3. Logistic-regression confusion matrix.
4. Logistic-regression decision-boundary visualization using `midterm` and `final`.
5. Manual matrix-based decision-boundary visualization.

## Reproducibility

The workflow has several reproducibility features:

- the data file is explicitly uploaded into the execution environment;
- output directories are created deterministically;
- numerical result tables are exported to CSV;
- selected results are also exported to Excel;
- generated figures are saved at 300 DPI;
- the input workbook is copied into the output package when possible;
- an inventory of generated files is written to `generated_files.txt`;
- all generated artifacts are collected into a single ZIP archive.

However, reproducibility is incomplete in the strict scientific sense because the environment and estimator randomness are not fully pinned. The manual matrix calculation also uses an explicit matrix inverse rather than a numerically safer pseudoinverse or least-squares solver.

## Limitations

### 1. In-Sample Evaluation

All three classifiers in Part 2 are evaluated on the same observations used for fitting. This can substantially overestimate generalization performance and prevents a scientifically defensible claim about performance on unseen data.

### 2. No Train/Test Split or Cross-Validation

The script does not create validation or test sets and does not perform cross-validation. Model comparison therefore reflects training-set behavior rather than robust model selection.

### 3. No Feature Scaling

The code does not standardize or otherwise rescale features before fitting the classifiers. This is particularly relevant for regularized or gradient-based methods because coefficient magnitudes and optimization behavior can depend on feature scale.

### 4. Default Hyperparameters

All estimators use scikit-learn defaults. The source does not include a documented hyperparameter-search procedure.

### 5. SGD Randomness

`SGDClassifier()` is created without `random_state`. Repeated executions can therefore produce different fitted parameters or predictions, depending on the environment and data.

### 6. Explicit Matrix Inversion

The manual derivation computes $(X^\top X)^{-1}$ explicitly. If $X^\top X$ is singular or poorly conditioned, `np.linalg.inv` can fail or produce numerically unstable results. A pseudoinverse or a linear least-squares solver would generally be more robust.

### 7. Decision-Boundary Plotting Formula

The Part 3 plotting equation uses the first fitted coefficient as the denominator rather than the second coefficient required when solving the fitted score equation for the vertical-axis variable. As documented above, this can make the drawn line differ from the actual logistic-regression boundary.

### 8. Dependence on Column Names

Several steps require exact column names. A workbook with different names or incompatible value formats can fail during execution.

### 9. No Missing-Value Handling

The source does not include explicit missing-value imputation, validation, or outlier handling. Missing or incompatible values may therefore propagate into model fitting or plotting.

### 10. Precision Edge Cases

The source calls `precision_score` without an explicit `zero_division` policy. A classifier that produces no positive predictions can therefore trigger a warning, depending on the data and scikit-learn version.

## Future Work

A stronger research-oriented extension would include:

1. a documented train/validation/test protocol or cross-validation;
2. feature scaling implemented through reproducible pipelines;
3. explicit hyperparameter selection for regularization and SGD optimization;
4. a fixed random seed and pinned dependency versions;
5. robust missing-value and data-quality validation;
6. numerically stable least-squares computation using `np.linalg.pinv` or a dedicated solver;
7. correction and regression testing of the Part 3 decision-boundary equation;
8. additional metrics such as recall, F1 score, ROC-AUC, and calibration analysis where scientifically appropriate;
9. confidence intervals or repeated resampling for performance uncertainty;
10. separation of model-training code, evaluation code, visualization code, and artifact-generation code into testable modules;
11. automated tests for feature selection, target encoding, matrix dimensions, and boundary equations;
12. environment capture through a pinned `requirements.txt` or equivalent reproducible environment specification.

## Research Interpretation

From a research-methodology perspective, the strongest aspect of the project is its explicit comparison of estimator-based machine learning with a transparent matrix formulation. The manual calculation exposes the algebra behind a linear score model and provides a useful bridge between numerical linear algebra and applied machine learning.

At the same time, the current implementation is best viewed as an educational and exploratory analysis rather than a complete empirical study. In particular, the absence of held-out evaluation, the use of default hyperparameters, the absence of feature scaling, and the documented decision-boundary plotting discrepancy limit the strength of any claims about predictive generalization.

For a research submission, the most defensible presentation is to emphasize methodological transparency, implementation of classical algorithms, reproducible artifact generation, and explicit identification of the current limitations.

## References

1. GitHub Docs, *Writing mathematical expressions*. GitHub supports LaTeX-formatted mathematics in Markdown and renders mathematical expressions with MathJax. See the official documentation: https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
2. scikit-learn documentation, *LinearRegression*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
3. scikit-learn documentation, *RidgeClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html
4. scikit-learn documentation, *SGDClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
5. scikit-learn documentation, *LogisticRegression*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
6. scikit-learn documentation, *Model evaluation: scoring and metrics*. https://scikit-learn.org/stable/modules/model_evaluation.html

## Source Provenance

This README documents the supplied source file `mltak_combined_colab(1).py`, a Colab-oriented Python workflow that uploads an Excel workbook, performs the four analyses described above, saves tables and figures, and automatically downloads a ZIP archive of the results. The source is explicitly marked as automatically generated from Colab and retains its original Colab execution structure.

