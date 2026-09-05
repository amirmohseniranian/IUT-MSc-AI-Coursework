# Character-Level Shahnameh Language Modeling with a Deep LSTM

A reproducible TensorFlow/Keras implementation for character-level recurrent language modeling of Persian text from the *Shahnameh*. The program loads a user-provided `shahnameh.txt`, normalizes the Persian Unicode text, creates a deterministic train/validation split, builds a character vocabulary, trains a two-layer LSTM language model, evaluates next-character prediction, and generates Persian text from a seed prompt.

> **Scope.** This README documents the implementation present in `ca4_shahnameh_deep_lstm_local_colab.py`. It does not claim that the generated text is metrically correct Persian poetry, nor does it report experimental values that are not produced by the script.

## Overview

The project represents the Shahnameh as a sequence of characters rather than words or subword tokens. Given a context of 128 character IDs, the model predicts the probability distribution of the next character at each position in the sequence.

The main pipeline is:

1. Load `/content/shahnameh.txt`.
2. Normalize Persian Unicode characters and remove combining marks and non-letter characters outside the retained Arabic/Persian Unicode block.
3. Represent the cleaned source as `PoemRecord` objects and record dataset provenance, including a SHA-256 checksum.
4. Shuffle records with seed `42` and create a `90% / 10%` train/validation split at the record level.
5. Build a character vocabulary with an explicit `<PAD>` token at ID `0`.
6. Encode characters as integer IDs and construct TensorFlow `tf.data.Dataset` objects for next-character prediction.
7. Train an `Embedding -> LSTM -> LSTM -> Dense -> Softmax` model.
8. Monitor validation loss with checkpointing, learning-rate reduction, early stopping, and epoch-end text sampling.
9. Evaluate validation loss and accuracy, derive perplexity, and generate 1,000 requested new characters.
10. Save metrics, training history, plots, model weights, generated text, provenance, and an experiment summary.

## Research Objective

The implementation is a compact experimental study of recurrent neural language modeling on classical Persian text. Its central task is conditional next-character prediction:

Given a character sequence

`x_1, x_2, ..., x_T`,

the network estimates the probability distribution

$P(x_{t+1} \mid x_1, \ldots, x_t)$

for each position `t`.

Character-level modeling is useful for raw text because it avoids the need for an external word-tokenization vocabulary and directly models spelling, character transitions, spaces, and the line structure retained by preprocessing. In this implementation, however, preprocessing is intentionally restrictive: only letters from the Arabic/Persian Unicode block are retained, while spaces are preserved and combining marks are removed.

## Scientific Background

### Character-level language modeling

Let the training vocabulary contain `V` symbols. Each character is mapped to an integer ID:

$x_t \in \{0, 1, \ldots, V-1\}$.

For a sequence of length `T`, the model produces a categorical distribution

$\hat{\mathbf{p}}_t \in [0,1]^V$

for the next character at each position.

The targets are shifted by one position. For an input sequence

`[x_1, x_2, ..., x_T]`

the corresponding target sequence is

`[x_2, x_3, ..., x_T, x_{T+1}]`.

The implementation constructs these pairs with:

```python
ds = ds.map(
    lambda chunk: (chunk[:-1], chunk[1:]),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
)
```

### Sparse categorical cross-entropy

Because the targets are integer character IDs rather than one-hot vectors, the model uses sparse categorical cross-entropy. For a target character `y_t` and predicted probability vector $\hat{\mathbf{p}}_t$:

$\mathcal{L}_t = -\log \hat{p}_t(y_t)$

The training objective is the mean cross-entropy over the evaluated prediction positions.

The code uses `tf.keras.losses.SparseCategoricalCrossentropy()` with the final layer configured as a probability-producing softmax.

### Perplexity

The script derives validation perplexity directly from the reported validation loss:

$\mathrm{PPL} = e^{\mathcal{L}_{\mathrm{val}}}$.

This is the exponential transformation of the cross-entropy used by the implementation.

### Character accuracy

The reported accuracy is sparse categorical accuracy: the fraction of character positions for which the highest-probability predicted class matches the integer target.

This metric is useful for monitoring exact next-character prediction, but it does not fully measure text or literary quality.

## Data and Preprocessing

### Local input

The active execution path expects:

```text
/content/shahnameh.txt
```

If the file is missing in Google Colab, the program opens the Colab upload dialog and accepts a file named `shahnameh.txt` or a single uploaded file.

The source is read as UTF-8 with an optional UTF-8 byte-order mark:

```python
raw_text = path.read_text(encoding="utf-8-sig")
```

### Persian text normalization

`normalize_persian_text()` performs the following:

* Removes the Unicode byte-order mark.
* Normalizes line endings to `\n`.
* Applies Unicode NFC normalization.
* Maps several Arabic-character variants to Persian or normalized forms, including `ي -> ی`, `ك -> ک`, and related forms.
* Replaces the zero-width non-joiner with a regular space.
* Removes left-to-right and right-to-left marks.
* Removes Unicode combining marks.
* Preserves ordinary spaces.
* Retains letters whose code points fall within `U+0600` through `U+06FF`.
* Collapses repeated spaces and removes empty lines.

Therefore, the model does **not** train on the complete raw Unicode source. The training corpus is the normalized output of this function.

### Record construction

The local loader creates one `PoemRecord` per non-empty normalized source line:

| Field            | Meaning                         |
| ---------------- | ------------------------------- |
| `id`             | Sequential local identifier     |
| `title`          | `Local Shahnameh line <id>`     |
| `full_url`       | Path of the local source file   |
| `metre`          | Empty for the local source path |
| `couplets_count` | `1` for line-based records      |
| `text`           | Normalized line text            |

If fewer than two records are created, the program falls back to chunking the normalized text into blocks of at least `SEQ_LENGTH + 1` characters, with a minimum chunk size of `5000`.

The original Ganjoor acquisition helpers remain in the source file for fidelity, but they are not part of the active data path in this local Colab workflow.

### Reproducible train/validation split

The records are copied and shuffled using:

```python
rng = random.Random(SEED)
rng.shuffle(shuffled)
```

with `SEED = 42`.

The requested split is:

```text
Training:   90%
Validation: 10%
```

The split is based on the number of records, not directly on character counts. If it would produce an empty validation set, the last training record is moved to validation.

Train and validation corpora are then created by joining their respective record texts with newline characters.

## Methodology

### Sequence Construction

The implementation uses:

* Sequence length: `128`
* Batch size: `128`

The number of complete non-overlapping sequences is computed as:

$N_{\mathrm{seq}} = \left\lfloor \frac{N}{L+1} \right\rfloor$

where `N` is the number of encoded characters and `L = 128`.

Only the first `N_seq * (L + 1)` characters are used for sequence construction; any shorter remainder is discarded.

Each chunk has length `129`; the final character is then shifted into the target:

```text
input:  x_1 ... x_128
target: x_2 ... x_129
```

Training batches are shuffled with a deterministic seed, repeated indefinitely, and consumed through the calculated `steps_per_epoch`. Validation data is not repeated and is evaluated over `ceil(N_seq / batch_size)` batches.

### Model Architecture

The model is named `Shahnameh_Deep_LSTM_LM` and uses the following structure:

```text
Integer character IDs
        |
        v
Embedding(vocab_size, 256, mask_zero=True)
        |
        v
SpatialDropout1D(0.15)
        |
        v
LSTM(512, return_sequences=True)
        |
        v
Dropout(0.25)
        |
        v
LSTM(512, return_sequences=True)
        |
        v
Dropout(0.25)
        |
        v
Dense(256, activation="gelu")
        |
        v
Dense(vocab_size, activation="softmax", dtype="float32")
```

### Embedding

Each character ID is mapped to a learned vector in a 256-dimensional embedding space.

The layer uses:

```python
mask_zero=True
```

because ID `0` is reserved for `<PAD>`.

### Recurrent stack

The network contains two LSTM layers with 512 recurrent units each. Both use `return_sequences=True`, so the full sequence of hidden states is passed to the next stage.

The source explicitly sets `dropout=0.0` and `recurrent_dropout=0.0` in both LSTM layers. Additional dropout is applied between the recurrent layers with standard `Dropout` layers.

### Projection and output

The second LSTM is followed by a 256-unit dense projection with GELU activation and then a vocabulary-sized softmax classifier.

At each sequence position `t`, the final layer produces:

$\hat{\mathbf{p}}_t = \operatorname{softmax}(\mathbf{z}_t)$

where $\mathbf{z}_t \in \mathbb{R}^V$ is the vocabulary logit vector and `V` is the character vocabulary size.

The output layer explicitly uses `dtype="float32"` so that the probability output remains in float32 when mixed precision is active.

### Optimization and Training

The optimizer is Adam with the following settings:

| Hyperparameter        |   Value |
| --------------------- | ------: |
| Initial learning rate | `0.002` |
| Gradient clip norm    |   `1.0` |
| Requested epochs      |    `10` |
| Batch size            |   `128` |
| Sequence length       |   `128` |
| Embedding dimension   |   `256` |
| LSTM units per layer  |   `512` |
| Inter-layer dropout   |  `0.25` |
| Spatial dropout       |  `0.15` |

The model is compiled with:

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=INITIAL_LR,
    clipnorm=1.0,
)
```

#### Callbacks

Three standard training callbacks are configured.

**Model checkpointing**

The best weights are saved according to minimum validation loss:

```python
monitor="val_loss"
mode="min"
save_best_only=True
save_weights_only=True
```

**Learning-rate reduction**

If validation loss does not improve for two epochs, the learning rate is multiplied by `0.5`, with a minimum of `1e-5`.

**Early stopping**

Training stops after four epochs without an improvement in validation loss, and the best weights are restored.

#### Epoch-end generation

`SamplePoemCallback` generates a short sample after each epoch using:

```text
به نام خداوند جان و خرد
```

using generation length `200`, temperature `0.75`, `top_k=20`, and `top_p=0.90`.

This callback is diagnostic. Samples printed during training are not used as training labels or as a formal evaluation benchmark.

### Text Generation

The final generation call requests `1000` **newly sampled characters** in addition to the seed prompt, using:

| Parameter                          |                     Value |
| ---------------------------------- | ------------------------: |
| Seed prompt                        | `به نام خداوند جان و خرد` |
| Requested newly sampled characters |                    `1000` |
| Temperature                        |                    `0.72` |
| Top-k                              |                      `20` |
| Top-p                              |                    `0.88` |

At each generation step, the latest `128` token IDs form the model context. If the seed is shorter than 128 characters, it is left-padded with the `<PAD>` token.

The probability assigned to `<PAD>` is then set to zero before sampling.

#### Temperature scaling

For a positive, non-unit temperature, the implementation applies:

$p_i' \propto \exp\left(\frac{\log p_i}{\tau}\right)$

where `p_i` is the original softmax probability and $\tau$ is the temperature.

A lower temperature concentrates probability mass around high-probability characters; a higher temperature produces a flatter distribution.

#### Top-k filtering

When `top_k > 0` and is smaller than the vocabulary size, only the `top_k` highest-probability candidates are retained before renormalization.

#### Top-p filtering

The remaining probabilities are sorted in descending order. Candidates whose cumulative probability exceeds `top_p` are removed, except that the highest-probability candidate is always retained. The resulting distribution is then renormalized.

## Mathematical Details

The implementation can be summarized by four core mathematical operations.

### Next-character objective

For an input sequence of character IDs $x_1, \ldots, x_T$, the model estimates a probability vector for the next character at each position:

$\hat{\mathbf{p}}_t = P(x_{t+1}\mid x_1,\ldots,x_t)$.

The sparse categorical cross-entropy used by Keras is:

$\mathcal{L}_t = -\log \hat{p}_t(y_t)$,

where $y_t$ is the integer ID of the target character. The reported validation loss is the mean cross-entropy over the evaluated prediction positions.

### Softmax output

The final dense layer converts logits $z_{t,i}$ into a probability distribution over the `V` vocabulary symbols:

$\hat{p}_{t,i} = \frac{\exp(z_{t,i})}{\sum_{j=1}^{V}\exp(z_{t,j})}$.

Thus, each sequence position produces `V` class probabilities.

### Perplexity

The script reports:

$\mathrm{PPL}=\exp(\mathcal{L}_{\mathrm{val}})$,

implemented as `exp(min(val_loss, 20.0))` to cap the exponent for numerical safety.

### Sampling temperature

For temperature $\tau>0$, the sampler transforms probabilities according to:

$p_i' \propto \exp\left(\frac{\log p_i}{\tau}\right)$.

After renormalization, top-k and top-p filtering are applied before the next character is sampled.

These equations describe the operations used by the source; they do not introduce an additional training objective or evaluation criterion.

## Evaluation

After training, the best saved weights are reloaded when available, and validation performance is computed using:

```python
eval_results = model.evaluate(
    val_ds,
    steps=validation_steps,
    verbose=0,
    return_dict=True,
)
```

The reported metrics are:

| Metric                       | Definition in the implementation                           |
| ---------------------------- | ---------------------------------------------------------- |
| Validation loss              | Sparse categorical cross-entropy                           |
| Validation accuracy          | Sparse categorical accuracy                                |
| Validation perplexity        | `exp(validation_loss)` with an exponent cap of `20`        |
| 4-gram repetition rate       | `1 - unique_ngrams / total_ngrams`                         |
| 4-gram novelty vs. train     | Fraction of generated 4-grams absent from training text    |
| Generated line count         | Number of non-empty generated lines                        |
| Generated line-length mean   | Mean character length of non-empty generated lines         |
| Generated line-length std    | Standard deviation of those lengths                        |
| Generated line-length median | Median of those lengths                                    |
| Training unique characters   | Number of distinct non-newline characters in training text |

### Interpretation of auxiliary generation metrics

The repetition statistic is a lexical-diversity diagnostic over character 4-grams. It is not a semantic repetition metric.

The novelty statistic measures whether generated character 4-grams occurred in the training text. A higher novelty value does not by itself imply better generalization or literary quality.

Line-length statistics summarize only the output's line structure. The program explicitly warns that these diagnostics are **not** a phonological scansion system and cannot establish that the generated text follows classical Persian metre.

## Reproducibility

The implementation sets the following:

```python
SEED = 42
```

The seed is applied to Python's `random`, NumPy, and TensorFlow/Keras. The implementation also sets:

```text
PYTHONHASHSEED=42
TF_DETERMINISTIC_OPS=1
TF_CUDNN_DETERMINISTIC=1
```

GPU memory growth is enabled when a GPU is detected. When a GPU is available, the script attempts to use TensorFlow mixed precision; if that setup fails, it falls back to float32.

The implementation therefore supports reproducibility, but exact bitwise identity can still depend on the execution environment, TensorFlow build, GPU hardware, and underlying kernels.

## Installation

The script can install missing runtime dependencies itself. The program specifies:

```bash
pip install "tensorflow>=2.15" numpy pandas matplotlib
```

A typical environment can be prepared with:

```bash
python -m pip install "tensorflow>=2.15" numpy pandas matplotlib
```

The code is organized primarily for Google Colab, where `/content` is used as the default working directory when available.

## Usage

### Google Colab

1. Upload `shahnameh.txt` to the Colab runtime.
2. Run `ca4_shahnameh_deep_lstm_local_colab.py`.
3. If the expected file is missing, the script attempts to open the Colab upload dialog.
4. The trained outputs are written to:

```text
/content/shahnameh_lstm_outputs/
```

The primary input filename expected by the active path is:

```text
/content/shahnameh.txt
```

### Local execution

Outside Colab, the script falls back to the current working directory for its base output directory, but the active loader is still called with:

```python
load_local_shahnameh_dataset("/content/shahnameh.txt")
```

Therefore, a non-Colab run requires adapting the input path in the source before execution. The source does not expose a command-line argument for the input file.

## Outputs

The program writes the following artifacts under `shahnameh_lstm_outputs`:

| File                                     | Purpose                                                     |
| ---------------------------------------- | ----------------------------------------------------------- |
| `raw_ganjoor_cache/`                     | Cache location for the retained Ganjoor acquisition helpers |
| `shahnameh_ganjoor_dataset.jsonl`        | JSON Lines representation of local `PoemRecord` records     |
| `shahnameh_cleaned.txt`                  | Concatenated train and validation text                      |
| `train_text.txt`                         | Training corpus                                             |
| `validation_text.txt`                    | Validation corpus                                           |
| `vocab_metadata.json`                    | Vocabulary, `char_to_id`, and `id_to_char` metadata         |
| `dataset_provenance.json`                | Source metadata and SHA-256 checksum                        |
| `split_metadata.json`                    | Record and character counts for the split                   |
| `best_shahnameh_lstm.weights.h5`         | Best model weights according to validation loss             |
| `shahnameh_generated_1000_chars.txt`     | Final generated text                                        |
| `training_history.csv`                   | Per-epoch Keras training history                            |
| `evaluation_metrics.csv`                 | Final evaluation and generation diagnostics                 |
| `training_evaluation_plots.png`          | Training/validation loss and accuracy plots                 |
| `experiment_report.json`                 | Machine-readable experiment summary                         |
| `README.txt`                             | Automatically generated output-directory note               |
| `CA4_Shahnameh_RNN_Complete_Outputs.zip` | ZIP archive containing generated files                      |

The code does not save the full Keras model architecture as a standalone model file; it saves only the model weights.

## Implementation Details

### Core source components

| Component                            | Role                                             |
| ------------------------------------ | ------------------------------------------------ |
| `PoemRecord`                         | Immutable record representation for text samples |
| `normalize_persian_text`             | Persian Unicode normalization and filtering      |
| `load_local_shahnameh_dataset`       | Local file loading and provenance generation     |
| `build_training_and_validation_text` | Deterministic record-level split                 |
| `build_vocabulary`                   | Character vocabulary construction                |
| `encode_text`                        | Character-to-ID encoding                         |
| `create_tf_dataset`                  | Next-character TensorFlow dataset construction   |
| `build_shahnameh_deep_lstm`          | LSTM language-model construction and compilation |
| `sample_token`                       | Temperature/top-k/top-p sampling                 |
| `generate_shahnameh_poetry`          | Autoregressive character generation              |
| `repetition_rate`                    | Character 4-gram repetition diagnostic           |
| `ngram_novelty`                      | Training-corpus 4-gram novelty diagnostic        |
| `line_length_statistics`             | Generated-line length summary                    |
| `SamplePoemCallback`                 | Epoch-end generation callback                    |

### Data-flow summary

```text
shahnameh.txt
     |
     v
Unicode normalization
     |
     v
Local PoemRecord objects
     |
     +--------------------+
     |                    |
     v                    v
train_text           validation_text
     |                    |
     v                    v
character IDs        character IDs
     |                    |
     v                    v
129-char chunks     129-char chunks
     |                    |
     v                    v
(X_t, X_t+1)        (X_t, X_t+1)
     |                    |
     +---------+----------+
               |
               v
Embedding
   -> LSTM
   -> LSTM
   -> Dense(GELU)
   -> Softmax
               |
               v
     next-character distribution
               |
        +------+------+
        |             |
        v             v
 validation      autoregressive
 metrics         generation
```

## Technical Notes and Audit Findings

### The active dataset source is local, not Ganjoor download

The file contains the original Ganjoor acquisition helpers, including HTTP retrieval, recursive category traversal, caching, and poem parsing. However, the executed pipeline calls `load_local_shahnameh_dataset("/content/shahnameh.txt")`, so those download helpers are not used by the active dataset path.

This distinction matters for reproduction: the local Shahnameh file is an external input and is not embedded in the Python source.

### The train/validation split is record-based

The split is performed before the training and validation strings are concatenated. In the active local representation, records normally correspond to non-empty source lines. Thus, validation data is held out at the record level rather than with a random character-level mask.

### Sequence windows are non-overlapping

The dataset builder uses consecutive chunks of `seq_len + 1` characters rather than a one-character sliding window. This reduces the number of training examples compared with an overlapping-window construction and discards any remainder that does not fill a complete chunk.

### `<PAD>` is structurally supported

`<PAD>` is assigned ID `0`, matching `mask_zero=True`. Padding is used when the generation context is shorter than 128 characters. The generation procedure also explicitly sets the `<PAD>` probability to zero before sampling.

### The vocabulary is built from the combined corpus

The vocabulary is constructed from:

```python
full_text = train_text + "\n" + val_text
```

As a result, character identities that appear only in validation text can still be included in the vocabulary. The model therefore has those validation characters in its output space even though the validation sequences are not used to optimize the weights.

### Validation perplexity is derived from reported loss

The code computes:

```python
val_perplexity = float(math.exp(min(val_loss, 20.0)))
```

The `min(..., 20.0)` cap prevents numerical overflow for extremely large losses. For finite losses below `20`, this is exactly `exp(val_loss)`.

### Generated length is a requested length, not the total output length

The generator starts with the normalized seed prompt and then performs `gen_length` sampling iterations. Thus, the requested `1000` characters are newly sampled characters appended to the seed, not the total output length. The resulting file length is stored separately as `actual_length`.

## Results Reporting

The source code is prepared to report results after execution, including:

* Validation cross-entropy loss
* Validation character accuracy
* Validation perplexity
* 4-gram repetition rate
* 4-gram novelty relative to training text
* Generated character counts
* Generated line-length statistics
* Number of completed epochs
* GPU names and mixed-precision policy

No numerical result values are embedded in this README because the supplied source code contains the computation logic rather than precomputed experiment results.

After a run, the authoritative machine-readable values are stored in:

```text
experiment_report.json
evaluation_metrics.csv
training_history.csv
```

## Limitations

1. **Character-level representation.** The model does not use linguistic word, morpheme, or subword units, so semantic and long-range linguistic structure must be learned indirectly from characters.

2. **Limited corpus controls.** The supplied program accepts a local `shahnameh.txt` file whose exact contents are external to the repository code. Corpus edition, transcription choices, completeness, duplication, and provenance beyond the recorded SHA-256 checksum are not established by the source alone.

3. **Preprocessing removes information.** Combining marks are removed, several Unicode characters are canonicalized, and retained text is restricted to letters in the `U+0600`-`U+06FF` block plus spaces. Characters outside that policy are not represented.

4. **Non-overlapping sequence segmentation.** The training-data construction discards incomplete remainders and does not use sliding windows.

5. **Accuracy is incomplete as a text-quality metric.** High next-character accuracy does not imply grammatical, semantic, stylistic, or poetic quality.

6. **Perplexity is corpus- and tokenizer-dependent.** It should be interpreted only in the context of this exact character vocabulary and preprocessing pipeline.

7. **Generation diagnostics are limited.** Repetition, novelty, and line-length statistics are heuristic diagnostics rather than standardized measures of literary quality.

8. **No formal metre evaluation.** The implementation does not perform syllabic or prosodic scansion. Its line-length statistics cannot establish conformity to classical Persian metre.

9. **No external benchmark.** The script does not compare this model with alternative architectures, tokenization strategies, baselines, or established language-model benchmarks.

10. **Hardware-dependent execution.** Runtime behavior and exact reproducibility can vary across TensorFlow versions, CPUs, GPUs, and mixed-precision environments.

## Future Work

The current implementation provides a clear baseline for recurrent Persian language modeling. Scientifically motivated extensions include:

* Compare character-level modeling with subword tokenization.
* Replace non-overlapping chunks with controlled overlapping windows.
* Evaluate larger or better-curated Shahnameh corpora with explicit edition metadata.
* Add held-out and cross-validation protocols designed to avoid stylistic or textual overlap.
* Compare the two-layer LSTM with GRU, Transformer, and modern causal language-model baselines.
* Evaluate generation using human judgments for fluency, coherence, style, and faithfulness to Persian literary conventions.
* Introduce formal prosodic analysis rather than using line length as a proxy for metre.
* Track parameter counts, training time, hardware, TensorFlow version, and dataset identifiers in an experiment registry.
* Add automated tests for normalization, vocabulary construction, sequence alignment, probability filtering, and output serialization.

