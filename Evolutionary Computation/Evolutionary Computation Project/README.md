# MATLAB-Reconstructed Evolutionary Optimization Workflows

> Faithful Python reconstruction of the supplied MATLAB Evolution Strategy (ES) and Traveling Salesman Problem (TSP) workflows, with deterministic validation, runtime metadata, machine-readable result export, convergence plots, and portable packaging.

## Overview

This project reconstructs four evolutionary-optimization workflows from the supplied MATLAB archive in Python:

1. **Evolution Strategy — main single run** for continuous optimization on the Ackley objective.
2. **Evolution Strategy — repeated runs** for stochastic replication and iteration-to-threshold analysis.
3. **Traveling Salesman Problem — main single run** using permutation-based evolutionary search.
4. **Traveling Salesman Problem — repeated runs** for repeated stochastic evaluation.

The implementation is intended for local execution or Google Colab. It preserves the algorithmic choices represented in the supplied workflow while adding explicit random seeding, validation routines, runtime metadata, JSON/CSV exports, convergence plots, route validation, an exact small-instance TSP verifier, and a results manifest/ZIP export.

The source file identifies the original Colab notebook as the reconstruction source. The Python file is a notebook-to-script export rather than a manually structured standalone package. [Original Colab notebook](https://colab.research.google.com/drive/1BrJ2bZkU0lZNIPNcw_snyWGGHXRhwVQx)

## Scientific scope

The repository is primarily a **reconstruction and reproducibility artifact**. It should not be interpreted as a claim of algorithmic novelty or as a benchmark study. The implemented methods are classical evolutionary optimization mechanisms: self-adaptive mutation strategies and recombination for ES, together with permutation crossovers, mutation operators, fitness-based selection, and survivor selection for TSP.

No performance claim is made beyond the executions produced by the supplied implementation under explicitly stated configurations and seeds.

## Motivation

The project addresses a practical research-engineering problem: reconstructing stochastic MATLAB optimization workflows in Python without silently changing the modeled algorithms, while keeping the resulting experiments auditable and reproducible.

The reconstruction emphasizes four properties:

- **Algorithmic fidelity:** operator choices, parameter conventions, stopping rules, and source-level quirks are preserved where identifiable.
- **Reproducibility:** random generators are seeded and runtime metadata are exported.
- **Validation:** optimization outputs are checked for finite values, expected dimensions, valid tours, and small-instance exact consistency.
- **Portability:** workbook-oriented outputs are replaced by JSON/CSV artifacts and standard image files suitable for local or Colab execution.

## Theoretical foundations

### 1. Evolution Strategy for continuous optimization

An ES maintains candidate vectors in a continuous search space. Let a candidate solution be

$$
x = (x_1, x_2, \ldots, x_d) \in \mathbb{R}^d.
$$

The implementation attaches a mutation-parameter state to every population member. Depending on the configured mutation family, this state is either a scalar, a vector of component-wise step sizes, or a dictionary containing step sizes and an angle matrix.

At each generation, the implementation performs:

1. parent sampling,
2. crossover/recombination,
3. mutation with probability `pm`,
4. objective evaluation,
5. survivor selection,
6. convergence recording,
7. threshold-based early stopping.

The number of children is

$$
\lambda = m\,\mu,
$$

where $\mu$ is the population size and $m$ is `nc_multiplier`.

The code supports two survivor modes corresponding to the classical $(\mu + \lambda)$ and $(\lambda)$ forms. The internal identifiers are retained exactly as they appear in the source: `landa+mua` and `landa_va_mua`.

### 2. Ackley objective

The main ES workflow uses the standard Ackley function in dimension $d$:

$$
f(x) = -20\exp\left(-0.2\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}\right) - \exp\left(\frac{1}{d}\sum_{i=1}^{d}\cos(2\pi x_i)\right) + 20 + e.
$$

The global minimum of this canonical objective is $0$ at the origin. The implementation uses the direct NumPy translation of this expression.

### 3. Alternative Rosenbrock objective

The source also contains an alternative objective named `rosenbrock_shifted_200`:

$$
f(x) = 200 + \sum_{i=1}^{d-1}\left[100\left(x_{i+1}-x_i^2\right)^2 + \left(x_i-1\right)^2\right].
$$

This function is implemented but is not used by the default workflow functions.

### 4. TSP objective

A TSP solution is represented as a permutation of city identifiers. For a route $r=(r_1,\ldots,r_n)$ and distance matrix $D$, the tour cost is the closed-cycle sum

$$
C(r) = \sum_{i=1}^{n} D_{r_i,r_{i+1}},
\qquad r_{n+1}=r_1.
$$

The implementation converts cost into a maximization fitness using

$$
F(r) = \frac{1}{C(r)}.
$$

A distance entry equal to `-1` is treated as an unavailable edge and replaced by an infinite traversal cost.

## Methodology

### Evolution Strategy representation

Each ES individual is represented by three fields:

| Field | Meaning |
|---|---|
| `variable` | Continuous decision vector. |
| `parameter` | Self-adaptive mutation state. |
| `cost` | Objective value to be minimized. |

Initial variables are sampled independently and uniformly from `[var_min, var_max]`.

A numerical floor of `0.01` is applied to nonnegative mutation step sizes through the helper `_soft_floor`.

### Mutation families

The source implements three mutation families.

#### Uncorrelated one-step mutation

A single step size is maintained for the entire vector. The log-normal scaling factor uses

$$
\tau = \frac{1}{\sqrt{d}}.
$$

The actual implementation samples the log-normal factor with `sigma=tau**2` and applies the scaled step size to an isotropic Gaussian covariance matrix.

#### Uncorrelated n-step mutation

A common log-normal factor and independent component-wise factors are used. The code uses

$$
\tau = \frac{1}{\sqrt{2\sqrt{d}}},
\qquad
\tau' = \frac{1}{\sqrt{2d}}.
$$

The resulting covariance is diagonal, with one mutation scale per coordinate.

#### Correlated mutation state

The correlated mode maintains both a vector of step sizes and an antisymmetric angle matrix. The angle matrix is updated using a small rotation scale of $5$ degrees and wrapped to the interval $[-\pi,\pi]$.

A critical source-fidelity detail is that the implementation does **not** construct a rotated covariance matrix from the angle state. It samples from `diag(sigma)`. The stored `alpha` state is therefore updated and exported, but it does not affect the Gaussian covariance used to perturb the decision vector. This is a characteristic of the supplied reconstruction, not a claim about a general correlated-ES formulation.

### ES crossover modes

Four crossover modes are implemented:

- `discrete`: component-wise parent selection for the decision vector and mutation parameters.
- `avg`: arithmetic averaging for the decision vector and mutation parameters.
- `avg_discrete`: arithmetic averaging for the decision vector with discrete parameter recombination.
- `discrete_avg`: discrete decision-vector recombination with averaged mutation parameters.

Crossover is attempted with probability `pc`. Parent pairs are drawn independently from the current population.

### ES mutation and bounds

When mutation fires, a Gaussian perturbation is sampled from the covariance constructed by the selected mutation family. The candidate is then projected component-wise into the configured bounds using clipping:

$$
 x'_i = \min\left(\max\left(x_i + \varepsilon_i,\,x_{\min}\right),\,x_{\max}\right),
$$

where $\varepsilon$ is the sampled Gaussian perturbation. The objective is recomputed after mutation.

### ES survivor selection

For `landa+mua`, the next population is selected from parents plus children. For `landa_va_mua`, only the children are eligible. In either case, candidates are sorted by objective value and the best $\mu$ are retained.

The ES implementation therefore performs deterministic ranking at the survivor-selection stage once the stochastic offspring have been generated.

## TSP implementation

### Instance loading

The code supports two data paths:

1. **TSPLIB-like coordinate files**, parsed by `parse_tsplib`.
2. **Text distance matrices**, loaded by `load_distance_matrix_text`.

The TSPLIB parser extracts `NAME`, `DIMENSION`, `EDGE_WEIGHT_TYPE`, and coordinates from `NODE_COORD_SECTION`.

For coordinate instances, the implementation computes the pairwise distance matrix directly as Euclidean distance:

$$
D_{ij} = \sqrt{(x_i-x_j)^2 + (y_i-y_j)^2}.
$$

The stored `EDGE_WEIGHT_TYPE` is not used to select among alternative TSPLIB distance conventions. Consequently, this reconstruction should be interpreted as using **raw Euclidean distances computed from the parsed coordinates**, rather than the rounded distance rule associated with every TSPLIB instance type.

### Permutation crossovers

The following permutation-preserving crossovers are implemented:

- **PMX** (`pmx`): partially mapped crossover with two cut positions.
- **Order-1 crossover** (`Order_1_crossover`): preserves one contiguous segment and fills remaining positions from the other parent in cyclic order.
- **Cycle crossover** (`Cycle_crossover`): decomposes the parent pair into position cycles and alternates parental contribution.
- **MOX** (`MOX`): preserves a parent prefix and appends the remaining cities in the order induced by the opposite parent.

Crossover is skipped when a random draw is not below `pc`, in which case the parents are copied.

### TSP mutation operators

Four route mutations are implemented:

- `swap`: exchange two cities.
- `insert`: move one selected city within the selected segment.
- `inversion`: reverse the selected segment.
- `scramble`: randomly shuffle the selected segment.

Two cut positions are sampled without replacement for the segment-based operators.

### TSP parent selection

The implementation provides four selection modes:

| Source identifier | Selection mechanism |
|---|---|
| `tournoment` | Tournament over a random half of the current population; the maximum-fitness member wins. |
| `roletwheel` | Numerically stabilized exponential weighting of fitness values. |
| `trancation` | Random choice among the top half of the population by fitness. |
| `fullrandom` | Uniform random selection. |

The misspelled identifiers are preserved because they are part of the source interface and are required for compatibility.

### TSP survivor selection

The source supports `landa+mua` and `landa_va_mua`. In addition, if fewer children than parents are available, the implementation falls back to the combined parent-and-child pool.

After constructing the eligible pool, the next population is built by repeatedly calling the configured selection method. Therefore, this stage is not equivalent to simple deterministic top-$\mu$ truncation: stochastic selection can choose the same high-fitness region repeatedly, depending on the configured selection operator.

## Workflow inventory

| Workflow | Purpose | Main configuration in source | Fast-validation configuration |
|---|---|---|---|
| 1 | ES main single run | $\mu=50$, $d=10$, 200,000 iterations, `pc=0.3`, `pm=0.9`, bounds `[-10,10]` | $\mu=10$, $d=5$, 100 iterations, bounds `[-30,30]` |
| 2 | ES repeated runs | 20 runs, same bounded ES configuration used by the source | 20 runs, 100-iteration cap |
| 3 | TSP main single run | 50 individuals, 10,000 iterations, 150 children, `pc=0.8`, `pm=0.2`, Order-1 + swap | 10 individuals, 80 iterations, 20 children |
| 4 | TSP repeated runs | 50 individuals, 200 iterations, 24 children, Order-1 + inversion | 10 individuals, 80 iterations, 24 children |

The workflow inventory is stored directly in `WORKFLOW_INVENTORY` and exported to JSON.

## Configuration

### ES defaults

```text
n_population = 10
var_length = 5
iteration = 100
pc = 0.3
pm = 0.9
var_min = -30.0
var_max = 30.0
nc_multiplier = 7
crossover_mode = "avg_discrete"
survivor_mode = "landa_va_mua"
mutation_method = "correlated"
threshold = 0.1
```

The source-level original configuration exposed by `run_workflows` is:

```text
n_population = 50
var_length = 10
iteration = 200000
pc = 0.3
pm = 0.9
var_min = -10
var_max = 10
nc_multiplier = 7
crossover_mode = "avg_discrete"
survivor_mode = "landa_va_mua"
mutation_method = "correlated"
threshold = 0.1
```

### TSP defaults

```text
n_population = 50
iterations = 1000
pc = 0.8
pm = 0.2
n_children = 150
crossover_mode = "Order_1_crossover"
mutation_mode = "swap"
selection_mode = "tournoment"
survivor_mode = "landa_va_mua"
stop_at_fitness = None
```

The source-level original main-workflow configuration uses 50 individuals, 10,000 iterations, 150 children, Order-1 crossover, swap mutation, tournament selection, and `landa_va_mua` survivor selection.

## Reproducibility

A NumPy `Generator` is seeded separately for each workflow run using `np.random.default_rng(seed)`. The repeated ES workflow uses `seed + i` for run index `i`, and the repeated TSP workflow follows the same pattern when the loop performs multiple executions.

The utility `set_global_seed` also seeds Python's standard `random` module and NumPy's legacy global RNG. Runtime metadata record the following:

- Python version
- platform information
- NumPy version
- pandas version
- matplotlib version
- optional SciPy version
- optional PyTorch version
- CUDA availability
- GPU name, when detectable
- experiment seed

This metadata is written to `summary/runtime_metadata.json`.

Reproducibility is therefore **seed-aware but does not guarantee bitwise-identical results across all machines and software stacks**. Differences in Python, NumPy, BLAS backends, operating systems, or plotting libraries may still affect execution details.

## Installation

### Requirements

The supplied script checks for and installs the following core dependencies when they are missing:

- `numpy`
- `pandas`
- `matplotlib`

SciPy and PyTorch are detected opportunistically for runtime metadata but are not required by the implemented workflows.

A modern Python 3 installation is required because the source uses type-annotation syntax such as `list[...]` and union types with `|`.

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Important source-file execution note

The supplied file is an exported notebook script and currently places

```python
from __future__ import annotations
```

after earlier module-level string cells. Python requires future imports to appear before ordinary executable statements. Consequently, the uploaded `.py` file is **not directly compilable in its current exported form**.

To execute the file as a standalone script, move `from __future__ import annotations` to the top of the module, immediately after the encoding declaration and module docstring, or remove it when using a Python version that already supports the annotation syntax used here.

This README documents the supplied implementation as written and does not silently alter that source-level issue.

## Usage

### Google Colab

The source was designed for Colab and contains a conditional download path for the final ZIP archive.

At a minimum, the intended final execution flow is:

```python
summary = run_workflows(BASE_DIR, OUTPUT_ROOT, SEED, mode="validation")
zip_path, manifest_path, manifest = collect_and_zip_results(
    OUTPUT_ROOT,
    BASE_DIR,
    manifest_extra=summary,
)
```

When executed in Colab, the script uses the Colab file-download API to retrieve the generated ZIP archive.

### Local execution of workflow functions

The workflow functions can be called directly from Python after importing or cleaning the notebook-export syntax as described above:

```python
from pathlib import Path

seed = 12345
result = run_es_workflow1(seed=seed, fast_validation=True)
print(result["best_cost"])

six_city_distance = ...
tsp_result = run_tsp_workflow3(
    six_city_distance,
    seed=seed,
    fast_validation=True,
)
print(tsp_result["best_cost"])
```

For a supplied TSPLIB-like coordinate file:

```python
instance = parse_tsplib("path/to/instance.tsp")
distance = euclidean_distance_matrix(instance.coordinates)
result = run_tsp_workflow3(distance, seed=12345, fast_validation=True)
```

## Validation framework

The source contains an explicit validation pipeline in `make_validation_results`.

### ES validation

The pipeline smoke-tests all three mutation families:

- `uncorrelated_1step`
- `uncorrelated_nstep`
- `correlated`

Each is executed for 30 iterations with a five-dimensional population. The checks verify finite best costs and the expected variable dimension.

### Exact small-instance TSP validation

The project defines a six-city distance matrix whose optimal cycle cost is obtained by exhaustive enumeration. The exact optimizer is deliberately restricted to instances with at most 10 cities because the search grows factorially.

The six-city validation then checks both:

- route validity, including permutation completeness and duplicate count;
- consistency between the exact optimum and the evolutionary result.

### Supplied-output forensic validation

When the original auxiliary files are present, `validate_supplied_tsp_outputs` compares saved routes and costs against the Euclidean distance convention used by this reconstruction. The check is optional because those archive files are not treated as hidden inputs to the standalone script.

## Validation results from the supplied implementation

A bounded validation run with seed `12345` was executed against the reconstructed functions after correcting only the notebook-export placement of the future import. The resulting checks were:

| Check | Result |
|---|---|
| ES mutation-family smoke validation | Validated; all three mutation families produced finite results with five-dimensional variables. |
| Workflow 1 ES main | Validated; 91 iterations executed, best cost approximately `0.08594946`. |
| Workflow 2 ES repeated | Validated; 20 runs, mean iterations `69.25`, 19 runs reached the threshold. |
| Six-city exact TSP consistency | Validated; exact optimum cost `6.0`, observed best cost `6.0`, route was a valid permutation. |
| Workflow 3 TSP main | Validated on the six-city validation matrix; best cost `6.0`. |
| Workflow 4 TSP repeated | Validated on the six-city validation matrix; 30 runs, mean iterations `80.0`. |
| Supplied TSP output-file forensic validation | Not validated in the standalone test because the original auxiliary archive files were not present in the execution directory. |

These figures are **execution results reproduced from this reconstructed implementation under the stated validation setup**, not general claims about algorithmic performance.

## Inputs and outputs

### Inputs

The workflows use:

- numerical objective functions for ES;
- in-memory distance matrices for TSP;
- optional TSPLIB-style coordinate files;
- optional text files containing distance matrices;
- a deterministic integer seed.

The source's example validation matrix is:

$$
D =
\begin{pmatrix}
0 & 1 & 10 & 10 & 10 & 1 \\
1 & 0 & 1 & 10 & 10 & 10 \\
10 & 1 & 0 & 1 & 10 & 10 \\
10 & 10 & 1 & 0 & 1 & 10 \\
10 & 10 & 10 & 1 & 0 & 1 \\
1 & 10 & 10 & 10 & 1 & 0
\end{pmatrix}.
$$

### ES outputs

`run_es` returns:

- configuration,
- seed,
- objective name,
- executed iteration count,
- best initial cost,
- best final cost,
- best decision vector,
- optional objective history,
- optional best-vector history.

### TSP outputs

`run_tsp` returns:

- configuration,
- seed,
- instance dimension,
- initial best route,
- executed iteration count,
- best fitness,
- best tour cost,
- best route,
- optional fitness history,
- optional cost history,
- optional best-route history,
- elapsed wall-clock time.

## Result artifacts

The export layer produces a structured results directory and a consolidated ZIP archive.

Typical artifact types include:

| Artifact | Purpose |
|---|---|
| `*_summary.json` | Machine-readable run summary. |
| `*_history.csv` | Per-iteration convergence data. |
| `*_convergence.png` | Convergence plot generated with Matplotlib. |
| `*_best_route.csv` | Best TSP permutation. |
| `runtime_metadata.json` | Runtime and dependency metadata. |
| `workflow_inventory.json` | Workflow descriptions and source mapping. |
| `original_configurations.json` | Source-scale configurations exposed by the reconstruction. |
| `validation_summary.json` | Validation checks and measured outcomes. |
| `results_manifest.json` | Artifact inventory with file types and sizes. |
| `MATLAB_Reconstruction_All_Workflow_Results.zip` | Archive containing the exported results directory. |

## Project structure

The supplied script is a monolithic notebook-exported workflow file. Conceptually, the implementation is organized as follows:

```text
MATLAB_Reconstruction/
├── Environment and imports
├── Reproducibility and runtime utilities
├── Workflow inventory
├── ES objective functions
├── ES representation and operators
│   ├── population initialization
│   ├── crossover
│   ├── mutation
│   └── survivor selection
├── TSP parsing and objectives
│   ├── TSPLIB parsing
│   ├── distance matrix construction
│   └── tour cost and fitness
├── TSP operators
│   ├── PMX
│   ├── Order-1
│   ├── Cycle crossover
│   ├── MOX
│   ├── swap
│   ├── insert
│   ├── inversion
│   └── scramble
├── TSP selection and survivor logic
├── Validation and artifact collection
├── Workflow wrappers
├── Example six-city configuration
└── Final automatic result export
```

## Implementation notes and source-faithful quirks

Several details are documented explicitly because they affect scientific interpretation:

1. **Notebook-export syntax:** the standalone `.py` file contains notebook markdown cells as string literals, which interact with the placement of the future import.
2. **Correlated ES covariance:** the `alpha` state is updated but is not applied to rotate the covariance before Gaussian sampling.
3. **TSPLIB distance semantics:** coordinate files are converted to raw Euclidean distances rather than all possible TSPLIB-specific edge-weight conventions.
4. **TSP fitness:** the implementation maximizes reciprocal tour cost rather than minimizing cost directly.
5. **TSP survivor selection:** for the main survivor mode, the new population is assembled through repeated stochastic selection from the eligible pool rather than direct truncation.
6. **Source identifiers:** spellings such as `tournoment`, `roletwheel`, `trancation`, `landa+mua`, and `landa_va_mua` are retained for compatibility with the reconstructed interface.
7. **Repeated workflow behavior:** `run_tsp_workflow4` performs `count` runs only in fast-validation mode; in its non-fast branch, the loop executes once. This behavior is preserved rather than reinterpreted.
8. **Optional forensic validation:** the source checks for original archive files only when they are physically available under the provided base directory.

## Limitations

The project should be evaluated as a reconstruction rather than as a comprehensive experimental study.

### Scientific limitations

- No statistical comparison against alternative evolutionary algorithms is provided.
- No external benchmark suite is bundled with the supplied script.
- Repeated-run analysis is limited to the configurations encoded in the reconstruction.
- The ES correlated mutation state does not mathematically influence the sampled covariance, so the implementation is not equivalent to a fully rotated correlated Gaussian mutation model.
- TSP coordinates are converted to raw Euclidean distances regardless of the source file's declared edge-weight type.
- The source does not establish computational-complexity measurements or scaling laws.

### Reproducibility limitations

- Exact bitwise reproducibility across environments is not guaranteed.
- Results depend on the exact NumPy random-generator behavior and numerical libraries available at execution time.
- The supplied script requires a small syntax-level cleanup before standalone execution because it is a direct notebook export.
- The original MATLAB support files referenced by the forensic validation routine are not included in this single uploaded Python source file.

## Future work

For a research-grade extension, the next steps should focus on methodological validation rather than only additional features:

1. Add a proper Python package structure with unit tests for every operator.
2. Separate the reconstructed implementation from notebook presentation code.
3. Add explicit experiment configuration files and versioned seeds.
4. Compare the reconstructed operators against independently tested reference implementations.
5. Validate TSPLIB edge-weight semantics per instance rather than assuming raw Euclidean distance.
6. Implement a mathematically complete correlated-ES covariance update if the research objective requires genuine rotational adaptation.
7. Add statistical summaries across larger numbers of independent seeds, including confidence intervals and effect sizes where appropriate.
8. Add benchmark coverage for standard continuous and combinatorial optimization suites.
9. Record hardware, software, and dependency versions in a machine-readable experiment manifest for each experiment.
10. Add automated continuous integration for syntax, unit, numerical, route-validity, and documentation checks.

## Research reproducibility checklist

Before treating an experiment as publication-grade, record the following:

- the exact Python version;
- NumPy, pandas, and Matplotlib versions;
- operating system and hardware;
- the experiment seed or seed set;
- the full configuration object;
- the exact input instance or distance matrix;
- the complete convergence history;
- the final route or continuous decision vector;
- the code revision used for execution;
- whether validation was run with original-scale or fast-validation settings.

## References

1. **GitHub Docs — Writing mathematical expressions.** GitHub documents `$...$` for inline mathematics and `$$...$$` for block mathematics in Markdown files, with MathJax used for rendering. [GitHub mathematical expressions documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
2. **GitHub Docs — Creating and highlighting code blocks.** GitHub documents fenced code blocks and language identifiers for syntax highlighting. [GitHub code block documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks)
3. **Original project source:** the supplied reconstruction file identifies the original Colab notebook as the source archive. [Original Colab notebook](https://colab.research.google.com/drive/1BrJ2bZkU0lZNIPNcw_snyWGGHXRhwVQx)

## Documentation status

This README describes the supplied Python reconstruction and explicitly records source-level behavior that affects reproducibility or scientific interpretation. It avoids claims of novelty, benchmark superiority, or experimental findings that are not established by the source and the bounded validation executions documented above.

