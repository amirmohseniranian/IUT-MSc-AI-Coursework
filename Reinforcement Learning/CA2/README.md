# Reinforcement Learning Assignment 2: Windy Gridworld with n-step Sarsa and Sarsa(λ)

## Overview

This repository contains a reproducible tabular reinforcement-learning implementation of the classic **Windy Gridworld** task. The source implements two on-policy temporal-difference learning families:

1. **n-step Sarsa** for $n \in \{1,5,10\}$.
2. **Sarsa(λ)** with replacing eligibility traces for $\lambda \in \{0,0.1,0.5,1\}$.

The experiment suite also includes a focused learning-rate ($\alpha$) sensitivity study that compares $n=5$ and $\lambda=0.5$ across $\alpha \in \{0.1,0.3,0.5,0.7\}$.

The implementation is designed for **Google Colab**. It uses deterministic environment dynamics, seeds NumPy and Python's random generator, saves per-episode results to CSV, generates high-resolution PNG figures, records configuration metadata as JSON, and packages the generated artifacts into a ZIP archive.

The source does not contain precomputed numerical results. Running the experiment pipeline generates the results and figures from the configured seeds.

> **Implementation note:** The uploaded file is a Colab-generated Python artifact. In its current layout, `from __future__ import annotations` appears after the initial module docstring, so the uploaded `.py` does not pass direct standalone Python compilation. The learning algorithms and experiment pipeline documented here match the supplied source.

## Scientific Motivation

Windy Gridworld is a compact benchmark for studying **temporal-difference control** with a stochastic policy and deterministic environment dynamics. The agent must learn an action-value function while repeatedly receiving a negative step reward until it reaches a fixed goal.

The project is useful for comparing how different temporal spans in Sarsa updates affect learning behavior:

- **One-step Sarsa** updates from the immediately observed reward and next action.
- **n-step Sarsa** delays the update until several rewards can contribute to the return.
- **Sarsa(λ)** distributes a temporal-difference error backward through an eligibility-trace mechanism.

The implementation follows this workflow: environment validation, policy definition, n-step Sarsa, Sarsa(λ), comparative experiments, learning-rate sensitivity, visualization, and artifact packaging.

## Problem Definition

The environment is a $7 \times 10$ grid with:

- Start state: $(3,0)$
- Goal state: $(3,7)$
- Four actions: **up**, **down**, **left**, and **right**
- Reward: $-1$ on every transition
- Episode termination: reaching the goal
- Column-dependent upward wind:

```math
W = (0,0,0,1,1,1,2,2,1,0).
```

The state space therefore has

```math
7 \times 10 = 70
```

tabular states, and the action space contains four discrete actions. The Q-table has shape $(70,4)$.

### Transition Dynamics

Let the current state be $(r,c)$ and the selected action be $a$. The action first changes the row or column by one grid cell. Both coordinates are clipped to the grid boundaries. The column-specific wind is then applied as an upward displacement, followed by another row clipping step.

The transition rule can be summarized as

```math
r' = clip(r + \Delta_r(a) - W[c'], 0, 6),
```

```math
c' = clip(c + \Delta_c(a), 0, 9),
```

where `clip` denotes clipping to the valid integer range and $(\Delta_r,\Delta_c)$ is the displacement associated with the selected action.

The episode terminates when

```math
(r',c') = (3,7).
```

The environment is deterministic: once the state and action are fixed, the next state and reward are fixed.

## Policy

The source uses an **$\epsilon$-greedy policy**.

With probability $\epsilon$, the policy samples uniformly from the four available actions. Otherwise, it selects a greedy action with the largest Q-value.

The implementation explicitly handles ties. In the usual case, tied greedy actions are sampled uniformly. When all four action values are equal, it uses a deterministic, goal-directed tie-breaker that prefers the action whose immediate successor is closest to the goal under the known deterministic wind map.

The configured exploration rate is

```math
\epsilon = 0.10.
```

This tie-breaking rule matters for reproducibility because the initial Q-table contains equal values.

## Mathematical Background

### Action-Value Function

For state $s$ and action $a$, the tabular learner stores an estimate

```math
Q(s,a).
```

The objective is to learn action values that favor policies reaching the goal in as few penalized time steps as possible.

Because each transition receives reward $-1$, minimizing the number of steps is equivalent to maximizing the undiscounted cumulative reward. The implementation, however, uses a configurable discount factor of $\gamma=0.99$, so its numerical return is discounted.

### One-Step Sarsa

For the one-step case, the temporal-difference target is

```math
G_t^{(1)} = R_{t+1} + \gamma Q(S_{t+1},A_{t+1}),
```

for a nonterminal transition. The corresponding update is

```math
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
G_t^{(1)}-Q(S_t,A_t)
\right].
```

At a terminal transition, the future bootstrap term is omitted.

### n-step Sarsa

For a general horizon $n$, the source constructs an n-step return by accumulating up to $n$ rewards and, if the terminal state has not been reached, bootstrapping from the state-action pair at time $t+n$:

```math
G_t^{(n)}
=
\sum_{i=t+1}^{\min(t+n,T)}
\gamma^{\,i-t-1}R_i
+
\mathbf{1}_{\{t+n<T\}}
\gamma^n Q(S_{t+n},A_{t+n}),
```

where $T$ denotes the terminal time.

The corresponding update is

```math
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
G_t^{(n)}-Q(S_t,A_t)
\right].
```

The implementation evaluates this procedure for

```math
n \in \{1,5,10\}.
```

For $n=1$, the procedure reduces to the one-step Sarsa family.

### Sarsa(λ) with Replacing Traces

The Sarsa(λ) implementation maintains an eligibility trace table $E$ with the same shape as the Q-table.

For a nonterminal transition, the TD error is

```math
\delta_t
=
R_{t+1}
+
\gamma Q(S_{t+1},A_{t+1})
-
Q(S_t,A_t).
```

For a terminal transition, the bootstrap term is removed:

```math
\delta_t
=
-1
-
Q(S_t,A_t).
```

The source uses **replacing traces**, setting the trace of the current state-action pair to one:

```math
E(S_t,A_t) \leftarrow 1.
```

The Q-table is then updated for all state-action pairs:

```math
Q \leftarrow Q + \alpha\,\delta_t\,E.
```

Finally, all traces are decayed:

```math
E \leftarrow \gamma\lambda E.
```

The experiment evaluates

```math
\lambda \in \{0,0.1,0.5,1\}.
```

The implementation therefore covers the configured trace range from no eligibility-trace contribution at $\lambda=0$ to the maximum configured trace parameter at $\lambda=1$.

## Relationship Between the Two Algorithm Families

The repository compares two different ways of extending one-step temporal-difference learning.

**n-step Sarsa** changes the number of rewards included in an explicit multi-step return before the update occurs.

**Sarsa(λ)** instead retains a one-step TD error and propagates that error backward through decaying eligibility traces.

This provides a controlled comparison of temporal credit-assignment mechanisms under the same environment, policy class, discount factor, and primary learning rate.

## Implementation Details

### State Representation

Grid coordinates are encoded as a compact integer index:

```math
\operatorname{index}(r,c) = r \times 10 + c.
```

The inverse mapping uses integer quotient and remainder operations:

```math
(r,c) = \operatorname{divmod}(\operatorname{index},10).
```

The goal state $(3,7)$ therefore has index

```math
3\times 10+7=37.
```

### Q-Table Initialization

Each experiment initializes a new Q-table with zeros:

```text
Q ∈ R^(70 × 4)
```

Training then starts from the fixed start state.

The project uses no function approximation, replay buffer, neural network, target network, or external dataset.

### Numerical Acceleration

The computational cores of both learning algorithms use Numba's `@njit(cache=True)`, compiling the inner training loops for numerical execution while preserving the tabular formulation.

The higher-level experiment utilities use NumPy and pandas for numerical organization and result aggregation, and Matplotlib produces the figures.

## Experimental Design

### Main Comparison

The main experiment fixes

```math
\alpha = 0.50,
\qquad
\gamma = 0.99,
\qquad
\epsilon = 0.10,
```

and runs:

| Algorithm family | Configurations | Episodes per seed | Seeds |
|---|---:|---:|---:|
| n-step Sarsa | $n=1,5,10$ | 100 | 13, 42, 123 |
| Sarsa(λ) | $\lambda=0,0.1,0.5,1$ | 100 | 13, 42, 123 |

The main comparison therefore contains seven algorithm configurations and three independent seeded runs per configuration.

### Learning-Rate Sensitivity

The source performs a focused $\alpha$ sweep using:

| Configuration | $\alpha$ values | Episodes per seed | Seeds |
|---|---:|---:|---:|
| n-step Sarsa, $n=5$ | 0.10, 0.30, 0.50, 0.70 | 80 | 13, 42 |
| Sarsa(λ), $\lambda=0.5$ | 0.10, 0.30, 0.50, 0.70 | 80 | 13, 42 |

This is intentionally narrower than a full grid over all algorithm parameters and isolates learning-rate effects for two representative configurations.

## Evaluation

The primary quantity recorded for each episode is **steps to goal**.

Because the reward is $-1$ per transition, fewer steps correspond to a less negative episode return. This README does not report numerical performance values because the supplied source does not contain precomputed result tables; those values are generated when the experiment suite is executed.

### Aggregation Across Seeds

For each algorithm configuration and episode, the source computes:

- mean steps,
- median steps,
- sample standard deviation.

It also computes summary statistics over the final 100 episodes:

- mean steps,
- median steps,
- sample standard deviation,
- best episode length,
- number of episodes,
- number of unique seeds.

For the alpha-sensitivity experiment, the nominal final window is also the last 100 episodes. Because that experiment runs only 80 episodes per seed, the implemented filter includes all available episodes in that subset.

This distinction matters for interpretation: the main experiment has a full 100-episode final window, whereas the alpha sweep has fewer than 100 episodes available per seed.

## Reproducibility

Reproducibility is addressed explicitly at several levels.

### Random Seeds

The source sets:

```python
random.seed(seed)
np.random.seed(seed)
```

It also seeds NumPy inside each Numba-compiled training core.

The main experiment uses seeds:

```text
13, 42, 123
```

and the alpha sweep uses:

```text
13, 42
```

### Deterministic Environment

Windy Gridworld dynamics are deterministic. Repeated execution with the same algorithm configuration and seed therefore follows the same environment transitions and seeded action-sampling process, subject to the execution environment and numerical software stack.

### Configuration Record

The complete experiment configuration is saved to:

```text
outputs/logs/experiment_configuration.json
```

The record includes the main hyperparameters, seeds, wind specification, start state, goal state, and tested n-step and $\lambda$ values.

### Environment Specification Artifact

The environment definition is also saved separately to:

```text
data/windy_gridworld_environment_specification.json
```

It records:

- grid shape,
- start state,
- goal state,
- wind strength by column,
- action mapping,
- per-step reward,
- configured discount factor,
- termination rule.

## Outputs

Running the full pipeline creates the following main artifacts.

### Tables

```text
outputs/tables/main_per_episode_results.csv
outputs/tables/main_summary.csv
outputs/tables/main_learning_curves.csv

outputs/tables/alpha_sensitivity_per_episode.csv
outputs/tables/alpha_sensitivity_summary.csv
outputs/tables/alpha_sensitivity_learning_curves.csv
```

### Figures

```text
outputs/figures/windy_gridworld_environment.png
outputs/figures/main_learning_curves.png
outputs/figures/alpha_effect.png
outputs/figures/final_performance_comparison.png
```

All figures are written at 300 DPI.

### Logs and Specification

```text
outputs/logs/experiment_configuration.json
data/windy_gridworld_environment_specification.json
```

### Submission Package

The pipeline creates:

```text
outputs/assignment_submission_package.zip
```

The package may include the source script, an available notebook copy, generated tables and figures, the environment specification, logs, and a validation/audit report when those files are present.

## Project Structure

A typical generated repository structure is:

```text
.
├── assignment_solution.py
├── assignment_solution.ipynb
├── changes_audit_and_validation_report.md
├── data/
│   └── windy_gridworld_environment_specification.json
└── outputs/
    ├── figures/
    │   ├── alpha_effect.png
    │   ├── final_performance_comparison.png
    │   ├── main_learning_curves.png
    │   └── windy_gridworld_environment.png
    ├── logs/
    │   └── experiment_configuration.json
    ├── tables/
    │   ├── alpha_sensitivity_learning_curves.csv
    │   ├── alpha_sensitivity_per_episode.csv
    │   ├── alpha_sensitivity_summary.csv
    │   ├── main_learning_curves.csv
    │   ├── main_per_episode_results.csv
    │   └── main_summary.csv
    ├── assignment_submission_package.zip
    └── submission_package/
```

Not every file is present before execution; the experiment pipeline creates the generated directories and artifacts.

## Dependencies

The source imports the following primary Python packages:

| Package | Role |
|---|---|
| `numpy` | Numerical arrays, random sampling, Q-tables, environment calculations |
| `pandas` | Per-episode records, grouping, summaries, and CSV export |
| `matplotlib` | Environment and experiment visualization |
| `numba` | JIT compilation of the two training cores |
| `IPython` | Displaying generated figures in the Colab workflow |

The standard-library components include `json`, `math`, `random`, `shutil`, `sys`, `dataclasses`, `pathlib`, and `typing`.

The source does not pin package versions, so exact byte-for-byte reproducibility across machines is not guaranteed unless the Python and dependency versions are also recorded.

## Installation

### Google Colab

The project is structured for Google Colab and uses paths relative to the current working directory.

A fresh Colab runtime normally includes the core scientific Python stack. Missing dependencies can be installed with:

```bash
pip install numpy pandas matplotlib numba ipython
```

Then place the source and any supporting files in the working directory.

### Local Python Environment

For local execution, install the imported dependencies in an isolated environment. Because the uploaded artifact has a Colab-generated module layout and its `from __future__ import annotations` statement is not at the beginning of the file, **standalone execution of the uploaded `.py` requires that packaging/layout issue to be corrected first**.

This README does not silently rewrite the source; the limitation is recorded here so that reproducibility claims remain faithful to the supplied artifact.

## Usage

The full pipeline is exposed through:

```python
results = execute()
```

Execution proceeds conceptually as follows:

```python
ensure_directories()
validate_environment()
save_environment_specification()

# Main comparison
#   n-step Sarsa: n = 1, 5, 10
#   Sarsa(lambda): lambda = 0, 0.1, 0.5, 1.0

# Alpha sensitivity
#   n = 5
#   lambda = 0.5

# Save CSV tables
# Save PNG figures
# Save configuration JSON
# Create submission ZIP
```

The source also displays the main and alpha-sensitivity summaries after execution, along with the generated PNG figures in the notebook/Colab workflow.

## Configuration

The central configuration is stored in an immutable `ExperimentConfig` dataclass.

| Parameter | Default |
|---|---:|
| `episodes` | 100 |
| `alpha_sweep_episodes` | 80 |
| `gamma` | 0.99 |
| `epsilon` | 0.10 |
| `main_alpha` | 0.50 |
| `alpha_values` | 0.10, 0.30, 0.50, 0.70 |
| `max_steps_per_episode` | 50,000 |
| `main_seeds` | 13, 42, 123 |
| `alpha_sweep_seeds` | 13, 42 |

The algorithm-specific parameter sets are:

```math
n \in \{1,5,10\},
```

and

```math
\lambda \in \{0,0.1,0.5,1\}.
```

### Important Configuration Observation

The environment follows the classic Windy Gridworld structure, but the source configures

```math
\gamma = 0.99
```

rather than $\gamma=1$. Therefore, the configured learning problem should not be described as fully undiscounted.

## Validation and Safety Checks

The source includes several explicit checks:

- The initial state must equal the configured start state.
- The goal state must map to tabular index 37.
- A zero-wind transition is checked explicitly.
- A wind-strength-two transition is checked explicitly.
- Invalid actions are rejected.
- Invalid $\alpha$, $\epsilon$, and $\lambda$ values are rejected by the public training functions.
- Non-finite Q-values cause Sarsa(λ) execution to fail.
- Episodes exceeding the configured maximum step count raise an error.

These checks make the environment and numerical training behavior easier to audit.

## Visualization

The pipeline generates four figures.

### Windy Gridworld schematic

`windy_gridworld_environment.png` shows the grid, start and goal positions, and the wind strength associated with each column.

### Main learning curves

`main_learning_curves.png` plots the mean steps-to-goal per episode across seeds for all requested n-step Sarsa and Sarsa(λ) configurations at the main learning rate $\alpha=0.5$.

### Learning-rate sensitivity

`alpha_effect.png` plots final-window mean steps against $\alpha$ for the representative $n=5$ and $\lambda=0.5$ configurations.

### Final-performance comparison

`final_performance_comparison.png` shows final-window mean steps with error bars based on the reported across-seed standard deviation.

## Interpretation Guidelines

The evaluation metric is **steps to goal**, so lower values are better.

Several methodological points should be kept in mind when interpreting the generated figures:

1. The main comparison uses only three seeds.
2. The alpha-sensitivity analysis uses two seeds.
3. The study covers a 100-episode main horizon and an 80-episode alpha-sweep horizon.
4. The alpha sweep compares representative configurations rather than every tested n-step and $\lambda$ value.
5. The environment is deterministic, but the behavior policy is stochastic because of $\epsilon$-greedy exploration.
6. The implementation uses $\gamma=0.99$.
7. The source does not report confidence intervals, statistical significance tests, or formal hypothesis tests.

These constraints do not invalidate the assignment experiment; they define the scope of the conclusions it supports.

## Reproducibility Checklist

A reproducible rerun should retain:

- the exact source file,
- the Python version,
- the package versions,
- the configured seeds,
- the generated `experiment_configuration.json`,
- the generated environment specification,
- the raw per-episode CSV files,
- the generated figures.

Seed values alone are not a complete software-environment specification because compiled numerical libraries and package versions can also affect execution.

## Limitations

The current implementation has several limitations that are relevant to a research-oriented repository.

### Source packaging

The uploaded `.py` is a Colab-generated artifact and currently fails direct compilation because `from __future__ import annotations` is not at the beginning of the module. This is a packaging/layout issue, not evidence that the underlying learning routines are mathematically invalid.

### Narrow experimental scope

The main experiment contains three seeds and 100 episodes per configuration. The alpha study contains two seeds and 80 episodes per configuration. These settings are appropriate for an assignment-scale experiment but are not sufficient to support broad claims of statistical superiority across environments.

### Single environment

All comparisons are performed on one tabular Windy Gridworld instance. The results therefore describe this environment and configuration rather than reinforcement-learning performance in general.

### Fixed exploration strategy

The exploration probability is fixed at $\epsilon=0.10$. The repository does not examine exploration schedules or alternative behavior policies.

### Fixed environment dynamics

Wind strengths, grid dimensions, start state, and goal state are defined through module-level configuration constants. The source does not benchmark multiple environment variants.

### No function approximation

The project uses a tabular Q-function. It therefore does not investigate neural-network approximation, representation learning, or large state spaces.

### Limited uncertainty analysis

The source reports descriptive mean, median, and standard deviation summaries but does not perform inferential statistical tests.

## Future Work

Natural extensions, consistent with the existing experimental design, include:

- increasing the number of independent seeds;
- extending the training horizon;
- evaluating additional learning rates and exploration schedules;
- comparing more representative configurations in the alpha study;
- adding confidence intervals or bootstrap uncertainty estimates;
- benchmarking multiple Windy Gridworld variants;
- separating training and evaluation episodes when assessing learned policies;
- recording exact Python and dependency versions;
- repairing the standalone source packaging and adding an automated continuous-integration test;
- adding unit tests for transition dynamics and update equations.

These extensions are methodological opportunities, not claims about results that are absent from the supplied source.

## Implementation Map

The source can be read in the following order:

| Source component | Function |
|---|---|
| `ExperimentConfig` | Central experiment configuration |
| `WindyGridworld` | Environment dynamics |
| `state_to_index` / `index_to_state` | Tabular state encoding |
| `epsilon_greedy_action` | Policy helper and tie handling |
| `_n_step_sarsa_core` | Compiled n-step Sarsa implementation |
| `n_step_sarsa` | Public n-step Sarsa interface |
| `_sarsa_lambda_core` | Compiled Sarsa(λ) implementation |
| `sarsa_lambda` | Public Sarsa(λ) interface |
| `run_method` | Seeded experiment execution |
| `summarize_runs` | Across-seed aggregation |
| `plot_environment` | Environment visualization |
| `plot_main_learning_curves` | Main learning-curve visualization |
| `plot_alpha_effect` | Alpha-sensitivity visualization |
| `plot_final_performance` | Final-window comparison |
| `run_all_experiments` | Full experiment suite |
| `create_submission_package` | ZIP packaging |
| `execute` | End-to-end entry point |

## Research Relevance

This project demonstrates several research-relevant engineering practices in a compact reinforcement-learning setting:

- explicit mathematical specification of a learning problem;
- separation of environment, policy, learning, evaluation, and packaging components;
- multi-seed experimentation;
- controlled hyperparameter comparison;
- deterministic environment validation;
- compiled numerical kernels;
- structured result serialization;
- reproducible artifact generation.

For an academic portfolio, the strongest contribution of this repository is not algorithmic novelty. It is the **transparent implementation and experimental comparison of established temporal-difference control methods under a clearly specified environment and reproducible configuration**.

## References

### Primary Algorithm Reference

R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd edition, MIT Press, 2018.

The uploaded source explicitly identifies the classic Windy Gridworld as the setting associated with Example 6.5 in the second edition (and corresponding older course numbering).

