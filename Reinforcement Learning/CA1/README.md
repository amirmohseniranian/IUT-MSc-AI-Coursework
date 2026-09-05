# Reinforcement Learning Homework 1 - Dynamic Programming on Gridworlds

A complete Python implementation of the dynamic-programming exercises included in a Reinforcement Learning homework based on Sutton and Barto. The notebook covers a 5 x 5 Gridworld, Bellman expectation updates, iterative policy evaluation, policy iteration, and value iteration.

> **Academic scope.** This repository documents the behavior of the supplied notebook as implemented. It does not claim results, functionality, or experimental findings that are absent from the source code. In particular, the implementation of **Exercise 15.3 is explicitly a placeholder framework** and is documented as such.

---

## Overview

The project studies a finite, discounted Markov decision process through a compact 5 x 5 Gridworld implementation. The code uses dynamic programming rather than sampling-based reinforcement-learning methods.

The notebook contains five assignment parts:

| Part | Topic                | Status                     |
| ---- | -------------------- | -------------------------- |
| 1    | Figure 5.3 Gridworld | Implemented                |
| 2    | Exercise 15.3        | Placeholder framework only |
| 3    | Policy Evaluation    | Implemented                |
| 4    | Policy Iteration     | Implemented                |
| 5    | Value Iteration      | Implemented                |

The implementation uses a shared 5 x 5 state space with four cardinal actions:

| Action index | Direction | Grid displacement       |
| -----------: | :-------: | :---------------------- |
|          `0` |    `^`    | one row upward          |
|          `1` |    `>`    | one column to the right |
|          `2` |    `v`    | one row downward        |
|          `3` |    `<`    | one column to the left  |

The main global parameters are:

| Parameter |   Value | Meaning               |
| --------- | ------: | --------------------- |
| `GAMMA`   |   `0.9` | Discount factor       |
| `THETA`   | `0.001` | Convergence tolerance |

The notebook also includes helper functions for policy visualization and CSV export.

---

## Scientific Background

The project is formulated as a finite, discounted Markov decision process (MDP). The state space is

$$
\mathcal{S} = \{0,\ldots,4\}\times\{0,\ldots,4\},
$$

which gives 25 grid states. Each nonterminal state has four actions.

For a state $s$ and action $a$, the environment returns a next state $s'$ and an immediate reward $r$. The discount factor is

$$
\gamma = 0.9.
$$

The central quantities are the state-value and action-value functions. For a policy $\pi$, the state-value function is

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
\sum_{t=0}^{\infty}
\gamma^t R_{t+1}
\mid S_0=s
\right].
$$

For a deterministic action choice, the one-step backup used by the implementation has the form

$$
r(s,a) + \gamma V(s').
$$

The notebook applies Bellman updates repeatedly until the maximum absolute change falls below the configured tolerance.

---

## Project Goals

The implementation illustrates the relationship between:

1. a specified transition-and-reward model,
2. Bellman expectation backups,
3. iterative policy evaluation,
4. policy improvement,
5. policy iteration, and
6. value iteration.

The code is intentionally small and transparent. Each algorithm is implemented directly with NumPy loops over the finite state and action spaces rather than being hidden behind a reinforcement-learning framework.

---

## Mathematical Formulation

### 1. Grid Representation

States are represented by row-column pairs,

$$
s=(r,c),
\qquad
r,c\in{0,1,2,3,4}.
$$

The implementation explicitly enumerates all 25 states.

The four actions are represented by integer indices:

$$
\mathcal{A}=\{0,1,2,3\}.
$$

The corresponding displacements are:

$$
0\mapsto(-1,0),
\qquad
1\mapsto(0,1),
\qquad
2\mapsto(1,0),
\qquad
3\mapsto(0,-1).
$$

### 2. Figure 5.3 Gridworld

Part 1 defines two special states.

For the state at row 0, column 1, every action produces:

$$
(0,1)\rightarrow(4,1),
\qquad
r=10.
$$

For the state at row 0, column 3, every action produces:

$$
(0,3)\rightarrow(2,3),
\qquad
r=5.
$$

For all other states, the action is applied according to the displacement table. If the resulting position falls outside the 5 x 5 grid, the agent remains in the current state and receives reward -1.

Otherwise, the agent moves to the requested neighboring state and receives reward 0.

Part 1 evaluates the uniform random policy, defined by

$$
\pi(a\mid s)=\frac{1}{4}
$$

for every state-action pair.

Accordingly, `solve_part1()` uses the following Bellman expectation update:

$$
V_{k+1}(s)
=
\sum_{a\in\mathcal{A}}
\frac{1}{4}
\left[
r(s,a)+\gamma V_k(s')
\right].
$$

The code stores the previous iterate in a copy of the value array and stops when

$$
\max_s
\left|
V_{k+1}(s)-V_k(s)
\right|
<
\theta.
$$

With the supplied implementation and parameters, the resulting value matrix is approximately:

```text
[[ 3.316  8.796  4.434  5.328  1.497]
 [ 1.529  2.999  2.256  1.913  0.553]
 [ 0.058  0.745  0.679  0.364 -0.398]
 [-0.966 -0.429 -0.348 -0.579 -1.177]
 [-1.850 -1.338 -1.222 -1.416 -1.969]]
```

The largest value occurs at the special state that receives the immediate reward of 10 and transfers to row 4, column 1.

### 3. Policy Evaluation

Part 3 uses a simpler transition model.

There are no special teleporting states. A valid move gives reward 0, while an attempted move outside the grid leaves the state unchanged and gives reward -1.

The evaluated policy is again the uniform random policy. The Bellman expectation equation is

$$
V^\pi(s)
=
\sum_{a\in\mathcal{A}}
\frac{1}{4}
\left[
r(s,a)+\gamma V^\pi(s')
\right].
$$

The implementation performs asynchronous in-place updates during each sweep:

1. read the current value of a state,
2. compute the average backup over all four actions,
3. write the new value immediately,
4. update the maximum change observed during the sweep.

The stopping condition is

$$
\Delta
=
\max_s
\left|
V_{\text{new}}(s)-V_{\text{old}}(s)
\right|
<
\theta.
$$

The resulting matrix is approximately:

```text
[[-2.674 -2.159 -2.010 -2.160 -2.676]
 [-2.159 -1.644 -1.494 -1.644 -2.160]
 [-2.010 -1.494 -1.345 -1.495 -2.011]
 [-2.160 -1.644 -1.495 -1.645 -2.161]
 [-2.676 -2.160 -2.011 -2.161 -2.677]]
```

The negative values arise because the random policy can repeatedly select actions that cross a boundary and incur the -1 reward.

### 4. Policy Iteration

Part 4 follows the standard two-stage policy-iteration procedure:

1. **Policy evaluation**
2. **Greedy policy improvement**

The initial deterministic policy assigns action `0`, corresponding to `^`, to every state.

#### Policy evaluation

For the current deterministic policy $\pi$, the implementation repeatedly applies

$$
V^\pi(s)
=
r(s,\pi(s))
+
\gamma V^\pi(s'),
$$

using the transition function defined for Part 3.

#### Policy improvement

For each state, the code evaluates all four actions:

$$
Q(s,a)
=
r(s,a)
+
\gamma V(s')
$$

and selects

$$
\pi_{\text{new}}(s)
=
\arg\max_{a\in\mathcal{A}}
Q(s,a).
$$

The policy is considered stable when no state's selected action changes during an improvement sweep.

The final policy is displayed as arrows. With the supplied code, it is:

```text
> > > > v
> > > > v
> > > > v
> > > > v
> > > > ^
```

The associated value estimates are close to zero but remain slightly negative because policy evaluation stops at the finite tolerance `THETA = 0.001`.

The returned values are therefore approximate values under the stopping criterion rather than exact symbolic solutions.

### 5. Value Iteration

Part 5 implements the Bellman optimality backup

$$
V_{k+1}(s)
=
\max_{a\in\mathcal{A}}
\left[
r(s,a)+\gamma V_k(s')
\right].
$$

The code updates the state values in place. Consequently, within a sweep, later states may use values that were already updated earlier in the same sweep. This is an asynchronous or in-place Bellman update rather than a fully synchronous copy-based implementation.

The iteration stops when

$$
\Delta
=
\max_s
\left|
V_{\text{new}}(s)-V_{\text{old}}(s)
\right|
<
\theta.
$$

After convergence, the greedy policy is extracted using

$$
\pi^*(s)
=
\arg\max_{a\in\mathcal{A}}
\left[
r(s,a)+\gamma V(s')
\right].
$$

With the supplied initialization and transition model, the value matrix remains at zero:

```text
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
```

The extracted greedy policy is:

```text
> > > > v
^ ^ ^ ^ ^
^ ^ ^ ^ ^
^ ^ ^ ^ ^
^ ^ ^ ^ ^
```

The zero value function is consistent with the environment permitting trajectories that avoid boundary penalties indefinitely after entering suitable cycles of zero-reward transitions.

---

## Assignment-by-Assignment Analysis

### Part 1 - Figure 5.3 Gridworld

The implementation encodes the main special-transition structure directly in `part1_transition()`.

The function first checks whether the current state is one of the two special states. If so, the selected action does not affect the destination or reward. Otherwise, the usual four-neighbor grid dynamics are applied.

The numerical solution is obtained by iterating Bellman expectation updates under the uniform random policy.

Output file:

```text
part1_gridworld_value.csv
```

### Part 2 - Exercise 15.3

The notebook does **not** provide a complete, problem-specific implementation of Exercise 15.3.

The source explicitly contains:

```python
# Exercise 15.3 placeholder environment implementation
# The same dynamic programming framework is provided here.
# Modify reward/transition definitions if your textbook edition
# uses a different figure.
```

Therefore, this README does not attribute a specific numerical solution to the exercise.

The notebook only prints:

```text
Exercise 15.3 implementation framework completed.
```

This distinction matters for reproducibility and academic accuracy: the current file documents a framework rather than a verified solution for a specific Exercise 15.3 environment.

### Part 3 - Policy Evaluation

The implementation evaluates the uniform random policy in the ordinary 5 x 5 Gridworld.

Output file:

```text
policy_evaluation.csv
```

### Part 4 - Policy Iteration

The implementation alternates between iterative policy evaluation and greedy policy improvement until the policy stabilizes.

Output files:

```text
policy_iteration_value.csv
policy_iteration_policy.csv
```

The policy file stores integer action indices rather than arrow characters.

The mapping is:

| Stored value | Direction |
| -----------: | :-------- |
|          `0` | `^`       |
|          `1` | `>`       |
|          `2` | `v`       |
|          `3` | `<`       |

### Part 5 - Value Iteration

The implementation applies the Bellman optimality operator and then extracts a deterministic greedy policy.

Output files:

```text
value_iteration_value.csv
value_iteration_policy.csv
```

---

## Algorithmic Comparison

| Algorithm | Backup type                            | Policy used during update                 | Output                                 |
| --------- | -------------------------------------- | ----------------------------------------- | -------------------------------------- |
| Part 1    | Bellman expectation                    | Uniform random                            | Value function                         |
| Part 3    | Bellman expectation                    | Uniform random                            | Value function                         |
| Part 4    | Policy evaluation + greedy improvement | Iteratively improved deterministic policy | Value function + policy                |
| Part 5    | Bellman optimality                     | Implicitly optimized over actions         | Optimal-value estimate + greedy policy |

The key conceptual distinction is between expectation and maximization.

Policy evaluation averages over the actions prescribed by a policy:

$$
V^\pi(s)
=
\sum_a
\pi(a\mid s)
\left[
r(s,a)+\gamma V^\pi(s')
\right].
$$

Value iteration instead selects the best one-step action:

$$
V^*(s)
=
\max_a
\left[
r(s,a)+\gamma V^*(s')
\right].
$$

Policy iteration combines these two ideas by first evaluating a fixed policy and then improving it greedily.

---

## Implementation Details

### Global Configuration

The notebook defines:

```python
GAMMA = 0.9
THETA = 0.001
```

`GAMMA` controls temporal discounting.

`THETA` is the convergence threshold used by the iterative procedures.

### State Enumeration

States are generated with:

```python
states = [(r,c) for r in range(5) for c in range(5)]
```

This creates the Cartesian product of the five row indices and five column indices.

### Policy Rendering

The helper function `show_policy(policy)` converts integer action indices into directional symbols:

```python
actions = {
    0: "^",
    1: ">",
    2: "v",
    3: "<"
}
```

The resulting 5 x 5 matrix is returned as a pandas `DataFrame`, making the policy easy to inspect in notebook output.

### CSV Export

The helper function

```python
def save_result(name, array):
    pd.DataFrame(array).to_csv(name + ".csv", index=False)
```

converts NumPy arrays to CSV files without including a pandas index column.

---

## Dependencies

The notebook imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
```

The implemented computations directly use:

* **NumPy** for numerical arrays and operations.
* **Pandas** for tabular policy display and CSV serialization.
* **Matplotlib** is imported but is not used by the supplied implementation.
* **`time`** is imported but is not used by the supplied implementation.

The code shown in the notebook does not require an external reinforcement-learning library.

A minimal environment can therefore be created with:

```bash
python -m pip install numpy pandas matplotlib jupyter
```

---

## Installation

Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install numpy pandas matplotlib jupyter
```

The notebook can then be opened with:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

## Usage

Open:

```text
RL_complete_homework_one_notebook(1).ipynb
```

and execute the cells from top to bottom.

The notebook then performs the following sequence:

1. imports the numerical and tabular libraries;
2. defines the common 5 x 5 state and action structures;
3. solves Part 1;
4. prints and exports the Part 1 value function;
5. executes the Part 2 placeholder;
6. evaluates the uniform policy for Part 3;
7. prints and exports the Part 3 value function;
8. runs policy iteration for Part 4;
9. prints and exports the resulting value function and policy;
10. runs value iteration for Part 5;
11. prints and exports the resulting value function and policy.

---

## Generated Outputs

The notebook writes the following CSV files to the current working directory:

| File                          | Contents                                        |
| ----------------------------- | ----------------------------------------------- |
| `part1_gridworld_value.csv`   | Part 1 value function                           |
| `policy_evaluation.csv`       | Part 3 evaluated value function                 |
| `policy_iteration_value.csv`  | Part 4 value function                           |
| `policy_iteration_policy.csv` | Part 4 deterministic policy encoded as integers |
| `value_iteration_value.csv`   | Part 5 optimal-value estimate                   |
| `value_iteration_policy.csv`  | Part 5 greedy policy encoded as integers        |

The policy CSV files contain integer action indices rather than arrow symbols.

---

## Reproducibility

The algorithms in this notebook are deterministic under a fixed Python/NumPy execution environment because the procedures do not use random-number generation.

There are no configurable random seeds in the source code.

For reproducible execution:

1. use the same notebook version;
2. use the same values of `GAMMA` and `THETA`;
3. execute the notebook from the first cell onward;
4. preserve the transition definitions exactly;
5. preserve the state and action indexing conventions.

The numerical outputs depend on the convergence tolerance because the iterative algorithms terminate when the maximum update falls below `THETA`.

---

## Verification of the Supplied Implementation

The supplied notebook was checked for consistency between the source code and this description. Several implementation details are relevant when interpreting the results.

### Deterministic versus stochastic policies

Part 1 and Part 3 use an explicit uniform action probability of one quarter for each of the four actions. Parts 4 and 5 use deterministic policies represented by integer action choices.

### Asynchronous updates

Part 3 updates the value array in place, so later states in the same sweep may use values updated earlier in that sweep.

Part 5 also updates states in place and computes the maximum action value from the currently available value array.

This differs from a strictly synchronous implementation, which would construct a complete copy of the previous value function for each sweep.

### Approximate convergence

Part 4 uses iterative evaluation with a tolerance of `0.001`. Consequently, even when the underlying fixed point is zero, the stored result can contain small residual errors rather than exact zeros.

### Argmax tie handling

Both policy improvement and greedy policy extraction use NumPy's `argmax`. When multiple actions have the same maximal value, `argmax` returns the first maximal index. Deterministic tie-breaking therefore follows NumPy's index ordering rather than a separately defined rule.

---

## Results

The following are the numerical outputs produced by the supplied implementation.

### Part 1

Approximate value function:

```text
[[ 3.316  8.796  4.434  5.328  1.497]
 [ 1.529  2.999  2.256  1.913  0.553]
 [ 0.058  0.745  0.679  0.364 -0.398]
 [-0.966 -0.429 -0.348 -0.579 -1.177]
 [-1.850 -1.338 -1.222 -1.416 -1.969]]
```

### Part 3

Approximate value function:

```text
[[-2.674 -2.159 -2.010 -2.160 -2.676]
 [-2.159 -1.644 -1.494 -1.644 -2.160]
 [-2.010 -1.494 -1.345 -1.495 -2.011]
 [-2.160 -1.644 -1.495 -1.645 -2.161]
 [-2.676 -2.160 -2.011 -2.161 -2.677]]
```

### Part 4

Approximate value function:

```text
[[-0.003 -0.003 -0.003 -0.002 -0.002]
 [-0.003 -0.003 -0.002 -0.002 -0.002]
 [-0.003 -0.002 -0.002 -0.002 -0.002]
 [-0.002 -0.002 -0.002 -0.002 -0.002]
 [-0.002 -0.002 -0.002 -0.002 -0.001]]
```

Policy:

```text
> > > > v
> > > > v
> > > > v
> > > > v
> > > > ^
```

### Part 5

Value function:

```text
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
```

Policy:

```text
> > > > v
^ ^ ^ ^ ^
^ ^ ^ ^ ^
^ ^ ^ ^ ^
^ ^ ^ ^ ^
```

These values report the behavior of the supplied program under the stated configuration. They should not be treated as independently re-derived textbook tables unless the corresponding textbook environment is exactly the one encoded in the notebook.

---

## Limitations

The current implementation has several limitations that should be considered before treating it as a general reinforcement-learning library.

### Exercise 15.3 is not fully implemented

The notebook itself labels Part 2 as a placeholder and instructs the user to modify the reward and transition definitions according to the relevant textbook figure.

### Hard-coded environment size

The implementation assumes a 5 x 5 grid in multiple places, including state construction, policy rendering, and array initialization.

### Hard-coded action space

There are exactly four actions corresponding to the four cardinal directions.

### No environment abstraction

The transition model is implemented directly in Python functions rather than through a reusable environment class or a standard RL-compatible interface.

### No general function approximation

The algorithms operate over explicit tabular state-value arrays and do not implement neural-network or other function-approximation methods.

### Limited experimentation

The notebook is an instructional implementation rather than a systematic benchmark suite. It does not include statistical comparisons across repeated runs, hyperparameter sweeps, or large-scale empirical studies.

### Unused imports

`matplotlib.pyplot` and `time` are imported but are not used in the supplied code.

---

## Future Work

The most direct extensions, consistent with the existing code structure, would be:

1. replace the Part 2 placeholder with the exact Exercise 15.3 transition and reward model;
2. generalize the Gridworld dimensions rather than hard-coding 5 x 5 arrays;
3. encapsulate states, actions, rewards, and transitions in a reusable environment abstraction;
4. expose `gamma` and `theta` as function arguments rather than global constants;
5. add convergence diagnostics such as iteration counts and residual histories;
6. compare synchronous and asynchronous dynamic-programming updates;
7. add automated tests for transition behavior and Bellman backups;
8. add explicit numerical checks against known textbook solutions;
9. record execution metadata for stronger computational reproducibility;
10. extend the project to model-free reinforcement-learning algorithms after the dynamic-programming foundations are established.

---

## Code Quality and Research-Reproducibility Notes

The notebook has several properties that are useful for academic inspection and reproducibility:

* the transition dynamics are explicitly encoded;
* the Bellman backups are visible rather than hidden inside a framework;
* the convergence threshold is explicit;
* the policy encoding is documented;
* value functions and policies are exported as CSV files;
* the implementation is small enough for direct code inspection;
* numerical outputs can be reproduced by executing the notebook from top to bottom.

For a research-grade extension, automated unit tests and a machine-readable environment specification would further improve verification.

---

## Suggested Repository Structure

A clean repository layout for this project is:

```text
.
|-- README.md
|-- RL_complete_homework_one_notebook(1).ipynb
|-- part1_gridworld_value.csv
|-- policy_evaluation.csv
|-- policy_iteration_value.csv
|-- policy_iteration_policy.csv
|-- value_iteration_value.csv
`-- value_iteration_policy.csv
```

The CSV files can be regenerated by rerunning the notebook.

---

## Theoretical Summary

The project demonstrates three closely related dynamic-programming mechanisms.

### Bellman expectation backup

For a fixed policy:

$$
V^\pi(s)
=
\sum_a
\pi(a\mid s)
\left[
r(s,a)+\gamma V^\pi(s')
\right].
$$

### Policy improvement

Given a current value function:

$$
\pi_{\text{new}}(s)
=
\arg\max_a
\left[
r(s,a)+\gamma V(s')
\right].
$$

### Bellman optimality backup

For the optimal value function:

$$
V^*(s)
=
\max_a
\left[
r(s,a)+\gamma V^*(s')
\right].
$$

These three equations provide the mathematical connection between Parts 1, 3, 4, and 5.

---

## References

1. Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, 2nd edition, MIT Press, 2018.
   https://incompleteideas.net/book/the-book-2nd.html

2. GitHub Docs, *Writing mathematical expressions*.
   https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions

3. GitHub Docs, *Basic writing and formatting syntax*.
   https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax

---
