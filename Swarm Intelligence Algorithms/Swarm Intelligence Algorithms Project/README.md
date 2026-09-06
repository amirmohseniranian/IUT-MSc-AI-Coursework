# Safe Robot Navigation in Dynamic Pedestrian Environments

## Multi-Algorithm Benchmark for Grid-Based, Continuous, and Swarm/Colony Path Planning

This repository contains a reproducible Python benchmark for safe 2D robot navigation in a synthetic environment with static obstacles and continuously moving pedestrian-like obstacles. The implementation compares eight path-planning and population-based optimization strategies under a shared trajectory representation and a shared spatiotemporal evaluation function.

The benchmark is designed around a simple but useful research question:

> How do discrete graph planning, continuous swarm optimization, and hybrid planning strategies trade off path length, geometric smoothness, obstacle clearance, collision exposure, execution time, and memory usage when the environment contains moving obstacles?

The implementation is an **algorithmic motion-planning benchmark, not a deep-learning model**. Its main research value is the controlled comparison of heterogeneous planning paradigms under the same synthetic environment and cost function.

---

## Overview

The source implements the following eight planners:

| Label in benchmark     | Implementation              | Planning representation        | Main role                                                     |
| ---------------------- | --------------------------- | ------------------------------ | ------------------------------------------------------------- |
| **D***                 | `DStarPlanner`              | Discrete occupancy grid        | D* Lite-style graph planning                                  |
| **PSO**                | `PSOPlanner(use_ldw=False)` | Continuous via-points + spline | Particle swarm optimization                                   |
| **LDW-PSO**            | `PSOPlanner(use_ldw=True)`  | Continuous via-points + spline | PSO with decreasing inertia weight                            |
| **D*-PSO (Proposed)**  | `DStarPSOHybrid`            | Grid guide + continuous spline | D* Lite guide followed by LDW-PSO refinement                  |
| **ABC**                | `ABCPlanner`                | Continuous via-points + spline | Artificial Bee Colony search                                  |
| **PSO-ABC (Proposed)** | `PSOABCHybrid`              | Continuous via-points + spline | PSO motion update combined with ABC-style partner exploration |
| **SMO**                | `SMOPlanner`                | Continuous via-points + spline | Spider Monkey Optimization-style search                       |
| **ACO**                | `ACOPlanner`                | Discrete occupancy grid        | Ant Colony Optimization                                       |

The two entries marked **Proposed** are implementation-specific hybrid strategies. The README does not claim that these hybrids are novel in the research literature; it describes only what is implemented in this codebase.

---

## Research Scope and Important Interpretation Notes

Several implementation details are scientifically important when interpreting benchmark results.

1. **The default benchmark generates five scenarios, but plans only on `scenario_1`.** The other generated scenarios are exported as dataset records and metadata; they are not independently evaluated by the eight planning algorithms in `run_full_pipeline`.
2. **The default scenarios are structurally identical.** The generator reseeds NumPy and Python's `random` module for each scenario, but the obstacle definitions themselves are deterministic constants and do not use those random draws. Consequently, the five default scenarios differ only in `scenario_id` while containing the same obstacle parameters.
3. **The experiment is not online replanning during execution.** Each planner is called once at a specified start time. Metaheuristic trajectories are evaluated against moving obstacles over the path traversal time, but the planner is not repeatedly replanned as the pedestrians move.
4. **D* is evaluated from a static occupancy grid at the planning time.** Its grid is generated at `t = 0` by default. The resulting path is subsequently scored using the moving-obstacle rollout.
5. **ACO is also planned on a single occupancy grid snapshot.** Its transition probabilities use pheromone and distance-to-goal information on that discrete grid; the final path is not dynamically replanned.
6. **GPU acceleration is partial rather than universal.** PyTorch tensors are used on CUDA when available, particularly for batched path-cost and collision calculations. Spline construction, several population-update loops, graph operations, and parts of ACO remain Python/NumPy/CPU controlled.
7. **The reported complexity values are simplified theoretical estimates stored by the implementation.** They do not include every constant, spline-generation cost, collision-evaluation cost, Python-loop overhead, or GPU synchronization effect.

These points should be stated explicitly in any academic analysis of the generated results.

---

## Scientific Background

### 1. Dynamic obstacle model

Each obstacle is represented by `PedestrianObstacle`. Static obstacles remain at a fixed center. Dynamic obstacles follow a bounded harmonic trajectory.

```math
x_i(t) = \operatorname{clip}\left(b_{x,i} + A_{x,i}\sin(\omega_i t + \phi_i),\;0.5,\;W-0.5\right)
```

```math
y_i(t) = \operatorname{clip}\left(b_{y,i} + A_{y,i}\cos(\omega_i t + \phi_i),\;0.5,\;H-0.5\right)
```

Here, $b_x$ and $b_y$ are base coordinates, $A_x$ and $A_y$ are motion amplitudes, $\omega$ is the angular frequency parameter used by the implementation, $\phi$ is the phase shift, and $W$ and $H$ are the workspace width and height. Static obstacles bypass the harmonic motion and use their base coordinates directly.

The default workspace is `10 × 10` with a grid resolution of `1.0`. The default robot start is `(0,0)` and the goal is `(6,7)`.

### 2. Occupancy grid construction

For grid-based planning, each cell is marked occupied when its center lies within the obstacle radius plus a safety margin. With cell center $c$ and obstacle center $o_i$:

```math
\|c-o_i\|_2 \leq r_i + m
```

where $r_i$ is the obstacle radius and the default safety margin is `0.4`.

The implementation evaluates the grid on the selected device and returns an integer occupancy array.

### 3. Continuous trajectory representation

Continuous planners optimize `num_via_points` two-dimensional points between the fixed start and goal. The complete control-point sequence is

```math
P = [p_{start}, v_1, v_2, \ldots, v_K, p_{goal}]
```

`generate_spline_path` then constructs a SciPy parametric spline with polynomial degree up to three and samples it at a fixed number of points. For fewer than three control points, the function falls back to linear interpolation. If spline construction fails, it falls back to distance-parameterized interpolation along the control points.

The function clips sampled coordinates to the fixed `10 × 10` workspace and then explicitly restores the first and last points to the exact requested start and goal coordinates.

### 4. Path length

For sampled path points $p_j$, the implementation computes the discrete path length as

```math
L = \sum_{j=0}^{T-2} \|p_{j+1}-p_j\|_2
```

### 5. Smoothness / bending-energy proxy

The code approximates the integral of squared curvature using changes in segment heading. For consecutive segment lengths $\Delta s_j$ and heading changes $\Delta\theta_j$, the implemented metric is

```math
S = \sum_j \frac{(\Delta\theta_j)^2}{\Delta s_j + 10^{-4}}
```

The small regularization term prevents division by a zero or near-zero local arc-length estimate.

This is a discrete curvature-energy proxy, not an exact symbolic integral of the continuous spline curvature.

### 6. Spatiotemporal collision evaluation

The planner evaluates obstacle positions at the time associated with each sampled path point. With path traversal speed $v$, the elapsed time at point $j$ is computed from cumulative sampled segment length:

```math
t_j = t_0 + \sum_{k=0}^{j-1}\frac{\Delta s_k}{\max(v,10^{-3})}
```

The default planners use `speed = 1.0`, so the numerical traversal time equals cumulative path length in workspace units.

For a path point $p_j$ and obstacle $i$, the signed boundary distance used in the implementation is

```math
d_{ij} = \|p_j-o_i(t_j)\|_2-r_i
```

The minimum clearance before reporting is the minimum value of $d_{ij}$ across all sampled path points and obstacles, then clipped to zero for the reported metric:

```math
c_{min} = \max\left(0,\min_{i,j} d_{ij}\right)
```

The collision counter is implementation-specific: it counts path-point/obstacle evaluations with $d_{ij} \leq 0.05$. Therefore, `Collision_Count` should be interpreted as a **near-contact / collision-threshold count over sampled spatiotemporal evaluations**, not as the number of distinct continuous collision events.

### 7. Shared optimization objective

PSO, LDW-PSO, D*-PSO, ABC, PSO-ABC, and SMO use the same cost structure:

```math
J = L + 0.1S + 500C + P_{clear}
```

where $L$ is path length, $S$ is the smoothness metric, and $C$ is the collision-threshold count.

The clearance penalty is active only when the reported minimum clearance is below `0.3`:

```math
P_{clear} = \frac{50}{c_{min}+0.1}
```

Otherwise, $P_{clear}=0$.

Because the reported clearance is clamped to a nonnegative value, the penalty remains numerically finite even when the underlying signed distance is negative.

---

## Methodology

### End-to-end pipeline

The benchmark follows this sequence:

1. Generate the synthetic pedestrian dataset and metadata.
2. Load `scenario_1` into `NavigationEnvironment`.
3. Instantiate all eight planners with fixed benchmark defaults.
4. Run each planner `num_runs` times using deterministic seed offsets.
5. Measure wall-clock execution time with CUDA synchronization before and after planning when CUDA is available.
6. Track peak memory using `tracemalloc` for CPU memory and PyTorch CUDA peak allocation for GPU memory.
7. Validate every returned path for finiteness, shape, exact endpoints, and workspace bounds.
8. Recompute path length, smoothness, minimum clearance, and collision count using the shared evaluator.
9. Export aggregate metrics, validation records, representative paths, dataset summaries, and figures.
10. Write an audit report and package the generated artifacts into a ZIP archive.

### Planner-specific methodology

#### D* Lite-style planner

`DStarPlanner` operates on the occupancy grid with 8-connected neighborhood expansion. It stores `g` and `rhs` values and uses a lexicographic two-component priority key.

For a node $s$, the implementation uses the standard D* Lite form:

```math
k(s) = \left[\min(g(s),rhs(s)) + h(s_{start},s) + k_m,\;\min(g(s),rhs(s))\right]
```

The one-step look-ahead value is updated from neighboring states:

```math
rhs(s) = \min_{s'\in N(s)}\left[g(s') + c(s,s')\right]
```

The heuristic is Euclidean distance on grid coordinates. Diagonal transitions therefore cost $\sqrt{2}$ and axis-aligned transitions cost $1$ when the grid resolution is `1.0`.

A critical implementation detail is that `plan()` clears `g`, `rhs`, and the open list and sets `k_m = 0` every time it is called. Thus, although the class implements D* Lite-style state variables and update logic, the default benchmark usage is effectively a **fresh graph-planning computation per call**, not a persistent incremental replanning experiment across successive map updates.

If no feasible path is recovered, the planner returns the two-point fallback `[start, goal]`, which may later receive collision penalties during evaluation.

#### PSO

`PSOPlanner` encodes a candidate solution as a flat vector containing `num_via_points * 2` coordinates. The particle velocity follows

```math
v_i \leftarrow wv_i + c_1r_1\odot(pbest_i-x_i) + c_2r_2\odot(gbest-x_i)
```

followed by coordinate and velocity clipping.

For the default configuration:

* `num_particles = 40`
* `num_via_points = 4`
* `num_iterations = 60`
* `c1 = 1.5`
* `c2 = 1.5`
* coordinate bounds = `[0, 10]`
* velocity bounds = `[-1, 1]`

The initial population is centered around either uniformly distributed points along the start-goal segment or a D* guide for the hybrid method, with Gaussian perturbations of standard deviation `0.8`.

#### LDW-PSO

The LDW variant uses the same PSO update but replaces the fixed inertia with the implementation's linearly decreasing weight:

```math
w_t = w_{max} - (w_{max}-w_{min})\frac{t}{I}
```

with `w_max = 0.9`, `w_min = 0.4`, and `I = 60` in the default configuration.

Because the loop runs from `t = 0` to `t = 59`, the final applied value is approximately `0.4083`; the code does not apply exactly `0.4` on an optimization iteration.

The non-LDW PSO baseline uses a fixed weight of `0.5`.

#### D*-PSO hybrid

`DStarPSOHybrid` combines two stages:

1. Generate a discrete global route using `DStarPlanner`.
2. Sample that route to initialize the four continuous via-points of an LDW-PSO search.

The resulting spline is therefore a continuous refinement initialized from the discrete planner's geometry. The method does **not** alter the D* objective itself; the hybridization occurs at the initialization interface between the discrete path and the continuous optimizer.

#### Artificial Bee Colony

`ABCPlanner` maintains a population of candidate via-point vectors and applies three phases per iteration:

* **Employed bees:** propose a partner-difference perturbation.
* **Onlooker bees:** sample sources according to fitness-derived probabilities and perform additional local search.
* **Scout bees:** reset stagnant solutions once the trial counter reaches `limit`.

The implemented candidate update is

```math
v_i = x_i + \phi\odot(x_i-x_k)
```

where $k\neq i$ is a randomly selected partner and each component of $\phi$ is sampled uniformly from `[-1, 1]`.

Default settings are `40` bees, `60` iterations, and `limit = 10`.

#### PSO-ABC hybrid

`PSOABCHybrid` combines two exploration mechanisms in one update:

```math
x_i^{new} = x_i + v_i + \phi\odot(x_i-x_k)
```

The velocity component follows a PSO-style update with a decreasing inertia weight, while the additional partner-difference term provides ABC-style population exploration. The implementation uses `40` individuals, `60` iterations, `c1 = c2 = 1.5`, velocity clipping to `[-1, 1]`, and a stagnation limit of `8`.

#### Spider Monkey Optimization

`SMOPlanner` splits the population into two groups by default and performs a local-leader phase followed by a global-leader phase. Candidate solutions are formed by combining movement toward a leader with a randomly selected population difference term.

The implementation uses:

* `40` monkeys
* `2` groups
* `60` iterations
* `4` via-points

Unlike ABC and PSO-ABC, the code does not implement a separate trial-based random restart mechanism in SMO.

#### Ant Colony Optimization

`ACOPlanner` operates directly on the discrete occupancy grid. The pheromone state is stored as a tensor indexed by source and destination grid coordinates.

For a candidate neighbor $j$ of the current node $i$, the implementation defines heuristic desirability from Euclidean distance to the goal:

```math
\eta_{ij} = \frac{1}{d(j,goal)+10^{-3}}
```

and uses the transition weight

```math
w_{ij} = \tau_{ij}^{\alpha}\eta_{ij}^{\beta}
```

with probabilities proportional to these weights.

Pheromone evaporates as

```math
\tau \leftarrow (1-\rho)\tau
```

and each successful ant deposits

```math
\Delta\tau = \frac{10}{\max(10^{-2},L_{ant})}
```

on every traversed directed edge.

The default configuration is `30` ants, `40` iterations, `alpha = 1.0`, `beta = 2.0`, and `rho = 0.2`. Ant walks are self-avoiding and use 8-connected free grid neighbors.

---

## Dataset and Environment Specification

The default generator creates:

| Property                |               Default |
| ----------------------- | --------------------: |
| Workspace width         |                `10.0` |
| Workspace height        |                `10.0` |
| Grid resolution         |                 `1.0` |
| Start                   |          `(0.0, 0.0)` |
| Goal                    |          `(6.0, 7.0)` |
| Scenarios generated     |                   `5` |
| Time steps per scenario |                 `100` |
| Obstacles per scenario  |                   `8` |
| Static obstacles        |                   `3` |
| Dynamic obstacles       |                   `5` |
| Dataset records         | `5 × 100 × 8 = 4,000` |
| Random seed             |                  `42` |

The eight default obstacles are defined directly in the source code. Three are static; five follow harmonic trajectories with different amplitudes, frequencies, and phase shifts.

The dataset generator exports:

* `data/pedestrian_navigation_dataset.csv`
* `data/pedestrian_navigation_metadata.json`

The CSV contains scenario ID, time step, obstacle ID, dynamic/static status, position, and radius. The JSON stores the full scenario configuration used to reconstruct each environment.

---

## Implementation Details

### Core dependencies

The implementation uses:

| Library                 | Purpose                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| Python standard library | Serialization, timing, random seeds, ZIP packaging, filesystem handling, priority queues |
| NumPy                   | Array operations, trajectory generation, numerical geometry                              |
| pandas                  | Tabular dataset and benchmark result export                                              |
| Matplotlib              | Benchmark figures and trajectory visualization                                           |
| SciPy                   | Parametric spline interpolation via `splprep` / `splev`                                  |
| tqdm                    | Progress bars                                                                            |
| PyTorch                 | Tensor operations, batched cost evaluation, CUDA acceleration, random tensors            |

### Runtime device selection

The source automatically selects CUDA when `torch.cuda.is_available()` is true and otherwise falls back to CPU.

The selected device and GPU name are included in the runtime summary. CUDA synchronization is used before and after timed planner execution so that asynchronous GPU work is included more faithfully in the reported wall-clock interval.

### Reproducibility controls

The source fixes the global seed to `42` for Python `random`, NumPy, and PyTorch. During benchmark repetitions, it uses run-specific seeds:

```text
42, 52, 62, 72, 82
```

when `num_runs = 5`.

CUDA random state is also reseeded when CUDA is available.

This improves reproducibility, but **bitwise-identical results across different GPU models, driver versions, PyTorch versions, or library versions are not guaranteed**.

---

## Installation

### Google Colab

The script was authored for a Colab workflow and contains Colab-specific helpers for notebook export and browser-based file download.

Recommended runtime:

```text
Runtime: Python 3
Hardware accelerator: GPU
```

### Local Python environment

Use Python 3 and install the required third-party packages:

```bash
python -m pip install --upgrade numpy pandas matplotlib scipy tqdm torch
```

For a CUDA-enabled local PyTorch installation, use the PyTorch wheel appropriate for the local CUDA/runtime configuration instead of assuming that the generic `pip install torch` command matches every GPU environment.

---

## Usage

### Run the complete benchmark

The provided source executes the full benchmark at module level. Running the file therefore performs the benchmark and then creates the audit report and submission archive.

```bash
python assignment_solution.py
```

The benchmark defaults are:

```python
run_full_pipeline(
    num_scenarios=5,
    time_steps=100,
    num_runs=5,
)
```

The source then calls:

```python
write_audit_report(run_summary)
package_submission()
```

and, in Colab, attempts to trigger a browser download of the ZIP archive.

### Important execution behavior

Because benchmark execution is placed at module level, **importing the script is not a lightweight library import**. Importing it will execute the complete benchmark unless the source is refactored to place the main execution block behind an `if __name__ == "__main__":` guard.

For reproducible research packaging, that refactoring would be a sensible future improvement.

---

## Outputs

A successful run is designed to create the following structure:

```text
data/
├── pedestrian_navigation_dataset.csv
└── pedestrian_navigation_metadata.json

outputs/
├── run_summary.json
├── tables/
│   ├── computation_costs_and_metrics.csv
│   ├── path_validation_results.csv
│   ├── planned_paths.csv
│   └── dataset_summary.csv
└── figures/
    ├── path_planning_comparison.png
    ├── dynamic_pedestrian_navigation.png
    └── computation_costs_chart.png

assignment_solution.py
assignment_solution.ipynb
changes_audit_and_validation_report.md
assignment_submission_package.zip
```

Not every file is guaranteed to exist outside the intended Colab workflow: the notebook export helper specifically depends on Colab internals, and the source package includes only files that exist at packaging time.

---

## Benchmark Metrics

For each algorithm, the benchmark records the mean and standard deviation of execution time across runs and the mean of several trajectory-level metrics.

| Metric               | Meaning                                                                     |
| -------------------- | --------------------------------------------------------------------------- |
| `Time_Mean_Sec`      | Mean wall-clock execution time per run                                      |
| `Time_Std_Sec`       | Standard deviation of execution time                                        |
| `Peak_Memory_KB`     | Mean measured peak memory indicator across runs                             |
| `Path_Length_Mean`   | Mean sampled path length                                                    |
| `Path_Length_Std`    | Standard deviation of sampled path length                                   |
| `Smoothness_Mean`    | Mean discrete curvature-energy proxy                                        |
| `Min_Clearance_Mean` | Mean reported minimum obstacle clearance                                    |
| `Collisions_Mean`    | Mean sampled collision-threshold count                                      |
| `Time_Complexity`    | Simplified theoretical complexity string stored by the implementation       |
| `Space_Complexity`   | Simplified theoretical space complexity string stored by the implementation |

### Path validation

Every benchmark run is separately checked for:

* finite coordinates;
* two-dimensional path shape with at least two points;
* start-point error below `1e-6`;
* goal-point error below `1e-6`;
* all coordinates inside the environment bounds.

The exported `valid` field is true only when all of these conditions are satisfied.

---

## Results

The source code does **not** embed a fixed numerical results table. Numerical benchmark values are generated at execution time and written to:

```text
outputs/tables/computation_costs_and_metrics.csv
```

Path-validation records are written to:

```text
outputs/tables/path_validation_results.csv
```

and aggregate runtime metadata are written to:

```text
outputs/run_summary.json
```

Therefore, this README intentionally does not report algorithm rankings, percentage improvements, mean scores, or statistical significance claims that are not present in the provided source artifact.

The generated figures are:

1. **`path_planning_comparison.png`** — trajectory comparison for the eight planners in the loaded scenario.
2. **`dynamic_pedestrian_navigation.png`** — four snapshots of the D*-PSO trajectory against obstacle positions at selected times.
3. **`computation_costs_chart.png`** — execution-time and smoothness trade-off visualization.

The four visualization times are `[0, 3, 6, 9]` seconds. The plotted robot locations are selected by path index rather than by solving the continuous inverse mapping from traversal time to path position; therefore, the snapshot visualization should be interpreted as a qualitative trajectory-evasion figure rather than as an exact simulated robot clock trace.

---

## Computational Complexity

The implementation records the following simplified estimates:

| Algorithm | Time estimate stored by code                                    | Space estimate stored by code |
| --------- | --------------------------------------------------------------- | ----------------------------- |
| D*        | $O((\lvert V \rvert+\lvert E \rvert)\log\lvert V \rvert)$       | $O(\lvert V \rvert)$          |
| PSO       | $O(IND)$                                                        | $O(ND)$                       |
| LDW-PSO   | $O(IND)$                                                        | $O(ND)$                       |
| D*-PSO    | $O((\lvert V \rvert+\lvert E \rvert)\log\lvert V \rvert + IND)$ | $O(\lvert V \rvert+ND)$       |
| ABC       | $O(IND)$                                                        | $O(ND)$                       |
| PSO-ABC   | $O(IND)$                                                        | $O(ND)$                       |
| SMO       | $O(IND)$                                                        | $O(ND)$                       |
| ACO       | $O(IM\lvert V \rvert)$                                          | $O(\lvert V \rvert^2)$        |

where $I$ is the number of optimization iterations, $N$ is the population size, $D$ is the continuous search dimension, $M$ is the number of ants, and $|V|$ and $|E|$ denote graph vertices and edges.

These are **high-level asymptotic labels used by the program**, not complete end-to-end performance models. In particular, continuous planners repeatedly construct splines and perform spatiotemporal collision evaluation, while ACO performs Python-level path construction and maintains a source-destination pheromone tensor.

---

