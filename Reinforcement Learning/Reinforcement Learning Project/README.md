# Robust Proximal Policy Optimization for `MountainCarContinuous-v0`

A reproducible implementation of **Proximal Policy Optimization (PPO)** for the continuous-action `MountainCarContinuous-v0` environment in Gymnasium. It uses a Gaussian actor-critic policy, Generalized Advantage Estimation (GAE), clipped PPO updates, entropy regularization, orthogonal initialization, deterministic evaluation, checkpointing, diagnostic export, and optional artifact download in Google Colab.

The repository documents a complete reinforcement-learning experiment rather than a minimal PPO example. It emphasizes explicit configuration, seed control, intermediate metric logging, evaluation artifacts, and machine-readable experiment metadata.

> **Implementation scope:** this README describes the uploaded Python source as implemented. It does not report experimental performance values that are not present in the source or in a supplied results artifact.

---

## Overview

The project trains an on-policy actor-critic agent with PPO on the one-dimensional continuous-control task `MountainCarContinuous-v0`.

The program:

1. Creates and configures the Gymnasium environment.
2. Seeds Python, NumPy, and PyTorch random number generators.
3. Normalizes observations into approximately the interval $[-1,1]$.
4. Uses separate two-hidden-layer multilayer perceptrons for the actor mean and critic.
5. Represents the policy with a state-independent trainable log standard deviation.
6. Collects fixed-length on-policy rollouts.
7. Computes GAE advantages and bootstrapped returns.
8. Performs multiple minibatch PPO optimization epochs.
9. Clips sampled actions to the environment's valid action interval.
10. Evaluates the learned policy deterministically over independent evaluation seeds.
11. Exports CSV metrics, plots, diagnostics, checkpoints, configuration metadata, a JSON summary, and an optional evaluation video.
12. Optionally archives the complete output directory and triggers a browser download when executed in Google Colab.

The implementation is contained in a single Python source file and uses the default training setup defined by the `Config` dataclass.

---

## Motivation

`MountainCarContinuous-v0` is a compact but informative continuous-control benchmark. The agent must control a car whose state consists of position and velocity and whose objective is to reach the target on the right-hand hill. Successful behavior depends on exploiting momentum: locally attractive actions are not necessarily sufficient to reach the goal.

The implementation addresses the exploration difficulty of this task by initializing the policy standard deviation at $1.0$ through `log_std = 0.0` and retaining a positive entropy bonus during training. These are engineering choices for exploration, not a demonstrated guarantee of avoiding local optima.

The task specification, action interval, observation bounds, reward structure, and termination behavior come from Gymnasium's `MountainCarContinuous-v0` environment. The official documentation is listed in the references section.

---

## Scientific Background

### Reinforcement Learning Formulation

The environment is modeled as a Markov decision process with state $s_t$, continuous action $a_t$, reward $r_t$, and transition dynamics. The neural-network actor parameterizes the policy $\pi_\theta(a_t \mid s_t)$, while a separate critic estimates the state value $V_\phi(s_t)$.

The discounted return from time step $t$ is

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k},
$$

where $\gamma \in [0,1)$ is the discount factor.

The implementation uses $\gamma = 0.99$.

### Proximal Policy Optimization

PPO updates the current policy with multiple minibatch gradient steps while limiting policy changes through a clipped probability ratio. For a sampled transition, the implementation computes

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)} {\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)} = \exp\!\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\mathrm{old}}}(a_t \mid s_t) \right).
$$

The implementation then uses the clipped surrogate form

$$
L_t^{\mathrm{CLIP}} = \min\!\left( r_t(\theta)\,\hat A_t,\, \operatorname{clip} \!\left(r_t(\theta),\,1-\epsilon,\,1+\epsilon\right)\hat A_t \right),
$$

where $\hat A_t$ is the estimated advantage and $\epsilon$ is the PPO clipping coefficient.

Because PyTorch minimizes the loss, the actor term is the negative of the clipped objective:

$$
\mathcal{L}_{\mathrm{actor}} = - \frac{1}{|\mathcal{B}|} \sum_{t\in\mathcal{B}} L_t^{\mathrm{CLIP}}.
$$

The implementation uses $\epsilon = 0.20$.

Schulman et al. introduced PPO as a policy-gradient method based on repeated minibatch optimization of a clipped surrogate objective. The original paper is listed in the references.

### Gaussian Policy

The actor predicts the mean of a Gaussian policy. For action dimension $i$,

$$
a_{t,i} \sim \mathcal{N} \!\left( \mu_{\theta,i}(s_t), \sigma_i^2 \right),
$$

where the standard deviation is represented through a trainable parameter

$$
\sigma_i = \exp(\log \sigma_i).
$$

The log standard deviation is state-independent in this implementation:

```python
self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
```

At initialization,

$$
\log \sigma_i = 0 \quad\Longrightarrow\quad \sigma_i = 1.
$$

The policy log-probability is summed across action dimensions:

$$
\log \pi_\theta(a_t\mid s_t) = \sum_i \log \mathcal{N} \!\left( a_{t,i}; \mu_{\theta,i}(s_t), \sigma_i^2 \right).
$$

Before `env.step(...)` is called, the environment action is clipped to the valid interval $[-1,1]$.

### Generalized Advantage Estimation

The implementation uses Generalized Advantage Estimation (GAE). It first computes the one-step temporal-difference residual

$$
\delta_t = r_t + \gamma V(s_{t+1}) \left(1-d_t\right) - V(s_t),
$$

where $d_t$ indicates a transition treated as terminal by the implementation.

The recursive GAE estimator is

$$
\hat A_t = \delta_t + \gamma\lambda \left(1-d_t\right) \hat A_{t+1},
$$

with $\lambda = 0.95$.

The code then constructs the value targets as

$$
\hat R_t = \hat A_t + V(s_t).
$$

GAE reduces the variance of policy-gradient estimates while providing a bias-variance trade-off controlled by $\lambda$.

### Entropy Regularization

The implementation adds an entropy bonus to discourage premature collapse of the Gaussian policy's exploration variance. For a Gaussian action distribution, PyTorch computes the entropy, which is summed over action dimensions.

The complete optimization loss is

$$
\mathcal{L} = \mathcal{L}_{\mathrm{actor}} + c_v\mathcal{L}_{\mathrm{critic}} - c_e\mathcal{H},
$$

where

- $\mathcal{L}_{\mathrm{actor}}$ is the negative clipped PPO surrogate objective.
- $\mathcal{L}_{\mathrm{critic}}$ is the value-function regression loss.
- $\mathcal{H}$ is the mean policy entropy.
- $c_v = 0.5$ is the value-loss coefficient.
- $c_e = 0.005$ is the entropy coefficient.

---

## Mathematical Formulation

### Observation Normalization

The source reads the environment's observation-space bounds and maps each observation component linearly into $[-1,1]$:

$$
\tilde{s}_j = \operatorname{clip} \left( 2 \frac{s_j-\ell_j}{u_j-\ell_j} -1,\, -1,\, 1 \right),
$$

where $\ell_j$ and $u_j$ are the lower and upper bounds for observation component $j$.

The normalized state is used as input to both the actor and critic.

### Actor-Critic Architecture

Let the normalized observation vector be $x\in\mathbb{R}^{d_s}$.

The actor mean network is a multilayer perceptron with two hidden layers:

$$
h_1 = \tanh(W_1x+b_1),
$$

$$
h_2 = \tanh(W_2h_1+b_2),
$$

$$
\mu_\theta(x)=W_3h_2+b_3.
$$

The critic uses the same hidden-layer pattern but produces a scalar value estimate:

$$
h^{V}_1 = \tanh(W^{V}_1x+b^{V}_1),
$$

$$
h^{V}_2 = \tanh(W^{V}_2h^{V}_1+b^{V}_2),
$$

$$
V_\phi(x)=W^{V}_3h^{V}_2+b^{V}_3.
$$

Both networks use a default hidden size of 64.

All linear layers use orthogonal initialization. The actor output layer uses an initialization scale of `0.01`, while the critic output layer uses a standard deviation of `1.0`.

### Critic Loss

The value function uses a mean squared-error objective scaled by $0.5$:

$$
\mathcal{L}_{\mathrm{critic}} = \frac{1}{2|\mathcal{B}|} \sum_{t\in\mathcal{B}} \left( V_\phi(s_t)-\hat R_t \right)^2.
$$

### Gradient Stabilization

After backpropagation, the source clips the global gradient norm:

```python
nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
```

with the default threshold

$$
\lVert g\rVert_2 \le 0.5
$$

after gradient clipping is applied.

---

## Environment

The target environment is:

```text
MountainCarContinuous-v0
```

Gymnasium defines a two-dimensional observation containing position and velocity and a one-dimensional continuous action with bounds $[-1,1]$.

In the implementation:

```python
ENV_ID = "MountainCarContinuous-v0"
```

The environment's observation bounds are read at runtime rather than hard-coded in the learning algorithm.

The training loop distinguishes:

- `terminated`: the environment reports successful task termination.
- `truncated`: the episode ends because of an external time limit.

The implementation treats either event as `done` when segmenting rollouts and resetting the environment.

---

## Methodology

### 1. Initialization

The training process:

- Loads Gymnasium.
- Seeds Python's `random` module.
- Seeds NumPy.
- Seeds PyTorch.
- Seeds CUDA generators when CUDA is available.
- Selects CPU or CUDA automatically unless explicitly configured.
- Creates the actor-critic model.
- Creates an Adam optimizer.

The default random seed is:

```text
42
```

### 2. Rollout Collection

The algorithm collects up to:

```text
2048 environment steps per rollout
```

while maintaining buffers for:

- normalized observations,
- raw sampled actions,
- old log-probabilities,
- rewards,
- done flags,
- value estimates.

The raw Gaussian sample is stored in the rollout buffer. Before interaction with the environment, the action is clipped:

```python
clipped_action = np.clip(raw_action_np, -1.0, 1.0)
```

### 3. GAE and Return Computation

After each rollout, the implementation estimates the next-state value and computes GAE advantages and return targets using:

```text
gamma       = 0.99
gae_lambda  = 0.95
```

### 4. PPO Optimization

For every collected rollout, the source performs:

```text
10 optimization epochs
```

with minibatches of size:

```text
64
```

For each minibatch it:

1. Re-evaluates the stored actions under the current policy.
2. Computes new log-probabilities.
3. Forms the PPO probability ratio.
4. Normalizes minibatch advantages.
5. Applies the clipped surrogate objective.
6. Computes the critic loss.
7. Adds entropy regularization.
8. Backpropagates.
9. Clips the gradient norm.
10. Performs an Adam optimizer step.

### 5. Episode Accounting

Training continues until:

```text
1000 completed episodes
```

have been recorded.

The source also tracks:

- episode return,
- episode length,
- success indicator.

The program prints progress every 20 completed episodes.

### 6. Checkpointing

Every 200 completed episodes, the program saves:

```text
checkpoints/agent_latest.pt
```

The checkpoint contains:

- current episode index,
- model state dictionary,
- optimizer state dictionary,
- serialized configuration.

### 7. Deterministic Evaluation

After training, the source evaluates the learned policy over:

```text
20 evaluation episodes
```

using deterministic action selection:

```python
action = model.get_action_and_value(..., deterministic=True)
```

This selects the actor mean rather than sampling from the Gaussian distribution.

Evaluation seeds are separated from training seeds by:

```text
eval_seed_offset = 100000
```

so evaluation episode $k$ uses:

$$
s_k = s_{\mathrm{train}} + 100000 + k.
$$

The evaluated action is clipped to the valid interval before it is passed to the environment.

---

## Configuration

The `Config` dataclass contains all primary hyperparameters.

| Parameter | Default | Role |
| --- | ---: | --- |
| `env_id` | `MountainCarContinuous-v0` | Gymnasium environment |
| `seed` | `42` | Training random seed |
| `total_episodes` | `1000` | Maximum completed training episodes |
| `rollout_steps` | `2048` | Environment steps collected per rollout |
| `max_steps_per_episode` | `999` | Maximum evaluation/training episode length used by the script |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE decay parameter |
| `clip_ratio` | `0.20` | PPO clipping range |
| `learning_rate` | `3e-4` | Adam learning rate |
| `train_epochs` | `10` | Optimization epochs per rollout |
| `minibatch_size` | `64` | PPO minibatch size |
| `value_coef` | `0.5` | Critic-loss coefficient |
| `entropy_coef` | `0.005` | Entropy-bonus coefficient |
| `max_grad_norm` | `0.5` | Global gradient-norm limit |
| `hidden_size` | `64` | Hidden-layer width |
| `eval_episodes` | `20` | Number of deterministic evaluation episodes |
| `eval_seed_offset` | `100000` | Separation between training and evaluation seeds |
| `checkpoint_every` | `200` | Checkpoint interval in completed episodes |
| `print_every` | `20` | Progress-report interval |
| `output_dir` | `/content/project_outputs_ppo` | Main artifact directory |
| `device` | `auto` | Automatic CPU/CUDA selection |
| `render_evaluation` | `True` | Enables RGB-frame capture for the first evaluation episode |
| `auto_download_colab` | `True` | Archives and downloads artifacts in Google Colab |

The final executable block constructs the configuration with 1000 episodes, the default output path, evaluation rendering enabled, and automatic Colab download enabled.

---

## Implementation Details

### Reproducibility Controls

The function `seed_everything` sets:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

and, when available:

```python
torch.cuda.manual_seed_all(seed)
```

This improves reproducibility across repeated executions, but exact bitwise reproducibility is not guaranteed because the source does not configure every PyTorch deterministic-kernel setting or pin package versions.

### Device Selection

`device="auto"` selects CUDA when PyTorch reports an available GPU and otherwise uses the CPU.

Requesting CUDA when no CUDA device is available raises an error rather than silently switching to CPU.

### Action Sampling

Training samples actions from:

```python
Normal(action_mean, action_std)
```

The standard deviation is derived from the trainable `actor_logstd`.

The source computes log-probabilities and entropy from the **unclipped Gaussian sample**, while the environment receives the **clipped** action. This matters when interpreting the algorithm as a mathematically exact bounded-action PPO formulation.

### Advantage Normalization

Advantages are normalized independently within each PPO minibatch:

$$
\hat A^{\mathrm{norm}}_t = \frac{\hat A_t-\mu_{\mathcal{B}}} {\sigma_{\mathcal{B}}+10^{-8}}.
$$

This is done immediately before computing the clipped actor objective.

### Optimizer

The source uses Adam:

```python
torch.optim.Adam(
    model.parameters(),
    lr=cfg.learning_rate,
    eps=1e-5
)
```

The learning rate is $3\times10^{-4}$ and $\epsilon_{\mathrm{Adam}}=10^{-5}$.

---

## Evaluation and Reported Metrics

Training metrics are written to:

```text
csv/episode_metrics.csv
```

with the following fields:

| Field | Meaning |
| --- | --- |
| `episode` | Completed episode index |
| `episode_return` | Sum of rewards within the episode |
| `episode_length` | Number of environment steps |
| `success` | `1` when the environment reports successful termination, otherwise `0` |

Evaluation metrics are written to:

```text
csv/evaluation_metrics.csv
```

with:

| Field | Meaning |
| --- | --- |
| `episode` | Evaluation episode index |
| `return` | Evaluation episode return |
| `length` | Evaluation episode length |
| `success` | Deterministic evaluation success indicator |

The final summary is written to:

```text
reports/final_results.json
```

and contains:

- total training episodes,
- mean return over the last 100 training episodes,
- success rate over the last 100 training episodes,
- mean evaluation return,
- evaluation success rate,
- mean evaluation episode length,
- maximum recorded position in the first evaluation diagnostic,
- wall-clock training time.

The uploaded Python source contains the mechanisms for producing these values but does **not** include a fixed set of final numeric experiment results. This README therefore does not report or imply specific performance numbers.

---

## Diagnostics and Visualization

### Training Curve

The script generates:

```text
plots/training_return.png
```

The plot contains:

- raw episode returns,
- a 50-episode moving average when at least 50 episodes are available.

The raw-return series comes directly from the stored episode metrics.

### Evaluation Diagnostics

For the first deterministic evaluation episode, the implementation stores:

- position,
- velocity,
- executed clipped action.

These arrays are saved as:

```text
logs/evaluation_diagnostics.npz
```

### Evaluation Video

When:

```text
render_evaluation = True
```

the source records RGB frames for the first evaluation episode and attempts to export:

```text
videos/evaluation_rollout.mp4
```

using `imageio` and the `libx264` codec.

If video export fails, the exception is reported and the remaining result serialization continues.

---

## Artifact Layout

After a successful training run, the configured output directory is organized as follows:

```text
project_outputs_ppo/
├── checkpoints/
│   └── agent_latest.pt
├── csv/
│   ├── episode_metrics.csv
│   └── evaluation_metrics.csv
├── logs/
│   └── evaluation_diagnostics.npz
├── metadata/
│   └── config.json
├── plots/
│   └── training_return.png
├── reports/
│   └── final_results.json
└── videos/
    └── evaluation_rollout.mp4
```

Optional artifacts may not exist in every run. For example, the evaluation video depends on rendering being enabled and video encoding succeeding.

---

## Project Structure

The uploaded source is a single Python module with the following logical components:

| Component | Responsibility |
| --- | --- |
| `Config` | Central experiment configuration |
| `seed_everything` | Random-seed initialization |
| `choose_device` | CPU/CUDA selection |
| `require_gym` | Runtime dependency validation |
| `layer_init` | Orthogonal neural-network initialization |
| `ActorCritic` | Gaussian actor and value-function networks |
| `normalize_obs` | Observation normalization |
| `compute_gae` | GAE advantages and return targets |
| `evaluate` | Deterministic policy evaluation and diagnostics |
| `download_colab_artifacts` | Artifact compression and optional browser download |
| `train` | End-to-end PPO training, logging, evaluation, and export |
| `smoke_test` | Minimal environment/model forward-pass validation |
| `__main__` block | Smoke test followed by configured training |

---

## Installation

When Gymnasium is unavailable, the source reports the following installation command:

```bash
pip install -q "gymnasium[classic-control]" imageio imageio-ffmpeg matplotlib
```

The program imports PyTorch and NumPy directly, so both must also be available in the Python environment.

A practical installation sequence is:

```bash
python -m pip install -U pip
python -m pip install numpy torch "gymnasium[classic-control]" imageio imageio-ffmpeg matplotlib
```

The uploaded source does not pin exact dependency versions. This establishes the required package families but does not guarantee bit-for-bit reproduction of a historical run.

---

## Usage

### Local Python Execution

Save the uploaded source as a Python file, for example:

```text
final_reinforcement_learning_project_ppo.py
```

Then execute:

```bash
python final_reinforcement_learning_project_ppo.py
```

The program runs the smoke test first and then starts PPO training.

### Google Colab

The source is structured for Google Colab and uses the output directory:

```text
/content/project_outputs_ppo
```

When `auto_download_colab=True`, the complete output directory is compressed into a ZIP archive and a browser download is requested in Colab.

A typical Colab setup is:

```python
!pip install -q "gymnasium[classic-control]" imageio imageio-ffmpeg matplotlib
```

Then run the uploaded script in the notebook or upload it to the Colab runtime.

---

## Smoke Test

Before training, the source runs `smoke_test()`.

The test checks that:

1. `MountainCarContinuous-v0` can be constructed.
2. The initial observation has shape `(2,)`.
3. The actor-critic model produces an action with shape `(1, 1)`.
4. The sampled action is finite.
5. The computed log-probability is finite.
6. The value estimate is finite.

A successful run prints:

```text
Smoke test: PASS
```

The smoke test is lightweight and checks basic integration and numerical sanity, not learning quality.

---

## Reproducibility

The experiment includes several reproducibility mechanisms:

- fixed default training seed (`42`);
- deterministic seed assignment for each training episode reset;
- separate deterministic evaluation seeds;
- serialized configuration in `metadata/config.json`;
- serialized model and optimizer state in checkpoints;
- CSV-based episode-level metrics;
- machine-readable JSON final summary;
- diagnostic arrays stored in NumPy's `.npz` format.

The training resets use:

$$
\text{seed}_{\mathrm{episode}} = 42+\text{episode index}.
$$

Evaluation uses a distinct offset of `100000`.

### Reproducibility Boundary

The following aspects are **not fully fixed by the uploaded source**:

- exact Python version;
- exact PyTorch version;
- exact NumPy version;
- exact Gymnasium version;
- GPU model and CUDA/cuDNN software stack;
- low-level deterministic-kernel settings;
- operating system;
- package lock file or environment manifest.

Therefore, the code is **seed-controlled and configuration-serialized**, but it is not a fully environment-pinned reproduction package.

For a research release, a stronger reproducibility package would also record exact dependency versions and hardware and software details.

---

## Limitations

The source is compact, but several implementation details are important for interpretation.

### 1. Clipped Environment Actions Versus Gaussian Log-Probabilities

Training samples an action from an unconstrained Gaussian distribution, stores that raw sample, computes its Gaussian log-probability, and only then clips the action to $[-1,1]$ before environment interaction.

This means the optimized probability model is not the exact probability density of the bounded action executed in the environment.

For this benchmark, the implementation may still work effectively, but it should not be described as a mathematically exact bounded-action Gaussian PPO formulation.

A principled alternative would model bounded actions explicitly, for example with a differentiable transformation such as a squashed Gaussian policy and the corresponding change-of-variables correction, or with another distribution that directly respects the environment's action support.

### 2. Time-Limit Truncation Handling

The rollout code sets:

```python
done = terminated or truncated
```

The GAE calculation uses a non-terminal mask that is zero for either event.

Consequently, a time-limit truncation is handled like true environment termination during bootstrapping.

This is consistent with the source, but it can introduce bias relative to a formulation that distinguishes environmental termination from externally imposed time limits.

### 3. Minibatch-Level Advantage Normalization

Advantages are normalized separately for each minibatch rather than once over the complete rollout. This changes the scaling applied to samples across minibatches and is a design choice of the implementation.

### 4. No Version Locking

The source contains no `requirements.txt`, lock file, Conda environment specification, Dockerfile, or recorded package-version manifest.

### 5. Single-Environment, Single-Agent Design

The implementation collects experience from one environment instance and does not use vectorized environments or parallel rollout workers.

### 6. No External Hyperparameter Sweep

The source defines one primary configuration rather than performing a systematic hyperparameter search. Consequently, the selected values should be understood as the project's chosen configuration, not as the result of a documented optimization study.

### 7. No Statistical Replication in the Source

The script uses a single training seed by default and evaluates deterministic behavior across 20 evaluation seeds. This provides a basic robustness check, but it is not a multi-seed study for estimating between-run variance.

---
