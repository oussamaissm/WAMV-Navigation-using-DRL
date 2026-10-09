# WAM-V Deep Reinforcement Learning: Goal Navigation

Deep reinforcement learning for autonomous point-to-point navigation of a **WAM-V** (Wave Adaptive Modular Vessel) unmanned surface vehicle. Three continuous-control algorithms, **SAC**, **PPO** and **TD3**, are trained and compared on the same goal-reaching task.

---

## Table of Contents

- [Overview](#overview)
- [Task Description](#task-description)
- [Algorithms](#algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Analysis](#analysis)

---

## Overview

The goal of this project is to learn a control policy that drives the WAM-V from a fixed start position to a target position, without a hand-tuned controller. Policies are trained with model-free DRL and evaluated on four random goal points inside a polygonal circuit.

## Task Description

| Item | Value |
|---|---|
| Vehicle | WAM-V USV |
| Start position | $(-800, 450)$ m |
| Goal points | P1 $(-825.1, 518.7)$, P2 $(-753.6, 437.7)$, P3 $(-888.4, 499.2)$, P4 $(-839.2, 420.7)$ |
| Action space | Continuous $\mathbf{a}_t = [a_L, a_R] \in [-1,1]^2$, mapped to left/right thruster thrust $[0,1000]$ |
| Observation space | $`\mathbf{o}_t = [e_x^b,\ e_y^b,\ e_\psi,\ u,\ v,\ r,\ v_d]`$: body-frame goal position error, heading error, surge/sway velocities, yaw rate, and desired speed |
| Reward | $R_t = -2.0d_t - 0.5\lvert e_{\psi,t}\rvert - 0.25\lvert v_t-v_d\rvert + 2.0\Delta d_t - 0.05\frac{\lvert T_L\rvert+\lvert T_R\rvert}{1000}$ |
| Simulator | Gazebo / VRX |

## Algorithms

| Algorithm | Type |
|---|---|
| **PPO** (Proximal Policy Optimization) | On-policy, stochastic actor, clipped policy objective, GAE |
| **SAC** (Soft Actor-Critic) | Off-policy, stochastic actor, twin critics, entropy regularization, automatic temperature tuning |
| **TD3** (Twin Delayed DDPG) | Off-policy, deterministic actor, twin critics, target policy smoothing, delayed policy updates |

## Evaluation Metrics

Each policy is evaluated on the four goal points (P1 to P4), and the mean over them is reported:

- **Time to goal [s]**: time until the goal is reached.
- **Path length [m]**: total distance travelled.
- **Rest distance to goal [m]**: distance between the final position and the goal.
- **Total reward**: cumulative episode reward.

In the bar charts, **faded bars mean the goal was not reached** for that point; solid bars are successful runs (and the means).

## Training Hardware and Software

All DRL experiments were trained locally using the same hardware and software environment to ensure a fair comparison between **SAC, PPO, and TD3**.

### Hardware

| Component            | Specification                                                      |
| -------------------- | ------------------------------------------------------------------ |
| **Machine**          | Lenovo ThinkPad L390                                               |
| **CPU**              | Intel Core i5-8365U @ 1.60 GHz (4 cores / 8 threads)               |
| **RAM**              | 16 GB                                                              |
| **Operating System** | Ubuntu 24.04.4 LTS                                                 |

### Software

| Component                   | Technology   |
| --------------------------- | ------------ |
| **Deep Learning Framework** | PyTorch      |
| **RL Environment API**      | Gymnasium    |
| **Robotics Middleware**     | ROS 2 Jazzy  |
| **Simulator**               | Gazebo / VRX |
| **Programming Language**    | Python       |

The DRL agents interact with the WAM-V simulation through a custom **Gymnasium environment**, which defines the observation space, continuous action space, reward function, episode termination conditions, and interaction with the Gazebo/VRX simulation.

All training was performed on the CPU.

## Results

### Metric comparison

![Algorithm comparison](Test%20results/metrics.png)

### Trajectories

Squares mark the final rest position of each vehicle.

![Trajectories](Test%20results/trajectories.png)

## Reward Weight Optimization

In some experiments, the WAM-V may take inefficient turns, make unnecessary corrections, or spend too much time aligning its heading. To investigate how reward coefficients can be selected systematically, this experiment uses a data-driven procedure to estimate the relative importance of three trajectory-quality components: distance, progress, and heading error.

The method generates simulated trajectories, extracts trajectory-level features, constructs pairwise preferences using a reference quality function, and estimates reward weights using a **Bradley–Terry preference model**.

The optimization learns three reward weights:

- $w_D$: distance-related reward weight.
- $w_P$: progress reward weight.
- $w_H$: heading-related reward weight.

A fourth parameter, $\beta$, controls the sensitivity of the preference model.

> **Important:** This experiment uses a simplified point-mass simulator with drift and stochastic noise. It is intended to study reward-weight estimation and does not reproduce the full WAM-V/VRX dynamics.

### 1. Trajectory generation

The experiment generates $N=10{,}000$ simulated trajectories, starting from a common initial position and navigating toward the same target.

The initial position and target are:

```math
\mathbf{p}_0=(0,0), \qquad \mathbf{g}=(10,10)
```

The simplified simulation dynamics are:

```math
\mathbf{p}_{t+1}
=
\mathbf{p}_t
+
d\frac{\mathbf{g}-\mathbf{p}_t}
{\|\mathbf{g}-\mathbf{p}_t\|}
\Delta t
+
\sigma\sqrt{\Delta t}\boldsymbol{\epsilon}_t
```

where:

- $\mathbf{p}_t$ is the current position.
- $\mathbf{g}$ is the target position.
- $d=0.5$ is the drift magnitude.
- $\sigma=1$ controls the stochastic noise.
- $\Delta t=0.1$ seconds.
- $\boldsymbol{\epsilon}_t\sim\mathcal{N}(0,I)$ is a two-dimensional Gaussian noise vector.

Each trajectory contains 800 simulation steps.

The simulator uses the same initial position and target for all trajectories, while stochastic noise produces different paths.

Example of trajectory simulation:

![Trajectories](Test%20results/trajectory_simulation.png)

### 2. Trajectory-quality features

Three primary trajectory metrics are extracted from each simulated trajectory: distance, progress, and heading error.

The implementation additionally constructs three nonlinear features for the reference quality function.

#### 2.1 Distance $D$

Distance measures the average normalized distance to the target throughout the trajectory:

```math
D=
\frac{1}{T+1}
\sum_{t=0}^{T}
\frac{\|\mathbf{g}-\mathbf{p}_t\|}{d_0}
```

where:

- $T=800$ is the number of simulation steps.
- $d_0$ is the initial distance to the target.
- $\mathbf{p}_t$ is the position at time $t$.

A lower value of $D$ indicates that the trajectory remains closer to the target on average.

#### 2.2 Progress $P$

Progress measures the reduction in distance to the target relative to the total distance travelled:

```math
P=
\frac{d_0-d_T}{\max(L,\epsilon)}
```

where:

- $d_0$ is the initial distance to the target.
- $d_T$ is the final distance to the target.
- $L$ is the total travelled path length.
- $\epsilon$ is a small positive constant used to avoid division by zero.

A higher value of $P$ indicates more efficient progress toward the target relative to the distance travelled.

#### 2.3 Heading error $H$

Heading error measures the average absolute angular difference between the direction of displacement and the direction toward the target.

The error is evaluated every $K=20$ simulation steps:

```math
H=
\frac{1}{N_c}
\sum_{k=1}^{N_c}
\frac{|e_{\psi,k}|}{\pi}
```

where:

- $e_{\psi,k}$ is the wrapped angular error at checkpoint $k$.
- $N_c$ is the number of checkpoints.

The angular error is wrapped to the interval $[-\pi,\pi)$.

A lower value of $H$ indicates better heading alignment.

#### 2.4 Nonlinear features

In addition to the three primary metrics, the implementation constructs three additional features:

```math
D^2,\qquad H^2,\qquad DH
```

The complete feature vector is:

```math
\mathbf{F}=
\begin{bmatrix}
D & P & H & D^2 & H^2 & DH
\end{bmatrix}^{T}
```

These features are used by the reference quality function to introduce nonlinear penalties.

**Importantly, the optimizer does not learn six independent reward weights.** It uses the first three features to learn the weights of the original distance, progress, and heading reward components. The additional features influence the reference preference ordering.

### 3. Training and test split

The 10,000 trajectories are randomly divided into two sets:

| Dataset | Percentage | Number of trajectories |
|---|---:|---:|
| Training | 80% | 8,000 |
| Test | 20% | 2,000 |
| Total | 100% | 10,000 |

The training set is used to estimate the reward weights and preference-model temperature. The test set is reserved for evaluating the resulting model.

### 4. Feature standardization and sign convention

Feature standardization is performed using the training-set statistics:

```math
z_f=\frac{f-\mu_f}{\sigma_f}
```

where $\mu_f$ and $\sigma_f$ are the training mean and standard deviation of feature $f$.

The same normalization statistics are then applied to the test set.

For the three primary features, the implementation uses the sign-adjusted representation:

```math
\mathbf{s}=
\begin{bmatrix}
-z(D)\\
+z(P)\\
-z(H)
\end{bmatrix}
```

This convention ensures that larger values correspond to better performance for each of the three reward components:

- Distance is negated because lower distance is better.
- Progress remains positive because higher progress is better.
- Heading error is negated because lower heading error is better.

The sign-adjusted training and test matrices initially contain all six features. Before optimization, the code selects only their first three columns.

Consequently, the learned reward model uses three weights rather than six.

### 5. Reference trajectory-quality function

Pairwise preferences are generated using a reference quality function.

In the implementation, the quality function is:

```math
\boxed{
Q(\tau)=
-z(D)+z(P)-z(H)
-z(D^2)-z(H^2)-z(DH)
}
```

Here, each $z(\cdot)$ represents the standardized value of the corresponding feature.

**Important implementation detail:** $z(D^2)$ means that the raw feature $D^2$ is standardized independently. It is not the same as $[z(D)]^2$. The same distinction applies to the other nonlinear features.

The reference quality function combines:

- Linear contributions from distance, progress, and heading error.
- A quadratic distance penalty.
- A quadratic heading penalty.
- A distance-heading interaction penalty.

The higher the reference quality score, the better the trajectory according to this synthetic evaluation criterion.

The reference function generates the preference labels. It is not itself the reward function whose three weights are being optimized.

### 6. Pairwise preference generation

The training trajectories are sorted from highest to lowest reference quality.

For each unique pair of trajectories $i<j$, the earlier trajectory is treated as preferred:

```math
i\succ j
\quad\Longleftrightarrow\quad
Q(\tau_i)>Q(\tau_j)
```

The preference model receives these pairwise rankings and the three sign-adjusted primary features.

It does not directly receive the reference quality scores as optimization targets.

For $N_{\text{train}}=8{,}000$ training trajectories, the number of unique pairs is:

```math
|\mathcal{P}_{\text{train}}|
=
\frac{N_{\text{train}}(N_{\text{train}}-1)}{2}
```

Therefore:

```math
\boxed{
|\mathcal{P}_{\text{train}}|
=
\frac{8000\times7999}{2}
=
31{,}996{,}000
}
```

The optimization uses all these pairs rather than sampling a subset.

For the 2,000 test trajectories:

```math
|\mathcal{P}_{\text{test}}|
=
\frac{2000\times1999}{2}
=
1{,}999{,}000
```

### 7. Bradley–Terry preference model

The Bradley–Terry model estimates the probability that trajectory $i$ is preferred over trajectory $j$:

```math
P(i\succ j)
=
\sigma\left(
\beta\mathbf{w}^{T}
(\mathbf{s}_i-\mathbf{s}_j)
\right)
```

where:

```math
\mathbf{w}=
\begin{bmatrix}
w_D & w_P & w_H
\end{bmatrix}^{T}
```

and:

- $\sigma(x)=1/(1+e^{-x})$ is the sigmoid function.
- $\mathbf{w}$ contains the three reward weights.
- $\beta$ is the learned inverse-temperature parameter.
- $\mathbf{s}_i-\mathbf{s}_j$ is the difference between the sign-adjusted primary features of two trajectories.

A higher predicted probability indicates a stronger preference for trajectory $i$.

The corresponding mean negative log-likelihood is:

```math
\mathcal{L}(\mathbf{w},\beta)
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
\log
\left[
1+
\exp\left(
-\beta\mathbf{w}^{T}
(\mathbf{s}_i-\mathbf{s}_j)
\right)
\right]
```

The optimization seeks weights that assign higher scores to trajectories preferred by the reference quality function.

### 8. Reward function and optimization constraints

The reward model being learned is:

```math
\boxed{
G(\tau)=
\beta
\left[
-w_Dz(D)
+w_Pz(P)
-w_Hz(H)
\right]
}
```

The weights satisfy:

```math
w_D+w_P+w_H=1
```

with:

```math
w_D,w_P,w_H\geq0
```

Each weight is also bounded above by 1.

The temperature is constrained by:

```math
0.1\leq\beta\leq200
```

The complete optimization problem is:

```math
\min_{\mathbf{w},\beta}
\mathcal{L}(\mathbf{w},\beta)
```

subject to the weight and temperature constraints above.

The three weights describe the relative importance of the primary reward components. The temperature controls how strongly the model translates score differences into predicted preferences.

Because the weights sum to one, they define a normalized distribution of relative importance.

### 9. Optimization methodology

The parameters are optimized using **Sequential Least Squares Programming (SLSQP)** from SciPy.

The optimization vector is:

```math
\mathbf{x}=
\begin{bmatrix}
w_D & w_P & w_H & \log\beta
\end{bmatrix}^{T}
```

The temperature is recovered using:

```math
\beta=e^{x_4}
```

This parameterization ensures that the temperature remains positive.

The optimization is initialized with approximately equal weights:

```math
\mathbf{w}_0=(0.33,0.33,0.33)
```

and:

```math
\beta_0=1
```

The maximum number of SLSQP iterations is 200.

The resulting parameters are:

```math
\mathbf{w}^{*}=
\begin{bmatrix}
w_D^{*} & w_P^{*} & w_H^{*}
\end{bmatrix}^{T}
```

and:

```math
\beta^{*}
```

### 10. Training and held-out evaluation

After optimization, the learned parameters are evaluated on the held-out test set.

The evaluation reports:

**Training loss**

The mean Bradley–Terry negative log-likelihood over all 31,996,000 training pairs.

**Test loss**

The mean Bradley–Terry negative log-likelihood over all 1,999,000 test pairs.

**Pairwise accuracy**

The fraction of test pairs for which the learned reward assigns a higher score to the trajectory preferred by the reference quality function.

The accuracy is defined as:

```math
\operatorname{Accuracy}
=
\frac{
\left|
\left\{
(i,j)\in\mathcal{P}_{\text{test}}:
G(\tau_i)>G(\tau_j)
\right\}
\right|
}{
|\mathcal{P}_{\text{test}}|
}
```

A higher pairwise accuracy indicates better agreement with the reference ranking.

The test loss and pairwise accuracy are calculated using the learned training parameters without retraining on the test trajectories.

### 11. Results

The optimization converged successfully using SLSQP, with the exact all-pairs Bradley–Terry loss evaluated over 31,996,000 training pairs.

The estimated parameters are:

| Parameter | Estimated value | Interpretation |
|---|---:|---|
| $w_D^*$ | 0.440010 | Relative importance of distance |
| $w_P^*$ | 0.171709 | Relative importance of progress |
| $w_H^*$ | 0.388282 | Relative importance of heading quality |
| $\beta^*$ | 117.104 | Learned inverse temperature |
| Weight sum | 1.000000 | Satisfies the normalization constraint |
| Training loss | 0.01082487 | Mean Bradley–Terry negative log-likelihood |
| Test loss | 0.010998 | Mean Bradley–Terry negative log-likelihood on held-out pairs |
| Test pairwise accuracy | 99.60% | Agreement with the reference ranking |

The optimization terminated successfully, and the learned weights satisfy the constraints:

```math
w_D^*+w_P^*+w_H^*=1
```

with all three weights non-negative.

The estimated weights suggest prioritizing distance-related performance and heading quality, while retaining a smaller contribution from progress.

The candidate normalized weights are:

```math
\boxed{
(w_D,w_P,w_H)
=
(0.440010,\ 0.171709,\ 0.388282)
}
```

The current results establish that the optimization successfully fitted a three-weight reward model to the synthetic preference ranking. Further experiments are necessary to determine whether these weights improve navigation performance in the full WAM-V simulation.

### 12. Application to WAM-V reinforcement learning

The resulting weights can be used as initial reward coefficients for experiments with reinforcement-learning algorithms such as SAC, PPO, and TD3.

The learned reward provides a data-driven alternative to selecting all three coefficients manually.

The overall workflow is:

1. Generate simulated trajectories.
2. Extract distance, progress, and heading metrics.
3. Construct a reference ranking using the quality function.
4. Generate all unique pairwise training preferences.
5. Optimize the three reward weights and inverse temperature.
6. Evaluate the learned model on held-out trajectories.
7. Transfer the candidate weights to the WAM-V reinforcement-learning environment.
8. Validate the resulting navigation behavior experimentally.