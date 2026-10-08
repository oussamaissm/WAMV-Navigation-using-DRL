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

In some experiments, the WAM-V could take inefficient turns or spend too much time correcting its heading. To improve the reward design, a data-driven procedure was used to estimate the relative importance of the main trajectory-quality features rather than relying entirely on manually selected reward coefficients.

The procedure generates a large set of simulated trajectories, extracts trajectory-level quality features, constructs pairwise preferences between trajectories, and learns reward weights using a **Bradley–Terry preference model**.

> **Note:** The current experiment uses a simplified point-mass simulator with drift and stochastic noise. It is used only to study reward-weight estimation and does not reproduce the full WAM-V/VRX dynamics.

### 1. Trajectory Generation

A total of $N=10{,}000$ trajectories are generated from the same initial position toward a common target.

The simplified dynamics are:

$$
\mathbf{p}_{t+1}
=
\mathbf{p}_t
+
d\frac{\mathbf{g}-\mathbf{p}_t}
{\lVert\mathbf{g}-\mathbf{p}_t\rVert}\Delta t
+
\sigma\sqrt{\Delta t}\boldsymbol{\epsilon}_t
$$

where:

* $\mathbf{p}_t$ is the trajectory position.
* $\mathbf{g}$ is the target position.
* $d=0.5$ is the drift magnitude.
* $\sigma=1$ controls stochastic noise.
* $\Delta t=0.1$ s.
* $\boldsymbol{\epsilon}_t\sim\mathcal{N}(0,I)$.

Each trajectory contains 800 simulation steps.

### 2. Trajectory-Quality Features

Three trajectory-level features are extracted.

#### Distance $D$

The average normalized distance to the target:

$$
D =
\frac{1}{T+1}
\sum_{t=0}^{T}
\frac{\lVert\mathbf{g}-\mathbf{p}_t\rVert}
{d_0}
$$

where $d_0$ is the initial distance to the target.

Lower $D$ indicates better navigation.

#### Progress $P$

Progress toward the target normalized by the travelled path length:

$$
P =
\frac{d_0-d_T}
{\max(L,\epsilon)}
$$

where $d_T$ is the final distance to the target and $L$ is the total travelled distance.

Higher $P$ indicates better navigation.

#### Heading Error $H$

Heading error is measured every $K=20$ simulation steps:

$$
H =
\frac{1}{N_c}
\sum_{k=1}^{N_c}
\frac{\lvert e_{\psi,k}\rvert}{\pi}
$$

where $e_{\psi,k}$ is the wrapped heading error and $N_c$ is the number of heading-error measurements.

Lower $H$ indicates better heading alignment.

### 3. Normalization and Sign Convention

The three features are standardized using statistics computed only on the training set:

$$
z_f = \frac{f-\mu_f}{\sigma_f}
$$

Because the three metrics do not have the same optimization direction, they are sign-adjusted so that **higher values always represent better trajectories**:

$$
\mathbf{s}
=
\begin{bmatrix}
-z(D)\\
+z(P)\\
-z(H)
\end{bmatrix}
$$

The trajectory score is then modeled as:

$$
G(\tau)
=
\beta
\left(
-w_D z(D)
+w_P z(P)
-w_H z(H)
\right)
$$

with:

$$
w_D+w_P+w_H=1,
\qquad
w_D,w_P,w_H\geq0
$$

The weights therefore represent the relative importance of distance, progress, and heading quality.

### 4. Preference Generation

The trajectories are ranked according to a trajectory-quality function. For every unique pair of trajectories $i<j$, the higher-quality trajectory is considered preferred:

$$
i \succ j
$$

For $N_{\text{train}}$ training trajectories, this produces:

$$
|\mathcal{P}|
=
\frac{N_{\text{train}}(N_{\text{train}}-1)}{2}
$$

pairwise comparisons.

With 8,000 training trajectories:

$$
|\mathcal{P}|
=
\frac{8000\times7999}{2}
=
31{,}996{,}000
$$

unique training pairs are evaluated.

### 5. Bradley–Terry Preference Model

The probability that trajectory $i$ is preferred over trajectory $j$ is modeled as:

$$
P(i\succ j)
=
\sigma
\left(
\beta\mathbf{w}^{T}
(\mathbf{s}_i-\mathbf{s}_j)
\right)
$$

where:

* $\sigma(x)=1/(1+e^{-x})$ is the sigmoid function.
* $\mathbf{w}=[w_D,w_P,w_H]$ contains the reward weights.
* $\beta$ is a learned inverse-temperature parameter.
* $\mathbf{s}_i-\mathbf{s}_j$ is the feature difference between the two trajectories.

The corresponding negative log-likelihood is:

$$
\mathcal{L}(\mathbf{w},\beta)
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
\log
\left[
1+
\exp
\left(
-\beta\mathbf{w}^{T}
(\mathbf{s}_i-\mathbf{s}_j)
\right)
\right]
$$

The optimization problem is:

$$
\min_{\mathbf{w},\beta}
\mathcal{L}(\mathbf{w},\beta)
$$

subject to:

$$
w_D+w_P+w_H=1
$$

$$
w_D,w_P,w_H\geq0
$$

and:

$$
0.1\leq\beta\leq200
$$

### 6. Optimization Methodology

The parameters are optimized using **Sequential Least Squares Programming (SLSQP)**.

The optimization variables are:

$$
\mathbf{x}
=
[w_D,w_P,w_H,\log\beta]
$$

Using $\log\beta$ guarantees that the temperature remains positive:

$$
\beta=e^{\log\beta}
$$

The exact loss and its gradient are evaluated over **all unique trajectory pairs**. To avoid storing all pairwise differences simultaneously, the pairs are processed in blocks of 512 trajectories.

This provides:

* Exact all-pairs maximum-likelihood estimation.
* $O(N^2)$ computational complexity.
* Reduced memory usage through block processing.

The learned parameters are:

$$
\theta^*
=
(w_D^*,w_P^*,w_H^*)
$$

and:

$$
\beta^*
$$

### 7. Equivalent Reward Coefficients

Because the optimization is performed on z-scored features, the learned model can also be expressed in terms of the original raw features.

Starting from:

$$
G
=
\beta
\left(
-w_D z(D)
+w_P z(P)
-w_H z(H)
\right)
$$

and using:

$$
z(f)=\frac{f-\mu_f}{\sigma_f}
$$

the equivalent raw-feature reward is:

$$
G
=
-a_DD+a_PP-a_HH+C
$$

where:

$$
a_D=\frac{\beta w_D}{\sigma_D},
\qquad
a_P=\frac{\beta w_P}{\sigma_P},
\qquad
a_H=\frac{\beta w_H}{\sigma_H}
$$

The constant $C$ does not affect the relative ranking of trajectories.

### 8. Train/Test Methodology

The 10,000 trajectories are randomly divided into:

* **80% training trajectories**
* **20% held-out test trajectories**

The feature normalization statistics are computed exclusively from the training set to avoid data leakage.

The reward weights and temperature are learned only from the training trajectories. The learned parameters are then evaluated on the held-out trajectories using the same Bradley–Terry loss and pairwise ranking procedure.

### 9. Results

The optimization produces the learned reward parameters:

$$
\theta^*
=
(w_D^*,w_P^*,w_H^*)
$$

and:

$$
\beta^*
$$

The resulting reward model is:

$$
G(\tau)
=
\beta^*
\left[
-w_D^*z(D)
+w_P^*z(P)
-w_H^*z(H)
\right]
$$

The experiment reports the following results:

| Result            | Description                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| $w_D^*$           | Learned importance of distance-to-goal                                         |
| $w_P^*$           | Learned importance of progress                                                 |
| $w_H^*$           | Learned importance of heading quality                                          |
| $\beta^*$         | Learned preference confidence / inverse temperature                            |
| Training loss     | Bradley–Terry negative log-likelihood on all training pairs                    |
| Test loss         | Bradley–Terry negative log-likelihood on all held-out pairs                    |
| Pairwise accuracy | Percentage of held-out trajectory pairs correctly ranked by the learned reward |

The pairwise accuracy is computed as:

$$
\text{Accuracy}
=
\frac{
\#\{(i,j):G(\tau_i)>G(\tau_j)\}
}{
|\mathcal{P}_{\text{test}}|
}
$$

This provides a direct measure of how well the learned reward reproduces the preference ordering on unseen trajectories.

### 10. Interpretation

The learned weights provide a quantitative indication of which trajectory characteristics are most important for the preference model.

For example, a larger $w_H$ means that heading quality contributes more strongly to the learned trajectory ranking, while a larger $w_P$ indicates that progress is more influential.

This approach can therefore be used as a **reward-design methodology** for the WAM-V DRL environment. Instead of selecting reward coefficients entirely by hand, trajectory preferences can be used to estimate the relative importance of the different reward components.

The learned coefficients can subsequently be incorporated into the DRL reward function and evaluated with SAC, PPO, and TD3.
