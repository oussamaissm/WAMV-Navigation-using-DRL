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