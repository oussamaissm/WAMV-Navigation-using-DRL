# WAM-V Deep Reinforcement Learning: Goal Navigation

Deep reinforcement learning for autonomous point-to-point navigation of a **WAM-V** (Wave Adaptive Modular Vessel) unmanned surface vehicle. Three continuous-control algorithms, **SAC**, **PPO** and **TD3**, are trained and compared on the same goal-reaching task.

> Sections marked `TODO` are placeholders: fill in the details of your own setup.

---

## Table of Contents

- [Overview](#overview)
- [Task Description](#task-description)
- [Algorithms](#algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Analysis](#analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

The goal of this project is to learn a control policy that drives the WAM-V from a fixed start position to a target position, without a hand-tuned controller. Policies are trained with model-free DRL and evaluated on four random goal points inside a polygonal circuit.

## Task Description

| Item | Value |
|---|---|
| Vehicle | WAM-V USV |
| Start position | (-800, 450) m |
| Goal points | P1 (-825.1, 518.7), P2 (-753.6, 437.7), P3 (-888.4, 499.2), P4 (-839.2, 420.7) |
| Episode time limit | ~383 s (episodes that hit this limit count as "goal not reached") |
| Action space | `TODO` (e.g. thrust / rudder, continuous) |
| Observation space | `TODO` (e.g. relative goal position, heading, velocities) |
| Reward | `TODO` (describe shaping: distance term, heading term, penalties...) |
| Simulator | `TODO` (e.g. Gazebo / VRX / custom) |

## Algorithms

| Algorithm | Type | Notes |
|---|---|---|
| **SAC** (Soft Actor-Critic) | Off-policy, maximum entropy | `TODO: hyperparameters` |
| **PPO** (Proximal Policy Optimization) | On-policy, clipped objective | `TODO: hyperparameters` |
| **TD3** (Twin Delayed DDPG) | Off-policy, deterministic | `TODO: hyperparameters` |

## Evaluation Metrics

Each policy is evaluated on the four goal points (P1 to P4), and the mean over them is reported:

- **Time to goal [s]**: time until the goal is reached (or the time limit if it is not).
- **Path length [m]**: total distance travelled.
- **Rest distance to goal [m]**: distance between the final position and the goal.
- **Total reward**: cumulative episode reward (closer to 0 is better).

In the bar charts, **faded bars mean the goal was not reached** for that point; solid bars are successful runs (and the means).

## Results

### Metric comparison

![Algorithm comparison](Test%20results/metrics.png)

### Trajectories

Squares mark the final rest position of each vehicle.

![Trajectories](Test%20results/trajectories.png)

### Summary

| Metric | SAC | PPO | TD3 |
|---|---|---|---|
| Mean time to goal [s] | ~293 | ~382 | ~382 |
| Mean path length [m] | ~265 | ~625 | ~195 |
| Mean rest distance [m] | ~36 | **~5** | ~190 |
| Mean total reward | ~-110,000 | **~-38,000** | ~-640,000 |

### Per-point results (approximate)

| Point | Metric | SAC | PPO | TD3 |
|---|---|---|---|---|
| P1 | Rest distance [m] | ~49 | ~5 | ~165 |
| P2 | Rest distance [m] | ~32 | ~5 | ~178 |
| P3 | Rest distance [m] | ~64 | ~6 | ~197 |
| P4 | Rest distance [m] | ~1 (reached, ~30 s) | ~5 | ~222 |

## Analysis

- **SAC** is the only algorithm that reached a goal (P4, in about 30 s with a ~47 m path, which is nearly a straight line). On P1 to P3 it overshoots the goal and makes large loops or spirals (especially on P2) and ends up 30 to 65 m away. It can solve the task but is inconsistent across goals.
- **PPO** gets closest to the goal on every point (about 5 m) and has the best mean reward, but never satisfies the goal condition. The trajectories show a near-straight approach followed by small circles around the goal. The ~620 to 640 m path lengths come from this loitering for the rest of the episode, not from a poor route. It looks like a station-keeping behaviour that does not settle within the goal tolerance.
- **TD3** fails to converge. It leaves the start along a curved path and finishes 165 to 220 m from the goal, far outside the circuit, which gives by far the worst reward.
- Overall, no algorithm is reliably successful yet. PPO is the most consistent in approach, SAC the only one with a successful run, and TD3 the weakest in this configuration.

> Caveat: results come from a single evaluation per goal point. Averaging over multiple seeds and episodes is needed before drawing firm conclusions.

## Installation

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`TODO`: add simulator / ROS / Gazebo installation steps if required.

## Usage

```bash
# Train
python train.py --algo sac   # or ppo / td3

# Evaluate and generate plots
python evaluate.py --algo sac ppo td3
```

`TODO`: replace with your actual scripts and arguments.

## Project Structure

```
.
├── assets/
│   ├── metrics.png          # metric comparison plots
│   └── trajectories.png     # trajectory plots
├── train.py                 # TODO
├── evaluate.py              # TODO
├── envs/                    # TODO: WAM-V environment
├── models/                  # TODO: saved policies
└── README.md
```

## Future Work

- Tune the reward function and goal tolerance. PPO ends about 5 m from the goal and loops there, which suggests the termination radius and the reward near the goal need adjusting.
- Train with multiple seeds and report mean and standard deviation.
- Add environmental disturbances (wind, waves, currents) and domain randomization.
- Add obstacle avoidance and path following along the circuit.
- Test curriculum learning for harder goal placements.

## License

`TODO`: add a license (e.g. MIT).
