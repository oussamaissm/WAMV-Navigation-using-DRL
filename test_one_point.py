"""
Send a trained WAM-V policy toward ONE specific (x, y) target and record
everything needed for a paper figure: the path taken, distance-to-goal
over time, heading error over time, speed over time, and reward over time.

Usage:
    python3 test_point.py --algo ppo --model_dir results_ppo/models/final \
        --goal_x -750 --goal_y 400

    python3 test_point.py --algo td3 --model_dir results_td3/models/final \
        --goal_x -750 --goal_y 400 --out_dir paper_figs/td3_point1
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import os, csv, argparse, math
import numpy as np
import torch
from drl_wamv import WamvEnv
import rclpy
from rclpy.signals import SignalHandlerOptions

STATE_DIM = 7
ACTION_DIM = 2


def load_policy(algo, model_dir, device):
    """Deterministic (noise-free) action function for the requested algorithm."""
    if algo == "ppo":
        from ppo import PPOAgent
        agent = PPOAgent(STATE_DIM, ACTION_DIM, device=device)
        agent.load(model_dir)
        agent.actor.eval()

        def act(state):
            s = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(s)  # no tanh squash in this PPO Actor; raw mean is the greedy action
            return np.clip(mean.cpu().numpy()[0], -1.0, 1.0).astype(np.float32)
        return act

    if algo == "td3":
        from td3 import TD3Agent
        agent = TD3Agent(STATE_DIM, ACTION_DIM, device=device)
        agent.load(model_dir)
        agent.actor.eval()
        return lambda state: agent.select_action(state, noise=False)

    if algo == "sac":
        from sac import SACAgent
        agent = SACAgent(STATE_DIM, ACTION_DIM, device=device)
        agent.load(model_dir)
        agent.actor.eval()
        return lambda state: agent.select_action(state, deterministic=True)

    raise ValueError(f"Unknown --algo '{algo}'. Choose from: ppo, td3, sac.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "td3", "sac"], required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--goal_x", type=float, required=True)
    parser.add_argument("--goal_y", type=float, required=True)
    parser.add_argument("--out_dir", default=None,
                         help="Defaults to test_point_results/<algo>_(<goal_x>,<goal_y>)")
    args = parser.parse_args()

    out_dir = args.out_dir or f"test_point_results/{args.algo}_({args.goal_x:.0f},{args.goal_y:.0f})"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    act = load_policy(args.algo, args.model_dir, device)
    print(f"Loaded {args.algo.upper()} policy from {args.model_dir} (device={device})")
    print(f"Target: ({args.goal_x:.2f}, {args.goal_y:.2f})")

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    env = WamvEnv(control_dt=0.2, max_episode_time=120.0,
                  trajectory_path=f"{out_dir}/_raw_env_trajectory.csv")

    traj_csv = f"{out_dir}/trajectory.csv"
    csv_f = open(traj_csv, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "step", "time_s", "x", "y", "goal_x", "goal_y", "distance_to_goal",
        "action_left", "action_right", "reward", "cumulative_reward",
        "heading_error_deg", "speed",
    ])

    rows = []  # kept in memory too, for plotting
    try:
        state, info = env.reset(goal=(args.goal_x, args.goal_y))
        xs, ys = [info["x"]], [info["y"]]
        step = 0
        cum_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = act(state)
            state, reward, terminated, truncated, info = env.step(action)
            step += 1
            cum_reward += float(reward)
            t = step * env.control_dt
            heading_err_deg = math.degrees(info["heading_error"])

            xs.append(info["x"]); ys.append(info["y"])
            row = [step, t, info["x"], info["y"], info["goal_x"], info["goal_y"],
                   info["distance_to_goal"], float(action[0]), float(action[1]),
                   reward, cum_reward, heading_err_deg, info["speed"]]
            rows.append(row)
            csv_w.writerow(row)

            print(f"step {step:4d} | t={t:6.1f}s | dist={info['distance_to_goal']:7.2f} m | "
                  f"hdg_err={heading_err_deg:6.1f} deg | reward={reward:7.2f}")

    finally:
        csv_f.close()
        env.close()
        rclpy.shutdown()

    success = bool(terminated)
    final_distance = info["distance_to_goal"]
    elapsed = info["elapsed_time"]
    print("\n===== Result =====")
    print(f"Reached goal: {'YES' if success else 'NO'}")
    print(f"Final distance to goal: {final_distance:.2f} m")
    print(f"Steps taken:             {step}")
    print(f"Time elapsed:            {elapsed:.1f} s")
    print(f"Cumulative reward:       {cum_reward:.2f}")
    print(f"Trajectory CSV:          {traj_csv}")

    # ---------------- Plots (for the paper) ----------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    rows = np.array(rows, dtype=float)
    t = rows[:, 1]
    dist = rows[:, 6]
    reward_series = rows[:, 9]
    cum_reward_series = rows[:, 10]
    hdg_err = rows[:, 11]
    speed = rows[:, 12]

    plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "savefig.dpi": 300})

    # 1. Trajectory (x, y) with start, goal, and tolerance radius
    fig, ax = plt.subplots(figsize=(7, 7))
    color = "tab:green" if success else "tab:red"
    ax.plot(xs, ys, color=color, linewidth=1.8, label="WAM-V path")
    ax.scatter([xs[0]], [ys[0]], color="black", marker="o", s=60, zorder=3, label="Start")
    ax.scatter([args.goal_x], [args.goal_y], color="gold", marker="*",
               s=200, edgecolor="black", zorder=3, label="Goal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{args.algo.upper()} trajectory to ({args.goal_x:.1f}, {args.goal_y:.1f}) "
                 f"— {'reached' if success else 'not reached'}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/trajectory.png")
    plt.close(fig)

    # 2. Distance-to-goal over time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, dist, color="tab:blue")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance to goal (m)")
    ax.set_title("Distance error over time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/distance_error.png")
    plt.close(fig)

    # 3. Heading error over time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, hdg_err, color="tab:orange")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Heading error (deg)")
    ax.set_title("Heading error over time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/heading_error.png")
    plt.close(fig)

    # 4. Speed over time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, speed, color="tab:purple")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed over time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/speed.png")
    plt.close(fig)

    # 5. Reward over time (per-step and cumulative)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(t, reward_series, color="tab:red")
    ax1.set_ylabel("Reward"); ax1.set_title("Per-step reward"); ax1.grid(alpha=0.3)
    ax2.plot(t, cum_reward_series, color="tab:green")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Cumulative reward")
    ax2.set_title("Cumulative reward"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/reward.png")
    plt.close(fig)

    # 6. Combined 2x2 summary figure (handy for quick review, not just the paper)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(xs, ys, color=color); axes[0, 0].scatter([xs[0]], [ys[0]], color="black")
    axes[0, 0].scatter([args.goal_x], [args.goal_y], color="gold", marker="*", s=150, edgecolor="black")
    axes[0, 0].set_title("Trajectory"); axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 1].plot(t, dist, color="tab:blue")
    axes[0, 1].set_title("Distance to goal (m)")
    axes[1, 0].plot(t, hdg_err, color="tab:orange"); axes[1, 0].set_title("Heading error (deg)")
    axes[1, 1].plot(t, cum_reward_series, color="tab:green"); axes[1, 1].set_title("Cumulative reward")
    for a in axes.flat:
        a.grid(alpha=0.3); a.set_xlabel("Time (s)" if a not in (axes[0, 0],) else "x (m)")
    fig.suptitle(f"{args.algo.upper()} — target ({args.goal_x:.1f}, {args.goal_y:.1f})")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/summary.png")
    plt.close(fig)

    print(f"\nPlots saved to: {out_dir}/")
    print("  trajectory.png, distance_error.png, heading_error.png, "
          "speed.png, reward.png, summary.png")


if __name__ == "__main__":
    main()
