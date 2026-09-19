"""
Evaluate a trained WAM-V navigation policy (PPO, TD3, or SAC).

Runs the policy deterministically (no exploration noise) for N episodes,
logs the full trajectory of each episode to its own CSV, computes
per-episode error/performance metrics, and prints an aggregate summary
at the end. Optionally plots the paths taken against their goals.

Usage:
    python3 evaluate.py --algo ppo --model_dir results_ppo/models/final --episodes 20
    python3 evaluate.py --algo td3 --model_dir results_td3/models/final --episodes 20
    python3 evaluate.py --algo sac --model_dir results/models/final     --episodes 20
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
    """Returns a state -> action function using each agent's deterministic
    (noise-free) behavior, since evaluation should measure what the policy
    has actually learned, not its exploration noise."""
    if algo == "ppo":
        from ppo import PPOAgent
        agent = PPOAgent(STATE_DIM, ACTION_DIM, device=device)
        agent.load(model_dir)
        agent.actor.eval()

        def act(state):
            s = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(s)  # PPO's actor has no tanh squash; use the raw mean
            return np.clip(mean.cpu().numpy()[0], -1.0, 1.0).astype(np.float32)
        return act, agent

    if algo == "td3":
        from td3 import TD3Agent
        agent = TD3Agent(STATE_DIM, ACTION_DIM, device=device)
        agent.load(model_dir)
        agent.actor.eval()
        return (lambda state: agent.select_action(state, noise=False)), agent

    if algo == "sac":
        from sac import SACAgent
        agent = SACAgent(STATE_DIM, ACTION_DIM, device=device)
        agent.load(model_dir)
        agent.actor.eval()
        return (lambda state: agent.select_action(state, deterministic=True)), agent

    raise ValueError(f"Unknown --algo '{algo}'. Choose from: ppo, td3, sac.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "td3", "sac"], required=True)
    parser.add_argument("--model_dir", required=True,
                         help="e.g. results_ppo/models/final")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out_dir", default="eval_results")
    parser.add_argument("--plot", action="store_true",
                         help="Save an aggregate PNG of all episode paths (requires matplotlib)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(f"{args.out_dir}/trajectories", exist_ok=True)

    act, agent = load_policy(args.algo, args.model_dir, device)
    print(f"Loaded {args.algo.upper()} policy from {args.model_dir} (device={device})")

    # Same SIGINT-safety fix as the training scripts: don't let rclpy tear
    # down the context on Ctrl+C before env.close() gets to run cleanly.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    env = WamvEnv(control_dt=0.2, max_episode_time=120.0,
                  trajectory_path=f"{args.out_dir}/_last_episode_raw.csv")

    summary_path = f"{args.out_dir}/eval_summary.csv"
    summary_f = open(summary_path, "w", newline="")
    summary_w = csv.writer(summary_f)
    summary_w.writerow([
        "episode", "success", "steps", "episode_reward",
        "final_distance_error", "straight_line_distance", "path_length",
        "path_efficiency", "mean_heading_error_deg", "max_heading_error_deg",
        "mean_speed", "elapsed_time_s",
    ])

    successes = 0
    all_rewards, all_final_dist, all_lengths, all_efficiency = [], [], [], []
    all_paths = []  # for optional plotting: list of (xs, ys, goal_x, goal_y, success)

    try:
        for ep in range(1, args.episodes + 1):
            state, info = env.reset()
            start_x, start_y = info["x"], info["y"]
            goal_x, goal_y = info["goal_x"], info["goal_y"]
            straight_line = math.hypot(goal_x - start_x, goal_y - start_y)

            traj_path = f"{args.out_dir}/trajectories/episode_{ep:03d}.csv"
            traj_f = open(traj_path, "w", newline="")
            traj_w = csv.writer(traj_f)
            traj_w.writerow([
                "step", "x", "y", "goal_x", "goal_y", "distance_to_goal",
                "action_left", "action_right", "reward",
                "heading_error_deg", "speed",
            ])

            xs, ys = [start_x], [start_y]
            heading_errors, speeds = [], []
            prev_x, prev_y = start_x, start_y
            path_length = 0.0
            ep_reward = 0.0
            step = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = act(state)
                state, reward, terminated, truncated, info = env.step(action)
                step += 1
                ep_reward += float(reward)

                path_length += math.hypot(info["x"] - prev_x, info["y"] - prev_y)
                prev_x, prev_y = info["x"], info["y"]
                xs.append(info["x"]); ys.append(info["y"])

                heading_err_deg = math.degrees(info["heading_error"])
                heading_errors.append(abs(heading_err_deg))
                speeds.append(info["speed"])

                traj_w.writerow([
                    step, info["x"], info["y"], info["goal_x"], info["goal_y"],
                    info["distance_to_goal"], float(action[0]), float(action[1]),
                    reward, heading_err_deg, info["speed"],
                ])

            traj_f.close()

            success = bool(terminated)
            final_distance = info["distance_to_goal"]
            efficiency = (straight_line / path_length) if path_length > 1e-6 else float("nan")
            mean_hdg_err = float(np.mean(heading_errors)) if heading_errors else float("nan")
            max_hdg_err = float(np.max(heading_errors)) if heading_errors else float("nan")
            mean_speed = float(np.mean(speeds)) if speeds else float("nan")

            successes += int(success)
            all_rewards.append(ep_reward)
            all_final_dist.append(final_distance)
            all_lengths.append(step)
            if not math.isnan(efficiency):
                all_efficiency.append(efficiency)
            all_paths.append((xs, ys, goal_x, goal_y, success))

            summary_w.writerow([
                ep, success, step, ep_reward, final_distance, straight_line,
                path_length, efficiency, mean_hdg_err, max_hdg_err, mean_speed,
                info["elapsed_time"],
            ])
            summary_f.flush()

            tag = "SUCCESS" if success else "FAIL   "
            print(f"Episode {ep:03d} | {tag} | steps={step:4d} | reward={ep_reward:10.2f} | "
                  f"final_err={final_distance:7.2f} m | efficiency={efficiency:.2f} | "
                  f"mean_hdg_err={mean_hdg_err:6.1f} deg")

    finally:
        summary_f.close()
        env.close()
        rclpy.shutdown()

    n = len(all_rewards)
    print("\n===== Evaluation summary =====")
    print(f"Algorithm:          {args.algo.upper()}")
    print(f"Episodes run:       {n}")
    print(f"Success rate:       {successes}/{n} ({100*successes/n:.1f}%)")
    print(f"Mean reward:        {np.mean(all_rewards):.2f} (+/- {np.std(all_rewards):.2f})")
    print(f"Mean final error:   {np.mean(all_final_dist):.2f} m (+/- {np.std(all_final_dist):.2f})")
    print(f"Mean episode length:{np.mean(all_lengths):.1f} steps")
    if all_efficiency:
        print(f"Mean path efficiency: {np.mean(all_efficiency):.2f} (1.0 = perfectly straight)")
    print(f"\nPer-episode trajectories: {args.out_dir}/trajectories/episode_*.csv")
    print(f"Summary CSV:              {summary_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            for xs, ys, gx, gy, success in all_paths:
                color = "tab:green" if success else "tab:red"
                ax.plot(xs, ys, color=color, alpha=0.6, linewidth=1)
                ax.scatter([gx], [gy], color=color, marker="*", s=80, zorder=3)
            ax.scatter([xs[0]], [ys[0]] if all_paths else [], color="black", marker="o")
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            ax.set_title(f"{args.algo.upper()} evaluation paths "
                         f"(green=success, red=fail, stars=goals)")
            ax.set_aspect("equal", adjustable="datalim")
            fig.tight_layout()
            plot_path = f"{args.out_dir}/trajectories_plot.png"
            fig.savefig(plot_path, dpi=150)
            print(f"Trajectory plot:          {plot_path}")
        except ImportError:
            print("matplotlib not installed — skipping --plot (pip install matplotlib)")


if __name__ == "__main__":
    main()
