"""
Plot the results produced by evaluate.py.

Reads:
    <eval_dir>/eval_summary.csv
    <eval_dir>/trajectories/episode_*.csv

Usage:
    python3 plot_eval_results.py --eval_dir eval_results --label PPO
    python3 plot_eval_results.py --eval_dir eval_results_td3 --label TD3 --out_dir eval_results_td3/plots
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GOAL_TOLERANCE_DEFAULT = 2.0  # matches GOAL_TOLERANCE in drl_wamv.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, help="Folder produced by evaluate.py")
    parser.add_argument("--out_dir", default=None, help="Defaults to <eval_dir>/plots")
    parser.add_argument("--label", default="Policy", help="Name shown in plot titles")
    parser.add_argument("--goal_tolerance", type=float, default=GOAL_TOLERANCE_DEFAULT)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.eval_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(args.eval_dir, "eval_summary.csv")
    df = pd.read_csv(summary_path)
    print(f"Loaded {len(df)} evaluation episodes from {summary_path}")

    plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "savefig.dpi": 300})

    # ---------------- 1. Success rate ----------------
    n_success = int(df["success"].sum())
    n_total = len(df)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([n_success, n_total - n_success],
           labels=[f"Success ({n_success})", f"Fail ({n_total - n_success})"],
           colors=["tab:green", "tab:red"], autopct="%1.0f%%", startangle=90)
    ax.set_title(f"{args.label}: success rate over {n_total} episodes")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/success_rate.png"); plt.close(fig)

    # ---------------- 2. Episode reward ----------------
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = df["success"].map({True: "tab:green", False: "tab:red"})
    ax.bar(df["episode"], df["episode_reward"], color=colors)
    ax.axhline(df["episode_reward"].mean(), color="black", linestyle="--",
               label=f"Mean = {df['episode_reward'].mean():.1f}")
    ax.set_xlabel("Episode"); ax.set_ylabel("Episode reward")
    ax.set_title(f"{args.label}: reward per evaluation episode")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/episode_reward.png"); plt.close(fig)

    # ---------------- 3. Final distance error per episode ----------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["episode"], df["final_distance_error"], color=colors)
    ax.axhline(args.goal_tolerance, color="gray", linestyle="--",
               label=f"Goal tolerance ({args.goal_tolerance} m)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Final distance to goal (m)")
    ax.set_title(f"{args.label}: final position error per episode")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/final_distance_error.png"); plt.close(fig)

    # ---------------- 4. Path efficiency per episode ----------------
    fig, ax = plt.subplots(figsize=(9, 4))
    eff = df["path_efficiency"].replace([np.inf, -np.inf], np.nan)
    ax.bar(df["episode"], eff, color=colors)
    ax.axhline(1.0, color="gray", linestyle="--", label="Perfectly straight path")
    ax.set_xlabel("Episode"); ax.set_ylabel("Path efficiency (straight-line / actual)")
    ax.set_title(f"{args.label}: path efficiency per episode")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/path_efficiency.png"); plt.close(fig)

    # ---------------- 5. Mean heading error per episode ----------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["episode"], df["mean_heading_error_deg"], color=colors)
    ax.set_xlabel("Episode"); ax.set_ylabel("Mean |heading error| (deg)")
    ax.set_title(f"{args.label}: mean heading error per episode")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/mean_heading_error.png"); plt.close(fig)

    # ---------------- 6. Episode length distribution ----------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["steps"], bins=min(20, max(5, n_total // 2)), color="tab:blue", edgecolor="black")
    ax.set_xlabel("Episode length (steps)"); ax.set_ylabel("Count")
    ax.set_title(f"{args.label}: episode length distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/episode_length_hist.png"); plt.close(fig)

    # ---------------- 7. All trajectories overlaid ----------------
    traj_files = sorted(glob.glob(os.path.join(args.eval_dir, "trajectories", "episode_*.csv")))
    if traj_files:
        fig, ax = plt.subplots(figsize=(8, 8))
        success_by_ep = dict(zip(df["episode"], df["success"]))
        for path in traj_files:
            ep_num = int(os.path.basename(path).split("_")[1].split(".")[0])
            tdf = pd.read_csv(path)
            success = bool(success_by_ep.get(ep_num, False))
            color = "tab:green" if success else "tab:red"
            ax.plot(tdf["x"], tdf["y"], color=color, alpha=0.5, linewidth=1)
            ax.scatter([tdf["goal_x"].iloc[-1]], [tdf["goal_y"].iloc[-1]],
                       color=color, marker="*", s=60, zorder=3)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"{args.label}: all evaluation trajectories "
                     f"(green=success, red=fail, stars=goals)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/all_trajectories.png"); plt.close(fig)

        # ---------------- 8. Distance-to-goal over time, all episodes overlaid ----------------
        fig, ax = plt.subplots(figsize=(9, 5))
        for path in traj_files:
            ep_num = int(os.path.basename(path).split("_")[1].split(".")[0])
            tdf = pd.read_csv(path)
            success = bool(success_by_ep.get(ep_num, False))
            color = "tab:green" if success else "tab:red"
            ax.plot(tdf["step"], tdf["distance_to_goal"], color=color, alpha=0.4, linewidth=1)
        ax.axhline(args.goal_tolerance, color="gray", linestyle="--",
                   label=f"Goal tolerance ({args.goal_tolerance} m)")
        ax.set_xlabel("Step"); ax.set_ylabel("Distance to goal (m)")
        ax.set_title(f"{args.label}: distance-to-goal over time (all episodes)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/distance_over_time_all_episodes.png"); plt.close(fig)
    else:
        print("No per-episode trajectory CSVs found — skipping trajectory plots "
              f"(expected under {args.eval_dir}/trajectories/)")

    print(f"\nSaved plots to: {out_dir}/")
    print("  success_rate.png, episode_reward.png, final_distance_error.png,")
    print("  path_efficiency.png, mean_heading_error.png, episode_length_hist.png,")
    print("  all_trajectories.png, distance_over_time_all_episodes.png")


if __name__ == "__main__":
    main()
