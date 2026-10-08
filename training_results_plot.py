"""
Plot training progress from the CSVs produced by train_ppo.py, train_td3.py,
and train_sac.py. Handles their different schemas automatically, and can
overlay multiple runs (e.g. PPO vs TD3 vs SAC) on the same axes for
comparison figures.

Usage (single run):
    python3 plot_training_curves.py --csv results_ppo/ppo_training.csv --label PPO

Usage (compare all three):
    python3 plot_training_curves.py \
        --csv results_ppo/ppo_training.csv --label PPO \
        --csv results_td3/td3_training.csv --label TD3 \
        --csv results/sac_training.csv     --label SAC \
        --out_dir training_plots
"""
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", required=True,
                         help="Path to a training CSV. Repeat for multiple runs.")
    parser.add_argument("--label", action="append", required=True,
                         help="Legend label for each --csv, in the same order.")
    parser.add_argument("--out_dir", default="training_plots")
    parser.add_argument("--smooth", type=int, default=20,
                         help="Rolling-average window (in rows) for the smoothed reward line")
    args = parser.parse_args()

    if len(args.csv) != len(args.label):
        raise ValueError("--csv and --label must be given the same number of times.")

    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "savefig.dpi": 300})

    runs = []
    for path, label in zip(args.csv, args.label):
        df = pd.read_csv(path)
        runs.append((label, df))
        print(f"Loaded {label}: {len(df)} rows from {path} (columns: {list(df.columns)})")

    colors = plt.cm.tab10.colors

    # ---------------- 1. Episode reward vs global_step (raw, faint) + smoothed ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, df) in enumerate(runs):
        c = colors[i % len(colors)]
        ax.plot(df["global_step"], df["episode_reward"], color=c, alpha=0.15, linewidth=1)
        ax.plot(df["global_step"], smooth(df["episode_reward"], args.smooth),
                color=c, linewidth=2, label=label)
    ax.set_xlabel("Environment step"); ax.set_ylabel("Episode reward")
    ax.set_title("Training reward over time")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/reward_curve.png"); plt.close(fig)

    # ---------------- 2. Actor loss vs global_step ----------------
    fig, ax = plt.subplots(figsize=(9, 4))
    any_plotted = False
    for i, (label, df) in enumerate(runs):
        if "actor_loss" in df.columns and df["actor_loss"].notna().any():
            c = colors[i % len(colors)]
            d = df.dropna(subset=["actor_loss"])
            ax.plot(d["global_step"], smooth(d["actor_loss"], args.smooth),
                    color=c, linewidth=1.5, label=label)
            any_plotted = True
    if any_plotted:
        ax.set_xlabel("Environment step"); ax.set_ylabel("Actor loss")
        ax.set_title("Actor loss over time")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.out_dir}/actor_loss.png")
    plt.close(fig)

    # ---------------- 3. Critic loss(es) vs global_step ----------------
    fig, ax = plt.subplots(figsize=(9, 4))
    any_plotted = False
    for i, (label, df) in enumerate(runs):
        c = colors[i % len(colors)]
        if "critic_loss" in df.columns:  # PPO: single critic
            ax.plot(df["global_step"], smooth(df["critic_loss"], args.smooth),
                    color=c, linewidth=1.5, label=f"{label} critic")
            any_plotted = True
        else:  # TD3 / SAC: twin critics
            if "critic1_loss" in df.columns:
                ax.plot(df["global_step"], smooth(df["critic1_loss"], args.smooth),
                        color=c, linewidth=1.5, linestyle="-", label=f"{label} critic1")
                any_plotted = True
            if "critic2_loss" in df.columns:
                ax.plot(df["global_step"], smooth(df["critic2_loss"], args.smooth),
                        color=c, linewidth=1.5, linestyle="--", label=f"{label} critic2")
                any_plotted = True
    if any_plotted:
        ax.set_xlabel("Environment step"); ax.set_ylabel("Critic loss")
        ax.set_title("Critic loss over time")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.out_dir}/critic_loss.png")
    plt.close(fig)

    # ---------------- 4. Algorithm-specific extra panel ----------------
    # PPO: entropy + approx_kl | SAC: alpha (temperature) | TD3: nothing extra
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plotted_left = plotted_right = False
    for i, (label, df) in enumerate(runs):
        c = colors[i % len(colors)]
        if "entropy" in df.columns:
            d = df.dropna(subset=["entropy"])
            axes[0].plot(d["global_step"], smooth(d["entropy"], args.smooth),
                         color=c, label=f"{label} entropy")
            plotted_left = True
        if "alpha" in df.columns:
            axes[0].plot(df["global_step"], df["alpha"], color=c, label=f"{label} alpha")
            plotted_left = True
        if "approx_kl" in df.columns:
            d = df.dropna(subset=["approx_kl"])
            axes[1].plot(d["global_step"], smooth(d["approx_kl"], args.smooth),
                         color=c, label=f"{label} approx KL")
            plotted_right = True
    axes[0].set_title("Entropy / temperature (α)"); axes[0].set_xlabel("Environment step")
    axes[1].set_title("PPO approx. KL divergence"); axes[1].set_xlabel("Environment step")
    for a, has_data in zip(axes, [plotted_left, plotted_right]):
        a.grid(alpha=0.3)
        if has_data:
            a.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/extra_diagnostics.png")
    plt.close(fig)

    # ---------------- 5. Buffer size growth (TD3 / SAC only) ----------------
    fig, ax = plt.subplots(figsize=(9, 4))
    any_plotted = False
    for i, (label, df) in enumerate(runs):
        if "buffer_size" in df.columns:
            c = colors[i % len(colors)]
            ax.plot(df["global_step"], df["buffer_size"], color=c, label=label)
            any_plotted = True
    if any_plotted:
        ax.set_xlabel("Environment step"); ax.set_ylabel("Replay buffer size")
        ax.set_title("Replay buffer growth")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.out_dir}/buffer_size.png")
    plt.close(fig)

    print(f"\nSaved plots to: {args.out_dir}/")
    print("  reward_curve.png, actor_loss.png, critic_loss.png,")
    print("  extra_diagnostics.png, buffer_size.png (when applicable)")


if __name__ == "__main__":
    main()
