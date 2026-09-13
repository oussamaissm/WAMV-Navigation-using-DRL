#!/usr/bin/env python3

"""
Robust plotting script for WAM-V SAC training.

Input files:
    results/sac_training.csv
    results/wamv_trajectory.csv

Output directory:
    results/plots/

Generated plots:
    - episode_reward.png
    - step_reward.png
    - actor_loss.png
    - critic_losses.png
    - alpha.png
    - wamv_trajectory.png
    - distance_to_goal.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

TRAINING_FILE = RESULTS_DIR / "sac_training.csv"
TRAJECTORY_FILE = RESULTS_DIR / "wamv_trajectory.csv"

# Smoothing windows
EPISODE_SMOOTHING = 10
STEP_SMOOTHING = 500


# ============================================================
# GENERAL UTILITIES
# ============================================================

def setup():

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("             WAM-V SAC RESULTS")
    print("=" * 60)

    print(f"Training file : {TRAINING_FILE}")
    print(f"Trajectory    : {TRAJECTORY_FILE}")
    print(f"Output dir    : {PLOTS_DIR}")
    print()


def save_plot(filename):

    output = PLOTS_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"[OK] Saved: {output}")

    plt.close()


def check_file(path):

    if not path.exists():

        print(f"[WARNING] File not found: {path}")
        return False

    if path.stat().st_size == 0:

        print(f"[WARNING] File is empty: {path}")
        return False

    return True


def check_columns(df, required, filename):

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print(
            f"[WARNING] {filename} is missing columns:"
            f" {missing}"
        )

        return False

    return True


def numeric_column(df, column):

    """
    Convert a column to numeric values.

    Invalid values become NaN.
    """

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def moving_average(series, window):

    """
    Compute a centered rolling mean.

    min_periods=1 prevents NaN values at the beginning.
    """

    if len(series) == 0:
        return series

    window = max(1, min(window, len(series)))

    return series.rolling(
        window=window,
        min_periods=1
    ).mean()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    if not check_file(TRAINING_FILE):

        return None

    try:

        df = pd.read_csv(TRAINING_FILE)

    except Exception as exc:

        print(
            f"[ERROR] Could not read "
            f"{TRAINING_FILE}: {exc}"
        )

        return None

    print("Training CSV:")
    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {list(df.columns)}")
    print()

    required = [
        "global_step",
        "episode",
        "episode_step",
        "episode_reward",
        "reward",
    ]

    if not check_columns(
        df,
        required,
        TRAINING_FILE
    ):

        return None

    # Convert important columns
    for column in [
        "global_step",
        "episode",
        "episode_step",
        "episode_reward",
        "reward",
        "actor_loss",
        "critic1_loss",
        "critic2_loss",
        "alpha",
        "buffer_size",
    ]:

        numeric_column(df, column)

    # Remove rows without the fundamental information
    df = df.dropna(
        subset=[
            "global_step",
            "episode",
            "reward"
        ]
    )

    # Sort chronologically
    df = df.sort_values(
        "global_step"
    ).reset_index(drop=True)

    if df.empty:

        print("[WARNING] Training CSV contains no valid rows.")

        return None

    return df


# ============================================================
# EPISODE REWARD
# ============================================================

def plot_episode_reward(df):

    print("\n--- Episode Reward ---")

    # --------------------------------------------------------
    # Extract one episode-level reward.
    #
    # The same episode_reward can appear on many rows.
    # We therefore keep the LAST value for every episode.
    # --------------------------------------------------------

    episode_df = (
        df.sort_values("global_step")
        .groupby("episode", as_index=False)
        .agg(
            episode_reward=(
                "episode_reward",
                "last"
            ),
            global_step=(
                "global_step",
                "last"
            )
        )
    )

    episode_df = episode_df.dropna(
        subset=["episode_reward"]
    )

    if episode_df.empty:

        print("[WARNING] No episode reward data.")
        return

    print(
        f"Episodes available: "
        f"{len(episode_df)}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        episode_df["episode"],
        episode_df["episode_reward"],
        linewidth=1,
        alpha=0.35,
        label="Episode reward"
    )

    if len(episode_df) > 1:

        smoothed = moving_average(
            episode_df["episode_reward"],
            EPISODE_SMOOTHING
        )

        plt.plot(
            episode_df["episode"],
            smoothed,
            linewidth=2,
            label=f"Moving average ({EPISODE_SMOOTHING})"
        )

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("SAC Training - Episode Reward")

    plt.grid(True, alpha=0.3)
    plt.legend()

    save_plot("episode_reward.png")


# ============================================================
# STEP REWARD
# ============================================================

def plot_step_reward(df):

    print("\n--- Step Reward ---")

    data = df[
        [
            "global_step",
            "reward"
        ]
    ].dropna()

    if data.empty:

        print("[WARNING] No step reward data.")
        return

    plt.figure(figsize=(11, 6))

    plt.plot(
        data["global_step"],
        data["reward"],
        linewidth=0.7,
        alpha=0.25,
        label="Reward"
    )

    smoothed = moving_average(
        data["reward"],
        STEP_SMOOTHING
    )

    plt.plot(
        data["global_step"],
        smoothed,
        linewidth=2,
        label=f"Moving average ({STEP_SMOOTHING})"
    )

    plt.xlabel("Environment Step")
    plt.ylabel("Reward")
    plt.title("SAC Training - Reward per Environment Step")

    plt.grid(True, alpha=0.3)
    plt.legend()

    save_plot("step_reward.png")


# ============================================================
# ACTOR LOSS
# ============================================================

def plot_actor_loss(df):

    print("\n--- Actor Loss ---")

    if "actor_loss" not in df.columns:

        print("[WARNING] actor_loss column not found.")
        return

    data = df[
        [
            "global_step",
            "actor_loss"
        ]
    ].dropna()

    if data.empty:

        print("[WARNING] No valid actor loss data.")
        return

    plt.figure(figsize=(11, 6))

    plt.plot(
        data["global_step"],
        data["actor_loss"],
        linewidth=0.8,
        alpha=0.7
    )

    plt.xlabel("Environment Step")
    plt.ylabel("Actor Loss")
    plt.title("SAC Actor Loss")

    plt.grid(True, alpha=0.3)

    save_plot("actor_loss.png")


# ============================================================
# CRITIC LOSSES
# ============================================================

def plot_critic_losses(df):

    print("\n--- Critic Losses ---")

    available = []

    if "critic1_loss" in df.columns:
        available.append("critic1_loss")

    if "critic2_loss" in df.columns:
        available.append("critic2_loss")

    if not available:

        print("[WARNING] No critic loss columns found.")
        return

    plt.figure(figsize=(11, 6))

    for column in available:

        data = df[
            [
                "global_step",
                column
            ]
        ].dropna()

        if data.empty:
            continue

        label = column.replace(
            "_",
            " "
        ).title()

        plt.plot(
            data["global_step"],
            data[column],
            linewidth=0.8,
            alpha=0.7,
            label=label
        )

    plt.xlabel("Environment Step")
    plt.ylabel("Critic Loss")
    plt.title("SAC Critic Losses")

    plt.grid(True, alpha=0.3)
    plt.legend()

    save_plot("critic_losses.png")


# ============================================================
# ALPHA
# ============================================================

def plot_alpha(df):

    print("\n--- Entropy Coefficient ---")

    if "alpha" not in df.columns:

        print("[WARNING] alpha column not found.")
        return

    data = df[
        [
            "global_step",
            "alpha"
        ]
    ].dropna()

    if data.empty:

        print("[WARNING] No valid alpha data.")
        return

    plt.figure(figsize=(11, 6))

    plt.plot(
        data["global_step"],
        data["alpha"],
        linewidth=1.2
    )

    plt.xlabel("Environment Step")
    plt.ylabel("Alpha")
    plt.title("SAC Entropy Coefficient")

    plt.grid(True, alpha=0.3)

    save_plot("alpha.png")


# ============================================================
# LOAD TRAJECTORY DATA
# ============================================================

def load_trajectory_data():

    if not check_file(TRAJECTORY_FILE):

        return None

    try:

        df = pd.read_csv(
            TRAJECTORY_FILE
        )

    except Exception as exc:

        print(
            f"[ERROR] Could not read "
            f"{TRAJECTORY_FILE}: {exc}"
        )

        return None

    print("\nTrajectory CSV:")
    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {list(df.columns)}")

    required = [
        "step",
        "x",
        "y",
        "goal_x",
        "goal_y",
        "distance",
    ]

    if not check_columns(
        df,
        required,
        TRAJECTORY_FILE
    ):

        return None

    for column in required:

        numeric_column(
            df,
            column
        )

    df = df.dropna(
        subset=[
            "step",
            "x",
            "y",
            "goal_x",
            "goal_y"
        ]
    )

    df = df.sort_values(
        "step"
    ).reset_index(drop=True)

    if df.empty:

        print("[WARNING] No valid trajectory data.")
        return None

    return df


# ============================================================
# TRAJECTORY
# ============================================================

def plot_trajectory(df):

    print("\n--- WAM-V Trajectory ---")

    plt.figure(figsize=(10, 8))

    # --------------------------------------------------------
    # Detect different goals.
    #
    # This is useful if the CSV contains several episodes
    # with randomized targets.
    # --------------------------------------------------------

    goals = (
        df[
            ["goal_x", "goal_y"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Different goals found: "
        f"{len(goals)}"
    )

    # --------------------------------------------------------
    # If there is only one goal, plot normally.
    # --------------------------------------------------------

    if len(goals) == 1:

        plt.plot(
            df["x"],
            df["y"],
            linewidth=2,
            label="WAM-V trajectory"
        )

        start_x = df["x"].iloc[0]
        start_y = df["y"].iloc[0]

        goal_x = goals["goal_x"].iloc[0]
        goal_y = goals["goal_y"].iloc[0]

        plt.scatter(
            start_x,
            start_y,
            s=100,
            marker="o",
            label="Start",
            zorder=5
        )

        plt.scatter(
            goal_x,
            goal_y,
            s=150,
            marker="*",
            label="Goal",
            zorder=5
        )

        # Straight-line reference
        plt.plot(
            [start_x, goal_x],
            [start_y, goal_y],
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Start → Goal"
        )

    # --------------------------------------------------------
    # Multiple goals.
    #
    # We cannot safely connect different episodes together,
    # so each goal segment is plotted independently.
    # --------------------------------------------------------

    else:

        grouped = df.groupby(
            ["goal_x", "goal_y"],
            sort=False
        )

        for index, ((goal_x, goal_y), group) in enumerate(grouped):

            group = group.sort_values("step")

            label = (
                "Trajectory"
                if index == 0
                else None
            )

            plt.plot(
                group["x"],
                group["y"],
                linewidth=1.5,
                label=label
            )

            # Mark goal
            plt.scatter(
                goal_x,
                goal_y,
                s=80,
                marker="*",
                alpha=0.7
            )

        # Overall first starting position
        plt.scatter(
            df["x"].iloc[0],
            df["y"].iloc[0],
            s=120,
            marker="o",
            label="First Start",
            zorder=5
        )

    plt.xlabel("X position [m]")
    plt.ylabel("Y position [m]")

    plt.title("WAM-V GPS Trajectory")

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_plot("wamv_trajectory.png")


# ============================================================
# DISTANCE TO GOAL
# ============================================================

def plot_distance(df):

    print("\n--- Distance to Goal ---")

    if "distance" not in df.columns:

        print("[WARNING] distance column not found.")
        return

    data = df[
        [
            "step",
            "distance"
        ]
    ].dropna()

    if data.empty:

        print("[WARNING] No distance data.")
        return

    plt.figure(figsize=(11, 6))

    plt.plot(
        data["step"],
        data["distance"],
        linewidth=1.2
    )

    plt.xlabel("Step")
    plt.ylabel("Distance to Goal [m]")
    plt.title("WAM-V Distance to Goal")

    plt.grid(True, alpha=0.3)

    save_plot("distance_to_goal.png")


# ============================================================
# SUMMARY
# ============================================================

def print_summary(training_df, trajectory_df):

    print("\n")
    print("=" * 60)
    print("                    SUMMARY")
    print("=" * 60)

    if training_df is not None:

        print(
            f"Training steps : "
            f"{training_df['global_step'].max():.0f}"
        )

        print(
            f"Episodes       : "
            f"{training_df['episode'].nunique()}"
        )

        rewards = (
            training_df
            .groupby("episode")["episode_reward"]
            .last()
            .dropna()
        )

        if not rewards.empty:

            print(
                f"Best episode reward : "
                f"{rewards.max():.2f}"
            )

            print(
                f"Worst episode reward: "
                f"{rewards.min():.2f}"
            )

            print(
                f"Last episode reward : "
                f"{rewards.iloc[-1]:.2f}"
            )

    if trajectory_df is not None:

        print(
            f"\nTrajectory points: "
            f"{len(trajectory_df)}"
        )

        final_distance = (
            trajectory_df["distance"]
            .dropna()
        )

        if not final_distance.empty:

            print(
                f"Final distance to goal: "
                f"{final_distance.iloc[-1]:.2f} m"
            )

            print(
                f"Minimum distance to goal: "
                f"{final_distance.min():.2f} m"
            )

        if "terminated" in trajectory_df.columns:

            terminated = (
                trajectory_df["terminated"]
                .astype(str)
                .str.lower()
                .eq("true")
                .any()
            )

            print(
                f"Goal reached: "
                f"{'YES' if terminated else 'NO'}"
            )

        if "truncated" in trajectory_df.columns:

            truncated = (
                trajectory_df["truncated"]
                .astype(str)
                .str.lower()
                .eq("true")
                .any()
            )

            print(
                f"Episode truncated: "
                f"{'YES' if truncated else 'NO'}"
            )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    setup()

    # --------------------------------------------------------
    # Training plots
    # --------------------------------------------------------

    training_df = load_training_data()

    if training_df is not None:

        try:
            plot_episode_reward(training_df)
        except Exception as exc:
            print(f"[ERROR] Episode reward plot: {exc}")

        try:
            plot_step_reward(training_df)
        except Exception as exc:
            print(f"[ERROR] Step reward plot: {exc}")

        try:
            plot_actor_loss(training_df)
        except Exception as exc:
            print(f"[ERROR] Actor loss plot: {exc}")

        try:
            plot_critic_losses(training_df)
        except Exception as exc:
            print(f"[ERROR] Critic loss plot: {exc}")

        try:
            plot_alpha(training_df)
        except Exception as exc:
            print(f"[ERROR] Alpha plot: {exc}")

    else:

        print(
            "\n[WARNING] Training plots skipped."
        )

    # --------------------------------------------------------
    # Trajectory plots
    # --------------------------------------------------------

    trajectory_df = load_trajectory_data()

    if trajectory_df is not None:

        try:
            plot_trajectory(trajectory_df)
        except Exception as exc:
            print(f"[ERROR] Trajectory plot: {exc}")

        try:
            plot_distance(trajectory_df)
        except Exception as exc:
            print(f"[ERROR] Distance plot: {exc}")

    else:

        print(
            "\n[WARNING] Trajectory plots skipped."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        training_df,
        trajectory_df
    )

    print("\nPlots are located in:")
    print(PLOTS_DIR.resolve())


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
