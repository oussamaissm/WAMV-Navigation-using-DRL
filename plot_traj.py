import argparse
import csv

import matplotlib.pyplot as plt


def load_trajectory(csv_path):
    steps = []
    xs = []
    ys = []
    goal_xs = []
    goal_ys = []
    distances = []
    rewards = []
    left_thrusts = []
    right_thrusts = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            steps.append(int(row["step"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            goal_xs.append(float(row["goal_x"]))
            goal_ys.append(float(row["goal_y"]))
            distances.append(float(row["distance"]))
            rewards.append(float(row["reward"]))
            left_thrusts.append(float(row["left_thrust"]))
            right_thrusts.append(float(row["right_thrust"]))

    return {
        "step": steps,
        "x": xs,
        "y": ys,
        "goal_x": goal_xs,
        "goal_y": goal_ys,
        "distance": distances,
        "reward": rewards,
        "left_thrust": left_thrusts,
        "right_thrust": right_thrusts,
    }


def plot_results(data, save_path=None, show=True):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # -------------------------------------------------
    # 1. Trajectory (x, y)
    # -------------------------------------------------
    ax = axes[0, 0]

    ax.plot(data["x"], data["y"], "-", color="tab:blue", label="Trajectory")
    ax.plot(data["x"][0], data["y"][0], "o", color="green",
            markersize=10, label="Start")
    ax.plot(data["goal_x"][-1], data["goal_y"][-1], "*", color="red",
            markersize=15, label="Goal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("WAM-V Trajectory")
    ax.legend()
    ax.axis("equal")
    ax.grid(True)

    # -------------------------------------------------
    # 2. Distance to goal over time
    # -------------------------------------------------
    ax = axes[0, 1]

    ax.plot(data["step"], data["distance"], color="tab:orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal (m)")
    ax.set_title("Distance to Goal")
    ax.grid(True)

    # -------------------------------------------------
    # 3. Reward per step
    # -------------------------------------------------
    ax = axes[1, 0]

    ax.plot(data["step"], data["reward"], color="tab:green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward per Step")
    ax.grid(True)

    # -------------------------------------------------
    # 4. Thrust commands
    # -------------------------------------------------
    ax = axes[1, 1]

    ax.plot(data["step"], data["left_thrust"], label="Left thrust")
    ax.plot(data["step"], data["right_thrust"], label="Right thrust")
    ax.set_xlabel("Step")
    ax.set_ylabel("Thrust")
    ax.set_title("Thruster Commands")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="results/wamv_trajectory.csv",
        help="Path to the trajectory CSV produced by the test run."
    )

    parser.add_argument(
        "--save",
        type=str,
        default="results/wamv_trajectory.png",
        help="Path to save the resulting figure (PNG)."
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window, only save the figure."
    )

    args = parser.parse_args()

    data = load_trajectory(args.csv)

    plot_results(
        data,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()