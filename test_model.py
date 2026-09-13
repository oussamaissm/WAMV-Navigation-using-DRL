import time
import math
import argparse

import numpy as np
import torch
import rclpy

import drl_wamv
from drl_wamv import WamvEnv
from sac import SACAgent


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--start-x", type=float, default=-800.32)
    parser.add_argument("--start-y", type=float, default=300.02)

    parser.add_argument("--goal-x", type=float, default=-720.0)
    parser.add_argument("--goal-y", type=float, default=370.0)

    parser.add_argument("--model", type=str,
                        default="results/models/final/actor.pt")

    parser.add_argument("--steps", type=int, default=2000)

    args = parser.parse_args()

    print("\n==============================")
    print("      WAM-V SAC TEST")
    print("==============================")

    print(f"Start : ({args.start_x:.2f}, {args.start_y:.2f})")
    print(f"Goal  : ({args.goal_x:.2f}, {args.goal_y:.2f})")
    print(f"Model : {args.model}")
    print("==============================\n")

    # ---------------------------------------------------------
    # IMPORTANT: WamvEnv.reset() teleporte TOUJOURS le WAM-V vers
    # les constantes du module START_WORLD_X / START_WORLD_Y, et
    # WorldReference calcule aussi son offset GPS->monde a partir
    # de ces memes constantes. Fixer env.start_x/env.start_y ne
    # sert a rien : il faut surcharger les constantes du module
    # AVANT d'appeler reset(), car Python resout les variables
    # globales au moment de l'appel, pas a la definition.
    # ---------------------------------------------------------

    drl_wamv.START_WORLD_X = args.start_x
    drl_wamv.START_WORLD_Y = args.start_y

    # ---------------------------------------------------------
    # ROS 2 initialization
    # ---------------------------------------------------------

    rclpy.init()

    # ---------------------------------------------------------
    # Create environment
    # ---------------------------------------------------------

    env = WamvEnv(
        control_dt=0.2,
        max_episode_time=args.steps * 0.2
    )

    # ---------------------------------------------------------
    # Create SAC agent
    # ---------------------------------------------------------

    agent = SACAgent(
        state_dim=7,
        action_dim=2
    )

    # ---------------------------------------------------------
    # Load trained actor
    # ---------------------------------------------------------

    checkpoint = torch.load(
        args.model,
        map_location="cpu"
    )

    agent.actor.load_state_dict(checkpoint)

    agent.actor.eval()

    print("Trained actor loaded successfully.\n")

    # ---------------------------------------------------------
    # Reset environment
    # ---------------------------------------------------------
    # reset() va :
    #   1. tirer un goal aleatoire (randomize_goal()) -> on l'ignore
    #   2. teleporter le WAM-V vers START_WORLD_X/Y (deja corrige
    #      ci-dessus pour correspondre a args.start_x/start_y)

    state, _ = env.reset()

    # -----------------------------------------------------
    # Restaurer le goal demande par l'utilisateur APRES reset,
    # et recalculer previous_distance pour que le reward
    # (terme "progress") reste coherent des le premier step.
    # -----------------------------------------------------

    env.goal_x = args.goal_x
    env.goal_y = args.goal_y

    env.previous_distance = math.hypot(
        env.goal_x - env.current_x,
        env.goal_y - env.current_y,
    )

    # Rebuild the initial observation for the requested goal
    state = env.get_observation()

    total_reward = 0.0

    # ---------------------------------------------------------
    # Testing loop
    # ---------------------------------------------------------

    for step in range(1, args.steps + 1):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        # -----------------------------------------------------
        # Deterministic action
        # -----------------------------------------------------
        with torch.no_grad():

            mean, _ = agent.actor(state_tensor)

            action = torch.tanh(mean)

        action = action.squeeze(0).numpy()

        # -----------------------------------------------------
        # Apply action
        # -----------------------------------------------------

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        state = next_state

        # -----------------------------------------------------
        # Display information
        # -----------------------------------------------------

        if step % 10 == 0:

            distance = np.sqrt(
                (env.current_x - args.goal_x) ** 2 +
                (env.current_y - args.goal_y) ** 2
            )

            print(
                f"Step {step:4d} | "
                f"Position=({env.current_x:8.2f}, "
                f"{env.current_y:8.2f}) | "
                f"Distance={distance:7.2f} | "
                f"Action=({action[0]:+.2f}, "
                f"{action[1]:+.2f})"
            )

        # -----------------------------------------------------
        # Episode finished
        # -----------------------------------------------------

        if terminated or truncated:

            if terminated:

                print("\n================================")
                print("       GOAL REACHED")
                print("================================")

            else:

                print("\n================================")
                print("       TEST TIMEOUT")
                print("================================")

            break

    print("\nTotal reward:", total_reward)

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    env.close()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()