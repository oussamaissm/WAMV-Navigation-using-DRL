#!/usr/bin/env python3

"""
Single-file DRL environment for the VRX WAM-V.

Architecture:

    VRX / Gazebo
         |
         +---- GPS  ---> ROS 2 subscriber
         |
         +---- IMU  ---> ROS 2 subscriber
         |
         +---- Thrusters <--- ROS 2 publishers
                              |
                              v
                         DRL environment
                              |
                              v
                         SAC / PPO

The Python code does NOT simulate the WAM-V dynamics.
Gazebo/VRX is the real plant.

Observation:
    [position_error_x,
     position_error_y,
     heading_error,
     u,
     v,
     r,
     desired_speed]

Action:
    [left_thruster, right_thruster]

GPS:
    /wamv/sensors/gps/gps/fix

IMU:
    /wamv/sensors/imu/imu/data

Thrusters:
    /wamv/thrusters/left/thrust
    /wamv/thrusters/right/thrust
"""

import math
import time
import threading

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64


# ============================================================
# Utility functions
# ============================================================

def wrap_angle(angle):
    """
    Wrap an angle to [-pi, pi].

    Example:
        +3.2 rad -> approximately -3.08 rad
        -3.2 rad -> approximately +3.08 rad
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(x, y, z, w):
    """
    Convert quaternion orientation to yaw angle.

    ROS uses quaternion:
        q = [x, y, z, w]

    Returns:
        yaw in radians.
    """

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)


# ============================================================
# State container
# ============================================================

class WamvState:
    """
    Stores the latest WAM-V state.

    Position:
        x, y

    Orientation:
        yaw

    Velocity:
        u = forward velocity
        v = lateral velocity
        r = yaw rate

    GPS is converted into a local XY coordinate system.
    """

    def __init__(self):

        self.x = 0.0
        self.y = 0.0

        self.yaw = 0.0

        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        self.gps_received = False
        self.imu_received = False

        self.last_time = None

        # Previous GPS position
        self.prev_x = None
        self.prev_y = None

        # Previous yaw
        self.prev_yaw = None

        self.lock = threading.Lock()


# ============================================================
# ROS 2 WAM-V interface
# ============================================================

class WamvNode(Node):

    def __init__(self):

        super().__init__("wamv_drl_node")

        self.state = WamvState()

        # ----------------------------------------------------
        # GPS reference
        # ----------------------------------------------------

        self.gps_lat0 = None
        self.gps_lon0 = None

        # ----------------------------------------------------
        # Subscribers
        # ----------------------------------------------------

        self.gps_sub = self.create_subscription(
            NavSatFix,
            "/wamv/sensors/gps/gps/fix",
            self.gps_cb,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            "/wamv/sensors/imu/imu/data",
            self.imu_cb,
            10
        )

        # ----------------------------------------------------
        # Thruster publishers
        # ----------------------------------------------------

        self.left_thruster_pub = self.create_publisher(
            Float64,
            "/wamv/thrusters/left/thrust",
            10
        )

        self.right_thruster_pub = self.create_publisher(
            Float64,
            "/wamv/thrusters/right/thrust",
            10
        )

        self.get_logger().info("WAM-V DRL node started.")

    # ========================================================
    # GPS callback
    # ========================================================

    def gps_cb(self, msg):

        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return

        with self.state.lock:

            # First GPS measurement becomes local origin.
            if self.gps_lat0 is None:

                self.gps_lat0 = msg.latitude
                self.gps_lon0 = msg.longitude

                self.get_logger().info(
                    f"GPS origin set: "
                    f"lat={self.gps_lat0:.8f}, "
                    f"lon={self.gps_lon0:.8f}"
                )

            # ------------------------------------------------
            # GPS -> local Cartesian coordinates
            #
            # Approximation:
            #
            # x = East
            # y = North
            # ------------------------------------------------

            earth_radius = 6378137.0

            lat0_rad = math.radians(self.gps_lat0)

            dx = math.radians(
                msg.longitude - self.gps_lon0
            ) * earth_radius * math.cos(lat0_rad)

            dy = math.radians(
                msg.latitude - self.gps_lat0
            ) * earth_radius

            x = dx
            y = dy

            # ------------------------------------------------
            # Estimate velocity from GPS
            # ------------------------------------------------

            now = time.monotonic()

            if self.state.prev_x is not None:

                dt = now - self.state.last_time

                # Avoid numerical problems.
                if 0.001 < dt < 1.0:

                    vx_world = (x - self.state.prev_x) / dt
                    vy_world = (y - self.state.prev_y) / dt

                    # Convert world velocity to body velocity.
                    #
                    # World:
                    #     vx = East
                    #     vy = North
                    #
                    # Body:
                    #     u = forward
                    #     v = lateral

                    yaw = self.state.yaw

                    self.state.u = (
                        math.cos(yaw) * vx_world
                        + math.sin(yaw) * vy_world
                    )

                    self.state.v = (
                        -math.sin(yaw) * vx_world
                        + math.cos(yaw) * vy_world
                    )

            self.state.x = x
            self.state.y = y

            self.state.prev_x = x
            self.state.prev_y = y

            self.state.last_time = now

            self.state.gps_received = True

    # ========================================================
    # IMU callback
    # ========================================================

    def imu_cb(self, msg):

        q = msg.orientation

        yaw = quaternion_to_yaw(
            q.x,
            q.y,
            q.z,
            q.w
        )

        with self.state.lock:

            now = time.monotonic()

            # ------------------------------------------------
            # Yaw rate
            # ------------------------------------------------

            if self.state.prev_yaw is not None:

                dt = now - self.state.last_time

                if 0.001 < dt < 1.0:

                    dyaw = wrap_angle(
                        yaw - self.state.prev_yaw
                    )

                    self.state.r = dyaw / dt

            self.state.yaw = yaw

            self.state.prev_yaw = yaw

            self.state.imu_received = True

    # ========================================================
    # Get current state
    # ========================================================

    def get_state(self):

        with self.state.lock:

            return {
                "x": self.state.x,
                "y": self.state.y,
                "yaw": self.state.yaw,
                "u": self.state.u,
                "v": self.state.v,
                "r": self.state.r,
                "gps_received": self.state.gps_received,
                "imu_received": self.state.imu_received,
            }

    # ========================================================
    # Publish thruster commands
    # ========================================================

    def publish_thrusters(self, left, right):

        left_msg = Float64()
        right_msg = Float64()

        left_msg.data = float(left)
        right_msg.data = float(right)

        self.left_thruster_pub.publish(left_msg)
        self.right_thruster_pub.publish(right_msg)

    # ========================================================
    # Stop WAM-V
    # ========================================================

    def stop(self):

        self.publish_thrusters(0.0, 0.0)


# ============================================================
# Trajectory
# ============================================================

class StraightTrajectory:
    """
    Simple straight-line trajectory.

    Example:

        start = (0, 0)
        goal  = (50, 0)

    The desired heading is automatically calculated.
    """

    def __init__(
        self,
        start_x=-800.0,
        start_y=450.0,
        goal_x=-720.0,
        goal_y=370.0,
        desired_speed=1.0,
    ):

        self.start_x = start_x
        self.start_y = start_y

        self.goal_x = goal_x
        self.goal_y = goal_y

        self.desired_speed = desired_speed

        self.heading = math.atan2(
            goal_y - start_y,
            goal_x - start_x
        )

    def get_reference(self):

        return {
            "x": self.goal_x,
            "y": self.goal_y,
            "heading": self.heading,
            "speed": self.desired_speed,
        }


# ============================================================
# DRL Environment
# ============================================================

class WamvEnv(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        control_dt=0.2,
        max_episode_time=120.0,
        thruster_min=0.0,
        thruster_max=1000.0,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        self.control_dt = control_dt
        self.max_episode_time = max_episode_time

        # ----------------------------------------------------
        # Thruster limits
        # ----------------------------------------------------

        self.thruster_min = thruster_min
        self.thruster_max = thruster_max

        # ----------------------------------------------------
        # Action space
        #
        # SAC will output:
        #
        # [-1, 1]
        #
        # We convert it to:
        #
        # [0, 1000]
        # ----------------------------------------------------

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Observation
        #
        # [ex,
        #  ey,
        #  epsi,
        #  u,
        #  v,
        #  r,
        #  desired_speed]
        #
        # Values are normalized/clipped to reasonable ranges.
        # ----------------------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # ROS
        # ----------------------------------------------------

        if not rclpy.ok():
            rclpy.init()

        self.node = WamvNode()

        # ----------------------------------------------------
        # Trajectory
        # ----------------------------------------------------

        self.trajectory = StraightTrajectory(
            start_x=-800.0,
            start_y=450.0,
            goal_x=-720.0,
            goal_y=370.0,
            desired_speed=1.0,
        )

        self.start_time = None

        self.previous_distance = None

        self.previous_x = None
        self.previous_y = None

    # ========================================================
    # ROS update
    # ========================================================

    def spin_ros(self, duration=0.05):

        """
        Give ROS 2 time to process GPS/IMU messages.
        """

        end_time = time.monotonic() + duration

        while time.monotonic() < end_time:

            rclpy.spin_once(
                self.node,
                timeout_sec=0.001
            )

    # ========================================================
    # Convert action
    # ========================================================

    def action_to_thrusters(self, action):

        """
        SAC outputs actions in [-1, 1].

        Convert to physical thruster command.

        Example with max=1000:

            -1 -> 0
             0 -> 500
            +1 -> 1000
        """

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        normalized = (action + 1.0) / 2.0

        thrusters = (
            self.thruster_min
            + normalized
            * (
                self.thruster_max
                - self.thruster_min
            )
        )

        return thrusters

    # ========================================================
    # Get observation
    # ========================================================

    def get_observation(self):

        state = self.node.get_state()

        reference = self.trajectory.get_reference()

        # ----------------------------------------------------
        # Position error in world frame
        # ----------------------------------------------------

        dx = reference["x"] - state["x"]
        dy = reference["y"] - state["y"]

        # ----------------------------------------------------
        # Transform position error into body frame.
        #
        # This is important because the policy should know
        # where the target is relative to the boat.
        # ----------------------------------------------------

        yaw = state["yaw"]

        ex = (
            math.cos(yaw) * dx
            + math.sin(yaw) * dy
        )

        ey = (
            -math.sin(yaw) * dx
            + math.cos(yaw) * dy
        )

        # ----------------------------------------------------
        # Heading error
        # ----------------------------------------------------

        epsi = wrap_angle(
            reference["heading"]
            - state["yaw"]
        )

        # ----------------------------------------------------
        # Observation
        # ----------------------------------------------------

        obs = np.array(
            [
                ex,
                ey,
                epsi,
                state["u"],
                state["v"],
                state["r"],
                reference["speed"],
            ],
            dtype=np.float32
        )

        return obs

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        # ----------------------------------------------------
        # Important:
        #
        # This does NOT reset Gazebo's physical WAM-V pose.
        #
        # For now reset only resets the RL episode.
        # Later we can add Gazebo model reset.
        # ----------------------------------------------------

        self.start_time = time.monotonic()

        self.previous_distance = None

        self.previous_x = None
        self.previous_y = None

        # Stop the boat before starting.
        self.node.stop()

        # Give ROS time to receive fresh data.
        for _ in range(20):

            self.spin_ros(0.01)

            state = self.node.get_state()

            if (
                state["gps_received"]
                and state["imu_received"]
            ):
                break

        # ----------------------------------------------------
        # Initial observation
        # ----------------------------------------------------

        obs = self.get_observation()

        state = self.node.get_state()

        reference = self.trajectory.get_reference()

        distance = math.sqrt(
            (reference["x"] - state["x"]) ** 2
            + (reference["y"] - state["y"]) ** 2
        )

        self.previous_distance = distance

        info = {
            "x": state["x"],
            "y": state["y"],
            "yaw": state["yaw"],
            "distance_to_goal": distance,
        }

        return obs, info

    # ========================================================
    # Step
    # ========================================================

    def step(self, action):

        # ----------------------------------------------------
        # Convert RL action to thruster values
        # ----------------------------------------------------

        left, right = self.action_to_thrusters(action)

        # ----------------------------------------------------
        # Send command to Gazebo/VRX
        # ----------------------------------------------------

        self.node.publish_thrusters(
            left,
            right
        )

        # ----------------------------------------------------
        # Wait for the physical simulation to evolve.
        #
        # During this time GPS/IMU callbacks update the state.
        # ----------------------------------------------------

        start = time.monotonic()

        while (
            time.monotonic() - start
            < self.control_dt
        ):

            rclpy.spin_once(
                self.node,
                timeout_sec=0.01
            )

        # ----------------------------------------------------
        # Get new state
        # ----------------------------------------------------

        state = self.node.get_state()

        reference = self.trajectory.get_reference()

        # ----------------------------------------------------
        # Position error
        # ----------------------------------------------------

        dx = reference["x"] - state["x"]
        dy = reference["y"] - state["y"]

        distance = math.sqrt(
            dx * dx + dy * dy
        )

        # ----------------------------------------------------
        # Heading error
        # ----------------------------------------------------

        heading_error = abs(
            wrap_angle(
                reference["heading"]
                - state["yaw"]
            )
        )

        # ----------------------------------------------------
        # Speed error
        # ----------------------------------------------------

        speed_error = abs(
            reference["speed"]
            - state["u"]
        )

        # ====================================================
        # Reward
        # ====================================================

        # 1. Distance penalty
        position_reward = -2.0 * distance

        # 2. Heading penalty
        heading_reward = -0.5 * heading_error

        # 3. Speed penalty
        speed_reward = -0.25 * speed_error

        # 4. Progress reward
        progress_reward = 0.0

        if self.previous_distance is not None:

            progress = (
                self.previous_distance
                - distance
            )

            progress_reward = (
                2.0 * progress
            )

        # 5. Thruster effort penalty
        effort = (
            abs(left)
            + abs(right)
        ) / (
            2.0 * self.thruster_max
        )

        effort_reward = -0.05 * effort

        # ----------------------------------------------------
        # Total reward
        # ----------------------------------------------------

        reward = (
            position_reward
            + heading_reward
            + speed_reward
            + progress_reward
            + effort_reward
        )

        # ----------------------------------------------------
        # Update distance
        # ----------------------------------------------------

        self.previous_distance = distance

        # ====================================================
        # Termination
        # ====================================================

        terminated = False

        # Goal reached
        goal_tolerance = 2.0

        if distance < goal_tolerance:

            terminated = True

            # Stop the WAM-V.
            self.node.stop()

            self.get_logger_safe(
                "Goal reached!"
            )

        # Boat too far from trajectory/goal
        '''failure_distance = 100.0

        if distance > failure_distance:

            terminated = True

            self.node.stop()'''

        # ====================================================
        # Time limit
        # ====================================================

        truncated = False

        if (
            time.monotonic()
            - self.start_time
            >= self.max_episode_time
        ):

            truncated = True

            self.node.stop()

        # ----------------------------------------------------
        # Observation for next step
        # ----------------------------------------------------

        obs = self.get_observation()

        info = {
            "x": state["x"],
            "y": state["y"],
            "yaw": state["yaw"],
            "u": state["u"],
            "v": state["v"],
            "r": state["r"],
            "distance_to_goal": distance,
            "left_thruster": left,
            "right_thruster": right,
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ========================================================
    # Safe logger helper
    # ========================================================

    def get_logger_safe(self, message):

        self.node.get_logger().info(message)

    # ========================================================
    # Close
    # ========================================================

    def close(self):

        self.node.stop()

        self.node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


# ============================================================
# Simple test
# ============================================================

def main():

    print("=" * 60)
    print("WAM-V DRL ENVIRONMENT TEST")
    print("=" * 60)

    env = WamvEnv(
        control_dt=0.2,
        max_episode_time=120.0,
        thruster_min=0.0,
        thruster_max=1000.0,
    )

    try:

        # ----------------------------------------------------
        # Reset environment
        # ----------------------------------------------------

        obs, info = env.reset()

        print("\nInitial observation:")
        print(obs)

        print("\nInitial info:")
        print(info)

        # ----------------------------------------------------
        # Test a few actions
        # ----------------------------------------------------

        for i in range(1000):

            # Example:
            # both thrusters = 50% command
            action = np.array(
                [0.0, 0.0],
                dtype=np.float32
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            print(
                f"\nStep {i + 1}"
            )

            print(
                f"Observation: {obs}"
            )

            print(
                f"Reward: {reward:.3f}"
            )

            print(
                f"Position: "
                f"({info['x']:.3f}, "
                f"{info['y']:.3f})"
            )

            print(
                f"Velocity: "
                f"u={info['u']:.3f}, "
                f"v={info['v']:.3f}, "
                f"r={info['r']:.3f}"
            )

            print(
                f"Thrusters: "
                f"L={info['left_thruster']:.1f}, "
                f"R={info['right_thruster']:.1f}"
            )

            if terminated or truncated:

                print("\nEpisode finished.")

                break

    finally:

        env.close()

        print("\nEnvironment closed.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()