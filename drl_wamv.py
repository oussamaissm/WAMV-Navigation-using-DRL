#!/usr/bin/env python3

import csv
import math
import os
import subprocess
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float64


WORLD_NAME = "sydney_regatta"
WAMV_NAME = "wamv"

# ---------------------------------------------------------
# WAM-V starting position
# ---------------------------------------------------------
START_WORLD_X = -800.0
START_WORLD_Y = 450.0

# ---------------------------------------------------------
# Circuit waypoints (closed loop). Random targets are sampled
# INSIDE the area enclosed by this polygon.
# ---------------------------------------------------------
WAYPOINTS = [
    (-800.0, 300.0),
    (-720.0, 370.0),
    (-700.0, 450.0),
    (-750.0, 530.0),
    (-900.0, 510.0),
]

# ---------------------------------------------------------
# Navigation parameters
# ---------------------------------------------------------
DESIRED_SPEED = 1.0
GOAL_TOLERANCE = 2.0

# ---------------------------------------------------------
# Reward parameters
# ---------------------------------------------------------
SUCCESS_BONUS = 100.0


class WamvNode:
    """ROS 2 interface to the real VRX WAM-V."""

    def __init__(self):
        self.node = rclpy.create_node("wamv_drl_node")

        self.left_pub = self.node.create_publisher(
            Float64, "/wamv/thrusters/left/thrust", 10
        )
        self.right_pub = self.node.create_publisher(
            Float64, "/wamv/thrusters/right/thrust", 10
        )

        self.gps_sub = self.node.create_subscription(
            NavSatFix,
            "/wamv/sensors/gps/gps/fix",
            self.gps_callback,
            10,
        )
        self.imu_sub = self.node.create_subscription(
            Imu,
            "/wamv/sensors/imu/imu/data",
            self.imu_callback,
            10,
        )

        self.gps_lat: Optional[float] = None
        self.gps_lon: Optional[float] = None
        self.gps_lat0: Optional[float] = None
        self.gps_lon0: Optional[float] = None

        self.imu = None

        self.prev_gps_x = None
        self.prev_gps_y = None
        self.prev_gps_time = None

        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        # Last published thrust values (used for the reward's effort term).
        self.left_thrust = 0.0
        self.right_thrust = 0.0

    def gps_callback(self, msg: NavSatFix):
        if not np.isfinite(msg.latitude) or not np.isfinite(msg.longitude):
            return

        now = time.monotonic()

        self.gps_lat = msg.latitude
        self.gps_lon = msg.longitude

        if self.gps_lat0 is None:
            self.gps_lat0 = msg.latitude
            self.gps_lon0 = msg.longitude

        x, y = self.gps_to_local(msg.latitude, msg.longitude)

        if self.prev_gps_time is not None:
            dt = now - self.prev_gps_time

            if dt > 1e-4:
                vx = (x - self.prev_gps_x) / dt
                vy = (y - self.prev_gps_y) / dt

                # GPS velocity is expressed in the local/world frame.
                # The body-frame surge/sway values need heading information,
                # so for this simple environment we keep the local values.
                self.u = float(vx)
                self.v = float(vy)

        self.prev_gps_x = x
        self.prev_gps_y = y
        self.prev_gps_time = now

    def imu_callback(self, msg: Imu):
        self.imu = msg
        self.r = float(msg.angular_velocity.z)

    def gps_to_local(self, lat, lon):
        """Approximate GPS displacement in meters around the first GPS point."""
        earth_radius = 6378137.0

        d_lat = math.radians(lat - self.gps_lat0)
        d_lon = math.radians(lon - self.gps_lon0)

        x = earth_radius * d_lon * math.cos(math.radians(self.gps_lat0))
        y = earth_radius * d_lat

        return float(x), float(y)

    def publish_thrust(self, left: float, right: float):
        left_msg = Float64()
        right_msg = Float64()

        left_msg.data = float(left)
        right_msg.data = float(right)

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

        self.left_thrust = float(left)
        self.right_thrust = float(right)

    def stop(self):
        self.publish_thrust(0.0, 0.0)

    def reset_wamv_pose(
            self,
            x: float,
            y: float,
            z: float = 0.0,
            yaw: float = 0.0,
            max_retries: int = 5,
    ) -> bool:
        """
        Reset the WAM-V pose in Gazebo using Gazebo Transport.

        Gazebo can occasionally be busy during long DRL training runs.
        Therefore, the reset command is retried several times.

        Returns
        -------
        bool
            True  -> Gazebo confirmed the reset.
            False -> all reset attempts failed.
        """

        # ---------------------------------------------------------
        # 1. Convert yaw angle to quaternion
        # ---------------------------------------------------------
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        # ---------------------------------------------------------
        # 2. Build the Gazebo Pose request
        # ---------------------------------------------------------
        request = (
            f'name: "{WAMV_NAME}", '
            f'position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{'
            f'x: 0.0, '
            f'y: 0.0, '
            f'z: {qz}, '
            f'w: {qw}'
            f'}}'
        )

        # ---------------------------------------------------------
        # 3. Gazebo service command
        # ---------------------------------------------------------
        command = [
            "gz",
            "service",

            "-s",
            f"/world/{WORLD_NAME}/set_pose",

            "--reqtype",
            "gz.msgs.Pose",

            "--reptype",
            "gz.msgs.Boolean",

            # Give Gazebo more time to answer
            "--timeout",
            "10000",

            "--req",
            request,
        ]

        # ---------------------------------------------------------
        # 4. Try several times
        # ---------------------------------------------------------
        for attempt in range(1, max_retries + 1):

            self.node.get_logger().info(
                f"Resetting WAM-V "
                f"(attempt {attempt}/{max_retries})..."
            )

            try:

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,

                    # Python timeout must be larger than
                    # the Gazebo CLI timeout.
                    timeout=15.0,
                )

            except subprocess.TimeoutExpired:

                self.node.get_logger().warn(
                    f"Gazebo reset timed out "
                    f"(attempt {attempt}/{max_retries})."
                )

                time.sleep(1.0)
                continue

            except FileNotFoundError:

                self.node.get_logger().error(
                    "The 'gz' command was not found. "
                    "Make sure Gazebo is correctly sourced."
                )

                return False

            # -----------------------------------------------------
            # 5. Collect stdout + stderr
            # -----------------------------------------------------
            output = (
                    result.stdout + "\n" + result.stderr
            ).strip()

            # -----------------------------------------------------
            # 6. Check command execution
            # -----------------------------------------------------
            if result.returncode != 0:
                self.node.get_logger().warn(
                    f"Gazebo reset failed "
                    f"(attempt {attempt}/{max_retries}).\n"
                    f"{output}"
                )

                time.sleep(1.0)
                continue

            # -----------------------------------------------------
            # 7. Check Gazebo confirmation
            # -----------------------------------------------------
            if "true" not in output.lower():
                self.node.get_logger().warn(
                    f"Gazebo reset returned without confirmation "
                    f"(attempt {attempt}/{max_retries}).\n"
                    f"{output}"
                )

                time.sleep(1.0)
                continue

            # -----------------------------------------------------
            # 8. SUCCESS
            # -----------------------------------------------------
            self.node.get_logger().info(
                f"WAM-V reset successfully to "
                f"({x:.2f}, {y:.2f}), "
                f"yaw={yaw:.3f} rad"
            )

            return True

        # ---------------------------------------------------------
        # 9. All attempts failed
        # ---------------------------------------------------------
        self.node.get_logger().error(
            f"Could not reset WAM-V after "
            f"{max_retries} attempts."
        )

        return False

    def reset_sensor_state(self):
        """Clear values derived from measurements before waiting for fresh data."""
        self.gps_lat = None
        self.gps_lon = None
        self.imu = None

        self.prev_gps_x = None
        self.prev_gps_y = None
        self.prev_gps_time = None

        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        self.left_thrust = 0.0
        self.right_thrust = 0.0


def point_in_polygon(x: float, y: float, polygon) -> bool:
    """
    Ray-casting point-in-polygon test.

    `polygon` is a list of (x, y) tuples describing a closed
    (or implicitly closed) simple polygon.
    """
    inside = False
    n = len(polygon)

    x1, y1 = polygon[0]

    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]

        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        ):
            inside = not inside

        x1, y1 = x2, y2

    return inside


def polygon_bounding_box(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    return min(xs), max(xs), min(ys), max(ys)


def polygon_centroid(polygon):
    """Simple average of vertices (good enough as a default goal)."""
    pts = polygon[:-1] if polygon[0] == polygon[-1] else polygon

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    return cx, cy


def sample_point_in_polygon(polygon, max_attempts: int = 1000):
    """
    Rejection-sample a random point strictly inside `polygon`,
    using its bounding box as the proposal distribution.
    """
    x_min, x_max, y_min, y_max = polygon_bounding_box(polygon)

    for _ in range(max_attempts):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        if point_in_polygon(x, y, polygon):
            return float(x), float(y)

    # Fallback: centroid is always inside for a convex-ish polygon.
    return polygon_centroid(polygon)


class WorldReference:
    """
    Convert the local GPS coordinate system into the VRX world coordinates.

    The first valid GPS measurement is associated with:
        (-800.32, 300.02)
    """

    def __init__(self):
        self.initialized = False
        self.offset_x = 0.0
        self.offset_y = 0.0

    def local_to_world(self, x_local, y_local):
        if not self.initialized:
            self.offset_x = START_WORLD_X - x_local
            self.offset_y = START_WORLD_Y - y_local
            self.initialized = True

        return (
            x_local + self.offset_x,
            y_local + self.offset_y,
        )


class TrajectoryLogger:
    """Save one episode trajectory to CSV."""

    def __init__(self, filename="results/wamv_trajectory.csv"):
        self.filename = filename
        self.data = []

    def start(self):
        self.data = []

    def log(
        self,
        step,
        x,
        y,
        goal_x,
        goal_y,
        distance,
        reward,
        left_thrust,
        right_thrust,
        terminated,
        truncated,
    ):
        self.data.append(
            {
                "step": step,
                "x": x,
                "y": y,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "distance": distance,
                "reward": reward,
                "left_thrust": left_thrust,
                "right_thrust": right_thrust,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    def save(self):
        if not self.data:
            return

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        fieldnames = list(self.data[0].keys())

        with open(self.filename, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)

        print(f"Trajectory saved to: {self.filename}")


class WamvEnv(gym.Env):
    """
    Gymnasium environment controlling the real VRX WAM-V.

    Observation:
        [error_x_body,
         error_y_body,
         heading_error,
         u,
         v,
         r,
         desired_speed]

    Action:
        [-1, 1] for left and right thrusters.

    Mapping:
        -1 -> 0 thrust
         0 -> 500 thrust
         1 -> 1000 thrust

    The target position is now randomized within
    [TARGET_X_MIN, TARGET_X_MAX] x [TARGET_Y_MIN, TARGET_Y_MAX]
    at the start of every episode (see randomize_goal()).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        control_dt: float = 0.2,
        max_episode_time: float = 120.0,
    ):
        super().__init__()

        self.control_dt = control_dt
        self.max_episode_time = max_episode_time

        self.node = WamvNode()
        self.reference = WorldReference()

        self.trajectory_logger = TrajectoryLogger()

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )

        # Current target (overwritten by randomize_goal() on every reset).
        self.goal_x, self.goal_y = polygon_centroid(WAYPOINTS)

        self.episode_step = 0
        self.start_time = None
        self.previous_distance = None

        self.current_x = START_WORLD_X
        self.current_y = START_WORLD_Y
        self.current_heading = 0.0

    def randomize_goal(self):
        """Generate a new random target inside the WAYPOINTS circuit."""
        self.goal_x, self.goal_y = sample_point_in_polygon(WAYPOINTS)

        self.node.node.get_logger().info(
            f"New target: ({self.goal_x:.2f}, {self.goal_y:.2f})"
        )

    def spin_once(self):
        rclpy.spin_once(
            self.node.node,
            timeout_sec=0.0,
        )

    def wait_for_sensors(self, timeout=5.0):
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            rclpy.spin_once(
                self.node.node,
                timeout_sec=0.1,
            )

            if (
                self.node.gps_lat is not None
                and self.node.imu is not None
            ):
                return True

        return False

    def get_world_position(self):
        if self.node.gps_lat is None or self.node.gps_lon is None:
            return self.current_x, self.current_y

        x_local, y_local = self.node.gps_to_local(
            self.node.gps_lat,
            self.node.gps_lon,
        )

        return self.reference.local_to_world(
            x_local,
            y_local,
        )

    def get_heading(self):
        if self.node.imu is None:
            return self.current_heading

        q = self.node.imu.orientation

        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        return math.atan2(sin_yaw, cos_yaw)

    @staticmethod
    def angle_normalize(angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    def get_observation(self):
        self.current_x, self.current_y = self.get_world_position()
        self.current_heading = self.get_heading()

        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y

        # Transform position error from world frame to body frame.
        cos_h = math.cos(self.current_heading)
        sin_h = math.sin(self.current_heading)

        error_x_body = cos_h * dx + sin_h * dy
        error_y_body = -sin_h * dx + cos_h * dy

        desired_heading = math.atan2(dy, dx)
        heading_error = self.angle_normalize(
            desired_heading - self.current_heading
        )

        return np.array(
            [
                error_x_body,
                error_y_body,
                heading_error,
                self.node.u,
                self.node.v,
                self.node.r,
                DESIRED_SPEED,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # -------------------------------------------------
        # 1. Stop the WAM-V before moving it.
        # -------------------------------------------------
        self.node.stop()

        # -------------------------------------------------
        # 2. Generate a new random target for this episode.
        # -------------------------------------------------
        self.randomize_goal()

        # -------------------------------------------------
        # 3. Reset the WAM-V to the fixed starting position,
        #    heading toward the freshly sampled target.
        # -------------------------------------------------
        initial_heading = math.atan2(
            self.goal_y - START_WORLD_Y,
            self.goal_x - START_WORLD_X,
        )

        reset_ok = self.node.reset_wamv_pose(
            x=START_WORLD_X,
            y=START_WORLD_Y,
            z=0.0,
            yaw=initial_heading,
        )

        if not reset_ok:
            raise RuntimeError("Could not reset WAM-V pose in Gazebo.")

        # -------------------------------------------------
        # 4. Reset sensor state.
        #    Do not reset gps_lat0/gps_lon0: they define the
        #    persistent GPS reference frame.
        # -------------------------------------------------
        self.node.reset_sensor_state()

        self.episode_step = 0
        self.start_time = time.monotonic()
        self.previous_distance = None

        self.current_x = START_WORLD_X
        self.current_y = START_WORLD_Y
        self.current_heading = initial_heading

        self.trajectory_logger.start()

        # -------------------------------------------------
        # 5. Wait for fresh measurements after teleporting.
        # -------------------------------------------------
        if not self.wait_for_sensors(timeout=5.0):
            self.node.stop()
            raise RuntimeError(
                "No fresh GPS/IMU measurements received after reset."
            )

        # -------------------------------------------------
        # 6. Build initial observation w.r.t. the NEW target.
        # -------------------------------------------------
        observation = self.get_observation()

        distance = math.hypot(
            self.goal_x - self.current_x,
            self.goal_y - self.current_y,
        )

        self.previous_distance = distance

        info = {
            "x": self.current_x,
            "y": self.current_y,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
            "distance_to_goal": distance,
            "reset_ok": reset_ok,
        }

        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        # [-1, 1] -> [0, 1000]
        left_thrust = float((action[0] + 1.0) * 500.0)
        right_thrust = float((action[1] + 1.0) * 500.0)

        self.node.publish_thrust(
            left_thrust,
            right_thrust,
        )

        # Let Gazebo and ROS produce the next sensor measurements.
        end_time = time.monotonic() + self.control_dt

        while time.monotonic() < end_time:
            rclpy.spin_once(
                self.node.node,
                timeout_sec=0.02,
            )

        self.episode_step += 1

        observation = self.get_observation()

        distance = math.hypot(
            self.goal_x - self.current_x,
            self.goal_y - self.current_y,
        )

        heading_error = float(observation[2])
        speed = math.sqrt(
            float(observation[3]) ** 2
            + float(observation[4]) ** 2
        )

        if self.previous_distance is None:
            progress = 0.0
        else:
            progress = self.previous_distance - distance

        self.previous_distance = distance

        # Reward:
        #   distance term      -> approach the goal
        #   heading term       -> point toward the goal
        #   speed term         -> approach desired speed
        #   progress term      -> reward actual progress
        #   effort term        -> avoid unnecessarily large thrust
        reward = (
            -2.0 * distance
            -0.5 * abs(heading_error)
            -0.25 * abs(speed - DESIRED_SPEED)
            +2.0 * progress
            -0.05 * (
                abs(left_thrust) + abs(right_thrust)
            ) / 1000.0
        )

        terminated = distance <= GOAL_TOLERANCE

        # Success bonus when the (randomized) goal is reached.
        if terminated:
            reward += SUCCESS_BONUS

            self.node.node.get_logger().info(
                f"Goal reached! Target=({self.goal_x:.2f}, {self.goal_y:.2f})"
            )

        elapsed = time.monotonic() - self.start_time
        truncated = elapsed >= self.max_episode_time

        if terminated or truncated:
            self.node.stop()

        self.trajectory_logger.log(
            step=self.episode_step,
            x=self.current_x,
            y=self.current_y,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            distance=distance,
            reward=reward,
            left_thrust=left_thrust,
            right_thrust=right_thrust,
            terminated=terminated,
            truncated=truncated,
        )

        if terminated or truncated:
            self.trajectory_logger.save()

        info = {
            "x": self.current_x,
            "y": self.current_y,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
            "distance_to_goal": distance,
            "heading_error": heading_error,
            "speed": speed,
            "progress": progress,
            "elapsed_time": elapsed,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.node.stop()
        self.trajectory_logger.save()

        if rclpy.ok():
            self.node.node.destroy_node()
            rclpy.shutdown()


def main():
    rclpy.init()

    env = WamvEnv(
        control_dt=0.2,
        max_episode_time=120.0,
    )

    try:
        observation, info = env.reset()

        print("Initial observation:", observation)
        print("Initial info:", info)

        # Test with neutral action:
        # [0, 0] -> left/right thrust = 500 / 500.
        for step in range(100):
            action = np.array(
                [0.0, 0.0],
                dtype=np.float32,
            )

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            print(
                f"step={step:03d} "
                f"x={info['x']:.2f} "
                f"y={info['y']:.2f} "
                f"goal=({info['goal_x']:.2f}, {info['goal_y']:.2f}) "
                f"distance={info['distance_to_goal']:.2f} "
                f"reward={reward:.3f}"
            )

            if terminated or truncated:
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()