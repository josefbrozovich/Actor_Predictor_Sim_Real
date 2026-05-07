import sys
import time
import copy
import cv2
import numpy as np
import torch
from pupil_apriltags import Detector

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


class TemporalLogic:
    def __init__(self):
        self.blue = False
        self.green = False

    def reset_memory(self):
        self.blue = False
        self.green = False

    @staticmethod
    def target_reached(tag_estimates, tag_id: int, threshold: float) -> bool:
        x = tag_estimates[tag_id]["x"]
        y = tag_estimates[tag_id]["y"]
        visible = tag_estimates[tag_id]["visible"]

        dist = np.sqrt(x * x + y * y)
        return (dist <= threshold) and (visible >= 0.5)

    def memory_reward_AB(self, tag_estimates, threshold: float) -> float:
        new_blue = self.target_reached(tag_estimates, tag_id=1, threshold=threshold)
        new_green = self.target_reached(tag_estimates, tag_id=2, threshold=threshold)

        self.blue = self.blue or new_blue
        self.green = self.green or new_green

        goal_reached = self.blue and self.green
        reward = 1.0 if goal_reached else 0.0

        if goal_reached:
            self.reset_memory()

        return reward


class Navigation_Buttons_real:
    def __init__(self, network_interface=None):
        # ----------------------------
        # AprilTag / camera parameters
        # ----------------------------
        self.TAG_SIZE = 0.20
        self.FX = 1000.0
        self.FY = 1000.0
        self.TAG_IDS = [1, 2]

        # ----------------------------
        # Safety / timing
        # ----------------------------
        self.MAX_VX = 0.2
        self.MAX_YAW_RATE = 0.4
        self.DT = 0.1

        # For discrete action mapping, if you want it later
        self.dvel = torch.tensor([-0.2, 0.0, 0.2, 0.0])
        self.dtheta = torch.tensor([0.0, 0.2, 0.0, -0.2])

        if network_interface is None:
            if len(sys.argv) < 2:
                raise RuntimeError("Usage: python Navigation_Buttons_real.py <network_interface>")
            network_interface = sys.argv[1]

        self.network_interface = network_interface

        # ----------------------------
        # Initialize Unitree DDS
        # ----------------------------
        ChannelFactoryInitialize(0, self.network_interface)

        # ----------------------------
        # LowState / IMU subscriber
        # ----------------------------
        self.low_state = None
        self.have_low_state = False

        # Most common Go2 low-state topic
        self.low_state_topic = "rt/lowstate"

        self.low_state_subscriber = ChannelSubscriber(self.low_state_topic, LowState_)
        self.low_state_subscriber.Init(self._low_state_callback, 10)

        # ----------------------------
        # Camera client
        # ----------------------------
        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()

        # ----------------------------
        # Sport control client
        # ----------------------------
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        self.sport_client.StopMove()

        # ----------------------------
        # AprilTag detector
        # ----------------------------
        self.detector = Detector(families="tagStandard52h13")

        self.tag_estimates = {
            1: {"x": 0.0, "y": 0.0, "visible": 0.0},
            2: {"x": 0.0, "y": 0.0, "visible": 0.0},
        }

        self.temporal_logic = TemporalLogic()
        self.info = {}

    # ============================================================
    # IMU / GRAVITY
    # ============================================================

    def _low_state_callback(self, msg):
        """
        Called automatically when a new Unitree LowState message arrives.
        """
        self.low_state = msg
        self.have_low_state = True

    @staticmethod
    def _quat_wxyz_to_projected_gravity(quat):
        """
        Convert Unitree IMU quaternion to projected gravity in robot body frame.

        Assumption:
            Unitree quaternion format is [w, x, y, z].

        Output:
            gravity vector expressed in robot body frame.

        Expected:
            robot upright: approximately [0, 0, -1]
        """
        w, x, y, z = quat

        # Rotation matrix: body frame -> world frame
        R_body_to_world = np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float32)

        # World frame -> body frame
        R_world_to_body = R_body_to_world.T

        # Unit gravity in world frame
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Gravity projected into robot body frame
        g_body = R_world_to_body @ g_world

        return g_body.astype(np.float32)

    def _get_projected_gravity(self):
        """
        Returns gravity vector in robot body frame.

        If low state has not arrived yet, returns the default upright gravity.
        """
        if not self.have_low_state or self.low_state is None:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

        quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)

        # Protect against uninitialized quaternion
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

        quat = quat / quat_norm

        return self._quat_wxyz_to_projected_gravity(quat)

    # ============================================================
    # OBSERVATION
    # ============================================================

    def _make_obs(self):
        """
        Observation structure:

        obs["policy"] =
        [
            gravity_x,
            gravity_y,
            gravity_z,

            tag_1_x,
            tag_1_y,
            tag_1_visible,

            tag_2_x,
            tag_2_y,
            tag_2_visible,
        ]

        Shape: [1, 9]
        """
        gravity = self._get_projected_gravity()

        obs = np.array([
            gravity[0],
            gravity[1],
            gravity[2],

            self.tag_estimates[1]["x"],
            self.tag_estimates[1]["y"],
            self.tag_estimates[1]["visible"],

            self.tag_estimates[2]["x"],
            self.tag_estimates[2]["y"],
            self.tag_estimates[2]["visible"],
        ], dtype=np.float32)

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        return {"policy": obs}

    # ============================================================
    # APRILTAG UPDATE
    # ============================================================

    def _update_tags_from_camera(self):
        code, data = self.client.GetImageSample()
        if code != 0:
            raise RuntimeError(f"Failed to get image from Unitree camera. code={code}")

        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if frame is None:
            raise RuntimeError("Could not decode image from Unitree camera.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cx = frame.shape[1] / 2.0
        cy = frame.shape[0] / 2.0

        # Mark all tags invisible unless detected this frame
        for tag_id in self.TAG_IDS:
            self.tag_estimates[tag_id]["visible"] = 0.0

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[self.FX, self.FY, cx, cy],
            tag_size=self.TAG_SIZE,
        )

        for det in detections:
            tag_id = det.tag_id

            if tag_id not in self.TAG_IDS:
                continue

            p_cam = det.pose_t.reshape(3)

            x_cam = float(p_cam[0])
            z_cam = float(p_cam[2])

            # Approximate camera frame -> robot frame conversion
            #
            # Camera:
            #   x_cam: right
            #   z_cam: forward
            #
            # Robot:
            #   x_robot: forward
            #   y_robot: left
            x_robot = z_cam
            y_robot = -x_cam

            self.tag_estimates[tag_id]["x"] = x_robot
            self.tag_estimates[tag_id]["y"] = y_robot
            self.tag_estimates[tag_id]["visible"] = 1.0

    # ============================================================
    # GYM-LIKE API
    # ============================================================

    def reset(self):
        self.sport_client.StopMove()
        self.temporal_logic.reset_memory()

        for tag_id in self.TAG_IDS:
            self.tag_estimates[tag_id]["x"] = 0.0
            self.tag_estimates[tag_id]["y"] = 0.0
            self.tag_estimates[tag_id]["visible"] = 0.0

        obs = self._make_obs()

        self.info = {
            "tag_estimates": copy.deepcopy(self.tag_estimates),
            "blue_memory": self.temporal_logic.blue,
            "green_memory": self.temporal_logic.green,
            "gravity": obs["policy"][0, 0:3].clone(),
            "have_low_state": self.have_low_state,
        }

        return obs, self.info

    def step(self, actions):
        """
        Expected action shape:

            actions = tensor([[vx, something, yaw_rate]])

        Your current code uses:
            vx       = actions[0, 0]
            yaw_rate = actions[0, 2]

        Unitree Sport API:
            Move(vx, vy, vyaw)
        """
        print(f"actions: {actions}")

        vx = float(actions[0, 0].item())
        yaw_rate = float(actions[0, 2].item())

        # Safety clipping
        vx = float(np.clip(vx, -self.MAX_VX, self.MAX_VX))
        yaw_rate = float(np.clip(yaw_rate, -self.MAX_YAW_RATE, self.MAX_YAW_RATE))

        # Command robot
        self.sport_client.Move(vx, 0.0, yaw_rate)

        time.sleep(self.DT)

        # Update AprilTags
        self._update_tags_from_camera()

        # Build observation
        obs = self._make_obs()

        # Compute reward
        reward_float = self.temporal_logic.memory_reward_AB(
            self.tag_estimates,
            threshold=0.5,
        )

        reward = torch.tensor([reward_float], dtype=torch.float32)
        terminated = torch.tensor([False])
        truncated = torch.tensor([False])

        self.info = {
            "tag_estimates": copy.deepcopy(self.tag_estimates),
            "blue_memory": self.temporal_logic.blue,
            "green_memory": self.temporal_logic.green,
            "vx": vx,
            "yaw_rate": yaw_rate,
            "gravity": obs["policy"][0, 0:3].clone(),
            "have_low_state": self.have_low_state,
        }

        return obs, reward, terminated, truncated, self.info

    def close(self):
        self.sport_client.StopMove()


if __name__ == "__main__":
    env = Navigation_Buttons_real()
    obs, info = env.reset()

    try:
        for i in range(100):
            # Test action shape must match step():
            # actions[0, 0] = vx
            # actions[0, 2] = yaw_rate
            action = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            print("step:", i)
            print("action:", action)
            print("obs:", obs["policy"])
            print("gravity:", info["gravity"])
            print("have_low_state:", info["have_low_state"])
            print("reward:", reward.item())
            print("info:", info)
            print()

    finally:
        env.close()