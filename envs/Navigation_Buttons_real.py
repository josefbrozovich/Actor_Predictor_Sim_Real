import sys
import time
import copy
import cv2
import numpy as np
import torch
from pupil_apriltags import Detector

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
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
        # AprilTag / camera parameters
        self.TAG_SIZE = 0.20
        self.FX = 1000.0
        self.FY = 1000.0
        self.TAG_IDS = [1, 2]

        # Safety limits
        self.MAX_VX = 0.2
        self.MAX_YAW_RATE = 0.4
        self.DT = 0.1

        # In real environment, everything will run on cpu
        self.dvel = torch.tensor([-0.2, 0, 0.2, 0])
        self.dtheta = torch.tensor([0, 0.2, 0, -0.2])


        if network_interface is None:
            if len(sys.argv) < 2:
                raise RuntimeError("Usage: python script.py <network_interface>")
            network_interface = sys.argv[1]

        self.network_interface = network_interface

        ChannelFactoryInitialize(0, self.network_interface)

        # Camera client
        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()

        # Sport control client
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        self.sport_client.StopMove()

        # AprilTag detector
        self.detector = Detector(families="tagStandard52h13")

        self.tag_estimates = {
            1: {"x": 0.0, "y": 0.0, "visible": 0.0},
            2: {"x": 0.0, "y": 0.0, "visible": 0.0},
        }

        self.temporal_logic = TemporalLogic()
        self.info = {}

    def _make_obs(self):
        # Placeholder gravity observation.
        # Later, replace this with Go2 IMU gravity projected into body frame.
        gravity_x = 0.0
        gravity_y = 0.0
        gravity_z = -1.0

        obs = np.array([
            gravity_x,
            gravity_y,
            gravity_z,

            self.tag_estimates[1]["x"],
            self.tag_estimates[1]["y"],
            self.tag_estimates[1]["visible"],

            self.tag_estimates[2]["x"],
            self.tag_estimates[2]["y"],
            self.tag_estimates[2]["visible"],
        ], dtype=np.float32)

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return {"policy": obs}

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

            # Approximate camera-to-robot-frame conversion:
            # robot x = forward distance
            # robot y = left/right offset
            x_robot = z_cam
            y_robot = -x_cam

            self.tag_estimates[tag_id]["x"] = x_robot
            self.tag_estimates[tag_id]["y"] = y_robot
            self.tag_estimates[tag_id]["visible"] = 1.0

    def reset(self):
        self.sport_client.StopMove()
        self.temporal_logic.reset_memory()

        for tag_id in self.TAG_IDS:
            self.tag_estimates[tag_id]["x"] = 0.0
            self.tag_estimates[tag_id]["y"] = 0.0
            self.tag_estimates[tag_id]["visible"] = 0.0

        self.info = {
            "tag_estimates": copy.deepcopy(self.tag_estimates),
            "blue_memory": self.temporal_logic.blue,
            "green_memory": self.temporal_logic.green,
        }

        return self._make_obs(), self.info

    def step(self, actions):
        """
        actions are discrete, 0, 1, 2, 3
        """

        print(f"actions: {actions}")

        vx = actions[0,0].item()
        yaw_rate = actions[0,2].item()

        # Unitree Sport API: Move(vx, vy, vyaw)
        self.sport_client.Move(vx, 0.0, yaw_rate)

        time.sleep(self.DT)

        self._update_tags_from_camera()

        obs = self._make_obs()

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
        }

        return obs, reward, terminated, truncated, self.info

    def close(self):
        self.sport_client.StopMove()


if __name__ == "__main__":
    env = Navigation_Buttons_real()
    obs, info = env.reset()

    try:
        for i in range(100):
            # Discrete test action:
            # 0 = stop, 1 = forward, 2 = left, 3 = right
            action = torch.randint(0, 4, (1,))

            # Or continuous test action:
            # action = torch.tensor([[0.1, 0.0]])

            obs, reward, terminated, truncated, info = env.step(action)

            print("step:", i)
            print("action:", action)
            print("obs:", obs["policy"])
            print("reward:", reward.item())
            print("info:", info)
            print()

    finally:
        env.close()