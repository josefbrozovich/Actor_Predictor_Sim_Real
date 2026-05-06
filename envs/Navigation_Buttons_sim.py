# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.envs import ManagerBasedRLEnv

import isaaclab_tasks.manager_based.navigation.mdp as mdp
# from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg


# LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()
LOW_LEVEL_ENV_CFG = UnitreeGo2FlatEnvCfg()


import torch

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import generated_commands


def tag_visible_from_xy(cmd: torch.Tensor):
    x = cmd[:, 0]
    y = cmd[:, 1]

    dist_sq = x * x + y * y

    min_dist_sq = 0.2 * 0.2
    max_dist_sq = 3.0 * 3.0

    tan_half_fov = math.tan(math.radians(70.0 / 2.0))

    visible = (
        (x > 0.0)
        & (torch.abs(y) < x * tan_half_fov)
        & (dist_sq > min_dist_sq)
        & (dist_sq < max_dist_sq)
    )

    return visible

def xy_only_with_visibility(env: ManagerBasedRLEnv, command_name: str):

    cmd = generated_commands(env, command_name=command_name)

    visible = tag_visible_from_xy(cmd)

    x = cmd[:, 0]
    y = cmd[:, 1]


    out = torch.zeros((cmd.shape[0], 3), device=cmd.device, dtype=cmd.dtype)
    out[:, 0] = x * visible
    out[:, 1] = y * visible
    out[:, 2] = visible.float()

    return out


# making ltl class
class temporal_logic:

    def reset_memory(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        # set memory
        if not hasattr(env, "blue"):
            env.blue = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            env.green = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # reset memory
        env.blue[env_ids] = False
        env.green[env_ids] = False


    def target_reached(env, threshold, name_pos):

        target_pos = env.command_manager.get_command(name_pos)[:, :2]

        reached = torch.norm(target_pos, dim=-1) <= threshold 
        visible = tag_visible_from_xy(target_pos)

        return reached & visible


    def memory_reward_AB(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
        """Updates the history of visited targets and returns them as an observation."""
        
        new_blue = temporal_logic.target_reached(env, threshold, "pose_command_blue")
        new_green = temporal_logic.target_reached(env, threshold, "pose_command_green")

        # update visited
        env.blue = env.blue | new_blue
        env.green = env.green | new_green

        goal_reached_mask = (env.blue&env.green).clone()

        # reward is 1 or 0
        reward = goal_reached_mask.float()

        # reset blue and green when goal is reached
        env.blue[goal_reached_mask] = False
        env.green[goal_reached_mask] = False

        return reward


@configclass
class EventCfg:
    """Configuration for events."""

    # reset physical memory
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_memory = EventTerm(
        func=temporal_logic.reset_memory,
        mode="reset",
    )

@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        # policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        policy_path="low_level_controller/go2.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # real life will have IMU data
        # Robot to ground
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # April Tag 1
        target_blue_local = ObsTerm(
            func=xy_only_with_visibility,
            params = {
                "command_name": "pose_command_blue"
            }
        )

        # April Tag 2
        target_green_local = ObsTerm(
            func=xy_only_with_visibility,
            params = {
                "command_name": "pose_command_green"
            }
        )

        # Last Action
        last_action = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    goal_reached = RewTerm(
        func=temporal_logic.memory_reward_AB,
        weight=5.0,
        params={
            "threshold": 0.5
        }
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command_blue = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(135, 135),
        debug_vis=True,
        # ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(0.0, 0.0), pos_y=(4, 4), heading=(0, 0)),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(0.0, 0.0), pos_y=(3, 3), heading=(0, 0)),
    )

    pose_command_green = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(360, 360),
        debug_vis=True,
        # ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(4, 4), pos_y=(0.0, 0.0), heading=(0, 0)),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(3, 3), pos_y=(0.0, 0.0), heading=(0, 0)),
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.8},
    )

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class Navigation_Buttons(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10

        self.episode_length_s = self.commands.pose_command_blue.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class Navigation_Buttons_PLAY(Navigation_Buttons):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

