import torch

from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType, TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg


@configclass
class AllegroHandCfg(BaseTaskCfg):
    episode_length = 600
    traj_filepath = None
    task_type = TaskType.TABLETOP_MANIPULATION

    objects = [
        RigidObjCfg(
            name="block",
            usd_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            mjcf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            urdf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            default_position=(0.0, -0.2, 0.56),
            default_orientation=(1.0, 0.0, 0.0, 0.0),
        ),
        RigidObjCfg(
            name="goal",
            usd_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            mjcf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            urdf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            default_position=(0.0, 0.0, 0.92),
            default_orientation=torch.nn.functional.normalize(torch.rand(4), p=2, dim=0),
            physics=PhysicStateType.XFORM,
        ),
    ]

    def reward_fn(self, states):
        # Reward constants
        dist_reward_scale = -10.0
        rot_reward_scale = 1.0
        rot_eps = 0.1
        action_penalty_scale = -0.0002
        success_tolerance = 0.1
        reach_goal_bonus = 250.0
        fall_dist = 0.24
        fall_penalty = 0.0

        # Handle both multi-env (IsaacGym) and single-env (Mujoco) formats
        rewards = []
        for env_state in states:
            # Extract necessary state information from env_state
            allegro_hand_state = env_state.get("allegro_hand", {})
            kuka_state = env_state.get("kuka", {})
            object_state = env_state.get("object", {})
            goal_state = env_state.get("goal", {})
            actions = env_state.get("actions", torch.zeros(16))  # Default to zeros if actions not provided

            # Extract object and goal poses
            object_pos = object_state.get("pos", torch.zeros(3))
            object_rot = object_state.get("rot", torch.tensor([1.0, 0.0, 0.0, 0.0]))  # w,x,y,z quaternion

            goal_pos = goal_state.get("pos", torch.zeros(3))
            goal_rot = goal_state.get("rot", torch.tensor([1.0, 0.0, 0.0, 0.0]))

            # Distance from object to goal
            goal_dist = torch.norm(object_pos - goal_pos, p=2)

            # Orientation alignment between object and goal
            # Convert quaternions to rotation matrices and compute angular distance
            quat_diff = self._quat_mul(object_rot, self._quat_conjugate(goal_rot))
            rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[0:3]), max=1.0))

            # Calculate reward components
            dist_rew = goal_dist * dist_reward_scale
            rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

            # Action penalty to encourage smooth motions
            action_penalty = torch.sum(actions**2)

            # Total reward
            reward = -dist_rew + rot_rew - action_penalty * action_penalty_scale

            # Success bonus
            if goal_dist < success_tolerance and rot_dist < success_tolerance:
                reward += reach_goal_bonus

            # Fall penalty if object is too far from goal
            if goal_dist >= fall_dist:
                reward += fall_penalty

            rewards.append(reward)

        # Return concatenated rewards
        return torch.tensor(rewards) if rewards else torch.tensor([0.0])

    def _quat_mul(self, a, b):
        """Multiply two quaternions."""
        x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
        x2, y2, z2, w2 = b[0], b[1], b[2], b[3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return torch.tensor([x, y, z, w])

    def _quat_conjugate(self, q):
        """Conjugate of quaternion."""
        return torch.tensor([-q[0], -q[1], -q[2], q[3]])

    def termination_fn(self, states):
        # Handle both multi-env (IsaacGym) and single-env (Mujoco) formats
        terminations = []
        for env_state in states:
            # Extract necessary state information
            robot_state = env_state.get("robots", {}).get("allegro_hand", {})
            block_state = env_state.get("objects", {}).get("block", {})
            goal_state = env_state.get("objects", {}).get("goal", {})
            progress = env_state.get("progress", 0)  # Current episode step count

            # Extract object and goal poses
            robot_pos = robot_state.get("pos", torch.zeros(3))
            block_pos = block_state.get("pos", torch.zeros(3))
            block_rot = block_state.get("rot", torch.tensor([1.0, 0.0, 0.0, 0.0]))
            goal_rot = goal_state.get("rot", torch.tensor([1.0, 0.0, 0.0, 0.0]))
            # Constants (matching those used in reward function)
            fall_dist = 0.24
            max_episode_length = 600  # From class definition

            # Calculate distance to goal
            goal_dist = torch.norm(block_pos - robot_pos, p=2)

            # Termination conditions:
            # 1. Episode timeout
            # 2. Object has fallen/moved too far from goal
            terminate = (progress >= max_episode_length - 1) or (goal_dist >= fall_dist)

            terminations.append(terminate)

        # Return concatenated terminations
        return torch.tensor(terminations) if terminations else torch.tensor([False])
