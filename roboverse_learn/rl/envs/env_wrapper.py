from __future__ import annotations

import gymnasium as gym

ISAACGYM_AVAILABLE = True

import logging
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.utils.setup_util import SimType, get_sim_env_class

log = logging.getLogger(__name__)


@contextmanager
def timing_context(name: str, verbose: bool = False, timing_dict: dict | None = None):
    """Context manager for timing code blocks"""
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"{name}: {elapsed_time:.4f}s")
    if timing_dict is not None:
        timing_dict[name] = timing_dict.get(name, 0.0) + elapsed_time


class RLEnv:
    """Wrapper for RL infra"""

    def __init__(self, sim_name: str, scenario: ScenarioCfg, seed: int | None = None, verbose: bool = False):
        # Set seed if provided
        if seed is not None:
            self.set_seed(seed)

        self.scenario = scenario
        env_class = get_sim_env_class(SimType(sim_name))
        self.env = env_class(scenario)

        # Additional variables for RL
        self._register_configs()
        self._set_up_buffers()

        # Add timing tracking
        self.verbose = verbose
        self.step_timings = {}
        self.total_steps = 0
        self.total_time = 0.0

        self.writer = None
        self.global_step = 0

    def _register_configs(self):
        """Register configurations for the environment."""
        self.max_episode_length = self.scenario.task.episode_length
        self.handler = self.env.handler
        self.headless = self.scenario.headless
        self.num_envs = self.scenario.num_envs
        self._task = self.scenario.task
        self._robot = self.scenario.robot
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rgb_observation = len(self.scenario.cameras) > 0
        self.obs_space = self.get_observation_space()

        if self.rgb_observation:
            self.camera_resolution_width = self.scenario.cameras[0].width
            self.camera_resolution_height = self.scenario.cameras[0].height

    def get_observation_space(self):
        """Get the observation space for the environment."""
        if self.rgb_observation:
            return gym.spaces.Box(
                low=0, high=255, shape=(3, self.camera_resolution_height, self.camera_resolution_width), dtype=np.uint8
            )
        else:
            return gym.spaces.Dict({"obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)})

    def _set_up_buffers(self):
        """Set up buffers for the environment."""
        self.reset_buffer = torch.zeros(self.num_envs, 1, device=self.device)
        self.success_buffer = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.rgb_observation:
            self.rgb_buffers = torch.zeros(
                self.num_envs, 3, self.camera_resolution_width, self.camera_resolution_height, device=self.device
            )

    def _randomize(self):
        """Randomize the environment."""
        pass

    def _sanity_check(self):
        """Sanity check the environment."""
        if not hasattr(self.handler.task, "reward_fn"):
            raise ValueError("Task does not have a reward function")
        if not hasattr(self.handler.task, "termination_fn"):
            log.warning("Task does not have a termination function. Assuming task is episodic.")

    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        if seed == -1 and torch.cuda.is_available():
            seed = torch.randint(0, 10000, (1,))[0].item()
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return seed

    def get_reward(self) -> torch.Tensor:
        """Calculate rewards based on task configuration."""
        # Get current states from the environment
        return self.env._get_reward()

    def get_termination(self) -> torch.Tensor:
        """Get termination states of all environments."""
        return self.env._get_termination()

    def get_success(self) -> torch.Tensor:
        """Get success states of all environments."""
        return self.success_buffer.clone()

    def get_timeout(self) -> torch.Tensor:
        """Check if environments have timed out."""
        timeout = self.timestep_buffer >= self.max_episode_length
        return timeout

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Step the environment forward."""
        self.total_steps += 1
        if self.verbose:
            print(f"\n=== Step {self.total_steps} ===")

        state, reward, success, timeout, info = self.env.step(action)
        observation = self.get_observation(state)

        done = success | timeout
        return (
            observation,
            reward,
            done,
            timeout,
            None,
        )

    def reset_handler(self, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        # Reset episode length buffer
        self.handler._episode_length_buf = [0 for _ in range(self.handler.num_envs)]

        # Create default states for resetting robot and objects to their initial positions
        default_states = []
        for _ in range(self.handler.num_envs):
            env_state = {"objects": {}, "robots": {}}

            # Set default positions/rotations for objects
            for obj in self.handler.objects:
                obj_state = {"pos": obj.default_position, "rot": obj.default_orientation}

                # Add joint positions for articulated objects
                if hasattr(obj, "default_joint_positions") and obj.default_joint_positions:
                    obj_state["dof_pos"] = obj.default_joint_positions

                env_state["objects"][obj.name] = obj_state

            # Set default position/rotation for robot
            robot_state = {"pos": self.handler.robot.default_position, "rot": self.handler.robot.default_orientation}

            # Add default joint positions for the robot
            if hasattr(self.handler.robot, "default_joint_positions") and self.handler.robot.default_joint_positions:
                robot_state["dof_pos"] = self.handler.robot.default_joint_positions

            env_state["robots"][self.handler.robot.name] = robot_state
            default_states.append(env_state)

        # Reset states using the handler's set_states method
        self.handler.set_states(default_states, env_ids=env_ids)

        # Then call the original checker reset
        self.handler.checker.reset(self.handler, env_ids=env_ids)

        # Simulate to apply the changes
        self.handler.simulate()

        # Get observation after reset
        obs = self.handler.get_observation()

        return obs, None

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset all environments."""
        # Call the handler's reset method which uses configurations
        self.reset_handler()

        # Reset internal state tracking buffers
        self.reset_buffer = torch.ones_like(self.reset_buffer)
        self.timestep_buffer = torch.zeros_like(self.timestep_buffer)
        self.success_buffer = torch.zeros_like(self.success_buffer)

        # Get observation after reset
        return self.get_observation(self.handler.get_states())

    def get_observation(self, states: dict) -> dict:
        """Process the observation from the handler."""
        obs = {}
        obs_tensor = torch.zeros(
            self.num_envs, 3, self.camera_resolution_height, self.camera_resolution_width, device=self.device
        )
        for state in states:
            if self.rgb_observation:
                rgb_frame = state["cameras"][0]["rgb"]
                rgb_frame = rgb_frame.view(
                    self.num_envs, 3, self.camera_resolution_height, self.camera_resolution_width
                )
                obs["obs"] = rgb_frame

            joint_dict = state["robots"][self.handler.robot.name]["dof_pos"]

        return obs

    def reset_idx(self, env_ids: list[int]) -> None:
        """Reset specific environments to initial configuration."""
        # Call the handler's reset method with specific environment IDs
        self.reset_handler(env_ids=env_ids)

        # Reset internal state tracking only for specified environments
        self.reset_buffer[env_ids] = torch.ones_like(self.reset_buffer[env_ids])
        self.timestep_buffer[env_ids] = torch.zeros_like(self.timestep_buffer[env_ids])
        self.success_buffer[env_ids] = torch.zeros_like(self.success_buffer[env_ids])

    def render(self) -> None:
        """Render the environment."""
        self.handler.render()

    def close(self) -> None:
        """Close the environment."""
        self.handler.close()

    def set_verbose(self, verbose: bool) -> None:
        """Enable or disable verbose timing output"""
        self.verbose = verbose

    def print_timing_stats(self) -> None:
        """Print timing statistics"""
        if self.total_steps == 0:
            return

        print("=== Timing Statistics ===")
        print(f"Total steps: {self.total_steps}")
        print(f"Total time: {self.total_time:.4f}s")
        print(f"Average time per step: {self.total_time / self.total_steps:.4f}s")
        print("\nBreakdown by operation:")
        for op, time_in_secs in self.step_timings.items():
            avg_time = time_in_secs / self.total_steps
            percentage = (time_in_secs / self.total_time) * 100
            print(f"{op:30s}: {avg_time:.4f}s ({percentage:.1f}%)")
        print("=====================")
