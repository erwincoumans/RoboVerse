from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.utils.setup_util import SimType, get_sim_env_class


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


class IsaacLabWrapper:
    """Wrapper around IsaaclabHandler for RL algorithms."""

    def __init__(self, scenario: ScenarioCfg, num_envs: int = 1, headless: bool = False, seed: int | None = None):
        """Initialize the wrapper with an IsaaclabHandler."""
        # Set seed if provided
        if seed is not None:
            self.set_seed(seed)

        scenario.num_envs = num_envs
        scenario.headless = headless
        env_class = get_sim_env_class(SimType(scenario.sim))
        env = env_class(scenario)
        self.handler = env.handler

        # Store configuration
        self.headless = headless
        self.num_envs = num_envs
        self._task = scenario.task
        self._robot = scenario.robot
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Additional variables for RL
        self.max_episode_length = 30 if not scenario.task.episode_length else scenario.task.episode_length
        self.reset_buffer = torch.zeros(self.num_envs, 1, device=self.device)
        self.timestep_buffer = torch.zeros(self.num_envs, 1, device=self.device)

        # Add success tracking
        self.success_buffer = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Add timing tracking
        self.verbose = True
        self.step_timings = {}
        self.total_steps = 0
        self.total_time = 0.0

        self.rgb_buffers = [[] for _ in range(self.num_envs)]
        self.writer = None
        self.global_step = 0

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

    def launch(self) -> None:
        """Launch the simulation."""
        self.handler.launch()

    def get_observation(self) -> dict[str, torch.Tensor]:
        """Get observations from the environment."""
        # Get observations from the handler
        handler_obs = self.handler.get_observation()

        # Get all states for extracting joint positions
        states = self.handler.get_states()

        # Extract joint positions from the states for each environment
        joint_positions = []
        for env_id in range(self.num_envs):
            robot_name = self._robot.name
            robot_state = states[env_id][robot_name]

            # Get robot joint positions
            robot_joint_pos = [pos for pos in robot_state["dof_pos"].values()]

            # Also get articulated object joint positions if available
            for obj_name, obj_state in states[env_id].items():
                if obj_name != robot_name and isinstance(obj_state, dict) and "dof_pos" in obj_state:
                    for joint_pos in obj_state["dof_pos"].values():
                        robot_joint_pos.append(joint_pos)

            joint_positions.append(robot_joint_pos)

        # Convert to torch tensor
        obs_tensor = torch.tensor(joint_positions, device=self.device, dtype=torch.float32)

        # Create observation dictionary
        observation = {"obs": obs_tensor}

        # Add RGB images if available
        if "rgb" in handler_obs and handler_obs["rgb"] is not None:
            try:
                rgb_tensors = []
                if isinstance(handler_obs["rgb"], list) and len(handler_obs["rgb"]) == self.num_envs:
                    rgb_tensors = [tensor.to(self.device) for tensor in handler_obs["rgb"]]
                else:
                    for env_id in range(self.num_envs):
                        if env_id < len(handler_obs["rgb"]):
                            rgb_tensors.append(handler_obs["rgb"][env_id].to(self.device))
                        else:
                            rgb_tensors.append(handler_obs["rgb"][0].to(self.device))

                observation["rgb"] = torch.stack(rgb_tensors, dim=0).float()
            except Exception as e:
                print(f"Warning: Could not process RGB images from handler: {e}")

        # Add depth images if available
        if "depth" in handler_obs and handler_obs["depth"] is not None:
            try:
                depth_tensors = []
                if isinstance(handler_obs["depth"], list) and len(handler_obs["depth"]) == self.num_envs:
                    depth_tensors = [tensor.to(self.device) for tensor in handler_obs["depth"]]
                else:
                    for env_id in range(self.num_envs):
                        if env_id < len(handler_obs["depth"]):
                            depth_tensors.append(handler_obs["depth"][env_id].to(self.device))
                        else:
                            depth_tensors.append(handler_obs["depth"][0].to(self.device))

                observation["depth"] = torch.stack(depth_tensors, dim=0).float()
            except Exception as e:
                print(f"Warning: Could not process depth images from handler: {e}")

        return observation

    def get_reward(self) -> torch.Tensor:
        """Calculate rewards based on task configuration."""
        # Get current states from the environment
        states = self.handler.get_states()

        # Case 1: Task has a direct reward_fn
        if hasattr(self.handler.task, "reward_fn"):
            return self.handler.task.reward_fn(states).to(self.device)

        # Case 2: Task has reward_functions and reward_weights
        from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg

        if isinstance(self.handler.task, BaseRLTaskCfg):
            final_reward = torch.zeros(self.handler.num_envs, device=self.device)
            for reward_func, reward_weight in zip(self.handler.task.reward_functions, self.handler.task.reward_weights):
                # Apply each reward function to the states and add the weighted result
                reward_component = reward_func(self.handler.get_states()) * reward_weight
                final_reward += reward_component
            return final_reward

        return torch.zeros(self.handler.num_envs, device=self.device)

    def get_termination(self) -> torch.Tensor:
        """Get termination states of all environments."""
        if hasattr(self.handler.task, "termination_fn"):
            return self.handler.task.termination_fn(self.handler.get_states())
        return torch.full((self.handler.num_envs,), False, device=self.device)

    def get_success(self) -> torch.Tensor:
        """Get success states of all environments."""
        success = self.handler.checker.check(self.handler)
        self.success_buffer = success.to(self.device)
        return self.success_buffer.clone()

    def get_timeout(self) -> torch.Tensor:
        """Check if environments have timed out."""
        timeout = self.timestep_buffer >= self.max_episode_length
        return timeout

    def step(
        self, action: list[dict]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment forward."""
        step_start = time.time()
        self.total_steps += 1

        if self.verbose:
            print(f"\n=== Step {self.total_steps} ===")

        self.timestep_buffer += 1

        # Step the environment using the handler
        with timing_context("step_simulation", self.verbose, self.step_timings):
            _, _, success, time_out, extras = self.handler.step(action)

        # Get observation and compute rewards
        with timing_context("get_observation", self.verbose, self.step_timings):
            observation = self.get_observation()

        with timing_context("compute_rewards", self.verbose, self.step_timings):
            reward = self.get_reward().to(self.device)
            timeout = self.get_timeout().to(self.device)
            termination = self.get_termination().to(self.device)

        # Handle resets for timed out environments
        with timing_context("handle_timeouts", self.verbose, self.step_timings):
            timeout_indices = torch.nonzero(timeout.squeeze(1)).squeeze(1).tolist()
            if timeout_indices:
                self.reset_idx(timeout_indices)
                updated_obs = self.get_observation()
                obs_tensor = observation["obs"]
                obs_tensor[timeout_indices] = updated_obs["obs"][timeout_indices]
                observation["obs"] = obs_tensor
                # Also update image observations if available
                for key in ["rgb", "depth"]:
                    if key in observation and key in updated_obs:
                        img_tensor = observation[key]
                        img_tensor[timeout_indices] = updated_obs[key][timeout_indices]
                        observation[key] = img_tensor

        # Handle resets for environments that are done (success or termination)
        with timing_context("handle_dones", self.verbose, self.step_timings):
            done_indices = torch.nonzero((success | termination) & ~timeout.squeeze(1)).squeeze(1).tolist()
            if done_indices:
                self.reset_idx(done_indices)
                updated_obs = self.get_observation()
                obs_tensor = observation["obs"]
                obs_tensor[done_indices] = updated_obs["obs"][done_indices]
                observation["obs"] = obs_tensor
                # Also update image observations if available
                for key in ["rgb", "depth"]:
                    if key in observation and key in updated_obs:
                        img_tensor = observation[key]
                        img_tensor[done_indices] = updated_obs[key][done_indices]
                        observation[key] = img_tensor

        dones = timeout.squeeze(1) | success | termination

        # Update timing statistics
        step_time = time.time() - step_start
        self.total_time += step_time

        if self.verbose:
            print(f"Step {self.total_steps} completed in {step_time:.4f}s")
            if success.any():
                success_envs = torch.nonzero(success).squeeze(-1).tolist()
                print(f"Success in environments: {success_envs}")
            if timeout.any():
                timeout_envs = torch.nonzero(timeout).squeeze(-1).tolist()
                print(f"Timeout in environments: {timeout_envs}")
            if termination.any():
                termination_envs = torch.nonzero(termination).squeeze(-1).tolist()
                print(f"Termination in environments: {termination_envs}")
            print("=" * 30 + "\n")

        reward = reward.unsqueeze(1)
        info = {"success": success}
        return (
            observation,
            reward,
            dones,
            timeout,
            info,
        )

    def reset_handler(self, env_ids: list[int] | None = None):
        """Reset the handler and checker for the specified environments."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        # Reset the handler using the existing handler.reset method
        self.handler.reset(env_ids=env_ids)

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset all environments."""
        # Call the handler's reset method
        self.reset_handler()

        # Reset internal state tracking buffers
        self.reset_buffer = torch.ones_like(self.reset_buffer)
        self.timestep_buffer = torch.zeros_like(self.timestep_buffer)
        self.success_buffer = torch.zeros_like(self.success_buffer)

        # Get observation after reset
        return self.get_observation()

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
        self.handler.refresh_render()

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
