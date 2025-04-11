from __future__ import annotations

import os
import time

import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

from roboverse_learn.rl.envs.env_wrapper import RLEnv


def set_np_formatting():
    """Formats numpy print"""
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Do simulator-specific imports first
    sim_name = cfg.environment.sim_name.lower()
    set_np_formatting()

    from metasim.cfg.scenario import ScenarioCfg
    from metasim.utils.setup_util import get_robot, get_task
    from roboverse_learn.rl.algos import get_algorithm

    if cfg.experiment.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        sim_device = f"cuda:{rank}"
        rl_device = f"cuda:{rank}"
    else:
        sim_device = f"cuda:{cfg.experiment.device_id}" if cfg.experiment.device_id >= 0 else "cpu"
        rl_device = f"cuda:{cfg.experiment.device_id}" if cfg.experiment.device_id >= 0 else "cpu"

    cprint("Start Building the Environment", "green", attrs=["bold"])
    task = get_task(cfg.train.task_name)
    robot = get_robot(cfg.train.robot_name)
    scenario = ScenarioCfg(task=task, robot=robot)
    scenario.cameras = []

    tic = time.time()
    # # Use our wrappers based on the simulator type
    # if sim_name == "isaacgym":
    #     # Import IsaacGymWrapper only when needed
    #     from roboverse_learn.rl.envs.isaacgym_wrapper import IsaacGymWrapper

    #     env = IsaacGymWrapper(
    #         scenario, cfg.environment.num_envs, headless=cfg.environment.headless, seed=cfg.experiment.seed
    #     )    scenario.sim_name = sim_name
    #     # Import MujocoWrapper only when needed
    #     from roboverse_learn.rl.envs.mujoco_wrapper import MujocoWrapper

    #     env = MujocoWrapper(
    #         scenario,sim_name
    #         seed=cfg.experiment.seed,
    #         rgb_observation=cfg.environment.rgb_observation,
    #     )
    #     env.launch()  # Initialize the environment
    # elif sim_name == "isaaclab":
    #     # For IsaacLab, we need a special import order as in replay_demo.py
    #     # The import must happen only after isaacgym is imported
    #     from roboverse_learn.rl.envs.isaaclab_wrapper import IsaacLabWrapper

    #     env = IsaacLabWrapper(
    #         scenario, cfg.environment.num_envs, headless=cfg.environment.headless, seed=cfg.experiment.seed
    #     )
    #     # IsaacLab must launch right after importing
    #     env.launch()  # Initialize the environment
    # else:
    # env_class = get_sim_env_class(SimType(sim_name))
    # env = env_class(scenario, cfg.environment.num_envs, headless=cfg.environment.headless)
    scenario.num_envs = cfg.environment.num_envs
    scenario.headless = cfg.environment.headless
    env = RLEnv(sim_name, scenario)
    # Set seed if the environment has a set_seed method
    if hasattr(env, "set_seed"):
        env.set_seed(cfg.experiment.seed)

    # Enable verbose mode if needed
    if hasattr(env, "set_verbose"):
        env.set_verbose(False)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    output_dif = os.path.join("outputs", cfg.experiment.output_name)
    os.makedirs(output_dif, exist_ok=True)

    # Initialize agent using the algorithm registry
    algo_name = cfg.train.algo.lower()
    agent = get_algorithm(algo_name, env=env, output_dif=output_dif, full_config=OmegaConf.to_container(cfg))

    # Example of accessing config parameters (you can remove these)
    log.info(f"Algorithm: {cfg.train.algo}")
    log.info(f"Number of environments: {cfg.environment.num_envs}")
    log.info(f"RL Device: {rl_device}")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.experiment.output_name,
        sync_tensorboard=True,
        mode=cfg.wandb.mode,
        settings=wandb.Settings(
            _disable_stats=True,  # Reduce overhead
            _disable_meta=True,  # Reduce overhead
        ),
    )

    if cfg.experiment.resume_training:
        agent.load(cfg.experiment.checkpoint)
    agent.train()
    wandb.finish()


if __name__ == "__main__":
    main()
