import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
# from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.agent import TRPOLagAgent
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all
from fsrl.utils.wrapper.guard_wrapper import create_env, VecWrapper
from fsrl.utils.wrapper.venv import SafeShmemVectorEnv
from tianshou.data import Batch
import sys

@dataclass
class EvalConfig:
    name: str = "Chase"
    cost: str = "0"
    label: str = "a301"
    path: str = "/home/yuqing/GUARD_FSRL/examples/mlp/logs/fast-safe-rl/" + name + '_Point_8Hazards-cost-' + cost +"/cvpo_cost" + cost + ".0-" + label
    best: bool = True
    eval_episodes: int = 20
    parallel_eval: bool = True
    device: str = "cpu"
    render: bool = False
    train_mode: bool = False


@pyrallis.wrap()
def eval(args: EvalConfig):
    path = "/home/yuqing/GUARD_FSRL/examples/mlp/logs/fast-safe-rl/" + args.name + '_Point_8Hazards-cost-' + args.cost +"/trpol_cost" + args.cost + ".0-" + args.label
    # print(path)
    cfg, model = load_config_and_model(path, args.best)

    task = cfg["task"]
    args_ = Batch(task=task, max_episode_steps=20000)
    demo_env = create_env(args_)
    # demo_env = gym.make(task)

    agent = TRPOLagAgent(
        env=demo_env,
        logger=BaseLogger(),
        device=args.device,
        use_lagrangian=cfg["use_lagrangian"],
        thread=cfg["thread"],
        seed=cfg["seed"],
        hidden_sizes=cfg["hidden_sizes"],
        unbounded=cfg["unbounded"],
        last_layer_scale=cfg["last_layer_scale"],
    )

    if args.parallel_eval:
        # test_envs = ShmemVectorEnv(
            # [lambda: gym.make(task) for _ in range(args.eval_episodes)]
        # )
        test_envs = VecWrapper(lambda: create_env(args_), SafeShmemVectorEnv, args.eval_episodes, args_)
    else:
        test_envs = create_env(args_)
        # test_envs = gym.make(task)

    rews, lens, cost, cost_dict = agent.evaluate(
        test_envs=test_envs,
        state_dict=model["model"],
        eval_episodes=args.eval_episodes,
        render=args.render,
        train_mode=args.train_mode
    )
    print("Traing mode: ", args.train_mode)
    print(f"Eval reward: {rews}, cost: {cost}, length: {lens}, cost_dict: {cost_dict}")
    list = [args.name, "Point", "Hazards", args.cost, rews, cost]
    import pandas as pd   
    data = pd.DataFrame([list])
    data.to_csv('./result/trpol_test.csv',mode='a',header=False,index=False)



if __name__ == "__main__":
    eval()
