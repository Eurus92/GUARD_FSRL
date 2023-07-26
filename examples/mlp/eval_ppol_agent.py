import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import os.path as osp
import yaml
import torch
import pandas as pd   

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
# from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.agent import PPOLagAgent
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all
from fsrl.utils.wrapper.guard_wrapper import create_env, VecWrapper, create_env_noconti, VecWrapper_noconti
from fsrl.utils.wrapper.venv import SafeShmemVectorEnv
from tianshou.data import Batch
import sys

@dataclass
class EvalConfig:
    name: str = "Chase"
    cost: str = "0"
    label: str = "a301"
    path: str = "/home/yuqing/GUARD_FSRL/examples/mlp/logs/fast-safe-rl/" + name + '_Point_8Hazards-cost-' + cost +"/cvpo_cost" + cost + ".0-" + label
    best: bool = False
    eval_episodes: int = 20
    parallel_eval: bool = True
    device: str = "cpu"
    render: bool = False
    train_mode: bool = False


@pyrallis.wrap()
def eval(args: EvalConfig):
    # cfg, model = load_config_and_model(args.path, args.best)
    # path = "/home/yuqing/GUARD_FSRL/examples/mlp/logs/fast-safe-rl/" + args.name + '_Point_8Hazards-cost-' + args.cost +"/cvpo_cost" + args.cost + ".0-" + args.label
    path = "/home/yuqing/GUARD_FSRL/examples/mlp/logs/fast-safe-rl/Goal_Arm3_8Hazards_Pillar-cost-" + args.cost + "/ppol_cost" + args.cost + ".0-" + args.label
    # path = "/home/yuqing/GUARD_FSRL/examples/mlp/logs/fast-safe-rl/haz1_pil1_vas1_noconti-cost-1000/ppol_cost1000.0-4f23"
    # path = "/home/yuqing/GUARD_FSRL/examples/mlp/logs/fast-safe-rl/Goal_Arm3_8Hazards_Pillar-cost-10/ppol-4845"
    if osp.exists(path):
        config_file = osp.join(path, "config.yaml")
        print(f"load config from {config_file}")
        with open(config_file) as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    else:
        raise ValueError(f"{path} doesn't exist!")
    # cfg, model = load_config_and_model(path, args.best)

    task = cfg["task"]
    # demo_env = gym.make(task)
    args_ = Batch(task=task, max_episode_steps=20000)
    demo_env = create_env(args_)
    # demo_env = create_env_noconti(args_)

    agent = PPOLagAgent(
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
        # test_envs = VecWrapper_noconti(lambda: create_env_noconti(args_), SafeShmemVectorEnv, args.eval_episodes, args_)
    else:
        # test_envs = gym.make(task)
        test_envs = create_env(args_)
        # test_envs = create_env_noconti(args_)
    cost_type = "hazard3DS+pillar"
    cost_limit=args.cost
    def load_model(suffix: str):
        model_file = "model" + suffix + ".pt"
        model_path = osp.join(path, "checkpoint/" + model_file)
        print(f"load model from {model_path}")
        model = torch.load(model_path)
        return model
    def Eval_Step(model: dict):
        rews, lens, cost, cost_dict = agent.evaluate(
            test_envs=test_envs,
            state_dict=model["model"],
            eval_episodes=args.eval_episodes,
            render=args.render,
            train_mode=args.train_mode
        )
        print("Traing mode: ", args.train_mode)
        print(f"Eval reward: {rews}, cost: {cost}, length: {lens}, cost_dict: {cost_dict}")
        list = [cost_type, cost_limit, rews, cost, cost_dict["cost_hazard3Ds"], cost_dict["cost_pillars"]]
        data = pd.DataFrame([list])
        data.to_csv('./result/ppol_multi.csv',mode='a',header=False,index=False)
        return rews, cost, cost_dict
    
    if args.best:
        model = load_model("best")
        rews, cost, cost_dict = Eval_Step(model)
    else:
        tot_rews, tot_cost = 0, 0
        tot_cost_dict = Batch(
            cost_hazards = 0.0,
            cost_ghosts = 0.0,
            cost_hazard3Ds = 0.0,
            cost_ghost3Ds = 0.0,
            cost_gremlins = 0.0,
            cost_pillars = 0.0,
            cost_vases = 0.0,
        )
        for i in range(160, 204, 4):
            model = load_model("_" + str(i))
            rews, cost, cost_dict = Eval_Step(model)
            tot_rews += rews
            tot_cost += cost
            tot_cost_dict += cost_dict
        list = [cost_type+"_avg", cost_limit, tot_rews/11, tot_cost/11, tot_cost_dict["cost_hazard3Ds"]/11, tot_cost_dict["cost_pillars"]/11]
        data = pd.DataFrame([list])
        data.to_csv('./result/ppol_multi.csv',mode='a',header=False,index=False)
        print(tot_rews/11, tot_cost/11, tot_cost_dict/11)
    
    
    


if __name__ == "__main__":
    eval()
