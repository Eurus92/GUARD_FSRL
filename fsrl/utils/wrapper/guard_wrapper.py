from guard.safe_rl_lib.utils.safe_rl_env_config import configuration
from guard.safe_rl_lib.utils.safe_rl_env_config_noconti import configuration as config_noconti
from guard.safe_rl_envs.safe_rl_envs.envs.engine import Engine as safe_rl_envs_Engine    
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union
import gym
from tianshou.data import Batch


def create_env(args):
    env = safe_rl_envs_Engine(configuration(args.task))
    try:
        env.spec = Batch(max_episode_steps=args.max_episode_steps)
    except:
        env.spec = Batch(max_episode_steps=args.step_per_epoch)
        # print("warn: there is no max_episode_steps in configuration")
    # env.spec["max_episode_steps"] = args.max_episode_steps
    return env  

def create_env_noconti(args):
    env = safe_rl_envs_Engine(config_noconti(args.task, args))
    # env.spec = Batch(max_episode_steps=args.max_episode_steps)
    return env  

class VecWrapper(gym.Wrapper):
    def __init__(self, make_env, worker, training_num, args):
        super().__init__(make_env())
        self.env = make_env()
        self.envs = worker([lambda: create_env(args) for _ in range(training_num)])
        self.len = int(training_num)
        # self.spec = args.max_episode_steps
        self.__len__ = training_num
    
    def __len__(self):
        # print(type(self.len))
        return int(self.len)

    def reset(self, ids: Optional[Union[int, List[int], np.ndarray]] = None, **gym_reset_kwargs):
        return self.envs.reset(ids, **gym_reset_kwargs)
    
    def step(self, action_map: np.ndarray, ids: Optional[Union[int, List[int], np.ndarray]] = None):
        return self.envs.step(action_map, ids)
    
    def render(self):
        return self.envs.render(mode="rgb_array")


class VecWrapper_noconti(gym.Wrapper):
    def __init__(self, make_env, worker, training_num, args):
        super().__init__(make_env())
        # self.env = make_env()
        self.envs = worker([lambda: make_env() for _ in range(training_num)])
        self.len = int(training_num)
        # self.spec = args.max_episode_steps
        self.__len__ = training_num
    
    def __len__(self):
        # print(type(self.len))
        return int(self.len)

    def reset(self, ids: Optional[Union[int, List[int], np.ndarray]] = None, **gym_reset_kwargs):
        return self.envs.reset(ids, **gym_reset_kwargs)
    
    def step(self, action_map: np.ndarray, ids: Optional[Union[int, List[int], np.ndarray]] = None):
        return self.envs.step(action_map, ids)