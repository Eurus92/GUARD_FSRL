from guard.safe_rl_lib.utils.safe_rl_env_config import configuration
from guard.safe_rl_envs.safe_rl_envs.envs.engine import Engine as safe_rl_envs_Engine    
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union
import gym


def create_env(args):
    env = safe_rl_envs_Engine(configuration(args.task))
    return env  

class InitWrapper(gym.Wrapper):
    def __init__(self, make_env, worker, training_num, args):
        super().__init__(make_env())
        self.env = make_env()
        self.envs = worker([lambda: create_env(args) for _ in range(training_num)])
        self.len = training_num
    
    def __len__(self):
        return self.len

    def reset(self, ids: Optional[Union[int, List[int], np.ndarray]] = None, **gym_reset_kwargs):
        return self.envs.reset(ids, **gym_reset_kwargs)
    
    def step(self, action_map: np.ndarray, ids: Optional[Union[int, List[int], np.ndarray]] = None):
        return self.envs.step(action_map, ids)