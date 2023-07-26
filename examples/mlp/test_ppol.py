import os
from dataclasses import asdict
import pandas as pd
import bullet_safety_gym
import torch
try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
# from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from fsrl.utils.wrapper.venv import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv,SafeShmemVectorEnv
from fsrl.agent import PPOLagAgent
from fsrl.config.ppol_cfg import (
    Bullet1MCfg,
    Bullet5MCfg,
    Bullet10MCfg,
    Mujoco2MCfg,
    Mujoco10MCfg,
    Mujoco20MCfg,
    MujocoBaseCfg,
    TrainCfg,
)
from fsrl.utils.net.common import ActorCritic
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from collections import OrderedDict
from guard.safe_rl_envs.safe_rl_envs.envs.engine import Engine
from guard.safe_rl_lib.utils.safe_rl_env_config import configuration
from fsrl.utils.wrapper.guard_wrapper import VecWrapper, create_env
TASK_TO_CFG = {
    # bullet safety gym tasks
    "SafetyCarRun-v0": Bullet1MCfg,
    "SafetyBallRun-v0": Bullet1MCfg,
    "SafetyBallCircle-v0": Bullet1MCfg,
    "SafetyCarCircle-v0": TrainCfg,
    "SafetyDroneRun-v0": TrainCfg,
    "SafetyAntRun-v0": TrainCfg,
    "SafetyDroneCircle-v0": Bullet5MCfg,
    "SafetyAntCircle-v0": Bullet10MCfg,
    # safety gymnasium tasks
    "SafetyPointCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyHalfCheetahVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyHopperVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetySwimmerVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyWalker2dVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyAntVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyHumanoidVelocityGymnasium-v1": Mujoco20MCfg,
}

def test(name, args: TrainCfg, task):
    args.task = task
    print(args.task)
    checkpoint = torch.load(name)
    print(args)
    env = create_env(args)
    worker = eval(args.worker)
    test_envs = VecWrapper(lambda: create_env(args), worker, 20, args)

    agent = PPOLagAgent(
        env=env,
        # logger=args.logger,
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        lr=args.lr,
        hidden_sizes=args.hidden_sizes,
        unbounded=args.unbounded,
        last_layer_scale=args.last_layer_scale,
        target_kl=args.target_kl,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        gae_lambda=args.gae_lambda,
        eps_clip=args.eps_clip,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_limit,
        rescaling=args.rescaling,
        gamma=args.gamma,
        max_batchsize=args.max_batchsize,
        reward_normalization=args.rew_norm,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
    )

    model = checkpoint['model']
    model_load = OrderedDict()
    for (k, v) in model.items():
        # print(k, v)
        if k[0] == '_':
            continue
        else:
            model_load[k] = v
            

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    net = Net(state_shape, hidden_sizes=(128, 128), device='cpu')
    actor = ActorProb(
        net, action_shape, max_action=max_action, unbounded=False, device='cpu'
    ).to('cpu')
    critic = [
        Critic(
            Net(state_shape, hidden_sizes=(128, 128), device='cpu'),
            device='cpu'
        ).to('cpu') for _ in range(2) # 2 means 1 reward + 1 cost 
    ]
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    actor_critic = ActorCritic(actor, critic)
    actor_critic.load_state_dict(model_load)
    agent.policy._actor_critic.load_state_dict(model_load)
    # from FSRL.fsrl.data import FastCollector
    from fsrl.data import FastCollector
    agent.policy.eval()
    collector = FastCollector(policy=agent.policy, env=test_envs)
    # print(collector)
    rews, cost = 0, 0
    cnt_ = 5
    for i in range(cnt_):
        result = collector.collect(n_episode=10, render=False)
        print(result)
        rews += result["rew"]
        cost += result["cost"]
    print(f"Final eval reward: {rews/cnt_}, cost_draw: {cost/cnt_}")
    return rews/cnt_, cost/cnt_



if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    img = Image.open("./figure/2q1H.png")
    img = np.array(img)
    print(img.shape)
    # import sys
    # name = "/home/yuqing/GUARD_FSRL/examples/mlp/logs/fast-safe-rl/"
    # task = sys.argv[1]
    # robot = sys.argv[2]
    # constr = sys.argv[3]
    # cost = sys.argv[4]
    # label = sys.argv[5]
    # agent = sys.argv[6]
    # train_task = task + '_' + robot + '_8' + constr
    # name += task + '_' + robot + '_8' + constr + '-cost-' + cost + '/ppol_cost' + cost + '.0-' + label
    # # name += task + '_' + robot + '_8' + constr + '-cost-' + cost + '/ppol-' + label
    # name += '/checkpoint/model_best.pt'
    # max_rew, max_cost = test(name, args=TrainCfg, task=train_task)
    # list = [task, robot, constr, cost, max_rew, max_cost]  # modif    
    # data = pd.DataFrame([list])
    # data.to_csv('./result/ppol_test.csv',mode='a',header=False,index=False)
