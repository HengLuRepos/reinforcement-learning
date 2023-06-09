import gymnasium as gym
import torch
import os
from config import config_cartpole, config_pendulum, config_cheetah
from policy_gradient import PolicyGradient
from ppo import PPO

pendulum = gym.make("Pendulum-v1")
config = config_pendulum(use_baseline=True,ppo=True,seed=1)
if not os.path.exists(config.output_path):
  os.makedirs(config.output_path)
pg_baseline = PPO(env=pendulum, config=config, seed=1)
pg_baseline.run()
