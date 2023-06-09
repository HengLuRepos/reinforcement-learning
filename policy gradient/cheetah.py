import gymnasium as gym
import torch
import os
from config import config_cheetah
from policy_gradient import PolicyGradient
from ppo import PPO

cheetah = gym.make("HalfCheetah-v4")
config = config_cheetah(use_baseline=True,ppo=True,seed=2)
if not os.path.exists(config.output_path):
  os.makedirs(config.output_path)
pg_baseline = PPO(env=cheetah, config=config, seed=2)
pg_baseline.run()
