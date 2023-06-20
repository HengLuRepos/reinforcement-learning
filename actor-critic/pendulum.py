import sys
sys.path.append('../')
import os
import gymnasium as gym
from sac import SAC
from policy_utils.config import Config
cheetah = gym.make("InvertedPendulum-v4")
config = Config(env_name="InvertedPendulum-v4", seed=1, buffer_batch_size=100)
if not os.path.exists(config.output_path):
  os.makedirs(config.output_path)
sac = SAC(env=cheetah, config=config, seed=1)
sac.train()

