import sys
sys.path.append('../')
import os
import gymnasium as gym
from sac import SAC
from policy_utils.config import Config
import torch
cheetah = gym.make("HalfCheetah-v4")
config = Config(env_name="HalfCheetah-v4", seed=1, buffer_batch_size=100, gamma=0.9)
if not os.path.exists(config.output_path):
  os.makedirs(config.output_path)
sac = SAC(env=cheetah, config=config, seed=1)
sac.qconfig = torch.quantization.get_default_qconfig('x86')
torch.quantization.prepare(sac, inplace=True)
sac.train()
torch.quantization.convert(sac, inplace=True)
sac.save_model()


