from actor_critic import AdvantageActorCritic
import gymnasium as gym
from policy_utils.config import Config
seed = 1
env = gym.make("InvertedPendulum-v4")
config = Config("InvertedPendulum-v4",seed=seed)
import os
if not os.path.exists(config.output_path):
  os.makedirs(config.output_path)
a2c = AdvantageActorCritic(env,seed=seed,config=config)
a2c.train()