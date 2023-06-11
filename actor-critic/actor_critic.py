import gymnasium as gym
import torch
import torch.nn as nn
from policy_utils.policy import CategoricalPolicy, GuassianPolicy
from policy_utils.utils import build_mlp, device, np2torch
class Actor(nn.Module):
  def __init__(self, env, config, seed):
    super.__init__()
    self.env = env
    self.config = config
    self.seed = seed
    self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
    self.observation_dim = self.env.observation_space.shape[0]
    self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

    self.init_policy()
  def init_policy(self):
    network = build_mlp(self.observation_dim,self.action_dim,self.config.layer_size, self.config.n_layers)
    self.policy = None
    if self.discrete:
      self.policy = CategoricalPolicy(network)
    else:
      self.policy = GuassianPolicy(network, self.action_dim)
  def forward(self, observations):
    pass
  def sample_path(self):
    pass

class Critic(nn.Module):
  def __init__(self, env):
    super().__init__()
    self.env = env
