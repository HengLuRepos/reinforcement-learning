import sys
sys.path.append('../')

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from policy_utils.policy import CategoricalPolicy, GuassianPolicy
from policy_utils.utils import build_mlp, device, np2torch, get_logger
import os

class Actor(nn.Module):
  def __init__(self, env, config):
    super().__init__()
    self.env = env
    self.observation_dim = self.env.observation_space.shape[0]
    self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
    self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
    self.config = config
    self.init_policy()
  def init_policy(self):
    network = build_mlp(self.observation_dim, 
                        self.action_dim, 
                        self.config.layer_size,
                        self.config.n_layers).to(device)
    self.policy = None
    if self.discrete:
      self.policy = CategoricalPolicy(network)
    else:
      self.policy = GuassianPolicy(network, self.action_dim)
    self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)
  def forward(self, ob):
    #take ONE observation, get action, log_prob
    ob = np2torch(ob)
    dist = self.policy.action_distribution(ob)
    action = dist.sample().detach().cpu().numpy()
    log_prob = dist.log_prob(np2torch(action)).detach().cpu().numpy()
    return action, log_prob
  def update_actor(self, ob, action, advantage):
    ob = np2torch(ob)
    action = np2torch(action)
    log_prob = self.policy.action_distribution(ob).log_prob(action)
    loss = -torch.sum(torch.mul(log_prob,advantage.detach()))
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
  
class Critic(nn.Module):
  def __init__(self, observation_dim, config):
    super().__init__()
    self.config = config
    self.observation_dim = observation_dim
    self.init_value()
  def init_value(self):
    self.value_network = build_mlp(self.observation_dim,
                        1,
                        self.config.layer_size,
                        self.config.n_layers).to(device)
    self.optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.config.lr)
  def forward(self, ob):
    ob = np2torch(ob)
    return torch.squeeze(self.value_network(ob))
  def update_critic(self, advantage):
    loss = torch.pow(advantage,2).mean()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
class A2C(nn.Module):
  def __init__(self, env, config, seed):
    super().__init__()
    self.env = env
    self.env.reset(seed=seed)
    self.config = config
    self.actor = Actor(self.env, self.config)
    self.critic = Critic(self.env.observation_space.shape[0], self.config)
    self.max_ep_num = self.config.max_ep_num
    self.logger = get_logger(self.config.log_path)
    if not os.path.exists(self.config.output_path):
      os.makedirs(self.config.output_path)
  def train(self):
    all_episode_return = []
    for episode in range(self.max_ep_num):
      done = False
      episode_return = 0
      state, _ = self.env.reset()
      while not done:
        action, log_prob = self.actor(state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        advantage = reward + self.critic(next_state) * (1.0 - done) * self.config.gamma - self.critic(state)
        self.critic.update_critic(advantage)
        self.actor.update_actor(state, action, advantage)
        state = next_state
        episode_return = episode_return * self.config.gamma + reward
      all_episode_return.append(episode_return)
      msg = "[EPISODE {}]: Epsodic reward: {:04.2f}".format(episode,episode_return)
      self.logger.info(msg)
    np.save(self.config.scores_output, all_episode_return)
