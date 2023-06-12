import sys
sys.path.append('../')

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from policy_utils.policy import CategoricalPolicy, GuassianPolicy
from policy_utils.utils import build_mlp, device, np2torch, get_logger
from collections import deque
#only for continuous state space
class Actor(nn.Module):
  def __init__(self, env, config):
    super().__init__()
    self.env = env
    self.config = config
    self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
    self.observation_dim = self.env.observation_space.shape[0]
    self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

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
    self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=self.config.lr)
  def forward(self, observations):
    return self.policy.action(observations)
  def sample_path(self, env, num_episodes=None):
    episode = 0
    paths = []
    episode_rewards = []
    t = 0

    while num_episodes or t < self.config.batch_size:
      state, _ = env.reset()
      states, actions, rewards, dones, log_probs, entropies = [], [], [], [], [], []
      episode_reward = 0

      for step in range(self.config.max_ep_len):
        states.append(state)
        action, log_prob, entropy= self.policy(states[-1][None])
        actions.append(action)
        log_probs.append(log_prob)
        entropies.append(entropy)
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        dones.append(done)
        episode_reward += reward
        t += 1
        if done or step == self.config.max_ep_len - 1:
          episode_rewards.append(episode_reward)
          break
        if (not num_episodes) and t == self.config.batch_size:
          break
      path = {
        "observation": np.array(states),
        "action": np.array(actions),
        "reward": np.array(rewards),
        "log_prob": np.array(log_probs),
        "done": np.array(dones),
        "entropy": np.array(entropies)
      }
      paths.append(path)
      episode += 1
      if num_episodes and episode >= num_episodes:
        break
    return paths, episode_rewards
  def update_actor(self, log_probs, advantages):
    #TODO: change this to PPO
    log_probs = np2torch(log_probs)
    loss = -torch.sum(torch.mul(log_probs,advantages.detach()))
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
                                   layer_size=self.config.layer_size,
                                   n_layers=self.config.n_layers).to(device)
    self.optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.config.lr)
  def update_value(self, observations, returns):
    observations = np2torch(observations)
    returns = np2torch(returns)
    for _ in range(self.config.value_update_freq):
      values = self.value_network(observations).squeeze()
      self.optimizer.zero_grad()
      loss = torch.nn.functional.mse_loss(values,returns)
      loss.backward()
      self.optimizer.step()
  def forward(self,x):
    x = np2torch(x)
    return self.value_network(x).squeeze()
  def calculate_return(self, paths):
    all_returns = []
    for path in paths:
      rewards = path["reward"]
      returns = deque([])
      total = 0
      for i in range(len(rewards)-1, -1, -1):
        total = rewards[i] + self.config.gamma * total
        returns.appendleft(total)
      all_returns.append(list(returns))
    return np.concatenate(all_returns)
#TODO: fix bugs
class AdvantageActorCritic(nn.Module):
  def __init__(self, env, seed, config):
    super().__init__()
    self.env = env
    self.seed = seed
    self.config = config
    self.env.reset(seed=self.seed)
    self.actor = Actor(self.env,self.config)
    self.critic = Critic(self.env.observation_space.shape[0], config)
    self.logger = get_logger(self.config.log_path)
  def calculate_advantage(self, observations, rewards):
    rewards = np2torch(rewards)
    values = self.critic(observations)
    advantages = rewards - values
    temp = torch.roll(values,-1)
    temp[len(temp) - 1] = 0
    advantages += self.config.gamma * temp
    return advantages
  def train(self):
    averaged_total_rewards = []
    for t in range(self.config.num_batches):
      paths, total_rewards = self.actor.sample_path(self.env)
      states = np.concatenate([path["observation"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      log_probs = np.concatenate([path["log_prob"] for path in paths])
      returns = self.critic.calculate_return(paths)


      self.critic.update_value(states, returns)
      advantages = self.calculate_advantage(states, rewards)
      self.actor.update_actor(log_probs, advantages)
      avg_reward = np.mean(total_rewards)
      std_reward = np.sqrt(np.var(total_rewards)/len(total_rewards))
      msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(t,avg_reward,std_reward)
      averaged_total_rewards.append(avg_reward)
      self.logger.info(msg)
    self.logger.info("- Training done.")
    np.save(self.config.scores_output, averaged_total_rewards)
