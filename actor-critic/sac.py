import sys
sys.path.append('../')

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
from  collections import deque
from policy_utils.utils import build_mlp, device, np2torch, get_logger

from a2c import Actor, Critic

class ReplayBuffer:
  def __init__(self, seed, max_buffer_len=None):
    self.max_buffer_len = max_buffer_len
    self.buffer = deque([])
    random.seed(seed)
  def update_buffer(self, record):
    #record: dict of "state", "action", "reward", "next_state", "done"
    self.buffer.append(record)
    if self.max_buffer_len and len(self.buffer) > self.max_buffer_len:
      self.buffer.popleft()
  def sample(self, batch_size):
    sampled_experience = random.sample(list(self.buffer), batch_size)
    states = np.vstack([record["state"] for record in sampled_experience])
    actions = np.vstack([record["action"] for record in sampled_experience])
    rewards = np.vstack([record["reward"] for record in sampled_experience])
    next_states = np.vstack([record["next_state"] for record in sampled_experience])
    log_probs = np.vstack([record["log_prob"] for record in sampled_experience])
    return states, actions, rewards, next_states, log_probs

class QNetwork(nn.Module):
  def __init__(self, observation_dim, action_dim, config):
    super().__init__()
    self.config = config
    self.network = build_mlp(observation_dim + action_dim,
                             1,
                             self.config.layer_size,
                             self.config.n_layers).to(device)
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.lr)
  def update_q_network(self, observations, actions, q_target):
    #q_target: Pytorch tensor
    inputs = self(observations,actions)
    loss = torch.nn.functional.mse_loss(inputs, q_target.detach())
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
  def forward(self, observation, action):
    
    return torch.squeeze(self.network(np2torch(np.concatenate([observation, action], axis=1))))

class SoftActor(Actor, nn.Module):
  def __init__(self, env, config):
    super().__init__(env=env, config=config)

  def update_actor(self, states, actions, q_values):
    states = np2torch(states)
    actions = np2torch(actions)
    log_probs = self.policy.action_distribution(states).log_prob(actions)
    loss = torch.mean(torch.pow(log_probs - q_values, 2))
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    

class SoftCritic(Critic, nn.Module):
  def __init__(self,observation_dim, config):
    super().__init__(observation_dim=observation_dim, config=config)
    self.tau = self.config.tau
  def update_critic(self, states, q_values, log_probs):
    log_probs = np2torch(log_probs)
    values = self(states)
    loss = torch.sum(torch.pow(values - q_values.detach() + log_probs, 2)).mean()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
  def hard_update(self, critic):
    with torch.no_grad():
      for param1, param2 in zip(self.parameters(), critic.parameters()):
        param1.copy_(self.tau * param2 + (1.0 - self.tau) * param1)

class SAC(nn.Module):
  def __init__(self, env, config, seed):
    super().__init__()
    self.env = env
    self.config = config
    self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
    self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
    self.observation_dim = self.env.observation_space.shape[0]
    self.q_network = QNetwork(self.observation_dim, self.action_dim, self.config)
    #self.q_target = QNetwork(self.observation_dim, self.action_dim, self.config)
    #self.q_network.load_state_dict(self.q_network.state_dict())
    self.logger = get_logger(self.config.log_path)
    self.env.reset(seed=seed)
    self.actor = SoftActor(self.env, self.config)
    self.critic = SoftCritic(self.observation_dim, self.config)
    self.critic_bar = SoftCritic(self.observation_dim, self.config)
    self.critic_bar.load_state_dict(self.critic.state_dict())
    self.buffer = ReplayBuffer(seed)
  def compute_q_target(self, next_states, rewards):
    #returns: Pytorch tensor
    rewards = torch.squeeze(np2torch(rewards))
    masked_next_values = self.critic_bar(next_states)
    return rewards + masked_next_values

  def train(self):
    all_episode_return = []
    for i in range(self.config.num_iter):
      #environmental step
      states, actions, rewards, next_states, log_probs = [], [], [], [], []
      state, _ = self.env.reset()
      for step in range(self.config.batch_size):
        states.append(state)
        action, log_prob = self.actor(state)
        log_probs.append(log_prob)
        actions.append(action)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_states.append(next_state)
        rewards.append(reward)
        record = {
          "state": state,
          "action": action,
          "reward": reward,
          "next_state": next_state,
          "log_prob": log_prob
        }
        self.buffer.update_buffer(record)
        state = next_state
        if done:
          state, _ = self.env.reset()
      

      #gradient loop
      for _ in range(self.config.update_gradient_freq):
        states, actions, rewards, next_states, log_probs = self.buffer.sample(self.config.batch_size)
        q_values = self.q_network(states, actions)
        self.critic.update_critic(states, q_values, log_probs)
        self.q_network.update_q_network(states, 
                                        actions, 
                                        self.compute_q_target(next_states,rewards))
        self.actor.update_actor(states, actions, q_values)
        self.critic_bar.hard_update(self.critic)
      
      
      #testing
      state, _ = self.env.reset()
      done = False
      episodic_reward = 0
      while not done:
        action, _ = self.actor(state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        episodic_reward = reward + self.config.gamma * episodic_reward
        state = next_state
      all_episode_return.append(episodic_reward)
      msg = "[EPISODE {}]: Epsodic reward: {:04.2f}".format(i,episodic_reward)
      self.logger.info(msg)
        
from policy_utils.config import Config
env = gym.make("InvertedPendulum-v4")
config = Config("InvertedPendulum-v4", seed=1, batch_size=2000)
sac = SAC(env, config, seed=1)
sac.train()


