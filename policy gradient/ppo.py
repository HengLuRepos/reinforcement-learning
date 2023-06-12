import torch
import torch.nn as nn
from policy_gradient import PolicyGradient
import gymnasium as gym
import numpy as np
from utils import np2torch, device

class PPO(PolicyGradient):
  def __init__(self, env, config, seed, logger=None):
    config.use_baseline = True
    super(PPO, self).__init__(env,config,seed,logger)
    self.episilon = self.config.episilon
  def update_policy(self,obs, actions,advantages,old_log):
    obs = np2torch(obs)
    actions = np2torch(actions)
    advantages = np2torch(advantages)
    old_log = np2torch(old_log)
    
    log_prob = self.policy.action_distribution(obs).log_prob(actions).to(device)
    r_theta = torch.exp(log_prob - old_log)
    clip_r_theta = torch.clip(r_theta, 1 - self.episilon , 1+ self.episilon)
    adv = torch.mul(r_theta,advantages)
    clip_adv = torch.mul(clip_r_theta,advantages.detach())
    loss = -torch.sum(torch.min(adv, clip_adv))
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
  def sample_path(self, env, episodes=None):
    episode = 0
    t = 0
    paths = []
    episode_rewards = []
    
    while episodes or t < self.config.batch_size:
      state, _ = env.reset()
      states, actions, old_logprobs, rewards = [], [], [], []
      episode_reward = 0
      
      for step in range(self.config.max_ep_len):
        states.append(state)
        action, old_logprob = self.policy.action(states[-1][None], return_log_probs = True)
        assert old_logprob.shape == (1,)
        action, old_logprob = action[0], old_logprob[0]
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        actions.append(action)
        old_logprobs.append(old_logprob)
        rewards.append(reward)
        episode_reward += reward
        t += 1
        if done or step == self.config.max_ep_len - 1:
          episode_rewards.append(episode_reward)
          break
        if (not episodes) and t == self.config.batch_size:
          break
      path = {
        "observation": np.array(states),
        "reward": np.array(rewards),
        "action": np.array(actions),
        "old_logprobs": np.array(old_logprobs)
      }
      paths.append(path)
      episode += 1
      if episodes and episode >= episodes:
        break
    return paths, episode_rewards
  def train(self):
    averaged_total_rewards = []
    for t in range(self.config.num_batches):
      paths, total_rewards = self.sample_path(self.env)
      obs = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])
      returns = self.get_all_returns(paths)
        
      advantages = self.calculate_advantage(returns,obs)
      for k in range(self.config.update_freq):
        self.baseline.update_baseline(returns, obs)
        self.update_policy(obs, actions, advantages,old_logprobs)
        
      avg_reward = np.mean(total_rewards)
      std_reward = np.sqrt(np.var(total_rewards)/len(total_rewards))
      msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(t,avg_reward,std_reward)
      averaged_total_rewards.append(avg_reward)
      self.logger.info(msg)
      
    self.logger.info("- Training done.")
    np.save(self.config.scores_output, averaged_total_rewards)
        