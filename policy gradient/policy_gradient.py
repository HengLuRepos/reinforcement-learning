import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from policy import CategoricalPolicy, GuassianPolicy
from utils import device, np2torch, build_mlp, get_logger
from baseline import Baseline
class PolicyGradient:
    def __init__(self, env, config, seed, logger=None):
        self.env = env
        self.seed = seed
        self.config = config
        self.logger = logger
        if logger is None:
          self.logger = get_logger(config.log_path)
        self.env.reset(seed=self.seed)
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        self.lr = self.config.lr
        self.init_policy()
        if self.config.use_baseline:
            self.baseline = Baseline(env,config)

    def init_policy(self):
        self.network = build_mlp(self.obs_dim, self.action_dim, self.config.layer_size, self.config.n_layers)
        if self.discrete:
            self.policy = CategoricalPolicy(self.network)
        else:
            self.policy = GuassianPolicy(self.network,self.action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
    def sample_path(self, env, episodes=None):
        t = 0
        episode = 0
        paths = []
        episode_rewards = []
        while episodes or t < self.config.batch_size:
            state, _ = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0
            for step in range(self.config.max_ep_len):
                states.append(state)
                action = self.policy.action(states[-1][None])[0][0]
                state, reward, terminated, truncated, _ = self.env.step(action)
                actions.append(action)
                rewards.append(reward)
                done = terminated or truncated
                episode_reward += reward
                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not episodes) and t == self.config.batch_size:
                    break
            path = {
                "observation": np.array(states),
                "action": np.array(actions),
                "reward": np.array(rewards),
            }
            paths.append(path)
            episode += 1
            
            if episodes and episode >= episodes:
                break
        return paths, episode_rewards
    def get_all_returns(self,paths):
        #get Gt at all timestamps for all path
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = np.zeros(rewards.shape)
            returns[len(rewards) - 1] = rewards[len(rewards) - 1]
            for index, reward in reversed(list(enumerate(rewards))):
                if index < len(rewards) - 1:
                    returns[index] = rewards[index] + self.config.gamma*returns[index+1]
            all_returns.append(returns)
        return np.concatenate(all_returns)
    def normalize_advantage(self, advantage):
        return (advantage - np.mean(advantage)) / np.std(advantage)
    def calculate_advantage(self, returns, obs):
        if self.config.use_baseline:
            advantage = self.baseline.calculate_advantage(returns,obs)
        else:
            advantage = returns
        if self.config.normalize_advantage:
            advantage = self.normalize_advantage(advantage)
        return advantage
    def update_policy(self, obs, actions, advantages):
        obs = np2torch(obs)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        log_prob = self.policy.action_distribution(obs).log_prob(actions).to(device)
        loss = -torch.sum(torch.mul(log_prob,advantages)).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def train(self):
      averaged_total_rewards = []
      for t in range(self.config.num_batches):
        paths, total_rewards = self.sample_path(self.env)
        obs = np.concatenate([path["observation"] for path in paths])
        actions = np.concatenate([path["action"] for path in paths])
        rewards = np.concatenate([path["reward"] for path in paths])
        returns = self.get_all_returns(paths)
        
        advantages = self.calculate_advantage(returns,obs)
        if self.config.use_baseline:
          self.baseline.update_baseline(returns, obs)
        self.update_policy(obs,actions,advantages)
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.sqrt(np.var(total_rewards)/len(total_rewards))
        msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(t,avg_reward,std_reward)
        averaged_total_rewards.append(avg_reward)
        self.logger.info(msg)
        
        
      self.logger.info("- Training done.")
      np.save(self.config.scores_output, averaged_total_rewards)
    def run(self):
      self.train()
        
        
        