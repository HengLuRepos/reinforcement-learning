import random
import sys
sys.path.append('../')

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from policy_utils.utils import build_mlp, device, np2torch, get_logger
from  collections import deque
import random

# may only work for continuous actions

class ReparameterizedGuassianPolicy(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.alpha = self.config.alpha
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.mu_network = build_mlp(self.observation_dim,
                                    self.action_dim,
                                    self.config.layer_size,
                                    self.config.n_layers).to(device)
        self.log_network = build_mlp(self.observation_dim,
                                     self.action_dim,
                                     self.config.layer_size,
                                     self.config.n_layers).to(device)
        self.action_scale = torch.FloatTensor((self.env.action_space.high -
                                               self.env.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((self.env.action_space.high +
                                              self.env.action_space.low) / 2.)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.pi_lr)

    def step(self, state):
        state = np2torch(state)
        mean = self.mu_network(state).to(device)
        log_std = self.log_network(state).to(device)
        std = torch.exp(log_std)
        dist = torch.distributions.MultivariateNormal(loc=mean, scale_tril=torch.diag_embed(std))
        action = dist.sample()
        return action.detach().cpu().numpy()

    def sample(self, obs):
        obs = np2torch(obs)
        mean = self.mu_network(obs).to(device)
        log_std = self.log_network(obs).to(device)
        std = torch.exp(log_std).to(device)
        dist = torch.distributions.MultivariateNormal(loc=mean, scale_tril=torch.diag_embed(std))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

    def rsample(self, obs):
        obs = np2torch(obs)
        mean = self.mu_network(obs).to(device)
        log_std = torch.clamp(self.log_network(obs).to(device), -20, 2)
        std = torch.exp(log_std).to(device)
        pi_dist = torch.distributions.Normal(mean,std)
        pi_act = pi_dist.rsample()
        logp_pi = pi_dist.log_prob(pi_act).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_act - torch.nn.functional.softplus(-2*pi_act))).sum(axis=1)
        return pi_act.to(device), logp_pi

    def update_actor(self, q_new, log_probs):
        loss = torch.mean(log_probs - q_new)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SoftQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, config):
        super().__init__()
        self.config = config
        self.network = build_mlp(observation_dim + action_dim,
                                 1,
                                 self.config.layer_size,
                                 self.config.n_layers).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.q_lr)

    def update_q_network(self, observations, actions, q_targets):
        inputs = self(observations, actions)
        loss = torch.nn.functional.mse_loss(inputs, q_targets.float().detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, obs, actions):
        # work for batched version.
        x = None
        if isinstance(actions, np.ndarray):
            x = np.concatenate([obs, actions], axis=1)
        else:
            obs = np2torch(obs)
            x = torch.cat((obs, actions), 1)
        out = self.network(np2torch(x))
        out = out.view(out.size(0), -1)
        return out
    
    def copy(self, q_network):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), q_network.parameters()):
                param1.copy_(param2)


class ReplayBuffer:
    def __init__(self, seed, config):
        self.buffer = deque([])
        self.config = config
        self.max_buffer_size = self.config.max_buffer_size
        self.sample_size = self.config.buffer_batch_size
        random.seed(seed)

    def update_buffer(self, experience):
        """
        :param experience:
            Python dict, {"state": np.ndarray,
                            "action": np.ndarray,
                            "reward": float,
                            "next_state": np.ndarray,
                            "done": bool}
        :return: None
        """
        self.buffer.append(experience)
        if self.max_buffer_size and len(self.buffer) > self.max_buffer_size:
            self.buffer.popleft()

    def sample(self):
        sampled_experience = random.sample(list(self.buffer), self.sample_size)
        states = np.vstack([record["state"] for record in sampled_experience])
        next_states = np.vstack([record["next_state"] for record in sampled_experience])
        actions = np.vstack([record["action"] for record in sampled_experience])
        rewards = np.array([record["reward"] for record in sampled_experience])
        done = np.array([record["done"] for record in sampled_experience])
        return states, actions, rewards, next_states, done


class SoftCritic(nn.Module):
    def __init__(self, observation_dim, config):
        super().__init__()
        self.config = config
        self.observation_dim = observation_dim
        self.network = build_mlp(self.observation_dim,
                                 1,
                                 self.config.layer_size,
                                 self.config.n_layers).to(device)
        self.alpha = self.config.alpha
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.v_lr)
        self.tau = self.config.tau

    def update_critic(self, states, q1, q2, log_probs):
        log_probs = np2torch(log_probs).unsqueeze(1)
        q_min = torch.min(q1, q2)
        inputs = self(states)
        loss = torch.nn.functional.mse_loss(inputs, q_min - log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def hard_update(self, critic):
        # update target network value
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), critic.parameters()):
                param1.copy_(self.tau * param2 + (1.0 - self.tau) * param1)

    def forward(self, states):
        states = np2torch(states)
        out = self.network(states)
        out = out.view(out.size(0), -1)
        return out
    
    def copy(self, q_network):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), q_network.parameters()):
                param1.copy_(param2)


class SAC:
    def __init__(self, env, config, seed):
        super().__init__()
        self.env = env
        self.config = config
        self.seed = seed
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.actor = ReparameterizedGuassianPolicy(self.env, self.config)
        self.critic = SoftCritic(self.observation_dim, self.config)
        self.critic_target = SoftCritic(self.observation_dim, self.config)
        self.critic_target.copy(self.critic)
        self.q1 = SoftQNetwork(self.observation_dim, self.action_dim, self.config)
        self.q2 = SoftQNetwork(self.observation_dim, self.action_dim, self.config)
        self.q2.copy(self.q1)
        self.logger = get_logger(self.config.log_path)
        self.buffer = ReplayBuffer(self.seed, self.config)
        self.env.reset(seed=self.seed)
        self.gamma = self.config.gamma

    def train(self):
        for p in self.critic_target.parameters():
            p.requires_grad = False
        all_episodic_rewards = []
        for i in range(self.config.num_iter):
            # exploration loop
            state, _ = self.env.reset()
            for step in range(self.config.explore_step):
                action = self.actor.step(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                experience = {
                    "state": np.array(state),
                    "action": np.array(action),
                    "reward": reward,
                    "next_state": np.array(next_state),
                    "done": done
                }
                self.buffer.update_buffer(experience)
                state = next_state
                if done:
                    state, _ = self.env.reset()

            # gradient loop
            for _ in range(self.config.update_gradient_freq):
                states, actions, rewards, next_states, done = self.buffer.sample()
                rewards = torch.tensor(rewards).unsqueeze(1).to(device)
                done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)
                q_targets = rewards + \
                            self.gamma * (1.0 - done) * self.critic_target(next_states)
                actions_sampled, log_probs_sampled = self.actor.sample(states)
                q1 = self.q1(states, actions_sampled).to(device)
                q2 = self.q2(states, actions_sampled).to(device)

                self.q1.update_q_network(states, actions, q_targets)
                self.q2.update_q_network(states, actions, q_targets)
                self.critic.update_critic(states, q1, q2, log_probs_sampled)

                r_actions, r_log_probs = self.actor.rsample(states)
                q1_new = self.q1(states, r_actions)
                self.actor.update_actor(q1_new, r_log_probs)

                self.critic_target.hard_update(self.critic)

            # evaluation loop
            state, _ = self.env.reset()
            episodic_reward = 0
            done = False
            while not done:
                action = self.actor.step(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episodic_reward = reward + self.gamma * episodic_reward
                state = next_state
                done = terminated or truncated
            msg = "[EPISODE {}]: Episodic reward: {:04.2f}".format(i, episodic_reward)
            all_episodic_rewards.append(episodic_reward)
            self.logger.info(msg)
        np.save(self.config.scores_output, all_episodic_rewards)
        self.logger.info("max episodic reward: {:04.2f}".format(max(all_episodic_rewards)))