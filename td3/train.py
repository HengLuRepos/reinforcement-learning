from config import Config
from td3 import ReplayBuffer, TD3, np2torch
import torch 
import gymnasium as gym
import numpy as np

env = gym.make("InvertedPendulum-v4")
A_MIN = env.action_space.low
A_MAX = env.action_space.high
config = Config(action_max=A_MAX, action_min = A_MIN)
buffer = ReplayBuffer(config)
td3 = TD3(4, 1, config)

action_size = env.action_space.shape[0]
all_ep_rewards = []
for ep in range(2000):
    state, info = env.reset()
    for i in range(config.start_steps):
        state = np2torch(state)
        mu = td3.mu_network(state).detach().squeeze().cpu().numpy() * A_MAX
        action = np.clip(mu + np.random.randn(action_size), a_min=A_MIN, a_max=A_MAX)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.remember(state, action, reward, next_state, done)
        if done:
            state, info = env.reset()
        else:
            state = next_state
    for i in range(20):
        states, actions, rewards, next_states, dones = buffer.sample()
        td3.update_q(states, actions, rewards, next_states, dones)
        if i % config.policy_delay == 0:
            td3.update_mu(states)
            td3.update_target_networks()
    
    state, info = env.reset()
    done = False
    ep_rewards = 0
    while not done:
        state = np2torch(state)
        action = td3.mu_network(state).detach().squeeze().cpu().numpy() * A_MAX
        next_state, reward, terminated, truncated, info = env.step(action)
        ep_rewards += reward
        done = terminated or truncated
        state = next_state
    all_ep_rewards.append(ep_rewards)
    if ep_rewards >= max(all_ep_rewards):
        td3.save_model()
    print("Iteration {}: Episodic reward: {}".format(ep, ep_rewards))

td3.save_model()
print("max_ep_reward: {}".format(max(all_ep_rewards)))
    