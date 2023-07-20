from config import Config
from ddpg import ReplayBuffer, DDPG, np2torch
import torch 
import gymnasium as gym
import numpy as np
import random
env = gym.make("InvertedPendulum-v4")
A_MIN = env.action_space.low
A_MAX = env.action_space.high
config = Config(action_max=A_MAX)
buffer = ReplayBuffer(config)
ddpg = DDPG(4, 1, config)
ddpg.load_model()
all_ep_rewards = []
np.random.seed(0)
seeds = np.random.randint(low=0, high=200, size=100)
for ep in range(100):
    state, info = env.reset(seed=int(seeds[ep]))
    done = False
    ep_rewards = 0
    while not done:
        state = np2torch(state)
        action = ddpg.mu_network(state).detach().squeeze().cpu().numpy() * A_MAX
        next_state, reward, terminated, truncated, info = env.step(action)
        ep_rewards += reward
        done = terminated or truncated
        state = next_state
    all_ep_rewards.append(ep_rewards)
    print("Iteration {}: Episodic reward: {}".format(ep, ep_rewards))
print("Mean reward: {}".format(np.mean(all_ep_rewards)))