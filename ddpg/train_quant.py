from config import Config
from ddpg import ReplayBuffer, DDPG, np2torch
import torch 
import gymnasium as gym
import numpy as np

env = gym.make("InvertedPendulum-v4")
A_MIN = env.action_space.low
A_MAX = env.action_space.high
config = Config(action_max=A_MAX)
buffer = ReplayBuffer(config)
ddpg = DDPG(4, 1, config)
ddpg.qconfig = torch.quantization.get_default_qconfig('x86')

ddpg_copy = DDPG(4, 1, config)
ddpg_copy.load_state_dict(ddpg.state_dict())
ddpg_copy.qconfig = torch.quantization.get_default_qconfig('x86')
torch.quantization.prepare(ddpg, inplace=True)
torch.quantization.prepare(ddpg_copy, inplace=True)
action_size = env.action_space.shape[0]
all_ep_rewards = []
for ep in range(2000):
    state, info = env.reset()
    for i in range(config.start_steps):
        state = np2torch(state)
        mu = ddpg.mu_network(state).detach().squeeze().cpu().numpy() * A_MAX
        action = np.clip(mu + np.random.randn(action_size), a_min=A_MIN, a_max=A_MAX)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.remember(state, action, reward, next_state, done)
        if done:
            state, info = env.reset()
        else:
            state = next_state
    for i in range(15):
        states, actions, rewards, next_states, dones = buffer.sample()
        ddpg.update_q(states, actions, rewards, next_states, dones)
        ddpg.update_mu(states)
        ddpg.update_target_networks()
    
    state, info = env.reset()
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
    if ep_rewards >= max(all_ep_rewards):
        ddpg_copy.load_state_dict(ddpg.state_dict())
    print("Iteration {}: Episodic reward: {}".format(ep, ep_rewards))
torch.quantization.convert(ddpg, inplace=True)
torch.quantization.convert(ddpg_copy, inplace=True)
ddpg_copy.save_model(path="./models/ddpg_quant.pt")
print("max_ep_reward: {}".format(max(all_ep_rewards)))
    