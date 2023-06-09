import torch
import torch.nn as nn
from utils import build_mlp, device, np2torch

class Baseline(nn.Module):
  def __init__(self, env, config):
    super().__init__()
    self.env = env
    self.config = config
    self.lr = self.config.lr
    observation_dim = self.env.observation_space.shape[0]
    self.network = build_mlp(observation_dim, 1, self.config.n_layers, self.config.layer_size).to(device)
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
  def forward(self, obs):
    output = self.network(obs.to(device)).squeeze()
    assert output.ndim == 1
    return output
  def calculate_advantage(self,returns,obs):
    obs = np2torch(obs)
    output = self(obs).cpu().detach().numpy()
    advantages = returns - output
    return advantages
  def update_baseline(self, returns, obs):
    returns = np2torch(returns)
    obs = np2torch(obs)
    loss = nn.functional.mse_loss(returns, self(obs))
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()