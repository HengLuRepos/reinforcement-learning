import torch
import torch.nn as nn
import torch.distributions as ptd

from policy_utils.utils import device, np2torch
class Policy:
    def action_distribution(self,obs):
        #given observations (torch tensors), return action distributions
        raise NotImplementedError
    def action(self, obs):
        #given obs, return sampled actions(and corresponding log_probs, entropies)
        obs = np2torch(obs)
        dist = self.action_distribution(obs)
        actions = dist.sample().cpu().numpy()[0]
        log_probs = dist.log_prob(np2torch(actions)).cpu().detach().numpy()
        entropy = dist.entropy().cpu().detach().numpy()
        return actions, log_probs, entropy

class CategoricalPolicy(Policy, nn.Module):
    def __init__(self, policy_network):
        nn.Module.__init__(self)
        #network is used to compute logits of each action
        self.network = policy_network.to(device)
    def action_distribution(self, obs):
        return ptd.Categorical(logits=self.network(obs).to(device))
    def forward(self,obs):
        return self.action(obs)

class GuassianPolicy(Policy, nn.Module):
    #diagnoal Guassian Policy
    def __init__(self, policy_network, action_dim):
        nn.Module.__init__(self)
        self.network = policy_network.to(device)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float)).to(device)
    def std(self):
        return torch.exp(self.log_std).to(device)
    def action_distribution(self, obs):
        mu = self.network(obs).to(device)
        return ptd.MultivariateNormal(loc=mu, scale_tril=torch.diag(self.std()))
    def forward(self,obs):
        return self.action(obs)

        
        
        
        