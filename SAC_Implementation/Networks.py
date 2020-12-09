import copy
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from .ranger import Ranger  # this is from ranger.py


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# CRITIC
class SoftQNetwork(nn.Module):
    """
    The SoftQNetwork is responsible for evaluating actions taken by the PolicyNet
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim,
            lr_critic,
            gpu_device,
            output_dim=1,
            init_w=3e-3,
    ):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)
        self.apply(weight_init)

        #self.optimizer = Ranger(self.parameters(), lr=lr_critic)
        #self.optimizer = optim.RMSprop(self.parameters(), lr=lr_critic)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)

        self.device = torch.device(f'cuda:{gpu_device}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = torch.cat([state.to(device=self.device), action.to(device=self.device)], 1)
        action_value = F.relu(self.linear1(action_value))
        action_value = F.relu(self.linear2(action_value))
        action_value_output = self.linear3(action_value)
        return action_value_output

    def update_params(self, new_params, tau):
        params = self.parameters()

        for param, target_param in zip(new_params, params):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

        # for k in params.keys():
        #     params[k] = params[k] * (1 - tau) + new_params[k] * tau
        # self.load_state_dict(params)


# POLICY
class PolicyNetwork(nn.Module):
    def __init__(
            self,
            input_dim,
            action_dim,
            hidden_dim,
            lr_policy,
            gpu_device,
            init_w=3e-3,
            log_std_min=-10,
            log_std_max=2,
    ):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)
        # self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        # self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        #self.optimizer = Ranger(self.parameters(), lr=lr_policy)
        #self.optimizer = optim.RMSprop(self.parameters(), lr=lr_policy)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_policy)

        self.device = torch.device(f'cuda:{gpu_device}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)

        # Squash it in -1 1
        # mean = torch.tanh(mean)

        # Squash log std
        log_std = torch.tanh(self.log_std_linear(x))
        log_std = self.log_std_min + 0.5*(self.log_std_max - self.log_std_min) * (log_std+1)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state.to(device=self.device))
        std = log_std.exp()

        noise = torch.randn_like(mean)
        pi = mean + noise * std

        residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
        log_pi = residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

        mean = torch.tanh(mean)
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
        pi = torch.tanh(pi)
        return mean, pi, log_pi
