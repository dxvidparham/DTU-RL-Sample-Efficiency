import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

"""
SAC uses three different networks:
a state value function V parameterized by ψ,
a soft Q-function Q parameterized by θ,
and a policy function π parameterized by ϕ
"""

# ACTOR
class ValueNetwork(nn.Module):
    """
    The ValueNetwork provides feedback to our PolicyNetwork if certain states are valueable or not.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        lr_value,
        output_dim=1,
        init_w=3e-3,
    ):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_value)

    def forward(self, state):
        state_value = F.relu(self.linear1(state))
        state_value = F.relu(self.linear2(state_value))
        state_value_output = self.linear3(state_value)

        return state_value_output


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
        output_dim=1,
        init_w=3e-3,
    ):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)

    def forward(self, state, action):
        action_value = torch.cat([state, action], 1)
        action_value = F.relu(self.linear1(action_value))
        action_value = F.relu(self.linear2(action_value))
        action_value_output = self.linear3(action_value)
        return action_value_output


# POLICY
class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim,
        lr_policy,
        init_w=3e-3,
        log_std_min=-20,
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
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_policy)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi
