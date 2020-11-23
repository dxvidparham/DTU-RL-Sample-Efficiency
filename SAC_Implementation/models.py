import torch
import torch.nn as nn
import torch.nn.functional as F

"""
SAC uses three different networks:
a state value function V parameterized by ψ,
a soft Q-function Q parameterized by θ,
and a policy function π parameterized by ϕ
"""

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class SoftQNetwork(nn.Module):

    def __init__(self, input_dim, action_dim, hidden_dim=256, output_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module):
    #! unfinished
    """Policy network"""

    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        super(PolicyNetwork, self).__init__()

    def forward(self, x):
        mean = None
        log_std = None
        return mean, log_std

    def sample(self, state):
        return action, log_prob, mean
