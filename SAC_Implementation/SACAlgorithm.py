import logging
import math
from copy import deepcopy

import LogHelper
import torch

from SAC_Implementation.Networks import *
from SAC_Implementation.ReplayBuffer import ReplayBuffer


def initialize_nets_and_buffer(state_dim: int,
                               action_dim: int,
                               q_hidden: int,
                               policy_hidden: int,
                               learning_rates: dict,
                               replay_buffer_size: int,
                               gpu_device: int
                               ) -> (
        SoftQNetwork, SoftQNetwork, SoftQNetwork, SoftQNetwork, PolicyNetwork, ReplayBuffer):
    """
    Method to initialize the neural networks as well as the replay buffer
    :param state_dim: Dimension of the state space
    :param action_dim: Dimension of the action space
    :param q_hidden: Hidden Size of the Q networks
    :param policy_hidden: Hidden Size of the Policy Network
    :param learning_rates: Learning Rates in an dict with keys "critic"(q-networks) and "actor"(policy)
    :param replay_buffer_size: Size of the replayBuffer
    :return: Returns the networks (Soft1, soft2, target1,target2, Policy, Buffer)
    """
    # We need to networks: 1 for the value function first
    soft_q1 = SoftQNetwork(state_dim, action_dim, q_hidden, learning_rates.get('critic'), gpu_device)
    soft_q2 = SoftQNetwork(state_dim, action_dim, q_hidden, learning_rates.get('critic'), gpu_device)

    # Then another one for calculating the targets
    soft_q1_targets = deepcopy(soft_q1)
    soft_q2_targets = deepcopy(soft_q1)

    policy = PolicyNetwork(state_dim, action_dim, policy_hidden, learning_rates.get('actor'), gpu_device)

    # Initialize the Replay Buffer
    buffer = ReplayBuffer(state_dim, action_dim,
                          replay_buffer_size)

    return soft_q1, soft_q2, soft_q1_targets, soft_q2_targets, policy, buffer


class SACAlgorithm:
    def __init__(self, env, param: dict):
        """

        :param env:
        :param param: dict which needs following parameter:
            [hidden_dim, lr_critic, lr_policy, alpha, tau, gamma, sample_batch_size]
        """
        self.param = param
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.device = torch.device(f'cuda:{param.get("gpu_device")}' if torch.cuda.is_available() else 'cpu')

        self.soft_q1, self.soft_q2, self.soft_q1_targets, self.soft_q2_targets, self.policy, self.buffer = initialize_nets_and_buffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            q_hidden=param.get('hidden_dim'),
            policy_hidden=param.get('hidden_dim'),
            learning_rates={
                'critic': param.get('lr_critic'),
                'actor': param.get('lr_actor')
            },
            replay_buffer_size=param.get('replay_buffer_size'),
            gpu_device=param.get('gpu_device')
        )
        self.alpha_decay_activated = param.get('alpha_decay_activated')
        if self.alpha_decay_activated:
            self.log_alpha = torch.tensor(np.log(param.get('init_alpha'))).to(self.device)
            self.log_alpha.requires_grad = True
            # set target entropy to -|A|
            self.target_entropy = -np.prod(self.action_dim)
            self.log_alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=param.get('alpha_lr'), betas=(param.get('alpha_beta'), 0.999)
            )
        else:
            self.alpha = param.get('alpha')

        self.sample_batch_size,  self.tau, self.gamma = (param.get('sample_batch_size'),
                                                                    param.get('tau'),
                                                                    param.get('gamma'))



    def _update_critic(self, state, action, y_hat):
        q1_forward = self.soft_q1(state.float(), action.float())
        q2_forward = self.soft_q2(state.float(), action.float())

        # Q1 Network
        q_loss = F.mse_loss(q1_forward.float(), y_hat.float().to(device=self.device)) + \
                 F.mse_loss(q2_forward.float(), y_hat.float().to(device=self.device))

        self.soft_q1.optimizer.zero_grad()
        self.soft_q2.optimizer.zero_grad()
        q_loss.backward()
        self.soft_q1.optimizer.step()
        self.soft_q2.optimizer.step()
        return q_loss

    def _calculate_target(self, state, action):
        with torch.no_grad():
            y_hat_q1 = self.soft_q1_targets(state.float(), action.float())
            y_hat_q2 = self.soft_q2_targets(state.float(), action.float())
            min_ = torch.min(y_hat_q1, y_hat_q2)
        return min_

    def _update_policy_alpha(self, state):
        action_new, _, log_pi = self.policy.sample(torch.Tensor(state))
        q1_forward = self.soft_q1(state.float(), action_new.float())
        q2_forward = self.soft_q2(state.float(), action_new.float())
        q_forward = torch.min(q1_forward, q2_forward)

        # Changed to an F.mse_loss from simple mean
        # policy_loss = F.mse_loss((self.alpha * action_entropy_new), q_forward)
        if self.alpha_decay_activated:
            entropy = -self.log_alpha.exp().detach() * log_pi
        else:
            entropy = -self.alpha * log_pi

        policy_loss = -(q_forward + entropy).mean()

        self.policy.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

        if self.alpha_decay_activated:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.log_alpha *
                          (-log_pi - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = 0

        return policy_loss.item(), alpha_loss

    def update(self, step):

        # Sample from Replay buffer
        # logging.warning("STEEEEEP 11")
        state, action, reward, new_state, done, _ = self.buffer.sample(batch_size=self.sample_batch_size)
        policy_loss, q_loss, alpha_loss = 0, 0, 0

        # Computation of targets
        # Here we are using 2 different Q Networks and afterwards choose the lower reward as regulator.
        if step % 2 == 0:
            # logging.warning("STEEEEEP 12")

            action_sample, _, log_pi = self.policy.sample(torch.Tensor(new_state))

            if self.alpha_decay_activated:
                entropy = -self.log_alpha.exp() * log_pi
            else:
                entropy = -math.exp(self.alpha) * log_pi
            y_hat_q = self._calculate_target(new_state, action_sample)

            # We calculate the estimated reward for the next state
            # DISCOUNT FACTOR
            y_hat = reward + self.gamma * (1 - done) * (y_hat_q.cpu() + entropy.cpu())

            # # UPDATES OF THE CRITIC NETWORK
            # logging.warning("STEEEEEP 13")
            q_loss = self._update_critic(state, action, y_hat)

        # Update Policy Network (ACTOR) and alpha
        if step % 2 == 0:
            policy_loss, alpha_loss = self._update_policy_alpha(state)

        self.soft_q1_targets.update_params(self.soft_q1.parameters(), self.tau)
        self.soft_q2_targets.update_params(self.soft_q2.parameters(), self.tau)

        # for graph
        return policy_loss, q_loss, alpha_loss

    def sample_action(self, state: torch.Tensor):
        action, _, log_pi = self.policy.sample(state)
        return action.detach().cpu().data.numpy(), log_pi
