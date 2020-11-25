import numpy as np
import torch
import logging


class ReplayBuffer(object):
    def __init__(self, obs_shape, action_shape, capacity):
        self.capacity = capacity

        self.obs = np.empty((capacity, obs_shape))
        self.next_obs = np.empty((capacity, obs_shape))
        self.action = np.empty((capacity, action_shape))
        self.reward = np.empty((capacity, 1))
        self.not_done = np.empty((capacity, 1))
        self.not_done_no_max = np.empty((capacity, 1))

        self.idx = 0
        self.last_save = 0
        self.full = False

        logging.debug("Initialized Replay Buffer...")

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max=0):
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.not_done[self.idx] = done
        self.not_done_no_max[self.idx] = done_no_max

        # We use rewrite if buffer is full.
        self.idx = (self.idx + 1) % self.capacity

        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obs[idxs]).float()

        actions = torch.as_tensor(self.action[idxs])

        rewards = torch.as_tensor(self.reward[idxs])

        next_obses = torch.as_tensor(self.next_obs[idxs]).float()

        not_dones = torch.as_tensor(self.not_done[idxs])
        not_dones_no_max = torch.as_tensor(self.not_done_no_max[idxs])

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
