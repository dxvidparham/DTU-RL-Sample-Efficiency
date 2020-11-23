import numpy as np
import torch


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = np.empty([capacity, 6])
        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):

        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max=0):
        self.memory[self.idx] = np.array([obs, action, reward, next_obs, not done, not done_no_max])
        self.idx = (self.idx + 1) % self.capacity

        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.memory[idxs][:,0]).float()

        actions = torch.as_tensor(self.memory[idxs][:,1])

        rewards = torch.as_tensor(self.memory[idxs][:,2])

        next_obses = torch.as_tensor(self.memory[idxs][:,3]).float()

        not_dones = torch.as_tensor(self.memory[idxs][:,4])
        not_dones_no_max = torch.as_tensor(self.memory[idxs][:,5])

        return obses, actions, rewards,next_obses, not_dones, not_dones_no_max


