import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


class Plotter:
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.rewards = []
        self.lengths = []
        self.policy_losses = []
        self.q1_losses = []
        self.q2_losses = []
        self.epsilons = []

    def add_to_lists(self, reward, length, policy_loss, q1_loss, q2_loss, epsilon):
        self.rewards.append(reward)
        self.lengths.append(length)
        self.policy_losses.append(policy_loss)
        self.q1_losses.append(q1_loss)
        self.q2_losses.append(q2_loss)
        self.epsilons.append(epsilon)

    def plot(self):

        _eps = min( self.num_episodes , len(self.rewards))

        plt.figure(figsize=(16, 9))
        plt.subplot(611)
        plt.title('training rewards')
        plt.plot(range(1, _eps + 1), self.rewards)
        plt.plot(moving_average(self.rewards))
        plt.xlim([0, _eps])
        plt.subplot(612)
        plt.title('training lengths')
        plt.plot(range(1, _eps + 1), self.lengths)
        plt.plot(range(1, _eps + 1), moving_average(self.lengths))
        plt.xlim([0, _eps])
        plt.subplot(613)
        plt.title('policy loss')
        plt.plot(range(1, _eps + 1), self.policy_losses)
        plt.plot(range(1, _eps + 1), moving_average(self.policy_losses))
        plt.xlim([0, _eps])
        plt.subplot(614)
        plt.title('q1 loss')
        plt.plot(range(1, _eps + 1), self.q1_losses)
        plt.plot(range(1, _eps + 1), moving_average(self.q1_losses))
        plt.xlim([0, _eps])
        plt.subplot(615)
        plt.title('q2 loss')
        plt.plot(range(1, _eps + 1), self.q2_losses)
        plt.plot(range(1, _eps + 1), moving_average(self.q2_losses))
        plt.xlim([0, _eps])
        plt.subplot(616)
        plt.title('epsilon')
        plt.plot(range(1, _eps + 1), self.epsilons)
        plt.xlim([0, _eps])
        plt.tight_layout()

        now = datetime.now().strftime("%d%m%Y%H%M%S")
        filename = f"plot_{now}.png"
        plt.savefig('figures/' + filename)



