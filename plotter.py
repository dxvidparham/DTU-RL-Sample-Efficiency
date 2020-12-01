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
        self.q_losses = []

    def add_to_lists(self, reward, length, policy_loss, q_loss):
        self.rewards.append(reward)
        self.lengths.append(length)
        self.policy_losses.append(policy_loss)
        self.q_losses.append(q_loss)

    def plot(self):

        _eps = min(self.num_episodes, len(self.rewards))

        plt.figure(figsize=(16, 9))
        plt.subplot(411)
        plt.title('training rewards')
        plt.plot(range(1, _eps + 1), self.rewards, label="Model")
        plt.plot(moving_average(self.rewards), label="Moving Average")
        plt.legend()
        plt.xlim([0, _eps])
        plt.subplot(412)
        plt.title('training lengths')
        plt.plot(range(1, _eps + 1), self.lengths)
        plt.plot(range(1, _eps + 1), moving_average(self.lengths))
        plt.xlim([0, _eps])
        plt.subplot(413)
        plt.title('policy loss')
        plt.plot(range(1, _eps + 1), self.policy_losses)
        plt.plot(range(1, _eps + 1), moving_average(self.policy_losses))
        plt.xlim([0, _eps])
        plt.subplot(414)
        plt.title('q loss')
        plt.plot(range(1, _eps + 1), self.q_losses)
        plt.plot(range(1, _eps + 1), moving_average(self.q_losses))
        plt.xlim([0, _eps])

        plt.tight_layout()

        now = datetime.now().strftime("%d_%m_%Y_-%H_%M_%S")
        filename = f"plot_{now}.png"
        plt.savefig('figures/' + filename)



