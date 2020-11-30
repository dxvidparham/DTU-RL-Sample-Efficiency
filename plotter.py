import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class Plotter():
    def __init__(
            self,
            num_episodes,
            rewards,
            lengths,
            losses,
            epsilons
    ):
        self.num_episodes = num_episodes
        self.rewards = rewards
        self.lengths = lengths
        self.losses = losses
        self.epsilons = epsilons

    # plot results
    def moving_average(a, n=10) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret / n

    def plot(self):
        plt.figure(figsize=(16, 9))
        plt.subplot(411)
        plt.title('training rewards')
        plt.plot(range(1, self.num_episodes + 1), self.rewards)
        plt.plot(self.moving_average(self.rewards))
        plt.xlim([0, self.num_episodes])
        plt.subplot(412)
        plt.title('training lengths')
        plt.plot(range(1, self.num_episodes + 1), self.lengths)
        plt.plot(range(1, self.num_episodes + 1), self.moving_average(self.lengths))
        plt.xlim([0, self.num_episodes])
        plt.subplot(413)
        plt.title('training loss')
        plt.plot(range(1, self.num_episodes + 1), self.losses)
        plt.plot(range(1, self.num_episodes + 1), self.moving_average(self.losses))
        plt.xlim([0, self.num_episodes])
        plt.subplot(414)
        plt.title('epsilon')
        plt.plot(range(1, self.num_episodes + 1), self.epsilons)
        plt.xlim([0, self.num_episodes])
        plt.tight_layout()


        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        filename = "plot_{0}.png".format(now)
        plt.savefig('figures/' + filename)



