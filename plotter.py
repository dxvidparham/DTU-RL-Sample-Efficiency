import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import mplcyberpunk

import LogHelper


def moving_average(a, n=10):
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
        self.a_losses = []
        self.total_steps = []
        self.time = []

    def add_to_lists(self, reward, length, policy_loss, q_loss, a_loss,total_steps, episode, time, log="INFO"):
        self.rewards.append(reward)
        self.lengths.append(length)
        self.policy_losses.append(policy_loss)
        self.q_losses.append(q_loss)
        self.a_losses.append(a_loss)
        self.total_steps.append(total_steps)
        self.time.append(time)

        if log is not None:
            LogHelper.log_episode(_episode=episode,
                                  step=total_steps,
                                  reward=reward,
                                  p_loss=policy_loss,
                                  q_loss=q_loss,
                                  a_loss=a_loss,
                                  time=time,
                                  level=log)

    def get_lists(self):
        return self.rewards, self.lengths, self.policy_losses, self.q_losses, self.total_steps, self.time, self.a_losses

    def plot(self):
        # #plt.style.use('fivethirtyeight')
        #plt.style.use('ggplot')
        plt.style.use('cyberpunk')
        # #plt.style.use('default')
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        ax[0][0].set_title('Rewards per Step')
        ax[0][0].plot(self.total_steps, self.rewards)
        ax[0][1].set_title('Policy loss per Step')
        logging.error(self.policy_losses)
        ax[0][1].plot(self.total_steps, self.policy_losses)
        ax[1][0].set_title('Q-Losses per Step')
        ax[1][0].plot(self.total_steps, self.q_losses)
        ax[1][1].set_title('Time per Step')
        ax[1][1].plot(self.total_steps, self.time)


        plt.tight_layout()
        now = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        filename = f"plot_{now}.png"
        plt.savefig('figures/' + filename)

        # #plt.style.use('fivethirtyeight')
        # plt.style.use('ggplot')
        # #plt.style.use('default')
        # plt.figure(figsize=(16, 9))
        # #plt.ylabel('Success Rate', fontsize=25)
        # #plt.xlabel('Iteration Number', fontsize=25, labelpad=-4)
        # plt.subplot(411)
        # plt.title('training rewards')
        # plt.plot(range(1, _eps + 1), self.rewards, label="Model")
        # plt.plot(moving_average(self.rewards), label="Moving Average")
        # plt.legend()
        # plt.xlim([0, _eps])
        # plt.subplot(412)
        # plt.title('training lengths')
        # plt.plot(range(1, _eps + 1), self.lengths)
        # plt.plot(range(1, _eps + 1), moving_average(self.lengths))
        # plt.xlim([0, _eps])
        # plt.subplot(413)
        # plt.title('policy loss')
        # plt.plot(range(1, _eps + 1), self.policy_losses)
        # plt.plot(range(1, _eps + 1), moving_average(self.policy_losses))
        # plt.xlim([0, _eps])
        # plt.subplot(414)
        # plt.title('q loss')
        # plt.plot(range(1, _eps + 1), self.q_losses)
        # plt.plot(range(1, _eps + 1), moving_average(self.q_losses))
        # plt.xlim([0, _eps])
        #
        # plt.tight_layout()
        # now = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        # filename = f"plot_{now}.png"
        # plt.savefig('figures/' + filename)
