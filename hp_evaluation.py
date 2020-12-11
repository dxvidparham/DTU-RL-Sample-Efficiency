"""
This files is to evaluate the hyperparameter testing. We saved the iterations into *.model files in the results folder.
They include the model, as well as the performance of each of them.

"""
import logging
from datetime import datetime
import numpy as np
from hyperopt import Trials

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level='INFO',
                    handlers=[logging.StreamHandler()]
                    )
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import pickle

from matplotlib import pyplot as plt

##

dir = "hp_trials"
filename = "hp_trials2"
ending = "trials"

parameter = ["hidden_dim", 'init_alpha']

if dir == "results":
    logging.warning("THIS IS DEPRECATED. PLEASE USE THE NEWER TRIALS FILES")

with open(f"{dir}/{filename}.{ending}", "rb") as f:
    evaluation = pickle.load(f)

evaluation = evaluation.results if dir == "hp_trials" else evaluation

#  dict_keys(['loss', 'status', 'model', 'max_reward', 'q_losses', 'policy_losses', 'rewards'])
print(len(evaluation))

figsizes = (15, 7)


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def plot(what: str, label: str = ''):
    plt.figure(figsize=figsizes)
    for _round in evaluation:
        paras = map(lambda para: f"{para}={_round['params'][para]:.4f}", parameter)
        label = ", ".join(paras)

        p = plt.plot(_round['total_steps'], moving_average(_round[what]), label=label)
        color = p[0].get_color()
        plt.plot(_round['total_steps'], _round[what],
                 c=color, alpha=0.2)

    label = what if what else label

    plt.title(
        f"{evaluation[0]['params']['env_domain'].capitalize()}:{evaluation[0]['params']['env_task'].capitalize()} - Optimization of {', '.join(parameter)}")
    plt.xlabel("Update steps")
    plt.ylabel(f"Average {label}")
    plt.legend()
    plt.tight_layout()
    date_now = datetime.now().strftime("%Y_%M_%d_%H_%M_%S")
    filename = f"{'_'.join(parameter)}_{what}_{date_now}"
    plt.savefig(f'hp_figures/{filename}.pdf')


plot('rewards')
plot('time')
plot('q_losses')
plot('policy_losses')

if __name__ == '__main__':
    pass
