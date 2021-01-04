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

# ONLY IMPORTANT IF EXECUTED AS SCRIPT
FILENAME = "alpha_expl_init_alpha_13_12_2020-22_13_05"


##
def load_and_display_model(dir="hp_trials", filename="FILENAME", parameter=['env_domain'], ending="model", save=False, num_steps=-1, figsizes=(5, 5)):
    if dir == "results":
        logging.warning("THIS IS DEPRECATED. PLEASE USE THE NEWER TRIALS FILES")

    with open(f"{dir}/{filename}.{ending}", "rb") as f:
        evaluation = pickle.load(f)

    evaluation = evaluation.results if dir == "hp_trials" else evaluation

    #  dict_keys(['loss', 'status', 'model', 'max_reward', 'q_losses', 'policy_losses', 'rewards'])
    print("Number of Rounds: ", len(evaluation))

    def moving_average(a, n=10):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret / n

    def plot(what: str, label: str = None):

        plt.figure(figsize=figsizes)
        for _round in evaluation:
            if parameter[0] is int:
                paras = map(lambda para: f"{para}={_round['params'][para]:.4f}", parameter)
            else:
                paras = map(lambda para: f"{para}={_round['params'][para]}", parameter)
            if num_steps == -1:
                steps = len(_round[what])
            else:
                steps = num_steps

            p = plt.plot(_round['total_steps'], moving_average(_round[what][:steps]), label=", ".join(paras))
            color = p[0].get_color()
            plt.plot(_round['total_steps'][:steps], _round[what][:steps],
                     c=color, alpha=0.2)

        label = label if label else what

        plt.title(
            f"{evaluation[0]['params']['env_domain'].capitalize()}:{evaluation[0]['params']['env_task'].capitalize()} - Optimization of {', '.join(parameter)}")
        plt.xlabel("Update steps")
        plt.ylabel(f"Average {label}")
        plt.legend()
        plt.tight_layout()
        date_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"{'_'.join(parameter)}_{what}_{date_now}"
        if save:
            plt.savefig(f'hp_figures/{filename}.pdf')
            print(f"File saved to {filename}")
        plt.show()

    plot('rewards')
    plot('time')
    plot('q_losses')
    plot('policy_losses')
    plot('alpha_losses', "applied alpha value")
    return list(map(lambda r: r['params'], evaluation))


dir = "hp_trials"
ending = "model"
parameter = ["init_alpha"]
# load_and_display_model(dir,ending,parameter)

if __name__ == '__main__':
    pass
