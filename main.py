import random
import torch
import datetime
import numpy as np
from hyperopt import hp

from argument_helper import parse
from LogHelper import setup_logging

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = (
    f"logging_{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"
)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# DEFAULT PARAMETERS WILL BE OVERWRITTEN BY THE
# ARGUMENT PARSER
parameter = {
    # Logging
    "log_level": "INFO",
    "log_file": f"{DEFAULT_LOG_DIR}/{DEFAULT_LOG_FILE}",
    # video
    "save_video": False,
    "recording_interval": 100,

    # Neural Network stuff
    "hidden_dim": 512,
    "lr-actor": 5e-4,
    "lr-critic": 1e-3,

    "policy_hidden_layers": 1,
    "q_hidden_layers": 1,

    # Parameter for RL
    "gamma": 0.98,
    "alpha": 0.01,
    "tau": 0.01,  # for target network soft update,

    # Environment
    "env_domain": "ball_in_cup",
    "env_task": "catch",
    "seed": 1,
    "frame-skip": 4,

    # Parameter for running RL
    "replay_buffer_size": 10 ** 6,
    "sample_batch_size": 128,
    "episodes": 300,
    "max_steps": 128,
    # Hyperparameter-tuning
    "max_evals": 5,

    # ID of the GPU to use
    "gpu_device": "0",

    #alpha
    "init_alpha": 0.1,
    "alpha_lr": 1e-4,
    "alpha_beta": 0.5,
    "alpha_decay_deactivate": False,

    # Initial sampling
    # Number of rounds which are sampled random
    "init_rounds": -1,
    "num_updates": 1
}

# HYPERPARAMETER training.
hyperparameter_space = {
    "hyperparmeter_round": "ball_in_cup_init_alpha_",
    "init_alpha": hp.quniform('init_alpha', 0, 0.5, 0.01),
    #"tau": hp.quniform('tau', 0.005, 0.3, 0.01),
    #"tau": hp.choice('tau', 0.01, 0.03, 0.005, 0.05, 0.1]),
    # "hidden_dim": hp.choice('hidden_dim', [512, 1024, 2048]),
}

args = parse(defaults=parameter)

set_seed(args.get('seed'))
# Setup the logging
setup_logging(args)
# The import must be done down here to allow the logging configuration
from SAC_Implementation import train

# START training. Set Max Eval to 1 to just train one episode.
train.prepare_hyperparameter_tuning({**args, **hyperparameter_space},
                                    max_evals=args['max_evals'])

##

