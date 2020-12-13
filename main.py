import datetime

from hyperopt import hp

from argument_helper import parse
from LogHelper import setup_logging

import torch
import random
import numpy as np

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = (
    f"logging_{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"
)

# DEFAULT PARAMETERS WILL BE OVERWRITTEN BY THE
# ARGUMENT PARSER
parameter = {
    # Logging
    "log_level": "INFO",
    "log_file": f"{DEFAULT_LOG_DIR}/{DEFAULT_LOG_FILE}",
    # video
    "save_video": True,
    "recording_interval": 100,
    # Neural Network stuff
    "hidden_dim": 256,
    "lr-actor": 5e-4,
    "lr-critic": 1e-3,
    # Parameter for RL
    "gamma": 0.98,
    "alpha": 1e-2,
    "tau": 0.01,  # for target network soft update,
    # Environment
    "env_domain": "walker", # cartpole, walker
    "env_task": "walk",  #balance, swingup, walk
    "seed": 1,
    "frame-skip": 8,
    # Parameter for running RL
    "replay_buffer_size": 10 ** 6,
    "sample_batch_size": 128,
    "episodes": 500,
    "max_steps": 250,
    # Hyperparameter-tuning
    "max_evals": 5,
    # ID of the GPU to use
    "gpu_device": "0",

    #alpha
    "init_alpha": 0.496,
    "alpha_lr": 1e-3,
    "alpha_beta": 0.9,
    "alpha_decay_deactivate": False
}

# HYPERPARAMETER training.
hyperparameter_space = {
    "hyperparmeter_round": "hidden_dim",
    #"init_alpha": hp.quniform('init_alpha', 0.001, 0.5, 0.001),
    #"alpha": hp.quniform('alpha', 0.0005, 0.1, 0.001),
    #"tau": hp.uniform('tau', 0, 0.05),
    "hidden_dim": hp.choice('hidden_dim', [256]),
    "num_updates": 1
}


args = parse(defaults=parameter)

# Setup the logging
setup_logging(args)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(args.get("seed"))


# The import must be done down here to allow the logging configuration
from SAC_Implementation import train

# START training. Set Max Eval to 1 to just train one episode.
train.prepare_hyperparameter_tuning(
    {**args, **hyperparameter_space}, max_evals=args["max_evals"]
)
