import random
import torch
import datetime
import numpy as np

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
    "recording_interval": 500,

    # Neural Network stuff
    "hidden_dim": 1024,
    "lr-actor": 5e-4,
    "lr-critic": 1e-3,

    # Parameter for RL
    "gamma": 0.99,
    "alpha": 1e-2,
    "tau": 0.01,  # for target network soft update,

    # Environment
    "env_domain": "walker",
    "env_task": "walk",
    "seed": 1,
    "frame-skip": 4,

    # Parameter for running RL
    "replay_buffer_size": 10 ** 6,
    "sample_batch_size": 128,
    "episodes": 1000,
    "max_steps": 250,
    # Hyperparameter-tuning
    "max_evals": 1,

    # ID of the GPU to use
    "gpu_device": "0",

    #alpha
    "init_alpha": 0.1,
    "alpha_lr": 1e-4,
    "alpha_beta": 0.5,
    "alpha_decay_deactivate": False
}

# HYPERPARAMETER training.
hyperparameter_space = {
    "hyperparmeter_round": "walker_hidden_dim",
    #"init_alpha": hp.quniform('init_alpha', 0, 0.4, 0.01),
    #"gamma": hp.quniform('gamma', 0.8,0.99,0.01),
    #"tau": hp.quniform('tau', 0.005,0.2,0.001),
    #"alpha": hp.quniform('alpha', 0.0005, 0.1, 0.001),
    #"tau": hp.uniform('tau', 0, 0.05),
    #"hidden_dim": hp.choice('hidden_dim', [1024, 2048, 4096]),
    "num_updates": 5
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
