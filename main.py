from LogHelper import setup_logging
import datetime

from hyperopt import hp

from argument_helper import parse

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = f"logging_{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"

# DEFAULT PARAMETERS WILL BE OVERWRITTEN BY THE
# ARGUMENT PARSER
parameter = {
    # Logging
    "log_level": "INFO",
    "log_file": f'{DEFAULT_LOG_DIR}/{DEFAULT_LOG_FILE}',

    # video
    "save_video": True,
    "recording_interval": 5,

    # Neural Network stuff
    "hidden_dim": 256,
    "lr-actor": 5e-4,
    "lr-critic": 1e-3,

    # Parameter for RL
    "gamma": 0.98,
    "alpha": 0.2,# Cartpole Balance: 1e-2,
    "tau": 0.01,  # for target network soft update,

    # Environment
    "env_domain": "cartpole",
    "env_task": "balance",
    "seed": 1,
    "frame-skip": 8,

    # Parameter for running RL
    "replay_buffer_size": 10 ** 6,
    "sample_batch_size": 128,
    "episodes": 100,
    "max_steps": 250,

    # Hyperparameter-tuning
    "max_evals": 1,

    # ID of the GPU to use
    "gpu_device": "1",
}

# HYPERPARAMETER training.
hyperparameter_space = {
    #"gamma": hp.uniform('gamma', 0.9, 1),
    # "alpha": hp.uniform('alpha', 0.0005, 0.0015),
    #"tau": hp.uniform('tau', 0, 0.05),
    "hidden_dim": hp.choice('hidden_dim', [256]),
    "policy_function": hp.choice('policy_function', [1, 2, 3])
}


args = parse(defaults=parameter)

# Setup the logging
setup_logging(args)
# The import must be done down here to allow the logging configuration
from SAC_Implementation import train

# START training. Set Max Eval to 1 to just train one episode.
train.prepare_hyperparameter_tuning({**args, **hyperparameter_space},
                                    max_evals=args['max_evals'])

