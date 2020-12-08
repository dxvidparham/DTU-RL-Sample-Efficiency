import datetime

from hyperopt import hp

from argument_helper import parse
from LogHelper import setup_logging

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
    "recording_interval": 5,
    # Neural Network stuff
    "hidden_dim": 256,
    "lr-actor": 5e-4,
    "lr-critic": 1e-3,
    # Parameter for RL
    "gamma": 0.98,
    "alpha": 1e-2,
    "tau": 0.01,  # for target network soft update,
    # Environment
    "env_domain": "cartpole",
    "env_task": "balance",
    "seed": 1,
    "frame-skip": 8,
    # Parameter for running RL
    "replay_buffer_size": 10 ** 6,
    "sample_batch_size": 128,
    "episodes": 500,
    "max_steps": 250,
    # Hyperparameter-tuning
    "max_evals": 10,
    # ID of the GPU to use
    "gpu_device": "1",
}

# HYPERPARAMETER training.
hyperparameter_space = {
    "hyperparmeter_round": "gamma",
    "gamma": hp.quniform("gamma", 0.7, 1, 0.01),
    # "alpha": hp.quniform('alpha', 0.0005, 0.1, 0.001),
    # "tau": hp.uniform('tau', 0, 0.05),
    "hidden_dim": hp.choice("hidden_dim", [256]),
    "num_updates": 1,
}


args = parse(defaults=parameter)

# Setup the logging
setup_logging(args)
# The import must be done down here to allow the logging configuration
from SAC_Implementation import train

# START training. Set Max Eval to 1 to just train one episode.
train.prepare_hyperparameter_tuning(
    {**args, **hyperparameter_space}, max_evals=args["max_evals"]
)
