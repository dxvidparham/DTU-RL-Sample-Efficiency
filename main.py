# Setting up the logging
import logging

import argparse
import datetime

from argument_helper import parse

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = f"logging_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"

parameter = {
    # Logging
    "log_level": "DEBUG",
    "log_file": f'{DEFAULT_LOG_DIR}{DEFAULT_LOG_FILE}',

    # Neural Network stuff
    "hidden_dim": 256,
    "lr-actor": 3e-4,
    "lr-critic": 3e-4,

    # Parameter for RL
    "gamma": .5,
    "alpha": .5,
    "tau": 0.09,

    # Environment
    "env_domain": "cartpole",
    "env_task": "balance",
    "seed": 5,
    "frame-skip": 4,

    # Parameter for running RL
    "replay_buffer_size": 10 ** 6,
    "sample_batch_size": 1024,
    "episodes": 2000,
    "max_steps": 10000,
}

args = parse(defaults=parameter)

logging.debug(f"{type(args.get('replay_buffer_size'))}")

# Setup the logging environment
level = logging.getLevelName(args.get('log_level'))
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=int(level),
                    handlers=[logging.FileHandler("./logs/my_log.log", mode='w'),
                              logging.StreamHandler()]
                    )
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# The import must be done down here to allow the logging configuration
from SAC_Implementation import SAC

# Running of the SAC
SAC.run_sac(hyperparameter_space=args)
