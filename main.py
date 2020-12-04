# Setting up the logging
import logging

from LogHelper import ColouredHandler, ColouredFormatter
from VideoRecorder import VideoRecorder

import argparse
import datetime

from hyperopt import hp

from argument_helper import parse

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = f"logging_{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"


parameter = {
    # Logging
    "log_level": "INFO",
    "log_file": f'{DEFAULT_LOG_DIR}/{DEFAULT_LOG_FILE}',

    # video
    "save_video": True,
    "recording_interval": 5,

    # Neural Network stuff
    "hidden_dim": 256,
    "lr-actor": 3e-4,
    "lr-critic": 3e-4,

    # Parameter for RL
    "gamma": 0.98,
    "alpha": 0.01,
    "tau": 0.01,  # for target network soft update,

    # Environment
    "env_domain": "walker",
    "env_task": "walk",
    "seed": 1,
    "frame-skip": 4,

    # Parameter for running RL
    "replay_buffer_size": 10 ** 6,
    "sample_batch_size": 1024,
    "episodes": 100,
    "max_steps": 10000,

    # Hyperparameter-tuning
    "max_evals": 3,

    # ID of the GPU to use
    "gpu_device": "1",
}

hyperparameter_space = {
    "gamma": hp.uniform('gamma', 0.9, 1),
    "alpha": hp.uniform('alpha', 0, 0.05),
    "tau": hp.uniform('tau', 0, 0.05),
    "hidden_dim": hp.choice('hidden_dim', [128, 256, 512]),
    "policy_function": hp.choice('policy_function', [1, 2, 3])
}
logging.info({**parameter, **hyperparameter_space})

args = parse(defaults=parameter)

logging.info(f"{type(args.get('replay_buffer_size'))}")

# Setup the logging environment
level = logging.getLevelName(args.get('log_level'))
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

format = '{asctime} {levelname:8} {message}'
date_format = '%Y-%m-%d %H:%M:%S'

h = ColouredHandler()
h.formatter = ColouredFormatter(format, date_format, '{')

file_handler = logging.FileHandler(parameter.get('log_file'),
                                   mode='a',
                                   )
file_handler.formatter = ColouredFormatter(format, date_format, '{')

# format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
logging.basicConfig(datefmt=date_format,
                    level=int(level),
                    handlers=[file_handler, h]
                    )

logger = logging.getLogger(__name__)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


# The import must be done down here to allow the logging configuration
from SAC_Implementation import train

params = []
train.prepare_hyperparameter_tuning({**args, **hyperparameter_space},
                                    max_evals=args['max_evals'])

# Running of the SAC
# train.run_sac(hyperparameter_space={**parameter, **hyperparameter_space}, video=video)

##
