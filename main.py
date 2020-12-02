# Setting up the logging
import logging
from VideoRecorder import VideoRecorder

import argparse
import datetime

from hyperopt import hp

from argument_helper import parse


DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = f"logging_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"


DEFAULT_VIDEO_DIR = "videos"

parameter = {
    # Logging
    "log_level": "DEBUG",
    "log_file": f'{DEFAULT_LOG_DIR}{DEFAULT_LOG_FILE}',

    #video
    "save_video": True,
    "recording_interval": 5,

    # Neural Network stuff
    "hidden_dim": 256,
    "lr-actor": 3e-4,
    "lr-critic": 3e-4,

    # Parameter for RL
    "gamma": .5,
    "alpha": .5,
    "tau": 0.01,

    # Environment
    "env_domain": "cartpole",
    "env_task": "balance",
    "seed": 1,
    "frame-skip": 4,

    # Parameter for running RL
    "replay_buffer_size": 10 ** 6,
    "sample_batch_size": 1024,
    "episodes": 100,
    "max_steps": 10000,

    # Hyperparameter-tuning
    "max_evals": 10,

    # ID of the GPU to use
    "gpu_device": 3,
}

hyperparameter_space = {
    #"gamma": hp.uniform('gamma', 0, 1),
    # "alpha": hp.uniform('alpha', 0, 1),
    #"tau": hp.uniform('tau', 0.8, 1),
     "hidden_dim": hp.choice('hidden_dim', [16, 64, 129, 256, 512])
}
logging.info({**parameter, **hyperparameter_space})

args = parse(defaults=parameter)

logging.info(f"{type(args.get('replay_buffer_size'))}")

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

#Initialize video object
video = VideoRecorder(DEFAULT_VIDEO_DIR if args.get('save_video') else None)


# The import must be done down here to allow the logging configuration
from SAC_Implementation import SAC



params = []
#SAC.prepare_hyperparameter_tuning({**args, **hyperparameter_space}, max_evals=args['max_evals'])

# Running of the SAC
SAC.run_sac(hyperparameter_space=args, video=video)
