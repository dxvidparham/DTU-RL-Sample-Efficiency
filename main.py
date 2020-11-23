# Setting up the logging
import logging

import argparse
import datetime

DEFAULT_LOG_DIR="logs"
DEFAULT_LOG_FILE=f"logging_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"

parser = argparse.ArgumentParser(description="Run the SAC-RL Agent.")

# Add Arguments depending on the Hyperparameter variables.
# The name of the argument is then the key of the hyperparameter dict.

parser.add_argument('--log_level',
                    type=str,
                    default='DEBUG',
                    # TODO Add more meaningful description
                    help='Hidden Dimension for something')

parser.add_argument('--log_file',
                    default=f'{DEFAULT_LOG_DIR}/{DEFAULT_LOG_FILE}',
                    type=str,
                    # TODO Add more meaningful description
                    help='Environment Name')

parser.add_argument('--hidden_dim',
                    type=int,
                    default=256,
                    # TODO Add more meaningful description
                    help='Hidden Dimension for something')

parser.add_argument('--learning_rate',
                    type=float,
                    default=3e-4,
                    # TODO Add more meaningful description
                    help='Learning Rate')

parser.add_argument('--discount-factor',
                    default=0.99,
                    type=float,
                    # TODO Add more meaningful description
                    help='Discount Factor')

parser.add_argument('--replay_buffer',
                    default=10**6,
                    type=float,
                    # TODO Add more meaningful description
                    help='')

parser.add_argument('--n_hidden_layer',
                    default=2,
                    type=int,
                    # TODO Add more meaningful description
                    help='')

parser.add_argument('--n_hidden',
                    default=256,
                    type=int,
                    # TODO Add more meaningful description
                    help='')

parser.add_argument('--target_smoothing',
                    default=5e-3,
                    type=float,
                    # TODO Add more meaningful description
                    help='')

parser.add_argument('--val_freq',
                    default=50,
                    type=float,
                    # TODO Add more meaningful description
                    help='Validation Frequency')

parser.add_argument('--env_name',
                    default='CartPole-v0',
                    type=str,
                    # TODO Add more meaningful description
                    help='Environment Name')

parser.add_argument('--episodes',
                    default=10,
                    type=int,
                    # TODO Add more meaningful description
                    help='Episodes for the Training')

args = vars(parser.parse_args())

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

# The import must be done down here to allow the logging configuration
from SAC_Implementation import SAC

# Running of the SAC
SAC.run_sac(hyperparameter_space=args)

