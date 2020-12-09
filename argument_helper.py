import argparse


def parse(defaults: dict) -> dict:
    parser = argparse.ArgumentParser(description="Run the SAC-RL Agent.")

    # #############################################################
    # Logging
    # #############################################################
    parser.add_argument('--log_level',
                        type=str,
                        default=defaults['log_level'],
                        # TODO Add more meaningful description
                        help='Hidden Dimension for something')

    parser.add_argument('--log_file',
                        default=defaults['log_file'],
                        type=str,
                        # TODO Add more meaningful description
                        help='Environment Name')

    # #############################################################
    # Video
    # #############################################################
    parser.add_argument('--save_video',
                        default=False,
                        action='store_true')
    parser.add_argument('--recording_interval',
                        default=defaults['recording_interval'],
                        type=int)

    # #############################################################
    # Neural Networks and its Parameters
    # #############################################################

    parser.add_argument('--hidden_dim',
                        type=int,
                        default=defaults['hidden_dim'],
                        # TODO Add more meaningful description
                        help='Hidden Dimension for something')

    parser.add_argument('--lr-actor',
                        type=float,
                        default=defaults['lr-actor'],
                        # TODO Add more meaningful description
                        help='Learning Rate')

    parser.add_argument('--lr-critic',
                        type=float,
                        default=defaults['lr-critic'],
                        # TODO Add more meaningful description
                        help='Learning Rate')

    # #############################################################
    # Parameter for RL
    # #############################################################

    parser.add_argument('--gamma',
                        default=defaults['gamma'],
                        type=float,
                        # TODO Add more meaningful description
                        help='')

    parser.add_argument('--alpha',
                        default=defaults['alpha'],
                        type=float,
                        # TODO Add more meaningful description
                        help='')

    parser.add_argument('--tau',
                        default=defaults['tau'],
                        type=float,
                        # TODO Add more meaningful description
                        help='')
    # ############################################################
    # alpha decay
    # ############################################################
    parser.add_argument('--init_alpha',
                        default=defaults['init_alpha'],
                        type=float,
                        # TODO Add more meaningful description
                        help='')
    parser.add_argument('--alpha_lr',
                        default=defaults['alpha_lr'],
                        type=float,
                        # TODO Add more meaningful description
                        help='')
    parser.add_argument('--alpha_beta',
                        default=defaults['alpha_beta'],
                        type=float,
                        # TODO Add more meaningful description
                        help='')

    parser.add_argument('--alpha_decay_activated',
                        default=True,
                        action='store_false')
    # ############################################################
    # Environment
    # ############################################################

    parser.add_argument('--env-domain',
                        default=defaults['env_domain'],
                        type=str,
                        # TODO Add more meaningful description
                        help='Domain Name for the task')

    parser.add_argument('--env-task',
                        default=defaults['env_task'],
                        type=str,
                        # TODO Add more meaningful description
                        help='Task Name')

    parser.add_argument('--frame-skip',
                        default=defaults['frame-skip'],
                        type=int,
                        help='Applying an action for several step. According to tutorial: Card_pole: 8, Finka Task: 2 and a default of 4')


    parser.add_argument('--seed',
                        default=defaults['seed'],
                        type=int,
                        # TODO Add more meaningful description
                        help='')

    # ############################################################
    # Parameter for the running of RL
    # ############################################################

    parser.add_argument('--replay_buffer_size',
                        default=defaults['replay_buffer_size'],
                        type=int,
                        # TODO Add more meaningful description
                        help='')

    # TODO Set higher episode times
    parser.add_argument('--episodes',
                        default=defaults['episodes'],
                        type=int,
                        # TODO Add more meaningful description
                        help='Episodes for the Training')

    # TODO Set higher episode times
    parser.add_argument('--sample_batch_size',
                        default=defaults["sample_batch_size"],
                        type=int,
                        # TODO Add more meaningful description
                        help='Episodes for the Training')

    parser.add_argument('--max_steps',
                        default=defaults['max_steps'],
                        type=int,
                        # TODO Add more meaningful description
                        help='Update Episodes for the Training (Offline-Policy-learning)')

    # ############################################################
    # Hyperparameter optimization stuff
    # ############################################################
    parser.add_argument('--max_evals',
                        default=defaults['max_evals'],
                        type=int,
                        # TODO Add more meaningful description
                        help='Number of Hyperparameter tests')

    # ############################################################
    # Specify GPU ID
    # ############################################################

    parser.add_argument('--gpu_device',
                        default=defaults['gpu_device'],
                        type=str,
                        # TODO Add more meaningful description
                        help='Specify the GPU to use. Range: 0-3')

    args = vars(parser.parse_args())
    return args