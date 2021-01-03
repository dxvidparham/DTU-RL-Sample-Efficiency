## Imports
import os
import pickle
import random
import time
from datetime import datetime
from typing import Dict

import numpy as np

import torch
from scipy.special.cython_special import hyperu

from SAC_Implementation.SACAlgorithm import SACAlgorithm
from VideoRecorder import VideoRecorder
from plotter import Plotter

from hyperopt import fmin, tpe, Trials, STATUS_OK

import logging
import dmc2gym
import LogHelper


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_hyperparameter_tuning(hyperparameter_space, max_evals=2):
    """
    Starting point for the hyperparameter training
    :param hyperparameter_space:
    :param max_evals:
    :return: Losses
    """
    try:
        filename = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        file_path = f"hp_trials/{hyperparameter_space.get('hyperparmeter_round')}_{filename}.model"

        trials = Trials()
        best = fmin(run_sac,
                    hyperparameter_space,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=max_evals,
                    trials_save_file=file_path
                    )

        logging.info("WE ARE DONE. THE BEST TRIAL IS:")
        LogHelper.print_dict({**hyperparameter_space, **best}, "Final Parameters")

        logging.info("--------------------------------------------")
        logging.info(f"For more information see {file_path}")
    except KeyboardInterrupt as e:
        logging.error("KEYBOARD INTERRUPT")
        raise


def run_sac(hyperparameter_space: dict) -> Dict:
    """
    Method to to start the SAC algorithm on a certain problem
    :param video: video object
    :param hyperparameter_space: Dict with the hyperparameter from the Argument parser
    :return:
    """
    # Print the hyperparameter
    LogHelper.print_big_log('Initialize Hyperparameter')
    LogHelper.print_dict(hyperparameter_space, "Hyperparameter")
    LogHelper.print_big_log("Start Training")

    set_seed(hyperparameter_space.get('seed'))

    # Initialize the environment
    env, action_dim, state_dim = initialize_environment(domain_name=hyperparameter_space.get('env_domain'),
                                                        task_name=hyperparameter_space.get('env_task'),
                                                        seed=hyperparameter_space.get('seed'),
                                                        frame_skip=hyperparameter_space.get('frame_skip'))

    # Create the SAC Algorithm
    sac = SACAlgorithm(env=env,
                       param={
                           "hidden_dim": hyperparameter_space.get('hidden_dim'),
                           "lr_critic": hyperparameter_space.get('lr_critic'),
                           "lr_actor": hyperparameter_space.get('lr_actor'),
                           "alpha": hyperparameter_space.get('alpha'),
                           "tau": hyperparameter_space.get('tau'),
                           "gamma": hyperparameter_space.get('gamma'),
                           "sample_batch_size": hyperparameter_space.get('sample_batch_size'),
                           "replay_buffer_size": hyperparameter_space.get('replay_buffer_size'),
                           "gpu_device": hyperparameter_space.get('gpu_device'),
                           "policy_function": hyperparameter_space.get('policy_function'),
                           "init_alpha": hyperparameter_space.get('init_alpha'),
                           "alpha_lr": hyperparameter_space.get('alpha_lr'),
                           "alpha_beta": hyperparameter_space.get('alpha_beta'),
                           "alpha_decay_deactivate": hyperparameter_space.get('alpha_decay_deactivate'),

                           "policy_hidden_layers": hyperparameter_space.get('policy_hidden_layers'),
                           "q_hidden_layers": hyperparameter_space.get('q_hidden_layers')
                       })

    video, plotter, recording_interval = initialize_plotting(hyperparameter_space)
    total_step = 0

    reward_velocity = 0

    try:
        for _episode in range(hyperparameter_space.get('episodes')):
            _start = time.time()

            logging.debug(f"Start EPISODE {_episode + 1}")

            ep_reward, policy_loss_incr, q_loss_incr, alpha_loss_incr, length = 0, [], [], [], 0
            # Observe state
            current_state = env.reset()

            # Run Step until done signal
            for step in range(hyperparameter_space.get('max_steps')):
                total_step += 1

                # Do the next step
                # logging.warning("STEEEEEP 5")
                action_mean = sac.sample_action(torch.Tensor(current_state))[0] if _episode > hyperparameter_space.get("init_rounds") \
                    else env.action_space.sample()

                # logging.warning("STEEEEEP 6")
                s1, r, done, _ = env.step(np.array(action_mean))

                # The last done is fake therefore we set it to true again
                if (step + 1) == int(hyperparameter_space.get('max_steps')):
                    done = False

                LogHelper.log_step(_episode, step, r, action_mean)

                # logging.warning("STEEEEEP 7")
                sac.buffer.add(obs=current_state, action=action_mean, reward=r, next_obs=s1, done=done)
                ep_reward += r

                # logging.warning("STEEEEEP 8")
                if bool(done): break

                # Update current step
                current_state = s1

                # logging.warning("STEEEEEP 9")
                # if sac.buffer.length > sac.sample_batch_size:
                if sac.buffer.length > 1000:
                    _polo, _qlo, _alo = [], [], []
                    # TODO REWRITE
                    update_steps = hyperparameter_space.get('max_steps') if total_step == hyperparameter_space.get(
                        'max_steps') else hyperparameter_space.get('num_updates')
                    for i in range(update_steps):
                        # Update the network
                        _metric = sac.update(step)
                        _polo.append(_metric[0])
                        _qlo.append(_metric[1])
                        _alo.append(_metric[2])
                    policy_loss_incr.append(sum(_polo) / len(_polo))
                    q_loss_incr.append(sum(_qlo) / len(_qlo))
                    alpha_loss_incr.append(sum(_alo) / len(_alo))
                    length = step

                if _episode % recording_interval == 0: video.record(env)

            if _episode % recording_interval == 0: video.save_and_reset(_episode)

            _end = time.time()

            avg_ploss = sum(policy_loss_incr) / len(policy_loss_incr) if len(policy_loss_incr) != 0 else -1
            avg_qloss = sum(q_loss_incr) / len(q_loss_incr) if len(q_loss_incr) != 0 else -1
            avg_aloss = sum(alpha_loss_incr) / len(alpha_loss_incr) if len(alpha_loss_incr) != 0 else -1

            _last_ploss = plotter.get_last_ploss()
            plotter.add_to_lists(reward=ep_reward,
                                 length=length,
                                 policy_loss=avg_ploss,
                                 q_loss=avg_qloss,
                                 a_loss=avg_aloss,
                                 total_steps=total_step,
                                 episode=_episode,
                                 time=_end - _start,
                                 log="INFO" if _episode % 1 == 0 else "DEBUG")

            if _last_ploss >= avg_ploss:
                reward_velocity = 0
            else:
                reward_velocity = reward_velocity + 1

            if reward_velocity > 15:
                max_reward = np.inf
                logging.error(f"TOO often the policy got bad: {reward_velocity}")
                #break

            if avg_qloss > 10000:
                max_reward = np.inf
                logging.error(f"ABORT DUE TO TOO HIGH QLOSS: {avg_qloss}")
                # break
            if avg_ploss > 10000:
                max_reward = np.inf
                logging.error(f"ABORT DUE TO TOO HIGH POLICY LOSS: {avg_ploss}")
                # break

    except KeyboardInterrupt as e:
        logging.error("KEYBOARD INTERRUPT")
        raise
    finally:
        # TODO ITS DEACTIVATED
        plotter.plot()
        pass

    rew, _, q_losses, policy_losses, total_step, timing, a_losses = plotter.get_lists()

    # Give back the error which should be optimized by the hyperparameter tuner
    max_reward = max(np.array(rew))

    return {'loss': -max_reward,
            'status': STATUS_OK,
            'model': sac,
            'max_reward': max_reward,
            'q_losses': q_losses,
            'policy_losses': policy_losses,
            'alpha_losses': a_losses,
            'rewards': rew,
            'total_steps': total_step,
            'time': timing,
            'params': hyperparameter_space}


def initialize_environment(domain_name, task_name, seed, frame_skip):
    """
    Initialize the Evironment
    :param domain_name:
    :param task_name:
    :param seed:
    :param frame_skip:
    :return:
    """
    LogHelper.print_step_log(f"Initialize Environment: {domain_name}/{task_name} ...")

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=seed,
                       frame_skip=frame_skip)

    # Debug logging to check environment specs
    s = env.reset()
    a = env.action_space.sample()
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    logging.debug(f'Sample state: {s}')
    logging.debug(f'Sample action:{a}')
    logging.debug(f'State DIM: {state_dim}')
    logging.debug(f'Action DIM:{action_dim}')

    return env, action_dim, state_dim


def initialize_plotting(hyperparameter_space: dict):
    # Initialize video object
    DEFAULT_VIDEO_DIR = "videos"

    if not os.path.exists('figures'):
        os.makedirs('figures')
    if not os.path.exists(DEFAULT_VIDEO_DIR):
        os.makedirs(DEFAULT_VIDEO_DIR)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('results'):
        os.makedirs('results')

    video = VideoRecorder(DEFAULT_VIDEO_DIR if hyperparameter_space.get('save_video') else None)

    # Init the Plotter
    plotter = Plotter(hyperparameter_space.get('episodes'))

    video.init()
    recording_interval = hyperparameter_space.get('recording_interval')
    return video, plotter, recording_interval
