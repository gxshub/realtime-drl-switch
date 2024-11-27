"""
Usage:
  evaluate <environment> [options]
  evaluate -h | --help

Options:
  -h --help              Show this screen.
  --episodes <count>     Number of episodes [default: 4].
  --no-display           Disable environment, agent, and rewards rendering.
  --test              Do not save model or log
"""

import datetime
from pathlib import Path

import numpy as np
from docopt import docopt
from stable_baselines3.common.logger import configure
from tqdm import tqdm

from rt_drl_safeguard.safeguard.highway_safeguard import SecondaryController
from rt_drl_safeguard.utils.factory import load_environment

OUTPUT_FOLDER = "contr_eval_results"


def main():
    opts = docopt(__doc__)
    env_config_file = opts['<environment>']
    test_mode = opts['--test']
    n_episodes = int(opts['--episodes'])

    env, _env_name = load_environment(env_config_file, n_envs=1)
    controller = SecondaryController(env)

    version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test_result_folder = Path(__file__).parent / _env_name / OUTPUT_FOLDER / version

    if test_mode:
        logger = configure(None, ['stdout'])
    else:
        logger = configure(str(test_result_folder), ['stdout', 'log'])

    logger.log("Options:\n{}".format(opts),
               "\nEnvironment configuration:\n{}".format(open(env_config_file).read()))

    avg_epi_rew, avg_epi_len, crash_rate = evaluate(env, controller, n_episodes)

    logger.log("----------------------\nNumber of episodes: {}".format(n_episodes),
               "\nAverage episode accumulated reward: {}".format(avg_epi_rew),
               "\nAverage episode length: {}".format(avg_epi_len),
               "\nCrash rate: {}".format(crash_rate))


def evaluate(env, controller, n_episodes):
    episode_rewards = []
    episode_lengths = []
    n_crashes = 0
    for _ in tqdm(range(n_episodes)):
        reward_acc = 0.
        timestep = 0
        done = truncated = False
        env.reset()
        while not (done or truncated):
            action = controller.control()
            # print("action: ", action)
            obs, reward, done, truncated, info = env.step(action)
            reward_acc += reward
            timestep += 1
        episode_rewards.append(reward_acc)
        episode_lengths.append(timestep)
        crashed = env.unwrapped.vehicle.crashed
        if crashed:
            n_crashes += 1
    return np.mean(episode_rewards), np.mean(episode_lengths), n_crashes / n_episodes


if __name__ == "__main__":
    main()
