"""
Usage:
  training <environment> <agent> [options]
  training -h | --help

Options:
  -h --help              Show this screen.
  --total-timesteps <count>     Number of episodes [default: 100].
  --processes <count>    Number of running processes [default: 4].
  --trial-mode          Do not save model or log
"""

import datetime
import time
from pathlib import Path

from docopt import docopt
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from rt_drl_safeguard.utils.factory import load_agent, load_environment

OUTPUT_FOLDER = 'out'
FINAL_MODEL = 'final_model'


def main():
    opts = docopt(__doc__)
    training(opts['<environment>'], opts['<agent>'], opts)


def training(environment_config, agent_config, options, total_timesteps=200):
    """
    Training an agent.
    :param environment_config: the path of the environment configuration file
    :param agent_config: the path of the agent configuration file
    :param options: the execution options
    :param total_timesteps: total number of training steps (not episodes)
    """

    total_timesteps = int(options['--total-timesteps'])
    n_proc = int(options['--processes'])
    trial_mode_on = options['--trial-mode']
    env, env_name = load_environment(environment_config, training=True, n_envs=n_proc)
    agent = load_agent(agent_config, env)

    version = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    model_saved_dir = Path(OUTPUT_FOLDER) / env_name / agent.__class__.__module__ / version

    eval_callback = None
    if not trial_mode_on:
        training_logger = configure(str(model_saved_dir), ['stdout', 'log', 'json', 'csv', 'tensorboard'])
        eval_callback = EvalCallback(env,
                                     best_model_save_path=str(model_saved_dir),  # file name 'best_model'
                                     log_path=str(model_saved_dir),
                                     eval_freq=1000, n_eval_episodes=10,
                                     deterministic=True, render=False)
    else:
        training_logger = configure(None, ['stdout'])
    training_logger.log("Options:\n{}".format(options))
    training_logger.log("Environment configuration:\n{}".format(open(options['<environment>']).read()))
    training_logger.log("Agent configuration:\n{}\n".format(open(options['<agent>']).read()))

    agent.set_logger(training_logger)

    start_time = time.time()
    # agent.learn(total_timesteps=total_timesteps, progress_bar=True)
    agent.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    duration = time.time() - start_time

    if not trial_mode_on:
        agent.save(str(model_saved_dir / FINAL_MODEL))
    training_logger.log("Total training time: {} seconds, per timestep: {} second(s)".format(
        duration, duration / total_timesteps))


if __name__ == "__main__":
    main()
