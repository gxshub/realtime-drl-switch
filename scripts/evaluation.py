"""
Usage:
  evaluation <environment> <agent> <checkpoint> [options]
  evaluation -h | --help

Options:
  -h --help              Show this screen.
  --episodes <count>     Number of episodes [default: 4].
  --processes <count>    Number of running processes [default: 4].
  --no-display           Disable environment, agent, and rewards rendering.
  --repeat <times>       Repeat several times [default: 1].
  --test              Do not save model or log
"""

import datetime
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from docopt import docopt
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from tqdm import tqdm

from rt_drl_safeguard.utils.factory import load_agent_class, load_environment

OUTPUT_FOLDER = "eval_results"


def main():
    opts = docopt(__doc__)
    env_config_file = opts['<environment>']
    agent_config_file = opts['<agent>']
    model_file = opts['<checkpoint>']
    test_mode = opts['--test']
    n_episodes = int(opts['--episodes'])
    n_proc = int(opts['--processes'])

    version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test_result_folder = Path(model_file).parent / OUTPUT_FOLDER / version

    if test_mode:
        logger = configure(None, ['stdout'])
    else:
        logger = configure(str(test_result_folder), ['stdout', 'log'])
    logger.log("Options:\n{}".format(opts))
    logger.log("Environment configuration:\n{}".format(open(env_config_file).read()))
    logger.log("Agent configuration:\n{}".format(open(agent_config_file).read()))
    logger.log("Load checkpoint file from: {}\n".format(model_file))

    env, _ = load_environment(env_config_file, n_envs=n_proc)
    agent = load_agent_class(agent_config_file).load(Path(model_file))

    # test(opts['<environment>'], opts['<agent>'], opts['<checkpoint>'], opts)
    avg_epi_rew, avg_epi_len, crash_rate = evaluate(env, agent, n_episodes)

    logger.log("Average episode accumulated reward: {}".format(avg_epi_rew))
    logger.log("Average episode length: {}".format(avg_epi_len))
    logger.log("Crash rate: {}".format(crash_rate))


def evaluate(
        env, agent, n_episodes,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        warn: bool = True,
):
    """
        Testing an agent.

    Note: This function is based on evaluate_policy() in sb3.
    https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html#evaluate_policy

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param env
    :param agent
    :param n_episodes
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    n_crashes = 0

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    n_episodes = np.sum(episode_count_targets)

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    pbar = tqdm(total=n_episodes)
    while (episode_counts < episode_count_targets).any():
        # eval_logger.log("Episode counts: {}".format(episode_counts))
        actions, states = agent.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if env.env_method("get_wrapper_attr", "vehicle", indices=[i])[0].crashed:
                        n_crashes += 1
                    pbar.update(1)

        observations = new_observations

        if render:
            env.render()

    # pbar.close()
    return np.mean(episode_rewards), np.mean(episode_lengths), n_crashes / n_episodes


if __name__ == "__main__":
    main()
