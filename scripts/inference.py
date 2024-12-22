"""
Usage:
  infer <environment> <agent> <checkpoint> [options]
  infer -h | --help

Options:
  -h --help              Show this screen.
  --episodes <count>     Number of episodes [default: 4].
  --processes <count>    Number of running processes [default: 1].
  --rtc-seed <time>      Random seed for real-time computing. If 0, not real-time. [default: 0]
  --random-delay         Randomised delay
  --safeguard            Whether the system is safeguarded or not.
  --no-display           Disable environment, agent, and rewards rendering.
  --test                 Do not save model or log in the test mode.
"""

import datetime
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

import numpy as np
from docopt import docopt
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from tqdm import tqdm

from rt_drl_safeguard.safeguard.highway_safeguard import TtcBasedController, DEFAULT_DELAY_TORRENCE
from rt_drl_safeguard.safeguard.highway_safeguard import HighwayAgentSafeguard
from rt_drl_safeguard.utils.factory import load_agent_class, load_environment
from rt_drl_safeguard.utils.highway_env_wrapper import RealtimeHighway
from rt_drl_safeguard.utils.randomization import exp_delay

Agent = TypeVar("Agent")
OUTPUT_FOLDER = "infer_results"


def main():
    opts = docopt(__doc__)
    env_config_file = opts['<environment>']
    agent_config_file = opts['<agent>']
    model_file = opts['<checkpoint>']
    n_episodes = int(opts['--episodes'])
    rtc_seed = float(opts['--rtc-seed'])
    if rtc_seed < 0:
        raise ValueError("rtc seed must be a non-negative value")
    random_delay = opts['--random-delay']
    safeguard = opts['--safeguard']
    n_proc = int(opts['--processes'])
    test_mode = opts['--test']

    version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test_result_folder = Path(model_file).parent / OUTPUT_FOLDER / version

    if test_mode:
        logger = configure(None, ['stdout'])
    else:
        logger = configure(str(test_result_folder), ['stdout', 'log'])

    logger.log("Options:\n{}".format(opts),
               "\nEnvironment configuration:\n{}".format(open(env_config_file).read()),
               "\nAgent configuration:\n{}".format(open(agent_config_file).read()),
               "\nLoad checkpoint file from: {}\n".format(model_file))

    agent = load_agent_class(agent_config_file).load(Path(model_file))

    if n_proc == 1:
        env, _ = load_environment(env_config_file, training=False, n_envs=n_proc)
        env = RealtimeHighway(env)
        avg_epi_rew, avg_epi_len, crash_rate = infer(env, agent, n_episodes, rtc_seed,
                                                     random_delay=random_delay, safeguard=safeguard)
    else:
        env, _ = load_environment(env_config_file, training=False, n_envs=n_proc, realtime=True)
        avg_epi_rew, avg_epi_len, crash_rate = infer_vec(env, agent, n_episodes, rtc_seed,
                                                         random_delay=random_delay, safeguard=safeguard)

    logger.log("----------------------\nNumber of episodes: {}".format(n_episodes),
               "\nAverage episode accumulated reward: {}".format(avg_epi_rew),
               "\nAverage episode length: {}".format(avg_epi_len),
               "\nCrash rate: {}".format(crash_rate))


def infer_vec(env,
              agent: "Agent",
              n_episodes: int,
              rtc_seed: float = 0,
              random_delay: bool = False,
              safeguard: bool = False,
              deterministic: bool = True,
              render: bool = False,
              callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
              warn: bool = True,
              ):
    """
        Inference in a real-time environment.

    This function is based on evaluate_policy() in sb3.
    https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html#evaluate_policy

    note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param env
    :param agent
    :param n_episodes: Number of episodes
    :param rtc_seed: randomization seed for real-time computing
    :param deterministic: Whether to use deterministic or stochastic actions
    :param safeguard
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
    with_delay = True if rtc_seed > 0 else False
    #sgs = [HighwayAgentSafeguard(env) for _ in range(n_envs)]
    while (episode_counts < episode_count_targets).any():
        # eval_logger.log("Episode counts: {}".format(episode_counts))
        actions, states = agent.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )

        if with_delay:
            if random_delay:
                # sample n_envs delays
                delays = exp_delay(rtc_seed, n_envs)
            else:
                delays = np.array([rtc_seed] * n_envs)
            for i in range(n_envs):
                #if safeguard:
                #    actions[i], delays[i], info = sgs[i].assure(actions[i], delays[i])
                #    print(info)
                env.env_method("set_delay", delays[i], indices=[i])
                # print("delay in env {}: {}".format(i, env.env_method("get_wrapper_attr", "delay", indices=[i])))

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
                    if info['crashed']:
                        n_crashes += 1
                    pbar.update(1)

        observations = new_observations

        if render:
            env.render()

    # pbar.close()
    return np.mean(episode_rewards), np.mean(episode_lengths), n_crashes / n_episodes


def infer(env,
          agent: "Agent",
          n_episodes: int,
          rtc_seed: float = 0,
          random_delay: bool = False,
          safeguard: bool = False):
    sg = HighwayAgentSafeguard(env)
    sec_controller = TtcBasedController(env)
    episode_rewards = []
    episode_lengths = []
    n_crashes = 0
    with_delay = True if rtc_seed > 0 else False
    n_agent_controls_steps = 0
    n_total_timesteps = 0
    road_info = None
    for _ in tqdm(range(n_episodes)):
        n_total_timesteps += 1
        reward_acc = 0.
        timestep = 0
        done = truncated = False
        obs, info = env.reset()
        delay = 0
        controlled_by_agent = True
        action = None
        while not (done or truncated):

            if not controlled_by_agent and sec_controller.switchable:
                controlled_by_agent = True
            if with_delay:
                # randomize delay
                if random_delay:
                    delay = exp_delay(rtc_seed)
                else:
                    delay = rtc_seed
            # print("sim delay:", delay)
            if controlled_by_agent and safeguard and delay > DEFAULT_DELAY_TORRENCE:
                controlled_by_agent = False
            if controlled_by_agent:
                action, _ = agent.predict(obs, deterministic=True)
                n_agent_controls_steps += 1
            else:
                delay = np.min([delay, DEFAULT_DELAY_TORRENCE])
                action = sec_controller.control()
            road_info = get_road_info(env)
            # print("action: {}, delay: {}".format(action, delay))
            env.delay = delay
            obs, reward, done, truncated, info = env.step(action)
            reward_acc += reward
            timestep += 1
        episode_rewards.append(reward_acc)
        episode_lengths.append(timestep)
        crashed = env.unwrapped.vehicle.crashed
        if crashed:
            n_crashes += 1
        print("!!!crashed: ", crashed)
        if crashed:
            print("controlled_by_agent: ", controlled_by_agent)
            print("action: ", action)
            print("before crash: " , road_info)
            print("crashed vehicle id: ", get_crashed_vehicle(env))
    print("proportion of timesteps controlled by agents: {} ".format(n_agent_controls_steps / n_total_timesteps))
    return np.mean(episode_rewards), np.mean(episode_lengths), n_crashes / n_episodes


def get_road_info(env):
    ego_vehicle = env.unwrapped.vehicle
    ego_speed = ego_vehicle.speed
    status = [{'speed': round(ego_speed,2), 'lane index': ego_vehicle.lane_index[2]}]
    num_roads = len(env.unwrapped.road.network.all_side_lanes(ego_vehicle.lane_index))
    for i in range(num_roads):
        ttc = np.inf
        on_lane_distance = np.inf
        relative_speed = 0
        is_crashed = False
        id = None
        ttc_r = np.inf
        on_lane_distance_r = -np.inf
        relative_speed_r = 0
        is_crashed_r = False
        id_r = None
        for other in env.unwrapped.road.vehicles:
            if other is not ego_vehicle and other.lane_index[2] == i:
            # if other is not ego_vehicle and other.lane_index == ego_vehicle.lane_index:
                margin = other.LENGTH / 2 + ego_vehicle.LENGTH / 2
                on_lane_distance_x = ego_vehicle.lane_distance_to(other) - margin
                # relative_speed_x = ego_speed - other.speed
                relative_speed_x = ego_speed - other.speed * np.dot(
                    other.direction, ego_vehicle.direction
                )
                ttc_x = on_lane_distance_x / relative_speed_x
                if 0 < on_lane_distance_x < on_lane_distance:
                    ttc = ttc_x
                    on_lane_distance = on_lane_distance_x
                    relative_speed = relative_speed_x
                    if other.crashed:
                        is_crashed = True
                    id = env.unwrapped.road.vehicles.index(other)
                if on_lane_distance_r < on_lane_distance_x < 0:
                    ttc_r = ttc_x
                    on_lane_distance_r = on_lane_distance_x
                    relative_speed_r = relative_speed_x
                    if other.crashed:
                        is_crashed_r = True
                    id_r = env.unwrapped.road.vehicles.index(other)

        status.extend([{'dist (f)': round(on_lane_distance,1),
                       'ttc (f)': round(ttc,1),
                       'rel spd (f)': round(relative_speed,1),
                       'crashed (f)': is_crashed,
                       'VID': id,
                       'lane': i},
                      {'dist (b)': round(on_lane_distance_r,1),
                       'ttc (b)': round(ttc_r,1),
                       'rel spd (b)': round(relative_speed_r,1),
                       'crashed (b)': is_crashed_r,
                       'VID': id_r,
                       'lane': i}])
    return status

def get_crashed_vehicle(env):
    vehicles = env.unwrapped.road.vehicles
    return [vehicles.index(x) for x in vehicles if x.crashed]


if __name__ == "__main__":
    main()
