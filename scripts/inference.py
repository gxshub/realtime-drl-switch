"""
Usage:
  inference <environment> <agent> <checkpoint> [options]
  inference <environment> [options]
  inference -h | --help

Options:
  -h --help              Show this screen.
  --episodes <count>     Number of episodes [default: 4].
  --processes <count>    Number of running processes [default: 1].
  --rtc-seed <time>      Random seed for real-time computing. If 0, not real-time. [default: 0]
  --random-delay         Randomised delay
  --safeguard            Whether the system is safeguarded or not.
  --controller-only      Use controller only
  --delay-tol <float>    Delay tolerance for controller [default: 0.2]
  --no-display           Disable environment, agent, and rewards rendering.
  --test                 Do not save model or log in the test mode.
  --verbose
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

from rt_drl_switch.factory import load_agent_class, load_environment
from rt_drl_switch.highway_safeguard import CON_PARAMS, CTRL_PARAMS, TtcBasedController
from rt_drl_switch.utils.randomization import exp_delay

Agent = TypeVar("Agent")
DEFAULT_SAFE_TTC = 2  # s


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
    safeguard = opts['--safeguard'] if not opts['--controller-only'] else False
    n_proc = int(opts['--processes'])
    controller_only = opts['--controller-only']
    delay_tol = float(opts['--delay-tol'])
    test_mode = opts['--test']
    verbose = opts['--verbose']

    version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # set up logger output
    if test_mode:
        logger = configure(None, ['stdout'])
    elif controller_only:
        _, _env_name = load_environment(env_config_file)
        result_folder = Path(__file__).parent / "out" / _env_name / "contr_infer_results" / version
        logger = configure(str(result_folder), ['stdout', 'log'])
    else:
        result_folder = Path(model_file).parent / "infer_results" / version
        logger = configure(str(result_folder), ['stdout', 'log'])

    logger.log("Options:\n{}".format(opts),
               "\nEnvironment configuration:\n{}".format(open(env_config_file).read()))

    if controller_only:
        agent = None
        logger.log("--Controller Only--")
    else:
        agent = load_agent_class(agent_config_file).load(Path(model_file))
        logger.log("Agent configuration:\n{}".format(open(agent_config_file).read()),
                   "\nLoad checkpoint file from: {}\n".format(model_file))

    if controller_only or safeguard:
        logger.log("controller parameters: {}".format(CTRL_PARAMS),
                   "\naction convertor parameters: {}".format(CON_PARAMS))

    if n_proc == 1:
        env, _ = load_environment(env_config_file, training=False, n_envs=1)
        # env = RealtimeHighway(env)
        controller = TtcBasedController(env)
        infer(env, agent, controller, n_episodes,
              rtc_seed=rtc_seed,
              random_delay=random_delay,
              safeguard=safeguard,
              controller_only=controller_only,
              delay_tol=delay_tol,
              logger=logger,
              verbose=verbose)
    else:
        # TODO
        env, _ = load_environment(env_config_file, training=False, n_envs=n_proc)
        infer_vec(env, agent, n_episodes, rtc_seed,
                  random_delay=random_delay, safeguard=safeguard)


def infer(env,
          agent,
          controller,
          n_episodes: int,
          rtc_seed: float = 0,
          random_delay: bool = False,
          safeguard: bool = False,
          controller_only: bool = False,
          delay_tol: float = np.inf,
          logger=None,
          verbose=False):
    episode_rewards = []
    episode_rewards_agent = []
    episode_rewards_controller = []
    episode_lengths = []
    n_crashes = 0
    with_delay = True if rtc_seed > 0 else False
    n_agent_controls_steps = 0
    n_on_lane = 0
    safe_ttc = DEFAULT_SAFE_TTC
    n_small_ttc = 0
    road_info = None
    for _ in tqdm(range(n_episodes)):
        reward_acc: float = 0
        timestep = 0
        done: bool
        done = truncated = False
        obs, info = env.reset()
        delay: float = 0
        overridden_by_controller = False
        action = None
        while not (done or truncated):
            # reset control
            if overridden_by_controller and controller.switchable:
                overridden_by_controller = False
            if with_delay:
                # randomized delay
                if random_delay:
                    delay = exp_delay(rtc_seed)
                else:
                    delay = rtc_seed
            # note. "timestep >= n" ensures agent controls at least n steps initially
            if not controller_only and safeguard and delay > delay_tol and timestep >= 3:
                overridden_by_controller = True
                # if controller.target_lane_index is None:
                #    controller.set_target_lane_index(env.unwrapped.vehicle.lane_index)
            if controller_only or overridden_by_controller:
                controller.target_lane_index = env.unwrapped.vehicle.lane_index
                action = controller.control()
            else:
                action, _ = agent.predict(obs, deterministic=True)
                n_agent_controls_steps += 1
            road_info = _nearby_vehicles_info(env, all_lanes=True)
            if controller_only or overridden_by_controller:
                env.elapse(delay, reset_steering=True)
            else:
                env.elapse(delay, reset_steering=False)
            obs, reward, done, truncated, info = env.step(action)
            timestep += 1
            if controller_only or overridden_by_controller:
                episode_rewards_controller.append(reward)
            else:
                episode_rewards_agent.append(reward)
            if info['rewards']['on_road_reward'] > 0:
                n_on_lane += 1
            i = road_info['ego']['lane index']
            if verbose:
                logger.log("[running info] timestep: {}, controlled by agent: {}, delay: {}, delay_tol: {}\n, " \
                           .format(timestep, not overridden_by_controller and not controller_only, delay, delay_tol),
                           "[running info] timestep: {}, lane: {}, forward speed: {}, position: {}, heading: {}, action: {}, " \
                           .format(timestep,
                                   road_info['ego']['lane index'],
                                   road_info['ego']['forward speed'],
                                   env.unwrapped.vehicle.position,
                                   env.unwrapped.vehicle.heading,
                                   action),
                           "front ttc: {}, ttc back: {}" \
                           .format(road_info[i]['front']['ttc'], road_info[i]['back']['ttc']))
                logger.log("[running info] timestep: {}, reward: {}, details: {}" \
                           .format(timestep, reward, info['rewards']))
            # count small ttc/s for statistics purposes
            if 0 <= road_info[i]['front']['ttc'] < safe_ttc or \
                    0 <= road_info[i]['back']['ttc'] < safe_ttc:
                n_small_ttc += 1
            reward_acc += reward
        episode_rewards.append(reward_acc)
        episode_lengths.append(timestep)
        if env.unwrapped.vehicle.crashed:
            n_crashes += 1
            logger.log("!!!crashed at timestep: {}\n".format(timestep),
                       "controlled_by_agent: {}\n".format(not (controller_only or overridden_by_controller)),
                       "road info before delay: {}\n".format(road_info),
                       "delay (s): {}\n".format(delay),
                       "action [acceleration, steering]: {}\n".format(action),
                       "crashed vehicles: {}".format(get_crashed_vehicle(env)))
    total_timesteps = np.sum(episode_lengths)
    logger.log("---------------------\nTotal number of episodes: {}".format(n_episodes))
    logger.log("Average episode reward: {}".format(np.mean(episode_rewards)))
    logger.log("Average episode Length: {}".format(np.mean(episode_lengths)))
    logger.log("Crash rate: {:.2%}".format(n_crashes / n_episodes))
    logger.log("Agent-controlled timesteps: {:.2%}".format(n_agent_controls_steps / total_timesteps))
    logger.log("On-lane: {:.2%}".format(n_on_lane / total_timesteps))
    logger.log("Small TTC (i.e. <{}s): {:.2%}".format(safe_ttc, n_small_ttc / total_timesteps))
    logger.log("Average timestep reward by agent: {}".format(np.nanmean(episode_rewards_agent)))
    logger.log("Average timestep reward by controller: {}".format(np.nanmean(episode_rewards_controller)))
    return


def _nearby_vehicles_info(env, all_lanes=False):
    ego_vehicle = env.unwrapped.vehicle
    ego_speed = ego_vehicle.speed * np.cos(ego_vehicle.heading)
    status = {'ego': {'forward speed': round(ego_speed, 2),
                      'lane index': ego_vehicle.lane_index[2],
                      'heading': ego_vehicle.heading}
              }
    if all_lanes:
        num_roads = len(env.unwrapped.road.network.all_side_lanes(ego_vehicle.lane_index))
        lane_idx = range(num_roads)
    else:
        lane_idx = [ego_vehicle.lane_index[2]]
    for i in lane_idx:
        ttc = np.inf
        on_lane_distance = np.inf
        lateral_distance = np.inf
        relative_speed = 0
        is_crashed = False
        vid = None
        ttc_r = np.inf
        on_lane_distance_r = -np.inf
        lateral_distance_r = np.inf
        relative_speed_r = 0
        is_crashed_r = False
        id_r = None
        for other in env.unwrapped.road.vehicles:
            if other is not ego_vehicle and other.lane_index[2] == i:
                margin = other.LENGTH / 2 + ego_vehicle.LENGTH / 2
                on_lane_distance_x = ego_vehicle.lane_distance_to(other) - margin
                lat_margin = other.WIDTH / 2 + ego_vehicle.WIDTH / 2
                # relative_speed_x = ego_speed - other.speed
                relative_speed_x = ego_speed - other.speed * np.dot(
                    other.direction, ego_vehicle.direction
                )
                ttc_x = on_lane_distance_x / relative_speed_x
                if 0 < on_lane_distance_x < on_lane_distance:
                    ttc = ttc_x
                    on_lane_distance = on_lane_distance_x
                    lateral_distance = np.abs(ego_vehicle.position[1] - other.position[1]) - lat_margin
                    relative_speed = relative_speed_x
                    if other.crashed:
                        is_crashed = True
                    vid = env.unwrapped.road.vehicles.index(other)
                if on_lane_distance_r < on_lane_distance_x < 0:
                    ttc_r = ttc_x
                    on_lane_distance_r = on_lane_distance_x
                    lateral_distance_r = np.abs(ego_vehicle.position[1] - other.position[1]) - lat_margin
                    relative_speed_r = relative_speed_x
                    if other.crashed:
                        is_crashed_r = True
                    id_r = env.unwrapped.road.vehicles.index(other)
        status[i] = {
            'front': {'vid': vid,
                      'dist': round(on_lane_distance, 1),
                      'ttc': round(ttc, 1),
                      'rel spd': round(relative_speed, 1),
                      'lat dist': round(lateral_distance, 1),
                      'crashed': is_crashed},
            'back': {'vid': id_r,
                     'dist': round(on_lane_distance_r, 1),
                     'ttc': round(ttc_r, 1),
                     'rel spd': round(relative_speed_r, 1),
                     'lat dist': round(lateral_distance_r, 1),
                     'crashed': is_crashed_r}
        }
    return status


def get_crashed_vehicle(env):
    vehicles = env.unwrapped.road.vehicles
    return [vehicles.index(x) for x in vehicles if x.crashed]


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
    # sgs = [HighwayAgentSafeguard(env) for _ in range(n_envs)]
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
                env.env_method("elapse", delays[i], indices=[i])

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
    return


if __name__ == "__main__":
    main()
