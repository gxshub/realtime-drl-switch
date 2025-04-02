import importlib
import json
import logging
import os
from typing import Any, Callable, Dict, Optional, Type, Union

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from rt_drl_safeguard.highway_env_v2 import HighwayEnvV2

logger = logging.getLogger(__name__)


def agent_factory(environment, agent_config):
    """
            Handles creation of sb3 agents.

        :param environment: the environment
        :param agent_config: configuration of the agent, must contain a '__class__' key
        :return: a new agent
        """
    if "__class__" in agent_config:
        path = agent_config['__class__'].split("'")[1]
        module_name, class_name = path.rsplit(".", 1)
        agent_class = getattr(importlib.import_module(module_name), class_name)
        # assuming sb3 agent is loaded
        policy = agent_config["policy"]
        agent_config.pop("policy")
        agent_config.pop("__class__")
        if policy is None:
            raise ValueError("The configuration should specify 'policy'")
        noise_type = agent_config.get("noise_type")
        if noise_type is not None:
            noise_std = agent_config.get("noise_std", 0.1)
            n_actions = environment.action_space.shape[-1]
            if noise_type == "normal":
                agent_config["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                 sigma=noise_std * np.ones(n_actions))
            elif noise_type == "ornstein-uhlenbeck":
                agent_config["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
                )
            else:
                raise ValueError("Noise type not supported (use 'normal' or 'ornstein-uhlenbeck')")
        agent_config.pop("noise_type", None)
        agent_config.pop("noise_std", None)
        agent = agent_class(policy, environment, **agent_config)
        return agent
    else:
        raise ValueError("The configuration should specify the agent __class__")


def load_agent(agent_config, env):
    """
        Load an agent from a configuration file.

    :param agent_config: dict or the path to the agent configuration file
    :param env: the environment with which the agent interacts
    :return: the agent
    """
    # Load config from file
    if not isinstance(agent_config, dict):
        agent_config = load_agent_config(agent_config)
    return agent_factory(env, agent_config)


def load_agent_class(agent_config):
    if not isinstance(agent_config, dict):
        agent_config = load_agent_config(agent_config)
    if "__class__" in agent_config:
        path = agent_config['__class__'].split("'")[1]
        module_name, class_name = path.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), class_name)
    else:
        raise ValueError("The configuration should specify the agent __class__")


def load_agent_config(config_path):
    """
        Load an agent configuration from file, with inheritance.
    :param config_path: path to a json config file
    :return: the configuration dict
    """
    return _load_config(config_path)


def env_factory(env_config):
    """
        Load an environment from a configuration file.

    :param env_config: the configuration, or path to the environment configuration file
    :return: the environment
    """
    # Load the environment config from file
    if not isinstance(env_config, dict):
        env_config = _load_config(env_config)

    # Make the environment
    if env_config.get("import_module", None):
        __import__(env_config["import_module"])
    try:
        # env = gym.make(env_config['id'], render_mode='rgb_array')
        # Save env module in order to be able to import it again
        # env.import_module = env_config.get("import_module", None)
        """create highway_env_v2 instance directly"""
        env = HighwayEnvV2()
    except KeyError:
        raise ValueError("The gym register id of the environment must be provided")
    except gym.error.UnregisteredEnv:
        # The environment is unregistered.
        print("import_module", env_config["import_module"])
        raise gym.error.UnregisteredEnv('Environment {} not registered. The environment module should be specified by '
                                        'the "import_module" key of the environment configuration'.format(
            env_config['id']))

    # Configure the environment, if supported
    try:
        env.unwrapped.configure(env_config)
        # Reset the environment to ensure configuration is applied
        env.reset()
    except AttributeError as e:
        logger.info("This environment does not support configuration. {}".format(e))

    # if realtime:
    #    env = RealtimeHighway(env)
    return env


def vec_env_factory(
        # env_id: Union[str, Callable[..., gym.Env]],
        env_config: Union[str, Dict],
        n_envs: int = 1,
        seed: Optional[int] = None,
        start_index: int = 0,
        monitor_dir: Optional[str] = None,
        wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
        vec_env_kwargs: Optional[Dict[str, Any]] = None,
        monitor_kwargs: Optional[Dict[str, Any]] = None,
        wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_config: environment configuration
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """

    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            env = env_factory(env_config)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env


def load_environment(env_config, training: bool = True, n_envs: int = 1):
    """
        Load an environment from a configuration file.

    :param env_config: the configuration, or path to the environment configuration file
    :param training ..
    :param n_envs number of environments
    :return: the environment, name
    """
    if not isinstance(env_config, dict):
        env_config = _load_config(env_config)

    if n_envs == 1:
        return env_factory(env_config), env_config.get("name", None)
    elif n_envs > 1:
        return (vec_env_factory(env_config,
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv),
                env_config.get("name", None))
    else:
        raise Exception("negative value for the number of environments")


def _load_config(config):
    with open(config) as f:
        f_type = f.name.split(".")[-1]
        if f_type == "json":
            return json.loads(f.read())
        elif f_type == "yml" or f_type == "yaml":
            return yaml.unsafe_load(f.read())
        else:
            raise Exception("unsupported configuration file format")
