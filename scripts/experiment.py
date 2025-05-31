"""
Usage:
  experiment <environment> [options]
  experiment -h | --help

Options:
  -h --help                Show this screen.
  -a --agent <file>        Agent configuration
  -c --checkpoint <file>   DRL agent checkpoint path
  -p --pctrl-only          Use primary controllers only
  -w --switch <file>       Switch configuration
  -s --sctrl <file>        Secondary controllers configuration
  -r --random-delay <file> Parameters for random delay time
  -d --deterministic-delay <value>  Deterministic delay time
  -e --episodes <num>      Number of episodes [default: 4].
  -o --output-dir <str>    Output (sub) direction
  -t --test                Do not save model or log in the test mode.
"""

import datetime
from pathlib import Path
from typing import TypeVar

from docopt import docopt
from stable_baselines3.common.logger import configure

from rt_drl_switch.controller import Controller
from rt_drl_switch.factory import load_agent_class, load_environment
from rt_drl_switch.switch import Switch
from rt_drl_switch.utils.configuration_manager import load_rule_based_ctrl_configs, load_switch_random_params, \
    load_config
from rt_drl_switch.utils.randomization import DelayTimeDistribution

Agent = TypeVar("Agent")

DEFAULT_TIME_INTERVALS = [0.0, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0, 2.0]
DEFAULT_SWITCH_CONFIG_FILE = str(Path(__file__).parent / "configs/HighwayEnv/switch/switch_dc0.yaml")

def main():
    opts = docopt(__doc__)
    print(opts)
    env_config_file = opts['<environment>']
    agent_config_file = opts['--agent']
    model_file = opts['--checkpoint']
    pri_ctrl_only = opts['--pctrl-only']
    switch_config_file = opts['--switch'] if opts['--switch'] else DEFAULT_SWITCH_CONFIG_FILE
    sec_ctrl_config_file = opts['--sctrl']
    rand_delay_time_param_file = opts['--random-delay']
    det_delay_time = float(opts['--deterministic-delay']) if opts['--deterministic-delay'] else None
    n_episodes = int(opts['--episodes'])
    dir_name = opts['--output-dir'] if opts['--output-dir'] else 'default'
    test_mode = opts['--test']

    # Load environment
    environment, env_name = load_environment(env_config_file, training=False, n_envs=1)
    # Set up logger
    if test_mode:
        logger = configure(None, ['stdout'])
    else:
        version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_folder = Path(__file__).parent / "out" / env_name / "experiment" / dir_name / version
        logger = configure(str(result_folder), ['stdout', 'log'])

    print("logger dir: ", logger.dir)

    logger.log("Options: \n{}".format(opts),
               "\nEnvironment configuration:\n{}".format(open(env_config_file).read()))

    # Load agent (primary controller)
    if agent_config_file is None:
        agent = None
        logger.log("No agent is provided.")
    else:
        if model_file is None:
            raise ValueError("Checkpoint file must be provided for an agent.")
        else:
            agent = load_agent_class(agent_config_file).load(Path(model_file))
            logger.log("Agent configuration:\n{}".format(open(agent_config_file).read()),
                       "\nLoad checkpoint file from: {}\n".format(model_file))
    primary_controller = Controller(agent, model_type='drl')

    # Create secondary controllers
    if sec_ctrl_config_file is None:
        secondary_controllers = None
    else:
        sec_ctrl_configs = load_rule_based_ctrl_configs(sec_ctrl_config_file)
        secondary_controllers = Controller.load_rule_based_controllers(environment, sec_ctrl_configs)
        logger.log("Secondary controller configurations: \n{}".format(sec_ctrl_configs))

    # Create time delay randomizer
    if det_delay_time is not None:
        delay_time = det_delay_time
        logger.log("Deterministic delay time: {}".format(delay_time))
    elif rand_delay_time_param_file is not None:
        # todo
        time_intervals = load_config(rand_delay_time_param_file)
        delay_time = DelayTimeDistribution.create_from_exp_dist(time_intervals, scale=0.2)
        logger.log(delay_time.get_distribution())
    else:
        time_intervals = DEFAULT_TIME_INTERVALS
        delay_time = DelayTimeDistribution.create_from_exp_dist(time_intervals, scale=0.2)
        logger.log(delay_time.get_distribution())

    # Create and run switch
    switch_config_params, params_selection_probs = load_switch_random_params(switch_config_file)
    logger.log("Switch configurations: \n{}".format(switch_config_params),
               "\nSwitch selection probabilities: {}".format(params_selection_probs))
    switch = Switch(environment,
                    primary_controller,
                    secondary_controllers,
                    switch_config_params,
                    params_selection_probs,
                    delay_time)
    switch.run(n_episodes, pri_ctrl_only=pri_ctrl_only, logger=logger)


if __name__ == "__main__":
    main()
