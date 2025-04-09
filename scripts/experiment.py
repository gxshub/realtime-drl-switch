"""
Usage:
  experiment <environment> <switch> [options]
  experiment -h | --help

Options:
  -h --help                Show this screen.
  -a --agent <file>        Agent file configuration
  -c --checkpoint <file>   DRL agent checkpoint path
  -p --pctrl-only          Use primary controllers only
  -s --sctrl <file>        Secondary controllers configuration
  -e --episodes <num>      Number of episodes [default: 4].
  -t --test                Do not save model or log in the test mode.
"""

import datetime
from pathlib import Path
from typing import TypeVar

from docopt import docopt
from stable_baselines3.common.logger import configure

from rt_drl_safeguard.controller import Controller
from rt_drl_safeguard.factory import load_agent_class, load_environment
from rt_drl_safeguard.switch import Switch
from rt_drl_safeguard.utils.configuration_manager import load_rule_based_ctrl_configs, load_switch_random_params
from rt_drl_safeguard.utils.randomization import DelayTimeDistribution

Agent = TypeVar("Agent")


def main():
    opts = docopt(__doc__)
    print(opts)
    env_config_file = opts['<environment>']
    switch_config_file = opts['<switch>']
    agent_config_file = opts['--agent']
    model_file = opts['--checkpoint']
    pri_ctrl_only = opts['--pctrl-only']
    sec_ctrl_config_file = opts['--sctrl']
    n_episodes = int(opts['--episodes'])
    test_mode = opts['--test']

    # Load environment
    environment, env_name = load_environment(env_config_file, training=False, n_envs=1)
    # Set up logger
    if test_mode:
        logger = configure(None, ['stdout'])
    else:
        version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_folder = Path(__file__).parent / "out" / env_name / "contr_infer_results" / version
        logger = configure(str(result_folder), ['stdout', 'log'])

    logger.log("Options: \n{}".format(opts),
               "\nEnvironment configuration:\n{}".format(open(env_config_file).read()))
    # Load agent (primary controller)
    if agent_config_file is None:
        agent = None
        logger.log("No agent is provided.")
    else:
        if model_file is None:
            raise ValueError("Checkpoint file must be provided for an agent model.")
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
    # Create switch and run
    switch_config_params, params_selection_probs = load_switch_random_params(switch_config_file)
    bins = [0, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0, 2.0]
    # delay_time_distribution = DelayTimeDistribution(bins)
    delay_time_distribution = DelayTimeDistribution.create_from_exp_dist(bins, scale=0.2)
    delay_time_distribution.print()
    switch = Switch(environment,
                    primary_controller,
                    secondary_controllers,
                    switch_config_params,
                    params_selection_probs,
                    delay_time_distribution)
    switch.run(n_episodes, pri_ctrl_only=pri_ctrl_only, logger=logger)


if __name__ == "__main__":
    main()
