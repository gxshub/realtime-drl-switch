from pathlib import Path

from rt_drl_switch.factory import load_agent_class, load_environment
from rt_drl_switch.rule_based_controller import TtcBasedController
from rt_drl_switch.utils.configuration_manager import load_rule_based_ctrl_configs, load_rule_based_ctrl_config


class Controller:
    def __init__(self, model, model_type=None, logger=None):
        self.model = model
        self.type = model_type
        self.logger = logger

    @staticmethod
    def load_drl_agent(agent_config_file, checkpoint_file):
        model = load_agent_class(agent_config_file).load(Path(checkpoint_file))
        controller = Controller(model, model_type='drl')
        return controller

    @staticmethod
    def load_rule_based_controller(env, ctrl_config_file, ctrl_name):
        configs_dict = load_rule_based_ctrl_configs(ctrl_config_file)
        cfg = configs_dict[ctrl_name]
        model = TtcBasedController(env, cfg)
        return Controller(model, model_type='rule_based')

    @staticmethod
    def load_rule_based_controllers(env, configs):
        controllers = {}
        for ctrl_name in configs:
            cfg = configs[ctrl_name]
            model = TtcBasedController(env, cfg)
            controllers[ctrl_name] = Controller(model, model_type='rule_based')
        return controllers

    def act(self, obs):
        if self.type == 'drl':
            action, _ = self.model.predict(obs, deterministic=True)
        elif self.type == 'rule_based':
            action = self.model.control()
        else:
            action = None

        return action
