from typing import List, TypeVar, Union, Dict

import numpy as np
from tqdm import tqdm

from rt_drl_safeguard.controller import Controller
from rt_drl_safeguard.utils.randomization import DelayTimeDistribution, SwitchConfigurationSampler

RTEnv = TypeVar("RTEnv")

WARMUP_STEPS = 3
MIN_CONSEC_SEC_CTRL_CALLS = 4


class Switch:
    def __init__(self,
                 env: RTEnv,
                 primary_controller: Union[Controller, None],
                 secondary_controllers: Union[Dict[str, Controller], None],
                 params_config: List[Dict],
                 probabilities: Union[np.ndarray, List, None],
                 delay_time: Union[DelayTimeDistribution, float]):

        if primary_controller is None and secondary_controllers is None:
            raise ValueError("At least one controller must be available.")

        if len(params_config) == 0:
            raise ValueError("Configurations must not be empty")
        elif probabilities is not None and len(probabilities) != len(params_config):
            raise ValueError("Configurations and probabilities must have the same dimension.")
        elif probabilities is None:
            probabilities = np.full(len(params_config), 1.0 / len(params_config))

        self.env = env
        self.params_sampler = SwitchConfigurationSampler(params_config, probabilities)
        self.primary_controller = primary_controller
        self.secondary_controllers = secondary_controllers
        self.delay_time = delay_time

    def run(self, num_episodes: int, pri_ctrl_only=False, logger=None):

        if logger is not None:
            logger.log("---Run Switch---")

        deterministic_config_in_use = True if self.params_sampler.size == 1 else False

        ep_rwrds = []
        ep_lengths = []
        num_episodes_actual = 0
        num_crashes = 0

        data_tran_freq = {"continue_with_sec_ctrl": 0, "can_switch_to_pri_ctrl": 0}

        params = self.params_sampler.get()

        for _ in tqdm(range(num_episodes)):

            if self.params_sampler.size > 0:
                params = self.params_sampler.sample()
            time_delay_threshold = float(params["thresh"]["time_delay"])
            sec_ctrl_id = params["ctrl_id"]
            cross_lane_headway_threshold = float(params["thresh"]["cross_lane_headway"])

            timestep = 0
            tot_rwrd = 0
            done = truncated = False
            obs, info = self.env.reset()

            using_sec_ctrl = False
            consec_sec_ctrl_calls = 0
            can_switch_to_pri_ctrl = True

            # Warm-up phase with primary controller
            for _ in range(WARMUP_STEPS):
                if done or truncated:
                    break
                action = self.primary_controller.act(obs)
                obs, _, done, truncated, _ = self.env.step(action)

            while not (done or truncated):

                if type(self.delay_time) is float:
                    elapsed_time = self.delay_time
                else:
                    elapsed_time = self.delay_time.sample()

                # If running the primary controller only
                if pri_ctrl_only:
                    action = self.primary_controller.act(obs)
                    self.env.elapse(elapsed_time)
                    obs, reward, done, truncated, info = self.env.step(action)
                else:
                    if can_switch_to_pri_ctrl and elapsed_time < time_delay_threshold:
                        action = self.primary_controller.act(obs)
                    else:
                        action = self.secondary_controllers[sec_ctrl_id].act(obs)
                        using_sec_ctrl = True
                        consec_sec_ctrl_calls += 1
                    self.env.elapse(elapsed_time)
                    obs, reward, done, truncated, info = self.env.step(action)

                    # Check if we can switch to the primary controller
                    #   after running the secondary controller for at least 4 steps
                    if using_sec_ctrl and consec_sec_ctrl_calls > MIN_CONSEC_SEC_CTRL_CALLS:
                        if self._cross_lane_headway > cross_lane_headway_threshold:
                            can_switch_to_pri_ctrl = True
                            consec_sec_ctrl_calls = 0
                            data_tran_freq["can_switch_to_pri_ctrl"] += 1
                        else:
                            data_tran_freq["continue_with_sec_ctrl"] += 1

                timestep += 1
                tot_rwrd += reward
                if logger is not None:
                    logger.log("[data] headway: {:.4f}, cross_lane_headway: {:4f}, ttc: {:.4f}, on lane: {}" \
                               .format(self._headway, self._cross_lane_headway, self._ttc, self._on_lane))

                if self.env.unwrapped.vehicle.crashed:
                    num_crashes += 1

            ep_rwrds.append(tot_rwrd)
            ep_lengths.append(timestep)
            num_episodes_actual += 1

        if logger is not None:
            logger.log("---Episode Summary---")
            logger.log("Total timestep count: {}".format(np.sum(ep_lengths)))
            logger.log("Episode count: {}".format(num_episodes_actual))
            logger.log("Average episode reward: {}, scaled: {}" \
                       .format(np.sum(ep_rwrds) / num_episodes_actual,
                               (np.sum(ep_rwrds) / num_episodes_actual) * 40 / (40 - WARMUP_STEPS)))
            logger.log("Crash rate (episode): {:.2%}".format(num_crashes / num_episodes_actual))
            logger.log("Crash rate (timestep): {:.4%}".format(num_crashes / np.sum(ep_lengths)))
            if type(self.delay_time) is not float:
                logger.log("[probabilities] - switch to pri or sec ctrls - primary: {}, secondary: {}" \
                           .format(
                    self.delay_time.cumulative_probability_below(float(params["thresh"]["time_delay"])),
                    self.delay_time.cumulative_probability_above(float(params["thresh"]["time_delay"]))))
            if not pri_ctrl_only and deterministic_config_in_use:
                logger.log("[counts] - exit sec ctrl after {} consec calls - yes: {}, no: {}" \
                           .format(MIN_CONSEC_SEC_CTRL_CALLS,
                                   data_tran_freq["can_switch_to_pri_ctrl"],
                                   data_tran_freq["continue_with_sec_ctrl"]))

    @property
    def _cross_lane_headway(self):
        ego_vehicle = self.env.vehicle
        ego_speed = ego_vehicle.speed
        headway = np.inf
        for other in self.env.road.vehicles:
            if other is not ego_vehicle:
                margin = other.LENGTH / 2 + ego_vehicle.LENGTH / 2
                on_lane_distance = ego_vehicle.lane_distance_to(other)
                if on_lane_distance >= 0 and ego_speed > 0:
                    headway_tmp = (on_lane_distance - margin) / ego_vehicle.speed
                    headway = min(headway, headway_tmp)
                elif on_lane_distance < 0 and other.speed > 0:
                    headway_tmp = (abs(on_lane_distance) - margin) / other.speed
                    headway = min(headway, headway_tmp)
        return headway

    @property
    def _headway(self):
        ego_vehicle = self.env.vehicle
        ego_speed = ego_vehicle.speed
        headway = np.inf
        for other in self.env.road.vehicles:
            if other.lane_index[2] == ego_vehicle.lane_index[2] and other is not ego_vehicle:
                margin = other.LENGTH / 2 + ego_vehicle.LENGTH / 2
                on_lane_distance = ego_vehicle.lane_distance_to(other)
                if on_lane_distance > 0 and ego_speed > 0:
                    headway_tmp = (on_lane_distance - margin) / ego_vehicle.speed
                    headway = min(headway, headway_tmp)
                elif on_lane_distance < 0 and other.speed > 0:
                    headway_tmp = (abs(on_lane_distance) - margin) / other.speed
                    headway = min(headway, headway_tmp)
        return headway

    @property
    def _ttc(self):
        ego_vehicle = self.env.vehicle
        ego_speed = ego_vehicle.speed
        ttc = np.inf
        for other in self.env.road.vehicles:
            if other.lane_index[2] == ego_vehicle.lane_index[2] and other is not ego_vehicle:
                margin = other.LENGTH / 2 + ego_vehicle.LENGTH / 2
                on_lane_distance = ego_vehicle.lane_distance_to(other)
                relative_speed = ego_speed - other.speed
                if on_lane_distance > 0 and relative_speed > 0:
                    ttc_tmp = (on_lane_distance - margin) / relative_speed
                    ttc = min(ttc, ttc_tmp)
                elif on_lane_distance < 0 and relative_speed < 0:
                    ttc_tmp = (abs(on_lane_distance) - margin) / abs(relative_speed)
                    ttc = min(ttc, ttc_tmp)
        return ttc

    @property
    def _on_lane(self):
        ego_vehicle = self.env.vehicle
        return ego_vehicle.on_road
