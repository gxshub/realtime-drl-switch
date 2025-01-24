from typing import TypeVar

import numpy as np
from highway_env import utils

from rt_drl_safeguard.safeguard.action_conversion import ActionConvertor
from rt_drl_safeguard.safeguard.meta_control_data import *

RTEnv = TypeVar("RTEnv")
Action = TypeVar("Action")

DEFAULT_DELAY_TOLERANCE = 0.5  # s
ACC_TIME_THD = 5.0  # s
SW_TIME_THD = 5.0  # s


class TtcBasedController:
    """A controller for safeguard directly based on ttc"""

    def __init__(
            self,
            env: RTEnv,
            delay_tolerance=DEFAULT_DELAY_TOLERANCE):
        self.env = env.unwrapped
        self.num_actions = SPEED_LEVELS
        self.delta_speed = DELTA_SPEED
        self.converter = ActionConvertor(env)
        self.delay_tolerance = delay_tolerance

    def compute_ttc(self, target_speed, target_lane_index):
        ego_vehicle = self.env.vehicle
        ego_speed = target_speed
        ttc = np.inf
        for other in self.env.road.vehicles:
            if other is not ego_vehicle and other.lane_index == target_lane_index:
                margin = other.LENGTH / 2 + ego_vehicle.LENGTH / 2
                if ego_vehicle.lane_distance_to(other) > 0:
                    on_lane_distance_x = ego_vehicle.lane_distance_to(other) - margin
                else:
                    on_lane_distance_x = ego_vehicle.lane_distance_to(other) + margin
                # relative_speed_x = ego_speed - other.speed
                relative_speed_x = ego_speed - other.speed * np.dot(
                    other.direction, ego_vehicle.direction
                )
                ttc_x = on_lane_distance_x / relative_speed_x
                if 0 < ttc_x < ttc:
                    ttc = ttc_x
        return ttc

    def control(self, acc_time_threshold=ACC_TIME_THD) -> np.ndarray:
        acc_time_threshold += self.delay_tolerance
        ego_lane_index = self.env.vehicle.lane_index
        # e.g., 0.5 indicates a speed change of +0.5*DELTA_SPEED
        # -2.0 indicates a speed change of -2.0*DELTA_SPEED
        speed_changes_coeffs = [0.1, 0.05, 0, -0.1, -0.25, -0.5, -1.0, -2.0, -3.0]
        for x in speed_changes_coeffs:
            target_speed = self.env.vehicle.speed + x * DELTA_SPEED
            if self.compute_ttc(target_speed, ego_lane_index) >= acc_time_threshold:
                return self._low_level_control(target_speed, ego_lane_index)
        # default
        return self._low_level_control(self.env.vehicle.speed, ego_lane_index)

    @property
    def switchable(self, delta_speed_x=1.2 * DELTA_SPEED) -> bool:
        switchable = True
        max_switchable_ttc = SW_TIME_THD + self.delay_tolerance
        ego_vehicle = self.env.vehicle
        ego_speed = ego_vehicle.speed
        for lane_index in self.env.road.network.all_side_lanes(ego_vehicle.lane_index):
            if 0 <= self.compute_ttc(ego_speed - delta_speed_x, lane_index) <= max_switchable_ttc or \
                    0 <= self.compute_ttc(ego_speed + delta_speed_x, lane_index) <= max_switchable_ttc:
                switchable = False
        # for lane_index in self.env.road.network.all_side_lanes(ego_vehicle.lane_index):
        #    if 0 <= self.compute_ttc(ego_speed + delta_speed_x, lane_index) <= max_switchable_ttc:
        #        switchable = False
        return switchable

    def _low_level_control(self, target_speed, target_lane):
        position = self.env.vehicle.position
        heading = self.env.vehicle.heading
        speed = self.env.vehicle.speed
        return self.converter.convert_to_control_parameters(target_speed=target_speed,
                                                            target_lane=target_lane,
                                                            position=position,
                                                            heading=heading,
                                                            speed=speed)


class PlanningBasedController:
    """A secondary controller used by the safeguard"""

    def __init__(
            self,
            env: RTEnv):
        self.env = env.unwrapped
        self.horizon = DEFAULT_HORIZON
        self.time_quantization = DEFAULT_TIME_QUANTIZATION
        self.num_actions = SPEED_LEVELS
        self.speed_levels = SPEED_LEVELS
        self.delta_speed = DELTA_SPEED
        self.idle_action = int((self.speed_levels + 1) / 2)
        self.action_converter = ActionConvertor(env)
        assert self.speed_levels % 2 == 1

    def control(self) -> np.ndarray:
        state, transition, reward, terminal = self._mdp()
        gamma = 0.95
        num_iterations = 10
        value = self._value_iteration(transition, reward, terminal, gamma, num_iterations)
        a_opt = self._get_best_action(state, value, transition)
        # print("a_opt: ", a_opt)
        # print("self._convert_to_lower_level_action(a_opt): ", self._convert_to_lower_level_action(a_opt))
        return self._convert_to_lower_level_action(a_opt)

    def _mdp(self):
        collision_reward = -1  # self.env.config["collision_reward"]
        right_lane_reward = 0  # self.env.config["right_lane_reward"]
        high_speed_reward = 0  # self.env.config["high_speed_reward"]
        lane_change_reward = 0  # self.env.config["lane_change_reward"]

        # Compute TTC grid
        grid = self._ttc_grid()

        # Compute current state
        grid_state = (self.idle_action, self.env.vehicle.lane_index[2], 0)
        # print("grid_state: ", grid_state)
        state = np.ravel_multi_index(grid_state, grid.shape)

        # Compute transition function
        transition = np.zeros((grid.size, self.num_actions), dtype=int)
        for s in range(transition.shape[0]):
            for a in range(transition.shape[1]):
                transition[s, a] = self._transition_model(s, a, grid)

        # Compute reward function
        v, l, t = grid.shape
        lanes = np.arange(l) / max(l - 1, 1)
        speeds = np.arange(v) / max(v - 1, 1)

        state_reward = (
                + collision_reward * grid
                + right_lane_reward
                * np.tile(lanes[np.newaxis, :, np.newaxis], (v, 1, t))
                + high_speed_reward
                * np.tile(speeds[:, np.newaxis, np.newaxis], (1, l, t))
        )

        state_reward = np.ravel(state_reward)
        action_reward = np.zeros(self.num_actions)
        action_reward[:2] = lane_change_reward
        reward = np.fromfunction(
            np.vectorize(lambda s, a: state_reward[s] + action_reward[a]),
            (np.size(state_reward), np.size(action_reward)),
            dtype=int,
        )

        # Compute terminal states
        collision = grid == 1
        end_of_horizon = np.fromfunction(
            lambda h, i, j: j == grid.shape[2] - 1, grid.shape, dtype=int
        )
        terminal = np.ravel(collision | end_of_horizon)
        return state, transition, reward, terminal

    def _transition_model(self, s: int, a: int, grid: np.ndarray) -> int:
        h, i, j = np.unravel_index(s, grid.shape)
        """
        :param h: speed index
        :param i: lane index
        :param j: time index
        :param a: action index
        :param grid: ttc grid specifying the limits of speeds, lanes, time and actions
        """
        left = False  # a == 0
        right = False  # a == 1
        speed_change = j == 0  # (a > 1) & (j == 0)
        a0 = self.idle_action  # +2

        if left:
            next_s = self._clip_position(h, i - 1, j + 1, grid)
        elif right:
            next_s = self._clip_position(h, i + 1, j + 1, grid)
        elif speed_change:
            next_s = self._clip_position(h + a - a0, i, j + 1, grid)
        else:
            next_s = self._clip_position(h, i, j + 1, grid)

        return next_s

    def _ttc_grid(self):
        vehicle = self.env.vehicle
        ego_speed = vehicle.speed
        target_speeds = np.zeros(self.speed_levels)
        for h in range(self.speed_levels):
            target_speeds[h] = ego_speed + (h - self.idle_action) * self.delta_speed
        road_lanes = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
        horizon = self.horizon
        time_quantization = self.time_quantization
        grid = np.zeros(
            (len(target_speeds), len(road_lanes), int(horizon / time_quantization))
        )
        for speed_index in range(grid.shape[0]):
            ego_speed = target_speeds[speed_index]
            for other in self.env.road.vehicles:
                if (other is vehicle) or (ego_speed == other.speed):
                    continue
                margin = other.LENGTH / 2 + vehicle.LENGTH / 2
                collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]
                for m, cost in collision_points:
                    distance = vehicle.lane_distance_to(other) + m
                    other_projected_speed = other.speed * np.dot(
                        other.direction, vehicle.direction
                    )
                    time_to_collision = distance / utils.not_zero(
                        ego_speed - other_projected_speed
                    )
                    if time_to_collision < 0:
                        continue
                    if self.env.road.network.is_connected_road(
                            vehicle.lane_index, other.lane_index, route=None, depth=3
                    ):
                        # Same road, or connected road with same number of lanes
                        if len(self.env.road.network.all_side_lanes(other.lane_index)) == len(
                                self.env.road.network.all_side_lanes(vehicle.lane_index)
                        ):
                            lane = [other.lane_index[2]]
                        # Different road of different number of lanes: uncertainty on future lane, use all
                        else:
                            lane = range(grid.shape[1])
                        # Quantize time-to-collision to both upper and lower values
                        for time in [
                            int(time_to_collision / time_quantization),
                            int(np.ceil(time_to_collision / time_quantization)),
                        ]:
                            if 0 <= time < grid.shape[2]:
                                # TODO: check lane overflow (e.g. vehicle with higher lane id than current road capacity)
                                grid[speed_index, lane, time] = np.maximum(
                                    grid[speed_index, lane, time], cost
                                )
        # indices = {1: self.env.vehicle.lane_index}
        # ix = [indices.get(dim, slice(None)) for dim in range(grid.ndim)]
        # print("tcc grid at current range", grid[ix])
        status = []
        for other in self.env.road.vehicles:
            if other is not vehicle and other.lane_index == vehicle.lane_index:
                margin = other.LENGTH / 2 + vehicle.LENGTH / 2
                on_lane_distance = vehicle.lane_distance_to(other) - margin
                relative_speed = vehicle.speed - other.speed
                # relative_speed = vehicle.speed - other.speed * np.dot(
                #    other.direction, vehicle.direction
                # )
                ttc = on_lane_distance / relative_speed
                status.append(
                    ({'on-lane distance': on_lane_distance}, {'relative speed': relative_speed}, {'ttc': ttc}))

        # distances.sort()
        print("!!!ttc status: ", status)
        return grid

    def _clip_position(self, h: int, i: int, j: int, grid: np.ndarray) -> int:
        """
        Clip a position in the TTC grid, so that it stays within bounds.

        :param h: speed index
        :param i: lane index
        :param j: time index
        :param grid: the ttc grid
        :return: The raveled index of the clipped position
        """

        h = np.clip(h, 0, grid.shape[0] - 1)
        i = np.clip(i, 0, grid.shape[1] - 1)
        j = np.clip(j, 0, grid.shape[2] - 1)
        indexes = np.ravel_multi_index((h, i, j), grid.shape)
        return indexes

    def _value_iteration(self, transition, reward, terminal, gamma, num_iterations):
        value = np.zeros(transition.shape[0])
        for _ in range(num_iterations):
            value = self._bellman_update(value, transition, reward, terminal, gamma)
        return value

    def _bellman_update(self, value, transition, reward, terminal, gamma):
        q_value = np.zeros(transition.shape)
        for s in range(transition.shape[0]):
            for a in range(transition.shape[1]):
                q_value[s, a] = value[transition[s, a]]
        q_value[terminal] = 0
        q_value = reward + gamma * q_value
        return q_value.max(axis=-1)

    def _get_best_action(self, state, value, transition) -> int:
        n_actions = self.num_actions
        v = -np.inf
        a_out = self.num_actions
        for a in reversed(range(self.num_actions)):
            if v < value[transition[state, a]]:
                v = value[transition[state, a]]
                a_out = a
        # print("q value: ", [value[transition[state, a]] for a in range(self.num_actions)])
        return a_out  # np.argmax([value[transition[state, a]] for a in range(self.num_actions)])

    def _convert_to_lower_level_action(self, action):
        position = self.env.vehicle.position
        heading = self.env.vehicle.heading
        speed = self.env.vehicle.speed
        return self.action_converter.convert(action, position, heading, speed)
