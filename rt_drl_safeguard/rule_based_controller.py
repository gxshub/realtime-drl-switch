import copy
from typing import List, Union, Tuple
from typing import TypeVar

import numpy as np
from highway_env.road.road import LaneIndex
from highway_env.vehicle.controller import MDPVehicle

RTEnv = TypeVar("RTEnv")
Action = TypeVar("Action")

DEFAULT_TTC_CONTROLLER_CONFIG = {
    "speed_change_range": [0.5, 0.25, 0, -2.5, -5.0, -10.0, -15.0],  # speed changes (m/s)
    "ttc_threshold": 4.5,  # s
    "max_speed": 18.0,  # m/s
    "min_speed": 0.0
}


class TtcBasedController:
    """A controller for safeguard directly based on ttc"""

    def __init__(
            self,
            env: RTEnv,
            config=None):
        self.env = env.unwrapped
        self.params = config if config else DEFAULT_TTC_CONTROLLER_CONFIG
        self.speed_change_range = self.params['speed_change_range']
        self.ttc_threshold = self.params['ttc_threshold']
        self.target_lane_index = None
        self.action_converter = \
            ControllerActionConvertor(env, self.params['max_speed'], self.params['min_speed'])

    def _compute_ttc(self, target_speed, lane_index):
        ego_vehicle = self.env.vehicle
        ego_speed = target_speed
        ttc = np.inf
        for other in self.env.road.vehicles:
            if other is not ego_vehicle and other.lane_index[2] == lane_index[2]:
                margin = other.LENGTH / 2 + ego_vehicle.LENGTH / 2
                if ego_vehicle.lane_distance_to(other) > 0:
                    on_lane_distance_with_margin = ego_vehicle.lane_distance_to(other) - margin
                else:
                    on_lane_distance_with_margin = ego_vehicle.lane_distance_to(other) + margin
                # relative_ahead_speed = ego_speed - other.speed
                relative_ahead_speed = ego_speed - other.speed * np.dot(
                    other.direction, ego_vehicle.direction
                )
                ttc_new = on_lane_distance_with_margin / relative_ahead_speed
                if 0 < ttc_new < ttc:
                    ttc = ttc_new
        return ttc

    def control(self) -> np.ndarray:
        target_index = self.env.vehicle.lane_index
        action = self.get_low_level_action(self.env.vehicle.speed, target_index)
        for speed_change in self.speed_change_range:
            target_speed = self.env.vehicle.speed + speed_change
            if self._compute_ttc(target_speed, target_index) >= self.ttc_threshold:
                action = self.get_low_level_action(target_speed, target_index)
                break
        return action

    def get_low_level_action(self, target_speed, target_lane):
        position = self.env.vehicle.position
        heading = self.env.vehicle.heading
        speed = self.env.vehicle.speed
        return self.action_converter.low_level_control(target_speed=target_speed,
                                                       target_lane=target_lane,
                                                       position=position,
                                                       heading=heading,
                                                       speed=speed)




class ControllerActionConvertor(MDPVehicle):

    def __init__(
            self,
            env,
            max_speed=20.0,
            min_speed=0.0
    ) -> None:
        road = copy.deepcopy(env.unwrapped.road)
        position = copy.deepcopy(env.unwrapped.vehicle.position)
        heading = 0.0
        speed = 0.0
        super().__init__(
            road, position, heading, speed
        )
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.acceleration_range = env.unwrapped.action_type.acceleration_range
        self.steering_range = env.unwrapped.action_type.steering_range

    def low_level_control(self,
                          target_speed: float,
                          target_lane: LaneIndex,
                          position: List[float],
                          heading: float,
                          speed: float) -> np.ndarray:
        self.set_position(position)
        self.set_heading(heading)
        self.set_speed(speed)
        self.on_state_update()

        self.target_speed = target_speed
        self.target_speed = np.clip(self.target_speed, self.min_speed, self.max_speed)
        self.target_lane_index = target_lane

        converted_action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed)
        }
        converted_action["steering"] = np.clip(
            converted_action["steering"], self.steering_range[0], self.steering_range[1]
        )
        converted_action["acceleration"] = np.clip(
            converted_action["acceleration"], self.acceleration_range[0], self.acceleration_range[1]
        )
        ac = lmap(converted_action["acceleration"], self.acceleration_range, [-1, 1])
        st = lmap(converted_action["steering"], self.steering_range, [-1, 1]) * 0.25
        # st = lmap(converted_action["steering"], self.steering_range, [-0.25, 0.25])

        return np.array([ac, st])

    def set_position(self, position):
        self.position = position

    def set_heading(self, heading):
        self.heading = heading

    def set_speed(self, speed):
        self.speed = speed


Interval = Union[
    np.ndarray,
    Tuple[float, float],
    List[float],
]


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
