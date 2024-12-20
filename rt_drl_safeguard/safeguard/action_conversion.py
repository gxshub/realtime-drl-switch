import copy
from typing import List, Union, Tuple

import numpy as np
from highway_env.vehicle.controller import MDPVehicle

from rt_drl_safeguard.safeguard.meta_control_data import MAX_SPEED, MIN_SPEED, SPEED_LEVELS, DELTA_SPEED


class ActionConvertor(MDPVehicle):

    def __init__(
            self,
            env
    ) -> None:
        road = copy.deepcopy(env.road)
        position = copy.deepcopy(env.unwrapped.vehicle.position)
        heading = 0.0
        speed = 0.0
        super().__init__(
            road, position, heading, speed
        )
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED
        self.acceleration_range = env.unwrapped.action_type.acceleration_range
        self.steering_range = env.unwrapped.action_type.steering_range
        self.speed_change_range = np.linspace(-1, 1, SPEED_LEVELS) * DELTA_SPEED
        # print("self.speed_change_range: ", self.speed_change_range)
        assert len(self.speed_change_range) == SPEED_LEVELS
        self.num_actions = SPEED_LEVELS

    def convert(self,
                action: int,
                position: List[float],
                heading: float,
                speed: float) -> np.ndarray:
        self.set_position(position)
        self.set_heading(heading)
        self.set_speed(speed)
        self.on_state_update()

        if action not in range(self.num_actions):
            raise RuntimeError("illegal action index ")

        self.target_speed = self.speed + self.speed_change_range[action]
        self.target_speed = np.clip(self.target_speed, MIN_SPEED, MAX_SPEED)
        self.target_speed = np.clip(self.target_speed, self.min_speed, self.max_speed)
        self.target_lane_index = self.lane_index

        converted_action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed),
        }
        converted_action["steering"] = np.clip(
            converted_action["steering"], self.steering_range[0], self.steering_range[1]
        )
        converted_action["acceleration"] = np.clip(
            converted_action["acceleration"], self.acceleration_range[0], self.acceleration_range[1]
        )
        ac = lmap(converted_action["acceleration"], self.acceleration_range, [-1, 1])
        st = lmap(converted_action["steering"], self.steering_range, [-1, 1])

        print("@@@ current speed: {}, target speed: {} ".format(self.speed, self.target_speed),
              "acceleration: {}, steering: {}".format(converted_action["acceleration"], converted_action["steering"]))
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


def do_every(duration: float, timer: float) -> bool:
    return duration < timer


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
