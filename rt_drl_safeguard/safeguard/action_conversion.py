import copy
from typing import List, Union, Tuple

import numpy as np
from highway_env.vehicle.controller import MDPVehicle

DEFAULT_MAX_SPEED = 40.0  # m/s
DEFAULT_MIN_SPEED = 0.0  # m/s
DEFAULT_DELTA_SPEED = 5.0  # m/s
META_SPEED_ACTIONS = ["FASTER", "IDLE", "SLOWER"]
META_LANE_ACTIONS = ["LEFT", "RIGHT"]


class ActionConvertor(MDPVehicle):

    def __init__(
            self,
            env
    ) -> None:
        road = copy.deepcopy(env.road)
        position = copy.deepcopy(env.vehicle.position)
        heading = 0.0
        speed = 0.0
        super().__init__(
            road, position, heading, speed
        )
        self.MAX_SPEED = DEFAULT_MAX_SPEED
        self.MIN_SPEED = DEFAULT_MIN_SPEED
        self.ACCELERATION_RANGE = env.action_type.acceleration_range
        self.STEERING_RANGE = env.action_type.steering_range

    def convert(self,
                action: str,
                position: List[float],
                heading: float,
                speed: float) -> np.ndarray:
        if action not in META_SPEED_ACTIONS + META_LANE_ACTIONS:
            raise RuntimeError(" '{}' is not a supported meta action".format(action))
        self.set_position(position)
        self.set_heading(heading)
        self.set_speed(speed)

        if action == "FASTER":
            self.target_speed = min(self.MAX_SPEED, self.target_speed + DEFAULT_DELTA_SPEED)
        elif action == "SLOWER":
            self.target_speed = max(self.MIN_SPEED, self.target_speed - DEFAULT_DELTA_SPEED)
        elif action == "RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                    self.position
            ):
                self.target_lane_index = target_lane_index
        elif action == "LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                    self.position
            ):
                self.target_lane_index = target_lane_index

        converted_action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed),
        }
        converted_action["steering"] = np.clip(
            converted_action["steering"], self.STEERING_RANGE[0], self.STEERING_RANGE[1]
        )
        converted_action["acceleration"] = np.clip(
            converted_action["acceleration"], self.ACCELERATION_RANGE[0], self.ACCELERATION_RANGE[1]
        )
        ac = lmap(converted_action["acceleration"], self.ACCELERATION_RANGE, [-1, 1])
        st = lmap(converted_action["steering"], self.STEERING_RANGE, [-1, 1])
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
