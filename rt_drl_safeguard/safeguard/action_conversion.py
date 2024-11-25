from typing import List, Optional, Dict

import numpy as np
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.controller import MDPVehicle

DEFAULT_MAX_SPEED = 40.0  # m/s
DEFAULT_MIN_SPEED = 0.0  # m/s
DEFAULT_DELTA_SPEED = 5.0  # m/s
META_SPEED_ACTIONS = ["FASTER", "IDLE", "SLOWER"]
META_LANE_ACTIONS = ["LEFT", "RIGHT"]


class ActionConvertor(MDPVehicle):

    def __init__(
            self,
            road: Road,
            position: List[float],
            heading: float = 0,
            speed: float = 0,
            target_lane_index: Optional[LaneIndex] = None,
            target_speed: Optional[float] = None,
            target_speeds: Optional[Vector] = None,
            route: Optional[Route] = None,
    ) -> None:
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, target_speeds, route
        )

    def convert(self,
                action: str,
                position: List[float],
                heading: float,
                speed: float) -> Dict[str, float]:
        if action not in META_SPEED_ACTIONS + META_LANE_ACTIONS:
            raise RuntimeError(" '{}' is not a supported meta action".format(action))
        self.set_position(position)
        self.set_heading(heading)
        self.set_speed(speed)

        if action == "FASTER":
            self.target_speed += DEFAULT_DELTA_SPEED
            self.target_speed = min(DEFAULT_MAX_SPEED, self.target_speed)
        elif action == "SLOWER":
            self.target_speed -= DEFAULT_DELTA_SPEED
            self.target_speed = max(DEFAULT_MIN_SPEED, self.target_speed)
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
            converted_action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return converted_action

    def set_position(self, position):
        self.position = position

    def set_heading(self, heading):
        self.heading = heading

    def set_speed(self, speed):
        self.speed = speed
