import math
from typing import Union, Tuple, TypeVar, TYPE_CHECKING

import numpy as np
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle

SAFE_TTC_FASTER = 2.0
SAFE_TTC_IDLE = 1.0
SAFE_TTC_URGENT = 0.5

RTEnv = TypeVar("RTEnv")
Action = TypeVar("Action")


class AbstractSafeguard(object):
    def __init__(self,
                 environment: RTEnv,
                 safe_ttc_faster: float = SAFE_TTC_FASTER,
                 safe_ttc_idle: float = SAFE_TTC_IDLE,
                 safe_ttc_urgent: float = SAFE_TTC_URGENT):
        self.env = environment
        self.safe_ttc_faser = safe_ttc_faster
        self.safe_ttc_idle = safe_ttc_idle
        self.safe_ttc_urgent = safe_ttc_urgent
        self.clock = 0.
        self.location = 0
        self.assurance_flag = "normal"

    def assure(self, time: float, action: Action) -> Tuple[float, Action, str]:
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class HighwayAgentSafeguard(AbstractSafeguard):
    def __init__(self,
                 env: RTEnv,
                 overhead: float = 0.01,
                 time_quantization: float = 0.5,
                 horizon: float = 5.0):
        super().__init__(env)
        # overhead (running time) for the safeguard
        self.overhead = overhead
        # TODO: add more speed indices for speed change
        self.num_speed_indices = 1
        self.num_lanes = len(self.env.unwrapped.road.network.all_side_lanes(env.unwrapped.vehicle.lane_index))
        self.grid = np.zeros(
            (self.num_speed_indices, self.num_lanes, int(horizon / time_quantization))
        )
        self.time_quantization = time_quantization
        self.horizon = horizon
        self.lane_index = env.unwrapped.vehicle.lane_index
        self.ttc = np.inf

    def assure(self, a: Action, t: float) -> Tuple[Action, float, str]:
        if self.location == 0:
            if t >= self.safe_ttc_urgent:
                return np.array([0., 0.]), self.safe_ttc_urgent, "urgent"
            else:
                return a, t, "normal"
        elif self.location == 1:
            if t >= self.safe_ttc_urgent:
                return np.array([-0.5, 0.]), self.safe_ttc_urgent, "urgent"
            else:
                if a[0] > -0.2:
                    a[0] = -0.2
                    return a, t, "modified"
                else:
                    return a, t, "normal"
        elif self.location == 2:
            if t >= self.safe_ttc_urgent:
                return np.array([0., 0.]), self.safe_ttc_urgent, "urgent"
            else:
                if a[0] > 0:
                    a[0] = 0
                    return a, t, "modified"
                else:
                    return a, t, "normal"

    def update(self):
        self._compute_ttc()
        if self.ttc < self.safe_ttc_idle:
            # non-safe location to keep speed
            self.location = 1
        elif self.safe_ttc_idle <= self.ttc < self.safe_ttc_faser:
            # non-safe location to accelerate
            self.location = 2
        else:
            # safe for all actions
            self.location = 0

    def _compute_ttc(self):
        self._compute_ttc_grid()
        # speed_index = self.env.unwrapped.vehicle.speed_index
        speed_index = 0
        lane_index = self.lane_index[2]
        # ttc the first index of collision prob 1
        ttc1_ary = np.where(self.grid[speed_index][lane_index] == 1.0)[0]
        if ttc1_ary.size > 0:
            self.ttc = ttc1_ary[0] * self.time_quantization
        else:
            self.ttc = np.inf

    def _compute_ttc_grid(self):
        """
        Compute the grid of predicted time-to-collision to each vehicle within the lane
        """
        # vehicle = vehicle or env.vehicle
        # road_lanes = env.road.network.all_side_lanes(env.vehicle.lane_index)
        # grid = np.zeros(
        #    (vehicle.target_speeds.size, len(road_lanes), int(horizon / time_quantization))
        # )
        ego_vehicle = self.env.unwrapped.vehicle
        for speed_index in range(self.grid.shape[0]):
            ego_speed = ego_vehicle.speed
            for other in self.env.unwrapped.road.vehicles:
                if (other is ego_vehicle) or (ego_speed == other.speed):
                    continue
                margin = other.LENGTH / 2 + ego_vehicle.LENGTH / 2
                collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]
                for m, cost in collision_points:
                    distance = ego_vehicle.lane_distance_to(other) + m
                    other_projected_speed = other.speed * np.dot(
                        other.direction, ego_vehicle.direction
                    )
                    time_to_collision = distance / utils.not_zero(
                        ego_speed - other_projected_speed
                    )
                    if time_to_collision < 0:
                        continue
                    if self.env.unwrapped.road.network.is_connected_road(
                            ego_vehicle.lane_index, other.lane_index, depth=3#, route=ego_vehicle.route
                    ):
                        # Same road, or connected road with same number of lanes
                        if len(self.env.unwrapped.road.network.all_side_lanes(other.lane_index)) == len(
                                self.env.unwrapped.road.network.all_side_lanes(ego_vehicle.lane_index)
                        ):
                            lane = [other.lane_index[2]]
                        # Different road of different number of lanes: uncertainty on future lane, use all
                        else:
                            lane = range(self.grid.shape[1])
                        # Quantize time-to-collision to both upper and lower values
                        for time in [
                            int(time_to_collision / self.time_quantization),
                            int(np.ceil(time_to_collision / self.time_quantization)),
                        ]:
                            if 0 <= time < self.grid.shape[2]:
                                # TODO: check lane overflow (e.g. vehicle with higher lane id than current road capacity)
                                self.grid[speed_index, lane, time] = np.maximum(
                                    self.grid[speed_index, lane, time], cost
                                )
