from functools import partial
from typing import TypeVar

from highway_env import utils

from rt_drl_safeguard.safeguard.action_conversion import *

RTEnv = TypeVar("RTEnv")
Action = TypeVar("Action")

DEFAULT_DELAY_TORRENCE = 0.25
DEFAULT_HORIZON = 10
DEFAULT_TIME_QUANTIZATION = 1


class HighwayAgentSafeguard:
    def __init__(self,
                 env: RTEnv,
                 delay_torrence: float = DEFAULT_DELAY_TORRENCE):
        self.env = env
        self.secondary_controller = SecondaryController(env)
        self.t0 = delay_torrence

    def assure(self, a: Action, t: float) -> Tuple[Action, float, str]:
        if t <= self.t0:
            return a, t, "normal"
        else:
            a = self.secondary_controller.control()
            return a, self.t0, "urgent"


class SecondaryController:
    """A secondary controller used by the safeguard"""

    def __init__(
            self,
            env: RTEnv):
        self.env = env.unwrapped
        # self.target_lane_index = self.env.controlled_vehicle.lane_index
        self.horizon = DEFAULT_HORIZON
        self.time_quantization = DEFAULT_TIME_QUANTIZATION
        self.speed_actions = META_SPEED_ACTIONS
        self.lane_actions = META_LANE_ACTIONS
        self.current_speed_index = self.speed_actions.index("IDLE")
        self.actions = self.speed_actions + self.lane_actions
        self.action_converter = ActionConvertor(env)

    def control(self) -> np.ndarray:
        state, transition, reward, terminal = self._mdp()
        gamma = 0.8
        num_iterations = 10
        value = self._value_iteration(transition, reward, terminal, gamma, num_iterations)
        a_opt = self._get_best_action(state, value, transition)
        return self._convert_to_lower_level_action(self.actions[a_opt])

    def _mdp(self):
        n_actions = len(self.actions)
        collision_reward = self.env.config["collision_reward"]
        right_lane_reward = self.env.config["right_lane_reward"]
        high_speed_reward = self.env.config["high_speed_reward"]
        lane_change_reward = self.env.config["lane_change_reward"]

        # Compute TTC grid
        grid = self._ttc_grid()

        # Compute current state
        grid_state = (self.current_speed_index, self.env.vehicle.lane_index[2], 0)
        state = np.ravel_multi_index(grid_state, grid.shape)

        # Compute transition function
        transition_model_with_grid = partial(self._transition_model, grid=grid)
        transition = np.fromfunction(
            transition_model_with_grid, (grid.size, n_actions), dtype=int
        )

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
        action_reward = [
            lane_change_reward,
            0,
            lane_change_reward,
            0,
            0,
        ]
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
        # Idle action (1) as default transition
        next_s = self._clip_position(h, i, j + 1, grid)
        left = a == 0
        right = a == 2
        faster = (a == 3) & (j == 0)
        slower = (a == 4) & (j == 0)

        next_s[left] = self._clip_position(h[left], i[left] - 1, j[left] + 1, grid)
        next_s[right] = self._clip_position(h[right], i[right] + 1, j[right] + 1, grid)
        next_s[faster] = self._clip_position(h[faster] + 1, i[faster], j[faster] + 1, grid)
        next_s[slower] = self._clip_position(h[slower] - 1, i[slower], j[slower] + 1, grid)

        """
        if left:
            next_s = self._clip_position(h, i - 1, j + 1, grid)
        elif right:
            next_s = self._clip_position(h, i + 1, j + 1, grid)
        elif faster:
            next_s = self._clip_position(h + 1, i, j + 1, grid)
        elif slower:
            next_s = self._clip_position(h - 1, i, j + 1, grid)
        else:
            # Idle action (1) as default transition
            next_s = self._clip_position(h, i, j + 1, grid)
        """
        return next_s

    def _ttc_grid(self):
        vehicle = self.env.vehicle
        ego_speed = vehicle.speed
        target_speeds = np.clip([ego_speed - DEFAULT_DELTA_SPEED,
                                 ego_speed,
                                 ego_speed + DEFAULT_DELTA_SPEED],
                                DEFAULT_MIN_SPEED,
                                DEFAULT_MAX_SPEED)
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
        # q_value = np.fromfunction(lambda s, a: value[transition[s,a]], transition.shape, dtype=int)
        q_value[terminal] = 0
        q_value = reward + gamma * q_value
        return q_value.max(axis=-1)

    def _get_best_action(self, state, value, transition):
        n_actions = len(self.actions)
        return np.argmax([value[transition[state, a]] for a in range(n_actions)])

    def _convert_to_lower_level_action(self, action):
        position = self.env.vehicle.position
        heading = self.env.vehicle.heading
        speed = self.env.vehicle.speed
        return self.action_converter.convert(action, position, heading, speed)
