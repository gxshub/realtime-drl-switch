from typing import SupportsFloat, Any, Tuple, Dict

from gymnasium import Wrapper
from gymnasium.core import WrapperActType, WrapperObsType
from highway_env.envs import HighwayEnv


class RealtimeHighway(Wrapper):
    def __init__(self, env: "HighwayEnv"):
        super().__init__(env)
        # Delay
        self.delay = 0.0
        self.delayed_frequency = 0

    def step(
            self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.delayed_frequency = int(self.delay * self.env.unwrapped.config["simulation_frequency"])
        # print("[RealtimeHighway] delay: {}, delayed_frequency: {}".format(self.delay, self.delayed_frequency))
        for _ in range(self.delayed_frequency):
            self.env.unwrapped.road.act()
            self.env.unwrapped.road.step(1 / self.env.unwrapped.config["simulation_frequency"])
            # self.env.unwrapped._automatic_rendering()
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    @property
    def delay(self) -> float:
        return self._delay

    @delay.setter
    def delay(self, value):
        self._delay = value

    def set_delay(self, value):
        self._delay = value


