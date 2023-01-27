from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from pdm4ar.exercises.ex04.structures import Action, Policy, ValueFunc, State, Cell


class GridMdp:
    def __init__(self, grid: NDArray[np.int], gamma: float = 0.7):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    def get_next_state(self, state: State, action: Action) -> State:

        if action == Action.NORTH:
            next_s = (state[0] - 1, state[1])
        elif action == Action.WEST:
            next_s = (state[0], state[1] - 1)
        elif action == Action.SOUTH:
            next_s = (state[0] + 1, state[1])
        elif action == Action.EAST:
            next_s = (state[0], state[1] + 1)
        elif action == Action.STAY:
            next_s = state

        return next_s

        pass

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""


        row, col = self.grid.shape

        # calculate the state reached applying the action to current state
        next_s = self.get_next_state(state, action)

        # if next state is out of the map, return P=0
        if next_state[0] < 0 or next_state[0] >= row:
            return float(0)

        if next_state[1] < 0 or next_state[1] >= col:
            return float(0)



        #if next state is different from the state reached, return P=0
        if next_state != next_s:
            return float(0)

        #if I'm not in the goal I can't chose the action Stay
        if self.grid[state] != Cell.GOAL and action == Action.STAY:
            return float(0)

        #in any other case, return P=1
        return float(1)

        pass

    def stage_reward(self, state: State, action: Action) -> float:

        row, col = self.grid.shape

        #calculate the state reached applying the action to current state
        next_s = self.get_next_state(state, action)

        #return reward based on the state we reached

        #if I'm out of map, return 0 (could be any value different from others)
        if next_s[0] < 0 or next_s[0] >= row or next_s[1] < 0 or next_s[1] >= col:
            return float(0)

        elif self.grid[next_s] == Cell.GOAL:
            return float(10)

        elif self.grid[next_s] == Cell.START or self.grid[next_s] == Cell.GRASS:
            return float(-1)

        elif self.grid[next_s] == Cell.SWAMP:
            return float(-10)

        pass




class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        pass
