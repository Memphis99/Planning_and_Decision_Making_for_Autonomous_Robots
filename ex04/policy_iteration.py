from typing import Tuple

import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy, Action, Cell
from pdm4ar.exercises_def.ex04.utils import time_function



class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        row, col = grid_mdp.grid.shape  # get the map dimensions

        cont = 1


        #create the baseline policy
        for i in range(row):
            for j in range(col):
                policy[(i, j)] = 1


        #compute the first value function using baseline policy
        prev_value_func = value_func.copy()  # get the value_func at previous step to have it fixed

        for i in range(0, row):
            for j in range(0, col):  # analyze every cell of the map
                state = (i, j)
                val = float(0)
                pol = policy[state] #get the previous policy for the cell

                reward = grid_mdp.stage_reward(state, pol)  # compute reward for the previous policy

                # if the state is not goal, stay has to be ignored
                if pol == Action.STAY and grid_mdp.grid[state] != Cell.GOAL:
                    val = 0

                # if with action I go out of map, action has to be ignored
                elif reward == 0:
                    val = 0

                # if action get me to a feasible state, check its value
                elif reward != 0:
                    for h in range(0, row):
                        for k in range(0, col):
                            next_state = (h, k)
                            prob = grid_mdp.get_transition_prob(state, pol, next_state)

                            # prob is !=0 only for the state I get to with the action
                            if prob != 0:
                                # val(a)=sum for all states[prob_of_transition*(reward(k) + gamma*value(state)]
                                val += prob * (reward + grid_mdp.gamma * prev_value_func[next_state])

                value_func[(i, j)] = val


        while cont > 0:
            cont = 0 #init the cont at every step
            prev_value_func = value_func.copy()  # get the value_func at previous step to have it fixed
            prev_policy = policy.copy() #save the previous policy to count the policy changes later
            for i in range(0, row):
                for j in range(0, col):  # analyze every cell of the map
                    state = (i, j)
                    reward = [0.0, 0.0, 0.0, 0.0, 0.0]  # init the action rewards at every cell
                    newval = float(-1001)  # init the max value of actions at every cell
                    val = [0.0, 0.0, 0.0, 0.0, 0.0]  # init the action value at every cell

                    for a in Action:  # analyze all the possible actions
                        reward[a] = grid_mdp.stage_reward(state, a)  # compute reward for the action

                        # if the state is not goal, stay has to be ignored
                        if a == Action.STAY and grid_mdp.grid[state] != Cell.GOAL:
                            val[a] = float(-1000)

                        # if with action I go out of map, action has to be ignored
                        elif reward[a] == 0:
                            val[a] = float(-1000)

                        # if action get me to a feasible state, check its value
                        elif reward[a] != 0:
                            for h in range(0, row):
                                for k in range(0, col):
                                    next_state = (h, k)
                                    prob = grid_mdp.get_transition_prob(state, a, next_state)

                                    # prob is !=0 only for the state I get to with the action
                                    if prob != 0:
                                        # val(a)=sum for all states[prob_of_transition*(reward(k) + gamma*value(state)]
                                        val[a] += prob * (reward[a] + grid_mdp.gamma * prev_value_func[next_state])

                        # if the val of the action is better than previously, I update the max
                        # and choose the action as policy (at least for now)
                        if val[a] >= newval:
                            newval = val[a]
                            policy[state] = a

                    # calculate the difference from new to old value for every cell
                    if policy[state] != prev_policy[state]:
                        cont += 1

                    # assign the final cell value to value function
                    value_func[state] = newval


        return value_func, policy
