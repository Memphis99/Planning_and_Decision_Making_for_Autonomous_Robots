import math
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, List

from pdm4ar.exercises.ex02.structures import X
from pdm4ar.exercises.ex03.structures import WeightedGraph


class InformedGraphSearch(ABC):
    @abstractmethod
    def path(self, graph: WeightedGraph, start: X, goal: X) -> Optional[List[X]]:
        # need to introduce weights!
        pass


class UniformCostSearch(InformedGraphSearch):
    def path(self, graph: WeightedGraph, start: X, goal: X) -> Optional[List[X]]:
        # todo

        #inizialization
        Q=[start]
        CtR={start: 0} #cost to reach
        P={start: None} #parents

        while len(Q) > 0: #Q is not empty
            CtRQ={q:CtR[q] for q in Q} #create subdict of Q elements
            cand=min(CtRQ, key=CtRQ.get) #I take the lowest Ctr element of Q
            Q.remove(cand)

            if cand==goal: #if the candidate is the goal I return the path
                path = [cand]
                par = P[cand]
                while par is not None:
                    path.append(par)
                    par = P[par]
                return list(reversed(path))

            for succ in graph.adj_list[cand]:
                newCtR= CtR[cand] + graph.get_weight(cand, succ)
                oldCtR= CtR.get(succ, math.inf) #if succ has no CtR, I set to inf

                if newCtR<oldCtR: #if the cost I found is less than the old one
                    CtR[succ]=newCtR #set the CtR of succ that I just found
                    P[succ]=cand #set the parent of succ
                    Q.append(succ) #add succ to the queue

        pass


class GreedyBestFirst(InformedGraphSearch):
    def path(self, graph: WeightedGraph, start: X, goal: X) -> Optional[List[X]]:
        # todo

        #inizialization
        Q = [start]
        V = [start]
        P = {start: None}

        #find goal and start euclidean coord
        goalc=np.array((graph.get_node_attribute(goal, "x"), graph.get_node_attribute(goal, "y")))
        startc=np.array((graph.get_node_attribute(start, "x"), graph.get_node_attribute(start, "y")))

        #create dict with eucl distances to goal, will be the euristic
        ED = {start: np.linalg.norm(goalc-startc)}


        while len(Q) > 0:
            EDQ = {q: ED[q] for q in Q}  # create subdict of Q elements
            cand = min(EDQ, key=EDQ.get)  # I take the lowest ED element of Q
            Q.remove(cand)

            if cand == goal:
                path = [cand]
                par = P[cand]
                while par is not None:
                    path.append(par)
                    par = P[par]
                return list(reversed(path))

            for succ in graph.adj_list[cand]:
                if succ not in V:
                    Q.append(succ)
                    V.append(succ)
                    P[succ] = cand

                    #calculate ED from succ to goal and add to dict
                    succc = np.array((graph.get_node_attribute(succ, "x"), graph.get_node_attribute(succ, "y")))
                    ED[succ]=np.linalg.norm(goalc-succc)
        pass


class Astar(InformedGraphSearch):
    def path(self, graph: WeightedGraph, start: X, goal: X) -> Optional[List[X]]:
        # todo
        # inizialization
        Q = [start]
        CtR = {start: 0}  # cost to reach
        P = {start: None}  # parents

        # find goal and start euclidean coord
        goalc = np.array((graph.get_node_attribute(goal, "x"), graph.get_node_attribute(goal, "y")))
        startc = np.array((graph.get_node_attribute(start, "x"), graph.get_node_attribute(start, "y")))

        # create dict with eucl distances to goal, will be the euristic
        ED = {start: np.linalg.norm(goalc - startc)}

        # create dict with euristic funct= sum of ED and CtR of elements
        EUR = {start: ED[start]+CtR[start]}


def compute_path_cost(wG: WeightedGraph, path: List[X]):
    """A utility function to compute the cumulative cost along a path"""
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
