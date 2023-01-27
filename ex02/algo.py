from abc import abstractmethod, ABC
from typing import List
from typing import Optional

from pdm4ar.exercises.ex02.structures import AdjacencyList, X

class GraphSearch(ABC):


    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        pass

class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        # todo implement here your solution
        Q=[start]
        V=[start]
        P={start: None}
        while len(Q)>0:
            cand=Q.pop(0)
            if cand == goal:
                path = [cand]
                par = P[cand]
                while par is not None:
                    path.append(par)
                    par = P[par]
                return list(reversed(path))
            for succ in graph[cand]:
                if succ not in V:
                    Q.insert(0, succ)
                    V.append(succ)
                    P[succ] = cand
        return None


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        # todo implement here your solution
        Q = [start]
        V = [start]
        P ={start: None}
        while len(Q) > 0:
            cand=Q.pop()
            if cand == goal:
                path = [cand]
                par = P[cand]
                while par is not None:
                    path.append(par)
                    par = P[par]
                return list(reversed(path))
            for succ in graph[cand]:
                if succ not in V:
                    Q.append(succ)
                    V.append(succ)
                    P[succ] = cand
        return None


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        # todo implement here your solution
        for d in range(1, len(graph)):
            Q = [start]
            V = [start]
            P = {start: None}
            L = {start: 1}
            while len(Q) > 0:
                cand = Q.pop(0)
                if cand == goal:
                    path = [cand]
                    par = P[cand]
                    while par is not None:
                        path.append(par)
                        par = P[par]
                    return list(reversed(path))
                if L[cand]<d:
                    for succ in graph[cand]:
                        if succ not in V:
                            Q.insert(0, succ)
                            V.append(succ)
                            P[succ] = cand
                            L[succ] = L[cand] +1
        return None
