"""Supply the official distance function its required mutable Episode cache slot."""
from types import SimpleNamespace
class GoalDistance:
    def __init__(self,metric,goal):
        self.metric=metric
        self.goals=[list(goal)]
        self.episode=SimpleNamespace(_shortest_path_cache=None)
    def measure(self,pathfinder,position):
        return self.metric.measure(pathfinder,position,self.goals,self.episode)
