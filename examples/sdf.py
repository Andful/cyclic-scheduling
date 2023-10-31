from typing import Tuple, List, Callable
from itertools import product
from cyclic_scheduling import CyclicSchedulingProblem
from collections import deque
from dataclasses import astuple, dataclass
from math import gcd

@dataclass(unsafe_hash=True)
class MultirateSDFActor:
    name: str
    execution_time: int
    processor: str
    repetition: int
    color: str

@dataclass
class MultirateSDFChannel:
    source: int
    target: int
    initial_tokens: int
    
class MultirateSDF:
    processors: List[str]
    actors: List[MultirateSDFActor]
    channels: List[MultirateSDFChannel]
        
    def __init__(self):
        self.processors = []
        self.actors = []
        self.channels = []
        
    def add_processor(self, processor: str):
        self.processors = self.processors + [processor]
        
    def add_actor(self, name: str, execution_time: int, processor: str, repetition: int, color: str = "#ffffff"):
        self.actors = self.actors + [MultirateSDFActor(name, execution_time, processor, repetition, color)]
    
    def add_channel(self, source: str, target: str, initial_tokens: int = 0):
        names = [a.name for a in self.actors]
        self.channels = self.channels + [MultirateSDFChannel(names.index(source), names.index(target), initial_tokens)]
        
    def into_cyclic_scheduling_problem(self, f: Callable[[str, int], str]):
        problem = CyclicSchedulingProblem()
        
        for processor in self.processors:
            problem.add_processor(processor)
            
        actors = {actor: [f(actor.name, i) for i in range(actor.repetition)] for actor in self.actors}
        
        for actor, hsdf_actors in actors.items():
            for name in hsdf_actors:
                problem.add_actor(name, actor.execution_time, actor.processor, actor.color)
                
        for channel in self.channels:
            sources = actors[self.actors[channel.source]]
            targets = deque(actors[self.actors[channel.target]])
            
            divisor = gcd(len(sources), len(targets))
            
            qa, qb, na, nb = len(sources), len(targets), len(targets)//divisor, len(sources)//divisor
            d = channel.initial_tokens

            for i, source in enumerate(sources):
                for k in range(na):
                    target = targets[((d + i*na + k) % (nb*qb))//nb]
                    problem.add_channel(source, target, (d + i*na + k)//(nb*qb))
        
        return problem