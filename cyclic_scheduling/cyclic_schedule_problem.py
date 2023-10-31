from dataclasses import astuple, dataclass
from typing import List, Dict, Any
import numpy as np
from itertools import permutations, chain, islice
from collections import OrderedDict
from ipywidgets import DOMWidget

def format_add(i: int) -> str:
    if i == 0:
        return ""
    elif i > 0:
        return f" + {i}" 
    else:
        return f" - {-i}" 

@dataclass
class Actor:
    name: str
    execution_time: int
    processor: str
    color: str

@dataclass
class Channel:
    source: int
    target: int
    initial_tokens: int
    additional: bool
        

    @staticmethod
    def from_data(actors_map: Dict[str, int], data):
        source = data['source']
        target = data['target']

        assert source in actors_map, f"Actor {source} undefined"
        assert target in actors_map, f"Actor {target} undefined"

        return Channel(actors_map[source], actors_map[target], data.get('initial-tokens', 0), False)

@dataclass
class CyclicSchedulingProblem:
    actors: List[Actor]
    channels: List[Channel]
    processors: OrderedDict[str, List[int]]

    def __init__(self):
        self.actors = []
        self.channels = []
        self.processors = OrderedDict()

    def add_processor(self, processor: str):
        if processor in self.processors:
            raise ValueError(f"{processor} is already present")
        self.processors = OrderedDict(chain(self.processors.items(), [(processor, [])]))

    def add_actor(self, name: str, execution_time: int, processor: str, color: str = "#ffffff"):
        if processor not in self.processors:
            raise ValueError(f'No processor "{processor}"')
        
        self.processors = OrderedDict((p, (e if p != processor else list(chain(e, [len(self.actors)]))))  for (p, e) in self.processors.items())
        self.actors = list(chain(self.actors, [Actor(name, execution_time, processor, color)]))

    def add_channel(self, source: str, target: str, initial_tokens: int = 0):
        source_index = list(islice((i for (i, a) in enumerate(self.actors) if a.name == source), 1))
        target_index = list(islice((i for (i, a) in enumerate(self.actors) if a.name == target), 1))
        if len(source_index) < 1:
            raise ValueError(f'No source actor "{source}"')
        if len(target_index) < 1:
            raise ValueError(f'No target actor "{target}"')
        self.channels = list(chain(self.channels, [Channel(source_index[0], target_index[0], initial_tokens, False)]))

    @staticmethod
    def validate_data(data: Any):
        from schema import Schema, And, Optional
        import re
        regex = re.compile("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")
        schema = Schema({
            'actors': [{
                'name': str,
                'execution-time': int,
                'processor': str,
                Optional('color'): And(str, lambda s: regex.match(s) is not None)
            }],
            'channels': [{
                'source': str,
                'target': str,
                Optional('initial-tokens'): int
            }]
        })
        schema.validate(data)

    @staticmethod
    def from_file(file: str) -> 'CyclicSchedulingProblem':
        import yaml
        with open(file, 'r') as content:
            data = yaml.safe_load(content)
        
        CyclicSchedulingProblem.validate_data(data)

        result = CyclicSchedulingProblem()

        processors = set(actor['processor'] for actor in data['actors'])
        for processor in processors:
            result.add_processor(processor)

        for actor in data['actors']:
            result.add_actor(actor['name'], actor['execution-time'], actor['processor'], actor.get('color', '#ffffff'))

        for channel in data['channels']:
            result.add_channel(channel['source'], channel['target'], channel.get('initial-tokens', 0))
        
        return result
    
    @staticmethod
    def _from_data(actors: List[Actor], channels: List[Channel], processors: OrderedDict[str, List[int]]) -> 'CyclicSchedulingProblem':
        result = CyclicSchedulingProblem()
        result.actors = actors
        result.channels = channels
        result.processors = processors

        return result

    def fire(self, name: str) -> 'CyclicSchedulingProblem':
        index = next(i for (i, a) in enumerate(self.actors) if a.name == name)
        def modify_channel(c: Channel) -> Channel:
            if (c.source == index and c.target == index) or (c.source != index and c.target != index):
                return c
            (source, target, initial_tokens, additional) = astuple(c)
            return Channel(source, target, (initial_tokens + 1) if source == index else (initial_tokens - 1), additional)
        return CyclicSchedulingProblem._from_data(
            self.actors,
            list(map(modify_channel, self.channels)),
            self.processors
        )

    def max_plus(self) -> 'MaxPlus':        
        return MaxPlus(self)
    
    def system_of_inequalities(self) -> 'SystemOfInequalities':        
        return SystemOfInequalities(self)
    
    def lp_formulation(self) -> 'MipFormulation':
        return MipFormulation(self, True)

    def mip_formulation(self, relaxed=False) -> 'MipFormulation':
        return MipFormulation(self, relaxed)

    def solution(self, relaxed: bool) -> 'CyclicSchedulingSolution':
        from pulp import LpProblem, LpVariable, LpAffineExpression, LpMaximize, GUROBI

        n_a = len(self.actors)
        problem = LpProblem(sense=LpMaximize)
        throughput = LpVariable("throughput", lowBound=0)
        u = np.array([LpVariable(f"u({a})", lowBound=0) for a in map(lambda a: a.name, self.actors)])            

        problem.setObjective(throughput)

        for (source, target, initial_tokens, _) in map(astuple, self.channels):
            execution_time = self.actors[source].execution_time
            problem += u[target] >= u[source] + throughput*execution_time - initial_tokens

        if not relaxed:
            k = np.array([[LpVariable(f"K({self.actors[i].name},{self.actors[j].name})", cat='Integer') if i < j and self.actors[i].processor == self.actors[j].processor else LpAffineExpression() for i in range(n_a)] for j in range(n_a)])
            k = k + (np.triu(np.ones((n_a, n_a)), k=1) - k.T)
            for (_, actors) in self.processors.items():
                for (source, target) in permutations(actors, 2):
                    initial_tokens = k[source, target]
                    execution_time = self.actors[source].execution_time
                    problem += (u[target] >= u[source] + throughput*execution_time - initial_tokens)
            for (_, actors) in self.processors.items():
                cycle_time_lb = sum(map(lambda i: self.actors[i].execution_time, actors))
                if cycle_time_lb > 0:
                    problem += (cycle_time_lb*throughput <= 1)
        problem.solve(GUROBI(msg=0))

        get_value = np.vectorize(lambda x: x.value())

        cycle_time = round(1/throughput.value())
        u = get_value(u)
        u = u - np.min(u)

        if relaxed:
            return CyclicSchedulingSolution(cycle_time, np.floor(cycle_time * u).tolist(), OrderedDict(), problem.solutionTime)
        
        k = get_value(k)
        return CyclicSchedulingSolution(
            cycle_time,
            np.floor(cycle_time * u).astype(int).tolist(),
            {(source, target): round(k[source, target]) for (_, actors) in self.processors.items() for (source, target) in permutations(actors, 2)},
            problem.solutionTime,
        )
    
    def solve(self) -> 'CyclicSchedulingProblem':
        from itertools import chain
        _, _, k, _ = astuple(self.solution(False))

        def to_channel(e: tuple[tuple[int, int], int]) -> Channel:
            ((source, target), initial_tokens) = e
            return Channel(source, target, initial_tokens, True)
        
        return CyclicSchedulingProblem._from_data(
            self.actors,
            list(chain(self.channels, map(to_channel, k.items()))),
            self.processors
        )
    
    def plot_asap(self) -> 'CyclicSchedulingPlot':
        return CyclicSchedulingPlot(self, False)
    
    def plot_with_processors(self) -> 'CyclicSchedulingPlot':
        return CyclicSchedulingPlot(self, True)
    
    def _repr_svg_(self):
        import graphviz

        dot = graphviz.Digraph()
        for (i, e) in enumerate(self.actors):
            dot.node(str(i), f"{e.name}\n{e.execution_time}")

        for (source, target, initial_tokens, additional) in map(astuple, self.channels):
            dot.edge(str(source), str(target), label=str(initial_tokens), style="dashed" if additional else None)

        return dot._repr_image_svg_xml()
    
    def solve_relaxation(self) -> 'CyclicSchedulingSolution':
        return self.solution(relaxed=True)

@dataclass
class CyclicSchedulingSolution:
    cycle_time: int
    t: List[int]
    k: Dict[tuple[int, int], int]
    solutionTime: int

@dataclass
class CyclicSchedulingPlotData:
    problem: CyclicSchedulingProblem
    cycle_time: int
    processors: List[str]
    t: List[int]
    actor2processors: List[int]

class CyclicSchedulingPlot(DOMWidget):
    from traitlets import Unicode
    from ._frontend import module_name, module_version

    _model_name = Unicode('CyclicSchedulingPlotModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode('CyclicSchedulingPlotView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)
    data = Unicode("data").tag(sync=True)

    def __init__(self, problem: CyclicSchedulingProblem, with_machine: bool = True):
        import json
        from dataclasses import asdict
        cycle_time, t, _, _ = astuple(problem.solution(relaxed=True))

        if with_machine:
            processors = list(problem.processors.keys())
            actor2processors = list([processors.index(e.processor) for e in problem.actors])
        else:
            processors = [a.name for a in problem.actors]
            actor2processors = list(range(len(problem.actors)))
        super().__init__()
        self.data = json.dumps(asdict(CyclicSchedulingPlotData(
            problem,
            cycle_time,
            processors,
            t,
            actor2processors
        )))

@dataclass
class MaxPlus:
    problem: CyclicSchedulingProblem

    def format_execution_time(self, t: int) -> str:
        if t == 0:
            return ""
        else:
            return f"{t}"
        
    def format_execution_time_latex(self, t: int) -> str:
        if t == 0:
            return ""
        else:
            return f"{t}"

    def format_channel(self, c: Channel) -> str:
        (source, target, initial_tokens, _) = astuple(c)
        actors = self.problem.actors
        return f"{actors[target].name}(k) \u2265 {self.format_execution_time(actors[source].execution_time)}{actors[source].name}(k{format_add(-initial_tokens)})"

    def format_channel_latex(self, c: Channel) -> str:
        (source, target, initial_tokens, _) = astuple(c)
        actors = self.problem.actors
        return f"{actors[target].name}(k) & \\ge & {self.format_execution_time_latex(actors[source].execution_time)}{actors[source].name}&(k{format_add(-initial_tokens)})"

    def __repr__(self) -> str:
        return "\n".join(map(lambda c: self.format_channel(c), self.problem.channels))
    
    def _repr_latex_(self) -> str:
        NEWLINE = "\\\\"
        return f'\\begin{{align}}{NEWLINE.join(map(lambda c: self.format_channel_latex(c), self.problem.channels))}\\end{{align}}\n'

@dataclass
class SystemOfInequalities:
    problem: CyclicSchedulingProblem

    def format_execution_time(self, t: int) -> str:
        if t == 0:
            return ""
        else:
            return f"{t} + "

    def format_channel(self, c: Channel) -> str:
        (source, target, initial_tokens, _) = astuple(c)
        actors = self.problem.actors
        return f"{actors[target].name}(k) \u2265 {self.format_execution_time(actors[source].execution_time)}{actors[source].name}(k{format_add(-initial_tokens)})"

    def format_channel_latex(self, c: Channel) -> str:
        (source, target, initial_tokens, _) = astuple(c)
        actors = self.problem.actors
        return f"{actors[target].name}(k) & \\ge & {self.format_execution_time(actors[source].execution_time)}{actors[source].name}&(k{format_add(-initial_tokens)})"

    def __repr__(self) -> str:
        return "\n".join(map(lambda c: self.format_channel(c), self.problem.channels))
    
    def _repr_latex_(self) -> str:
        newline = "\\\\ "
        return f'\\begin{{align}}{newline.join(map(lambda c: self.format_channel_latex(c), self.problem.channels))}\\end{{align}}\n'


@dataclass
class MipFormulation:
    problem: CyclicSchedulingProblem
    relaxed: bool
    def format_channel(self, c: Channel) -> str:
        (source, target, initial_tokens, _) = astuple(c)
        actors = self.problem.actors
        return f"u({actors[target].name}) \u2265 {actors[source].execution_time}t + u({actors[source].name}){format_add(-initial_tokens)}"
    def __repr__(self) -> str:
        from itertools import product
        actors = self.problem.actors
        NEWLINE = "\n"
        not_relaxed = f"""
{NEWLINE.join(f"u({actors[j].name}) ≥ {actors[i].execution_time}t + u({actors[i].name}) + K({actors[i].name},{actors[j].name})" for (_, acs) in self.problem.processors.items() for (i, j) in product(acs, repeat=2) if i != j)}
{NEWLINE.join(f"K({actors[i].name},{actors[j].name}) + K({actors[j].name},{actors[i].name}) = 1" for (_, acs) in self.problem.processors.items() for (i, j) in product(acs, repeat=2) if i != j)}
{",".join(f"K({actors[i].name},{actors[j].name}),K({actors[j].name},{actors[i].name})" for (_, acs) in self.problem.processors.items() for (i, j) in product(acs, repeat=2) if i != j)} ∈ \u2124
"""
        return f"""Maximize t

{NEWLINE.join(map(lambda c: self.format_channel(c), self.problem.channels))}{not_relaxed if not self.relaxed else ""}
"""
    def format_channel_latex(self, source: int, target: int, constant: str) -> str:
        actors = self.problem.actors
        (sa, ta) = (actors[source], actors[target])
        et = sa.execution_time
        if et == 0:
            addition = " & "
        elif et == 1:
            addition = " & \\tau "
        else:
            addition = f" {et} & \\tau "
        return f"u({ta.name}) & \\ge & {addition} & + & u({actors[source].name}) & {constant}"
    
    def _repr_latex_(self) -> str:
        from itertools import product, combinations
        actors = self.problem.actors
        NEWLINE = "\\\\"
        IN_INTEGER = "\\in \\mathbb{Z}"

        not_relaxed = f"""
{NEWLINE.join(map(lambda c: self.format_channel_latex(c[0], c[1], f"+ K({actors[c[0]].name},{actors[c[1]].name})"), ((source, target) for (_, acs) in self.problem.processors.items() for (source, target) in product(acs, repeat=2) if source != target)))}\\\\
{NEWLINE.join(map(lambda c: f"K({actors[c[0]].name},{actors[c[1]].name}) + K({actors[c[1]].name},{actors[c[0]].name}) & = & 1", ((source, target) for (_, acs) in self.problem.processors.items() for (source, target) in combinations(acs, 2) if source != target)))}\\\\
{NEWLINE.join(map(lambda c: f"K({actors[c[0]].name},{actors[c[1]].name}) {IN_INTEGER}", ((source, target) for (_, acs) in self.problem.processors.items() for (source, target) in product(acs, repeat=2) if source != target)))}
"""
        return f"""\\begin{{align}}\\text{{Maximize}} \\quad \\tau & & \\\\

{NEWLINE.join(map(lambda c: self.format_channel_latex(c.source, c.target, format_add(-c.initial_tokens)), self.problem.channels))}\\\\{not_relaxed if not self.relaxed else ""}
\\end{{align}}
"""