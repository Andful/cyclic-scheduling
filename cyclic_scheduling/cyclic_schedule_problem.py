from dataclasses import astuple, dataclass
from typing import List, Dict, Any
import numpy as np
from itertools import permutations
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
        

    @staticmethod
    def from_data(actors_map: Dict[str, int], data):
        source = data['source']
        target = data['target']

        assert source in actors_map, f"Actor {source} undefined"
        assert target in actors_map, f"Actor {target} undefined"

        return Channel(actors_map[source], actors_map[target], data.get('initial-tokens', 0))

@dataclass
class CyclicSchedulingProblem:
    actors: List[Actor]
    channels: List[Channel]
    processors: Dict[str, List[int]]

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

        actors = [Actor(a['name'], a['execution-time'], a['processor'], a.get('color', '#ffffff')) for a in data['actors']]
        actors_map = dict((a['name'], i) for i,a in enumerate(data['actors']))
        channels = [Channel.from_data(actors_map, c) for c in data['channels']]
        processors = dict()
        for i, actor in enumerate(data['actors']):
            processor = actor['processor']
            if processor in processors:
                elements = processors[processor]
            else:
                elements = []
                processors[processor] = elements
            elements.append(i)
        
        return CyclicSchedulingProblem(actors, channels, processors)

    def format_channel(self, c: Channel) -> str:
            (source, target, initial_tokens) = astuple(c)
            return f"{self.actors[target].name}(k) \u2265 {self.format_execution_time(self.actors[source].execution_time)}{self.actors[source].name}(k{self.format_initial_tokens(initial_tokens)})"

    def max_plus(self) -> 'MaxPlus':        
        return MaxPlus(self)
    
    def system_of_inequalities(self) -> 'SystemOfInequalities':        
        return SystemOfInequalities(self)
    
    def lp_formulation(self) -> 'MipFormulation':
        return MipFormulation(self, True)

    def mip_formulation(self, relaxed=False) -> 'MipFormulation':
        return MipFormulation(self, relaxed)
    
    def additional_systems_of_inequalities(self, k: np.ndarray) -> str:
        channels = [Channel(source, target, k[source, target]) for (_, actors) in self.processors.items() for (source, target) in permutations(actors, 2)]
        return "\n".join(map(lambda c: self.format_channel(c), channels))


    def dependencies_to_maxplus_matrix(self, channels: List[Channel]) -> np.ndarray:
        result = np.fromiter(map(lambda _: None, range(4)), dtype=np.object_)
    
    def matrix(self) -> str:
        possible_initial_tokens = set(c.initial_tokens for c in self.channels)
        initial_token_map = { t: c for c in self.channels for t in possible_initial_tokens if c.initial_tokens == t }
    
    def solve(self, relaxed=False) -> 'CyclicSchedulingSolution':
        from pulp import LpProblem, LpVariable, LpAffineExpression, LpMaximize, PULP_CBC_CMD

        n_a = len(self.actors)
        problem = LpProblem(sense=LpMaximize)
        throughput = LpVariable("throughput", lowBound=0)
        u = np.array([LpVariable(f"u({a})", lowBound=0) for a in map(lambda a: a.name, self.actors)])
        if not relaxed:
            k = np.array([[LpVariable(f"K({self.actors[i].name},{self.actors[j].name})", cat='Integer') if i < j and self.actors[i].processor == self.actors[j].processor else LpAffineExpression() for i in range(n_a)] for j in range(n_a)])
            k = k + (np.triu(np.ones((n_a, n_a)), k=1) - k.T)

        problem.setObjective(throughput)

        for (source, target, initial_tokens) in map(astuple, self.channels):
            execution_time = self.actors[source].execution_time
            problem += u[target] >= u[source] + throughput*execution_time - initial_tokens

        if not relaxed:
            for (_, actors) in self.processors.items():
                for (source, target) in permutations(actors, 2):
                    initial_tokens = k[source, target]
                    execution_time = self.actors[source].execution_time
                    problem += (u[target] >= u[source] + throughput*execution_time - initial_tokens)
        problem.solve(PULP_CBC_CMD(msg=0))

        get_continuous = np.vectorize(lambda x: x.value(), otypes=[float])

        cycle_time = round(1/throughput.value())

        processors = list(self.processors.keys())

        return CyclicSchedulingSolution(
            self,
            [a.name for a in self.actors] if relaxed else processors,
            [] if relaxed else [Channel(source, target, round(k[source, target].value())) for (_, actors) in self.processors.items() for (source, target) in permutations(actors, 2)], #todo fix
            np.round(cycle_time*get_continuous(u)).astype(int).tolist(),
            list(range(len(self.actors))) if relaxed else [processors.index(a.processor) for a in self.actors],
            cycle_time
        )
    
    def _repr_svg_(self):
        import graphviz

        dot = graphviz.Digraph()
        for (i, e) in enumerate(self.actors):
            dot.node(str(i), f"{e.name}\n{e.execution_time}")

        for (source, target, initial_tokens) in map(astuple, self.channels):
            dot.edge(str(source), str(target), label=str(initial_tokens))

        return dot._repr_image_svg_xml()
    
    def solve_relaxation(self) -> 'CyclicSchedulingProblem':
        return self.solve(relaxed=True)

@dataclass
class CyclicSchedulingSolution:
    problem: CyclicSchedulingProblem
    processors: List[str]
    additional_channels: List[Channel]
    t: List[int]
    actor2processors: Dict[int, int]
    cycle_time: int

    def _repr_svg_(self):
        import graphviz

        dot = graphviz.Digraph()
        for (i, e) in enumerate(self.problem.actors):
            dot.node(str(i), f"{e.name}\n{e.execution_time}")

        for (source, target, initial_tokens) in map(astuple, self.problem.channels):
            dot.edge(str(source), str(target), label=str(initial_tokens) if initial_tokens != 0 else None)

        for (source, target, initial_tokens) in map(astuple, self.additional_channels):
            dot.edge(str(source), str(target), label=str(initial_tokens), style="dashed")

        return dot._repr_image_svg_xml()
    
    def plot(self) -> 'CyclicSchedulingPlot':
        from json import dumps
        from dataclasses import asdict
        return CyclicSchedulingPlot(dumps(asdict(self)))
    
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

    def __init__(self, data: str):
        
        super().__init__()
        self.data = data

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
        (source, target, initial_tokens) = astuple(c)
        actors = self.problem.actors
        return f"{actors[target].name}(k) \u2265 {self.format_execution_time(actors[source].execution_time)}{actors[source].name}(k{format_add(-initial_tokens)})"

    def format_channel_latex(self, c: Channel) -> str:
        (source, target, initial_tokens) = astuple(c)
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
        (source, target, initial_tokens) = astuple(c)
        actors = self.problem.actors
        return f"{actors[target].name}(k) \u2265 {self.format_execution_time(actors[source].execution_time)}{actors[source].name}(k{format_add(-initial_tokens)})"

    def format_channel_latex(self, c: Channel) -> str:
        (source, target, initial_tokens) = astuple(c)
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
        (source, target, initial_tokens) = astuple(c)
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