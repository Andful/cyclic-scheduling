import yaml
from dataclasses import astuple, dataclass, asdict
from typing import List, Dict
import numpy as np
from pulp import LpProblem, LpVariable, LpAffineExpression, LpMaximize
from itertools import permutations

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

        return Channel(actors_map[source], actors_map[target], data['initial-tokens'])

@dataclass
class SystemOfInequalities:
    actors: List[Actor]
    channels: List[Channel]

    def format_execution_time(self, t: int) -> str:
        if t == 0:
            return ""
        else:
            return f"{t} \u2297 "
        
    def format_execution_time_latex(self, t: int) -> str:
        if t == 0:
            return ""
        else:
            return f"{t} &\\otimes& "
    
    def format_initial_tokens(self, t: int) -> str:
        if t > 0:
            return f" - {t}"
        elif t == 0:
            return ""
        else:
            return f" + {-t}"

    def format_channel(self, c: Channel) -> str:
        (source, target, initial_tokens) = astuple(c)
        return f"{self.actors[target].name}(k) \u2265 {self.format_execution_time(self.actors[source].execution_time)}{self.actors[source].name}(k{self.format_initial_tokens(initial_tokens)})"

    def format_channel_latex(self, c: Channel) -> str:
        (source, target, initial_tokens) = astuple(c)
        return f"{self.actors[target].name}(k) &\\ge& {self.format_execution_time_latex(self.actors[source].execution_time)}{self.actors[source].name}(k{self.format_initial_tokens(initial_tokens)})"

    def __repr__(self) -> str:
        return "\n".join(map(lambda c: self.format_channel(c), self.channels))
    
    def _repr_latex_(self) -> str:
        newline = "\\\\"
        return f'$\\displaystyle \n\\begin{{align}}{newline.join(map(lambda c: self.format_channel_latex(c), self.channels))}\\end{{align}}\n$'

class CyclicSchedulingProblem:
    actors: List[Actor]
    channels: List[Channel]
    processors: Dict[str, List[int]]

    def __init__(self, data) -> None:
        self.actors = [Actor(a['name'], a['execution-time'], a['processor'], a.get('color', '#ffffff')) for a in data['actors']]
        actors_map = dict((a['name'], i) for i,a in enumerate(data['actors']))
        self.channels = [Channel.from_data(actors_map, c) for c in data['channels']]
        self.processors = dict()
        for i, actor in enumerate(data['actors']):
            processor = actor['processor']
            if processor in self.processors:
                elements = self.processors[processor]
            else:
                elements = []
                self.processors[processor] = elements
            elements.append(i)

    def format_channel(self, c: Channel) -> str:
            (source, target, initial_tokens) = astuple(c)
            return f"{self.actors[target].name}(k) \u2265 {self.format_execution_time(self.actors[source].execution_time)}{self.actors[source].name}(k{self.format_initial_tokens(initial_tokens)})"

    def systems_of_inequalities(self) -> SystemOfInequalities:        
        return SystemOfInequalities(self.actors, self.channels)
    
    def additional_systems_of_inequalities(self, k: np.ndarray) -> str:
        channels = [Channel(source, target, k[source, target]) for (_, actors) in self.processors.items() for (source, target) in permutations(actors, 2)]
        return "\n".join(map(lambda c: self.format_channel(c), channels))


    def dependencies_to_maxplus_matrix(self, channels: List[Channel]) -> np.ndarray:
        result = np.fromiter(map(lambda _: None, range(4)), dtype=np.object_)
    
    def matrix(self) -> str:
        possible_initial_tokens = set(c.initial_tokens for c in self.channels)
        initial_token_map = { t: c for c in self.channels for t in possible_initial_tokens if c.initial_tokens == t }
    
    def solve(self):
        n_a = len(self.actors)
        problem = LpProblem(sense=LpMaximize)
        throughput = LpVariable("throughput", lowBound=0)
        u = np.array([LpVariable(f"u({a})", lowBound=0) for a in map(lambda a: a.name, self.actors)])
        k = np.array([[LpVariable(f"K({self.actors[i].name},{self.actors[j].name})", cat='Integer') if i < j and self.actors[i].processor == self.actors[j].processor else LpAffineExpression() for i in range(n_a)] for j in range(n_a)])
        k = k + (np.triu(np.ones((n_a, n_a)), k=1) - k.T)

        problem.setObjective(throughput)

        for (source, target, initial_tokens) in map(astuple, self.channels):
            execution_time = self.actors[source].execution_time
            problem += u[target] >= u[source] + throughput*execution_time - initial_tokens

        for (_, actors) in self.processors.items():
            for (source, target) in permutations(actors, 2):
                initial_tokens = k[source, target ]
                execution_time = self.actors[source].execution_time
                problem += (u[target] >= u[source] + throughput*execution_time - initial_tokens)
        problem.solve()

        get_integer = np.vectorize(lambda x: round(x.value()), otypes=[int])
        get_continuous = np.vectorize(lambda x: x.value(), otypes=[float])

        cycle_time = round(1/throughput.value())

        print(self.systems_of_inequalities())
        print("===Additional Constraints===")
        print(self.additional_systems_of_inequalities(get_integer(k)))

        import webbrowser, os
        from urllib.parse import urlencode, urlunparse
        import json

        plot_path = os.path.abspath("./plot/dist/index.html")
        url = urlunparse(
            (
                'file',
                '',
                plot_path,
                '',
                urlencode({
                    'data': json.dumps({
                        'processors': list(self.processors.keys()),
                        'actors': list(map(asdict, self.actors)),
                        'cycle_time': cycle_time,
                        't': np.round(cycle_time * get_continuous(u)).astype(int).tolist()
                    })
                }),
                ''
            )
        )
        webbrowser.open(url, new=2)