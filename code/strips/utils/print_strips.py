from collections import deque
from typing import Set

import sympy
from graphviz import Digraph
from pythomata.impl.symbolic import SymbolicDFA
from sympy import satisfiable


def find_sink_states(dfa: SymbolicDFA):
    """Find sink states."""
    result = set()
    for s in filter(lambda x: not dfa.is_accepting(x), dfa.states):
        transitions = list(dfa.get_transitions_from(s))
        if len(transitions) != 1:
            continue
        start, guard, end = transitions[0]
        if isinstance(guard, sympy.logic.boolalg.BooleanTrue) and end == s:
            result.add(s)
    return result


def from_fluents_to_string(fluents: Set[str]):
    return "\n".join(sorted(fluents))


def get_model(expr) -> Set[str]:
    """Get model."""
    models = list(satisfiable(expr, all_models=True))
    assert len(models) == 1, f"Expected one model, found {len(models)}"
    return {str(prop) for prop, value in models[0].items() if value}


def print_strips(dfa: SymbolicDFA, fluents: Set[str], actions: Set[str]) -> Digraph:
    """Print STRIPS."""
    graph = Digraph()

    index_to_fluents_str = {0: "init"}
    # check that the above is a injective mapping!
    # that is, each fluents configuration is associated to
    # exactly one DFA state
    fluents_str_to_index = {"init": 0}
    sink_states = find_sink_states(dfa)

    rejecting_states = set()
    transitions = set()

    # breadth-first exploration
    # why? Because we only know the "meaning" of a state
    # from the transitions leading to it
    queue = deque()
    visited = set()
    to_be_visited = set()
    queue.append(0)
    while len(queue) != 0:
        current_state = queue.pop()
        visited.add(current_state)
        to_be_visited.discard(current_state)
        for start, guard, end in dfa.get_transitions_from(current_state):

            # if destination sink state, ignore it
            if end in sink_states:
                visited.add(end)
                tail = index_to_fluents_str[start]
                graph.edge(tail, "sink", label="else")
                continue

            # else, get the action and the fluents from the guards
            tail = index_to_fluents_str[start]
            atoms = get_model(guard)
            current_actions = atoms.intersection(actions)
            assert len(current_actions) == 1, f"Expected one action, found {len(current_actions)}"
            action = next(iter(current_actions))
            current_fluents = atoms.difference(actions)
            assert current_fluents.issubset(fluents)

            fluents_str = from_fluents_to_string(current_fluents)
            if end not in visited and end not in to_be_visited:
                to_be_visited.add(end)
                queue.append(end)
                index_to_fluents_str[end] = fluents_str
                fluents_str_to_index[fluents_str] = end

                if not dfa.is_accepting(end):
                    # at least one occurrence as non-accepting
                    # state is enough to label this state as
                    # rejecting.
                    rejecting_states.add(fluents_str)

            if (tail, action, fluents_str) not in transitions:
                graph.edge(tail, fluents_str, label=action)
                transitions.add((tail, action, fluents_str))

    for fluents_str in fluents_str_to_index:
        if fluents_str not in rejecting_states:
            graph.node(fluents_str, shape="doublecircle")
    return graph
