"""
This module implements the STRIPS-to-LTLf translation explained
in page 2, second column, of the paper:

    https://www.cs.rice.edu/~vardi/papers/ijcai13.pdf

I.e. in the proof of PSPACE-hardness.

In other words, from a STRIPS problem-domain instance,
build an equivalent LTLf formula such that all the
good plans for the STRIPS problem are traces accepted
by the LTLf formula.

The "start" action is used only at the initial state,
and never used again later in the trace,
so to avoid conflicts with other potential actions.
"""
import itertools
import re
from copy import copy
from typing import Dict, Tuple, List, Set, Optional, Union

LEGAL_FLUENT_NAME = re.compile("[a-z_][a-z0-9_]*")


class Strips:

    def __init__(self,
                 init: List[List[str]],
                 goal: List[List[str]],
                 actions: Dict[str, Tuple[str, List[str], List[str]]],
                 constraints: Optional[List[str]] = None):
        """
        Initialize a Strips domain and problem.

        To check the accepted LTLf format, please
        look at https://marcofavorito.me/tl-grammars/.

        :param init: A list of *positive* literals. The negative literals
          will be constructed automatically.
        :param goal: a list of literal, to be interpreted in conjunction.
        :param actions: a dictionary where:

            - keys are action names
            - values are triples:
                - the first element is the precondition formula
                - the second element is a list of add
                - the third element is a list of del
        :param constraints: a list of formulas, to be considered in conjunction.
        """
        self.init = init
        self.goal = goal
        self.actions = actions
        self.constraints = constraints if constraints else []
        self._check_consistency()
        self._fluents = self._collect_fluents()

    @property
    def fluents(self) -> Set[str]:
        """Get the fluents."""
        return self._fluents

    def _extract_fluent_from_literal(self, literal):
        """Extract the fluent name from the literal."""
        if "~" in literal:
            # if '~' in literal, it must be the first character
            assert literal[0] == "~", "tilde not in first character"
            fluent_name = literal[1:]
        else:
            fluent_name = literal
        if not LEGAL_FLUENT_NAME.fullmatch(fluent_name):
            raise ValueError(f"fluent name '{fluent_name}' does not comply with regex '{LEGAL_FLUENT_NAME.pattern}'.")

        return fluent_name

    def _collect_fluents(self):
        """Compute the fluents."""
        result = set()
        result = result.union(map(self._extract_fluent_from_literal, list(set([item for sublist in self.init for item in sublist]))))
        result = result.union(map(self._extract_fluent_from_literal, list(set([item for sublist in self.goal for item in sublist]))))
        for action, (_precondition, adds, dels) in self.actions.items():
            result = result.union(map(self._extract_fluent_from_literal, adds))
            result = result.union(map(self._extract_fluent_from_literal, dels))
        return result

    def _check_consistency(self):
        assert len(self.actions) > 0, "at least one action"
        assert len(self.init) > 0, "at least one clause"
        assert len(self.goal) > 0, "at least one literal in the goal"
        assert all(len(clause) > 0 for clause in self.init), "some clause is empty"


def to_ltlf(problem: Strips):
    """Transform a STRIPS problem to LTLf."""
    conjunction = []
    all_fluents = problem.fluents

    # the "start" action is a dummy
    start_action = "start"
    conjunction.append("G(start -> X(G(~start)))")

    # initial conjunction includes "start" action
    initial_condition_formulas = []
    for initial_condition in problem.init:
        negated_fluents = all_fluents.difference(initial_condition)
        occurring_part = " & ".join(initial_condition)
        negative_part = " & ".join([f"~{neg}" for neg in negated_fluents])
        extended_initial_condition = f"({occurring_part} & {negative_part})"
        initial_condition_formulas.append(extended_initial_condition)
    full_initial_conditions = " | ".join(initial_condition_formulas)
    initial_conditions_and_start = f"{start_action} & ({full_initial_conditions})"
    conjunction.append(initial_conditions_and_start)

    # eventually, get the goal
    goals = [f"({' & '.join([f for f in x])})" for x in problem.goal]
    goal = f"F({' | '.join(goals)})"
    conjunction.append(goal)

    for action, (precondition, adds, dels) in problem.actions.items():
        unused_fluents = copy(all_fluents)
        # add precondition for action
        conjunction.append(f"G(X[!]({action}) -> ({precondition}))")

        # add "add" and "del" effects for action
        dels_negated = ["~" + del_atom for del_atom in dels]
        adds_and_dels_string = " & ".join(adds + dels_negated)
        conjunction.append(f"G(X[!]({action}) -> X[!]({adds_and_dels_string}))")
        unused_fluents = unused_fluents.difference(adds)
        unused_fluents = unused_fluents.difference(dels)

        # add the frame axion
        if len(unused_fluents) > 0:
            conjunction.append(f"G(X[!]({action}) ->({' & '.join([f'({unused} <-> X[!]({unused}))' for unused in unused_fluents])}))")

    # add condition of "exactly one action"
    all_actions = list(problem.actions.keys()) + ["start"]
    at_least_one_action = " | ".join(all_actions)
    at_most_one_action_conditions = []
    for (action_1, action_2) in itertools.product(all_actions, all_actions):
        if action_1 == action_2:
            continue
        at_most_one_action_conditions.append(f"({action_1} -> ~{action_2})")

    at_most_one_action = " & ".join(at_most_one_action_conditions)
    exactly_one_action = f"G(({at_least_one_action}) & ({at_most_one_action}))"
    conjunction.append(exactly_one_action)

    if len(problem.constraints) > 0:
        conjunction.extend(problem.constraints)

    return " & ".join(conjunction)
