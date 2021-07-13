import os
from datetime import datetime
from typing import cast, List, Optional, Set

import gym
import numpy as np
from gym import ObservationWrapper
from logaut import ltl2dfa
from pylogics.parsers import parse_ltl
from pythomata.core import DFA
from pythomata.impl.symbolic import SymbolicDFA
from temprl.wrapper import TemporalGoalWrapper, TemporalGoal, StepController

from src.strips_to_ltlf import Strips, to_ltlf
from utils.print_strips import print_strips
from utils.qlearning import q_function_learn, q_function_test, load_Q, save_Q, history

env = gym.make('Taxi-v3')


def decode(obs: int):
    """
    Expose the env's decode function for the fluent extraction process.
    """
    return env.decode(obs)


action_fluents = [
    "pickup_g",
    "pickup_r",
    "pickup_y",
    "pickup_b",
    "dropoff_g",
    "dropoff_r",
    "dropoff_y",
    "dropoff_b"
]


def my_step_function(fluents: Set[str]) -> bool:
    """
    Return true if fluents contains at least one of the action fluents
    :param fluents: The fluents at the current timestep
    :return: True if fluents contains at least one of the action fluents
    """
    return len(fluents.intersection(action_fluents)) != 0


def my_fluent_extractor(obs: int,
                        action: int):
    """
    Fluent extraction method.

    :param obs: The env's observation.
    :param action: The action.
    :return: The set of fluents.
    """
    taxi_row, taxi_col, passenger_location, destination = list(decode(obs))

    idx_to_loc = {
        0: '_at_r',
        1: '_at_g',
        2: '_at_y',
        3: '_at_b',
        4: '_on_t'
    }
    actions = {
        4: 'pickup_',
        5: 'dropoff_'
    }
    pos_to_col = {
        (0, 0): 0,
        (0, 4): 1,
        (4, 0): 2,
        (4, 3): 3
    }

    fluents = [f'p{idx_to_loc[passenger_location]}',
               f'd{idx_to_loc[destination]}']

    if history.start:
        fluents.append('start')
        history.start = False

    if action in actions.keys() and (taxi_row, taxi_col) in pos_to_col.keys():
        if action == 4:  # pickup
            if passenger_location == 4 and not history.was_on_taxi:
                fluents.append(f'{actions[action] + idx_to_loc[pos_to_col[(taxi_row, taxi_col)]][-1]}')
                history.was_on_taxi = True
        else:  # dropoff
            if pos_to_col[(taxi_row, taxi_col)] == passenger_location and history.was_on_taxi:
                fluents.append(f'{actions[action] + idx_to_loc[passenger_location][-1]}')
                history.was_on_taxi = False

    return set(fluents)


def make_env_from_dfa(dfa: DFA,
                      goal_reward: float = 100.0) -> gym.Env:
    """
    Make the environment.

    :param dfa: the automaton that constitutes the goal.
    :param goal_reward: the reward associated to the goal.
    :return: the wrapped Gym environment.
    """
    tg = TemporalGoal(automaton=dfa,
                      reward=goal_reward)

    tgw = TemporalGoalWrapper(
        env,
        [tg],
        fluent_extractor=my_fluent_extractor,
        step_controller=StepController(step_func=my_step_function,
                                       allow_first=True)
    )

    # currently, the observation is a pair of (features, list_of_dfa_states)
    # we need to flatten them.
    obs_wrapper = ObservationWrapper(tgw)
    obs_wrapper.observation = lambda obs: tuple([*list(decode(obs[0]))[:4],
                                                 *obs[1]])

    return obs_wrapper


def plot_rewards(rewards: List[int],
                 title: str):
    """
    Plot and save rewards.

    :param rewards: A list of rewards.
    :param title: The title of the figure.
    """
    import matplotlib.pyplot as plt
    os.makedirs('../results', exist_ok=True)

    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(rewards, '-', label=f'Rewards', linewidth=2, color='#33ccff')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    fig.legend(loc='center right')
    # save figure
    plt.savefig(f'../results/{title}.png')


def run_learner(seed: int,
                n_eps: int,
                q_name: Optional[str]):
    """
    Run the learning and testing process.

    :param seed: The random seed.
    :param n_eps: The number of training episodes.
    :param q_name: Optional Q table filename.
    """
    np.random.seed(seed)

    strips = Strips(
        init=[["p_at_r", "d_at_r"], ["p_at_r", "d_at_g"], ["p_at_r", "d_at_y"], ["p_at_r", "d_at_b"],
              ["p_at_g", "d_at_r"], ["p_at_g", "d_at_g"], ["p_at_g", "d_at_y"], ["p_at_g", "d_at_b"],
              ["p_at_y", "d_at_r"], ["p_at_y", "d_at_g"], ["p_at_y", "d_at_y"], ["p_at_y", "d_at_b"],
              ["p_at_b", "d_at_r"], ["p_at_b", "d_at_g"], ["p_at_b", "d_at_y"], ["p_at_b", "d_at_b"]],
        goal=[["p_at_r", "d_at_r"], ["p_at_g", "d_at_g"], ["p_at_y", "d_at_y"], ["p_at_b", "d_at_b"]],
        actions={
            "pickup_g": ("~p_on_t & p_at_g", ["p_on_t"], ["p_at_g"]),
            "pickup_r": ("~p_on_t & p_at_r", ["p_on_t"], ["p_at_r"]),
            "pickup_y": ("~p_on_t & p_at_y", ["p_on_t"], ["p_at_y"]),
            "pickup_b": ("~p_on_t & p_at_b", ["p_on_t"], ["p_at_b"]),
            "dropoff_g": ("p_on_t", ["p_at_g"], ["p_on_t"]),
            "dropoff_r": ("p_on_t", ["p_at_r"], ["p_on_t"]),
            "dropoff_y": ("p_on_t", ["p_at_y"], ["p_on_t"]),
            "dropoff_b": ("p_on_t", ["p_at_b"], ["p_on_t"]),
        }
    )
    formula_str = to_ltlf(strips)
    print(f"Computed formula: {formula_str}")
    formula = parse_ltl(formula_str)
    dfa = ltl2dfa(formula, backend="lydia")
    dfa = cast(SymbolicDFA, dfa)

    fluents = strips.fluents
    actions = set(strips.actions.keys())
    d = print_strips(dfa, fluents, actions.union({"start"}))
    d.render("taxi_strips_plangraph")

    # dfa_dot_file = os.path.join("../dfa_graph")
    # dfa.to_graphviz().render(dfa_dot_file)

    tgw = make_env_from_dfa(dfa)
    tgw.seed(seed)
    if q_name is not None:
        Q = load_Q(name=q_name,
                   n=env.action_space.n)
    else:
        Q, rewards = q_function_learn(tgw, nb_episodes=n_eps)
        save_Q(Q=Q)
        plot_rewards(rewards=rewards,
                     title=f'Train rewards STRIPS')

    # test Q learning
    results = q_function_test(tgw, Q)
    plot_rewards(rewards=results.get('episodic_rewards'),
                 title=f'Test rewards STRIPS')

    t = datetime.now()

    # nicer printout of results
    with open(f"../results/test_{t.strftime('%Y%m%d%H%M%S')}.log", 'w') as f:
        f.write(f'Tests for STRIPS: {formula_str}\n')
        for run in range(len(results.keys()) - 1):
            f.write(f'Episode {run + 1}:')
            states, rewards = results[run].get('states'), results[run].get('rewards')
            tot_reward = 0
            for s, r in zip(states, rewards):
                f.write(f'{s}Reward: {r}\n')
                tot_reward += r
            f.write(f'Episode ended; total reward: {tot_reward}\n')
    print('Test results saved to log file.')

    tgw.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DFA Q Learning Tester')
    parser.add_argument('--seed', type=int, dest='random_seed', default=12345,
                        help='RNG seed for reproducibility (default: 123456)')
    parser.add_argument('--n', type=int, dest='n_episodes', default=2500,
                        help='Number of training episodes (default: 2500)')
    parser.add_argument('--q_values', type=str, dest='q_values', default=None,
                        help='Optional; Starting values of Q Table')
    args = parser.parse_args()

    run_learner(seed=args.random_seed,
                n_eps=args.n_episodes,
                q_name=args.q_values)
