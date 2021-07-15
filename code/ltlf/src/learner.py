import os
from datetime import datetime
from typing import Dict, Any, Optional, List

import gym
import numpy as np
from flloat.parser.ltlf import LTLfParser
from flloat.semantics import PLInterpretation
from pythomata.dfa import DFA
from temprl.wrapper import TemporalGoalWrapper, TemporalGoal

from utils.qlearning import q_function_learn, q_function_test, save_Q, load_Q
from goals import *

env = gym.make('Taxi-v3')


def decode(obs: int):
    """
    Expose the env's decode function for the fluent extraction process.
    """
    return env.decode(obs)


def fluent_extraction(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    """ Globally accessible method. """
    pass


available_goals = {
    'env_base': {
        'formula': 'a U (b U c)',
        'fluent_extractor': env_base_goal
    },
    'through_center': {
        'formula': 'a U (b U (c U d))',
        'fluent_extractor': pass_through_center
    },
    'through_1corner': {
        'formula': 'a U (b U (c U d))',
        'fluent_extractor': pass_through_1_corner
    },
    'through_2corners': {
        'formula': 'a U (b U (c U (d U e)))',
        'fluent_extractor': pass_through_2_corners
    },
    'through_3corners': {
        'formula': 'a U (b U (c U (d U (e U f))))',
        'fluent_extractor': pass_through_3_corners
    }
}


def my_fluent_extractor(obs: int,
                        action: int) -> PLInterpretation:
    """
    Fluent extraction wrapper method.

    :param obs: The env's observation.
    :param action: The action.
    :return: The set of fluents.
    """
    taxi_row, taxi_col, passenger_location, destination = list(decode(obs))

    pos_to_colors = {
        (0, 0): 'Red',
        (0, 4): 'Green',
        (4, 0): 'Yellow',
        (4, 3): 'Blue'
    }
    idx_colors = {
        0: 'Red',
        1: 'Green',
        2: 'Yellow',
        3: 'Blue',
        4: 'In_Taxi'
    }

    fluent = fluent_extraction(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors)

    return PLInterpretation({fluent})


def make_env_from_dfa(dfa: DFA,
                      goal_reward: float = 5000.0,
                      reward_shaping: bool = True) -> gym.Env:
    """
    Make the environment.

    :param dfa: the automaton that constitutes the goal.
    :param goal_reward: the reward associated to the goal.
    :param reward_shaping: apply automata-based reward shaping.
    :return: the wrapped Gym environment.
    """
    tg = TemporalGoal(automaton=dfa,
                      reward=goal_reward,
                      reward_shaping=reward_shaping,
                      zero_terminal_state=False,
                      extract_fluents=my_fluent_extractor)

    tgw = TemporalGoalWrapper(
        env,
        [tg],
        combine=lambda obs, qs: tuple((*obs, *qs)),
        feature_extractor=(lambda obs, action: (obs,))
    )

    return tgw


def plot_rewards(rewards: List[int],
                 title: str) -> None:
    """
    Plot and save rewards.

    :param rewards: A list of rewards.
    :param title: The title of the figure.
    """
    import matplotlib.pyplot as plt
    os.makedirs('../results', exist_ok=True)

    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(rewards, '-', label=f'Rewards', linewidth=1, color='#33ccff')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    fig.legend(loc='center right')
    # save figure
    plt.savefig(f'../results/{title}.png')


def run_learner(seed: int,
                n_eps: int,
                q_name: Optional[str],
                selected_goal: str,
                reward_shaping: bool) -> None:
    """
    Run the learning and testing process.

    :param seed: The random seed.
    :param n_eps: The number of training episodes.
    :param q_name: Optional Q table filename.
    :param selected_goal: The goal we train for.
    :param reward_shaping: Whether to apply reward shaping or not.
    """
    global fluent_extraction
    np.random.seed(seed)

    # parse the formula
    parser = LTLfParser()
    formula = available_goals.get(selected_goal).get('formula', None)
    fluent_extraction = available_goals.get(selected_goal).get('fluent_extractor', None)
    parsed_formula = parser(formula)
    dfa = parsed_formula.to_automaton()

    dfa_dot_file = os.path.join(f"../results/{selected_goal}")
    dfa.to_dot(dfa_dot_file)

    tgw = make_env_from_dfa(dfa, reward_shaping=reward_shaping)
    tgw.seed(seed)

    if q_name is not None:
        Q = load_Q(name=q_name,
                   n=env.action_space.n)
    else:
        Q, train_rewards = q_function_learn(tgw, nb_episodes=n_eps, eps=0.9)
        save_Q(Q=Q)
        plot_rewards(rewards=train_rewards,
                     title=f'Train rewards {selected_goal}' + (
                         ' (reward shaping)' if reward_shaping else ''))

    # test Q learning
    results = q_function_test(tgw, Q)
    plot_rewards(rewards=results.get('episodic_rewards'),
                 title=f'Test.png rewards {selected_goal}' + (' (reward shaping)' if reward_shaping else ''))

    t = datetime.now()

    # nicer printout of results
    with open(f"../results/test_{t.strftime('%Y%m%d%H%M%S')}.log", 'w') as f:
        f.write(f'Tests for {selected_goal}: {formula}\n')
        f.write(f'Reward shaping: {reward_shaping}\n')
        f.write(f'N steps: {n_eps}\n')
        for run in range(len(results.keys()) - 1):
            f.write(f'Episode {run + 1}:')
            states, rewards = results[run].get('states'), results[run].get('rewards')
            tot_reward = 0
            for s, r in zip(states, rewards):
                f.write(f'{s}Reward: {r}\n')
                tot_reward += r
            f.write(f'Episode ended; total reward: {tot_reward}\n')
    print('Test.png results saved to log file.')

    tgw.close()

    import json
    with open(f'../results/{selected_goal}' + (' (reward shaping)' if reward_shaping else '') + '.json', 'w') as f:
        json.dump({
            'train_rewards': train_rewards,
            'test_rewards': results.get('episodic_rewards')
        }, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DFA Q Learning Tester')
    parser.add_argument('--seed', type=int, dest='random_seed', default=12345,
                        help='RNG seed for reproducibility (default: 123456)')
    parser.add_argument('--n', type=int, dest='n_episodes', default=2500,
                        help='Number of training episodes (default: 2500)')
    parser.add_argument('--q_values', type=str, dest='q_values', default=None,
                        help='Optional; Starting values of Q Table')
    parser.add_argument('--goal', type=str, dest='goal', default='env_base',
                        help=f'Temporal Goal. One of {list(available_goals.keys())}')
    parser.add_argument('--shape_rewards', type=bool, dest='reward_shaping', default=False,
                        help='Toggle reward shaping (default: False)')
    args = parser.parse_args()

    # run_learner(seed=args.random_seed,
    #             n_eps=args.n_episodes,
    #             q_name=args.q_values,
    #             selected_goal=args.goal,
    #             reward_shaping=args.reward_shaping)

    # print(f'Running env_base with no Reward Shaping...')
    # run_learner(seed=42,
    #             n_eps=50000,
    #             q_name=None,
    #             selected_goal='env_base',
    #             reward_shaping=False)
    # print(f'Running env_base with Reward Shaping...')
    # run_learner(seed=42,
    #             n_eps=50000,
    #             q_name=None,
    #             selected_goal='env_base',
    #             reward_shaping=True)
    #
    # print(f'Running through_center with no Reward Shaping...')
    # run_learner(seed=42,
    #             n_eps=50000,
    #             q_name=None,
    #             selected_goal='through_center',
    #             reward_shaping=False)
    # print(f'Running through_center with Reward Shaping...')
    # run_learner(seed=42,
    #             n_eps=50000,
    #             q_name=None,
    #             selected_goal='through_center',
    #             reward_shaping=True)
    #
    # print(f'Running through_1corner with no Reward Shaping...')
    # run_learner(seed=42,
    #             n_eps=50000,
    #             q_name=None,
    #             selected_goal='through_1corner',
    #             reward_shaping=False)
    # print(f'Running through_1corner with Reward Shaping...')
    # run_learner(seed=42,
    #             n_eps=50000,
    #             q_name=None,
    #             selected_goal='through_1corner',
    #             reward_shaping=True)

    # print(f'Running through_2corners with no Reward Shaping...')
    # run_learner(seed=42,
    #             n_eps=50000,
    #             q_name=None,
    #             selected_goal='through_2corners',
    #             reward_shaping=False)
    # print(f'Running through_2corners with Reward Shaping...')
    # run_learner(seed=42,
    #             n_eps=50000,
    #             q_name=None,
    #             selected_goal='through_2corners',
    #             reward_shaping=True)

    print(f'Running through_3corners with no Reward Shaping...')
    run_learner(seed=42,
                n_eps=50000,
                q_name=None,
                selected_goal='through_3corners',
                reward_shaping=False)
    print(f'Running through_3corners with Reward Shaping...')
    run_learner(seed=42,
                n_eps=50000,
                q_name=None,
                selected_goal='through_3corners',
                reward_shaping=True)
