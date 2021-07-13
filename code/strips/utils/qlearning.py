# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Marco Favorito
#
# ------------------------------
#
# This file is part of temprl.
#
# temprl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# temprl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with temprl.  If not, see <https://www.gnu.org/licenses/>.
#

"""Naive implementation of Q-learning."""
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, List, Optional

import gym
import numpy as np


class History:
    """ Shared history object; needed for fluents generation """

    def __init__(self):
        self.was_on_taxi = False
        self.start = True


history = History()

logging.basicConfig(level=logging.DEBUG)


def q_function_learn(
        env: gym.Env, nb_episodes=100, alpha=0.1, eps=0.1, gamma=0.9
) -> Tuple[Dict[Any, np.ndarray], List[int]]:
    """
    Learn a Q-function from a Gym env using vanilla Q-Learning.

    :param env: the environment
    :param nb_episodes: the number of episodes
    :param alpha: the learning rate
    :param eps: the epsilon parameter in eps-greedy exploration
    :param gamma: the discount factor
    :returns: the Q function, a dictionary from states to array of Q values for every action.
    """
    global history
    nb_actions = env.action_space.n
    Q: Dict[Any, np.ndarray] = defaultdict(
        lambda: np.random.randn(
            nb_actions,
        )
                * 0.01
    )
    rewards = []

    def choose_action(state):
        if np.random.random() < eps:
            return np.random.randint(0, nb_actions)
        else:
            return np.argmax(Q[state])

    for i in range(nb_episodes):
        history.was_on_taxi = False
        history.start = True
        state = env.reset()
        done = False
        episodic_reward = 0
        while not done:
            action = choose_action(state)
            state2, reward, done, info = env.step(action)
            Q[state][action] += alpha * (
                    reward + gamma * np.max(Q[state2]) - Q[state][action]
            )
            state = state2
            episodic_reward += reward
        rewards.append(episodic_reward)
        logging.debug(f"Episode {i} Done; Episode reward: {episodic_reward}")
    return Q, rewards


def q_function_test(
        env: gym.Env, Q: Dict[Any, np.ndarray], nb_episodes=10
) -> np.ndarray:
    """
    Test a Q-function against a Gym env.

    :param env: the environment
    :param Q: the action-value function
    :param nb_episodes: the number of episodes
    :returns: a list of rewards collected for every episode.
    """
    results = {
        'episodic_rewards': []
    }
    for i in range(nb_episodes):
        results[i] = {
            'states': [],
            'rewards': []
        }
        history.was_on_taxi = False
        history.start = True
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[state])
            state2, reward, done, info = env.step(action)
            total_reward += reward
            state = state2

            results[i]['states'].append(env.render(mode='ansi'))
            results[i]['rewards'].append(reward)
        results['episodic_rewards'].append(total_reward)

    return results


def wrap_observation(
        env: gym.Env, observation_space: gym.spaces.Space, observe: Callable
):
    """Wrap a Gym environment with an observation wrapper."""

    class _wrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = observation_space

        def observation(self, observation):
            return observe(observation)

    return _wrapper(env)


def save_Q(Q: Dict[Any, np.array]):
    """ Save the Q table. """
    import pickle
    with open('learned_q_keys.pkl', 'wb') as f:
        pickle.dump(list(Q.keys()), f)
    with open('learned_q_values.pkl', 'wb') as f:
        pickle.dump(list(Q.values()), f)


def load_Q(name: str,
           n: int) -> Dict[Any, np.array]:
    """ Load the Q table from file. """
    import pickle
    ks, vs = None, None
    with open(f'{name}_keys.pkl', 'rb') as f:
        ks = pickle.load(f)
    with open(f'{name}_values.pkl', 'rb') as f:
        vs = pickle.load(f)
    Q: Dict[Any, np.ndarray] = defaultdict(
        lambda: np.random.randn(n, ) * 0.01
    )
    for k, v in zip(ks, vs):
        Q[k] = v
    return Q
