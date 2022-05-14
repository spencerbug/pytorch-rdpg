from copy import deepcopy
import pytest
import gym
import argparse

from main import parse_args
from rdpg import RDPG

def test_update_policy():
    env = gym.make('Pendulum-v1')
    args=parse_args()
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    rdpg = RDPG(env, nb_states, nb_actions, args)
    
    i=10
    state0=None
    while (i > 0):
        i-=1
        if state0 is None:
            state0=deepcopy(env.reset())
        action=rdpg.agent.select_action(state0)
        state, reward, done, info =rdpg.env.step(action)
        state=deepcopy(state)
        rdpg.memory.append(state0, action, reward, done)
        state0=deepcopy(state)
    
    rdpg.agent.reset_lstm_hidden_state(done=False)
    rdpg.update_policy()