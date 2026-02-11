"""
Replay Buffer for storing transitions.
"""

import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, state_dim, n_agents, n_actions):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, state, actions, reward, next_obs, next_state, done):
        """
        Store a transition.
        obs: (n_agents, obs_dim)
        state: (state_dim)
        actions: (n_agents)
        reward: float
        next_obs: (n_agents, obs_dim)
        next_state: (state_dim)
        done: bool
        """
        transition = (obs, state, actions, reward, next_obs, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack batch
        obs_batch = np.array([x[0] for x in batch])
        state_batch = np.array([x[1] for x in batch])
        actions_batch = np.array([x[2] for x in batch])
        reward_batch = np.array([x[3] for x in batch])
        next_obs_batch = np.array([x[4] for x in batch])
        next_state_batch = np.array([x[5] for x in batch])
        done_batch = np.array([x[6] for x in batch])
        
        return (obs_batch, state_batch, actions_batch, reward_batch, 
                next_obs_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)
