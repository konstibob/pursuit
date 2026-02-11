"""
Environment wrapper for PettingZoo pursuit_v4.
Handles observation flattening and global state construction.
"""

import numpy as np
import torch
from pettingzoo.sisl import pursuit_v4
from config import ENV_CONFIG

class PursuitEnvWrapper:
    def __init__(self):
        self.env = pursuit_v4.parallel_env(**ENV_CONFIG)
        self.env.reset()
        
        # Get dimensions
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        
        # Action space (Discrete)
        self.n_actions = self.env.action_space(self.agents[0]).n
        
        # Observation space (7, 7, 3) -> Flattened to 147
        sample_obs = self.env.observation_space(self.agents[0]).shape
        self.obs_dim = np.prod(sample_obs)
        
        # State space (Global State) = Concatenation of all agent observations
        self.state_dim = self.obs_dim * self.n_agents
        
    def reset(self):
        obs_dict, _ = self.env.reset()
        return self._process_obs(obs_dict)

    def step(self, actions):
        """
        Step environment with a list/array of actions.
        actions: list or array of shape (n_agents,)
        """
        # Convert list of actions to dict for PettingZoo
        act_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        
        obs_dict, rewards_dict, terms, truncs, infos = self.env.step(act_dict)
        
        # Process observations and state
        obs, state = self._process_obs(obs_dict)
        
        # Extract rewards (shared global reward)
        # In pursuit_v4, rewards are shared if configured, but let's take mean or sum
        # usually they are identical for shared_reward=True
        reward = sum(rewards_dict.values()) / self.n_agents
        
        # Check done
        done = any(terms.values()) or any(truncs.values())
        
        return obs, state, reward, done, infos

    def _process_obs(self, obs_dict):
        """
        Convert dict of observations to:
        1. obs: (n_agents, obs_dim) array
        2. state: (state_dim,) array (flattened global state)
        """
        obs_list = []
        for agent in self.agents:
            # Flatten observation: (7, 7, 3) -> (147,)
            flat_o = obs_dict[agent].flatten()
            obs_list.append(flat_o)
            
        obs = np.array(obs_list, dtype=np.float32)
        state = obs.flatten()  # Concatenate all agent obs for global state
        
        return obs, state

    def close(self):
        self.env.close()
