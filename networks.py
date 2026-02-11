"""
Neural Networks for QMIX (Agent Network + Mixing Network).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    """
    Individual Agent Q-Network.
    Input: Observation
    Output: Q-values for each action
    """
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class QMatrixMixer(nn.Module):
    """
    QMIX Mixing Network.
    Input: Agent Q-values, Global State
    Output: Q_tot
    """
    def __init__(self, n_agents, state_dim, embed_dim):
        super(QMatrixMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        
        # Hypernetworks to generate weights for mixing network
        # Conditioned on state
        
        # Layer 1 weights (output shape: n_agents * embed_dim)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        # Layer 1 bias
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        
        # Layer 2 weights (output shape: embed_dim * 1)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        # Layer 2 bias (output shape: 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, agent_qs, states):
        """
        agent_qs: (batch_size, n_agents)
        states: (batch_size, state_dim)
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        
        # Layer 1
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Layer 2
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states)
        b2 = b2.view(-1, 1, 1)
        
        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.view(bs, -1)
        return q_tot
