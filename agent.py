"""
QMIX Agent Logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from networks import AgentNetwork, QMatrixMixer
from config import TRAIN_CONFIG

class QMIXAgent:
    def __init__(self, obs_dim, state_dim, n_agents, n_actions):
        self.config = TRAIN_CONFIG
        self.device = torch.device(self.config["device"])
        
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        # Initialize Networks
        self.agent_net = AgentNetwork(obs_dim, self.config["hidden_dim"], n_actions).to(self.device)
        self.target_agent_net = AgentNetwork(obs_dim, self.config["hidden_dim"], n_actions).to(self.device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        
        self.mixer = QMatrixMixer(n_agents, state_dim, self.config["mixing_embed_dim"]).to(self.device)
        self.target_mixer = QMatrixMixer(n_agents, state_dim, self.config["mixing_embed_dim"]).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # Optimizer (optimize both agent net and mixer)
        self.optimizer = optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()), 
            lr=self.config["lr"]
        )
        
        self.steps_done = 0
        self.epsilon = self.config["epsilon_start"]
        
    def select_actions(self, obs):
        """
        Epsilon-greedy action selection.
        obs: (n_agents, obs_dim) numpy array
        """
        self.update_epsilon()
        
        if random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions, self.n_agents)
        
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(self.device)
            q_vals = self.agent_net(obs_t)
            actions = q_vals.argmax(dim=1).cpu().numpy()
            return actions

    def update_epsilon(self):
        decay_steps = self.config["epsilon_decay"]
        end_eps = self.config["epsilon_end"]
        start_eps = self.config["epsilon_start"]
        
        if self.steps_done < decay_steps:
            self.epsilon = start_eps - (start_eps - end_eps) * (self.steps_done / decay_steps)
        else:
            self.epsilon = end_eps
            
    def train(self, buffer):
        if len(buffer) < self.config["batch_size"]:
            return 0.0
            
        self.steps_done += 1
        
        # Sample Batch
        batch = buffer.sample(self.config["batch_size"])
        obs, state, actions, reward, next_obs, next_state, done = batch
        
        # Convert to Tensor
        obs = torch.FloatTensor(obs).to(self.device)          # (B, N, O)
        state = torch.FloatTensor(state).to(self.device)      # (B, S)
        actions = torch.LongTensor(actions).to(self.device)   # (B, N)
        reward = torch.FloatTensor(reward).to(self.device).reshape(-1, 1)    # (B, 1)
        next_obs = torch.FloatTensor(next_obs).to(self.device)# (B, N, O)
        next_state = torch.FloatTensor(next_state).to(self.device) # (B, S)
        done = torch.FloatTensor(done).to(self.device).reshape(-1, 1)        # (B, 1)
        
        # --- Computations ---
        
        # 1. Get current Q_tot
        # Get Q values for all actions: (B, N, n_actions)
        q_vals = self.agent_net(obs) 
        # Select Q values for chosen actions: (B, N, 1)
        chosen_action_q_vals = q_vals.gather(2, actions.unsqueeze(2)).squeeze(2)
        # Mix them: (B, 1)
        q_tot = self.mixer(chosen_action_q_vals, state)
        
        # 2. Get Target Q_tot (Double DQN style)
        with torch.no_grad():
            # Get next Q values from target network
            target_q_vals = self.target_agent_net(next_obs)
            # Use online network to select best actions (Double DQN)
            # argmax over actions dim
            best_actions = self.agent_net(next_obs).argmax(dim=2).unsqueeze(2)
            # Gather target Qs
            target_chosen_q_vals = target_q_vals.gather(2, best_actions).squeeze(2)
            # Mix target Qs
            target_q_tot = self.target_mixer(target_chosen_q_vals, next_state)
            
            # Bellman Target
            loss_target = reward + self.config["gamma"] * (1 - done) * target_q_tot
            
        # 3. Loss
        loss = F.mse_loss(q_tot, loss_target)
        
        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_net.parameters(), self.config["grad_norm_clip"])
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.config["grad_norm_clip"])
        self.optimizer.step()
        
        # 5. Update Target Networks
        if self.steps_done % self.config["target_update_freq"] == 0:
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            
        return loss.item()
