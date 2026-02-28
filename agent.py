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
        
        self.agent_ids = torch.eye(n_agents).to(self.device)
        
        self.agent_net = AgentNetwork(obs_dim + n_agents, self.config["hidden_dim"], n_actions).to(self.device)
        self.target_agent_net = AgentNetwork(obs_dim + n_agents, self.config["hidden_dim"], n_actions).to(self.device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        
        self.mixer = QMatrixMixer(n_agents, state_dim, self.config["mixing_embed_dim"]).to(self.device)
        self.target_mixer = QMatrixMixer(n_agents, state_dim, self.config["mixing_embed_dim"]).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        self.optimizer = optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()), 
            lr=self.config["lr"]
        )
        
        self.steps_done = 0
        self.epsilon = self.config["epsilon_start"]
        
    def select_actions(self, obs, evaluate=False, eval_epsilon=0.0):
        if not evaluate:
            self.update_epsilon()
            eps = self.epsilon
        else:
            eps = eval_epsilon
            
        if random.random() < eps:
            return np.random.randint(0, self.n_actions, self.n_agents)
        
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(self.device)
            input_t = torch.cat([obs_t, self.agent_ids], dim=-1)
            q_vals = self.agent_net(input_t)
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
        batch = buffer.sample(self.config["batch_size"])
        obs, state, actions, reward, next_obs, next_state, done = batch
        bs = self.config["batch_size"]
        
        obs = torch.FloatTensor(obs).to(self.device)
        state = torch.FloatTensor(state).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).reshape(-1, 1)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).reshape(-1, 1)
        
        ids = self.agent_ids.unsqueeze(0).expand(bs, -1, -1)
        
        q_input = torch.cat([obs, ids], dim=-1)
        q_vals = self.agent_net(q_input)
        chosen_action_q_vals = q_vals.gather(2, actions.unsqueeze(2)).squeeze(2)
        q_tot = self.mixer(chosen_action_q_vals, state)
        
        with torch.no_grad():
            next_q_input = torch.cat([next_obs, ids], dim=-1)
            target_q_vals = self.target_agent_net(next_q_input)
            best_actions = self.agent_net(next_q_input).argmax(dim=2).unsqueeze(2)
            target_chosen_q_vals = target_q_vals.gather(2, best_actions).squeeze(2)
            target_q_tot = self.target_mixer(target_chosen_q_vals, next_state)
            loss_target = reward + self.config["gamma"] * (1 - done) * target_q_tot
            
        loss = F.mse_loss(q_tot, loss_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_net.parameters(), self.config["grad_norm_clip"])
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.config["grad_norm_clip"])
        self.optimizer.step()
        
        if self.steps_done % self.config["target_update_freq"] == 0:
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            
        return loss.item()

    def save_model(self, path):
        torch.save({
            "agent_net": self.agent_net.state_dict(),
            "mixer": self.mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_net.load_state_dict(checkpoint["agent_net"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        
        # Update target nets as well
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
