"""
Configuration for custom QMIX on PettingZoo pursuit_v4.
"""

import torch

# Environment Configuration (User Specified)
ENV_CONFIG = {
    "x_size": 8,
    "y_size": 8,
    #"render_mode": "human", 
    "n_pursuers": 4,
    "n_evaders": 15,
    "n_catch": 2,
    "freeze_evaders": True, #freeze_evaders: False for harder experiment
    "surround": True,   #Surround: True for harder experiment
}

# Training Hyperparameters
TRAIN_CONFIG = {
    "lr": 0.0005,
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 50000,
    "epsilon_start": 0.1,
    "epsilon_end": 0.01,
    "epsilon_decay": 20000,  # Decay over first X steps
    "target_update_freq": 200,  # Update target network every X steps
    "total_episodes": 100,
    "hidden_dim": 64,
    "mixing_embed_dim": 32,
    "grad_norm_clip": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
