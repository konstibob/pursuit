import torch

# Experiment Definitions (1-12)
COMMON_PARAMS = {
    "n_catch": 2,
    "max_cycles": 500,
}

EXPERIMENTS = {
    # 8x8 Experiments
    1: {"id": 1, "name": "8x8_surround_freeze", "x_size": 8, "y_size": 8, "surround": True, "freeze_evaders": True, "n_pursuers": 4, "n_evaders": 15},
    2: {"id": 2, "name": "8x8_surround_active", "x_size": 8, "y_size": 8, "surround": True, "freeze_evaders": False, "n_pursuers": 4, "n_evaders": 15},
    3: {"id": 3, "name": "8x8_touch_freeze", "x_size": 8, "y_size": 8, "surround": False, "freeze_evaders": True, "n_pursuers": 4, "n_evaders": 15},
    4: {"id": 4, "name": "8x8_touch_active", "x_size": 8, "y_size": 8, "surround": False, "freeze_evaders": False, "n_pursuers": 4, "n_evaders": 15},
    
    # 12x12 Experiments
    5: {"id": 5, "name": "12x12_surround_freeze", "x_size": 12, "y_size": 12, "surround": True, "freeze_evaders": True, "n_pursuers": 6, "n_evaders": 22},
    6: {"id": 6, "name": "12x12_surround_active", "x_size": 12, "y_size": 12, "surround": True, "freeze_evaders": False, "n_pursuers": 6, "n_evaders": 22},
    7: {"id": 7, "name": "12x12_touch_freeze", "x_size": 12, "y_size": 12, "surround": False, "freeze_evaders": True, "n_pursuers": 6, "n_evaders": 22},
    8: {"id": 8, "name": "12x12_touch_active", "x_size": 12, "y_size": 12, "surround": False, "freeze_evaders": False, "n_pursuers": 6, "n_evaders": 22},
    
    # 16x16 Experiments
    9: {"id": 9, "name": "16x16_surround_freeze", "x_size": 16, "y_size": 16, "surround": True, "freeze_evaders": True, "n_pursuers": 8, "n_evaders": 30},
    10: {"id": 10, "name": "16x16_surround_active", "x_size": 16, "y_size": 16, "surround": True, "freeze_evaders": False, "n_pursuers": 8, "n_evaders": 30},
    11: {"id": 11, "name": "16x16_touch_freeze", "x_size": 16, "y_size": 16, "surround": False, "freeze_evaders": True, "n_pursuers": 8, "n_evaders": 30},
    12: {"id": 12, "name": "16x16_touch_active", "x_size": 16, "y_size": 16, "surround": False, "freeze_evaders": False, "n_pursuers": 8, "n_evaders": 30},
}

for key in EXPERIMENTS:
    EXPERIMENTS[key].update(COMMON_PARAMS)
    

TOTAL_EPISODES = 50
MAX_CYCLES = COMMON_PARAMS["max_cycles"]
EPSILON_DECAY_STEPS = 25000  # Fixed at 50 episodes * 500 steps for consistent behavior

TRAIN_CONFIG = {
    "lr": 0.0005,             # Reverted to 0.0005 for faster convergence
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 50000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "eval_epsilon": 0.05,
    "epsilon_decay": EPSILON_DECAY_STEPS,
    "target_update_freq": 1000,
    "total_episodes": TOTAL_EPISODES, 
    "eval_episodes": 5,
    "num_evaluations": TOTAL_EPISODES // 25,
    "hidden_dim": 64,
    "mixing_embed_dim": 32,
    "grad_norm_clip": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
