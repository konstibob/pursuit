"""
Main Training Loop for QMIX on PettingZoo Pursuit.
Includes interactive experiment selection and existing model detection.
"""

from environment import PursuitEnvWrapper
from buffer import ReplayBuffer
from agent import QMIXAgent
from logger import ExperimentLogger
from config import TRAIN_CONFIG, EXPERIMENTS, COMMON_PARAMS
import numpy as np
import time
import sys
import torch
import os

def run_evaluation(env, agent, logger, eval_id, training_episode):
    """
    Run an evaluation phase and log results.
    Returns average reward.
    """
    logger.start_evaluation_phase(eval_id)
    eval_episodes = TRAIN_CONFIG.get("eval_episodes", 20)
    
    total_phase_reward = 0
    
    # print(f"\n[Evaluation {eval_id}] Starting (Episode {training_episode})...")
    
    # Use eval_epsilon from config (default 0.0 if not set)
    eval_eps = TRAIN_CONFIG.get("eval_epsilon", 0.0)
    
    for ep in range(1, eval_episodes + 1):
        obs, state = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done:
            actions = agent.select_actions(obs, evaluate=True, eval_epsilon=eval_eps)
            next_obs, next_state, reward, done, _ = env.step(actions)
            obs = next_obs
            state = next_state
            ep_reward += reward
            steps += 1
            
        logger.log_evaluation_step(ep, steps, ep_reward)
        total_phase_reward += ep_reward
        
    avg_reward = total_phase_reward / eval_episodes
    print(f"[Evaluation {eval_id}] Completed. Avg Reward: {avg_reward:.4f}")
    
    return avg_reward

def select_experiment():
    print("\n" + "="*80)
    print(" Select Experiment Configuration:")
    print("="*80)
    print(f"{'ID':<4} | {'Name':<25} | {'Grid':<6} | {'Purs/Evad':<10} | {'Surround':<8} | {'Freeze':<8}")
    print("-" * 80)
    
    for exp_id, conf in EXPERIMENTS.items():
        print(f"{exp_id:<4} | {conf['name']:<25} | {conf['x_size']}x{conf['y_size']:<3} | {conf['n_pursuers']}/{conf['n_evaders']:<7} | {str(conf['surround']):<8} | {str(conf['freeze_evaders']):<8}")
    print("-" * 80)
    print(f"{'0':<4} | Run All Experiments (1-12) Sequentially")
    print("=" * 80)
    
    while True:
        try:
            selection = input("\nEnter Experiment ID (0-12): ")
            exp_id = int(selection)
            if exp_id == 0:
                return 0
            if exp_id in EXPERIMENTS:
                return exp_id
            else:
                print("Invalid ID. Please enter a number between 0 and 12.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main(override_exp_id=None):
    # 0. Rendering Prompt (Interactive mode only)
    render_mode = None
    if override_exp_id is None:
        render_input = input("Do you want to view the experiment (render window)? (y/n): ").lower()
        if render_input == 'y':
            render_mode = "human"
            print("[System] Rendering enabled.")
        else:
            print("[System] Rendering disabled.")

    # 1. Experiment Selection (Interactive or Batch Override)
    if override_exp_id is not None:
        if override_exp_id in EXPERIMENTS:
            exp_ids = [override_exp_id]
        else:
            print(f"Invalid Override ID: {override_exp_id}")
            return
    else:
        selection = select_experiment()
        if selection == 0:
            exp_ids = sorted(EXPERIMENTS.keys())
            print(f"\n[System] Running ALL {len(exp_ids)} experiments sequentially.")
        else:
            exp_ids = [selection]
    
    # --- EXPERIMENT LOOP ---
    for exp_id in exp_ids:
        exp_config = EXPERIMENTS[exp_id].copy()
        exp_config['render_mode'] = render_mode
        print(f"\n" + "#"*40)
        print(f" Starting Experiment {exp_id}: {exp_config['name']}")
        print("#"*40)

        # Check for existing model
        existing_dir = ExperimentLogger.check_existing(exp_config["name"])
        
        start_fresh = True
        resume_path = None
        
        if existing_dir and override_exp_id is None:
            print(f"\n[System] Found existing trained model in: {existing_dir}")
            while True:
                choice = input("Do you want to (e)valuate this model or (t)rain a new one? (e/t): ").lower()
                if choice == 'e':
                    start_fresh = False
                    resume_path = existing_dir
                    print("[System] Loading existing model for evaluation...")
                    break
                elif choice == 't':
                    start_fresh = True
                    print("[System] Starting fresh training...")
                    break
                else:
                    print("Invalid choice. Please enter 'e' or 't'.")
        elif existing_dir and override_exp_id is not None:
            start_fresh = True
            print(f"[System] Batch mode: Overwriting existing directory {existing_dir}")
        
        # 2. Initialize Logger
        if start_fresh:
            logger = ExperimentLogger(exp_config) # New folder
        else:
            logger = ExperimentLogger(exp_config, resume_path=resume_path) # Use existing folder
        
        # 3. Initialize Environment
        env = PursuitEnvWrapper(config=exp_config)
        
        agent = QMIXAgent(
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            n_agents=env.n_agents,
            n_actions=env.n_actions
        )
        
        if not start_fresh:
            model_path = os.path.join(resume_path, "model.pt")
            if os.path.exists(model_path):
                agent.load_model(model_path)
                print(f"Loaded model from {model_path}")
            else:
                print(f"Error: model.pt not found in {resume_path}. Starting fresh.")
                start_fresh = True
                
        if start_fresh:
            buffer = ReplayBuffer(
                capacity=TRAIN_CONFIG["buffer_size"],
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                n_agents=env.n_agents,
                n_actions=env.n_actions
            )
        
        print(f"\nConfiguration:")
        print(f"  Agents: {env.n_agents}")
        print(f"  Grid: {exp_config['x_size']}x{exp_config['y_size']}")
        print(f"  Device: {agent.device}")
        print(f"  Render Mode: {render_mode}")
        
        # --- TRAINING PHASE ---
        if start_fresh:
            print("\n=== Starting Training Phase ===")
            
            total_episodes = TRAIN_CONFIG["total_episodes"]
            num_evaluations = TRAIN_CONFIG.get("num_evaluations", 5)
            eval_interval = total_episodes // num_evaluations
            
            print(f"\n[System] Epsilon Schedule:")
            print(f"  Start: {TRAIN_CONFIG['epsilon_start']}")
            print(f"  End:   {TRAIN_CONFIG['epsilon_end']}")
            print(f"  Decay Steps: {TRAIN_CONFIG['epsilon_decay']}")
            
            if TRAIN_CONFIG['epsilon_decay'] is None:
                 print("[WARNING] Epsilon decay is None! Defaulting to 20000.")
                 agent.epsilon_decay = 20000
            
            agent.config["epsilon_decay"] = TRAIN_CONFIG["epsilon_decay"]
            agent.update_epsilon() 
            
            best_eval_reward = float('-inf')
            
            try:
                # Pre-Training Evaluation (Eval 0)
                avg_reward = run_evaluation(env, agent, logger, 0, 0)
                logger.log_evaluation_summary(0, 0, avg_reward)
                
                if avg_reward > best_eval_reward:
                     best_eval_reward = avg_reward
                
                for episode in range(1, total_episodes + 1):
                    obs, state = env.reset()
                    done = False
                    total_reward = 0
                    steps = 0
                    loss_val = 0
                    loss_count = 0
                    
                    while not done:
                        actions = agent.select_actions(obs, evaluate=False)
                        next_obs, next_state, reward, done, _ = env.step(actions)
                        buffer.push(obs, state, actions, reward, next_obs, next_state, done)
                        
                        loss = agent.train(buffer)
                        if loss > 0:
                            loss_val += loss
                            loss_count += 1
                        
                        obs = next_obs
                        state = next_state
                        total_reward += reward
                        steps += 1
                        
                    avg_loss = loss_val / loss_count if loss_count > 0 else 0
                    logger.log_episode(episode, steps, total_reward, agent.epsilon, avg_loss)
    
                    if episode % eval_interval == 0:
                        eval_id = episode // eval_interval
                        avg_reward = run_evaluation(env, agent, logger, eval_id, episode)
                        
                        if avg_reward > best_eval_reward:
                            print(f"  -> New Best Model! (Previous Best: {best_eval_reward:.4f})")
                            best_eval_reward = avg_reward
                            logger.save_model(agent, filename="model.pt")
                        else:
                            print(f"  -> Model is worse than previous best model ({best_eval_reward:.4f}). Not saving to model.pt")
                        
                        logger.log_evaluation_summary(eval_id, episode, avg_reward)
    
            except KeyboardInterrupt:
                print("\nTraining interrupted by user. Saving current model...")
                logger.save_model(agent, filename="interrupted_model.pt")
                break # Exit the experiment loop as well
                
        if not start_fresh:
            print("\n=== Starting Evaluation Only Phase ===")
            run_evaluation(env, agent, logger, 999, 0)
    
        print(f"\nExperiment {exp_id} Completed!")
        env.close()

if __name__ == "__main__":
    main()
