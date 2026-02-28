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

def run_evaluation(env, agent, logger, eval_id, training_episode, category="training", num_episodes=None):
    """
    Run an evaluation phase and log results.
    Returns average reward.
    """
    logger.start_evaluation_phase(eval_id, category=category)
    eval_episodes = num_episodes if num_episodes is not None else TRAIN_CONFIG.get("eval_episodes", 20)
    
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
    render_mode = None
    if override_exp_id is None:
        render_input = input("Do you want to view the experiment (render window)? (y/n): ").lower()
        if render_input == 'y':
            render_mode = "human"
            print("[System] Rendering enabled.")
        else:
            print("[System] Rendering disabled.")

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
    
    # --- BATCH CONFIGURATION ---
    batch_mode = None
    batch_episodes = None
    is_batch = (selection == 0)

    if is_batch:
        print("\n=== Batch Configuration ===")
        while True:
            choice = input("Do you want to (e)valuate ALL models or (t)rain ALL new ones? (e/t): ").lower()
            if choice in ['e', 't']:
                batch_mode = choice
                break
            print("Invalid choice. Please enter 'e' or 't'.")
        
        try:
            if batch_mode == 't':
                ep_prompt = f"How many episodes do you want to train EACH experiment for? (Default: {TRAIN_CONFIG['total_episodes']}): "
                default_eps = TRAIN_CONFIG['total_episodes']
            else:
                ep_prompt = f"How many episodes do you want to evaluate EACH model for? (Default: {TRAIN_CONFIG.get('eval_episodes', 20)}): "
                default_eps = TRAIN_CONFIG.get('eval_episodes', 20)
                
            ep_input = input(ep_prompt)
            batch_episodes = int(ep_input) if ep_input.strip() else default_eps
        except ValueError:
            print(f"[System] Invalid input. Using default: {default_eps}")
            batch_episodes = default_eps

    # --- EXPERIMENT LOOP ---
    for exp_id in exp_ids:
        exp_config = EXPERIMENTS[exp_id].copy()
        exp_config['render_mode'] = render_mode
        print(f"\n" + "#"*40)
        print(f" Starting Experiment {exp_id}: {exp_config['name']}")
        print("#"*40)

        existing_dir = ExperimentLogger.check_existing(exp_config["name"])
        
        if is_batch:
            start_fresh = (batch_mode == 't')
            resume_path = existing_dir if not start_fresh else None
        else:
            start_fresh = True
            resume_path = None
            if existing_dir:
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
        
        if start_fresh:
            logger = ExperimentLogger(exp_config) # New folder
        else:
            logger = ExperimentLogger(exp_config, resume_path=resume_path) # Use existing folder
        
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
            
            if is_batch:
                total_episodes = batch_episodes
            else:
                try:
                    ep_input = input(f"How many episodes do you want to train for? (Default: {TRAIN_CONFIG['total_episodes']}): ")
                    if ep_input.strip():
                        total_episodes = int(ep_input)
                    else:
                        total_episodes = TRAIN_CONFIG["total_episodes"]
                except ValueError:
                    print(f"[System] Invalid input. Using default: {TRAIN_CONFIG['total_episodes']}")
                    total_episodes = TRAIN_CONFIG["total_episodes"]

            epsilon_decay_steps = TRAIN_CONFIG["epsilon_decay"]
            eval_interval = 25
            num_evaluations = max(1, total_episodes // eval_interval)
            
            print(f"\n[System] Epsilon Schedule:")
            print(f"  Start: {TRAIN_CONFIG['epsilon_start']}")
            print(f"  End:   {TRAIN_CONFIG['epsilon_end']}")
            print(f"  Decay Steps: {epsilon_decay_steps}")
            print(f"  Evaluation every {eval_interval} episodes ({num_evaluations} total evaluations)")
            
            agent.config["epsilon_decay"] = epsilon_decay_steps
            agent.update_epsilon() 
            
            best_eval_reward = float('-inf')
            
            try:
                avg_reward = run_evaluation(env, agent, logger, 0, 0, category="training")
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
                        avg_reward = run_evaluation(env, agent, logger, eval_id, episode, category="training")
                        
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
                break
            finally:
                pass
                
        if not start_fresh:
            print("\n=== Starting Evaluation Only Phase ===")
            if is_batch:
                num_eval_episodes = batch_episodes
            else:
                try:
                    eval_ep_input = input(f"How many evaluation episodes? (Default: {TRAIN_CONFIG.get('eval_episodes', 20)}): ")
                    if eval_ep_input.strip():
                        num_eval_episodes = int(eval_ep_input)
                    else:
                        num_eval_episodes = TRAIN_CONFIG.get('eval_episodes', 20)
                except ValueError:
                    print(f"[System] Invalid input. Using default: {TRAIN_CONFIG.get('eval_episodes', 20)}")
                    num_eval_episodes = TRAIN_CONFIG.get('eval_episodes', 20)
                
            run_evaluation(env, agent, logger, "final", 0, category="final", num_episodes=num_eval_episodes)
    
        print(f"\nExperiment {exp_id} Completed!")
        env.close()

if __name__ == "__main__":
    main()
