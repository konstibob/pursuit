import os
import csv
import json
import torch
import glob
import shutil
from datetime import datetime

class ExperimentLogger:
    def __init__(self, experiment_config, resume_path=None):
        """
        Initialize logger.
        If resume_path is provided, use that directory.
        Otherwise create new: trained_agents/{exp_name}_{timestamp}/
        """
        self.exp_config = experiment_config
        self.base_dir = "trained_agents"
        
        if resume_path:
            self.run_dir = resume_path
            print(f"\n[Logger] Resuming/Evaluating in directory: {self.run_dir}")
            # Ensure csv exists or append? 
            # If evaluating, we switch to results.csv anyway.
            # If retraining, we might overwrite or append. 
            # For simplicity, if we check existing and choose "Train", we usually create NEW folder.
            # If we choose "Evaluate", we use EXISTING folder.
            # So resume_path implies EVALUATION or Resumption.
            
            # Setup CSV path (default to metrics.csv for consistency, but start_evaluation will switch it)
            self.csv_path = os.path.join(self.run_dir, "metrics.csv")
            self.csv_header = ["episode", "steps", "avg_step_reward", "epsilon", "loss"]
            
        else:
            # Create NEW directory
            # No timestamp, just the experiment name
            self.run_dir = os.path.join(self.base_dir, self.exp_config["name"])
            
            # If exists, delete it (overwrite)
            if os.path.exists(self.run_dir):
                print(f"[Logger] Overwriting existing directory: {self.run_dir}")
                shutil.rmtree(self.run_dir)
                
            os.makedirs(self.run_dir, exist_ok=True)
            
            # Save config
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(experiment_config, f, indent=4)
                
            # Initialize metrics.csv
            self.csv_path = os.path.join(self.run_dir, "metrics.csv")
            self.csv_header = ["episode", "steps", "total_reward", "epsilon", "loss"]
            
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_header)
                
            print(f"\n[Logger] Experiment output directory: {self.run_dir}")

    @staticmethod
    def check_existing(exp_name):
        """
        Check if there is an existing run for this experiment name.
        """
        base_dir = "trained_agents"
        path = os.path.join(base_dir, exp_name)
        if os.path.exists(path):
            return path
        return None
        
    def log_episode(self, episode, steps, total_reward, epsilon, loss):
        """
        Append episode metrics to current CSV.
        """
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, steps, round(total_reward, 2), round(epsilon, 3), round(loss, 3)])
            
            print(f"Episode: {episode} | "f"Steps: {steps} | "f"Reward: {round(total_reward, 2)} | "f"Epsilon: {round(epsilon, 3)} | "f"Loss: {round(loss, 3)}")

    def start_evaluation_phase(self, eval_id):
        """
        Prepare for a new evaluation phase.
        Creates a new CSV for this specific evaluation run.
        """
        eval_dir = os.path.join(self.run_dir, "evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        
        self.current_eval_csv = os.path.join(eval_dir, f"eval_{eval_id}.csv")
        with open(self.current_eval_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode_in_eval", "steps", "total_reward"])
            
    def log_evaluation_step(self, episode_in_eval, steps, total_reward):
        """
        Log a single episode within an evaluation phase.
        """
        if hasattr(self, 'current_eval_csv'):
            with open(self.current_eval_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode_in_eval, steps, round(total_reward, 2)])

    def log_evaluation_summary(self, eval_id, training_episode, avg_reward):
        """
        Log the summary of an evaluation phase.
        """
        summary_path = os.path.join(self.run_dir, "evaluation_summary.csv")
        
        # Create if not exists
        if not os.path.exists(summary_path):
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["eval_id", "training_episode", "avg_reward"])
        
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([eval_id, training_episode, round(avg_reward, 4)])

    def save_model(self, agent, filename="model.pt"):
        """
        Save the agent's model state dicts.
        """
        save_path = os.path.join(self.run_dir, filename)
        torch.save({
            "agent_net": agent.agent_net.state_dict(),
            "mixer": agent.mixer.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon
        }, save_path)
        print(f"[Logger] Model saved to {save_path}")
