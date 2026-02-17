import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Style settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def load_experiments(base_dir="trained_agents"):

    experiments = []
    # Get all subdirectories in base_dir
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for d in dirs:
        run_path = os.path.join(base_dir, d)
        config_path = os.path.join(run_path, "config.json")
        eval_path = os.path.join(run_path, "evaluation_summary.csv")
        metrics_path = os.path.join(run_path, "metrics.csv")
        
        if os.path.exists(config_path) and os.path.exists(eval_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load Reward Data
            df_eval = pd.read_csv(eval_path)
            
            # Load Loss Data (Optional)
            df_metrics = None
            if os.path.exists(metrics_path):
                df_metrics = pd.read_csv(metrics_path)
            
            experiments.append({
                "id": config.get("id"),
                "name": config.get("name"),
                "x_size": config.get("x_size"),
                "y_size": config.get("y_size"),
                "surround": config.get("surround"),
                "freeze": config.get("freeze_evaders"),
                "type": "surround" if config.get("surround") else "touch",
                "dynamics": "freeze" if config.get("freeze_evaders") else "active",
                "full_type": f"{'surround' if config.get('surround') else 'touch'}_{'freeze' if config.get('freeze_evaders') else 'active'}",
                "eval_data": df_eval,
                "metrics_data": df_metrics
            })
            
    return experiments

def plot_perspective_a(experiments, output_dir="graphs"):

    types = ["surround_freeze", "surround_active", "touch_freeze", "touch_active"]
    scaling_dir = os.path.join(output_dir, "task")
    os.makedirs(scaling_dir, exist_ok=True)
    
    for t in types:
        plt.figure(figsize=(10, 6))
        subset = [e for e in experiments if e["full_type"] == t]
        
        # Sort by grid size
        subset.sort(key=lambda x: x["x_size"])
        
        for exp in subset:
            label = f"{exp['x_size']}x{exp['y_size']}"
            plt.plot(exp["eval_data"]["training_episode"], exp["eval_data"]["avg_reward"], 
                     marker='o', label=label)
            
        plt.title(f"Scaling Comparison: {t.replace('_', ' ').title()}")
        plt.xlabel("Training Episode")
        plt.ylabel("Average Evaluation Reward")
        plt.legend(title="Grid Size")
        plt.tight_layout()
        
        path = os.path.join(scaling_dir, f"scaling_{t}.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved {path}")

def plot_perspective_b(experiments, output_dir="graphs"):

    grids = sorted(list(set([(e["x_size"], e["y_size"]) for e in experiments])))
    mapsize_dir = os.path.join(output_dir, "mapsize")
    os.makedirs(mapsize_dir, exist_ok=True)
    
    for g in grids:
        plt.figure(figsize=(10, 6))
        subset = [e for e in experiments if (e["x_size"], e["y_size"]) == g]
        
        for exp in subset:
            plt.plot(exp["eval_data"]["training_episode"], exp["eval_data"]["avg_reward"], 
                     marker='o', label=exp["full_type"].replace('_', ' ').title())
            
        plt.title(f"Task Comparison: {g[0]}x{g[1]} Grid")
        plt.xlabel("Training Episode")
        plt.ylabel("Average Evaluation Reward")
        plt.legend(title="Experiment Type")
        plt.tight_layout()
        
        path = os.path.join(mapsize_dir, f"task_comparison_{g[0]}x{g[1]}.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved {path}")

def plot_training_stability(experiments, output_dir="graphs"):

    stability_dir = os.path.join(output_dir, "stability")
    os.makedirs(stability_dir, exist_ok=True)
    
    for exp in experiments:
        if exp["metrics_data"] is not None:
            plt.figure(figsize=(10, 6))
            df = exp["metrics_data"]
            
            # Use moving average to smooth loss
            window = max(1, len(df) // 20)
            df["loss_smooth"] = df["loss"].rolling(window=window).mean()
            
            plt.plot(df["episode"], df["loss"], alpha=0.3, color='gray', label="Raw Loss")
            plt.plot(df["episode"], df["loss_smooth"], color='blue', label=f"Smooth Loss (window={window})")
            
            plt.yscale('log')
            plt.title(f"Training Stability: {exp['name']}")
            plt.xlabel("Episode")
            plt.ylabel("Loss (Log Scale)")
            plt.legend()
            plt.tight_layout()
            
            path = os.path.join(stability_dir, f"loss_{exp['name']}.png")
            plt.savefig(path)
            plt.close()
            
    print(f"Saved loss stability plots in {stability_dir}")

def main():
    output_dir = "graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading experiment data...")
    experiments = load_experiments()
    
    if not experiments:
        print("No experiment data found in trained_agents/")
        return
        
    print(f"Found {len(experiments)} experiments.")
    
    print("\nGenerating Scaling Comparison (Perspective A) in 'graphs/task/'...")
    plot_perspective_a(experiments, output_dir)
    
    print("\nGenerating Task Comparison (Perspective B) in 'graphs/mapsize/'...")
    plot_perspective_b(experiments, output_dir)
    
    print("\nGenerating Training Stability (Loss) in 'graphs/stability/'...")
    plot_training_stability(experiments, output_dir)
    
    print("\nAll graphs generated successfully!")

if __name__ == "__main__":
    main()
