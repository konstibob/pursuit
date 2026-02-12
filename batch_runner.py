from main import main
import time

def run_batch_experiments():
    print("="*50)
    print("STARTING BATCH EXPERIMENT RUNNER")
    print("Executes experiments 1 through 12 sequentially.")
    print("="*50)
    
    experiments_to_run = list(range(1, 13))
    
    for exp_id in experiments_to_run:
        print(f"\n\n>>> STARTING EXPERIMENT ID: {exp_id} <<<")
        try:
            main(override_exp_id=exp_id)
            print(f">>> FINISHED EXPERIMENT ID: {exp_id} <<<")
        except Exception as e:
            print(f"!!! CRITICAL ERROR IN EXPERIMENT {exp_id} !!!")
            print(e)
            print("Skipping to next experiment...")
        
        # Small cooldown
        time.sleep(2)

if __name__ == "__main__":
    run_batch_experiments()
