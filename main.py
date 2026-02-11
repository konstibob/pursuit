"""
Main Training Loop for QMIX on PettingZoo Pursuit.
"""

from environment import PursuitEnvWrapper
from buffer import ReplayBuffer
from agent import QMIXAgent
from config import TRAIN_CONFIG
import numpy as np
import time

def main():
    print("Initializing Environment...")
    env = PursuitEnvWrapper()
    
    print("Initializing Agent...")
    agent = QMIXAgent(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions
    )
    
    print("Initializing Replay Buffer...")
    buffer = ReplayBuffer(
        capacity=TRAIN_CONFIG["buffer_size"],
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions
    )
    
    print(f"\nConfiguration:")
    print(f"  Agents: {env.n_agents}")
    print(f"  Obs Dim: {env.obs_dim}")
    print(f"  State Dim: {env.state_dim}")
    print(f"  Device: {agent.device}")
    print("\nStarting Training...\n")
    
    # Metrics
    episode_rewards = []
    
    for episode in range(1, TRAIN_CONFIG["total_episodes"] + 1):
        obs, state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        loss_val = 0
        loss_count = 0
        
        while not done:
            # 1. Select Actions
            actions = agent.select_actions(obs)
            
            # 2. Step Environment
            next_obs, next_state, reward, done, _ = env.step(actions)
            
            # 3. Store Transition (Using shared reward for all agents)
            # Replay buffer stores reward as float (shared)
            buffer.push(obs, state, actions, reward, next_obs, next_state, done)
            
            # 4. Train
            loss = agent.train(buffer)
            if loss > 0:
                loss_val += loss
                loss_count += 1
            
            # 5. Update pointers
            obs = next_obs
            state = next_state
            total_reward += reward  # Since it's shared, we track one value
            steps += 1
            
        # End of Episode
        avg_loss = loss_val / loss_count if loss_count > 0 else 0
        episode_rewards.append(total_reward)
        
        # Calculate running average (last 10)
        avg_reward_10 = np.mean(episode_rewards[-10:])
        
        print(f"Episode {episode:3d} | "
              f"Steps: {steps:3d} | "
              f"Reward: {total_reward:6.2f} | "
              f"Avg Reward (10): {avg_reward_10:6.2f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Loss: {avg_loss:.4f}")

    print("\nTraining Completed!")
    env.close()

if __name__ == "__main__":
    main()
