import os
import pandas as pd
import matplotlib.pyplot as plt
from environment import HaberBoschEnv
from stable_baselines3 import PPO

def validate_policy(model, env, num_episodes=10, steps_per_episode=20, output_file="SRSI_results_NH3_only.csv"):
    """Validate trained policy and collect results."""
    all_results = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        print(f"\n--- Validation Episode {ep + 1}/{num_episodes} ---")
        
        for step in range(steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            physical_action = info["physical_action"]

            results_row = {
                "Episode": ep + 1,
                "Step": step + 1,
                "Reward": reward,
                "CumulativeReward": ep_reward,
                "Obs_NH3_MassFlow": obs[0],
                "Obs_NH3_MassFrac": obs[1],
                "AgentAction_0": action[0],
                "AgentAction_1": action[1],
                "PhysicalAction_K100_PRES": physical_action[0],
                "PhysicalAction_REAC_TEMP": physical_action[1]
            }
            all_results.append(results_row)

            if done or truncated:
                print(f"  Episode terminated at step {step + 1}. Cumulative Reward: {ep_reward:.2f}")
                break
        
        print(f"  Episode {ep + 1} finished. Cumulative Reward: {ep_reward:.2f}")

    # Save results
    df = pd.DataFrame(all_results)
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", output_file), index=False)
    print(f"[INFO] Results saved to results/{output_file}")
    
    return df


def plot_results(df):
    """Visualize validation results."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    df.groupby("Step")["Reward"].mean().plot(title="Mean Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    df.groupby("Step")["PhysicalAction_K100_PRES"].mean().plot(label="K-100 PRES (bar)")
    df.groupby("Step")["PhysicalAction_REAC_TEMP"].mean().plot(label="REAC_TEMP (K)")
    plt.title("Mean Physical Action Values per Step")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ASPEN_ARCHIVE = r"C:/Users/Ahmed/Downloads/e-Ammonia SRSI/ASPEN simulations/Green ammonia.apw"
    
    # Load environment and trained model
    env = HaberBoschEnv(ASPEN_ARCHIVE)
    model = PPO.load("PPO_Aspen_Final", env=env)
    
    # Validate
    print("\n[INFO] Running policy validation...")
    df = validate_policy(model, env, num_episodes=10, steps_per_episode=20)
    
    # Visualize
    plot_results(df)
    
    env.close()
