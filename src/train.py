import os
from environment import HaberBoschEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Configuration
ASPEN_ARCHIVE = r"C:/Users/Ahmed/Downloads/e-Ammonia SRSI/ASPEN simulations/Green ammonia.apw"
TOTAL_TIMESTEPS = 10000

# Initialize environment
env = HaberBoschEnv(ASPEN_ARCHIVE)

# Create directories
os.makedirs('./checkpoints/', exist_ok=True)
os.makedirs('./aspen_tensorboard/', exist_ok=True)
os.makedirs('./results/', exist_ok=True)

# Setup callback for periodic checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path='./checkpoints/',
    name_prefix='ammonia_aspen_model'
)

# Initialize PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./aspen_tensorboard/"
)

# Train
print("\n[INFO] Starting PPO training...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
model.save("PPO_Aspen_Final")
print("[INFO] Training complete. Model saved.")

env.close()
