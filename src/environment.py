import numpy as np
import gymnasium as gym
from gymnasium import spaces
from aspen_integration import AspenController

class HaberBoschEnv(gym.Env):
    """
    Gymnasium environment for Haber-Bosch process optimization.
    
    State: [NH3 mass flow, NH3 mass fraction]
    Actions: [K-100 pressure, R-100 temperature]
    """
    
    def __init__(self, archive_path):
        super().__init__()
        self.aspen = AspenController(archive_path)
        
        # Physical constraints
        self.min_action_values = np.array([15, 500], dtype=np.float32)  # [pressure_bar, temp_K]
        self.max_action_values = np.array([300, 800], dtype=np.float32)

        # Agent receives normalized actions [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation: [NH3 mass flow, NH3 mass fraction]
        self.observation_space = spaces.Box(
            low=0, high=1e5, shape=(2,), dtype=np.float32
        )

    def _scale_action(self, normalized_action):
        """Convert normalized action [-1, 1] to physical values."""
        scaled = self.min_action_values + (normalized_action + 1) / 2 * (
            self.max_action_values - self.min_action_values
        )
        return scaled

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_obs = self.aspen.reset()
        return initial_obs, {}

    def step(self, action):
        """Execute one environment step."""
        scaled_action = self._scale_action(action)
        obs, reward, terminated = self.aspen.step(scaled_action[0], scaled_action[1])
        truncated = False
        
        return obs, reward, terminated, truncated, {
            "physical_action": scaled_action,
            "normalized_action": action
        }

    def close(self):
        self.aspen.close()
