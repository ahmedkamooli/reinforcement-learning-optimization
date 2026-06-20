import win32com.client
import numpy as np

class AspenController:
    """Manages Aspen Plus process simulation and control."""
    
    def __init__(self, archive_path):
        try:
            self.aspen = win32com.client.gencache.EnsureDispatch("Apwn.Document")
            self.aspen.InitFromArchive2(archive_path)
            self.aspen.Visible = False
            print("[INFO] Aspen Plus initialized and archive loaded.")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to initialize Aspen Plus: {e}")
            raise

    def reset(self):
        """Reinitialize Aspen simulation."""
        try:
            self.aspen.Reinit()
            print("[INFO] Aspen simulation reinitialized.")
        except Exception as e:
            print(f"[ERROR] Failed to reinitialize Aspen: {e}")
        return np.zeros(2, dtype=np.float32)

    def step(self, pressure, temperature):
        """
        Execute one simulation step with given control parameters.
        
        Args:
            pressure: K-100 reactor pressure (bar)
            temperature: R-100 reactor temperature (K)
            
        Returns:
            obs: [NH3 mass flow, NH3 mass fraction]
            reward: Combined reward metric
            terminated: Episode termination flag
        """
        print("\n[DEBUG] Setting Action Parameters:")
        self.aspen.Tree.FindNode(r"/Data/Blocks/K-100/Input/PRES").Value = float(pressure)
        print(f"  K-100 PRES = {pressure:.2f}")
        self.aspen.Tree.FindNode(r"/Data/Blocks/R-100/Input/REAC_TEMP").Value = float(temperature)
        print(f"  REAC TEMP = {temperature:.2f}")

        try:
            self.aspen.Engine.Run2()
            print("[INFO] Aspen simulation run successfully.")
        except Exception as e:
            print(f"[ERROR] Aspen simulation failed: {e}")
            return np.zeros(2, dtype=np.float32), -10.0, True

        obs = np.zeros(2, dtype=np.float32)
        try:
            obs[0] = float(self.aspen.Tree.FindNode(r"/Data/Streams/NH3/Output/MASSFLOW/MIXED/NH3").Value)
            print(f"  MASSFLOW NH3 = {obs[0]:.4f}")
            obs[1] = float(self.aspen.Tree.FindNode(r"/Data/Streams/NH3/Output/MASSFRAC/MIXED/NH3").Value)
            print(f"  MASSFRAC NH3 = {obs[1]:.4f}")
        except Exception as e:
            print(f"[ERROR] Failed to retrieve observations: {e}")
            return np.zeros(2, dtype=np.float32), -10, True

        reward = obs[0] * obs[1]
        terminated = False
        if obs[0] < 1e-5 or obs[1] < 1e-5:
            terminated = True
            reward = -10.0
            print(f"[WARNING] Invalid observation. Reward: {reward:.2f}")

        print(f"[DEBUG] Calculated Reward: {reward:.2f}")
        return obs, reward, terminated

    def close(self):
        """Close Aspen Plus."""
        try:
            self.aspen.Close(False)
            print("[INFO] Aspen Plus closed.")
        except Exception as e:
            print(f"[ERROR] Closing Aspen: {e}")
