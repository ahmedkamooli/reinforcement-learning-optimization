# Reinforcement Learning Optimization of Green Ammonia Production

## Project Overview

This project optimizes the **Haber-Bosch process** for green ammonia production using Proximal Policy Optimization (PPO). The RL agent learns to control reactor pressure and temperature to maximize ammonia yield while minimizing energy consumption, integrated directly with Aspen Plus simulations.

**Status:** Completed | **Duration:** June 2025 - Present | **Scholar:** SRSI, KAUST

---

## Technical Approach

### Architecture

**Environment (Gymnasium + Aspen Plus)**
- State: [NH3 mass flow (kg/s), NH3 mass fraction (%)]
- Actions: [K-100 pressure (15-300 bar), R-100 temperature (500-800 K)]
- Reward: Maximize (NH3 flow × NH3 fraction)
- Closed-loop: Agent → Aspen simulation → Feedback

**Agent (PPO - Stable-Baselines3)**
- Policy: MLP with 2 hidden layers (128 units)
- Learning rate: 3e-4 (default)
- Training: 10,000 timesteps

**Process Integration**
- Aspen Plus COM automation via pywin32
- Real simulation of industrial Haber-Bosch reactor
- Dynamic pressure/temperature control

### Key Technologies
- **Gymnasium:** RL environment framework
- **Stable-Baselines3:** PPO implementation
- **Aspen Plus:** Chemical process simulation
- **Python:** Core development

---

## Results

- **Best ammonia yield discovered:** 48 kg/h (vs. baseline setpoints)
- **Convergence:** ~6,000 episodes
- **Validation:** 10 episodes × 20 steps with deterministic policy
- **Output:** CSV data + reward curves

---

## Project Structure

```
src/
├── aspen_integration.py  # Aspen Plus control wrapper
├── environment.py        # Gymnasium environment
├── train.py              # PPO training script
└── validate.py           # Policy validation & visualization

results/
└── SRSI_results_NH3_only.csv  # Validation output

checkpoints/              # Periodic model saves
```

---

## Installation & Usage

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
python train.py
```
Trains PPO agent for 10,000 timesteps. Models saved to `checkpoints/` and final model to `PPO_Aspen_Final.zip`.

### Validation & Analysis
```bash
python validate.py
```
Validates trained policy, generates results CSV and plots.

---

## Technical Challenges & Solutions

1. **Aspen Plus Integration**
   - Challenge: Bridging COM API with Python RL loop
   - Solution: Wrapper class handling initialization, stepping, and error recovery

2. **Action Normalization**
   - Challenge: RL agents work with normalized [-1, 1] actions
   - Solution: Scale function maps to physical constraints [15-300 bar, 500-800 K]

3. **Simulation Failures**
   - Challenge: Invalid parameters cause Aspen crashes
   - Solution: Error handling + reward penalties for invalid states

---

## Future Work

- [ ] Compare with SAC, A3C algorithms
- [ ] Multi-objective optimization (yield vs. energy)
- [ ] Transfer learning to different reactor configs
- [ ] Real-world experimental validation

---

## Author

**Ahmed Kamal**  
SRSI Scholar | Electrical & Computer Engineering | University of Washington  
Email: kamooli@uw.edu

---

## License

MIT License
