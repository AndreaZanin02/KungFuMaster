# Kung-Fu Master RL: Deep Q-Network & PPO

<img width="190" height="240" alt="Image" src="https://github.com/user-attachments/assets/9f4e1ddf-be7e-4765-b6e0-d37bbc033805" />

Deep Reinforcement Learning project focused on mastering the Atari 2600 classic **Kung-Fu Master** (`ALE/KungFuMaster-v5`).
This repository features a fully custom Double-DQN implementation and a parallelized PPO setup via Stable-Baselines3. 

The core of this project goes beyond standard environment wrappers: it includes RAM reverse-engineering to extract hidden game states from game's RAM (boss health, player lifes, milestones) for dynamic reward shaping, and a custom image preprocessing pipeline to help the agent isolate critical threats (like throwing knives).

## Key Features
* **Custom Double DQN Architecture:** A built-from-scratch CNN-based Double DQN with a pre-allocated circular Replay Buffer, optimized to run on laptops with 16GB RAM without triggering disk swapping.
* **Stable-Baselines3 PPO Integration:** A secondary training pipeline utilizing PPO with `SubprocVecEnv` for vectorized parallel training.
* **ALE RAM Reverse-Engineering:** Custom scripts (`TEST_RamExploration.py`) to map out the 128-byte Atari RAM, successfully identifying bytes for player health, lives, boss health, X-coordinates, and milestone progression.
* **Dynamic Reward Shaping:** The `KungFuMasterRewardShaper` replaces standard score-based rewards with a dense reward signal based on actual game progression (damage dealt to bosses, penalties for taking damage, milestone exploration bonuses).
* **Targeted Frame Preprocessing:** A custom computer vision pipeline (`KungFuPreprocessing` & `MaskFasciaGioco`) that masks non-playable UI areas to reduce noise and uses OpenCV dilation to highlight incoming enemy knives before grayscale conversion.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/AndreaZanin02/KungFuMaster.git
   ```
2. Install requirements:
   ```bash
   cd KungFuMaster
   pip install -r requirements.txt
   ```
3. Start the trainings:
   ```bash
   python train_dqn.py --seed 42 --total-timesteps 15000000
   ```
   ```bash
   python train_ppo.py --seed 42 --n-envs 8
   ```

## Behind the Scenes: RAM & Vision
### RAM-Based Reward Shaping
Standard Atari environments only reward the agent when the score increases. It's an unbalanced mode: each basic enemy offers 100 or 200 points to the agent and they spawn in infinite waves. Completing the level, however, simply grants the player 2000 points.
This project reads the ALE RAM to extract dense signals.
By exploring the memory, we identified:

* **Byte 75:** Player Health
* **Byte 76:** Boss Health
* **Byte 29:** Player Lives
* **Byte 6:** Screen Milestone

This allows the agent to learn defensive behaviors (penalized for losing health) and aggressive boss tactics (rewarded per HP of damage dealt to the boss), rather than blindly chasing score points.

### Advanced Preprocessing Pipeline
To help the CNN focus on lethal threats, `TEST_FramePreprocessing.py` validates a custom pipeline. It uses `cv2.inRange` and `cv2.dilate` to isolate the RGB value `[74, 74, 74]` (the color of the throwing knives). This forces the knives to appear as bright white pixels in the final 84x84 grayscale observation, drastically improving the agent's reaction time to projectiles.

## Repository Structure
* `agent.py`: Custom PyTorch implementation of the Double DQN agent and Replay Buffer.
* `train_dqn.py` / `train_ppo.py`: Main entry points for training loops.
* `evaluate.py`: Standalone script for rigorous model evaluation and behavior analysis.
* `utils.py`: Contains the custom environments wrappers, reward shaper, and preprocessing logic.
* `config.py`: Dataclasses holding hyperparameters for hardware-specific constraints (Fast testing, 16GB RAM, 64GB RAM).
* `logger.py` / `plot_results.py`: Tools for tracking metrics and visualizing training curves.
* `TEST_*.py`: Diagnostic scripts for manual RAM debugging and visual pipeline testing.
