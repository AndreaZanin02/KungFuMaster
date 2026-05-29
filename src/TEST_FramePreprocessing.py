import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
from utils import KungFuPreprocessing, MaskFasciaGioco

# DEBUG FILE: Run this file to see how my custom pre-processing pipeline
# transforms the game frames for the network input

gym.register_envs(ale_py)

def main():
    env_id = "ALE/KungFuMaster-v5"
    STEP_DA_AVANZARE = 100

    env_base = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env_processed = KungFuPreprocessing(env_base, frame_skip=4, screen_size=84)
    env_masked = MaskFasciaGioco(env_processed)

    obs_final, _ = env_masked.reset(seed=42)

    print(f"Skipping {STEP_DA_AVANZARE} steps in order to make enemies spawn")
    for _ in range(STEP_DA_AVANZARE):
        obs_final, _, done, truncated, _ = env_masked.step(0)
        if done or truncated:
            obs_final, _ = env_masked.reset()

    obs_base = env_masked.unwrapped.ale.getScreenRGB()
    
    obs_fat = env_processed.get_intermediate_rgb(obs_base)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(obs_base)
    axes[0].set_title("Original (210x160 RGB)")
    axes[0].axis('off')

    axes[1].imshow(obs_fat)
    axes[1].set_title("Knife preprocessing")
    axes[1].axis('off')

    axes[2].imshow(obs_final, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title("Input frame for the network (84x84 gray)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    env_masked.close()

if __name__ == "__main__":
    main()