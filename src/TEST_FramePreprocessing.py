import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from gymnasium.wrappers import AtariPreprocessing
from utils import KungFuPreprocessing, MaskFasciaGioco


# DEBUG FILE: Run this file to see how my custom pre-processing pipeline
# transforms the game frames for the network input


SKIPPED_STEPS = 100


gym.register_envs(ale_py)


# Arrows
def add_pipeline_arrow(axA, axB):

    con = ConnectionPatch(
        xyA=(1.0, 0.5), xyB=(0.0, 0.5),      
        coordsA="axes fraction", coordsB="axes fraction",
        axesA=axA, axesB=axB,
        arrowstyle="-|>",                     
        linewidth=3, color="black",           
        mutation_scale=25,                    
        shrinkA=15, shrinkB=15                
    )
    axA.add_artist(con)


def main():
    env_id = "ALE/KungFuMaster-v5"

    # Setup custom environment
    env_base = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env_processed = KungFuPreprocessing(env_base, frame_skip=4, screen_size=84)
    env_masked = MaskFasciaGioco(env_processed)

    # Setup official Atari environment
    env_official_base = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env_official = AtariPreprocessing(env_official_base, frame_skip=4, screen_size=84, grayscale_obs=True)

    obs_final, _ = env_masked.reset(seed=42)
    obs_official, _ = env_official.reset(seed=42)

    # Waiting for knife thrower
    print(f"Skipping {SKIPPED_STEPS} steps in order to make enemies spawn")
    for _ in range(SKIPPED_STEPS):
        obs_final, _, done, truncated, _ = env_masked.step(0)
        if done or truncated:
            obs_final, _ = env_masked.reset()
            
        obs_official, _, done_off, truncated_off, _ = env_official.step(0)
        if done_off or truncated_off:
            obs_official, _ = env_official.reset()

    # Frame extraction
    obs_base = env_masked.unwrapped.ale.getScreenRGB()
    obs_fat = env_processed.get_intermediate_rgb(obs_base)
    obs_base_official = env_official.unwrapped.ale.getScreenRGB()

    # Plotting
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 6, figure=fig)

    # Axis first line
    ax_top1 = fig.add_subplot(gs[0, 1:3])
    ax_top2 = fig.add_subplot(gs[0, 3:5])

    # Axis second line
    ax_bot1 = fig.add_subplot(gs[1, 0:2])
    ax_bot2 = fig.add_subplot(gs[1, 2:4])
    ax_bot3 = fig.add_subplot(gs[1, 4:6])

    # Pictures
    ax_top1.imshow(obs_base_official)
    ax_top1.set_title("Original")
    ax_top1.axis('off')
    ax_top2.imshow(obs_official, cmap='gray', vmin=0, vmax=255)
    ax_top2.set_title("Official Atari Preprocessing (84x84)")
    ax_top2.axis('off')
    ax_bot1.imshow(obs_base)
    ax_bot1.set_title("Original")
    ax_bot1.axis('off')
    ax_bot2.imshow(obs_fat)
    ax_bot2.set_title("Knife preprocessing")
    ax_bot2.axis('off')
    ax_bot3.imshow(obs_final, cmap='gray', vmin=0, vmax=255)
    ax_bot3.set_title("Input frame for the network (84x84)")
    ax_bot3.axis('off')

    # Arrow between pictures
    add_pipeline_arrow(ax_top1, ax_top2)
    add_pipeline_arrow(ax_bot1, ax_bot2)
    add_pipeline_arrow(ax_bot2, ax_bot3)

    plt.tight_layout()
    plt.show()

    env_masked.close()
    env_official.close()


if __name__ == "__main__":
    main()
