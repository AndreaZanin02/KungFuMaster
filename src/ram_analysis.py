import gymnasium as gym
import ale_py
import numpy as np
from PIL import Image

gym.register_envs(ale_py)

env = gym.make("ALE/KungFuMaster-v5", obs_type="ram", render_mode="rgb_array", frameskip=1)
env.reset(seed=42)

# Movement actions
ACTION_NOOP = 0
ACTION_RIGHT = 2
ACTION_LEFT = 3
ACTION_START = 7  # RIGHTFIRE is for "press Start" in the menu

print("\nFase 1: Menu and waiting (400 frame)")
env.step(ACTION_START)

# Waiting for animations
for _ in range(400):
    env.step(ACTION_NOOP)

# --- Debug image ---
screen = env.render()
img = Image.fromarray(screen)
img.save("debug.png")

# Save starting ram
ram_base = env.unwrapped.ale.getRAM().copy()
clone_state = env.unwrapped.clone_state()

print("Fase 2: Movement tests (30 steps)")

# Test 1: Noise
for _ in range(30):
    env.step(ACTION_NOOP)
ram_noop = env.unwrapped.ale.getRAM().copy()

# Test 2: Left walk
env.unwrapped.restore_state(clone_state)
for _ in range(30):
    env.step(ACTION_LEFT)
ram_left = env.unwrapped.ale.getRAM().copy()

# Test 3: Right walk
env.unwrapped.restore_state(clone_state)
for _ in range(30):
    env.step(ACTION_RIGHT)
ram_right = env.unwrapped.ale.getRAM().copy()

# Difference analysis
time_bytes = np.where(ram_base != ram_noop)[0]

left_diff = np.where((ram_base != ram_left) & (ram_left != ram_noop))[0]
right_diff = np.where((ram_base != ram_right) & (ram_right != ram_noop))[0]

print(f"\n[!] Time changing bytes (not movement correlated): {time_bytes}")
print(f"Possible X bytes (Left test): {left_diff}")
print(f"Possible X bytes (Right test): {right_diff}")

candidates = np.unique(np.concatenate((left_diff, right_diff)))

print("\n--- Analysis (Base -> SX -> DX) ---")
if len(candidates) == 0:
    print("No candidates. Check debug.png for detail")
else:
    for idx in candidates:
        print(f"Byte[{idx:3}]: Base={ram_base[idx]:3} | Left={ram_left[idx]:3} | Right={ram_right[idx]:3}")

