import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

def analizza_salute():
    # Loading environment
    env = gym.make("ALE/KungFuMaster-v5", obs_type="ram", frameskip=1, render_mode=None)
    obs, info = env.reset(seed=42)
    
    print("\nFase 1: Menu and waiting (400 frame)")
    env.step(7)

    # Waiting for animations
    for _ in range(400):
        env.step(0)

    # Starting the test
    current_lives = info.get('lives', 0)
    ram_history = []
    lives_history = []
    
    print("Doing NOOP action waiting for the death...")
    
    # Recording episode
    while True:
        obs, _, _, _, info = env.step(0)  # 0 = NOOP
        ram_history.append(obs.copy())
        lives_history.append(info['lives'])
        
        # Waiting 60 frames after death in order to monitoring reset of the life byte
        if info['lives'] < current_lives:
            print(f"Agent died at frame: {len(ram_history)}!")
            for _ in range(60):
                obs, _, _, _, _ = env.step(0)
                ram_history.append(obs.copy())
            break
            
    env.close()
    
    ram_history = np.array(ram_history, dtype=np.int32)
    death_frame = len(lives_history) - 1
    
    start_ram = ram_history[0]
    death_ram = ram_history[death_frame]
    
    candidates = []
    
    # Analysing the 128 bytes of the ram
    print("\n--- Filtering ---")
    for i in range(128):
        byte_trace = ram_history[:death_frame, i]
        
        # Condition 1: the life bytes is lower than its value at the start of the episode
        if start_ram[i] > death_ram[i]:
            
            # Differences step-by-step
            diffs = np.diff(byte_trace)
            
            # Condition 2: monotonic decreasing
            if np.all(diffs <= 0):
                
                # Counting how many times the byte value is changed (number of hits)
                drop_count = np.sum(diffs < 0)
                candidates.append((i, drop_count, start_ram[i], death_ram[i]))

    print(f"Found {len(candidates)} monotonic decrising candidates:\n")
    for cand in candidates:
        print(f"Byte {cand[0]:3d} | Starting value: {cand[2]:3d} | At death: {cand[3]:3d} | N° of hits (drops): {cand[1]}")
        
    # Plotting
    if len(candidates) > 0:
        plt.figure(figsize=(12, 6))
        
        for cand in candidates:
            idx = cand[0]
            plt.plot(ram_history[:, idx], label=f"Byte {idx}", linewidth=2)
            
        plt.axvline(x=death_frame, color='red', linestyle='--', label="Death")
        
        plt.title("Time analysis of the candidate bytes")
        plt.xlabel("Frame")
        plt.ylabel("Byte value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("ram_health_candidates.png", dpi=150)
        
        print("\nPlot saved in'ram_health_candidates.png'.")
    else:
        print("\nNo candidate bytes")

if __name__ == "__main__":
    analizza_salute()