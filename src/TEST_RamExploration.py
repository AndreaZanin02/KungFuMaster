"""
    RAM Debugging and Exploration Tool for ALE/KungFuMaster-v5
    This script contains various manual testing suites to identify RAM addresses 
    corresponding to specific game states (health, lives, player position, camera, and boss patterns).
    
    Current test results:
        BYTE 6 --> screen portion --> boss located in ram[6] = 11
        BYTE 124 --> character X position in the screen
        BYTE 75 --> charachter's health --> from 39 to 0
        BYTE 29 --> character's lives --> from 3 to 0 (4 lives) --> game over while ram[29] = 255 overflow
        Byte 76 --> boss health --> from 39 to 0
"""

import gymnasium as gym
import pygame
import ale_py


# Human keys actions:   arrows --> movements   space --> attack   right+space --> kick
def get_manual_action(keys):
    if keys[pygame.K_SPACE] and keys[pygame.K_DOWN]: return 9
    elif keys[pygame.K_SPACE] and keys[pygame.K_UP]: return 10
    elif keys[pygame.K_SPACE] and keys[pygame.K_RIGHT]: return 7
    elif keys[pygame.K_SPACE] and keys[pygame.K_LEFT]: return 8
    elif keys[pygame.K_SPACE]: return 11
    elif keys[pygame.K_UP]: return 1
    elif keys[pygame.K_RIGHT]: return 2
    elif keys[pygame.K_LEFT]: return 3
    elif keys[pygame.K_DOWN]: return 4
    return 0


# Basic exploration: Prints values for known RAM addresses (Health and Lives) at every step
def investigate_health_and_lives():
    print("--- Running: Health and Lives Investigation ---")
    env = gym.make("ALE/KungFuMaster-v5", render_mode="human")
    obs, _ = env.reset()

    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(3)
        
        # Extract RAM
        ram = env.unwrapped.ale.getRAM()
        health_byte = int(ram[75])
        lives_byte = int(ram[29])
        
        print(f"Byte 75 (Health): {health_byte} | Byte 29 (Lives): {lives_byte} | Action: 3")
        
        done = terminated or truncated

    env.close()


# Monitors all 128 bytes to see which ones decrease during a single life (potential damage),
# and checks for sudden positive spikes exactly on the death frame (respawn/reset)
def investigate_death_drops():
    print("--- Running: Death Byte Detection ---")
    print("Anchoring search to the lives byte. Let the character get killed...")
    
    env = gym.make("ALE/KungFuMaster-v5", render_mode="human")
    env.reset()

    # Advance a few frames to bypass initial load
    for _ in range(50):
        env.step(1)

    prev_ram = env.unwrapped.ale.getRAM().copy()
    prev_lives = int(prev_ram[29])

    # Tracks how many times each byte decreases during a single life
    byte_decreases_while_alive = {i: 0 for i in range(128)}
    done = False

    while not done:
        env.step(0)
        current_ram = env.unwrapped.ale.getRAM()
        current_lives = int(current_ram[29])
        
        # Track what decreases while alive (potential damage taken)
        if current_lives == prev_lives:
            for i in range(128):
                if current_ram[i] < prev_ram[i]:
                    byte_decreases_while_alive[i] += 1
                    
        # Detect the exact death frame to see what resets positively
        elif current_lives < prev_lives:
            print(f"\n[!] DEATH DETECTED! Lives dropped to {current_lives}")
            
            for i in range(128):
                curr_val = int(current_ram[i])
                prev_val = int(prev_ram[i])
                
                # If a byte spiked (energy respawn) AND was decreasing while alive (damage)
                # Using a +5 margin to filter out noise and minor fluctuations
                if curr_val > prev_val + 5 and byte_decreases_while_alive[i] >= 2: 
                    print(f" - Byte {i:3d} | Jumped from {prev_val:3d} to {curr_val:3d} | Decreased {byte_decreases_while_alive[i]} times before death")
                    
            # Reset decrease counters to monitor the next life independently
            byte_decreases_while_alive = {i: 0 for i in range(128)}
        
        prev_ram = current_ram.copy()
        prev_lives = current_lives
        
        # Exit after Game Over triggers
        if current_lives == 0:
            done = True

    env.close()


# Tracks bytes that change consistently when the player character passes the center of the screen
# (triggering world scrolling)
def investigate_camera_scrolling():
    print("--- Running: Camera & World Scrolling Tracking ---")
    print("MANUAL CONTROLS ACTIVE")
    print("Walk LEFT. When Thomas reaches the center (X <= 140), camera tracking will begin!")
    print("Press TAB during tracking to print results.")

    env = gym.make("ALE/KungFuMaster-v5", render_mode="human")
    env.reset()

    start_world_ram = None
    prev_world_ram = None
    reversals = {i: 0 for i in range(128)}
    last_dir = {i: 0 for i in range(128)}

    done = False
    tracking_camera = False

    while not done:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        action = get_manual_action(keys)

        obs, reward, terminated, truncated, info = env.step(action)
        current_ram = env.unwrapped.ale.getRAM()
        thomas_x = int(current_ram[124])

        # Activate tracking once Thomas reaches the screen center
        if thomas_x <= 140 and not tracking_camera:
            print("\nThe agent is at the center. Starting world scroll tracking...")
            tracking_camera = True
            start_world_ram = current_ram.copy()
            prev_world_ram = current_ram.copy()

        if tracking_camera:
            for i in range(128):
                diff = int(current_ram[i]) - int(prev_world_ram[i])
                
                # Filter out standard overflow/underflow artifacts
                if diff == 0 or diff > 200 or diff < -200: 
                    continue
                    
                if diff > 0:
                    if last_dir[i] == -1: reversals[i] += 1
                    last_dir[i] = 1
                elif diff < 0:
                    if last_dir[i] == 1: reversals[i] += 1
                    last_dir[i] = -1
                    
            prev_world_ram = current_ram.copy()

        if keys[pygame.K_TAB] and tracking_camera:
            print("\n--- WORLD SCROLLING RESULTS (WORLD X) ---")
            for i in range(128):
                net_delta = int(current_ram[i]) - int(start_world_ram[i])
                # Looking for bytes that moved significantly, with few reversals, excluding Screen X (124)
                if abs(net_delta) > 5 and reversals[i] <= 3 and i != 124:
                    print(f"🔥 BYTE {i:3d} | Delta: {net_delta:+4d} | Reversals: {reversals[i]:2d} | Current Value: {current_ram[i]}")
            pygame.time.wait(1000)

        done = terminated or truncated
        pygame.time.wait(30)

    env.close()


# Eliminates any bytes that change while the player is completely idle
# (filtering out timers, animations, or enemies), leaving only position-dependent bytes
def investigate_idle_filtering():
    print("--- Running: Idle Byte Filtering (The Guillotine) ---")
    print("1. Remain completely IDLE for a few seconds (this eliminates timers like Byte 0 and 3).")
    print("2. Then walk LEFT, pass the screen center, and proceed a bit.")
    print("3. Press TAB to see which bytes survived the filter.")

    env = gym.make("ALE/KungFuMaster-v5", render_mode="human")
    env.reset()

    # Start by considering all 128 bytes as potential coordinates
    suspects = set(range(128))
    start_ram = env.unwrapped.ale.getRAM().copy()
    prev_ram = start_ram.copy()

    # Initial forcing of specific RAM states
    env.unwrapped.ale.setRAM(6, 9)  

    done = False
    while not done:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        # Force max health and infinite lives for uninterrupted testing
        env.unwrapped.ale.setRAM(75, 39) 
        env.unwrapped.ale.setRAM(29, 3)  

        action = get_manual_action(keys)
        is_idle = (action == 0)

        obs, reward, terminated, truncated, info = env.step(action)
        current_ram = env.unwrapped.ale.getRAM()
        
        # If the player is idle, the world should be static.
        # Any byte moving during this idle state is a timer, animation, or enemy
        if is_idle:
            for i in list(suspects):
                if current_ram[i] != prev_ram[i]:
                    suspects.remove(i)
                    
        # Print surviving bytes
        if keys[pygame.K_TAB]:
            print(f"\n--- SURVIVING BYTES ({len(suspects)}) ---")
            found = False
            for i in suspects:
                net_delta = int(current_ram[i]) - int(start_ram[i])
                # Only print surviving bytes that actualli moved during the walk
                if abs(net_delta) > 0:
                    print(f"BYTE {i:3d} | Net Delta: {net_delta:+4d} | Current Value: {current_ram[i]}")
                    found = True
            if not found:
                print("All moving bytes were eliminated. Walk left further and try again.")
            
            pygame.time.wait(500) # Debounce tab key

        prev_ram = current_ram.copy()
        done = terminated or truncated
        pygame.time.wait(30)

    env.close()


# Analyzes RAM for byte patterns that match boss mechanics (sudden drops for damage taken,
# slow increments for health regeneration)
def investigate_boss_patterns():
    print("--- Running: Boss Health & Pattern Investigation ---")
    print("MANUAL CONTROLS ACTIVE")
    print("- Press 'R' to toggle damage/regen tracking.")
    print("- Press 'TAB' to display suspected boss health bytes.")

    env = gym.make("ALE/KungFuMaster-v5", render_mode="human", repeat_action_probability=0.0)
    env.reset()

    # Data structures to count behavioral occurrences
    drops = {i: 0 for i in range(128)}   # Counts sudden byte decreases (damage)
    regens = {i: 0 for i in range(128)}  # Counts minor byte increases (healing/regen ticks)

    prev_ram = env.unwrapped.ale.getRAM().copy()
    tracking_active = False
    r_pressed = False
    tab_pressed = False

    # Force starting position near the boss
    env.unwrapped.ale.setRAM(6, 11) 

    done = False
    while not done:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        # Keep infinite health/lives cheat active to focus on the boss
        env.unwrapped.ale.setRAM(75, 39) 
        env.unwrapped.ale.setRAM(29, 3)  

        action = get_manual_action(keys)
        obs, reward, terminated, truncated, info = env.step(action)
        current_ram = env.unwrapped.ale.getRAM()
        
        # Toggle tracking with 'R' key
        if keys[pygame.K_r] and not r_pressed:
            tracking_active = not tracking_active
            status = "ACTIVATED" if tracking_active else "DEACTIVATED"
            print(f"\nDamage/Regen tracking {status}.")
            r_pressed = True
        elif not keys[pygame.K_r]:
            r_pressed = False

        # Pattern analysis
        if tracking_active:
            for i in range(128):
                diff = int(current_ram[i]) - int(prev_ram[i])
                
                # A drop of at least 2 is likely a hit taken (filtering underflows)
                if diff <= -2 and diff >= -20: 
                    drops[i] += 1
                # An increase of 1 or 2 is likely a regeneration tick
                elif diff == 1 or diff == 2:
                    regens[i] += 1

        # Print results with 'TAB'
        if keys[pygame.K_TAB] and not tab_pressed:
            print("\n--- BOSS PATTERN RESULTS ---")
            found_suspects = False
            for i in range(128):
                # We want bytes that have taken hits (drops > 0) AND healed (regens > 0)
                if drops[i] > 0 and regens[i] > 0:
                    print(f"BYTE {i:3d} | Damage taken: {drops[i]:2d} | Healing ticks: {regens[i]:2d} | Current Value: {current_ram[i]}")
                    found_suspects = True
                    
            if not found_suspects:
                print("No bytes showing the damage/regen pattern yet. Keep fighting the boss!")
            tab_pressed = True
        elif not keys[pygame.K_TAB]:
            tab_pressed = False

        prev_ram = current_ram.copy()
        done = terminated or truncated
        pygame.time.wait(30)

    env.close()

if __name__ == "__main__":

    """ Uncomment the specific test suite you wish to run"""
    # investigate_health_and_lives()
    # investigate_death_drops()
    # investigate_camera_scrolling()
    # investigate_idle_filtering()
    investigate_boss_patterns()
