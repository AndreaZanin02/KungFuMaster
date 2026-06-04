import argparse
import csv
from pathlib import Path
import numpy as np
import scipy.stats as stats
from stable_baselines3 import PPO
from agent import DQNAgent
from utils import get_device, make_eval_env, set_seed


# Evaluates the custom DQN model with a different seed for each episode.
def evaluate_dqn(model_path: Path, env_cfg: dict, base_seed: int, n_episodes: int) -> list:

    device = get_device()
    env = make_eval_env(
        env_id=env_cfg["env_id"], 
        seed=base_seed,
        screen_size=env_cfg["screen_size"], 
        frame_skip=env_cfg["frame_skip"], 
        frame_stack=env_cfg["frame_stack"]
    )
    
    if not model_path.exists():
        raise FileNotFoundError(f"DQN model not found at {model_path}")

    agent = DQNAgent(action_dim=env.action_space.n, device=device, input_channels=env_cfg["frame_stack"])
    agent.load(model_path)
    agent.policy_net.eval()

    rewards = []
    print("\nStarting DQN Evaluation:")
    for i in range(n_episodes):
        current_seed = base_seed + i
        obs, _ = env.reset(seed=current_seed)
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(np.array(obs), epsilon=0.0) 
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        rewards.append(episode_reward)
        print(f"  DQN Ep {i + 1}/{n_episodes} (Seed: {current_seed}): Score = {episode_reward:.1f}")
        
    env.close()
    return rewards


# Evaluates the Stable Baselines 3 PPO model with a different seed for each episode.
def evaluate_ppo(model_path: Path, env_cfg: dict, base_seed: int, n_episodes: int) -> list:

    device = get_device()
    env = make_eval_env(
        env_id=env_cfg["env_id"], 
        seed=base_seed,
        screen_size=env_cfg["screen_size"], 
        frame_skip=env_cfg["frame_skip"], 
        frame_stack=env_cfg["frame_stack"]
    )
    
    check_path = model_path if model_path.suffix == ".zip" else model_path.with_suffix(".zip")
    if not check_path.exists():
        raise FileNotFoundError(f"PPO model not found at {check_path}")

    model = PPO.load(str(model_path), device=str(device))

    rewards = []
    print("\nStarting PPO Evaluation:")
    for i in range(n_episodes):
        current_seed = base_seed + i
        obs, _ = env.reset(seed=current_seed)
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _ = model.predict(np.array(obs), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated
            
        rewards.append(episode_reward)
        print(f"  PPO Ep {i + 1}/{n_episodes} (Seed: {current_seed}): Score = {episode_reward:.1f}")
        
    env.close()
    return rewards


# Calculates statistical tests and returns a dictionary containing all the metrics for CSV export
def calculate_and_print_statistics(dqn_rews: list, ppo_rews: list, alpha: float = 0.05) -> dict:
    
    dqn_rews = np.array(dqn_rews)
    ppo_rews = np.array(ppo_rews)
    
    results = {
        "dqn_mean": dqn_rews.mean(), "dqn_std": dqn_rews.std(), "dqn_max": dqn_rews.max(),
        "ppo_mean": ppo_rews.mean(), "ppo_std": ppo_rews.std(), "ppo_max": ppo_rews.max(),
        "alpha": alpha
    }
    
    print(" STATISTICAL RESULTS ")
    
    print(f"DQN -> Mean: {results['dqn_mean']:.2f} | Std: {results['dqn_std']:.2f} | Max: {results['dqn_max']:.2f}")
    print(f"PPO -> Mean: {results['ppo_mean']:.2f} | Std: {results['ppo_std']:.2f} | Max: {results['ppo_max']:.2f}")
    print("-" * 55)

    # 1. Normality Test
    _, p_dqn = stats.shapiro(dqn_rews)
    _, p_ppo = stats.shapiro(ppo_rews)
    results["dqn_shapiro_p"] = p_dqn
    results["ppo_shapiro_p"] = p_ppo
    
    print("Normality Test (Shapiro-Wilk)")
    print(f"   DQN p-value: {p_dqn:.4e} -> {'Normal' if p_dqn > alpha else 'NOT Normal'}")
    print(f"   PPO p-value: {p_ppo:.4e} -> {'Normal' if p_ppo > alpha else 'NOT Normal'}")
    
    # 2. Welch's t-test
    t_stat, p_welch = stats.ttest_ind(dqn_rews, ppo_rews, equal_var=False)
    results["welch_p"] = p_welch
    
    print("\nWelch's t-test")
    print(f"   t-statistic: {t_stat:.4f} | p-value: {p_welch:.4e}")
    if p_welch < alpha:
        results["welch_winner"] = "DQN" if t_stat > 0 else "PPO"
        print(f"   Result:      Significant difference. {results['welch_winner']} is better.")
    else:
        results["welch_winner"] = "None"
        print("   Result:      No statistically significant difference.")

    # 3. Mann-Whitney U Test
    u_stat, p_mw = stats.mannwhitneyu(dqn_rews, ppo_rews, alternative='two-sided')
    results["mw_p"] = p_mw
    
    print("\nMann-Whitney U Test")
    print(f"   U-statistic: {u_stat:.4f} | p-value: {p_mw:.4e}")
    
    if p_mw < alpha:
        median_dqn = np.median(dqn_rews)
        median_ppo = np.median(ppo_rews)
        results["mw_winner"] = "DQN" if median_dqn > median_ppo else "PPO"
        print(f"   Result:      Significant difference. {results['mw_winner']} is stochastically superior.")
    else:
        results["mw_winner"] = "None"
        print("   Result:      No statistically significant difference.")
    print("="*55)
    
    return results


# Saves the aggregated statistical results into a cleanly formatted CSV file
def save_results_to_csv(results: dict, dqn_rews: list, ppo_rews: list, output_dir: Path):
    
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"statistical_comparison.csv"
    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "DQN", "PPO", "P-Value", "Winner / Notes"])
        writer.writerow(["Mean Reward", f"{results['dqn_mean']:.2f}", f"{results['ppo_mean']:.2f}", "-", "-"])
        writer.writerow(["Std Deviation", f"{results['dqn_std']:.2f}", f"{results['ppo_std']:.2f}", "-", "-"])
        writer.writerow(["Max Reward", f"{results['dqn_max']:.2f}", f"{results['ppo_max']:.2f}", "-", "-"])
        writer.writerow([])
        
        dqn_normality = "Normal" if results['dqn_shapiro_p'] > results['alpha'] else "NOT Normal"
        ppo_normality = "Normal" if results['ppo_shapiro_p'] > results['alpha'] else "NOT Normal"
        writer.writerow(["Shapiro-Wilk (Normality)", dqn_normality, ppo_normality, 
                         f"DQN: {results['dqn_shapiro_p']:.4e} | PPO: {results['ppo_shapiro_p']:.4e}", "-"])
                         
        writer.writerow(["Welch's t-test", "-", "-", f"{results['welch_p']:.4e}", results['welch_winner']])
        writer.writerow(["Mann-Whitney U Test", "-", "-", f"{results['mw_p']:.4e}", results['mw_winner']])

    # Scores for every episodes
    raw_data_path = output_dir / f"raw_scores.csv"
    with open(raw_data_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "DQN_Reward", "PPO_Reward"])
        
        for i in range(len(dqn_rews)):
            writer.writerow([i + 1, dqn_rews[i], ppo_rews[i]])

    print(f"\nStatistic summary saved at: {summary_path}")
    print(f"Raw scored saved at: {raw_data_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Statistical comparison between DQN and PPO on KungFuMaster")
    parser.add_argument("--dqn-model", type=str, default="../results/DQN/modelDQN.pt", 
                        help="Path to the DQN model file (.pt)")
    parser.add_argument("--ppo-model", type=str, default="../results/PPO/modelPPO.zip", 
                        help="Path to the PPO model file (.zip)")
    parser.add_argument("--episodes", type=int, default=50, 
                        help="Number of episodes for evaluation")
    parser.add_argument("--base-seed", type=int, default=1000, 
                        help="Base seed. Each episode will use base_seed + episode_index")
    parser.add_argument("--results-dir", type=str, default="../results", 
                        help="Directory where the CSV report will be saved")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.base_seed)

    env_cfg = {
        "env_id": "ALE/KungFuMaster-v5",
        "screen_size": 84,
        "frame_skip": 4,
        "frame_stack": 4
    }

    try:
        dqn_rewards = evaluate_dqn(Path(args.dqn_model), env_cfg, args.base_seed, args.episodes)
        ppo_rewards = evaluate_ppo(Path(args.ppo_model), env_cfg, args.base_seed, args.episodes)
        
        results_dict = calculate_and_print_statistics(dqn_rewards, ppo_rewards)
        
        save_results_to_csv(results_dict, dqn_rewards, ppo_rewards, Path(args.results_dir)) 

    except Exception as e:
        print(f"\nExecution Error: {e}")


if __name__ == "__main__":
    main()