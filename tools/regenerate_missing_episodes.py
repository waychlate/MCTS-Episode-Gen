import os
import time
import gc
import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
from rl_agents.agents.tree_search.mcts import MCTSAgent

# Dataset base directory (can be overridden via environment variable)
DATASET_DIR = os.environ.get("DATASET_DIR", "/blue/iruchkin/khek.do/dataset_episodes_1000")
ENV_DURATION = 200  # 1000 steps at 5 Hz
AGENT_BUDGET = 150

# The 4 target episodes and their split folders
TARGET_EPISODES = [
    {"split": "train", "ep_num": 772},
    {"split": "train", "ep_num": 1102},
    {"split": "test",  "ep_num": 741},
    {"split": "test",  "ep_num": 776},
]

def get_min_ttc(obs, lane_id, num_lanes=4):
    """Calculates min TTC across lanes, zeroing out boundary padding and selecting ego speed index 1."""
    obs_clean = obs.copy()

    # Zero out out-of-bounds lanes
    if lane_id == 0:
        obs_clean[:, 0, :] = 0
    elif lane_id == num_lanes - 1:
        obs_clean[:, 2, :] = 0

    # Select ego vehicle's current velocity state (Index 1)
    lane_timelines = obs_clean[1, :, :]
    lane_ttcs = []

    for lane_idx in range(lane_timelines.shape[0]):
        timeline = lane_timelines[lane_idx]
        risky_steps = np.where(timeline > 0)[0]
        if len(risky_steps) > 0:
            earliest_crash_time = (risky_steps[0] + 1) * 0.2
            lane_ttcs.append(earliest_crash_time)
        else:
            lane_ttcs.append(15.0)

    return min(lane_ttcs)

def get_seed_for_episode(csv_path, fallback_ep_num):
    """Reads the exact seed from an existing CSV file if present, else computes fallback."""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'seed' in df.columns and len(df) > 0:
                seed = int(df['seed'].iloc[0])
                print(f"  Found existing seed {seed} in {csv_path}")
                return seed
        except Exception as e:
            print(f"  Warning: Could not read seed from {csv_path}: {e}")
    fallback = fallback_ep_num * 1000
    print(f"  Using fallback seed {fallback}")
    return fallback

def make_env():
    return gym.make(
        "highway-fast-v0",
        config={
            "collision_reward": -1.0,
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.1,
            "target_speeds": [15, 22, 30],
            "lane_change_reward": -0.1,
            "vehicles_count": 30,
            "lanes_count": 4,
            "duration": ENV_DURATION,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "normalize_reward": False,
            "observation": {
                "type": "TimeToCollision",
                "horizon": 15,
            },
        },
        render_mode="rgb_array"
    )

agent_config = {
    "budget": AGENT_BUDGET,
    "gamma": 0.90,
    "env_preprocessors": [],
}

def main():
    print("==========================================================")
    print(" REGENERATING MISSING / CORRUPTED EPISODES (CSV + NPZ)   ")
    print(f" Dataset Directory: {DATASET_DIR}")
    print(f" Duration: {ENV_DURATION}s (1000 steps) | Budget: {AGENT_BUDGET}")
    print("==========================================================")

    env = make_env()

    for item in TARGET_EPISODES:
        split = item["split"]
        ep_num = item["ep_num"]
        split_dir = os.path.join(DATASET_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        csv_file = os.path.join(split_dir, f"episode_{ep_num:04d}_data.csv")
        visuals_file = os.path.join(split_dir, f"episode_{ep_num:04d}_visuals.npz")

        print(f"\n>>> Processing Episode {ep_num:04d} [{split.upper()}] <<<")
        seed = get_seed_for_episode(csv_file, ep_num)
        
        attempt = 0
        success = False

        while not success:
            actual_seed = seed + attempt
            print(f"  Running rollout with Seed: {actual_seed} (Attempt: {attempt})...")
            start_time = time.perf_counter()

            try:
                agent = MCTSAgent(env, agent_config)
                obs, info = env.reset(seed=actual_seed)

                episode_data = []
                video_tensor = None
                done = truncated = False
                was_corrupted = False
                step = 0

                while not (done or truncated):
                    action = agent.act(obs)
                    ego = env.unwrapped.vehicle
                    current_ttc = get_min_ttc(obs, ego.lane_index[2])

                    step_entry = {
                        "seed": actual_seed,
                        "episode": ep_num,
                        "step": step,
                        "lane_id": ego.lane_index[2],
                        "action": action,
                        "target_speed": ego.target_speed,
                        "obs_ttc": current_ttc,
                    }
                    step_entry.update(ego.to_dict())

                    # Capture visual frame directly into preallocated buffer
                    frame = env.render()
                    if video_tensor is None:
                        max_steps = (ENV_DURATION * 5) + 20
                        video_tensor = np.zeros((max_steps, frame.shape[0], frame.shape[1], frame.shape[2]), dtype=np.uint8)
                    video_tensor[step] = frame

                    obs, reward, done, truncated, info = env.step(action)
                    step += 1

                    step_entry["reward"] = reward
                    step_entry["done"] = done
                    step_entry["crashed"] = int(ego.crashed)
                    episode_data.append(step_entry)

                    if ego.crashed:
                        was_corrupted = True
                        break

                if was_corrupted:
                    print(f"  [Crash detected on attempt {attempt}, retrying with seed offset...]")
                    attempt += 1
                    continue

                # Save Data CSV
                df = pd.DataFrame(episode_data)
                df.to_csv(csv_file, index=False)
                print(f"  Saved CSV: {csv_file} ({len(df)} steps, {os.path.getsize(csv_file):,} bytes)")

                # Save Visuals NPZ
                np.savez_compressed(visuals_file, visuals=video_tensor[:step])
                print(f"  Saved NPZ: {visuals_file} ({step} frames, {os.path.getsize(visuals_file):,} bytes)")

                duration = time.perf_counter() - start_time
                print(f"  Successfully regenerated episode_{ep_num:04d} in {duration:.1f}s!")
                success = True

            finally:
                if 'agent' in locals():
                    del agent
                if 'video_tensor' in locals():
                    del video_tensor
                if 'episode_data' in locals():
                    del episode_data
                gc.collect()

    env.close()
    print("\n==========================================================")
    print(" ALL 4 TARGET EPISODES REGENERATED AND VALIDATED!")
    print("==========================================================")

if __name__ == "__main__":
    main()
