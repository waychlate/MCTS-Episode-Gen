import os
import sys
import time
import gc
import argparse
from multiprocessing import Pool
import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
from rl_agents.agents.tree_search.mcts import MCTSAgent

# Candidate dataset paths (auto-detects HiPerGator /blue storage or local output folder)
CANDIDATE_PATHS = [
    os.environ.get("DATASET_DIR", ""),
    "/blue/iruchkin/khek.do/dataset_episodes_1000",
    "output/dataset_episodes_1000",
    "output/new/dataset_episodes_1000",
    "/home/khek.do/MCTS-Episode-Gen/output/dataset_episodes_1000"
]

def find_dataset_dir():
    for path in CANDIDATE_PATHS:
        if path and os.path.exists(path):
            return path
    # Default fallback
    return "output/dataset_episodes_1000"

DATASET_DIR = find_dataset_dir()
ENV_DURATION = 200  # 1000 steps at 5 Hz
AGENT_BUDGET = 150

# The 4 target episodes to regenerate
TARGET_EPISODES = [
    {"split": "train", "ep_num": 772},
    {"split": "train", "ep_num": 1102},
    {"split": "test",  "ep_num": 741},
    {"split": "test",  "ep_num": 776},
]

def get_min_ttc(obs, lane_id, num_lanes=4):
    """Calculates min TTC across lanes, zeroing out boundary padding and selecting ego speed index 1."""
    obs_clean = obs.copy()

    # Zero out out-of-bounds relative lanes
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
                print(f"  [Ep {fallback_ep_num:04d}] Found existing seed {seed} in {csv_path}")
                return seed
        except Exception as e:
            print(f"  [Ep {fallback_ep_num:04d}] Warning: Could not read seed from {csv_path}: {e}")
    fallback = fallback_ep_num * 1000
    print(f"  [Ep {fallback_ep_num:04d}] Using fallback seed {fallback}")
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

def process_single_episode(item):
    """Worker function to generate a single episode (CSV + Visual NPZ)."""
    split = item["split"]
    ep_num = item["ep_num"]
    split_dir = os.path.join(DATASET_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

    csv_file = os.path.join(split_dir, f"episode_{ep_num:04d}_data.csv")
    visuals_file = os.path.join(split_dir, f"episode_{ep_num:04d}_visuals.npz")

    print(f"\n[Worker] Starting Episode {ep_num:04d} [{split.upper()}] in {split_dir}...")
    seed = get_seed_for_episode(csv_file, ep_num)

    env = make_env()
    attempt = 0
    success = False

    while not success:
        actual_seed = seed + attempt
        print(f"  [Ep {ep_num:04d}] Running rollout with Seed: {actual_seed} (Attempt: {attempt})...")
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
                print(f"  [Ep {ep_num:04d}] Crash detected on attempt {attempt}, retrying with seed offset...")
                attempt += 1
                continue

            # Save Data CSV
            df = pd.DataFrame(episode_data)
            df.to_csv(csv_file, index=False)

            # Save Visuals NPZ
            np.savez_compressed(visuals_file, visuals=video_tensor[:step])

            duration = time.perf_counter() - start_time
            print(f"  [Ep {ep_num:04d} SUCCESS] Saved CSV ({len(df)} rows) and NPZ ({step} frames, {os.path.getsize(visuals_file):,} bytes) in {duration:.1f}s!")
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
    return ep_num

def main():
    parser = argparse.ArgumentParser(description="Parallel Episode Regeneration")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()

    # Check if running inside a SLURM Job Array (--array=1-4)
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    
    if slurm_task_id is not None:
        task_idx = int(slurm_task_id) - 1
        if 0 <= task_idx < len(TARGET_EPISODES):
            target = TARGET_EPISODES[task_idx]
            print(f"--- SLURM Array Task {slurm_task_id}/4: Regenerating Episode {target['ep_num']:04d} [{target['split']}] ---")
            print(f"--- Target Directory: {DATASET_DIR} ---")
            process_single_episode(target)
            print(f"--- SLURM Array Task {slurm_task_id} Completed Successfully! ---")
            return
        else:
            print(f"Error: SLURM_ARRAY_TASK_ID {slurm_task_id} out of range (1-{len(TARGET_EPISODES)})")
            return

    # Multiprocessing across 4 workers locally or on a single node
    num_workers = min(args.workers, len(TARGET_EPISODES))
    print("==================================================================")
    print(f" PARALLEL EPISODE REGENERATION WITH {num_workers} CONCURRENT WORKERS")
    print(f" Target episodes: {[item['ep_num'] for item in TARGET_EPISODES]}")
    print(f" Dataset Directory: {DATASET_DIR}")
    print("==================================================================")

    start_all = time.perf_counter()
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_episode, TARGET_EPISODES)

    total_time = time.perf_counter() - start_all
    print("\n==================================================================")
    print(f" ALL 4 EPISODES {results} REGENERATED IN PARALLEL IN {total_time:.1f} SECONDS!")
    print("==================================================================")

if __name__ == "__main__":
    main()
