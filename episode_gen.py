import os
import re
import time
import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
from rl_agents.agents.tree_search.mcts import MCTSAgent

def get_next_episode_index(directory="output/"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    pattern = re.compile(r"^episode_(\d+)_(data\.csv|placeholder)$")

    taken_indexes = set()
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            taken_indexes.add(int(match.group(1)))

    candidate = 1
    while candidate in taken_indexes:
        candidate += 1

    placeholder_path = os.path.join(directory, f"episode_{candidate:04d}_placeholder") 
    try:
        with open(placeholder_path, "x") as f:
            f.write("claimed")
        return candidate
    except FileExistsError:
        time.sleep(0.1) 
        return get_next_episode_index(directory)


env = gym.make(
    "highway-v0",
    config={
        # Reward weights
        "collision_reward": -1.0,     # default: -1.0
        "high_speed_reward": 0.4,      # default: 0.4
        "right_lane_reward": 0.1,     # default: 0.1
        "target_speeds": [20, 30], # default: [20, 30]
        "lane_change_reward": -0.1,

        "vehicles_count": 50,
        "lanes_count": 4,

        "duration": 2,
        "simulation_frequency": 15,
        "policy_frequency": 5,

        "normalize_reward": False,
    },
    render_mode="rgb_array"
)

agent_config = {
    "budget": 50, # How many simulations per step
    "gamma": 0.90,
    "env_preprocessors": [],
}

agent = MCTSAgent(env, agent_config)

# print(env.unwrapped.config)  # Confirm config is set

EPISODES_TO_GENERATE = 3
episodes_saved = 0

attempt_counter = 0

while (episodes_saved < EPISODES_TO_GENERATE):
    current_ep = get_next_episode_index()

    seed = (current_ep * 73) + attempt_counter
    obs, info = env.reset(seed=seed)

    print(f"Generating Episode {current_ep:04d} with Seed: {seed} | Attempt: {attempt_counter}")

    # Increment the seed so that if an attempt fails it won't try on the same seed
    seed += 1

    episode_data = []
    episode_images = []
    done = truncated = False
    was_corrupted = False # Track if attempt failed (ex. car crashed)

    episode = current_ep
    step = 0

    while not (done or truncated):
        # Agent planning
        action = agent.act(obs)
        
        ego = env.unwrapped.vehicle
        step_entry = {
            "seed": seed,
            "episode": episode,
            "step": step,
            "x": ego.position[0],
            "y": ego.position[1],
            "vx": ego.velocity[0],
            "vy": ego.velocity[1],
            "cos_h": ego.direction[0],
            "sin_h": ego.direction[1],
            "cos_d": ego.destination_direction[0],
            "sin_d": ego.destination_direction[1],
            "long_off": ego.lane_offset[0],
            "lat_off": ego.lane_offset[1],
            "ang_off": ego.lane_offset[2],
            "lane_id": ego.lane_index[2],
            "action": action,
            "target_speed": ego.target_speed,
            "crashed": int(ego.crashed)
        }

        episode_images.append(env.render())

        obs, reward, done, truncated, info = env.step(action)
        step += 1

        if ego.crashed:
            was_corrupted = True
            break

        # To observe each step uncomment line below
        # print(f"action={action}, reward={reward}, lane={ego.lane_index}, speed={ego.speed:.1f}, crashed={ego.crashed}")

        step_entry["reward"] = reward
        step_entry["done"] = done
        episode_data.append(step_entry)

    placeholder_file = f"output/episode_{current_ep:04d}_placeholder"
    
    if was_corrupted:
        print(f"There was a crash on episode {current_ep:04d}, redoing the episode with a different seed...")
        attempt_counter += 1

        if os.path.exists(placeholder_file):
            os.remove(placeholder_file)
        continue
    
    df = pd.DataFrame(episode_data)
    df.to_csv(f"output/episode_{current_ep:04d}_data.csv", index=False)

    video_tensor = np.array(episode_images, dtype=np.uint8)
    np.savez_compressed(f"output/episode_{current_ep:04d}_visuals.npz", visuals=video_tensor)

    # Clean up placeholder file
    if os.path.exists(placeholder_file):
        os.remove(placeholder_file)

    print(f"Successfully saved episode_{current_ep:04d}!")

    attempt_counter = 0
    episodes_saved += 1

env.close()
print("Quota has been completed.")