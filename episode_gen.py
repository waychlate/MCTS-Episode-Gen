import os
import re
import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
from rl_agents.agents.tree_search.mcts import MCTSAgent

def get_next_episode_index(directory="output/"):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 1

    pattern = re.compile(r"^episode_(\d+)_data\.csv$")
    highest_index = 0
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            file_num = int(match.group(1))
            if file_num > highest_index:
                highest_index = file_num
    return highest_index + 1

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

        "duration": 8,
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

# print(env.unwrapped.config)  # Confirm weights are set

BATCH_SIZE = 100

start_index = get_next_episode_index()
end_index = start_index + BATCH_SIZE

for current_ep in range(start_index, end_index):
    print(f"Generating Episode {current_ep:04d}")
    seed = current_ep

    obs, info = env.reset(seed=seed)
    episode_data = []
    episode_images = []

    # Episode Loop
    episode = current_ep
    step = 0

    done = truncated = False
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

        # For observing
        # print(f"action={action}, reward={reward}, lane={ego.lane_index}, speed={ego.speed:.1f}, crashed={ego.crashed}")

        step_entry["reward"] = reward
        step_entry["done"] = done
        episode_data.append(step_entry)
    
    df = pd.DataFrame(episode_data)
    df.to_csv(f"output/episode_{current_ep:04d}_data.csv")

    video_tensor = np.array(episode_images, dtype=np.uint8)
    np.savez_compressed(f"output/episode_{current_ep:04d}_visuals.npz", visuals=video_tensor)

env.close()