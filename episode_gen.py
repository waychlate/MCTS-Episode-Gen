import gymnasium as gym
import copy
import highway_env
import numpy as np
from rl_agents.agents.tree_search.mcts import MCTSAgent

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
    "budget": 200, # How many simulations per step
    "gamma": 0.90,
    "env_preprocessors": [],
}

seed = 1 # Arbitrary seed
obs, info = env.reset(seed=seed) # Set seed while testing

agent = MCTSAgent(env, agent_config)

episode_data = []
episode_frames = []

# print(env.unwrapped.config)  # Confirm weights are set

# Episode Loop
episode = 0
step = 0
done = truncated = False
while not (done or truncated):
    # Agent planning
    action = agent.act(obs)
    
    ego = env.unwrapped.vehicle
    step_entry = {
        "seed": seed,
        "episode": episode, # Turn into a loop to collect lots of data
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

    episode_frames.append(env.render())

    obs, reward, done, truncated, info = env.step(action)
    step += 1

    # For observing
    print(f"action={action}, reward={reward}, lane={ego.lane_index}, speed={ego.speed:.1f}, crashed={ego.crashed}")

    step_entry["reward"] = reward
    step_entry["done"] = done
    episode_data.append(step_entry)

env.close()

import pandas as pd
df = pd.DataFrame(episode_data)
df.to_csv("driving_episode_1.csv", index=False)

np.save("test.npy", np.array(episode_frames))