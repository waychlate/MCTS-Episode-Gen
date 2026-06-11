import os
import re
import time
import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
from rl_agents.agents.tree_search.mcts import MCTSAgent

OUTPUT_DIRECTORY = "/blue/iruchkin/khek.do/output"
ENV_DURATION = 2
AGENT_BUDGET = 15

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

        "duration": ENV_DURATION,
        "simulation_frequency": 15,
        "policy_frequency": 5,

        "normalize_reward": False,
    },
    render_mode="rgb_array"
)

agent_config = {
    "budget": AGENT_BUDGET, # How many simulations per step
    "gamma": 0.90,
    "env_preprocessors": [],
}


agent = MCTSAgent(env, agent_config)

obs, info = env.reset()

import pprint

pprint.pprint(env.unwrapped.vehicle.to_dict())
done = truncated = False


episode_data = []
step = 0

while not (done or truncated):
    # Agent planning
    action = agent.act(obs)
    
    ego = env.unwrapped.vehicle
    step_entry = { 
        "step": step,
        "lane_id": ego.lane_index[2],
        "action": action,
        "target_speed": ego.target_speed,
        "crashed": int(ego.crashed)
    }

    episode_data.append(step_entry)
    episode_data.append(ego.to_dict())

    obs, reward, done, truncated, info = env.step(action)
    step += 1

env.close()

pprint.pprint(episode_data)
