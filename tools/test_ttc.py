import highway_env
import gymnasium as gym

# Assumes you've wrapped your config properly
env = gym.make("highway-v0", config={
    "observation": {
        "type": "TimeToCollision",
        "horizon": 10,
    }
})

obs, info = env.reset()
print("Observation Shape:", obs.shape)
print("Sample Observation Matrix:\n", obs)