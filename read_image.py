import numpy as np
from PIL import Image

file = f"episode_0001_visuals.npz"
data = np.load(f"output/{file}")

frame_data = data['visuals'][8]
img = Image.fromarray(frame_data, mode="RGB")
img.show()