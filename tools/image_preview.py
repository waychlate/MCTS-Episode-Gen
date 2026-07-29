from pathlib import Path
import numpy as np
from PIL import Image

directory = Path('../output/mcts_dataset_1000')
file_name = 'episode_0030_visuals.npz'

data = np.load(directory / file_name)
array = data['visuals']

print(array[1].shape)

img = Image.fromarray(array[0])
img.show()