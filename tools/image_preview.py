from pathlib import Path
import numpy as np
from PIL import Image

directory = Path('output/processed_output/processed_visual_test')
file_name = 'episode_0190_visuals.npz'

data = np.load(directory / file_name)
array = data['visuals']

print(array[1].shape)

img = Image.fromarray(array[0])
img.show()