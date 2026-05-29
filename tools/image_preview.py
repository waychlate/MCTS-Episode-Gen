from pathlib import Path
import numpy as np
from PIL import Image

directory = Path('output/preprocessed_train')
file_name = 'episode_0002_visuals.npz'

data = np.load(directory / file_name)
array = data['visuals']

print(array[0].shape)

img = Image.fromarray(array[0])
img.show()