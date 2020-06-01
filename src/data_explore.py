import h5py
import numpy as np
from PIL import Image
import io

filename = "../ds/flowers.hdf5"

f = h5py.File(filename, 'r')

dataset_keys = [str(k) for k in f['train'].keys()]

example_name = dataset_keys[0]

example = f['train'][example_name]

right_image = bytes(np.array(example['img']))

right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))

right_image.show()

print(right_image)