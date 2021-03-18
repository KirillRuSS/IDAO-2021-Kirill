import os
import random

import numpy as np
from PIL import Image


import config as c
from utils.dataset import get_train_data, get_test_data
from distribution_generation import DistributionGenerator

imgs, y = get_train_data(available_energy_values=[1, 3, 6, 10, 20, 30], input_shape=(576, 576), values_linear_transformation=False, center_by_max=False, short_load=True)
imgs = np.array(imgs)[:,:,:,0]
y = np.array(y)[:, 1].reshape((-1, 1)).astype(np.float)

dataset_dir = os.path.join(c.DATASET_DIR, 'generated_data')

file_name = '{0}.png'
path = os.path.join(dataset_dir, file_name)

file_name_clean = '{0}_clean.png'
path_clean = os.path.join(dataset_dir, file_name_clean)

file_name_noise = '{0}_noise.png'
path_noise = os.path.join(dataset_dir, file_name_noise)

dist_gen = DistributionGenerator(imgs, y)

for i in range(100):
    noise = dist_gen.get_noise_img()
    matrix, matrix_noise = dist_gen.get_distribution(0.99, 0.06, 0.9, random.random()*50, matrix_noise=noise.copy())
    file_name = str(i)

    im = Image.fromarray(matrix_noise).convert("L")
    im.save(path.format(file_name))

    im = Image.fromarray(matrix).convert("L")
    im.save(path_clean.format(file_name))

    im = Image.fromarray(noise).convert("L")
    im.save(path_noise.format(file_name))