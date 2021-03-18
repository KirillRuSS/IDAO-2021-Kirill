import os
import re
from tqdm import tqdm
from shutil import copyfile
from joblib import Parallel, delayed

import config as c
from utils.image_processing import *


def get_x(file, data_dir, input_shape, values_linear_transformation=True, center_by_max=False):
    file_path = os.path.join(data_dir, file)
    x = x_preprocessing(x_loader(file_path), input_shape, values_linear_transformation, center_by_max)
    return x


def get_y(file, data_dir, reaction_type):
    file_path = os.path.join(data_dir, file)
    y = y_loader(file_path, reaction_type)
    return y


def get_train_data(available_energy_values=[3, 10, 30],
                   input_shape=c.INPUT_SHAPE,
                   values_linear_transformation=True,
                   center_by_max=False,
                   short_load=False) -> (list, list):
    x, y = [], []
    for reaction_type in ['NR']:  # 'ER',
        data_dir = os.path.join(c.DATASET_DIR, 'train', reaction_type)
        for root, dirs, files in os.walk(data_dir):
            if short_load:
                files = files[:300]
            x += Parallel(n_jobs=c.NUM_CORES)\
                (delayed(get_x)(file, data_dir, input_shape, values_linear_transformation, center_by_max) for file in tqdm(files))
            y += Parallel(n_jobs=c.NUM_CORES)(delayed(get_y)(file, data_dir, reaction_type) for file in tqdm(files))

    x = np.array(x)
    y = np.array(y)

    available_data = np.isin(y[:, 1], available_energy_values)
    x = x[available_data]
    y = y[available_data]

    return x, y


def get_test_data():  # -> np.ndarray:
    x = []
    data_dir = os.path.join(c.DATASET_DIR, 'public_test')
    for root, dirs, files in os.walk(data_dir):
        for file in tqdm(files):
            file_path = os.path.join(data_dir, file)
            x.append(x_loader(file_path))

    # x = np.array(x)
    return x


if __name__ == '__main__':
    get_train_data(center_by_max=True)
