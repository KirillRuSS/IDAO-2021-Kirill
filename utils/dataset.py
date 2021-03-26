import glob
import os
import re
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from joblib import Parallel, delayed

import config as c
from utils.image_processing import *


def get_x(file, data_dir, input_shape, values_linear_transformation=True, center_by_max=False, distance_matrices=True):
    file_path = os.path.join(data_dir, file)
    x = x_preprocessing(x_loader(file_path), input_shape, values_linear_transformation, center_by_max, distance_matrices)
    return x


def get_y(file, data_dir, reaction_type):
    file_path = os.path.join(data_dir, file)
    y = y_loader(file_path, reaction_type)
    return y


def get_generated_data(short_load=False):
    data_dir = os.path.join(c.DATASET_DIR, 'generated_data')

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.match(r'[0-9]+.png', f)]
    files_clean = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.match(r'[0-9]+_clean.png', f)]
    files_noise = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.match(r'[0-9]+_noise.png', f)]

    if short_load:
        files = files[:30000]
        files_clean = files_clean[:30000]
        files_noise = files_noise[:30000]
    images = np.array(Parallel(n_jobs=c.NUM_CORES)(delayed(x_loader)(file) for file in tqdm(files)))
    images_clean = np.array(Parallel(n_jobs=c.NUM_CORES)(delayed(x_loader)(file) for file in tqdm(files_clean)))
    images_noise = np.array(Parallel(n_jobs=c.NUM_CORES)(delayed(x_loader)(file) for file in tqdm(files_noise)))

    return images, images_clean, images_noise


def get_train_data(available_energy_values=[3, 10, 30],
                   input_shape=c.INPUT_SHAPE,
                   values_linear_transformation=True,
                   center_by_max=False,
                   short_load=False,
                   distance_matrices=True,
                   return_as_dataframe=False):
    x, y = [], []
    for reaction_type in ['NR', 'ER']:
        data_dir = os.path.join(c.DATASET_DIR, 'train', reaction_type)
        for root, dirs, files in os.walk(data_dir):
            if short_load:
                files = files[:300]
            x += Parallel(n_jobs=c.NUM_CORES)\
                (delayed(get_x)(file, data_dir, input_shape, values_linear_transformation, center_by_max, distance_matrices) for file in tqdm(files))
            y += Parallel(n_jobs=c.NUM_CORES)(delayed(get_y)(file, data_dir, reaction_type) for file in tqdm(files))

    if not return_as_dataframe:
        x = np.array(x)
        y = np.array(y)

        available_data = np.isin(y[:, 1], available_energy_values)
        x = x[available_data]
        y = y[available_data]

        return x, y
    else:
        df = pd.DataFrame(y, columns=['t', 'e'])
        df['img_'+str(input_shape[0])] = x
        return df


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
    get_generated_data(short_load=True)
