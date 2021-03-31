import os
import fnmatch
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import config as c
from utils.image_processing import *


def get_x(file, data_dir, input_shape, values_linear_transformation=True, center_by_max=False, distance_matrices=True):
    file_path = os.path.join(data_dir, file)
    x = x_preprocessing(x_loader(file_path), input_shape, values_linear_transformation, center_by_max, distance_matrices)
    return x


def get_circular_ratio_and_bright_sum(file, data_dir):
    file_path = os.path.join(data_dir, file)
    img = x_loader(file_path)
    img = crop_image(img, (250, 250))
    circular_ratio = get_circular_ratio(img)
    bright_sum = get_bright_sum(img)
    return np.array([circular_ratio, bright_sum])


def get_y(file, data_dir, reaction_type):
    file_path = os.path.join(data_dir, file)
    y = y_loader(file_path, reaction_type)
    return y


def get_train_data(available_energy_values=[3, 10, 30],
                   input_shape=c.INPUT_SHAPE,
                   values_linear_transformation=True,
                   center_by_max=False,
                   short_load=False,
                   distance_matrices=True,
                   return_as_dataframe=False):
    x, y = [], []
    x_circular_ratio_and_bright_sum = []
    for reaction_type in ['NR', 'ER']:
        data_dir = os.path.join(c.DATASET_DIR, 'train', reaction_type)
        for root, dirs, files in os.walk(data_dir):
            files = fnmatch.filter(files, '*.png')
            if short_load:
                files = files[:300]
            x += Parallel(n_jobs=c.NUM_CORES) \
                (delayed(get_x)(file, data_dir, input_shape, values_linear_transformation, center_by_max, distance_matrices) for file in tqdm(files))

            x_circular_ratio_and_bright_sum += Parallel(n_jobs=c.NUM_CORES) \
                (delayed(get_circular_ratio_and_bright_sum)(file, data_dir) for file in tqdm(files))

            y += Parallel(n_jobs=c.NUM_CORES)(delayed(get_y)(file, data_dir, reaction_type) for file in tqdm(files))

    x_circular_ratio_and_bright_sum = np.array(x_circular_ratio_and_bright_sum)
    df = pd.DataFrame(y, columns=['t', 'e'])
    df['circular_ratio'] = x_circular_ratio_and_bright_sum[:, 0]
    df['bright_sum'] = x_circular_ratio_and_bright_sum[:, 1]
    df['img_' + str(input_shape[0])] = x
    return df


def get_test_data(input_shape=c.INPUT_SHAPE):
    x = []
    file_names = []
    x_circular_ratio_and_bright_sum = []
    data_dir = os.path.join(c.DATASET_DIR, 'public_test')
    for root, dirs, files in os.walk(data_dir):
        files = fnmatch.filter(files, '*.png')
        x += Parallel(n_jobs=c.NUM_CORES)\
            (delayed(get_x)(file, data_dir, input_shape, False, False, False) for file in tqdm(files))

        x_circular_ratio_and_bright_sum += Parallel(n_jobs=c.NUM_CORES) \
            (delayed(get_circular_ratio_and_bright_sum)(file, data_dir) for file in tqdm(files))
        file_names += files

    x_circular_ratio_and_bright_sum = np.array(x_circular_ratio_and_bright_sum)
    df = pd.DataFrame(file_names, columns=['file_names'])
    df['id'] = df['file_names'].map(lambda file_name: file_name[:-4])
    df['circular_ratio'] = x_circular_ratio_and_bright_sum[:, 0]
    df['bright_sum'] = x_circular_ratio_and_bright_sum[:, 1]
    df['img_' + str(input_shape[0])] = x
    return df



def get_private_test_data(input_shape=c.INPUT_SHAPE):
    x = []
    file_names = []
    x_circular_ratio_and_bright_sum = []
    data_dir = os.path.join(c.DATASET_DIR, 'private_test')
    for root, dirs, files in os.walk(data_dir):
        files = fnmatch.filter(files, '*.png')
        x += Parallel(n_jobs=c.NUM_CORES) \
            (delayed(get_x)(file, data_dir, input_shape, False, False, False) for file in tqdm(files))

        x_circular_ratio_and_bright_sum += Parallel(n_jobs=c.NUM_CORES) \
            (delayed(get_circular_ratio_and_bright_sum)(file, data_dir) for file in tqdm(files))
        file_names += files

    x_circular_ratio_and_bright_sum = np.array(x_circular_ratio_and_bright_sum)
    df = pd.DataFrame(file_names, columns=['file_names'])
    df['id'] = df['file_names'].map(lambda file_name: file_name[:-4])
    df['circular_ratio'] = x_circular_ratio_and_bright_sum[:, 0]
    df['bright_sum'] = x_circular_ratio_and_bright_sum[:, 1]
    df['img_' + str(input_shape[0])] = x
    return df


if __name__ == '__main__':
    get_train_data(short_load=True, return_as_dataframe=True)
