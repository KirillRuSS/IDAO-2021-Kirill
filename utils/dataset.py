import os
import re
from tqdm import tqdm
from shutil import copyfile

import config as c
from utils.image_processing import *


def get_train_data():# -> (np.ndarray, np.ndarray):
    x, y = [], []
    for reaction_type in ['NR']:#'ER',
        data_dir = os.path.join(c.DATASET_DIR, 'train', reaction_type)
        for root, dirs, files in os.walk(data_dir):
            for file in tqdm(files):
                file_path = os.path.join(data_dir, file)
                x.append(x_preprocessing(x_loader(file_path)))
                y.append(y_loader(file_path, reaction_type))

    #x = np.array(x)
    #y = np.array(y)
    return x, y


def get_test_data():# -> np.ndarray:
    x = []
    data_dir = os.path.join(c.DATASET_DIR, 'public_test')
    for root, dirs, files in os.walk(data_dir):
        for file in tqdm(files):
            file_path = os.path.join(data_dir, file)
            x.append(x_loader(file_path))

    #x = np.array(x)
    return x
