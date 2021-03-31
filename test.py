import random
import pandas as pd
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import config as c
import utils.image_processing as ipr
from utils.dataset import get_train_data, get_test_data, get_private_test_data

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config as c
import utils.image_processing as ipr
from utils.dataset import get_train_data, get_test_data, get_private_test_data


def main():
    c.DATASET_DIR = 'test_dataset'
    df_private = get_private_test_data(input_shape=(100, 100))
    df_test = get_test_data(input_shape=(100, 100))

    submission = pd.concat([df_private, df_test])
    submission['t'] = 1
    submission['e'] = 1
    submission = submission[['id', 't', 'e']]
    submission = submission.rename(columns={"t": "classification_predictions", "e": "regression_predictions"})
    print(submission.head())
    submission.to_csv('submission.csv', index=False)

    print('Все ок!')


if __name__ == '__main__':
    main()
