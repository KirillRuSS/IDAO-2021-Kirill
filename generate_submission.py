import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config as c
import utils.image_processing as ipr
from utils.dataset import get_train_data, get_test_data, get_private_test_data


def main():
    df_test = get_test_data(input_shape=(100, 100))
    df_private = get_private_test_data(input_shape=(100, 100))


    submission = pd.concat([df_private, df_test])
    submission['t'] = 1
    submission['e'] = 1
    submission = submission[['id', 't', 'e']]
    submission = submission.rename(columns={"t": "classification_predictions", "e": "regression_predictions"})

    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    c.DATASET_DIR = 'tests'
    main()
