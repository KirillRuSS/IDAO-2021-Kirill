import gc
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt

import config as c
import utils.image_processing as ipr
import utils.model as model_factory
from utils.dataset import get_train_data, get_test_data


def main():
    model = k.models.load_model('model.h5')

    x, y = get_train_data()
    x = np.array(x)
    y = np.array(y)[:, 1].reshape((-1, 1)).astype(np.float)
    r = model.predict(x)

    plt.hist(r.reshape(-1), log=True, bins=100, alpha=0.5)
    plt.show()

    print(np.stack((y, model.predict(x)), axis=2).reshape(-1, 2).tolist())




if __name__ == '__main__':
    main()
    k.backend.clear_session()
    gc.collect()