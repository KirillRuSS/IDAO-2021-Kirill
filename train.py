import gc
import random
import time

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as k

import config as c
import utils.image_processing as ipr
import utils.model as model_factory
from utils.dataset import get_train_data, get_test_data


def main():
    model = model_factory.get_model(
        input_shape=c.INPUT_SHAPE,
        weights_file=None
    )
    print(model.summary())

    metrics = [
        k.metrics.mean_squared_error,
        k.metrics.categorical_accuracy
    ]

    optimizer = k.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        loss=k.losses.mean_squared_error,
        metrics=metrics
    )

    tb = k.callbacks.TensorBoard(
        log_dir='logs/{}{}_logs'.format(time.strftime('%Y_%m_%d_%H_%M_%S_'), c.FILE_PREFIX),
        write_graph=True,
        update_freq='batch'
    )

    callbacks = [
        tb,
    ]


    x, y = get_train_data()
    y = y[:, 1].reshape((-1, 1)).astype(np.float)

    model.fit(x, y, epochs=c.EPOCHS, verbose=1, batch_size=c.BATCH_SIZE, validation_split=0.3, callbacks=callbacks)

    model.save("model.h5")
    print(np.stack((y, model.predict(x)), axis=2).reshape(-1, 2)[:30])




if __name__ == '__main__':
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    tf.random.set_seed(c.SEED)

    main()

    k.backend.clear_session()
    gc.collect()
