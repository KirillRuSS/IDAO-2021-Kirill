import gc
import random
import time

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as k

import config as c
import matplotlib.pyplot as plt
import utils.image_processing as ipr
from utils.dataset import get_generated_data


def main():
    model = model_factory.get_model(
        input_shape=(64, 64, 1),
        weights_file=None
    )
    print(model.summary())

    metrics = [
        k.metrics.mean_squared_error,
        k.metrics.binary_crossentropy
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


    images, images_clean, images_noise = get_generated_data(short_load=False)
    images = images.reshape(-1, 64, 64, 1)
    images_clean = images_clean.reshape(-1, 64, 64, 1)
    images_clean = (images_clean > 0) * 1.0
    images_noise = images_noise.reshape(-1, 64, 64, 1)

    model.fit(images, images_clean, epochs=c.EPOCHS, verbose=1, batch_size=c.BATCH_SIZE, validation_split=0.3, callbacks=callbacks)
    model.save("model.h5")

    predict_images = model.predict(images)
    for predict_image, clean_image in zip(predict_images, images_clean):
        fig, axs = plt.subplots(2)
        axs[0].imshow(predict_image)
        axs[1].imshow(clean_image)
        plt.show()




if __name__ == '__main__':
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    tf.random.set_seed(c.SEED)

    main()

    k.backend.clear_session()
    gc.collect()
