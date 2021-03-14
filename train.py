import gc
import random
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


    x, y = get_train_data()
    x = np.array(x)
    y = np.array(y)[:, 1].reshape((-1, 1)).astype(np.float)

    model.fit(x, y, epochs=c.EPOCHS, verbose=1, batch_size=c.BATCH_SIZE, validation_split=0.3)

    print(model.predict(x[:30]))
    print(y[:30])
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [k.backend.function([inp], [out]) for out in outputs]  # evaluation functions
    model.save("model.h5")

    print(np.stack((y, model.predict(x)), axis=2).reshape(-1, 2).tolist())
    # Testing
    #for i in range(3):
    #    layer_outs = [func([x[i:i+1]]) for func in functors]
    #    print(layer_outs)




if __name__ == '__main__':
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    tf.random.set_seed(c.SEED)

    main()

    k.backend.clear_session()
    gc.collect()
