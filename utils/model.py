import tensorflow.keras as k
from tensorflow.keras.models import load_model, Model
from typing import Iterable


def get_model(input_shape: Iterable = (None, None, 1),
              weights_file: str = None) -> Model:
    """
    :param input_shape: размерность входа
    :param weights_file: путь к файлу весов
    :return: загруженна в мапять модеь
    """
    if weights_file:
        model = load_model(weights_file, compile=False)
        return model


    base_model = create_base_model(input_shape=input_shape)

    #head_model = create_head_model(base_model.output)

    #full_model = k.models.Model(inputs=[base_model.input], outputs=head_model)

    return base_model


def create_base_model(input_shape: Iterable = (None, None, None)) -> k.models.Model:
    """
    Создает тело модели
    :param input_shape: размерность входа
    """
    input_layer = k.layers.Input(shape=input_shape)
    conv1 = k.layers.Conv2D(32, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(input_layer)
    conv1 = k.layers.Conv2D(32, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv1)
    merge = k.layers.concatenate([input_layer, conv1], axis=3)
    conv2 = k.layers.Conv2D(32, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(merge)
    conv2 = k.layers.Conv2D(32, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = k.layers.Conv2D(32, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = k.layers.Conv2D(1, 1, activation='sigmoid')(conv2)
    return Model(input_layer, conv3)


def create_head_model(base_model_output: k.layers.Layer) -> k.layers.Layer:
    """
    Создает голову модели
    :param base_model_output: выход тела модели
    :return: последний слой головы модели
    """
    x = base_model_output

    x = k.layers.GlobalAveragePooling2D()(x)
    x = k.layers.Dense(1, activation='relu', kernel_initializer='ones')(x)
    return x