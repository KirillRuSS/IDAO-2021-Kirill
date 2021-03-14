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

    head_model = create_head_model(base_model.output)

    full_model = k.models.Model(inputs=[base_model.input], outputs=head_model)

    return full_model


def create_base_model(input_shape: Iterable = (None, None, None)) -> k.models.Model:
    """
    Создает тело модели
    :param input_shape: размерность входа
    """
    input_layer = k.layers.Input(shape=input_shape)
    #x = k.layers.Dropout(0.1)(input_layer)
    x = input_layer
    x = k.layers.Conv2D(8, 1, activation='tanh')(x)
    x = k.layers.Conv2D(8, 3, activation='tanh', padding='same')(x)
    x = k.layers.Conv2D(8, 1, activation='tanh')(x)
    x = k.layers.Conv2D(2, 1, activation='tanh')(x)
    x = k.layers.Multiply()([input_layer, x])
    x = k.layers.Conv2D(1, 1, activation='relu')(x)

    #x = k.layers.Dropout(0.5)(x)
    #x = k.layers.Dense(classes, activation='sigmoid')(x)
    return Model(input_layer, x)


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