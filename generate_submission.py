import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import delayed, Parallel
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

import config as c
from utils.dataset import *


def predict(df: pd.DataFrame, particle_types_model_ER3_NR6, particle_types_model_ER10_NR20, mean_hist_matrix):
    df['is_center_shifted'] = ((df.circular_ratio < 0.8) & (df.bright_sum > 5))
    df['img_80'] = df['img_80'].map(lambda img: img - np.mean((img[:25] + img[-25:]), axis=0).reshape(1, 80) / 2 + 100.4)
    df['circular_std'] = Parallel(n_jobs=c.NUM_CORES)(delayed(get_circular_std)(img) for img in tqdm(df['img_80']))
    df['circular_sum'] = Parallel(n_jobs=c.NUM_CORES)(delayed(get_circular_sum)(img) for img in tqdm(df['img_80']))
    df['circular_kurtosis'] = Parallel(n_jobs=c.NUM_CORES)(delayed(get_circular_kurtosis)(img) for img in tqdm(df['img_80']))
    df['ellipse_coefficient'] = Parallel(n_jobs=c.NUM_CORES)(delayed(get_ellipse_coefficient)(img) for img in tqdm(df['img_80']))
    df['hist_matrix'] = Parallel(n_jobs=c.NUM_CORES)(delayed(get_histogram_matrix)(img, 16) for img in tqdm(df['img_80']))
    for i in range(len(mean_hist_matrix)):
        df['error_' + str(i)] = Parallel(n_jobs=c.NUM_CORES)(delayed(mean_absolute_error)(mean_hist_matrix[i], hist_matrix) for hist_matrix in df['hist_matrix'])
    mask = create_circular_mask(80, 80, radius=8)
    df['sum'] = df['img_80'].map(lambda img: np.sum(img[mask]) / np.sum(mask) - 100.4)

    class_borders = np.array([7.5, 30, 8.5])
    predicting_class = np.array([[1, 6, 20], [3, 10, 30]])

    x = df[['bright_sum', 'circular_ratio', 'circular_std', 'circular_kurtosis', 'circular_sum', 'error_0', 'error_1', 'error_2', 'error_3', 'error_4', 'ellipse_coefficient']].to_numpy()

    df['particle_types_predict_ER10_NR20'] = particle_types_model_ER10_NR20.predict(x)
    df['particle_types_predict_ER3_NR6'] = particle_types_model_ER3_NR6.predict(x)

    df['e'] = 0

    df['e'] += predicting_class[0, 0] * ((~df['is_center_shifted']) & (df['sum'] < class_borders[0]) & (df['particle_types_predict_ER3_NR6'] < 0.5))
    df['e'] += predicting_class[1, 0] * ((~df['is_center_shifted']) & (df['sum'] < class_borders[0]) & (df['particle_types_predict_ER3_NR6'] >= 0.5))

    df['e'] += predicting_class[0, 1] * (
                (~df['is_center_shifted']) & (df['sum'] > class_borders[0]) & (df['sum'] < class_borders[1]) & (df['particle_types_predict_ER3_NR6'] < 0.5))
    df['e'] += predicting_class[1, 1] * (
                (~df['is_center_shifted']) & (df['sum'] > class_borders[0]) & (df['sum'] < class_borders[1]) & (df['particle_types_predict_ER3_NR6'] >= 0.5))

    df['e'] += predicting_class[0, 2] * ((~df['is_center_shifted']) & (df['sum'] > class_borders[1]) & (df['particle_types_predict_ER3_NR6'] < 0.5))
    df['e'] += predicting_class[1, 2] * ((~df['is_center_shifted']) & (df['sum'] > class_borders[1]) & (df['particle_types_predict_ER3_NR6'] >= 0.5))

    df['e'] += predicting_class[0, 1] * ((df['is_center_shifted']) & (df['bright_sum'] < class_borders[2]))
    df['e'] += predicting_class[0, 2] * ((df['is_center_shifted']) & (df['bright_sum'] > class_borders[2]))

    df['t'] = ((~df['is_center_shifted']) & (df['sum'] < class_borders[0]) & (df['particle_types_predict_ER3_NR6'] > 0.5)) | \
                      ((~df['is_center_shifted']) & (df['sum'] > class_borders[0]) & (df['sum'] < class_borders[1]) & (df['particle_types_predict_ER3_NR6'] > 0.5)) | \
                      ((~df['is_center_shifted']) & (df['sum'] > class_borders[1]) & (df['particle_types_predict_ER3_NR6'] > 0.5))

    return df


def main():
    particle_types_model_ER3_NR6 = pickle.load(open('checkpoints/particle_types_model_ER3_NR6.pickle', 'rb'))
    particle_types_model_ER10_NR20 = pickle.load(open('checkpoints/particle_types_model_ER10_NR20.pickle', 'rb'))
    mean_hist_matrix = pickle.load(open('checkpoints/mean_hist_matrix.pickle', 'rb'))

    df_test = get_test_data(input_shape=(80, 80))
    df_test = predict(df_test, particle_types_model_ER3_NR6, particle_types_model_ER10_NR20, mean_hist_matrix)
    df_test = df_test[['id', 't', 'e']]

    df_private = get_private_test_data(input_shape=(80, 80))
    df_private = predict(df_private, particle_types_model_ER3_NR6, particle_types_model_ER10_NR20, mean_hist_matrix)
    df_private = df_private[['id', 't', 'e']]


    submission = pd.concat([df_private, df_test])
    submission = submission.fillna(1, axis=0)

    submission = submission[['id', 't', 'e']]
    submission.t = submission.t.astype(int)
    submission.e = submission.e.astype(int)
    submission = submission.rename(columns={"t": "classification_predictions", "e": "regression_predictions"})

    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    c.DATASET_DIR = 'tests'
    main()
