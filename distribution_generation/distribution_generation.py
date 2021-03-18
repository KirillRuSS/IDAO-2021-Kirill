import random
import timeit

import numpy as np
import config as c
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, rosen, rosen_der, Bounds

from spector import get_img_spector
import utils.image_processing as ipr
import utils.model as model_factory
from utils.dataset import get_train_data
from utils.image_processing import crop_image

Nfeval = 1


def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.8f}   {2: 3.8f}   {3: 3.8f}   {4: 3.8f}   {5: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], rosen(Xi)))
    Nfeval += 1


def fit(images, y):
    images = images[(y == 20).reshape(-1)]
    distribution_generator = DistributionGenerator(images)

    res = minimize(distribution_generator.calculate_distribution_score,
                   (0.99, 0.06, 0.8, 21),
                   callback=callbackF,
                   method='Nelder-Mead',
                   options={'disp': True})
    print(res)


class DistributionGenerator:
    def __init__(self, images, images_energy, w=64):
        self.w = w
        self.images = images
        self.images_energy = images_energy

    def get_noise_img(self):
        return crop_image(random.choice(self.images), (self.w, self.w), False, center=(self.w // 2, self.w // 2))

    def get_distribution(self, sigma_r=0.99, average_r=0.06, p_eng=0.8, energy_limit=1.0, n=100000, matrix_noise=None):
        positions = np.random.normal(0.0, sigma_r, (n, 2)) * average_r

        energies = np.random.geometric(p=p_eng, size=n)
        positions = (positions * self.w + self.w // 2).astype(int)

        total_energy = 0
        energy_limit *= 500
        position_count = 0
        matrix = np.zeros((self.w, self.w))
        for position, energie in zip(positions, energies):
            if total_energy >= energy_limit:
                break
            if min(position > 0) & min(position < self.w):
                matrix[position[0], position[1]] += energie
                total_energy += energie
                position_count += 1

        if matrix_noise is not None:
            matrix_noise += matrix
        return matrix, matrix_noise

    def calculate_distribution_score(self, coefficient, energy):
        n = 1000
        images_with_current_energy = self.images[self.images_energy == energy]

        sigma_r, average_r, p_eng, energy_limit = coefficient[0], coefficient[1], coefficient[2], coefficient[3]
        if not (0 < sigma_r < 1 and 0 < average_r < 1 and 0 < p_eng < 1 and 0 < energy_limit < 100):
            return 1e6

        self.get_distribution_spector(sigma_r=sigma_r,
                                      average_r=average_r,
                                      p_eng=p_eng,
                                      energy_limit=energy_limit,
                                      matrix_noise=self.get_noise_img())
        distribution_spector = np.sum(Parallel(n_jobs=c.NUM_CORES)
                                      (delayed(self.get_distribution_spector)(sigma_r=sigma_r,
                                                                              average_r=average_r,
                                                                              p_eng=p_eng,
                                                                              energy_limit=energy_limit,
                                                                              matrix_noise=self.get_noise_img()) for i in range(n)), axis=0)

        imgs_spector = np.sum(Parallel(n_jobs=c.NUM_CORES)
                              (delayed(get_img_spector)(crop_image(random.choice(images_with_current_energy), (self.w, self.w), False)) for i in range(n)), axis=0)

        print(mean_squared_error(imgs_spector / n, distribution_spector / n))
        return mean_squared_error(imgs_spector / n, distribution_spector / n)

    def get_distribution_spector(self, sigma_r=0.99, average_r=5.0, p_eng=0.2, energy_limit=1.0, matrix_noise=None):
        img_distribution = self.get_distribution(sigma_r=sigma_r,
                                                 average_r=average_r,
                                                 p_eng=p_eng,
                                                 energy_limit=energy_limit,
                                                 matrix_noise=matrix_noise)
        return get_img_spector(img_distribution[1])


if __name__ == '__main__':
    images, y = get_train_data(available_energy_values=[1, 3, 6, 10, 20, 30],
                               input_shape=(576, 576),
                               values_linear_transformation=False,
                               center_by_max=False, short_load=True)
    images = np.array(images)[:, :, :, 0]
    y = np.array(y)[:, 1].reshape((-1, 1)).astype(np.float)

    fit(images, y)
