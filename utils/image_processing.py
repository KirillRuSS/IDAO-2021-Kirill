import re

import cv2
import scipy as sp
import numpy as np
import typing as t
from PIL import Image
import scipy.ndimage
from scipy.stats import kurtosis

import config as c


def y_loader(file_path, reaction_type) -> list:
    if reaction_type == 'ER':
        reaction_type = 0
    else:
        reaction_type = 1

    energy = int(re.search(r'\d+', re.search(r'_\d+_keV_', file_path)[0])[0])
    return [reaction_type, energy]


def x_loader(file_path) -> np.ndarray:
    img = Image.open(file_path)
    return np.array(img, np.float32)


def color_combining(image: np.ndarray, sigma: int = 10) -> np.ndarray:
    """
    Комбинирует смазаное и исходное изображения
    :param image: изображение
    :param sigma: каофициент смазывания изображения
    :return: объединенное изображение
    """
    blurry_image = cv2.GaussianBlur(image, (0, 0), sigma)
    image = cv2.addWeighted(image, 4, blurry_image, -4, 128)
    return image


def normalize_channels(img: np.array) -> np.ndarray:
    """
    Производит нормализацию изображения
    :param img: изображение
    :return: нормализованое изображение
    """
    img = np.arctan(img / np.mean(img.flatten()))
    return img


def x_preprocessing(x: np.ndarray, input_shape, values_linear_transformation=True, center_by_max=False, distance_matrices=True) -> np.ndarray:
    if c.DIST_MATRIX is None:
        c.DIST_MATRIX = np.mgrid[0:576:1, 0:576:1] - 288
        c.DIST_MATRIX = np.sum(c.DIST_MATRIX ** 2, axis=0) ** 0.5
    if values_linear_transformation:
        x -= 100
        x /= 255

    if distance_matrices:
        x = np.stack((x, c.DIST_MATRIX), axis=2)
    #x = x[270:310, 270:310, :] / 255

    x = crop_image(x, input_shape, center_by_max)
    return x.astype(np.float32)


def crop_image(img, input_shape, center_by_max=False, center=None):
    if center is None:
        center = (img.shape[0]//2, img.shape[1]//2)
    if center_by_max:
        img_gaussian = sp.ndimage.filters.gaussian_filter(img[:, :, 0], [3.0, 3.0]) * (img[:, :, 1]<10)
        center = (img_gaussian.argmax() // 576, img_gaussian.argmax() % 576)

    cut_shape = (center[0] - input_shape[0] // 2, center[1] - input_shape[1] // 2,
                 center[0] + input_shape[0] // 2, center[1] + input_shape[1] // 2)
    img = img[cut_shape[0]:cut_shape[2], cut_shape[1]:cut_shape[3]]
    return img.copy()


def y_preprocessing(y: np.ndarray) -> np.ndarray:
    new_y = np.zeros(c.CLASS_COUNT)
    new_y[y] = 1
    return new_y


def image_preprocessing(image):
    image = image[270:310, 270:310]
    image = sp.ndimage.filters.gaussian_filter(image, [1.0, 1.0], mode='constant')
    image = np.array(image) ** 30
    # image = (image > np.quantile(image, 0.7)) * image + (image <= np.quantile(image, 0.7))*np.quantile(image, 0.7)
    ##image = sp.ndimage.filters.gaussian_filter(image, [5.0, 5.0], mode='constant')
    # image = np.array(image) ** 20

    # image = image[0:100, 0:100]
    # image = np.sum(np.gradient(image), axis=0) + image
    # image[0, 0] = 0.005
    return image.astype(np.float32)


def create_circular_mask(h, w, center=None, radius=None):
    # Создает круговую маску изображения, с требуемым радиусом
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_height_map(img):
    # Получает граффик высот пятна
    m = np.zeros((15))
    for r in range(0, 30, 2):
        mask = create_circular_mask(80, 80, radius=r) * (~create_circular_mask(80, 80, radius=r - 2))
        m[r // 2] = np.sum(mask * (img - 100.4)) / np.sum(mask)
    m -= m.min()
    m /= m.max()
    return m


def get_circular_std(img):
    return np.std(img * create_circular_mask(80, 80, radius=20))


def get_circular_sum(img):
    return np.sum(img * create_circular_mask(80, 80, radius=20))


def get_circular_kurtosis(img):
    img = sp.ndimage.filters.gaussian_filter(img, [5.0, 5.0])
    height_map = get_height_map(img)
    return kurtosis(height_map)


def get_circular_ratio(img):
    img_gaussian = sp.ndimage.filters.gaussian_filter(img.copy(), [3.0, 3.0])
    gaussian_mask = img_gaussian > np.quantile(img_gaussian, 0.995)
    r = np.sqrt(np.sum(gaussian_mask) / np.pi)
    mask = create_circular_mask(250, 250, radius=r)
    return np.sum(mask * gaussian_mask) / np.sum(mask + gaussian_mask)


def get_bright_sum(img):
    img_gaussian = sp.ndimage.filters.gaussian_filter(img.copy(), [3.0, 3.0])
    gaussian_mask = img_gaussian > np.quantile(img_gaussian, 0.98)  # 112
    return np.sum(img[gaussian_mask] - 100.4) / 1000
