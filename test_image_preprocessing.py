import os
import random
import re

import config as c
import scipy as sp
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def image_preprocessing(image):
    image = image[208:368, 208:368]
    #image = sp.ndimage.filters.gaussian_filter(image, [1.0, 1.0], mode='constant')
    #image = np.array(image) ** 30
    #image = (image > np.quantile(image, 0.7)) * image + (image <= np.quantile(image, 0.7))*np.quantile(image, 0.7)
    ##image = sp.ndimage.filters.gaussian_filter(image, [5.0, 5.0], mode='constant')
    #image = np.array(image) ** 20


    #image = image[0:100, 0:100]
    #image = np.sum(np.gradient(image), axis=0) + image
    #image[0, 0] = 0.005
    return image


plt.rcParams["figure.figsize"] = (40, 20)
reaction_types = [['ER_1keV', 'ER_3keV', 'ER_6keV', 'ER_10keV', 'ER_20keV', 'ER_30keV'],
                  ['NR_1keV', 'NR_3keV', 'NR_6keV', 'NR_10keV', 'NR_20keV', 'NR_30keV']]


for n in range(10):
    fig, axs = plt.subplots(2, 6)
    for reaction_type in range(2):
        for i, energy in enumerate(reaction_types[reaction_type]):
            _, _, filenames = next(os.walk(os.path.join(c.DATASET_DIR+'\\data', energy)))

            path = random.choice(filenames)
            """
            img = mpimg.imread(os.path.join(os.path.join(c.DATASET_DIR, energy), path))
            if reaction_type == 0:
                axs[reaction_type, i-1].set_title(energy)
                img = sp.ndimage.filters.gaussian_filter(img, [1.0, 1.0], mode='constant')
                img = np.array(img) ** 10
                img = img[250:350, 250:350]
                axs[reaction_type, i-1].imshow(img)
            else:
                axs[reaction_type, i + 1].set_title(energy)
                img = sp.ndimage.filters.gaussian_filter(img, [1.0, 1.0], mode='constant')
                img = np.array(img) ** 10
                img = img[250:350, 250:350]
                axs[reaction_type, i + 1].imshow(img)
            """
            img = mpimg.imread(os.path.join(os.path.join(c.DATASET_DIR+'\\data', energy), path))
            img = image_preprocessing(img)
            img *= 256
            #img = img*(img < 100)*(img > 90)# + img*(img > 102)

            #img -= img.min()
            #img /= img.max()
            axs[reaction_type, i].set_title(energy)
            axs[reaction_type, i].imshow(img)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("-70+0")
    plt.show()
