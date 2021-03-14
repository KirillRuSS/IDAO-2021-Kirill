import os
import random
import config as c
import scipy as sp
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def image_preprocessing(image):
    image = np.array(image) ** 10
    image[200:400, 200:400] = 0

    image = image[image > 0.00001]
    image = image[image < 0.001]
    #image = image[250:350, 250:350]
    #image = np.sum(np.gradient(image), axis=0) + image
    #image[0, 0] = 0.005
    return image


plt.rcParams["figure.figsize"] = (40, 20)
reaction_types = [['ER_1keV', 'ER_3keV', 'ER_6keV', 'ER_10keV', 'ER_20keV', 'ER_30keV'],
                  ['NR_1keV', 'NR_3keV', 'NR_6keV', 'NR_10keV', 'NR_20keV', 'NR_30keV']]


for k in range(1):
    #fig, axs = plt.subplots(2, 6)
    for reaction_type in range(2):
        for i, energy in enumerate(reaction_types[reaction_type]):
            _, _, filenames = next(os.walk(os.path.join(c.DATASET_DIR, energy)))
            if (i + reaction_type) % 2 == 0:
                continue
            s = []
            hist_data = np.zeros(10)
            for filename in filenames:
                img = mpimg.imread(os.path.join(os.path.join(c.DATASET_DIR, energy), filename))
                img = image_preprocessing(img)
                #s = s + img.reshape((-1)).tolist()
                #img = img.reshape((-1))
                #hist_data = hist_data + np.histogram(img, bins=10, range=(0.00001, 0.001))[0]
                s.append(np.sum(img))


            #axs[reaction_type, i].set_title(energy)
            #axs[reaction_type, i].hist(s, bins=100)

            print(np.mean(s), np.std(s), np.min(s), np.max(s))

            s = ''
            for h in hist_data:
                s += "{:.2E} ".format(h)
            print(energy)
            #print(s)

    #thismanager = plt.get_current_fig_manager()
    #thismanager.window.wm_geometry("-70+0")

    #plt.show()
    print('!')
