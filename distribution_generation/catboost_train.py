import glob
import os
import re
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

import config as c
from utils.dataset import get_generated_data
from utils.image_processing import *

images, images_clean, images_noise = get_generated_data(short_load=True)
images = images.reshape(-1, 64, 64, 1)
images_clean = images_clean.reshape(-1, 64, 64, 1)
# images_clean = (images_clean > 0) * 1.0
images_noise = images_noise.reshape(-1, 64, 64, 1)
for img in images:
    print(np.sum(img)-100.4*64*64)

def get_train_data(image):
    x = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if j==0 and i==0:
                continue
            x.append(np.roll(np.roll(image, i, axis=0), j, axis=1).reshape(-1))
    x = np.stack(x, axis=1)
    return x


data = Parallel(n_jobs=c.NUM_CORES)(delayed(get_train_data)(image) for image in tqdm(images_noise))
x = np.array(data)
y = images_noise.reshape(images_noise.shape[0], -1, 1)

x = x.reshape(-1, 8)
y = y.reshape(-1, 1)

model = CatBoostRegressor(iterations=100,
                          learning_rate=3e-2,
                          l2_leaf_reg=3.0,  # any pos value
                          depth=6,  # int up to 16
                          min_data_in_leaf=1,  # 1,2,3,4,5
                          rsm=1,  # 0.01 .. 1.0
                          langevin=False,
                          task_type="GPU",
                          devices='0:1')
                          #diffusion_temperature=10000)

reg = LinearRegression().fit(x, y)
print(reg.score(x, y))
preds = reg.predict(x)
print(preds[y < 100][:15])
print(preds[y > 120][:15])
print(y[y > 0][:15])

model.fit(x, y)
preds = np.array(model.predict(x))
model.save_model('CatBoostRegressor')

print(preds)
for i in range(30):
    fig, axs = plt.subplots(2)
    axs[0].imshow(y.reshape((-1, 64, 64))[i])
    axs[1].imshow(preds.reshape((-1, 64, 64))[i])
    print(np.sum(y.reshape((-1, 64, 64))[i]))
    print(np.sum(preds.reshape((-1, 64, 64))[i]))
    plt.show()
print('')