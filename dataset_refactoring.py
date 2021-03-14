import os
import re
from shutil import copyfile

dir_old = "C:\\Users\\Kirill\\Documents\\DataSets\\idao_dataset\\train\\ER"
dir_new = "C:\\Users\\Kirill\\Documents\\DataSets\\idao_dataset\\data\\ER_"

for root, dirs, files in os.walk(dir_old):
    for file in files:
        path_old = os.path.join(dir_old, file)
        energy = re.search(r'\d+', re.search(r'_\d+_keV_', file)[0])[0]

        dir = dir_new + energy + 'keV'
        if not os.path.isdir(dir):
            os.mkdir(dir)

        path_new = os.path.join(dir, re.search(r'[\d.-]+', file)[0])
        path_new += '.png'
        copyfile(path_old, path_new)
        print(path_new)