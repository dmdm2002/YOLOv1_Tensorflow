import pandas as pd
import numpy as np
import operator
import glob
import cv2
import matplotlib.pyplot as plt

w = 676
h = 380
train_y_path = f'D:/Datasets/car_detection/data/training_label'

tr_label_file_path = sorted(glob.glob(f'{train_y_path}/*'))

a = pd.read_csv(tr_label_file_path[0])
print(a)

a['xmin'] = float((224.0 / w)) * a['xmin']
a['ymin'] = float((224.0 / h)) * a['ymin']
a['xmax'] = float((224.0 / w)) * a['xmax']
a['ymax'] = float((224.0 / h)) * a['ymax']

print('============================================')

print(a)

print('============================================')

a['x'] = (a['xmin'] + a['xmax']) / 2.0
a['y'] = (a['ymin'] + a['ymax']) / 2.0
a['w'] = a['xmax'] - a['xmin']
a['h'] = a['ymax'] - a['ymin']
a_arr = np.array(a)
print(a)

img = cv2.imread(f'D:/Datasets/car_detection/data/training_images/{a_arr[0][0]}', cv2.IMREAD_UNCHANGED)

cv2.imshow('a', img)
cv2.waitKey()