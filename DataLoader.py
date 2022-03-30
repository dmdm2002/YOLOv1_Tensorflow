import tensorflow as tf
import glob
import os
import operator
import numpy as np
import pandas as pd

from Options import params


class Loader(params):
    def __init__(self):
        super(Loader, self).__init__()

        self.train_x_path = f'{self.root}/training_images'
        train_csv_path = f'{self.root}/training_label'

        test_x_path = f'{self.root}/testing_images'
        test_csv_path = f'{self.root}/testing_label'

        self.tr_file_csv = sorted(glob.glob(f'{train_csv_path}/*'))
        self.te_file_csv = sorted(glob.glob(f'{test_csv_path}/*'))

        self.tr_img_ds, self.tr_box_ds = self.get_box_info()



    def get_box_info(self):
        tr_data_df = pd.read_csv(self.tr_file_csv[0])
        te_data_df = pd.read_csv(self.te_file_csv[0])

        tr_img_path, tr_label_arr = self.Preprocessing(tr_data_df)

        # Make Full Path
        tr_img_fullpath = [f'{self.train_x_path}/{img}' for img in tr_img_path]

        tr_img_ds = tf.data.Dataset.from_tensor_slices(tr_img_fullpath)
        tr_box_ds = tf.data.Dataset.from_tensor_slices(tr_label_arr)

        return tr_img_ds, tr_box_ds

        # te_data_arr = self.Preprocessing(te_data_df)

    def Preprocessing(self, df):
        label_datas = []

        df['xmin'] = float((float(self.W) / self.original_W)) * df['xmin']
        df['ymin'] = float((float(self.H) / self.original_H)) * df['ymin']
        df['xmax'] = float((float(self.W) / self.original_W)) * df['xmax']
        df['ymax'] = float((float(self.H) / self.original_H)) * df['ymax']

        df['x'] = (df['xmin'] + df['xmax']) / 2.0
        df['y'] = (df['ymin'] + df['ymax']) / 2.0
        df['w'] = df['xmax'] - df['xmin']
        df['h'] = df['ymax'] - df['ymin']
        df['class'] = 0

        df['x_cell'] = df['x'] // 32
        df['y_cell'] = df['y'] // 32

        df['x_center_inCell'] = (df['x'] - df['x_cell'] * 32.0)/32.0 # 11
        df['y_center_inCell'] = (df['y'] - df['y_cell'] * 32.0)/32.0 # 12

        df['w'] = df['w'] / float(self.W)
        df['h'] = df['h'] / float(self.H)

        img_list = df['image']
        box_df = df.drop(['image'], axis=1)
        box_arr = np.array(box_df)

        for data in box_arr:
            label = self.make_label_data(data)
            label_datas.append(label)

        return np.array(img_list), np.array(label_datas)

    def make_label_data(self, data):
        label = np.zeros((7, 7, 25), dtype=float)

        label[int(data[10])][int(data[9])][0] = data[11]
        label[int(data[10])][int(data[9])][1] = data[12]
        label[int(data[10])][int(data[9])][2] = data[6]
        label[int(data[10])][int(data[9])][3] = data[7]
        label[int(data[10])][int(data[9])][4] = 1.0
        label[int(data[10])][int(data[9])][int(data[8]) + 5] = 1.0

        return label

    def decode_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, 3)
        img = tf.image.resize(img, [self.W, self.H]) / 255.

        return img

    def load(self):
        A_ds = self.tr_img_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = tf.data.Dataset.zip((A_ds, self.tr_box_ds))

        return ds

    def configure_for_performance(self, ds, shuffle=False):
        if shuffle == True:
            ds = ds.shuffle(buffer_size=self.cnt)
            ds = ds.batch(self.bathSZ)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        elif shuffle == False:
            ds = ds.batch(self.bathSZ)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

# l = Loader()
# ds = l.load()
#
# ds_iter = iter(ds)
# for step in range(1):
#     img, box = next(ds_iter)
#     print(img)
#     print(box)