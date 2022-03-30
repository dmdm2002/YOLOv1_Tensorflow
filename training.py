import tensorflow as tf
import glob
import os
import operator
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.optimizers import adam_experimental
from Options import params
from Model import YoLo_Model
from DataLoader import Loader
from LossFunction import *


class Trainner(YoLo_Model):
    def __init__(self):
        super(Trainner, self).__init__()

        self.model = self.Yolo_v1()
        self.l = Loader()
        self.tr_ds = self.l.load()
        self.optimizer = adam_experimental.Adam(learning_rate=1e-3)
        self.loss_mean = keras.metrics.Mean()

    def lr_schedule(self, epoch, lr):  # epoch는 0부터 시작
        if epoch >= 0 and epoch < 75:
            lr = 0.001 + 0.009 * (float(epoch) / (75.0))  # 가중치를 0.001 ~ 0.0075로 변경
            return lr
        elif epoch >= 75 and epoch < 105:
            lr = 0.001
            return lr
        else:
            lr = 0.0001
            return lr

    # @tf.function
    def train_step(self, img, label):
        with tf.GradientTape() as tape:
            logits = self.model(img, training=True)
            bz_loss = yolo_multitask_loss(label, logits)

        grads = tape.gradient(bz_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.loss_mean.update_state(bz_loss)

        return bz_loss

    def fit_train(self):
        os.makedirs(self.ckp_path, exist_ok=True)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        manager = tf.train.CheckpointManager(ckpt, self.ckp_path, max_to_keep=None)

        for epoch in range(self.EPOCHS):
            tr_iter_shuffle = self.l.configure_for_performance(self.tr_ds, shuffle=True)
            tr_iter = iter(tr_iter_shuffle)

            for step in range(self.cnt//self.bathSZ):
                img, label = next(tr_iter)
                bz_loss = self.train_step(img, label)

                if step % 10 == 0:
                    print(f'step ({step}/{self.cnt})  Loss : {bz_loss}')

            result_loss = self.loss_mean.result()
            self.loss_mean.reset_states()

            manager.save()

            print(f'Train [ Epoch ({epoch}/{self.EPOCHS})   Loss : {result_loss}    ]')

