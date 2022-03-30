import keras
from keras.layers import *
from keras.applications import densenet, vgg19
from keras.initializers import *
from keras.regularizers import *
from Options import params


class YoLo_Model(params):
    def __init__(self):
        super(YoLo_Model, self).__init__()
        self.backbone = 'densenet'

    def conv_block(self, x, n_filters, size, strides=1, pool=False):
        x = Conv2D(filters=n_filters, kernel_size=size, padding='same', strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        if pool:
            x = MaxPool2D(pool_size=2)(x)

        return x

    def get_DarkNet19(self):
        input_layer = Input(shape=(self.H, self.W, 3))
        x = self.conv_block(input_layer, 32, 3, pool=True)
        x = self.conv_block(x, 64, 3, pool=True)

        x = self.conv_block(x, 128, 3)
        x = self.conv_block(x, 64, 1)
        x = self.conv_block(x, 128, 3, pool=True)

        x = self.conv_block(x, 256, 3)
        x = self.conv_block(x, 128, 1)
        x = self.conv_block(x, 256, 3, pool=True)

        x = self.conv_block(x, 512, 3)
        x = self.conv_block(x, 256, 1)
        x = self.conv_block(x, 512, 3)
        x = self.conv_block(x, 256, 1)
        x = self.conv_block(x, 512, 3, pool=True)

        x = self.conv_block(x, 1024, 3)
        x = self.conv_block(x, 512, 1)
        x = self.conv_block(x, 1024, 3)
        x = self.conv_block(x, 512, 1)
        x = self.conv_block(x, 1024, 3)

        x = self.conv_block(x, 1000, 1)
        output_layer = GlobalAveragePooling2D()(x)

        return keras.Model(inputs=input_layer, outputs=output_layer)

    def Yolo_v1(self):
        initializer = initializers_v2.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        leaky_relu = LeakyReLU(alpha=0.1)
        regularizer = l2(5e-4)  # L2 규제 == weight decay.

        if self.backbone == 'densenet':
            backbone = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        elif self.backbone == 'darknet19':
            backbone = self.get_DarkNet19()

        else:
            backbone = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        if self.backbone != 'darknet19':
            backbone.trainable = False
            x = backbone.output
            x = self.conv_block(x, 1024, 3)
            x = self.conv_block(x, 1024, 3, pool=True)

            x = self.conv_block(x, 1024, 3)
            x = self.conv_block(x, 1024, 3)

            x = GlobalAveragePooling2D()(x)
            x = Dense(4096, activation=leaky_relu, kernel_initializer=initializer, kernel_regularizer=regularizer,
                      name="detection_linear1", dtype='float32')(x)
            x = Dropout(0.5)(x)

            x = Dense(1470, kernel_initializer=initializer, kernel_regularizer=regularizer,
                      name="detection_linear2", dtype='float32')(x)
            output_layer = Reshape((7, 7, 30), name='output', dtype='float32')(x)

            YOLO = keras.Model(inputs=backbone.input, outputs=output_layer)

            return YOLO

        else:
            return backbone