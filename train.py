import os
#using GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
#use 80% of the GPU memory
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)

import keras
from keras.layers import Conv2D,MaxPool2D,Input,BatchNormalization
from keras.layers import Activation,concatenate,AveragePooling2D,UpSampling2D
from keras.models import Model,load_model
import h5py
from keras.callbacks import ModelCheckpoint
from keras import optimizers

#create a dense_block for deeper
def dense_block4(x0,growth_rate,bn_num):
    k = growth_rate   #define the width of the network(12,32,40)

    x = BatchNormalization(momentum=bn_num)(x0)
    x = Activation('relu')(x)
    x = Conv2D(4 * k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x1 = concatenate([x0, x])
    x = BatchNormalization(momentum=bn_num)(x1)
    x = Activation('relu')(x)
    x = Conv2D(4 * k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x2 = concatenate([x0, x1, x])
    x = BatchNormalization(momentum=bn_num)(x2)
    x = Activation('relu')(x)
    x = Conv2D(4 * k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x3 = concatenate([x0, x1, x2, x])
    x = BatchNormalization(momentum=bn_num)(x3)
    x = Activation('relu')(x)
    x = Conv2D(4 * k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x4 = concatenate([x0, x1, x2, x3, x])
    return x4

def dense_block6(x0,growth_rate,bn_num):
    k = growth_rate   #define the width of the network(12,32,40)

    x = BatchNormalization(momentum=bn_num)(x0)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x1=concatenate([x0,x])
    x = BatchNormalization(momentum=bn_num)(x1)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x2 = concatenate([x0,x1,x])
    x = BatchNormalization(momentum=bn_num)(x2)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x3 = concatenate([x0, x1, x2,x])
    x = BatchNormalization(momentum=bn_num)(x3)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x4 = concatenate([x0, x1, x2, x3, x])
    x = BatchNormalization(momentum=bn_num)(x4)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x5 = concatenate([x0, x1, x2, x3,x4, x])
    x = BatchNormalization(momentum=bn_num)(x5)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=bn_num)(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)
    x6 = concatenate([x0, x1, x2, x3, x4,x5, x])
    return x6

file_train = h5py.File(r'train64.h5','r')
train_data = file_train['data'][:].transpose(0,3,2,1)
file_test = h5py.File(r'test64.h5','r')
test_data = file_test['data'][:].transpose(0,3,2,1)

input_img = Input(shape=(64,64,3))

x = BatchNormalization(momentum=0.9)(input_img)
x = Activation('relu')(x)
x = Conv2D(32,(5,5),padding='same')(x)

#dense_block 1
x = dense_block4(x,12,0.9)
#transition layer 1
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Conv2D(16,(1,1),padding='same')(x)
x = AveragePooling2D((2,2))(x)
#dense_block 2
x = dense_block6(x,12,0.9)

x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Conv2D(16,(1,1),padding='same')(x)
x = AveragePooling2D((2,2))(x)
#dense block 3
x = dense_block6(x,12,0.9)

x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)

x_big = Conv2D(1,(1,1),padding='same')(x) # 1/1 middle layer

x_small = Conv2D(3,(1,1),padding='same')(x)
x = AveragePooling2D((2,2))(x_small)
x = dense_block6(x,12,0.9) # 1/4 middle layer
x = UpSampling2D((2,2))(x)
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Conv2D(3,(1,1),padding='same')(x)
#x = AveragePooling2D((2,2))(x)

x = concatenate([x_big, x])
########....
x = dense_block6(x,12,0.9)

x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Conv2D(16,(1,1),padding='same')(x)
#x = UpSampling2D((2,2))(x)

x = dense_block6(x,12,0.9)

x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Conv2D(16,(1,1),padding='same')(x)
x = UpSampling2D((2,2))(x)

x = dense_block4(x,32,0.9)

x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Conv2D(16,(1,1),padding='same')(x)
x = UpSampling2D((2,2))(x)

x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Conv2D(3,(3,3),padding='same')(x)
autoencoder = Model(input_img,x)

autoencoder.compile(optimizer='adam', loss='mse')

#sgd = optimizers.SGD(lr=0.00001,decay=1e-8,momentum=0.99,nesterov=True)
#autoencoder.compile(optimizer='sgd', loss='mse')

checkpoint=ModelCheckpoint(filepath=r'models/0/{epoch:02d}-{val_loss:.7f}.hdf5',period=1)
autoencoder.fit(train_data, train_data,
                epochs=100,
                batch_size=96,
                shuffle=True,
                validation_data=(test_data, test_data),callbacks=[checkpoint])
