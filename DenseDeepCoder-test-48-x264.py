#import tensorflow as tf
import keras
from keras.layers import Conv2D,MaxPool2D,Input,BatchNormalization,Lambda
from keras.layers import Activation,concatenate,AveragePooling2D,UpSampling2D
from keras.models import Model,load_model
import h5py
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import cv2
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import argparse
import sys,os

sys.path.append('/home/chentong/deepcoder/WeightedQUAN')
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet')
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet/blockpredict')
import block
import msssim
import utili
import copy

K.set_learning_phase(0)
#import pydot

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

def dense_block6_0(x0,growth_rate,bn_num):
    k = growth_rate   #define the width of the network(12,32,40)

    x = BatchNormalization(momentum=bn_num)(x0)
    x_mid = Activation('relu')(x)
    #x = Lamaba(lambda x:np.round(x*64))

    x = Conv2D(4*k, (1, 1), padding='same')(x_mid)
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

ImageIndex = sys.argv[1]
QuanBits = int(sys.argv[2])
ModelIndex = sys.argv[3]

#mod = load_model(r'C:\Users\lhj\Desktop\Adaptive compression\model4\27-0.0028142.hdf5')
#mod.save_weights(r'C:\Users\lhj\Desktop\Adaptive compression\weights\weights.h5')
img = plt.imread('/home/chentong/deepcoder/WeightedQUAN/kodar/kodim0'+ImageIndex+'.bmp')
H,W,C = img.shape
im = img.reshape(1,H,W,C)/255.0

input_img = Input(shape=(H,W,3))
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
x = Conv2D(4,(1,1),padding='same')(x)

########
x = AveragePooling2D((2,2))(x)
########

x = dense_block6_0(x,12,0.9)

x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Conv2D(16,(1,1),padding='same')(x)
x = UpSampling2D((2,2))(x)

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
#autoencoder.summary()
autoencoder.load_weights('/home/chentong/deepcoder/WeightedQUAN/DenseNet/weights.h5')

encoder = K.function([autoencoder.input],[autoencoder.get_layer('average_pooling2d_3').output])


decoder = K.function([autoencoder.get_layer('average_pooling2d_3').output],[autoencoder.layers[-1].output])

encoder_maps = encoder([im])[0]

encoder_map = copy.deepcopy(encoder_maps)
#print (encoder_maps.shape,np.max(encoder_maps),np.min(encoder_maps))

max_val = np.max(encoder_maps)
min_val = np.min(encoder_maps)

QUAN_LEV = 2**0
encoder_maps = (encoder_maps - min_val) * 1.0 / (max_val - min_val)

mid_channel = 4
for i in range(mid_channel):
	cv2.imwrite('res'+str(i)+'.png',encoder_maps[0][:,:,i]*255.0)

os.system("ffmpeg  -f image2 -i ./res%d.png -c:v libx264 -r 20 -qp " + str(QuanBits) + " res.h264")
os.system("rm ./res*.png")
os.system("ffmpeg -i  ./res.h264 -r 20 -f image2 decode%d.png")

for i in range(0,mid_channel):
    encoder_map[0][:,:,i] = plt.imread('./decode'+str(i+1)+'.png')[:,:,0]

os.system("rm ./decode*.png")

recons = decoder([encoder_map*1.0/QUAN_LEV*(max_val - min_val) + min_val])[0]

recons[recons > 1] = 1
recons[recons < 0] = 0
ms_ssim = msssim.MultiScaleSSIM(im*255.0, recons*255.0, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None)

recons = recons.reshape(H,W,3)

#res = img - recons
print(ms_ssim)

plt.imsave('/home/chentong/deepcoder/WeightedQUAN/DenseNet/DeepCoder-20170928/result/result-x264-'+ImageIndex+'-'+str(QuanBits)+'-'+ModelIndex+'.bmp',recons)
#cv2.imwrite('/home/chentong/deepcoder/WeightedQUAN/DenseNet/residual.bmp',10*res)
