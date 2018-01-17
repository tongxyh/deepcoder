#using GPU 1
#set this before Keras / Tensorflow is imported.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras
from keras.layers import Conv2D,MaxPool2D,Input,BatchNormalization,Lambda
from keras.layers import Activation,concatenate,AveragePooling2D,UpSampling2D
from keras.models import Model,load_model
import h5py
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import argparse
import sys
import scipy
import copy


sys.path.append('/home/chentong/deepcoder/WeightedQUAN')
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet')
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet/blockpredict')
import block
import msssim
import utili

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

parser = argparse.ArgumentParser(description='DONT KNOW WHAT TO SHOW')

parser.add_argument('--Index', type=str, dest="index", default= '1', help='ImageIndex in Kodark Images')
parser.add_argument('--QP', type=int, default= 8, dest="qp", help='ImageIndex in Kodark Images')
parser.add_argument('--Model', type=str, default= '12', dest="model", help='ImageIndex in Kodark Images')
parser.add_argument('--channel', type=int, default= 0, dest="channel", help='Channel Index in Feature maps')

args = parser.parse_args()

ImageIndex = args.index
QuanBits = args.qp
ModelIndex = args.model

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

if(int(args.model) == 24):
# 1/24
    x = Conv2D(2,(1,1),padding='same')(x)
else:
# 1/12
    x = Conv2D(4,(1,1),padding='same')(x)

x = dense_block6(x,12,0.9)

x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Conv2D(16,(1,1),padding='same')(x)

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
autoencoder.load_weights('/home/chentong/deepcoder/WeightedQUAN/DenseNet/weights/weight-1-'+ModelIndex+'.hdf5')

encoder = K.function([autoencoder.input],[autoencoder.get_layer('conv2d_36').output])

decoder = K.function([autoencoder.get_layer('conv2d_36').output],[autoencoder.layers[-1].output])

encoder_maps = encoder([im])[0] # feature maps
encoder_map = copy.deepcopy(encoder_maps)

#Normalization & Quantization
max_val = np.max(encoder_maps)
min_val = np.min(encoder_maps)

#QUAN_LEV = 2**QuanBits
QUAN_LEV = 1.0
encoder_maps = (encoder_maps - min_val) / (max_val - min_val)

mid_channel=2
for i in range(mid_channel):
	cv2.imwrite('res'+str(i)+'.png',encoder_maps[0][:,:,i]*255.0)

os.system("ffmpeg  -f image2 -i ./res%d.png -c:v libx264 -r 20 -qp 25 res.h264")
os.system("ffmpeg -i  ./res.h264 -r 20 -f image2 decode%d.png")

for i in range(0,mid_channel):
    encoder_map[0][:,:,i] = plt.imread('./decode'+str(i+1)+'.png')[:,:,0]

'''
#huffman
ModelLevel = int(ModelIndex)/3.0

avgbits,codec0 = utili.huffman_coding(encoder_maps, 0 , QUAN_LEV,H,W)
avgbits = avgbits/ModelLevel + np.double(utili.huffman_head(codec0)) / H / W

bi_avg, bi_res = block.crop(encoder_maps)
bi_avgbits,codec1 = utili.huffman_coding(bi_avg, 0 ,QUAN_LEV,H,W)
bi_resbits,codec2 = utili.huffman_coding(bi_res,-QUAN_LEV,QUAN_LEV,H,W)
bi_prebits = bi_avgbits / ModelLevel / 16.0 + bi_resbits / ModelLevel + np.double(utili.huffman_head(codec1)) / H / W + np.double(utili.huffman_head(codec2)) / H / W
'''
#decode
recons = decoder([encoder_map*(1.0/QUAN_LEV)*(max_val - min_val) + min_val])[0]
#res = autoencoder.predict(im)

ms_ssim = msssim.MultiScaleSSIM(im*255.0, recons*255.0, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None)

'''
print("orginal bits:", avgbits,"after prediction:", bi_prebits)
if avgbits < bi_prebits:
    print("bpp: ",avgbits," MS-SSIM:",ms_ssim)
else:
    print("bpp: ",bi_prebits," MS-SSIM:",ms_ssim)
'''
print(ms_ssim)
#recons = recons.reshape(H,W,3)*255.0
recons = recons.reshape(H,W,3)

plt.imsave('/home/chentong/deepcoder/WeightedQUAN/DenseNet/DeepCoder-20170928/result/x264-'+ImageIndex+'-'+str(QuanBits)+'-'+ModelIndex+'-'+str(args.channel)+'.bmp',recons)
