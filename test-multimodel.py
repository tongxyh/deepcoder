import os
import sys
gpu_idx = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt


sys.path.append('/home/chentong/deepcoder/WeightedQUAN')
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet') #utili.py
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet/blockpredict') #block
import blockpred as bpred
import msssim
import utili

G = tf.get_default_graph()
def quantizer(x):
    with G.gradient_override_map({"round": "Identity"}):
        g_x = tf.round(x)
        return g_x

def deepcoder_v3(in_gt,IMG_W,IMG_H,IMG_C):
    x = tf.keras.layers.Conv2D(4, (1, 1), padding='same',name = "Conv2D_0")(in_gt)
    x_0 = tf.keras.layers.AveragePooling2D((2,2))(x)
    q_x = quantizer(x_0)
    x1 = tf.keras.layers.Conv2D(4, (1, 1), padding='same',name= "Conv2D_1")(q_x)
    x2 = tf.keras.layers.UpSampling2D((2,2))(x1)
    output = tf.keras.layers.Conv2D(3, (1, 1), padding='same',name = "Conv2D_2")(x2)
    return x_0,q_x,output

def deepcoder_v4(in_gt,IMG_W,IMG_H,IMG_C):
    x = tf.keras.layers.Conv2D(4, (3, 3), padding='same',name = "Conv2D_0")(in_gt)
    x_0 = tf.keras.layers.AveragePooling2D((2,2))(x)
    q_x = quantizer(x_0)
    x1 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',name= "Conv2D_1")(q_x)
    x2 = tf.keras.layers.UpSampling2D((2,2))(x1)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "Conv2D_2")(x2)
    return x_0,q_x,output

def deepcoder_v9(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_0")(in_gt)
    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(conv_0)
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_1")(pool_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',name = "Conv2D_2")(pool_1)
    q_x = quantizer(conv_2)
    dconv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_0")(q_x)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0)
    dconv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_1")(up_0)
    up_1 = tf.keras.layers.UpSampling2D((2,2))(dconv_1)
    dconv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_2")(up_1)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_3")(dconv_2)
    return conv_2,q_x,output

def deepcoder_v10(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_0")(in_gt)
    pool_0 = tf.keras.layers.AveragePooling2D((4,4))(conv_0)
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_1")(pool_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',name = "Conv2D_2")(pool_1)

    q_x = quantizer(conv_2)

    dconv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_0")(q_x)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0)
    dconv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_1")(up_0)
    up_1 = tf.keras.layers.UpSampling2D((4,4))(dconv_1)
    dconv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_2")(up_1)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_3")(dconv_2)
    return conv_2,q_x,output

def deepcoder_v11(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_0")(in_gt)
    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(conv_0)
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_1")(pool_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_2")(pool_1)
    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(conv_2)
    conv_3 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',name = "Conv2D_3")(pool_2)

    q_x = quantizer(conv_3)

    dconv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_0")(q_x)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0)
    dconv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_1")(up_0)
    up_1 = tf.keras.layers.UpSampling2D((2,2))(dconv_1)
    dconv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_2")(up_1)
    up_2 = tf.keras.layers.UpSampling2D((2,2))(dconv_2)
    dconv_3 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_3")(up_2)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_4")(dconv_3)
    return conv_3,q_x,output

test_img = plt.imread("/home/chentong/deepcoder/WeightedQUAN/kodar/kodim02.bmp")
#print(test_img)
[IMG_H,IMG_W,IMG_C] = test_img.shape

ModelIndex_Range = sys.argv[1]
train_idx = sys.argv[2]
input_idx = "kodim02"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    in_gt = tf.placeholder(tf.float32, shape=[1, IMG_H, IMG_W, IMG_C])
    x_0,q_x,out = deepcoder_v11(in_gt,IMG_W,IMG_H,IMG_C)
    saver = tf.train.Saver()
    for ModelIndex in range(0,int(ModelIndex_Range)):
        saver.restore(sess, "/home/chentong/deepcoder/WeightedQUAN/DenseNet/DeepCoder-20170928/models/"+str(train_idx)+"/model.ckpt-"+str(ModelIndex))

        bin_data,recons = sess.run([q_x,out],feed_dict={in_gt : [test_img/255.0]})
        #huffman
        #ModelLevel
        ModelLevel = 48.0/3.0 #

        QUAN_LEV = 16
        vmax = np.max(bin_data)
        vmin = np.min(bin_data)

        avgbits,codec0 = utili.huffman_coding(bin_data, -QUAN_LEV , QUAN_LEV,IMG_H,IMG_W)
        avgbits = avgbits/ModelLevel + np.double(utili.huffman_head(codec0)) / IMG_H / IMG_W

        bi_avg, bi_res = bpred.crop(bin_data)
        bi_avgbits,codec1 = utili.huffman_coding(bi_avg, -QUAN_LEV , QUAN_LEV,IMG_H,IMG_W)
        bi_resbits,codec2 = utili.huffman_coding(bi_res,-QUAN_LEV,QUAN_LEV,IMG_H,IMG_W)
        bi_prebits = bi_avgbits / ModelLevel / 16.0 + bi_resbits / ModelLevel + np.double(utili.huffman_head(codec1)) / IMG_H / IMG_W + np.double(utili.huffman_head(codec2)) / IMG_H / IMG_W

        recons[recons > 1] = 1
        recons[recons < 0] = 0

        #ms-ssim
        ms_ssim = msssim.MultiScaleSSIM(np.reshape(test_img,[1, IMG_H, IMG_W, IMG_C]), recons*255.0, max_val=255, filter_size=11, filter_sigma=1.5,
                           k1=0.01, k2=0.03, weights=None)

        #print(ModelIndex,avgbits,bi_prebits,ms_ssim,vmin,vmax)
        print avgbits,bi_prebits,ms_ssim
        plt.imsave("./result/train-"+str(train_idx)+"/Input-"+ input_idx + "-Model-" + str(ModelIndex) + ".png" ,recons[0])
