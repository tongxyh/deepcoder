import os
import sys
gpu_idx = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

import tensorflow as tf
#import h5py
import numpy as np
import matplotlib.pyplot as plt


sys.path.append('/home/chentong/deepcoder/WeightedQUAN')
sys.path.append('/home/chentong/deepcoder/python-huffman')
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet') #utili.py
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet/blockpredict') #block
import blockpred as bpred
import msssim
import utili
import tofile
import time

def deep_decoder_v11(q_x):
    dconv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_0")(q_x)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0)
    dconv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_1")(up_0)
    up_1 = tf.keras.layers.UpSampling2D((2,2))(dconv_1)
    dconv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_2")(up_1)
    up_2 = tf.keras.layers.UpSampling2D((2,2))(dconv_2)
    dconv_3 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_3")(up_2)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_4")(dconv_3)
    return output

test_img = plt.imread("/home/chentong/deepcoder/WeightedQUAN/kodar/kodim01.bmp")
#print(test_img)
IMG_H, IMG_W, IMG_C = test_img.shape

ModelIndex = sys.argv[1]
train_idx = sys.argv[2]
input_idx = "kodim01"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #huffman decode

    time_start = time.time()

    bin_data = tofile.read("Input-"+ input_idx + "-Model-"+ModelIndex+"-train-"+str(train_idx)+'.deepc')
    time_decode_huffman = time.time()

    N,H,W,C = bin_data.shape

    q_x = tf.placeholder(tf.float32, shape=[N, H, W, C])
    out = deep_decoder_v11(q_x)

    saver = tf.train.Saver()
    saver.restore(sess, "/home/chentong/deepcoder/WeightedQUAN/DenseNet/DeepCoder-20170928/models/"+str(train_idx)+"/model.ckpt-"+ModelIndex)

    [recons] = sess.run([out],feed_dict={q_x : bin_data})

    recons[recons > 1] = 1
    recons[recons < 0] = 0

    plt.imsave("./result/Input-"+ input_idx + "-Model-"+ModelIndex+"-train-"+str(train_idx)+".png",recons[0])
    time_end=time.time()
    print('huffman decode cost:',time_decode_huffman-time_start ,'s')
    print('totally cost',time_end-time_start,'s')
    #ms-ssim
    ms_ssim = msssim.MultiScaleSSIM(np.reshape(test_img,[1, IMG_H, IMG_W, IMG_C]), recons*255.0, max_val=255, filter_size=11, filter_sigma=1.5,
                       k1=0.01, k2=0.03, weights=None)
    print('ms_ssim:', ms_ssim)
