import os
import sys
gpu_idx = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

import tensorflow as tf
#import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/chentong/deepcoder/python-huffman')
sys.path.append('/home/chentong/deepcoder/DenseNet') #utili.py
sys.path.append('/home/chentong/deepcoder/DenseNet/blockpredict') #block
import blockpred as bpred
import msssim
import utili
import tofile
import time
import model

ModelIndex = sys.argv[1]
train_idx = sys.argv[2]

input_idx = sys.argv[4]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #huffman decode

    time_start = time.time()

    bin_data = tofile.read("bin/"+input_idx + "-Model-"+ModelIndex+"-train-"+str(train_idx)[-2:]+'.deepc')
    time_decode_huffman = time.time()

    N,H,W,C = bin_data.fmaps_info.fmaps_shape
    IMG_H, IMG_W, IMG_C = bin_data.iminfo.imshape

    #q_x = tf.placeholder(tf.float32, shape=[N, H, W, C])
    in_gt = tf.placeholder(tf.float32, shape=[1, IMG_H, IMG_W, IMG_C])

    q_x = model.deepcoder_bn_encoder(in_gt)
    out = model.deepcoder_bn_decoder(q_x)

    saver = tf.train.Saver()
    saver.restore(sess, "models/"+str(train_idx)+"/model.ckpt-"+ModelIndex)

    in_data = (bin_data.data+0.49)/31.98
    print(in_data)
    [recons] = sess.run([out],feed_dict={q_x : in_data})

    recons[recons > 1] = 1
    recons[recons < 0] = 0
    recons = recons[:,:IMG_H,:IMG_W,:]

    print(recons[0])

    plt.imsave("./result/Input-"+ input_idx + "-Model-"+ModelIndex+"-train-"+str(train_idx)[-2:]+".png",recons[0])
    time_end=time.time()
    print('huffman decode cost:',time_decode_huffman-time_start ,'s')
    print('totally cost',time_end-time_start,'s')
