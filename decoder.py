# Single Image Decoder
# Updated 2018.03.30 by Tong Chen

import os
import sys
gpu_idx = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

import tensorflow as tf
#import h5py
import numpy as np
import matplotlib.pyplot as plt

from tools import blockpred as bpred
from tools import utili,tofile
import model
import time

#ModelDir & InputDir & OutputDir
ModelDir = sys.argv[1]
InputDir = sys.argv[2]
OutputDir = sys.argv[3]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    time_start = time.time()

    bin_data = tofile.read(InputDir)
    time_decode_huffman = time.time()

    N,H,W,C = bin_data.fmaps_info.fmaps_shape
    IMG_H, IMG_W, IMG_C = bin_data.iminfo.imshape

    #q_x = tf.placeholder(tf.float32, shape=[N, H, W, C])
    in_gt = tf.placeholder(tf.float32, shape=[1, IMG_H, IMG_W, IMG_C])

    q_x = model.deepcoder_bn_encoder(in_gt,TRAIN_FLAG=False)
    out = model.deepcoder_bn_decoder(q_x,TRAIN_FLAG=False)

    saver = tf.train.Saver()
    saver.restore(sess, ModelDir)

    Bits = 42
    in_data = (bin_data.data+0.49)/(Bits-0.02)

    [recons] = sess.run([out],feed_dict={q_x : in_data})

    recons[recons > 1] = 1
    recons[recons < 0] = 0
    recons = recons[:,:IMG_H,:IMG_W,:]

    plt.imsave(OutputDir,recons[0])

    time_end=time.time()
    print('Huffman Decode Time:',time_decode_huffman-time_start ,'s')
    print('Network Forward Time',time_end-time_decode_huffman,'s')
    print('Total Time',time_end-time_start,'s')
