# Single Image Encoder
# Updated 2018.03.30 by Tong Chen

import os
import sys
gpu_idx = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tools import blockpred as bpred
from tools import utili,tofile
import model

def bits_cal(arr,H,W,QUAN_B=0,QUAN_LEV=256):
    sum_ele = arr.shape[0] * arr.shape[1] * arr.shape[2] * arr.shape[3]
    avgbits,codec0 = utili.huffman_coding(arr, QUAN_B , QUAN_LEV,H,W)
    '''
    avgbits = avgbits * sum_ele * 1.0 / H / W + np.double(utili.huffman_head(codec0)) / H / W

    bi_avg, bi_res = bpred.crop(arr.transpose(0,2,3,1))
    bi_avgbits,codec1 = utili.huffman_coding(bi_avg,-QUAN_LEV,QUAN_LEV,H,W)
    bi_resbits,codec2 = utili.huffman_coding(bi_res,-QUAN_LEV,QUAN_LEV,H,W)
    prebits = bi_avgbits*sum_ele/16.0/H/W + bi_resbits*sum_ele*1.0/H/W + np.double(utili.huffman_head(codec1)) / H / W + np.double(utili.huffman_head(codec2)) / H / W
    '''
    return codec0

#ModelDir & InputDir & OutputDir
ModelDir = sys.argv[1]
InputDir = sys.argv[2]
OutputDir = sys.argv[3]

test_img = plt.imread(InputDir)
[H_ORG,W_ORG,C] = test_img.shape
# divided by 8
H_PAD = int(8 * np.ceil(H_ORG / 8.0))
W_PAD = int(8 * np.ceil(W_ORG / 8.0))
# print(H_PAD,W_PAD)

im = np.zeros([H_PAD,W_PAD,C],dtype = 'float32')
im[:H_ORG,:W_ORG,:] = test_img

H, W, C = im.shape
print("Image Size: ",test_img.shape)
print("Image Size After Padding: ",im.shape)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    in_gt = tf.placeholder(tf.float32, shape=[1, H, W, C])
    x_0,q_x,out = model.deepcoder_bn(in_gt,TRAIN_FLAG=False)
    saver = tf.train.Saver()
    saver.restore(sess, ModelDir)

    fmaps_data,recons = sess.run([q_x,out],feed_dict={in_gt : [im]})

    Bits = 42 # use tf.FLAGS in model.py to store this

    fmaps_data = np.round(fmaps_data*(Bits-0.02)-0.49)

    codec0 = bits_cal(fmaps_data,H_ORG,W_ORG,QUAN_B=0,QUAN_LEV=Bits-1)
    tofile.write(OutputDir,test_img,fmaps_data,codec0)
