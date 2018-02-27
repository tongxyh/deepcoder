import os
import sys
gpu_idx = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/chentong/deepcoder/python-huffman')
sys.path.append('/home/chentong/deepcoder/DenseNet') #utili.py
sys.path.append('/home/chentong/deepcoder/DenseNet/blockpredict') #block
import blockpred as bpred
#import msssim
import utili
import tofile
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

ModelIndex = sys.argv[1]
train_idx = sys.argv[2]

filedir = sys.argv[4]
input_idx = sys.argv[5]

test_img = plt.imread(filedir)
[H_ORG,W_ORG,C] = test_img.shape
print(test_img)
# divided by 8
H_PAD = int(8 * np.ceil(H_ORG / 8.0))
W_PAD = int(8 * np.ceil(W_ORG / 8.0))
# print(H_PAD,W_PAD)

im = np.zeros([H_PAD,W_PAD,C],dtype = 'float32')
im[:H_ORG,:W_ORG,:] = test_img

H, W, C = im.shape
print(im)
print("Image Size: ",im.shape)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    in_gt = tf.placeholder(tf.float32, shape=[1, H, W, C])
    x_0,q_x,out = model.deepcoder_bn(in_gt)
    saver = tf.train.Saver()
    saver.restore(sess, "models/"+str(train_idx)+"/model.ckpt-"+ModelIndex)

    x_data,fmaps_data,recons = sess.run([x_0,q_x,out],feed_dict={in_gt : [im]})
    #print("fmaps_data",fmaps_data)
    print("other_data",x_data)
    print("recons:",recons)
    #print(bin_data)
    fmaps_data = np.round(fmaps_data*31.98-0.49)
    #huffman
    #print("bin_data",int(bin_data))
    #print(bin_data)
    codec0 = bits_cal(fmaps_data,H_ORG,W_ORG,QUAN_B=0,QUAN_LEV=31)
    #print(codec0)
    tofile.write("bin/"+input_idx + "-Model-"+ModelIndex+"-train-"+str(train_idx)[-2:]+'.deepc',test_img,fmaps_data,codec0)
