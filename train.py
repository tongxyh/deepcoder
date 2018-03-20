import os
#using GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import sys
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet') #utili.py
import h5py
import numpy as np
import time
import model

'''
@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, x):
    # compute custom gradient
    # ...

'''

'''
G = tf.get_default_graph()
def quantize(x)
    with G.gradient_override_map({"Sign": "QuantizeGrad"}):
        E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
        return tf.sign(x / E) * E
'''

G = tf.get_default_graph()
def quantizer(x):
    with G.gradient_override_map({"Round": "Identity"}):
        g_x = tf.round(x)
        #print(g_x)
        return g_x

G = tf.get_default_graph()

def quantizer_norm(x):
    with G.gradient_override_map({"Round": "Identity"}):
        Bits = 41.98
        vmin = tf.stop_gradient(tf.reduce_min(x)) #stop_gradient
        vmax = tf.stop_gradient(tf.reduce_max(x)) #stop_gradient

        g_x_norm = Bits*(x - vmin)/ (vmax - vmin) - 0.49
        #g_x = (tf.round(g_x_norm) + 0.49)/Bits*(vmax - vmin) + vmin
    return g_x_norm,x

def cal_rloss_1(x,vmin=-0.5,vmax=31.5,nbins=32):
    interval = (vmax - vmin) / nbins

    hist = []
    hist_0 = tf.histogram_fixed_width(x,[vmin,vmax],nbins=nbins,dtype=tf.int32,name="hist_0")
    hist_1 = tf.histogram_fixed_width(x,[vmin+0.5,vmax-0.5],nbins=nbins-1,dtype=tf.int32,name="hist_0")

    hist_1 = tf.cast(hist_1,tf.float32)
    ele_sum = tf.reduce_sum(hist_1)
    hist_1 = tf.stop_gradient(hist_1 / ele_sum)

    hist_0 = tf.cast(hist_0,tf.float32)
    ele_sum = tf.reduce_sum(hist_0)
    hist_0 = tf.stop_gradient(hist_0 / ele_sum)

    for i in range(nbins-1):
        hist.append(hist_0[i])
        hist.append(hist_1[i])
    hist.append(hist_0[nbins-1])
    #print(hist)

    rloss = 0.0
    interval = interval/2.0
    g = (hist[1] - hist[0])/(interval*2.0)
    mask = tf.less(x,1*interval+(vmin+0.5))
    d = tf.boolean_mask(x,mask)
    rloss = tf.cond(tf.shape(d)[0] > 0,lambda: - tf.reduce_sum(tf.log((d - vmin) * g  + hist[0] + 0.00000001)) + rloss,lambda: rloss)

    for i in range(1,2*nbins - 3):
        g = (hist[i+1] - hist[i])/interval
        mask = tf.logical_and(tf.less(x,(i+1)*interval+vmin+0.5),tf.greater_equal(x,i*interval+vmin+0.5))

        d = tf.boolean_mask(x,mask)
        rloss = tf.cond(tf.shape(d)[0] > 0,lambda: - tf.reduce_sum(tf.log((d - (vmin+0.5+i*interval)) * g  + hist[i] + 0.00000001)) + rloss,lambda: rloss)

    g = (hist[2*nbins-2] - hist[2*nbins-3])/(interval*2.0)
    mask = tf.greater_equal(x,(2*nbins-3)*interval+vmin+0.5)
    d = tf.boolean_mask(x,mask)
    rloss = tf.cond(tf.shape(d)[0] > 0,lambda: - tf.reduce_sum(tf.log((d - (vmin+0.5+(2*nbins-3)*interval)) * g  + hist[2*nbins-3] + 0.00000001)) + rloss,lambda: rloss)

    return rloss/ele_sum , hist

def cal_rloss(x,vmin,vmax,nbins):
    interval = (vmax - vmin) / nbins
    #vmax = tf.argmax(tf.argmax(tf.argmax(tf.argmax(x,axis = 3),axis = 2),axis = 1),axis=0)  #it return index not value
    #vmax_1 = tf.argmax(tf.argmax(tf.argmax(tf.argmax(x,axis = -1),axis = -1),axis = -1),axis=-1) #same
    #vmin = tf.cast(vmin, tf.float32) # int to float - correct
    #print(vmin,vmax)

    hist = tf.histogram_fixed_width(x,[vmin,vmax],nbins=nbins,dtype=tf.int32,name="hist")
    #print(hist)
    hist = tf.cast(hist,tf.float32)
    ele_sum = tf.reduce_sum(hist)
    hist = tf.stop_gradient(hist / ele_sum)
    #g = tf.Variable(tf.zeros([nbins], tf.float32),trainable=False)
    #d = tf.zeros(x.shape, tf.float32)

    rloss = 0.0
    for i in range(0,nbins - 1):
        g = (hist[i+1] - hist[i])/interval
        mask = tf.logical_and(tf.less(x,(i+1.5)*interval+vmin),tf.greater_equal(x,(i+0.5)*interval+vmin))

        d = tf.boolean_mask(x,mask)
        rloss = tf.cond(tf.shape(d)[0] > 0,lambda: - tf.reduce_sum(tf.log((d - (vmin+(i+0.5)*interval)) * g  + hist[i] + 0.0001)) + rloss,lambda: rloss)
        #if tf.shape(d) > 0:
        #    a = a + 1
            #g[i].assign(d[1])
        #    rloss = - tf.reduce_mean(tf.log((d - (vmin+(i+0.5)*interval)) * g  + hist[i] + 0.0001)) + rloss
    #(x-vmin) * a[tf.ceil((x-vmin)*256.0/(vmax-vmin))] + hist[tf.ceil((x-vmin)*256.0/(vmax-vmin))]
    return rloss/ele_sum
'''
    def forward(self, x):
        q_x = tf.round(x)
        q_x = tf.fake_quant_with_min_max_args_gradient(gradients,inputs,min=-6,max=6,num_bits=8)
        y,idx,count = tf.unique_with_counts(q_x)
        print(y,idx,count)
        return q_x,y,idx,count
        #return K.dot(x, self.kernel)
'''

file_train = h5py.File(r'../FeatureVisualization/train64.h5','r')
train_data = file_train['data'][:].transpose(0,3,2,1)

datasize = train_data.shape[0]

IMG_W = 64
IMG_H = 64
IMG_C = 3

train_index = 12
ModelIndex = 80

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    batchsize = 50
    lr = 0.0001*0.1 # stage 2

    #lamda = 0.001
    lamda = 0.0005

    #optimizer = tf.train.AdamOptimizer(rate).minimize(cross_entropy, global_step=step)

    in_gt = tf.placeholder(tf.float32, shape=[batchsize, IMG_H, IMG_W, IMG_C])
    x_0,q_x,out = model.deepcoder_bn(in_gt,TRAIN_FLAG=True) # x_0 Normalized bin_data

    lr1 = tf.placeholder(tf.float32, name="lr")


    rloss,d = cal_rloss_1(x_0,vmin=-0.5,vmax=63.5,nbins=64)
    dloss = tf.reduce_mean(tf.square(out - in_gt))

    loss = lamda * rloss + dloss

    tf.summary.scalar('rloss', rloss)
    tf.summary.scalar('dloss', dloss)
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('bin_data', q_x)
    #tf.summary.tensor_summary('ditrib',rloss[0])

    optimizer = tf.train.AdamOptimizer(learning_rate = lr1).minimize(loss)
    #step = tf.Variable(0, trainable=False)
    #lr = tf.train.exponential_decay(0.0001, step, 1, 0.9999)
    #optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss,global_step=step)

    #g_after_round = tf.gradients(loss,q_x)
    #g_before_round = tf.gradients(loss,x_0)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log/rd_v1/train-'+str(train_index), sess.graph)
    #test_writer = tf.summary.FileWriter('./log/test-0')
    #load_model
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore(sess, "/home/chentong/deepcoder/WeightedQUAN/DenseNet/DeepCoder-20170928/models/rd_v1/"+str(train_index)+"/model.ckpt-"+str(ModelIndex))

    #step = tf.Variable(0, trainable=False)
    #lr = tf.train.exponential_decay(0.0001, step, 1, 0.9999)
    #optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss,global_step=step)

    #g_after_round = tf.gradients(loss,q_x)
    #g_before_round = tf.gradients(loss,x_0)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log/rd_v1/train-'+str(train_index), sess.graph)
    #test_writer = tf.summary.FileWriter('./log/test-0')


    #saver.restore(sess, "/home/penglu/Desktop/lp/model.ckpt")

    time_start = time.time()

    for epoch in range(100):

        shuffle_idx = np.random.permutation(datasize)
        batch_num = int(datasize/batchsize)
        ls = 0
        #lr = learningrate * (np.minimum((4 - epoch/1000.), 3.)/3)
        for i in range(batch_num):
            summary,_,vq_x,vrloss,vdloss,vloss = sess.run([merged,optimizer,q_x,rloss,dloss,loss],feed_dict={in_gt : train_data[shuffle_idx[batchsize*i:batchsize*(i+1)]], lr1:lr})
            ls = ls + vloss;

            #vloss avg
            #dloss avg

            train_writer.add_summary(summary, i+epoch*batch_num)

            batch_end_time = time.time()
            time_each_batch = (batch_end_time - time_start)/(i+1+epoch*batch_num+0.0)

            time_spent = batch_end_time - time_start
            time_remain = time_each_batch * ((100-epoch)*batch_num-i-1)

            time_hour = np.floor(time_spent/3600.0)
            time_min = np.floor((time_spent - time_hour * 3600.0)/60.0)
            time_sec = (time_spent - time_hour * 3600.0)%60.0
            time_print = str(int(time_hour)) + 'h ' + str(int(time_min)) + 'min ' + str(int(time_sec)) + 'sec'

            time_hr = np.floor(time_remain/3600.0)
            time_mr = np.floor((time_remain - time_hr * 3600.0)/60.0)
            time_sr = (time_remain - time_hr * 3600.0)%60.0
            timer_print = str(int(time_hr)) + 'h ' + str(int(time_mr)) + 'min ' + str(int(time_sr)) + 'sec'

            #print('epoch:',epoch,'loss: ',ls/(i+1),"r: ", vrloss,"d: ",vdloss)
            print 'epoch:',epoch,'loss:',ls/(i+1),'r:', vrloss,'d:',vdloss, 'time spent:', time_print, 'time remain:',timer_print

        saver.save(sess, "/home/chentong/deepcoder/WeightedQUAN/DenseNet/DeepCoder-20170928/models/rd_v1/"+str(train_index)+"/model.ckpt",global_step=epoch+81)

train_writer.close()
