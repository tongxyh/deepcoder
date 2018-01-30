import os
#using GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import sys
sys.path.append('/home/chentong/deepcoder/WeightedQUAN/DenseNet') #utili.py
import h5py
import numpy as np

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

'''
def quantizer_norm(x):
    g_x = tf.round(x*tf.reduce_max(x))/10.0
    return g_x
'''

def cal_rloss(x,value_range = [-16.0,16.0],nbins=64):
    #vmax = tf.reduce_max(x)
    #vmin = tf.reduce_min(x)
    vmin = -16.0
    vmax = 16.0
    interval = (vmax - vmin) / nbins
    #vmax = tf.argmax(tf.argmax(tf.argmax(tf.argmax(x,axis = 3),axis = 2),axis = 1),axis=0)  #it return index not value
    #vmax_1 = tf.argmax(tf.argmax(tf.argmax(tf.argmax(x,axis = -1),axis = -1),axis = -1),axis=-1) #same
    #vmin = tf.cast(vmin, tf.float32) # int to float - correct
    #print(vmin,vmax)

    hist = tf.histogram_fixed_width(x,[vmin,vmax],nbins=nbins,dtype=tf.int32,name="hist")
    #print(hist)
    hist = tf.cast(hist,tf.float32)
    ele_sum = tf.reduce_sum(hist)
    hist = hist / ele_sum
    #g = tf.Variable(tf.zeros([nbins], tf.float32),trainable=False)
    #d = tf.zeros(x.shape, tf.float32)

    rloss = 0.0
    a = tf.Variable(0. ,tf.float32)
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
    return rloss/ele_sum,d
'''
    def forward(self, x):
        q_x = tf.round(x)
        q_x = tf.fake_quant_with_min_max_args_gradient(gradients,inputs,min=-6,max=6,num_bits=8)
        y,idx,count = tf.unique_with_counts(q_x)
        print(y,idx,count)
        return q_x,y,idx,count
        #return K.dot(x, self.kernel)
'''
def conv(x, Wx, Wy,inputchannels, outputchannels, stridex=1, stridey=1, padding='SAME', transpose=False, name='conv'):
    w = tf.get_variable(name+"/w",[Wx, Wy, inputchannels, outputchannels], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(name+"/b",[outputchannels], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, w, strides=[1,stridex,stridey,1], padding=padding) + b
    return conv

def deepcoder(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_0")(in_gt)
    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(conv_0)
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_1")(pool_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_2")(pool_1)
    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(conv_2)
    conv_3 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',name = "Conv2D_3")(pool_2)

    q_x = quantizer(conv_3)
    #q_1 = quantizer_v2()
    #q_2 = quantizer_v4()
    #q_3 = quantizer_v6()
    #q_4 = quantizer_v8()
    #q_x = mask_fusion(q_1,q_2,q_3,q_4)

    dconv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_0")(q_x)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0)
    dconv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_1")(up_0)
    up_1 = tf.keras.layers.UpSampling2D((2,2))(dconv_1)
    dconv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_2")(up_1)
    up_2 = tf.keras.layers.UpSampling2D((2,2))(dconv_2)
    dconv_3 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_3")(up_2)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_4")(dconv_3)
    return conv_3,q_x,output

file_train = h5py.File(r'../FeatureVisualization/train64.h5','r')
train_data = file_train['data'][:].transpose(0,3,2,1)

#file_test = h5py.File(r'../FeatureVisualization/test64.h5','r')
#test_data = file_test['data'][:].transpose(0,3,2,1)

datasize = train_data.shape[0]

IMG_W = 64
IMG_H = 64
IMG_C = 3

train_index = 11

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    batchsize = 50
    lr = 0.0001
    lamda = 0.001

    in_gt = tf.placeholder(tf.float32, shape=[batchsize, IMG_H, IMG_W, IMG_C])
    x_0,q_x,out = deepcoder(in_gt,IMG_W,IMG_H,IMG_C)
    #lr1 = tf.placeholder(tf.float32, name="lr")

    rloss,d = cal_rloss(x_0) # vmin_0,vmin_0_f
    dloss = tf.reduce_mean(tf.square(out - in_gt))

    loss = lamda * rloss + dloss

    tf.summary.scalar('rloss', rloss)
    tf.summary.scalar('dloss', dloss)
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('bin_data', q_x)
    #tf.summary.tensor_summary('ditrib',rloss[0])

    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    #g_after_round = tf.gradients(loss,q_x)
    #g_before_round = tf.gradients(loss,x_0)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log/rd_v0/train-'+str(train_index), sess.graph)
    #test_writer = tf.summary.FileWriter('./log/test-0')

    tf.global_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=100)
    #saver.restore(sess, "/home/penglu/Desktop/lp/model.ckpt")
    for epoch in range(100):
        shuffle_idx = np.random.permutation(datasize)
        batch_num = int(datasize/batchsize)
        ls = 0
        #lr = learningrate * (np.minimum((4 - epoch/1000.), 3.)/3)
        for i in range(batch_num):
            summary,_,vq_x,vrloss,vdloss,vloss,vd = sess.run([merged,optimizer,q_x,rloss,dloss,loss,d],feed_dict={in_gt : train_data[shuffle_idx[batchsize*i:batchsize*(i+1)]]})
            ls = ls + vloss;

            #vloss avg
            #dloss avg

            print('epoch:',epoch,'loss: ',ls/(i+1),"r: ", vrloss,"d: ",vdloss)
            #print(vrloss,vdloss)
            train_writer.add_summary(summary, batch_num+epoch*batchsize)
        #loss = evaluate(test_data)
        #sess.run([merged,q_x,rloss,dloss,loss],feed_dict={in_gt : batch[batchsize*i:batchsize*(i+1),:,:,:]})

        saver.save(sess, "/home/chentong/deepcoder/WeightedQUAN/DenseNet/DeepCoder-20170928/models/"+str(train_index)+"/model.ckpt",global_step=epoch)

train_writer.close()
