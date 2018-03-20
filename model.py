# Model Description
# Updated 2018.03.20 by Tong Chen

import tensorflow as tf

G = tf.get_default_graph()
def quantizer_norm(x):
    with G.gradient_override_map({"Round": "Identity"}):
        Bits = 63.98
        vmin = tf.stop_gradient(tf.reduce_min(x)) #stop_gradient
        vmax = tf.stop_gradient(tf.reduce_max(x)) #stop_gradient

        g_x_norm = Bits* x - 0.49
        g_x = (tf.round(g_x_norm) + 0.49)/Bits
    return g_x_norm,g_x

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

def deepcoder_rd_v1_0(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_0")(in_gt)
    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(conv_0)
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_1")(pool_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_2")(pool_1)
    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(conv_2)
    conv_3 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',name = "Conv2D_3")(pool_2)

    q_x_norm , q_x = quantizer_norm(conv_3)
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
    return q_x_norm,q_x,output

def deepcoder_rd_v1_3(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_0")(in_gt)
    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(conv_0)
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_1")(pool_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_2")(pool_1)
    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(conv_2)
    conv_3 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',activation='relu',name = "Conv2D_3")(pool_2)

    q_x_norm , q_x = quantizer_norm(conv_3)

    dconv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0")(q_x)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0)
    dconv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1")(up_0)
    up_1 = tf.keras.layers.UpSampling2D((2,2))(dconv_1)
    dconv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2")(up_1)
    up_2 = tf.keras.layers.UpSampling2D((2,2))(dconv_2)
    dconv_3 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_3")(up_2)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',activation='relu',name = "dConv2D_4")(dconv_3)
    return q_x_norm,q_x,output

def deepcoder_rd_v1_5(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_0")(in_gt)
    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(conv_0)
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_1")(pool_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "Conv2D_2")(pool_1)
    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(conv_2)
    conv_3 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',name = "Conv2D_3")(pool_2)

    q_x_norm , q_x = quantizer_norm(conv_3)

    dconv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_0_0")(q_x)
    dconv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_0_1")(dconv_0_0)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0_1)
    dconv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_1_0")(up_0)
    dconv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_1_1")(dconv_1_0)
    up_1 = tf.keras.layers.UpSampling2D((2,2))(dconv_1_1)
    dconv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_2_0")(up_1)
    dconv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_2_1")(dconv_2_0)
    up_2 = tf.keras.layers.UpSampling2D((2,2))(dconv_2_1)
    dconv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name = "dConv2D_3_0")(up_2)
    dconv_3_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',name= "dConv2D_3_1")(dconv_3_0)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_3_2")(dconv_3_1)
    return q_x_norm,q_x,output

def deepcoder_rd_v1_8(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_0")(in_gt)
    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(conv_0)
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_1")(pool_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_2")(pool_1)
    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(conv_2)
    conv_3 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',activation='relu',name = "Conv2D_3")(pool_2)

    q_x_norm , q_x = quantizer_norm(conv_3)

    dconv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_0")(q_x)
    dconv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_1")(dconv_0_0)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0_1)
    dconv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_0")(up_0)
    dconv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_1")(dconv_1_0)
    up_1 = tf.keras.layers.UpSampling2D((2,2))(dconv_1_1)
    dconv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_0")(up_1)
    dconv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_1")(dconv_2_0)
    up_2 = tf.keras.layers.UpSampling2D((2,2))(dconv_2_1)
    dconv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_3_0")(up_2)
    dconv_3_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_3_1")(dconv_3_0)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_3_2")(dconv_3_1)
    return q_x_norm,q_x,output

def deepcoder_rd_v1_9(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_0")(in_gt)
    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(conv_0)
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_1")(pool_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_2")(pool_1)
    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(conv_2)
    conv_3 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',activation='sigmoid',name = "Conv2D_3")(pool_2)

    q_x_norm , q_x = quantizer_norm(conv_3)
    #q_1 = quantizer_v2()
    #q_2 = quantizer_v4()
    #q_3 = quantizer_v6()
    #q_4 = quantizer_v8()
    #q_x = mask_fusion(q_1,q_2,q_3,q_4)

    dconv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_0")(q_x)
    dconv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_1")(dconv_0_0)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0_1)
    dconv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_0")(up_0)
    dconv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_1")(dconv_1_0)
    up_1 = tf.keras.layers.UpSampling2D((2,2))(dconv_1_1)
    dconv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_0")(up_1)
    dconv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_1")(dconv_2_0)
    up_2 = tf.keras.layers.UpSampling2D((2,2))(dconv_2_1)
    dconv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_3_0")(up_2)
    dconv_3_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_3_1")(dconv_3_0)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_3_2")(dconv_3_1)
    return q_x_norm,q_x,output

def deepcoder_rd_v1_10(in_gt,IMG_W,IMG_H,IMG_C):
    conv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_0_0")(in_gt)
    conv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_0_1")(conv_0_0)
    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(conv_0_1)
    conv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_1_0")(pool_0)
    conv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_1_1")(conv_1_0)
    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(conv_1_1)
    conv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_2_0")(pool_1)
    conv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_2_1")(conv_2_0)
    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(conv_2_1)
    conv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_3_0")(pool_2)
    conv_3_1 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',activation='sigmoid',name = "Conv2D_3_1")(conv_3_0)

    q_x_norm , q_x = quantizer_norm(conv_3_1)
    #q_1 = quantizer_v2()
    #q_2 = quantizer_v4()
    #q_3 = quantizer_v6()
    #q_4 = quantizer_v8()
    #q_x = mask_fusion(q_1,q_2,q_3,q_4)

    dconv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_0")(q_x)
    dconv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_1")(dconv_0_0)
    up_0 = tf.keras.layers.UpSampling2D((2,2))(dconv_0_1)
    dconv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_0")(up_0)
    dconv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_1")(dconv_1_0)
    up_1 = tf.keras.layers.UpSampling2D((2,2))(dconv_1_1)
    dconv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_0")(up_1)
    dconv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_1")(dconv_2_0)
    up_2 = tf.keras.layers.UpSampling2D((2,2))(dconv_2_1)
    dconv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_3_0")(up_2)
    dconv_3_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_3_1")(dconv_3_0)
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_3_2")(dconv_3_1)
    return q_x_norm,q_x,output

#v31
def deepcoder_bn(in_gt,TRAIN_FLAG=True):
    conv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_0_0")(in_gt)
    bn_0_0 = tf.layers.batch_normalization(conv_0_0,training=TRAIN_FLAG)

    conv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_0_1")(bn_0_0)
    bn_0_1 = tf.layers.batch_normalization(conv_0_1,training=TRAIN_FLAG)

    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(bn_0_1)

    conv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_1_0")(pool_0)
    bn_1_0 = tf.layers.batch_normalization(conv_1_0,training=TRAIN_FLAG)

    conv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_1_1")(bn_1_0)
    bn_1_1 = tf.layers.batch_normalization(conv_1_1,training=TRAIN_FLAG)

    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(bn_1_1)

    conv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_2_0")(pool_1)
    bn_2_0 = tf.layers.batch_normalization(conv_2_0,training=TRAIN_FLAG)

    conv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_2_1")(bn_2_0)
    bn_2_1 = tf.layers.batch_normalization(conv_2_1,training=TRAIN_FLAG)

    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(bn_2_1)

    conv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_3_0")(pool_2)
    bn_3_0 = tf.layers.batch_normalization(conv_3_0,training=TRAIN_FLAG)

    conv_3_1 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',activation='sigmoid',name = "Conv2D_3_1")(bn_3_0)

    q_x_norm , q_x = quantizer_norm(conv_3_1)

    dconv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_0")(q_x)
    dbn_0_0 = tf.layers.batch_normalization(dconv_0_0,training=TRAIN_FLAG)

    dconv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_1")(dbn_0_0)
    dbn_0_1 = tf.layers.batch_normalization(dconv_0_1,training=TRAIN_FLAG)

    up_0 = tf.keras.layers.UpSampling2D((2,2))(dbn_0_1)

    dconv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_0")(up_0)
    dbn_1_0 = tf.layers.batch_normalization(dconv_1_0,training=TRAIN_FLAG)

    dconv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_1")(dbn_1_0)
    dbn_1_1 = tf.layers.batch_normalization(dconv_1_1,training=TRAIN_FLAG)

    up_1 = tf.keras.layers.UpSampling2D((2,2))(dbn_1_1)

    dconv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_0")(up_1)
    dbn_2_0 = tf.layers.batch_normalization(dconv_2_0,training=TRAIN_FLAG)

    dconv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_1")(dbn_2_0)
    dbn_2_1 = tf.layers.batch_normalization(dconv_2_1,training=TRAIN_FLAG)

    up_2 = tf.keras.layers.UpSampling2D((2,2))(dbn_2_1)
    dconv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_3_0")(up_2)
    dbn_3_0 = tf.layers.batch_normalization(dconv_3_0,training=TRAIN_FLAG)

    dconv_3_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_3_1")(dbn_3_0)
    dbn_3_1 = tf.layers.batch_normalization(dconv_3_1,training=TRAIN_FLAG)

    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_3_2")(dbn_3_1)
    return q_x_norm,q_x,output

def deepcoder_bn_encoder(in_gt,TRAIN_FLAG=True):
    conv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_0_0")(in_gt)
    bn_0_0 = tf.layers.batch_normalization(conv_0_0,training=TRAIN_FLAG)

    conv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_0_1")(bn_0_0)
    bn_0_1 = tf.layers.batch_normalization(conv_0_1,training=TRAIN_FLAG)

    pool_0 = tf.keras.layers.AveragePooling2D((2,2))(bn_0_1)

    conv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_1_0")(pool_0)
    bn_1_0 = tf.layers.batch_normalization(conv_1_0,training=TRAIN_FLAG)

    conv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_1_1")(bn_1_0)
    bn_1_1 = tf.layers.batch_normalization(conv_1_1,training=TRAIN_FLAG)

    pool_1 = tf.keras.layers.AveragePooling2D((2,2))(bn_1_1)

    conv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_2_0")(pool_1)
    bn_2_0 = tf.layers.batch_normalization(conv_2_0,training=TRAIN_FLAG)

    conv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_2_1")(bn_2_0)
    bn_2_1 = tf.layers.batch_normalization(conv_2_1,training=TRAIN_FLAG)

    pool_2 = tf.keras.layers.AveragePooling2D((2,2))(bn_2_1)

    conv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "Conv2D_3_0")(pool_2)
    bn_3_0 = tf.layers.batch_normalization(conv_3_0,training=TRAIN_FLAG)

    conv_3_1 = tf.keras.layers.Conv2D(4, (3, 3), padding='same',activation='sigmoid',name = "Conv2D_3_1")(bn_3_0)

    q_x_norm , q_x = quantizer_norm(conv_3_1)
    return q_x

def deepcoder_bn_decoder_name(q_x,TRAIN_FLAG=True):
    dconv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_0")(q_x)
    dbn_0_0 = tf.layers.batch_normalization(dconv_0_0,training=TRAIN_FLAG, name="dbn_0_0")

    dconv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_1")(dbn_0_0)
    dbn_0_1 = tf.layers.batch_normalization(dconv_0_1,training=TRAIN_FLAG, name="dbn_0_1")

    up_0 = tf.keras.layers.UpSampling2D((2,2))(dbn_0_1)

    dconv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_0")(up_0)
    dbn_1_0 = tf.layers.batch_normalization(dconv_1_0,training=TRAIN_FLAG, name="dbn_1_0")

    dconv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_1")(dbn_1_0)
    dbn_1_1 = tf.layers.batch_normalization(dconv_1_1,training=TRAIN_FLAG, name="dbn_1_1")

    up_1 = tf.keras.layers.UpSampling2D((2,2))(dbn_1_1)

    dconv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_0")(up_1)
    dbn_2_0 = tf.layers.batch_normalization(dconv_2_0,training=TRAIN_FLAG, name="dbn_2_0")

    dconv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_1")(dbn_2_0)
    dbn_2_1 = tf.layers.batch_normalization(dconv_2_1,training=TRAIN_FLAG, name="dbn_2_1")

    up_2 = tf.keras.layers.UpSampling2D((2,2))(dbn_2_1)
    dconv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_3_0")(up_2)
    dbn_3_0 = tf.layers.batch_normalization(dconv_3_0,training=TRAIN_FLAG, name="dbn_3_0")

    dconv_3_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_3_1")(dbn_3_0)
    dbn_3_1 = tf.layers.batch_normalization(dconv_3_1,training=TRAIN_FLAG, name="dbn_3_1")

    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_3_2")(dbn_3_1)
    return output

def deepcoder_bn_decoder(q_x,TRAIN_FLAG=True):
    dconv_0_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_0")(q_x)
    dbn_0_0 = tf.layers.batch_normalization(dconv_0_0,training=TRAIN_FLAG)

    dconv_0_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_0_1")(dbn_0_0)
    dbn_0_1 = tf.layers.batch_normalization(dconv_0_1,training=TRAIN_FLAG)

    up_0 = tf.keras.layers.UpSampling2D((2,2))(dbn_0_1)

    dconv_1_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_0")(up_0)
    dbn_1_0 = tf.layers.batch_normalization(dconv_1_0,training=TRAIN_FLAG)

    dconv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_1_1")(dbn_1_0)
    dbn_1_1 = tf.layers.batch_normalization(dconv_1_1,training=TRAIN_FLAG)

    up_1 = tf.keras.layers.UpSampling2D((2,2))(dbn_1_1)

    dconv_2_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_0")(up_1)
    dbn_2_0 = tf.layers.batch_normalization(dconv_2_0,training=TRAIN_FLAG)

    dconv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_2_1")(dbn_2_0)
    dbn_2_1 = tf.layers.batch_normalization(dconv_2_1,training=TRAIN_FLAG)

    up_2 = tf.keras.layers.UpSampling2D((2,2))(dbn_2_1)
    dconv_3_0 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name = "dConv2D_3_0")(up_2)
    dbn_3_0 = tf.layers.batch_normalization(dconv_3_0,training=TRAIN_FLAG)

    dconv_3_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu',name= "dConv2D_3_1")(dbn_3_0)
    dbn_3_1 = tf.layers.batch_normalization(dconv_3_1,training=TRAIN_FLAG)

    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same',name = "dConv2D_3_2")(dbn_3_1)
    return output
