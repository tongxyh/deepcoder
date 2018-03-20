# Save a trained model to pb model file
# 2018.3.20 written by Tong Chen

import tensorflow as tf
import model
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# Add TF FLAG
tf.app.flags.DEFINE_string("modelpath",None,"Path of Model that is to be converted.")
tf.app.flags.DEFINE_string("modelname",None,"New Model Name.")

FLAGS = tf.app.flags.FLAGS

in_gt = tf.placeholder(tf.float32, shape=[1, None, None, 3],name='Input_Img')
q_x = model.deepcoder_bn_encoder(in_gt)

de_in = tf.placeholder(tf.float32, shape=[1, None, None, 4],name='Input_Decoder')
out = model.deepcoder_bn_decoder(de_in)

def main():
    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.modelpath)

        #save encoder
        graph = convert_variables_to_constants(sess, sess.graph_def,["div"])
        tf.train.write_graph(graph, '.', FLAGS.modelname+'-encoder.pb', as_text=False)

        #save decoder
        graph = convert_variables_to_constants(sess, sess.graph_def,["dConv2D_3_2/BiasAdd"])
        tf.train.write_graph(graph, '.', FLAGS.modelname+'-decoder.pb', as_text=False)

        #from tensorflow.python.tools import inspect_checkpoint as chkp
        #chkp.print_tensors_in_checkpoint_file(FLAGS.modelpath,tensor_name='',all_tensors=False) #set False to only print tensor name and shape

        #for v in sess.graph.get_operations():
        #    print(v.name)

        #tf.train.write_graph(tf.get_default_graph().as_graph_def(), "./", 'deepc.pb', as_text=False)
        #tf.import_graph_def()
        #g = tf.contrib.quantize.create_eval_graph(tf.get_default_graph())
        #draw model graph in tensorboard to check

if __name__ == "__main__":
    #tf.app.run()
    main()

# TODO: test only-quantize_weights
# find out what's happening in fold_batch_norms
