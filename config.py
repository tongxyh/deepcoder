import tensorflow as tf

# system setting
tf.app.flags.DEFINE_string('gpu', '1' ,'the gpu to be used.')

tf.app.flags.DEFINE_string('train_dir', 'xxx', 'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string('modelpath','models/rd_v2/0/','Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string('tboardpath','log/rd_v2/train-0', 'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string('train_dataset','../../dataset/train64.h5','Directory of dataset to be used for training.')

tf.app.flags.DEFINE_string('test_dataset','../../dataset/test64.h5','Directory of dataset to be used for test.')

tf.app.flags.DEFINE_bool('restore', False, 'restore flag.')

tf.app.flags.DEFINE_bool('model_to_restore', None, 'path of model to restore.')

tf.app.flags.DEFINE_float('lr', 0.0001,'learning rate.')

#
tf.app.flags.DEFINE_bool('Stream_to_H5',True,'save encoded fmaps to h5')

if __name__ == "__main__":
    tf.app.run()
