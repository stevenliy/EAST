import os, sys
import tensorflow as tf
import numpy
import copy
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import dtypes
import model
if len(sys.argv) != 3:
    print ("")
    print("Usage: python3 " + sys.argv[0] + " <meta file> <ckpt file>")
    print ("")
    sys.exit(1)

meta_file = sys.argv[1]
ckpt_file = sys.argv[2]

# meta_saver = tf.train.import_meta_graph(meta_file)
# ckpt_reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
# graph = tf.get_default_graph()
# summary_writer = tf.summary.FileWriter('logs', graph)

# variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
with tf.get_default_graph().as_default():
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess,ckpt_file)
        tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
        graph = tf.get_default_graph()
        summary_writer = tf.summary.FileWriter('logs', graph)
    
    # freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, model_path, 'out','save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")

    print("done")


