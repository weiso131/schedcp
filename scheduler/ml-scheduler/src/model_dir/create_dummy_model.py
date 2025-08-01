#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
import numpy as np
import shutil

# Remove old model directory if exists
if os.path.exists('model_path'):
    shutil.rmtree('model_path')

# Create a simple graph
tf.disable_eager_execution()
tf.reset_default_graph()

# Define placeholders and operations
input_placeholder = tf.placeholder(tf.float32, shape=[None, 10], name='serving_default_input')
W1 = tf.Variable(tf.random.normal([10, 64]), name='W1')
b1 = tf.Variable(tf.zeros([64]), name='b1')
hidden1 = tf.nn.relu(tf.matmul(input_placeholder, W1) + b1)

W2 = tf.Variable(tf.random.normal([64, 64]), name='W2')
b2 = tf.Variable(tf.zeros([64]), name='b2')
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

W3 = tf.Variable(tf.random.normal([64, 1]), name='W3')
b3 = tf.Variable(tf.zeros([1]), name='b3')
output = tf.nn.sigmoid(tf.matmul(hidden2, W3) + b3, name='StatefulPartitionedCall')

# Create session and initialize variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Save the model
    builder = tf.saved_model.builder.SavedModelBuilder('model_path')
    
    # Create signature
    inputs = {'input': tf.saved_model.utils.build_tensor_info(input_placeholder)}
    outputs = {'output': tf.saved_model.utils.build_tensor_info(output)}
    
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        }
    )
    
    builder.save()
    
print("Dummy model saved successfully to model_path/")