from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils as tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops

from .bilinear_upsample_weights import bilinear_upsample_weights
from . import TensorflowUtils as utils



MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
MODEL_DIR = 'data/vgg_model/'

def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with arg_scope(
      [layers.conv2d],
      activation_fn=None,
      weights_initializer=tf.contrib.layers.xavier_initializer(),
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      biases_initializer=init_ops.zeros_initializer(),
      padding='SAME') as arg_sc:
    return arg_sc

def vgg_mfcn(inputs, is_training, weight_decay=5e-4, dropout_keep_prob=0.5, first_no_subsample=6, no_pool=False, num_classes=2, init=True):
    model_data = utils.get_model_data(MODEL_DIR, MODEL_URL)
    mean_pixel = np.mean(model_data['normalization'][0][0][0], axis=(0, 1))
    weights = np.squeeze(model_data['layers'])

    layers_name = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5',
    )

    inputs_shape = tf.shape(inputs)
    processed_image = tf.subtract(inputs, mean_pixel)
    net = processed_image

    end_points = collections.OrderedDict()

    for i, name in enumerate(layers_name):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = kernels.transpose(1,0,2,3)
            bias = bias.reshape(-1)
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            if init:
                print("Setting up vgg initialized conv layer: {}".format(name))
                kernels_init = tf.constant_initializer(kernels, verify_shape=True)
                bias_init = tf.constant_initializer(bias, verify_shape=True)
            else:
                kernels = tf.contrib.layers.xavier_initializer()
                bias = init_ops.zeros_initializer()
            idx = int(name[4])
            dilation_rate = 1 if idx < first_no_subsample+1 else 2**(idx-first_no_subsample)
            net = layers.conv2d(net, kernels.shape[-1], kernels.shape[0:2], weights_initializer=kernels_init, biases_initializer=bias_init, rate=dilation_rate, scope=name)
            net = layers_lib.batch_norm(net, is_training=is_training, activation_fn=nn_ops.relu, scope=name)
        elif kind == 'relu':
            continue
        elif kind == 'pool':
            idx = int(name[4])
            if idx < first_no_subsample:
                net = layers_lib.max_pool2d(net, kernel_size=2, stride=2, padding="SAME", scope=name)
            else:
                if not no_pool:
                    net = layers_lib.max_pool2d(net, kernel_size=2, stride=1, padding="SAME", scope=name)
        end_points[name] = net 
    
    net = layers.conv2d(net, 4096, [7, 7], activation_fn=nn_ops.relu, scope='conv6')
    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

    net = layers.conv2d(net, 4096, [1, 1], activation_fn=nn_ops.relu, scope='conv7')
    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

    net = layers.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='conv8')

    def skip_connection(input):
        # now to upscale to actual image size
        with tf.variable_scope('32x_to_16x'):
            deconv_shape1 = tf.shape(end_points["pool4"])
            conv_t11 = tf.nn.conv2d_transpose(input, \
                tf.get_variable('bilinear_kernel1', dtype=tf.float32, shape=[4,4,2,2], \
                    initializer=tf.constant_initializer(bilinear_upsample_weights(2,num_classes,num_classes), verify_shape=True), \
                    regularizer=regularizers.l2_regularizer(weight_decay)), \
                [deconv_shape1[0],deconv_shape1[1],deconv_shape1[2],num_classes], strides=[1, 2, 2, 1], padding="SAME")
            conv_t12 = layers.conv2d(end_points["pool4"], num_classes, [1, 1], activation_fn=None, scope='conv_skip1')
            fuse_1 = tf.add(conv_t11, conv_t12, name="fuse_1")

        with tf.variable_scope('16x_to_8x'):
            deconv_shape2 = tf.shape(end_points["pool3"])
            conv_t21 = tf.nn.conv2d_transpose(fuse_1, \
                tf.get_variable('bilinear_kernel2', dtype=tf.float32, shape=[4,4,2,2], \
                    initializer=tf.constant_initializer(bilinear_upsample_weights(2,num_classes,num_classes), verify_shape=True), \
                    regularizer=regularizers.l2_regularizer(weight_decay)), \
                [deconv_shape2[0],deconv_shape2[1],deconv_shape2[2],num_classes], strides=[1, 2, 2, 1], padding="SAME")
            conv_t22 = layers.conv2d(end_points["pool3"], num_classes, [1, 1], activation_fn=None, scope='conv_skip2')
            fuse_2 = tf.add(conv_t21, conv_t22, name="fuse_2")

        with tf.variable_scope('8x_to_1x'):
            conv_t3 = tf.nn.conv2d_transpose(fuse_2, \
                tf.get_variable('bilinear_kernel3', dtype=tf.float32, shape=[16,16,2,2], \
                    initializer=tf.constant_initializer(bilinear_upsample_weights(8,num_classes,num_classes), verify_shape=True), \
                    regularizer=regularizers.l2_regularizer(weight_decay)), \
                [inputs_shape[0], inputs_shape[1], inputs_shape[2], num_classes], strides=[1, 8, 8, 1], padding="SAME")

        return conv_t3

    with tf.variable_scope('mask_pred'):
        logits_msk = skip_connection(net)
        preds_msk = tf.cast(tf.argmax(logits_msk,3),tf.int32)
        preds_msk_map = tf.nn.softmax(logits_msk)[:,:,:,1]

    with tf.variable_scope('edge_pred'):
        logits_edg = skip_connection(net)
        preds_edg = tf.cast(tf.argmax(logits_edg,3),tf.int32)
        preds_edg_map = tf.nn.softmax(logits_edg)[:,:,:,1]

    return logits_msk, logits_edg, preds_msk, preds_edg, preds_msk_map, preds_edg_map



