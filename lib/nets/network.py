from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

from model.config import cfg
from utils.snippets import compute_target
from utils.visualization import draw_gt_boxes, draw_predicted_boxes

class Network(object):
  def __init__(self):
    self._predictions = {}
    self._losses = {}
    self._targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = []
    self._score_summaries = {}
    self._event_summaries = {}
    self._variables_to_fix = {}

  def _add_gt_image(self):
    # add back mean
    image = self._image + cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    self._gt_image = tf.reverse(image, axis=[-1])

  def _add_org_image_summary(self):
    if self._gt_image is None:
      self._add_gt_image()
    
    return tf.summary.image('ORG', self._gt_image)

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    if self._gt_image is None:
      self._add_gt_image()
    image = tf.py_func(draw_gt_boxes, 
                      [self._gt_image, self._gt_boxes],
                      tf.float32, name="gt_boxes")
    
    return tf.summary.image('GROUND_TRUTH', image)

  def _add_pred_summary(self):
    # also visualize the predictions of the network
    if self._gt_image is None:
      self._add_gt_image()
    image = tf.py_func(draw_predicted_boxes,
                       [self._gt_image,
                       self._predictions['cls_prob'],
                       self._gt_boxes],
                       tf.float32, name="pred_boxes")
    return tf.summary.image('PRED', image)

  def _add_zero_summary(self, tensor):
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _crop_pool_layer(self, bottom, rois, batch_ids, name):
    with tf.variable_scope(name):
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, 
                                       rois, 
                                       batch_ids, 
                                       [pre_pool_size, pre_pool_size], 
                                       name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _target_layer(self, name):
    with tf.variable_scope(name):
      rois, batch_ids, roi_overlaps, labels = tf.py_func(compute_target,
                                                        [self._memory_size, self._gt_boxes, self._feat_stride[0]],
                                                        [tf.float32, tf.int32, tf.float32, tf.int32],
                                                        name="target_layer")

      rois.set_shape([None, 4])
      labels.set_shape([None, 1])

      self._targets['rois'] = rois
      self._targets['roi_overlaps'] = roi_overlaps
      self._targets['batch_ids'] = batch_ids
      self._targets['labels'] = labels

      self._score_summaries.update(self._targets)

      return rois, batch_ids

  def _region_classification(self, fc7, is_training, initializer):
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, 
                                       scope='cls_score')
    cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob

    return cls_prob

  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError

  def _build_network(self, is_training=True):
    # select initializers
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

    net_conv = self._image_to_head(is_training)
    with tf.variable_scope(self._scope, self._scope):
      # get the region of interest
      rois, batch_ids = self._target_layer("target")
      # region of interest pooling
      pool5 = self._crop_pool_layer(net_conv, rois, batch_ids, "pool5")

    fc7 = self._head_to_tail(pool5, is_training)
    with tf.variable_scope(self._scope, self._scope):
      # region classification
      cls_prob = self._region_classification(fc7, is_training, initializer)

    self._score_summaries.update(self._predictions)

    return rois, cls_prob

  def _add_losses(self):
    with tf.variable_scope('loss') as scope:
      # RCNN, class loss
      cls_score = self._predictions["cls_score"]
      label = tf.reshape(self._targets["labels"], [-1])
      cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

      self._losses['cross_entropy'] = cross_entropy
      loss = cross_entropy
      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      self._losses['total_loss'] = loss + regularization_loss

      self._event_summaries.update(self._losses)

    return loss

  def create_architecture(self, mode, num_classes, tag=None):
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._memory_size = tf.placeholder(tf.int32, shape=[2])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._num_gt = tf.placeholder(tf.int32, shape=[])
    self._tag = tag

    self._num_classes = num_classes
    self._mode = mode

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag is not None

    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)): 
      rois, cls_prob = self._build_network(training)

    layers_to_output = {'rois': rois}

    if not testing:
      self._add_losses()
      layers_to_output.update(self._losses)
      val_summaries = []
      with tf.device("/cpu:0"):
        val_summaries.append(self._add_gt_image_summary())
        val_summaries.append(self._add_pred_summary())
        for key, var in self._event_summaries.items():
          val_summaries.append(tf.summary.scalar(key, var))
        for key, var in self._score_summaries.items():
          self._add_score_summary(key, var)
        for var in self._act_summaries:
          self._add_act_summary(var)

      self._summary_op = tf.summary.merge_all()
      self._summary_op_val = tf.summary.merge(val_summaries)

    layers_to_output.update(self._predictions)

    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat

  def _parse_dict(self, blobs):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'], 
                 self._memory_size: blobs['memory_size'],
                 self._num_gt: blobs['num_gt']}
    return feed_dict

  # only useful during testing mode
  def test_image(self, sess, blobs):
    cls_score, cls_prob = sess.run([self._predictions["cls_score"],
                                    self._predictions['cls_prob']],
                                   feed_dict=self._parse_dict(blobs))
    return cls_score, cls_prob

  def get_summary(self, sess, blobs):
    summary = sess.run(self._summary_op_val, feed_dict=self._parse_dict(blobs))
    return summary

  def train_step(self, sess, blobs, train_op):
    loss_cls, loss, _ = sess.run([self._losses['cross_entropy'],
                                  self._losses['total_loss'],
                                  train_op],
                                 feed_dict=self._parse_dict(blobs))
    return loss_cls, loss

  def train_step_with_summary(self, sess, blobs, train_op, summary_grads):
    loss_cls, loss, summary, gsummary, _ = sess.run([self._losses['cross_entropy'],
                                           self._losses['total_loss'],
                                           self._summary_op,
                                           summary_grads,
                                           train_op],
                                          feed_dict=self._parse_dict(blobs))
    return loss_cls, loss, summary, gsummary

  def train_step_no_return(self, sess, blobs, train_op):
    sess.run([train_op], feed_dict=self._parse_dict(blobs))

  # Empty function here, just in case
  def fixed_parameters(self, sess):
    return

