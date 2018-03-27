from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

from nets.base_memory import BaseMemory
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

from model.config import cfg
from utils.snippets import update_weights
from utils.visualization import draw_predicted_boxes, draw_predicted_boxes_attend, draw_weights

class AttendMemory(BaseMemory):
  def __init__(self):
    BaseMemory.__init__(self)
    self._predictions["confid"] = []
    self._predictions["weights"] = []
    # self._attends = []
    self._attend_means = []

  def _update_weights(self, labels, cls_prob, iter):
    weights = tf.py_func(update_weights,
                        [labels, cls_prob],
                        tf.float32, 
                        name="weights_%02d" % iter)

    weights.set_shape([None])
    self._predictions["weights"].append(weights)

    # It is the label for the next iteration
    self._score_summaries[iter+1].append(weights)

  def _add_attend_mean_summary(self, iter):
    image = tf.py_func(draw_weights,
                      [self._attend_means[iter]],
                      tf.uint8,
                      name="attend")
    return tf.summary.image('MEAN-%02d' % iter, image)

  def _add_pred_memory_summary(self, iter):
    # also visualize the predictions of the network
    if self._gt_image is None:
      self._add_gt_image()
    if iter == 0 or cfg.MEM.LOSS != 'weight':
      image = tf.py_func(draw_predicted_boxes_attend,
                         [self._gt_image,
                         self._predictions['cls_prob'][iter],
                         self._gt_boxes,
                         self._predictions['confid_prob'][iter]],
                         tf.float32, name="pred_boxes")
    else:
      image = tf.py_func(draw_predicted_boxes_attend,
                         [self._gt_image,
                         self._predictions['cls_prob'][iter],
                         self._gt_boxes,
                         self._predictions['confid_prob'][iter],
                         self._predictions["weights"][iter-1]],
                         tf.float32, name="pred_boxes")
    return tf.summary.image('PRED-%02d' % iter, image)

  def _add_pred_attend_summary(self):
    # also visualize the predictions of the network
    if self._gt_image is None:
      self._add_gt_image()
    image = tf.py_func(draw_predicted_boxes,
                       [self._gt_image,
                       self._predictions['attend_cls_prob'],
                       self._gt_boxes],
                       tf.float32, name="attend_boxes")
    return tf.summary.image('PRED-attend', image)

  def _confidence_init(self, fc7, is_training):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    if cfg.MEM.SAT:
      fc7 = tf.stop_gradient(fc7,name="sat_init")
    confid = slim.fully_connected(fc7, 1, 
                                 weights_initializer=initializer,
                                 trainable=is_training,
                                 activation_fn=None, 
                                 biases_initializer=None,
                                 scope='confid')
    self._base_confidence = confid
    self._score_summaries[0].append(confid)

  def _confidence_iter(self, mem_fc7, is_training, name, iter):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.C_STD)
    with tf.variable_scope(name):
      if cfg.MEM.SAT:
        mem_fc7 = tf.stop_gradient(mem_fc7,name="sat_iter%02d"%iter)
      if cfg.MEM.CC == 'mem':
        confid_mem = slim.fully_connected(mem_fc7, 1, 
                                          weights_initializer=initializer,
                                          trainable=is_training,
                                          activation_fn=None,
                                          biases_initializer=None,
                                          scope="confid_mem")
        self._predictions['confid'].append(confid_mem)
        self._score_summaries[iter].append(confid_mem)
      elif cfg.MEM.CC == 'no':
        confid_mem = slim.fully_connected(mem_fc7, 1, 
                                          weights_initializer=initializer,
                                          trainable=is_training,
                                          activation_fn=None,
                                          biases_initializer=tf.constant_initializer(0.0),
                                          scope="confid_mem")
        if iter == 0:
          self._predictions['confid'].append(self._base_confidence)
          self._score_summaries[iter].append(self._base_confidence)
        else:
          self._predictions['confid'].append(confid_mem)
          self._score_summaries[iter].append(confid_mem)
      else:
        raise NotImplementedError

  def _aggregate_pred(self, name):
    with tf.variable_scope(name):
      comb_confid = tf.stack(self._predictions['confid'], axis=2, name='comb_confid')
      comb_attend = tf.nn.softmax(comb_confid, dim=2, name='comb_attend')
      self._predictions['confid_prob'] = tf.unstack(comb_attend, axis=2, name='unstack_attend')
      if cfg.MEM.LOSS == 'avg':
        comb_score = tf.stack(self._predictions["cls_score"], axis=2, name='comb_score')
      elif cfg.MEM.LOSS == 'sep' or cfg.MEM.LOSS == 'weight':
        comb_score = tf.stop_gradient(tf.stack(self._predictions["cls_score"], axis=2, name='comb_score'),
                                      name="comb_score_nb")
      else:
        raise NotImplementedError
      cls_score = tf.reduce_sum(tf.multiply(comb_score, comb_attend, name='weighted_cls_score'), 
                                axis=2, name='attend_cls_score')
      cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
      cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

      self._predictions["attend_cls_score"] = cls_score
      self._predictions["attend_cls_prob"] = cls_prob
      self._predictions["attend_cls_pred"] = cls_pred
      self._predictions['attend'] = self._predictions['confid_prob']

    return cls_prob

  def _build_conv(self, is_training):
    # Get the head
    net_conv = self._image_to_head(is_training)
    with tf.variable_scope(self._scope, self._scope):
      # get the region of interest
      rois, batch_ids, inv_rois, inv_batch_ids = self._target_memory_layer("target")
      # region of interest pooling
      pool5 = self._crop_rois(net_conv, rois, batch_ids, "pool5")
      pool5_nb = tf.stop_gradient(pool5, name="pool5_nb")

    # initialize the normalization vector, note here it is the batch ids
    count_matrix_raw, self._count_crops = self._inv_crops(self._count_base, inv_rois, batch_ids, "count_matrix")
    self._count_matrix = tf.stop_gradient(count_matrix_raw, name='cm_nb')
    self._count_matrix_eps = tf.maximum(self._count_matrix, cfg.EPS, name='count_eps')
    self._score_summaries[0].append(self._count_matrix)

    fc7 = self._head_to_tail(pool5, is_training)
    # First iteration
    with tf.variable_scope(self._scope, self._scope):
      # region classification
      cls_score_conv, cls_prob_conv = self._cls_init(fc7, is_training)
      if cfg.MEM.CC == 'no':
        self._confidence_init(fc7, is_training)

    return cls_score_conv, pool5_nb, \
           rois, batch_ids, inv_rois, inv_batch_ids

  def _build_pred(self, is_training, mem, cls_score_conv, rois, batch_ids, iter):
    if cfg.MEM.CT_L:
      mem_net = self._context(mem, is_training, "context", iter)
    else:
      mem_net = mem
    mem_ct_pool5 = self._crop_rois(mem_net, rois, batch_ids, "mem_ct_pool5")
    mem_fc7 = self._fc_iter(mem_ct_pool5, is_training, "fc7", iter) 
    cls_score_mem = self._cls_iter(mem_fc7, is_training, "cls_iter", iter)
    self._confidence_iter(mem_fc7, is_training, "confid_iter", iter)
    cls_score, cls_prob, cls_pred = self._comb_conv_mem(cls_score_conv, cls_score_mem, 
                                                        "comb_conv_mem", iter)

    return cls_score, cls_prob, cls_pred

  def _build_memory(self, is_training, is_testing):
    # initialize memory
    mem = self._mem_init(is_training, "mem_init")
    # convolution related stuff
    cls_score_conv, pool5_nb, \
    rois, batch_ids, inv_rois, inv_batch_ids = self._build_conv(is_training)
    # Separate first prediction
    reuse = None
    # Memory iterations
    self._labels = tf.reshape(self._targets["labels"], [-1])
    for iter in xrange(cfg.MEM.ITER):
      print('ITERATION: %02d' % iter)
      self._mems.append(mem)
      with tf.variable_scope('SMN', reuse=reuse):
        # Use memory to predict the output
        cls_score, cls_prob, cls_pred = self._build_pred(is_training, 
                                                         mem, 
                                                         cls_score_conv, 
                                                         rois, batch_ids, iter)

        if iter == cfg.MEM.ITER - 1:
          break

        # Update the memory with all the regions
        mem = self._build_update(is_training, mem, 
                                pool5_nb, cls_score, cls_prob, cls_pred,
                                rois, batch_ids, inv_rois, inv_batch_ids, 
                                iter)

        if is_training and cfg.MEM.LOSS == 'weight':
          self._update_weights(self._labels, cls_prob, iter)

      if iter == 0:
        reuse = True

    # Need to finalize the class scores, regardless of whether loss is computed
    cls_prob = self._aggregate_pred("aggregate")

    return rois, cls_prob

  def _add_memory_losses(self, name):
    cross_entropy = []
    assert len(self._predictions["cls_score"]) == cfg.MEM.ITER
    if cfg.MEM.LOSS == 'weight':
      assert len(self._predictions["weights"]) == cfg.MEM.ITER - 1
    with tf.variable_scope(name):
      # Then add ones for later iterations
      for iter in xrange(cfg.MEM.ITER):
        # RCNN, class loss
        cls_score = self._predictions["cls_score"][iter]
        ce_ins = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, 
                                                                labels=self._labels,
                                                                name="ce_ins_%02d" % iter)
        if iter > 0 and cfg.MEM.LOSS == 'weight':
          weight = self._predictions["weights"][iter-1]
          ce = tf.reduce_sum(tf.multiply(weight, ce_ins, name="weight_%02d" % iter), name="ce_%02d" % iter)
        else:
          ce = tf.reduce_mean(ce_ins, name="ce_%02d" % iter)
        cross_entropy.append(ce)

      # The final most important score
      cls_score = self._predictions["attend_cls_score"]
      ce_final = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, 
                                                                               labels=self._labels,
                                                                               name="ce_ins"), name="ce")
      self._losses['cross_entropy_final'] = ce_final
 
      if cfg.MEM.LOSS == 'avg':
        # Just use a single one out
        self._losses['cross_entropy'] = self._losses['cross_entropy_final']
      elif cfg.MEM.LOSS == 'sep' or cfg.MEM.LOSS == 'weight':
        ce_rest = tf.stack(cross_entropy[1:], name="cross_entropy_rest")
        self._losses['cross_entropy_image'] = cross_entropy[0]
        self._losses['cross_entropy_memory'] = tf.reduce_mean(ce_rest, name='cross_entropy')
        self._losses['cross_entropy'] = self._losses['cross_entropy_image'] \
                                      + cfg.MEM.W * self._losses['cross_entropy_memory'] \
                                      + cfg.MEM.WF * self._losses['cross_entropy_final']
      else:
        raise NotImplementedError

      loss = self._losses['cross_entropy']
      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      self._losses['total_loss'] = loss + regularization_loss

      self._event_summaries.update(self._losses)

    return loss

  def _add_attend(self, iter):
    attend_reshape = tf.reshape(self._predictions['attend'][iter], [-1, 1, 1, 1], name="reshape_attend_%d" % iter)
    attend = tf.multiply(self._count_crops, attend_reshape, name="attend_%d" % iter)
    sum_attend = tf.reduce_sum(attend, axis=0, keep_dims=True, name="sum_attend_%d" % iter)
    # self._attends.append(sum_attend)
    self._attend_means.append(tf.div(sum_attend, self._count_matrix_eps, name="mean_attend_%d" % iter))

  def _create_summary(self):
    val_summaries = []
    with tf.device("/cpu:0"):
      val_summaries.append(self._add_gt_image_summary())
      val_summaries.append(self._add_pred_attend_summary())
      for iter in xrange(cfg.MEM.ITER):
        self._add_attend(iter)
      for iter in xrange(cfg.MEM.ITER):
        val_summaries.append(self._add_pred_memory_summary(iter))
        val_summaries.append(self._add_memory_summary(iter))
        # val_summaries.append(self._add_attend_summary(iter))
        val_summaries.append(self._add_attend_mean_summary(iter))
        for var in self._score_summaries[iter]:
          self._add_score_iter_summary(iter, var)
      for key, var in self._event_summaries.items():
        val_summaries.append(tf.summary.scalar(key, var))
      for var in self._act_summaries:
        self._add_zero_summary(var)

    self._summary_op = tf.summary.merge_all()
    self._summary_op_val = tf.summary.merge(val_summaries)

  # take the last predicted output
  def test_image(self, sess, blobs):
    cls_score, cls_prob = sess.run([self._predictions["attend_cls_score"],
                                    self._predictions['attend_cls_prob']],
                                   feed_dict=self._parse_dict(blobs))
    return cls_score, cls_prob

class vgg16_memory(AttendMemory, vgg16):
  def __init__(self):
    AttendMemory.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'vgg_16'

class resnetv1_memory(AttendMemory, resnetv1):
  def __init__(self, num_layers=50):
    AttendMemory.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._num_layers = num_layers
    self._scope = 'resnet_v1_%d' % num_layers
    resnetv1._decide_blocks(self)

class mobilenetv1_memory(AttendMemory, mobilenetv1):
  def __init__(self):
    AttendMemory.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
    self._scope = 'MobilenetV1'
