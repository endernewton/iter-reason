from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

from nets.network import Network
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

from model.config import cfg
from utils.snippets import compute_target_memory
from utils.visualization import draw_predicted_boxes, draw_memory

class BaseMemory(Network):
  def __init__(self):
    self._predictions = {}
    self._predictions["cls_score"] = []
    self._predictions["cls_prob"] = []
    self._predictions["cls_pred"] = []

    self._losses = {}
    self._targets = {}
    self._layers = {}
    self._gt_image = None
    self._mems = []
    self._act_summaries = []
    self._score_summaries = [[] for _ in xrange(cfg.MEM.ITER)]
    self._event_summaries = {}
    self._variables_to_fix = {}

  def _add_score_iter_summary(self, iter, tensor):
    tf.summary.histogram('SCORE-%02d/' % iter + tensor.op.name, tensor)

  def _add_memory_summary(self, iter):
    image = tf.py_func(draw_memory,
                      [self._mems[iter]],
                      tf.float32,
                      name="memory")
    return tf.summary.image('MEM-%02d' % iter, image)

  def _add_pred_memory_summary(self, iter):
    # also visualize the predictions of the network
    if self._gt_image is None:
      self._add_gt_image()
    image = tf.py_func(draw_predicted_boxes,
                       [self._gt_image,
                       self._predictions['cls_prob'][iter],
                       self._gt_boxes],
                       tf.float32, name="pred_boxes")
    return tf.summary.image('PRED-%02d' % iter, image)

  def _target_memory_layer(self, name):
    with tf.variable_scope(name):
      rois, batch_ids, labels, inv_rois, inv_batch_ids = tf.py_func(compute_target_memory,
                                                          [self._memory_size, self._gt_boxes, self._feat_stride[0]],
                                                          [tf.float32, tf.int32, tf.int32, tf.float32, tf.int32],
                                                          name="target_memory_layer")

      rois.set_shape([None, 4])
      labels.set_shape([None, 1])
      inv_rois.set_shape([None, 4])

      self._targets['rois'] = rois
      self._targets['batch_ids'] = batch_ids
      self._targets['labels'] = labels
      self._targets['inv_rois'] = inv_rois
      self._targets['inv_batch_ids'] = inv_batch_ids

      self._score_summaries[0].append(rois)
      self._score_summaries[0].append(labels)
      self._score_summaries[0].append(inv_rois)

    return rois, batch_ids, inv_rois, inv_batch_ids

  def _crop_rois(self, bottom, rois, batch_ids, name, iter=0):
    with tf.variable_scope(name):
      crops = tf.image.crop_and_resize(bottom, rois, batch_ids, 
                                       [cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE],
                                       name="crops")
      self._score_summaries[iter].append(crops)
    return crops

  def _inv_crops(self, pool5, inv_rois, inv_batch_ids, name):
    with tf.variable_scope(name):
      inv_crops = tf.image.crop_and_resize(pool5, inv_rois, inv_batch_ids, self._memory_size,
                                           extrapolation_value=0, # difference is 0 outside
                                           name="inv_crops")
      # Add things up (make sure it is relu)
      inv_crop = tf.reduce_sum(inv_crops, axis=0, keep_dims=True, name="reduce_sum")

    return inv_crop, inv_crops

  # The initial classes, only use output from the conv features
  def _cls_init(self, fc7, is_training):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

    self._score_summaries[0].append(cls_score)
    self._score_summaries[0].append(cls_pred)
    self._score_summaries[0].append(cls_prob)

    return cls_score, cls_prob

  def _mem_init(self, is_training, name):
    mem_initializer = tf.constant_initializer(0.0)
    # Kinda like bias
    if cfg.TRAIN.BIAS_DECAY:
      mem_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    else:
      mem_regularizer = tf.no_regularizer

    with tf.variable_scope('SMN'):
      with tf.variable_scope(name):
        mem_init = tf.get_variable('mem_init', 
                                   [1, cfg.MEM.INIT_H, cfg.MEM.INIT_W, cfg.MEM.C], 
                                   initializer=mem_initializer, 
                                   trainable=is_training,
                                   regularizer=mem_regularizer)
        self._score_summaries[0].append(mem_init)
        # resize it to the image-specific size
        mem_init = tf.image.resize_bilinear(mem_init, self._memory_size, 
                                            name="resize_init")

    return mem_init

  def _context_conv(self, net, conv, scope):
    net = slim.conv2d(net, cfg.MEM.C, [conv, conv], scope=scope)
    return net

  def _context(self, net, is_training, name, iter):
    num_layers = cfg.MEM.CT_L
    xavier = tf.contrib.layers.variance_scaling_initializer()

    assert num_layers % 2 == 1
    conv = cfg.MEM.CT_CONV
    with tf.variable_scope(name):
      with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
                          activation_fn=None, 
                          trainable=is_training,
                          weights_initializer=xavier,
                          biases_initializer=tf.constant_initializer(0.0)):
        net = self._context_conv(net, cfg.MEM.CT_FCONV, "conv1")
        for i in xrange(2, num_layers+1, 2):
          net1 = tf.nn.relu(net, name="relu%02d" % (i-1))
          self._act_summaries.append(net1)
          self._score_summaries[iter].append(net1)
          net1 = self._context_conv(net1, conv, "conv%02d" % i)
          net2 = tf.nn.relu(net1, name="relu%02d" % i)
          self._act_summaries.append(net2)
          self._score_summaries[iter].append(net2)
          net2 = self._context_conv(net2, conv, "conv%02d" % (i+1))
          net = tf.add(net, net2, "residual%02d" % i)

    return net

  def _fc_iter(self, mem_pool5, is_training, name, iter):
    xavier = tf.contrib.layers.variance_scaling_initializer()
    
    with tf.variable_scope(name):
      mem_fc7 = slim.flatten(mem_pool5, scope='flatten')
      with slim.arg_scope([slim.fully_connected], 
                          activation_fn=tf.nn.relu, 
                          trainable=is_training,
                          weights_initializer=xavier,
                          biases_initializer=tf.constant_initializer(0.0)):
        for i in xrange(cfg.MEM.FC_L):
          mem_fc7 = slim.fully_connected(mem_fc7, 
                                        cfg.MEM.FC_C, 
                                        scope="mem_fc%d" % (i+6))
          self._act_summaries.append(mem_fc7)
          self._score_summaries[iter].append(mem_fc7)

    return mem_fc7

  def _cls_iter(self, mem_fc7, is_training, name, iter):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.C_STD)
    with tf.variable_scope(name):
      cls_score_mem = slim.fully_connected(mem_fc7, self._num_classes, 
                                          weights_initializer=initializer,
                                          activation_fn=None,
                                          trainable=is_training,
                                          biases_initializer=tf.constant_initializer(0.0),
                                          scope="cls_score_mem")
      self._score_summaries[iter].append(cls_score_mem)

    return cls_score_mem

  def _comb_conv_mem(self, cls_score_conv, cls_score_mem, name, iter):
    with tf.variable_scope(name):
      # take the output directly from each iteration
      if iter == 0:
        cls_score = cls_score_conv
      else:
        cls_score = cls_score_mem
      cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
      cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

      self._predictions['cls_score'].append(cls_score)
      self._predictions['cls_pred'].append(cls_pred)
      self._predictions['cls_prob'].append(cls_prob)

      self._score_summaries[iter].append(cls_score)
      self._score_summaries[iter].append(cls_pred)
      self._score_summaries[iter].append(cls_prob)

    return cls_score, cls_prob, cls_pred

  def _bottomtop(self, pool5, cls_prob, is_training, name, iter):
    initializer=tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.STD)
    initializer_fc=tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.STD * cfg.MEM.FP_R)
    with tf.variable_scope(name):
      with slim.arg_scope([slim.fully_connected, slim.conv2d], 
                          activation_fn=None,
                          trainable=is_training):
        # just make the representation more dense
        map_prob = slim.fully_connected(cls_prob, 
                                        cfg.MEM.C,
                                        weights_initializer=initializer_fc,
                                        biases_initializer=None,
                                        scope="map_prob")
        map_comp = tf.reshape(map_prob, [-1, 1, 1, cfg.MEM.C], name="map_comp")
        pool5_comp = slim.conv2d(pool5, 
                                 cfg.MEM.C, 
                                 [1, 1], 
                                 weights_initializer=initializer,
                                 biases_initializer=tf.constant_initializer(0.0),
                                 scope="pool5_comp")

        pool5_comb = tf.add(map_comp, pool5_comp, name="addition")
        pool5_comb = tf.nn.relu(pool5_comb, name="pool5_comb")

        self._score_summaries[iter].append(map_prob)
        self._score_summaries[iter].append(pool5_comp)
        self._score_summaries[iter].append(pool5_comb)
        self._act_summaries.append(pool5_comb)

    return pool5_comb

  def _bottom(self, pool5, is_training, name, iter):
    initializer=tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.STD)
    with tf.variable_scope(name):
      with slim.arg_scope([slim.fully_connected, slim.conv2d], 
                          activation_fn=None,
                          trainable=is_training):
        # just make the representation more dense
        pool5_comp = slim.conv2d(pool5, 
                                 cfg.MEM.C, 
                                 [1, 1], 
                                 activation_fn=tf.nn.relu,
                                 weights_initializer=initializer,
                                 biases_initializer=tf.constant_initializer(0.0),
                                 scope="pool5_comp")
        self._score_summaries[iter].append(pool5_comp)
        self._act_summaries.append(pool5_comp)

    return pool5_comp

  def _topprob(self, cls_prob, is_training, name, iter):
    initializer_fc=tf.random_normal_initializer(mean=0.0, 
                                                stddev=cfg.MEM.STD * cfg.MEM.FP_R)
    with tf.variable_scope(name):
      # just make the representation more dense
      map_prob = slim.fully_connected(cls_prob, 
                                      cfg.MEM.C,
                                      activation_fn=tf.nn.relu,
                                      trainable=is_training,
                                      weights_initializer=initializer_fc,
                                      biases_initializer=tf.constant_initializer(0.0),
                                      scope="map_prob")
      map_comp = tf.reshape(map_prob, [-1, 1, 1, cfg.MEM.C], name="map_comp")
      map_pool = tf.tile(map_comp, [1, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1], name="map_pool")
      self._score_summaries[iter].append(map_prob)
      self._act_summaries.append(map_prob)

    return map_pool

  def _toppred(self, cls_pred, is_training, name, iter):
    initializer_fc=tf.random_normal_initializer(mean=0.0, 
                                                stddev=cfg.MEM.STD * cfg.MEM.FP_R)
    with tf.variable_scope(name):
      cls_pred_hot = tf.one_hot(cls_pred, self._num_classes, name="encode")
      # just make the representation more dense
      map_pred = slim.fully_connected(cls_pred_hot, 
                                      cfg.MEM.C,
                                      activation_fn=tf.nn.relu,
                                      trainable=is_training,
                                      weights_initializer=initializer_fc,
                                      biases_initializer=tf.constant_initializer(0.0),
                                      scope="map_pred")
      map_comp = tf.reshape(map_pred, [-1, 1, 1, cfg.MEM.C], name="map_comp")
      map_pool = tf.tile(map_comp, [1, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1], name="map_pool")
      self._score_summaries[iter].append(map_pred)
      self._act_summaries.append(map_pred)

    return map_pool

  def _input(self, net, is_training, name, iter):
    num_layers = cfg.MEM.IN_L
    in_conv = cfg.MEM.IN_CONV
    xavier = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name): 
      # the first part is already done
      for i in xrange(2, num_layers+1):
        net = slim.conv2d(net, cfg.MEM.C, [in_conv, in_conv], 
                          activation_fn=tf.nn.relu, 
                          trainable=is_training,
                          weights_initializer=xavier,
                          biases_initializer=tf.constant_initializer(0.0),
                          scope="conv%02d" % i)
        self._score_summaries[iter].append(net)
        self._act_summaries.append(net)

    return net

  def _mem_update(self, pool5_mem, pool5_input, is_training, name, iter):
    feat_initializer = tf.random_normal_initializer(mean=0.0, 
                          stddev=cfg.MEM.U_STD)
    mem_initializer = tf.random_normal_initializer(mean=0.0, 
                          stddev=cfg.MEM.U_STD * cfg.MEM.FM_R)
    feat_gate_initializer = tf.random_normal_initializer(mean=0.0, 
                          stddev=cfg.MEM.U_STD / cfg.MEM.VG_R)
    mem_gate_initializer = tf.random_normal_initializer(mean=0.0, 
                          stddev=cfg.MEM.U_STD * cfg.MEM.FM_R / cfg.MEM.VG_R)
    mconv = cfg.MEM.CONV
    with tf.variable_scope(name):
      with slim.arg_scope([slim.conv2d], 
                          activation_fn=None, 
                          biases_initializer=tf.constant_initializer(0.0),
                          trainable=is_training): 
        # compute the gates and features
        p_input = slim.conv2d(pool5_input, 
                              cfg.MEM.C, 
                              [mconv, mconv], 
                              weights_initializer=feat_initializer,
                              scope="input_p")
        p_reset = slim.conv2d(pool5_input, 
                              1, 
                              [mconv, mconv],
                              weights_initializer=feat_gate_initializer,
                              scope="reset_p")
        p_update = slim.conv2d(pool5_input, 
                              1, 
                              [mconv, mconv],
                              weights_initializer=feat_gate_initializer,
                              scope="update_p")
        # compute the gates and features from the hidden memory
        m_reset = slim.conv2d(pool5_mem, 
                              1,
                              [mconv, mconv],
                              weights_initializer=mem_gate_initializer,
                              biases_initializer=None,
                              scope="reset_m")
        m_update = slim.conv2d(pool5_mem, 
                              1,
                              [mconv, mconv], 
                              weights_initializer=mem_gate_initializer,
                              biases_initializer=None,
                              scope="update_m")
        # get the reset gate, the portion that is kept from the previous step
        reset_gate = tf.sigmoid(p_reset + m_reset, name="reset_gate")
        reset_res = tf.multiply(pool5_mem, reset_gate, name="m_input_reset")
        m_input = slim.conv2d(reset_res, 
                              cfg.MEM.C,
                              [mconv, mconv], 
                              weights_initializer=mem_initializer,
                              biases_initializer=None,
                              scope="input_m")
        # Non-linear activation
        pool5_new = tf.nn.relu(p_input + m_input, name="pool5_new")
        # get the update gate, the portion that is taken to update the new memory
        update_gate = tf.sigmoid(p_update + m_update, name="update_gate")
        # the update is done in a difference manner
        mem_diff = tf.multiply(update_gate, pool5_new - pool5_mem, 
                              name="mem_diff") 

      self._score_summaries[iter].append(p_reset)
      self._score_summaries[iter].append(p_update)
      self._score_summaries[iter].append(m_reset)
      self._score_summaries[iter].append(m_update)
      self._score_summaries[iter].append(reset_gate)
      self._score_summaries[iter].append(update_gate)
      self._score_summaries[iter].append(mem_diff)

    return mem_diff

  def _input_module(self, pool5_nb, 
                    cls_score_nb, cls_prob_nb, cls_pred_nb, 
                    is_training, iter):
    pool5_comb = self._bottomtop(pool5_nb, cls_score_nb, is_training, "bottom_top", iter)
    pool5_input = self._input(pool5_comb, is_training, "pool5_input", iter)

    return pool5_input

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

    return cls_score_conv, pool5_nb, \
           rois, batch_ids, inv_rois, inv_batch_ids

  def _build_pred(self, is_training, mem, cls_score_conv, rois, batch_ids, iter):
    if cfg.MEM.CT_L:
      mem_net = self._context(mem, is_training, "context", iter)
    else:
      mem_net = mem
    mem_ct_pool5 = self._crop_rois(mem_net, rois, batch_ids, "mem_ct_pool5", iter)
    mem_fc7 = self._fc_iter(mem_ct_pool5, is_training, "fc7", iter) 
    cls_score_mem = self._cls_iter(mem_fc7, is_training, "cls_iter", iter) 
    cls_score, cls_prob, cls_pred = self._comb_conv_mem(cls_score_conv, cls_score_mem, 
                                                        "comb_conv_mem", iter)

    return cls_score, cls_prob, cls_pred

  def _build_update(self, is_training, mem, pool5_nb, cls_score, cls_prob, cls_pred,
                    rois, batch_ids, inv_rois, inv_batch_ids, iter):
    cls_score_nb = tf.stop_gradient(cls_score, name="cls_score_nb")
    cls_prob_nb = tf.stop_gradient(cls_prob, name="cls_prob_nb")
    cls_pred_nb = tf.stop_gradient(cls_pred, name="cls_pred_nb")
    pool5_mem = self._crop_rois(mem, rois, batch_ids, "pool5_mem", iter)
    pool5_input = self._input_module(pool5_nb, 
                                    cls_score_nb, cls_prob_nb, 
                                    cls_pred_nb, is_training, iter)
    mem_update = self._mem_update(pool5_mem, pool5_input, is_training, "mem_update", iter) 
    mem_diff, _ = self._inv_crops(mem_update, inv_rois, inv_batch_ids, "inv_crop")
    self._score_summaries[iter].append(mem_diff)
    # Update the memory
    mem_div = tf.div(mem_diff, self._count_matrix_eps, name="div")
    mem = tf.add(mem, mem_div, name="add")
    self._score_summaries[iter].append(mem)

    return mem

  def _build_memory(self, is_training, is_testing):
    # initialize memory
    mem = self._mem_init(is_training, "mem_init")
    # convolution related stuff
    cls_score_conv, pool5_nb, \
    rois, batch_ids, inv_rois, inv_batch_ids = self._build_conv(is_training)
    # Separate first prediction
    reuse = None
    # Memory iterations
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
        
      if iter == 0:
        reuse = True

    return rois, cls_prob

  def _add_memory_losses(self, name):
    cross_entropy = []
    assert len(self._predictions["cls_score"]) == cfg.MEM.ITER
    with tf.variable_scope(name):
      label = tf.reshape(self._targets["labels"], [-1])
      for iter in xrange(cfg.MEM.ITER):
        # RCNN, class loss
        cls_score = self._predictions["cls_score"][iter]
        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, 
                                                                           labels=label))
        cross_entropy.append(ce)

      ce_rest = tf.stack(cross_entropy[1:], name="cross_entropy_rest")
      self._losses['cross_entropy_image'] = cross_entropy[0]
      self._losses['cross_entropy_memory'] = tf.reduce_mean(ce_rest, name='cross_entropy')
      self._losses['cross_entropy'] = self._losses['cross_entropy_image'] \
                                    + cfg.MEM.WEIGHT * self._losses['cross_entropy_memory']

      loss = self._losses['cross_entropy']
      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      self._losses['total_loss'] = loss + regularization_loss

      self._event_summaries.update(self._losses)

    return loss

  def _create_summary(self):
    val_summaries = []
    with tf.device("/cpu:0"):
      val_summaries.append(self._add_gt_image_summary())
      for iter in xrange(cfg.MEM.ITER):
        val_summaries.append(self._add_pred_memory_summary(iter))
        val_summaries.append(self._add_memory_summary(iter))
        for var in self._score_summaries[iter]:
          self._add_score_iter_summary(iter, var)
      for key, var in self._event_summaries.items():
        val_summaries.append(tf.summary.scalar(key, var))
      for var in self._act_summaries:
        self._add_zero_summary(var)

    self._summary_op = tf.summary.merge_all()
    self._summary_op_val = tf.summary.merge(val_summaries)

  def create_architecture(self, mode, num_classes, tag=None):
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._memory_size = tf.placeholder(tf.int32, shape=[2])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._count_base = tf.ones([1, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1])
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
      rois = self._build_memory(training, testing)

    layers_to_output = {'rois': rois}

    if not testing:
      self._add_memory_losses("loss")
      layers_to_output.update(self._losses)
      self._create_summary()

    layers_to_output.update(self._predictions)

    return layers_to_output

  # take the last predicted output
  def test_image(self, sess, blobs):
    cls_score, cls_prob = sess.run([self._predictions["cls_score"][-1],
                                    self._predictions['cls_prob'][-1]],
                                   feed_dict=self._parse_dict(blobs))
    return cls_score, cls_prob

  # Test the base output
  def test_image_iter(self, sess, blobs, iter):
    cls_score, cls_prob = sess.run([self._predictions["cls_score"][iter],
                                    self._predictions['cls_prob'][iter]],
                                   feed_dict=self._parse_dict(blobs))
    return cls_score, cls_prob

class vgg16_memory(BaseMemory, vgg16):
  def __init__(self):
    BaseMemory.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'vgg_16'

class resnetv1_memory(BaseMemory, resnetv1):
  def __init__(self, num_layers=50):
    BaseMemory.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._num_layers = num_layers
    self._scope = 'resnet_v1_%d' % num_layers
    resnetv1._decide_blocks(self)

class mobilenetv1_memory(BaseMemory, mobilenetv1):
  def __init__(self):
    BaseMemory.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
    self._scope = 'MobilenetV1'
