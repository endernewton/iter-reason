from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from utils.cython_bbox import bbox_overlaps
try:
  import cPickle as pickle
except ImportError:
  import pickle

# Just return the ground truth boxes for a single image
def compute_target(memory_size, gt_boxes, feat_stride):
  factor_h = (memory_size[0] - 1.) * feat_stride
  factor_w = (memory_size[1] - 1.) * feat_stride
  num_gt = gt_boxes.shape[0]

  x1 = gt_boxes[:, [0]] / factor_w
  y1 = gt_boxes[:, [1]] / factor_h
  x2 = gt_boxes[:, [2]] / factor_w
  y2 = gt_boxes[:, [3]] / factor_h

  rois = np.hstack((y1, x1, y2, x2))
  batch_ids = np.zeros((num_gt), dtype=np.int32)
  # overlap to regions of interest
  roi_overlaps = np.ones((num_gt), dtype=np.float32)
  labels = np.array(gt_boxes[:, 4], dtype=np.int32)

  return rois, batch_ids, roi_overlaps, labels

def compute_patch(im_info, gt_boxes):
  factor_h = im_info[0] - 1.
  factor_w = im_info[1] - 1.
  num_gt = gt_boxes.shape[0]

  x1 = gt_boxes[:, [0]] / factor_w
  y1 = gt_boxes[:, [1]] / factor_h
  x2 = gt_boxes[:, [2]] / factor_w
  y2 = gt_boxes[:, [3]] / factor_h

  im_rois = np.hstack((y1, x1, y2, x2))
  im_batch_ids = np.zeros((num_gt), dtype=np.int32)

  return im_rois, im_batch_ids

def normalize(graph):
  norm = 1. / np.maximum(np.sum(graph, axis=1, keepdims=True), cfg.EPS)
  return graph * norm

# Also return the reverse index of rois
def compute_target_memory(memory_size, gt_boxes, feat_stride):
  minus_h = memory_size[0] - 1.
  minus_w = memory_size[1] - 1.
  num_gt = gt_boxes.shape[0]

  x1 = gt_boxes[:, [0]] / feat_stride
  y1 = gt_boxes[:, [1]] / feat_stride
  x2 = gt_boxes[:, [2]] / feat_stride
  y2 = gt_boxes[:, [3]] / feat_stride

  # h, w, h, w
  rois = np.hstack((y1, x1, y2, x2))
  rois[:, 0::2] /= minus_h
  rois[:, 1::2] /= minus_w
  batch_ids = np.zeros((num_gt), dtype=np.int32)
  labels = np.array(gt_boxes[:, 4], dtype=np.int32)

  # h, w, h, w
  inv_rois = np.empty_like(rois)
  inv_rois[:, 0:2] = 0.
  inv_rois[:, 2] = minus_h
  inv_rois[:, 3] = minus_w
  inv_rois[:, 0::2] -= y1
  inv_rois[:, 1::2] -= x1

  # normalize coordinates
  inv_rois[:, 0::2] /= np.maximum(y2 - y1, cfg.EPS)
  inv_rois[:, 1::2] /= np.maximum(x2 - x1, cfg.EPS)

  inv_batch_ids = np.arange((num_gt), dtype=np.int32)

  return rois, batch_ids, labels, inv_rois, inv_batch_ids

def update_weights(labels, cls_prob):
  num_gt = labels.shape[0]
  index = np.arange(num_gt)
  cls_score = cls_prob[index, labels]
  big_ones = cls_score >= 1. - cfg.MEM.B
  # Focus on the hard examples
  weights = 1. - cls_score
  weights[big_ones] = cfg.MEM.B
  weights /= np.maximum(np.sum(weights), cfg.EPS)
  
  return weights
