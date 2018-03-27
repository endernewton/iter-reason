"""Compute minibatch blobs for training a region classification network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, is_training=True):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  scales = cfg.TRAIN.SCALES if is_training else cfg.TEST.SCALES
  max_scale = cfg.TRAIN.MAX_SIZE if is_training else cfg.TEST.MAX_SIZE
  random_scale_inds = npr.randint(0, high=len(scales), size=num_images)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = get_image_blob(roidb, random_scale_inds, scales, max_scale)

  blobs = {'data': im_blob}
  
  # gt boxes: (x1, y1, x2, y2, cls)
  gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  # height, width, scale
  blobs['im_info'] = np.array([im_blob.shape[1], 
                              im_blob.shape[2], 
                              im_scales[0]], dtype=np.float32)
  blobs['memory_size'] = np.ceil(blobs['im_info'][:2] / cfg.BOTTLE_SCALE).astype(np.int32)
  blobs['num_gt'] = np.int32(gt_boxes.shape[0])

  return blobs

def get_image_blob(roidb, scale_inds, scales, max_scale):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = scales[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    max_scale)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
