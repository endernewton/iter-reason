from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Memory options
#
__C.MEM = edict()

# Number of memory iterations
__C.MEM.ITER = 2

# Height of the memory
__C.MEM.INIT_H = 20
# Width of the memory
__C.MEM.INIT_W = 20

# Channel of the memory
__C.MEM.C = 512

# Basic stds in the memory
__C.MEM.STD = 0.01
# Base stds in the memory update function for input features
__C.MEM.U_STD = 0.01
# Region classification
__C.MEM.C_STD = 0.01

# Feature to memory ratio
__C.MEM.FM_R = 1.
# Value to gate ratio
__C.MEM.VG_R = 1.
# FC to Pool ratio when combing the input
__C.MEM.FP_R = 1.

# Conv kernel size for memory
__C.MEM.CONV = 3

# Canonical region size
__C.MEM.CROP_SIZE = 7

# Context aggregation
__C.MEM.CT_L = 3
__C.MEM.CT_CONV = 3
__C.MEM.CT_FCONV = 3

# Input feature
__C.MEM.IN_L = 2
__C.MEM.IN_CONV = 3
# Input the total scores or not
__C.MEM.IN_A = False

# Memory final fc layer channels
__C.MEM.FC_C = 4096
__C.MEM.FC_L = 2

# Clipping the gradients
__C.MEM.CLIP = 0.

# How to take care of the loss
__C.MEM.LOSS = 'avg'

# Input memory
__C.MEM.INPUT = 'combs'

# Update memory
__C.MEM.U = 'relu'
__C.MEM.UA = 'avg'

# For segmentation, how to crop the memory
__C.MEM.SEG_MEM = False

# The weight for the memory based prediction
__C.MEM.W = 1.
# Final supervision weight
__C.MEM.WF = 1.

__C.MEM.B = 1.

# Combining methods for prediction and confidence
__C.MEM.COMB = 'base'
# Combining confidences
__C.MEM.CC = 'mem'

# Stop grediant for attention
__C.MEM.SAT = False

#
# Training options
#
__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.RATE = 0.0005

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0001

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = [30000]

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 20

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 2

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_ITERS = 500

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 180

# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# How much to jitter the ground-truth bounding box, right now closed
__C.TRAIN.BBOX_THRESH = 1.

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'res101'

#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

#
# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize. 
# if true, the region will be resized to a square of 2xPOOLING_SIZE, 
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

#
# MobileNet options
#

__C.MOBILENET = edict()

# Whether to regularize the depth-wise filters during training
__C.MOBILENET.REGU_DEPTH = False

# Number of fixed layers during training, by default the bottom 5 of 14 layers is fixed
# Range: 0 (none) to 12 (all)
__C.MOBILENET.FIXED_LAYERS = 5

# Weight decay for the mobilenet weights
__C.MOBILENET.WEIGHT_DECAY = 0.00004

# Depth multiplier
__C.MOBILENET.DEPTH_MULTIPLIER = 1.

#
# MISC
#

# Class names, for visualization purposes
__C.CLASSES = None

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Scale of the head spatially
__C.BOTTLE_SCALE = 16.

# EPS, a small number for numerical issue
__C.EPS = 1e-14


def get_output_dir(imdb, weights_filename):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def get_output_tb_dir(imdb, weights_filename):
  """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
