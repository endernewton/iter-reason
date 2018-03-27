"""The data layer used during training to train the region classifier.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from data.minibatch import get_minibatch
import numpy as np
import time

class DataLayer(object):
  """Data layer used for training."""

  def __init__(self, roidb, num_classes, random=False):
    """Set the roidb to be used by this layer during training."""
    self._roidb = roidb
    self._num_classes = num_classes
    # Also set a random flag
    self._random = random
    self._shuffle_roidb_inds()

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    # If the random flag is set, 
    # then the database is shuffled according to system time
    # Useful for the validation set
    if self._random:
      st0 = np.random.get_state()
      millis = int(round(time.time() * 1000)) % 4294967295
      np.random.seed(millis)
    
    self._perm = np.random.permutation(np.arange(len(self._roidb)))
    # Restore the random state
    if self._random:
      np.random.set_state(st0)
      
    self._cur = 0

  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
      self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    self._cur += cfg.TRAIN.IMS_PER_BATCH

    return db_inds

  def forward(self):
    """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    db_inds = self._get_next_minibatch_inds()
    minibatch_db = [self._roidb[i] for i in db_inds]
    return get_minibatch(minibatch_db)
      
