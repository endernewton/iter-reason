from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from datasets.voc_eval import voc_ap
from model.config import cfg
from data.layer import DataLayer
from data.minibatch import get_minibatch

class imdb(object):
  """Image database."""

  def __init__(self, name, classes=None):
    self._name = name
    if not classes:
      self._classes = []
    else:
      self._classes = classes
    self._image_index = []
    self._obj_proposer = 'gt'
    self._roidb = None
    self._roidb_handler = self.default_roidb
    # Use this dict for storing dataset specific config options
    self._data_layer = DataLayer
    self._minibatch = get_minibatch
    self.config = {}

  @property
  def name(self):
    return self._name

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def image_index(self):
    return self._image_index

  @property
  def data_layer(self):
    return self._data_layer

  @property
  def minibatch(self):
    return self._minibatch

  @property
  def roidb_handler(self):
    return self._roidb_handler

  @roidb_handler.setter
  def roidb_handler(self, val):
    self._roidb_handler = val

  def set_proposal_method(self, method):
    method = eval('self.' + method + '_roidb')
    self.roidb_handler = method

  @property
  def roidb(self):
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   gt_overlaps
    #   gt_classes
    #   flipped
    if self._roidb is not None:
      return self._roidb
    self._roidb = self.roidb_handler()
    return self._roidb

  @property
  def cache_path(self):
    cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
    if not os.path.exists(cache_path):
      os.makedirs(cache_path)
    return cache_path

  @property
  def num_images(self):
    return len(self.image_index)

  def image_path_at(self, i):
    raise NotImplementedError

  def default_roidb(self):
    raise NotImplementedError

  def _score(self, all_scores):
    scs = [0.] * self.num_classes
    scs_all = [0.] * self.num_classes
    valid = [0] * self.num_classes
    for i in xrange(1, self.num_classes):
      ind_this = np.where(self.gt_classes == i)[0]  
      scs_all[i] = np.sum(all_scores[ind_this, i])
      if ind_this.shape[0] > 0:
        valid[i] = ind_this.shape[0]
        scs[i] = scs_all[i] / ind_this.shape[0]

    mcls_sc = np.mean([s for s, v in zip(scs,valid) if v])
    mins_sc = np.sum(scs_all) / self.gt_classes.shape[0]
    return scs[1:], mcls_sc, mins_sc, valid[1:]

  def _accuracy(self, all_scores):
    acs = [0.] * self.num_classes
    acs_all = [0.] * self.num_classes
    valid = [0] * self.num_classes

    # Need to remove the background class
    max_inds = np.argmax(all_scores[:, 1:], axis=1) + 1
    max_scores = np.empty_like(all_scores)
    max_scores[:] = 0.
    max_scores[np.arange(self.gt_classes.shape[0]), max_inds] = 1.

    for i in xrange(1, self.num_classes):
      ind_this = np.where(self.gt_classes == i)[0]
      acs_all[i] = np.sum(max_scores[ind_this, i])
      if ind_this.shape[0] > 0:
        valid[i] = ind_this.shape[0]
        acs[i] = acs_all[i] / ind_this.shape[0]

    mcls_ac = np.mean([s for s, v in zip(acs,valid) if v])
    mins_ac = np.sum(acs_all) / self.gt_classes.shape[0]
    return acs[1:], mcls_ac, mins_ac

  def _average_precision(self, all_scores):
    aps = [0.] * self.num_classes
    valid = [0] * self.num_classes

    ind_all = np.arange(self.gt_classes.shape[0])
    num_cls = self.num_classes
    num_ins = ind_all.shape[0]

    for i, c in enumerate(self._classes):
      if i == 0:
        continue
      gt_this = (self.gt_classes == i).astype(np.float32)
      num_this = np.sum(gt_this)
      if i % 10 == 0:
        print('AP for %s: %d/%d' % (c, i, num_cls))
      if num_this > 0:
        valid[i] = num_this
        sco_this = all_scores[ind_all, i]

        ind_sorted = np.argsort(-sco_this)

        tp = gt_this[ind_sorted]
        max_ind = num_ins - np.argmax(tp[::-1])
        tp = tp[:max_ind]
        fp = 1. - tp

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / float(num_this)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        aps[i] = voc_ap(rec, prec)

    mcls_ap = np.mean([s for s, v in zip(aps,valid) if v])

    # Compute the overall score
    max_inds = np.argmax(all_scores[:, 1:], axis=1) + 1
    max_scores = np.empty_like(all_scores)
    max_scores[:] = 0.
    max_scores[ind_all, max_inds] = 1.
    pred_all = max_scores[ind_all, self.gt_classes]
    sco_all = all_scores[ind_all, self.gt_classes]
    ind_sorted = np.argsort(-sco_all)

    tp = pred_all[ind_sorted]
    fp = 1. - tp

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / float(num_ins)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    mins_ap = voc_ap(rec, prec)
    return aps[1:], mcls_ap, mins_ap

  def evaluate_detail(self, all_scores, output_dir, roidb=None):
    if roidb is None:
      roidb = self.roidb
    all_scores = np.vstack(all_scores)
    # all_scores = np.minimum(all_scores, 1.0)
    self.gt_classes = np.hstack([r['gt_classes'] for r in roidb])

    scs, mcls_sc, mins_sc, valid = self._score(all_scores)
    acs, mcls_ac, mins_ac = self._accuracy(all_scores)
    aps, mcls_ap, mins_ap = self._average_precision(all_scores)

    for i, cls in enumerate(self._classes):
      if cls == '__background__' or not valid[i-1]:
        continue
      if valid[i-1]:
        print(('{} {:d} {:.4f} {:.4f}'.format(cls, 
                                              valid[i-1], 
                                              acs[i-1], 
                                              aps[i-1])))

    print('~~~~~~~~')
    # print('Accuracies | APs:')
    # for ac, ap, vl in zip(acs, aps, valid):
    #   if vl:
    #     print(('{:.3f} {:.3f}'.format(ac, ap)))
    print(('mean-cls: {:.3f} {:.3f}'.format(mcls_ac, mcls_ap)))
    print(('mean-ins: {:.3f} {:.3f}'.format(mins_ac, mins_ap)))
    print('~~~~~~~~')
    print(('{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(mcls_sc, 
                                                              mcls_ac, 
                                                              mcls_ap, 
                                                              mins_sc, 
                                                              mins_ac, 
                                                              mins_ap)))
    print('~~~~~~~~')

    return acs, aps

  def evaluate(self, all_scores, output_dir, roidb=None):
    if roidb is None:
      roidb = self.roidb
    all_scores = np.vstack(all_scores)
    # all_scores = np.minimum(all_scores, 1.0)
    self.gt_classes = np.hstack([r['gt_classes'] for r in roidb])

    scs, mcls_sc, mins_sc, valid = self._score(all_scores)
    acs, mcls_ac, mins_ac = self._accuracy(all_scores)
    aps, mcls_ap, mins_ap = self._average_precision(all_scores)

    for i, cls in enumerate(self._classes):
      if cls == '__background__' or not valid[i-1]:
        continue
      print(('{} {:d} {:.4f} {:.4f} {:.4f}'.format(cls, 
                                                  valid[i-1], 
                                                  scs[i-1], 
                                                  acs[i-1], 
                                                  aps[i-1])))

    print('~~~~~~~~')
    # print('Scores | Accuracies | APs:')
    # for sc, ac, ap, vl in zip(scs, acs, aps, valid):
    #   if vl:
    #     print(('{:.3f} {:.3f} {:.3f}'.format(sc, ac, ap)))
    print(('mean-cls: {:.3f} {:.3f} {:.3f}'.format(mcls_sc, mcls_ac, mcls_ap)))
    print(('mean-ins: {:.3f} {:.3f} {:.3f}'.format(mins_sc, mins_ac, mins_ap)))
    print('~~~~~~~~')
    print(('{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(mcls_sc, 
                                                              mcls_ac, 
                                                              mcls_ap, 
                                                              mins_sc, 
                                                              mins_ac, 
                                                              mins_ap)))
    print('~~~~~~~~')

    return mcls_sc, mcls_ac, mcls_ap, mins_sc, mins_ac, mins_ap

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2

