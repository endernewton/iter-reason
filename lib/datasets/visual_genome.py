from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
try:
  import cPickle as pickle
except ImportError:
  import pickle
import json
import cv2
import numpy as np

from datasets.imdb import imdb
from model.config import cfg

class visual_genome(imdb):
  def __init__(self, image_set, count=5):
    imdb.__init__(self, 'visual_genome_%s_%d' % (image_set, count))
    self._image_set = image_set
    self._root_path = osp.join(cfg.DATA_DIR, 'visual_genome')
    self._name_file = osp.join(self._root_path, 'synsets.txt')
    self._anno_file = osp.join(self._root_path, self._image_set + '.json')
    self._image_file = osp.join(self._root_path, 'image_data.json')
    with open(self._name_file) as fid:
      lines = fid.readlines()
      self._raw_names = []
      self._raw_counts = []
      for line in lines:
        name, cc = line.strip().split(':')
        cc = int(cc)
        self._raw_names.append(name)
        self._raw_counts.append(cc)
      self._len_raw = len(self._raw_names)

    self._raw_counts = np.array(self._raw_counts)
    # First class is always background
    self._vg_inds = [0] + list(np.where(self._raw_counts >= count)[0])
    self._classes = ['__background__']

    for idx in self._vg_inds:
      if idx == 0:
        continue
      vg_name = self._raw_names[idx]
      self._classes.append(vg_name)

    self._classes = tuple(self._classes)
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self.set_proposal_method('gt')
    # Call to get one
    self.roidb

  def _load_annotation(self):
    gt_roidb = []

    with open(self._anno_file) as fid:
      annos = json.load(fid)

    with open(self._image_file) as fid:
      images = json.load(fid)
      image_ids = [item['image_id'] for item in images]
      image_dicts = dict(zip(image_ids, range(len(image_ids))))

    for i in xrange(len(annos)):
      anno = annos[i]
      idx = image_dicts[anno['image_id']]
      image = images[idx]
      image_path = image['url'].replace('https://cs.stanford.edu/people/rak248/','VG/').encode('ascii')
      self._image_index.append(image_path)
      if i % 100 == 0:
        print('%d: %s' % (i, image_path))
      width = image['width']
      height = image['height']

      objects = anno['objects']
      num_objs = sum([len(x['synsets']) for x in objects])

      boxes = np.zeros((num_objs, 4), dtype=np.uint16)
      gt_classes = np.zeros((num_objs), dtype=np.int32)
      seg_areas = np.zeros((num_objs), dtype=np.float32)

      ix = 0
      for obj in objects:
        names = obj['synsets']
        x = obj['x']
        y = obj['y']
        w = obj['w']
        h = obj['h']

        # need to verify if x and y are 0 based
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w - 1, width - 1)
        y2 = min(y + h - 1, height - 1)

        if x1 > x2 or y1 > y2:
          continue

        for n in names:
          if n in self.classes:
            cls = self._class_to_ind[n]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix += 1

      boxes = boxes[:ix,:]
      gt_classes = gt_classes[:ix]
      seg_areas = seg_areas[:ix]

      gt_roidb.append({'width': width,
                      'height': height,
                      'boxes' : boxes,
                      'gt_classes': gt_classes,
                      'flipped' : False,
                      'seg_areas': seg_areas})
    return gt_roidb

  def image_path_at(self, i):
    return osp.join(self._root_path, self._image_index[i])

  def gt_roidb(self):
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    image_file = osp.join(self.cache_path, self.name + '_gt_image.pkl')
    if osp.exists(cache_file) and osp.exists(image_file):
      with open(cache_file, 'rb') as fid:
        gt_roidb = pickle.load(fid)
      print('{} gt roidb lovgd from {}'.format(self.name, cache_file))
      with open(image_file, 'rb') as fid:
        self._image_index = pickle.load(fid)
      print('{} gt image lovgd from {}'.format(self.name, image_file))
      return gt_roidb

    gt_roidb = self._load_annotation()
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    with open(image_file, 'wb') as fid:
      pickle.dump(self._image_index, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt image to {}'.format(image_file))
    return gt_roidb
