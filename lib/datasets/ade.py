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

class ade(imdb):
  def __init__(self, image_set, count=5):
    imdb.__init__(self, 'ade_%s_%d' % (image_set,count))
    self._image_set = image_set
    self._root_path = osp.join(cfg.DATA_DIR, 'ADE')
    self._name_file = osp.join(self._root_path, 'objectnames.txt')
    self._count_file = osp.join(self._root_path, 'objectcounts.txt')
    self._anno_file = osp.join(self._root_path, self._image_set + '.txt')
    with open(self._anno_file) as fid:
      image_index = fid.readlines()
      self._image_index = [ii.strip() for ii in image_index]
    with open(self._name_file) as fid:
      raw_names = fid.readlines()
      self._raw_names = [n.strip().replace(' ', '_') for n in raw_names]
      self._len_raw = len(self._raw_names)
    with open(self._count_file) as fid:
      raw_counts = fid.readlines()
      self._raw_counts = np.array([int(n.strip()) for n in raw_counts])

    # First class is always background
    self._ade_inds = [0] + list(np.where(self._raw_counts >= count)[0])
    self._classes = ['__background__']

    for idx in self._ade_inds:
      if idx == 0:
        continue
      ade_name = self._raw_names[idx]
      self._classes.append(ade_name)

    self._classes = tuple(self._classes)
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self.set_proposal_method('gt')

  def _load_text(self, text_path):
    class_keys = {}
    with open(text_path) as fid:
      lines = fid.readlines()
      for line in lines:
        columns = line.split('#')
        key = '%s_%s' % (columns[0].strip(), columns[1].strip())
        # Just get the class ID
        class_name = columns[4].strip().replace(' ', '_')
        if class_name in self._class_to_ind:
          class_keys[key] = self._class_to_ind[class_name]
      total_num_ins = len(lines)

    return class_keys, total_num_ins

  def _load_annotation(self):
    gt_roidb = []

    for i in xrange(self.num_images):
      image_path = self.image_path_at(i)
      if i % 10 == 0:
        print(image_path)
      # Estimate the number of objects from text file
      text_path = image_path.replace('.jpg', '_atr.txt')
      class_keys, total_num_ins = self._load_text(text_path)

      valid_num_ins = 0
      boxes = np.zeros((total_num_ins, 4), dtype=np.uint16)
      gt_classes = np.zeros((total_num_ins), dtype=np.int32)
      seg_areas = np.zeros((total_num_ins), dtype=np.float32)

      # First, whole objects
      label_path = image_path.replace('.jpg', '_seg.png')
      seg = cv2.imread(label_path)
      height, width, _ = seg.shape

      # OpenCV has reversed RGB
      instances = seg[:, :, 0]
      unique_ins = np.unique(instances)

      for t, ins in enumerate(list(unique_ins)):
        if ins == 0:
          continue
        key = '%03d_%d' % (t, 0)
        if key in class_keys:
          ins_seg = np.where(instances == ins)
          x1 = ins_seg[1].min()
          x2 = ins_seg[1].max()
          y1 = ins_seg[0].min()
          y2 = ins_seg[0].max()
          boxes[valid_num_ins, :] = [x1, y1, x2, y2]
          gt_classes[valid_num_ins] = class_keys[key]
          seg_areas[valid_num_ins] = ins_seg[0].shape[0]
          valid_num_ins += 1

      # Then deal with parts
      level = 1
      while True:
        part_path = image_path.replace('.jpg', '_parts_%d.png' % level)
        if osp.exists(part_path):
          seg = cv2.imread(part_path)
          instances = seg[:, :, 0]
          unique_ins = np.unique(instances)

          for t, ins in enumerate(list(unique_ins)):
            if ins == 0:
              continue
            key = '%03d_%d' % (t, level)
            if key in class_keys:
              ins_seg = np.where(instances == ins)
              x1 = ins_seg[1].min()
              x2 = ins_seg[1].max()
              y1 = ins_seg[0].min()
              y2 = ins_seg[0].max()
              boxes[valid_num_ins, :] = [x1, y1, x2, y2]
              gt_classes[valid_num_ins] = class_keys[key]
              seg_areas[valid_num_ins] = ins_seg[0].shape[0]
              valid_num_ins += 1

          level += 1
        else:
          break

      boxes = boxes[:valid_num_ins, :]
      gt_classes = gt_classes[:valid_num_ins]
      seg_areas = seg_areas[:valid_num_ins]

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
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      with open(image_file, 'rb') as fid:
        self._image_index = pickle.load(fid)
      print('{} gt image loaded from {}'.format(self.name, image_file))
      return gt_roidb

    gt_roidb = self._load_annotation()
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    with open(image_file, 'wb') as fid:
      pickle.dump(self._image_index, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt image to {}'.format(image_file))
    return gt_roidb

  def _find_parts(self, part_counts, seg, part_seg, level, class_keys):
    part_instances = part_seg[:, :, 0]
    obj_instances = seg[:, :, 0]
    part_unique_ins = np.unique(part_instances)
    obj_unique_ins, obj_map = np.unique(obj_instances, return_inverse=True)
    obj_map = obj_map.reshape(obj_instances.shape)
    bins = np.empty((len(obj_unique_ins)), dtype=np.uint32)

    for t, ins in enumerate(list(part_unique_ins)):
      if ins == 0:
        continue
      key_part = '%03d_%d' % (t, level)
      if key_part in class_keys:
        ins_seg = np.where(part_instances == ins)
        obj_pixels = obj_map[ins_seg[0], ins_seg[1]].astype(np.uint16)
        # Do a pooling
        bins.fill(0)
        for pix in obj_pixels:
          bins[pix] += 1
        obj_t = np.argmax(bins)
        key_obj = '%03d_%d' % (obj_t, level-1)
        if key_obj in class_keys:
          part_cls = class_keys[key_part]
          obj_cls = class_keys[key_obj]
          part_name = self.classes[part_cls]
          obj_name = self.classes[obj_cls]
          query = '%s||%s' % (part_name, obj_name)
          if query in part_counts:
            part_counts[query] += 1
          else:
            print(query)
            part_counts[query] = 1

    return part_counts

  def part_relationships(self):
    part_counts = {}

    for i in xrange(self.num_images):
      image_path = self.image_path_at(i)
      text_path = image_path.replace('.jpg', '_atr.txt')
      class_keys, _ = self._load_text(text_path)
      if i % 10 == 0:
        print(image_path)
      # Estimate the number of objects from text file
      level = 1
      seg_path = image_path.replace('.jpg', '_seg.png')
      while True:
        part_path = image_path.replace('.jpg', '_parts_%d.png' % level)
        if osp.exists(part_path):
          if level == 1:
            seg = cv2.imread(seg_path)
          else:
            seg = part_seg
          part_seg = cv2.imread(part_path)
          part_counts = self._find_parts(part_counts, seg, part_seg, level, class_keys)
          level += 1
        else:
          break

    return part_counts

  # Should be removed, no bad images available
  def find_bad_images(self):
    for i in xrange(self.num_images):
      if i % 10 == 0:
        print('%d/%d' % (i, self.num_images))
      image_path = self.image_path_at(i)
      im = cv2.imread(image_path)
      h_im, w_im, _ = im.shape

      seg_path = image_path.replace('.jpg', '_seg.png')
      seg = cv2.imread(seg_path)
      h_seg, w_seg, _ = seg.shape

      if h_im != h_seg or w_im != w_seg:
        print('SEG:%s' % image_path)
        continue

      level = 1
      while True:
        part_path = image_path.replace('.jpg', '_parts_%d.png' % level)
        if osp.exists(part_path):
          part_seg = cv2.imread(part_path)
          h_part, w_part, _ = part_seg.shape

          if h_im != h_part or w_im != w_part:
            print('P%d:%s' % (level, image_path))
            break

          level += 1
        else:
          break

  # Do some left-right flipping here
  def _find_flipped_classes(self):
    self._flipped_classes = np.arange(self.num_classes, dtype=np.int32)
    for i, cls_name in enumerate(self.classes):
      if cls_name.startswith('left_'):
        query = cls_name.replace('left_', 'right_')
        idx = self._class_to_ind[query]
        # Swap for both left and right
        self._flipped_classes[idx] = i
        self._flipped_classes[i] = idx

  def append_flipped_images(self):
    self._find_flipped_classes()
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
               'gt_classes': self._flipped_classes[self.roidb[i]['gt_classes']],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2
