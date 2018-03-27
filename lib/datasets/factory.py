"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.vrd import vrd
from datasets.ade import ade
from datasets.adeseg import adeseg
from datasets.visual_genome import visual_genome

import numpy as np

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up ade_<split>_5
for split in ['train', 'val', 'mval', 'mtest']:
  name = 'ade_{}_5'.format(split)
  __sets[name] = (lambda split=split: ade(split))

# Set up vg_<split>_5,10
for split in ['train', 'val', 'test']:
  name = 'visual_genome_{}_5'.format(split)
  __sets[name] = (lambda split=split: visual_genome(split))
  

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
