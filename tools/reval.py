#!/usr/bin/env python
# Reval = re-eval. Re-evaluate saved detections.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from datasets.factory import get_imdb
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os, sys, argparse
import numpy as np


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Re-evaluate results')
  parser.add_argument('output_dir', nargs=1, help='results directory',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to re-evaluate',
                      default='ade_mtest_5', type=str)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def from_results(imdb_name, output_dir, args):
  imdb = get_imdb(imdb_name)
  with open(os.path.join(output_dir, 'results.pkl'), 'rb') as f:
    results = pickle.load(f)

  print('Evaluating detections')
  imdb.evaluate(results, output_dir)


if __name__ == '__main__':
  args = parse_args()

  output_dir = os.path.abspath(args.output_dir[0])
  imdb_name = args.imdb_name
  from_results(imdb_name, output_dir, args)
