from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
import nets.seg_memory as seg_memory
import nets.check_memory as check_memory
import nets.gcheck_memory as gcheck_memory

import nets.base_memory as base_memory
import nets.attend_memory as attend_memory

import nets.gbase_memory as gbase_memory
import nets.gattend_memory as gattend_memory

import nets.comb_memory as comb_memory

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a region classification network')
  parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
                        help='model to test',
                        default=None, type=str)
  parser.add_argument('--visualize', dest='visualize', help='whether to show results',
                        action='store_true')
  parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the initialization weights
  if args.model:
    filename = os.path.splitext(os.path.basename(args.model))[0]
  else:
    filename = os.path.splitext(os.path.basename(args.weight))[0]

  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  imdb = get_imdb(args.imdb_name)
  cfg.CLASSES = imdb.classes

  for i in range(len(imdb.image_index)):
    imdb.roidb[i]['image'] = imdb.image_path_at(i)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
  
  net_base, net_tag = args.net.split('_')

  # Define the graphs
  if net_tag in ('gbase','sgbase','gattend','sepgat','comb'):
    if imdb.name.startswith('ade'):
      cfg.GRA.W_B = ('IK','iIK','LR','iLR','PF','iPF','PO','iPO','SM')
      cfg.GRA.W2V_C = 900
    elif imdb.name.startswith('visual_genome'):
      cfg.GRA.W_B = ('000_along.r.01_513444', '001_have.v.01_270403',
                     '002_be.v.01_185338', '003_in.r.01_175449',
                     '004_wear.v.01_100031', '005_behind.r.01_28694',
                     '006_about.r.07_21774', '007_next.r.01_17033',
                     '008_sit.v.01_16717', '009_stand.v.01_13526')
      cfg.GRA.W2V_C= 300
    else:
      cfg.GRA.W_B = ()

    cfg.GRA.LW = len(cfg.GRA.W_B)
    cfg.GRA.LR = len(cfg.GRA.R_B)

  if 'adeseg' in args.imdb_name:
    assert net_tag == 'base'
    memory = seg_memory
    iter_test = True
  elif net_tag == 'base' or net_tag == 'sepbase':
    memory = base_memory
    iter_test = True
  elif net_tag == 'attend' or net_tag == 'sepat':
    memory = attend_memory
    iter_test = False
  elif net_tag == 'check':
    memory = check_memory
    iter_test = False
  elif net_tag == 'gcheck':
    memory = gcheck_memory
    iter_test = False
  elif net_tag == 'gbase' or net_tag == 'sgbase':
    memory = gbase_memory
    iter_test = True
  elif net_tag == 'gattend' or net_tag == 'sepgat' or net_tag == 'noinv':
    memory = gattend_memory
    iter_test = False
  elif net_tag == 'comb':
    memory = comb_memory
    iter_test = False
  else:
    raise NotImplementedError

  # load network
  if net_base == 'vgg16':
    net = memory.vgg16_memory()
  elif net_base == 'res50':
    net = memory.resnetv1_memory(num_layers=50)
  elif net_base == 'res101':
    net = memory.resnetv1_memory(num_layers=101)
  elif net_base == 'res152':
    net = memory.resnetv1_memory(num_layers=152)
  elif net_base == 'mobile':
    net = memory.mobilenetv1_memory()
  else:
    raise NotImplementedError

  # load model
  net.create_architecture("TEST", imdb.num_classes, tag='default')

  if args.model:
    print(('Loading model check point from {:s}').format(args.model))
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    print('Loaded.')
  else:
    print(('Loading initial weights from {:s}').format(args.weight))
    sess.run(tf.global_variables_initializer())
    print('Loaded.')

  test_net(sess, net, imdb, imdb.roidb, filename, args.visualize, iter_test=iter_test)

  sess.close()
