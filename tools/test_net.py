from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net_base
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1


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
  parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
  parser.add_argument('--visualize', dest='visualize', help='whether to show results',
                        action='store_true')
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
  # load network
  if args.net == 'vgg16':
    net = vgg16()
  elif args.net == 'res50':
    net = resnetv1(num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(num_layers=152)
  elif args.net == 'mobile':
    net = mobilenetv1()
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

  test_net_base(sess, net, imdb, imdb.roidb, filename, args.visualize)

  sess.close()
