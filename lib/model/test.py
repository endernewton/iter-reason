from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os

from model.config import cfg, get_output_dir
from utils.timer import Timer
from utils.cython_bbox import bbox_overlaps
from utils.visualization import draw_predicted_boxes_test


def im_detect(sess, imdb, net, roidb):
  blobs = imdb.minibatch(roidb, is_training=False)
  _, scores = net.test_image(sess, blobs)

  return scores, blobs

def im_detect_iter(sess, imdb, net, roidb, iter):
  blobs = imdb.minibatch(roidb, is_training=False)
  _, scores = net.test_image_iter(sess, blobs, iter)

  return scores, blobs

def test_net(sess, net, imdb, roidb, weights_filename, visualize=False, iter_test=False):
  if iter_test:
    for iter in xrange(cfg.MEM.ITER):
      test_net_memory(sess, net, imdb, roidb, weights_filename, visualize, iter)
  else:
    test_net_base(sess, net, imdb, roidb, weights_filename, visualize)

def test_net_base(sess, net, imdb, roidb, weights_filename, visualize=False):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(roidb)
  output_dir = get_output_dir(imdb, weights_filename)
  output_dir_image = os.path.join(output_dir, 'images')
  if visualize and not os.path.exists(output_dir_image):
    os.makedirs(output_dir_image)
  all_scores = [[] for _ in range(num_images)]

  # timers
  _t = {'score' : Timer()}

  for i in range(num_images):
    _t['score'].tic()
    all_scores[i], blobs = im_detect(sess, imdb, net, [roidb[i]])
    _t['score'].toc()

    print('score: {:d}/{:d} {:.3f}s' \
        .format(i + 1, num_images, _t['score'].average_time))

    if visualize and i % 10 == 0:
      basename = os.path.basename(imdb.image_path_at(i)).split('.')[0]
      im_vis, wrong = draw_predicted_boxes_test(blobs['data'], all_scores[i], blobs['gt_boxes'])
      if wrong:
        out_image = os.path.join(output_dir_image, basename + '.jpg')
        print(out_image)
        cv2.imwrite(out_image, im_vis)

  res_file = os.path.join(output_dir, 'results.pkl')
  with open(res_file, 'wb') as f:
    pickle.dump(all_scores, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  mcls_sc, mcls_ac, mcls_ap, mins_sc, mins_ac, mins_ap = imdb.evaluate(all_scores, output_dir)
  eval_file = os.path.join(output_dir, 'results.txt')
  with open(eval_file, 'w') as f:
    f.write('{:.3f} {:.3f} {:.3f} {:.3f}'.format(mins_ap, mins_ac, mcls_ap, mcls_ac))

def test_net_memory(sess, net, imdb, roidb, weights_filename, visualize=False, iter=0):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(roidb)
  output_dir = get_output_dir(imdb, weights_filename + "_iter%02d" % iter)
  output_dir_image = os.path.join(output_dir, 'images')
  if visualize and not os.path.exists(output_dir_image):
    os.makedirs(output_dir_image)
  all_scores = [[] for _ in range(num_images)]

  # timers
  _t = {'score' : Timer()}

  for i in range(num_images):
    _t['score'].tic()
    all_scores[i], blobs = im_detect_iter(sess, imdb, net, [roidb[i]], iter)
    _t['score'].toc()

    print('score: {:d}/{:d} {:.3f}s' \
        .format(i + 1, num_images, _t['score'].average_time))

    if visualize and i % 10 == 0:
      basename = os.path.basename(imdb.image_path_at(i)).split('.')[0]
      im_vis, wrong = draw_predicted_boxes_test(blobs['data'], all_scores[i], blobs['gt_boxes'])
      if wrong:
        out_image = os.path.join(output_dir_image, basename + '.jpg')
        print(out_image)
        cv2.imwrite(out_image, im_vis)

  res_file = os.path.join(output_dir, 'results.pkl')
  with open(res_file, 'wb') as f:
    pickle.dump(all_scores, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  mcls_sc, mcls_ac, mcls_ap, mins_sc, mins_ac, mins_ap = imdb.evaluate(all_scores, output_dir)
  eval_file = os.path.join(output_dir, 'results.txt')
  with open(eval_file, 'w') as f:
    f.write('{:.3f} {:.3f} {:.3f} {:.3f}'.format(mins_ap, mins_ac, mcls_ap, mcls_ac))

