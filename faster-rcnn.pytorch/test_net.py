# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

from data_loader.data_load import Action_dataset
from data_loader.data_load import *
from dataset_config import dataset
from dataset_config.transforms import *

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
#from roi_data_layer.roidb import combined_roidb
#from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
#from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import multiprocessing
multiprocessing.set_start_method('spawn', True)

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)

  #temporal shift module
  parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
  parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
  parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')
  parser.add_argument('--pretrain', type=str, default='imagenet')
  parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
  parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')
  parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# dataset arguments from temporal segment networks
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='virat', type=str)
  parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
  parser.add_argument('--train_list', type=str, default="")
  parser.add_argument('--val_list', type=str, default="")
  parser.add_argument('--num_segments', type=int, default=3)
  parser.add_argument('--dense_sample', default=False, action="store_true", 
  help='use dense sample for video dataset')
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="detections",
                      type=str)

  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "virat":args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  #on_proposal', args.dataset, args.modality, args.net, 'segment%d' % args.num_segments,
  #       'e{}'.format(args.max_epochs)])
  #if args.pretrain != 'imagenet':
  #      args.store_name += '_{}'.format(args.pretrain)
  #print('storing name: ' + args.store_name)
  
  num_class, args.train_list, args.val_list,args.test_list, train_path, val_path, test_path= dataset.return_dataset(args.dataset,args.modality)
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "/home/malar/git_sam/Action-Proposal-Networks/faster-rcnn.pytorch/cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  #imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  #imdb.competition_mode(on=True)

  #print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    #fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN = resnet(num_class,modality = 'RGB',num_layers =50, base_model ='resnet50', n_segments =8,
               n_div =args.shift_div , place = args.shift_place,temporal_pool = args.temporal_pool,
               pretrain = args.pretrain,shift = args.shift,
               class_agnostic=args.class_agnostic)
  
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  '''replace_dict = []
  sd = checkpoint['model']
  for k, v in checkpoint['model'].items():
                replace_dict.append((k.replace('RCNN_base.4.0.conv1.net.weight',
                'RCNN_base.4.0.conv1.weight')
                .replace('RCNN_base.4.1.conv1.net.weight','RCNN_base.4.1.conv1.weight')
                .replace('RCNN_base.4.2.conv1.net.weight','RCNN_base.4.2.conv1.weight')
                .replace('RCNN_base.5.0.conv1.net.weight','RCNN_base.5.0.conv1.weight')
                .replace('RCNN_base.5.1.conv1.net.weight','RCNN_base.5.1.conv1.weight') 
                .replace('RCNN_base.5.2.conv1.net.weight','RCNN_base.5.2.conv1.weight')
                .replace('RCNN_base.5.3.conv1.net.weight','RCNN_base.5.3.conv1.weight')
                .replace('RCNN_base.6.0.conv1.net.weight','RCNN_base.6.0.conv1.weight')
                .replace('RCNN_base.6.1.conv1.net.weight','RCNN_base.6.1.conv1.weight')
                .replace('RCNN_base.6.2.conv1.net.weight','RCNN_base.6.2.conv1.weight')
                .replace('RCNN_base.6.3.conv1.net.weight','RCNN_base.6.3.conv1.weight')
                .replace('RCNN_base.6.4.conv1.net.weight','RCNN_base.6.4.conv1.weight')
                .replace('RCNN_base.6.5.conv1.net.weight','RCNN_base.6.5.conv1.weight'),k))
  for k_new, k in replace_dict:
            sd[k_new] = sd.pop(k)
  '''
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)


  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 15

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  input_size =600
  input_mean = [0.485, 0.456, 0.406]
  input_std = [0.229, 0.224, 0.225]
  normalize = GroupNormalize(input_mean, input_std)
  test_loader = torch.utils.data.DataLoader(
        Action_dataset(test_path, args.test_list, num_segments=args.num_segments,
                   modality=args.modality,random_shift=False, test_mode=True,
                   input_size = input_size,
                   transform=torchvision.transforms.Compose([ 
                       ToTorchFormatTensor(div=1),
                       normalize]),dense_sample =False,uniform_sample=False,
                 random_sample = False,strided_sample = False),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,
        drop_last=True)

  save_name = 'faster_rcnn_10'
  num_sequence = len(test_loader)
  #all_boxes = [[[] for _ in xrange(num_images)]
  #             for _ in xrange(imdb.num_classes)]
  all_boxes = [[[[] for _ in xrange(args.num_segments)] for _ in xrange(num_sequence)]for _ in xrange(num_class)]
      
  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
          os.makedirs(output_dir)
  #output_dir = get_output_dir(args.dataset, save_name)

  #dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
  #                      imdb.num_classes, training=False, normalize = False)
  #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
  #                          shuffle=False, num_workers=0,
  #                          pin_memory=True)

  data_iter = iter(test_loader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_sequence):

      data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              gt_boxes.resize_(data[1].size()).copy_(data[1])
              num_boxes.resize_(data[2].size()).copy_(data[2])
              im_info.resize_(data[3].size()).copy_(data[3])
              img_path = data[4]
              imgsize1 = im_data.size(3)
              imgsize2 = im_data.size(4)
              channel = im_data.size(2)
              im_data = im_data.view(-1,channel,imgsize1,imgsize2)
              im_info = im_info.view(-1,3)
              gt_boxes= gt_boxes.view(-1,30,44)
              num_boxes = num_boxes.view(-1)
        
      detect_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/" + "final_detections"
      if not os.path.exists(detect_dir):
          os.makedirs(detect_dir)
 
      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      #scores = scores.view(-1,40)
      boxes = rois.data[:, :, 1:5]
      boxes = boxes.view(-1, 4).expand(1,-1,4).contiguous()

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * num_class)

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))
      
      
      pred_boxes /= data[3][0][1][2].item()   #make it more efficient

      scores = scores.squeeze()
      rolled_shape = pred_boxes.size(1) // 8
      pred_boxes = pred_boxes.view(8,rolled_shape,4*num_class)
      #pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
       for frame in xrange(0,args.num_segments):
        str_path = ' '.join([str(elem) for elem in img_path[frame]]) 
        im = cv2.imread(str_path)
        im2show = np.copy(im)
        for j in xrange(1, num_class):
          inds = torch.nonzero(scores[frame,:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[frame,:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[frame,inds, :]
            else:
              cls_boxes = pred_boxes[frame,inds, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              for activity, value in activity2id.items(): 
                  if value == j:
                    im2show = vis_detections(im2show, activity, cls_dets.cpu().numpy(), 0.5)
            all_boxes[j][i][frame] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i][frame] = empty_array

      # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][frame][:, -1]
                                    for j in xrange(1, num_class)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, num_class):
                  keep = np.where(all_boxes[j][i][frame][:, -1] >= image_thresh)[0]
                  all_boxes[j][i][frame] = all_boxes[j][i][frame][keep, :]
                
        if (frame == 0):
          os.chdir(detect_dir)
          detect_seq_dir = "final_detections" + str(i)
          if not os.path.exists(detect_seq_dir):
             os.makedirs(detect_seq_dir)
          os.chdir(detect_seq_dir)
 
        image_file=str(str_path).strip().split('/frames/')[1]
        frame_num= str(image_file).split(".")[0]
        result = 'pred_{:06d}.png'.format(int(frame_num))
        cv2.imwrite(result, im2show)
        #pdb.set_trace()

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_sequence, detect_time, nms_time))
      sys.stdout.flush()


  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
