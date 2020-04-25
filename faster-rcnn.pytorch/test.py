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
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms import soft_nms,py_cpu_nms
from model.rpn.bbox_transform import bbox_transform_inv,cpu_bbox_overlaps_batch, \
          bbox_transform_batch,cpu_bbox_overlaps,clip_boxes

from model.utils.net_utils import save_net, load_net, precision_recall, \
        compute_ap,vis_detections,gt_visuals,AverageMeter
from model.faster_rcnn.resnet import resnet

from data_loader.data_load import activity2id
from model.nms.nms import soft_nms,py_cpu_nms,avg_iou

from model.roi_layers import nms
import multiprocessing
multiprocessing.set_start_method('spawn', True)



def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  #loss type
  parser.add_argument('--loss_type',type=str,default='sigmoid',help="""\
      Loss type for training the network ('softmax', 'sigmoid', 'focal').\
      """)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')



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
  parser.add_argument('--uniform_sample', default=False, action="store_true", 
  help='use dense sample for video dataset')
  parser.add_argument('--random_sample', default=False, action="store_true", 
  help='use dense sample for video dataset')
  parser.add_argument('--strided_sample', default=False, action="store_true", 
  help='use dense sample for video dataset')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int) 
# set log and model root folders
  parser.add_argument('--root_log',type=str, default='log')
  parser.add_argument('--root_model', type=str, default='checkpoint')

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')
#visualisation 
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def main():

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "virat":args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 3]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  num_class, args.train_list, args.val_list,args.test_list, train_path, val_path, test_path= dataset.return_dataset(args.dataset,args.modality)
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "/home/malar/git_sam/Action-Proposal-Networks/faster-rcnn.pytorch/cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

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
               class_agnostic=args.class_agnostic,loss_type =args.loss_type)
  
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
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

  input_size =600
  input_mean = [0.485, 0.456, 0.406]
  input_std = [0.229, 0.224, 0.225]
  normalize = GroupNormalize(input_mean, input_std)
  test_loader = torch.utils.data.DataLoader(
        Action_dataset(test_path,num_class, args.test_list, num_segments=args.num_segments,
                   modality=args.modality,random_shift=False, test_mode=True,
                   input_size = input_size,
                   transform=torchvision.transforms.Compose([ 
                       ToTorchFormatTensor(div=1),
                       normalize]),dense_sample =False,uniform_sample=False,
                 random_sample = False,strided_sample = False),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,
        drop_last=True)
  test_iters_per_epoch = int(len(test_loader.dataset) / args.batch_size)

  data_iter = iter(test_loader)
  batch_time = AverageMeter()

  fasterRCNN.eval()
  all_boxes = [[[[]for _ in range(num_class)] for _ in range(args.batch_size * args.num_segments)]
               for _ in range(test_iters_per_epoch)]
  #limit the number of proposal per image across all the class
  max_per_image = 15
  bins = 9 
  score_threshold = 0.1
  tp_labels = np.zeros((num_class,bins),dtype=int)
  fp_labels = np.zeros((num_class,bins),dtype=int)
  fn_labels = np.zeros((num_class,bins),dtype=int)
  all_pred_box,all_gtbb,all_gtlabels,all_scores=[],[],[],[]
  end = time.time()
  with torch.no_grad():
      for step in range (test_iters_per_epoch):
        data = next(data_iter)
        im_data.resize_(data[0].size()).copy_(data[0])
        gt_boxes.resize_(data[1].size()).copy_(data[1])
        num_boxes.resize_(data[2].size()).copy_(data[2])
        im_info.resize_(data[3].size()).copy_(data[3])
        imgsize1 = im_data.size(3)
        imgsize2 = im_data.size(4)
        channel = im_data.size(2)
        im_data = im_data.view(-1,channel,imgsize1,imgsize2)
        im_info = im_info.view(-1,3)
        gt_boxes= gt_boxes.view(-1,15,num_class+4)
        num_boxes = num_boxes.view(-1)
        img_path = data[4]
        rois, cls_prob, bbox_pred = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      

      # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
       #calculate the scores and boxes
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        batch_size = rois.shape[0]
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(batch_size, -1, 4 * num_class)
       #transforms the image to x1,y1,x2,y2, format and clips the coord to images
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 8)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 8)   

       #convert the prediction boxes and gt_boxes to the image size
        gtbb = gt_boxes[:,:,0:4]
        gtlabels = gt_boxes[:,:,4:]
        pred_boxes /= data[3][0][1][2].item()
        gtbb /= data[3][0][1][2].item()
      
       #move the groudtruth to cpu
        gtbb = gtbb.cpu().numpy()
        gtlabels = gtlabels.cpu().numpy()

       #calculate the predictions for each class in an image
        for image in range(pred_boxes.shape[0]):
          for class_id in range(1,num_class):
           inds = torch.nonzero(scores[image,:,class_id]>score_threshold).view(-1)
         # if there is det
           if inds.numel() > 0:
             cls_scores = scores[image,inds,class_id]
             _, order = torch.sort(cls_scores, 0, True)
             cls_boxes = pred_boxes[image,inds, class_id * 4:(class_id + 1) * 4]
             cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
             cls_dets = cls_dets[order,:]
             keep = nms(cls_boxes[order, :], cls_scores[order],cfg.TEST.NMS)
             cls_dets = cls_dets[keep.view(-1)]  
             all_boxes[step][image][class_id] = cls_dets.cpu().numpy()

          if max_per_image > 0:
            image_scores = []
            image_det = []
            for class_id in range(1,num_class):
             if len(all_boxes[step][image][class_id]) !=0:
              #stack the scores to get the number of proposals in an image
               sc = ([all_boxes[step][image][class_id][:, -1]])
               image_scores=np.append(image_scores,sc)
              
            if len(image_scores) > max_per_image:
            #take the image threshold value taking 15th entry  
              image_thresh = np.sort(image_scores)[-max_per_image]
              for class_id in range(1, num_class):
                if len(all_boxes[step][image][class_id]) > 0:
                  keep = np.where(all_boxes[step][image][class_id][:, -1] >= image_thresh)[0]
                  all_boxes[step][image][class_id] = all_boxes[step][image][class_id][keep, :]
                else :
                  all_boxes[step][image][class_id] = []
         #if all_boxes[step][image].any()  
          det = np.asarray(all_boxes[step][image]) 
          if det.size > 0:
            all_boxes[step][image] = avg_iou(all_boxes[step][image],0.7)
        
          #calculate tp_fp_fn
          compute_tp_fp(all_boxes[step][image],gtlabels[image],gtbb[image], \
                      num_class,tp_labels,fp_labels,fn_labels,bins)
        output = ('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(step,(test_iters_per_epoch), batch_time=batch_time))
        print(output)
     
  print(' completed step:{}'.format(step))
  print('tp_labels: {}'.format(tp_labels))
  print('fp_labels: {}'.format(fp_labels))
  print('fn_labels: {}'.format(fn_labels))
  ap = precision_recall(tp_labels,fp_labels,fn_labels,num_class)
  for n in range(1,num_class):
        for key, v in activity2id.items():
          if v == n :
            print(f"Class '{n}' ({key}) - AveragePrecision: {ap[n-1]}")
              
  #print the mean average precision      
  print(f"mAP: {np.mean(ap)}")

def compute_tp_fp(all_boxes,gtlabels,gtbb,num_class,tp_labels,fp_labels,fn_labels,bins):
    bin_dict = {
                  0: lambda x: 1 if x>= 0.1 else 0,
                  1: lambda x: 1 if x>= 0.2 else 0,
                  2: lambda x: 1 if x>= 0.3 else 0,
                  3: lambda x: 1 if x>= 0.4 else 0,
                  4: lambda x: 1 if x>= 0.5 else 0,
                  5: lambda x: 1 if x>= 0.6 else 0,
                  6: lambda x: 1 if x>= 0.7 else 0,
                  7: lambda x: 1 if x>= 0.8 else 0,
                  8: lambda x: 1 if x>= 0.9 else 0
                }
    
    #get the non -zero groundtruth for that image
    index=np.unique(np.nonzero(gtbb)[0])
    gtbox = gtbb[index,:]       #[gt,4]
    gtlabel = gtlabels[index,:] #[gt,40]
    all_boxes = np.asarray(all_boxes)
    

    # compute tp fp and fn for all class in that image 
    #change---------------
    #for class_id in range(1,num_class):

      #keep track of propposal and gt
    is_gt_detected =[] 
    is_proposal_labeled =[]
      
      #get the detections and groudtruths for each class
      #all_detectbb = all_boxes[class_id]
    pred_bb = all_boxes[:,:4]
    pred_label = all_boxes[:,4:]
      #labels = gtlabel[:,class_id]
      
      #variables for the match table
    num_boxes_per_img = gtbox.shape[0]
    num_proposal = pred_bb.shape[0]
      
      #check if there is a detection for that class
    if all_boxes.any() !=0:
       overlaps = cpu_bbox_overlaps(pred_bb, gtbox)
       
       #if there is a detection and gt 
       if (num_proposal>0) and (num_boxes_per_img !=0):
            match_table = np.zeros((num_proposal,num_boxes_per_img),dtype=float)
            for p in range(num_proposal):
                for g in range(num_boxes_per_img):
                    if overlaps[p][g] >= 0.5:
                        match_table[p,g] = overlaps[p][g]
                    
            best_match = match_table.max()
            while best_match >= 0.5:
                match_pos = np.where(match_table == best_match)
                p = match_pos[0][0]
                detectclass = pred_label[p]
                g = match_pos[1][0]
                gtclass = gtlabel[g]
                is_gt_detected += [g]
                is_proposal_labeled+=[p]
                for class_id in range(1,num_class):
                  gt_label = gtclass[class_id]
                  detect_label = detectclass[class_id]
                  if gt_label == 1:
                  #if there is groundtruth and pred_score matches with the bin = tp
                   for bin_id in range(bins):
                    bin_value=  bin_dict[bin_id](detect_label)
                    if bin_value == 1 : 
                     tp_labels[class_id][bin_id] += 1
                
                    else: #there is a gt but pred_scores doesnt match with bin = fn
                     fn_labels[class_id][bin_id] +=1
                
                  else: #no gt but pred_scores matches bin = fp
                   for bin_id in range(bins):
                    bin_value=  bin_dict[bin_id](detect_label)
                    if bin_value == 1 : 
                      fp_labels[class_id][bin_id] +=1 
                
                match_table[p,:] = np.zeros(match_table[p,:].shape)
                match_table[:,g] = np.zeros(match_table[:,g].shape)

                best_match = match_table.max()   
            
            #count the left out proposals as fp      
            for k in range(num_proposal): 
             if k not in is_proposal_labeled:
               for class_id in range(1,num_class):
                  detect_label = pred_label[k][class_id]
                  for bin_id in range(bins):
                      bin_value=  bin_dict[bin_id](detect_label)
                      if bin_value == 1 : 
                         fp_labels[class_id][bin_id] +=1   
                   
            
           
            #count the left out groundtruths as fn
            for gt in range(num_boxes_per_img):
                if gt not in is_gt_detected:
                  for class_id in range(1,num_class):
                    ground_label = gtlabel[gt][class_id]
                    if ground_label == 1:                    
                      fn_labels[class_id,:] += 1   
      #for no detections for that class  
    else:
       if gtlabel.any() != 0:
        for gt in range(num_boxes_per_img):
          for class_id in range(1,num_class):
                    ground_label = gtlabel[gt][class_id]
                    if ground_label == 1:                    
                      fn_labels[class_id,:] += 1  

  
if __name__ == '__main__':
   main() 