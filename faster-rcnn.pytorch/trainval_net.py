# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision
from statistics import mean 

from torch.utils.data.sampler import Sampler
from data_loader.data_load import Action_dataset
from dataset_config import dataset
from dataset_config.ava_dataset import Ava
from dataset_config.transforms import *
from tensorboardX import SummaryWriter

import multiprocessing
multiprocessing.set_start_method('spawn', True)
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient,precision_recall, \
        compute_ap,vis_detections,gt_visuals,AverageMeter
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

from model.nms.nms import soft_nms,py_cpu_nms,avg_iou

from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv,cpu_bbox_overlaps_batch, \
          bbox_transform_batch,cpu_bbox_overlaps,clip_boxes

from data_loader.data_load import activity2id
from model.faster_rcnn.resnet import resnet

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
  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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

#loss type
  parser.add_argument('--loss_type',type=str,default='sigmoid',help="""\
      Loss type for training the network ('softmax', 'sigmoid', 'focal').\
      """)

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

def main():
  args = parse_args()
  print('Called with args:')
  print(args)

  best_meanap =0
  meanap = 0
  if args.dataset == "virat":args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 3]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  if args.dataset == "ava":args.set_cfgs = ['ANCHOR_SCALES', '[2,4,6]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  #args.store_name = '_'.join(
  #      ['Action_proposal', args.dataset, args.modality, args.net, 'segment%d' % args.num_segments,
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
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  
  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.USE_GPU_NMS = args.cuda

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
          os.makedirs(output_dir)
  
  #visualisation 
  vis = args.vis
  #dataloader
  input_size =600
  input_mean = [0.485, 0.456, 0.406]
  input_std = [0.229, 0.224, 0.225]
  normalize = GroupNormalize(input_mean, input_std)

  if args.dataset == 'virat':
    train_loader = torch.utils.data.DataLoader(
        Action_dataset(train_path,num_class, args.train_list, num_segments=args.num_segments,
                   modality=args.modality,random_shift=True, test_mode=False,
                   input_size = input_size,
                   transform=torchvision.transforms.Compose([ 
                       ToTorchFormatTensor(div=1),
                       normalize]),dense_sample =args.dense_sample,uniform_sample=args.uniform_sample,
                 random_sample = args.random_sample,strided_sample = args.strided_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU
  
    val_loader = torch.utils.data.DataLoader(
        Action_dataset(val_path, num_class,args.val_list, num_segments=args.num_segments,
                   modality=args.modality,random_shift=True, test_mode=False,
                   input_size = input_size,
                   transform=torchvision.transforms.Compose([ 
                       ToTorchFormatTensor(div=1),
                       normalize]),dense_sample =args.dense_sample,uniform_sample=args.uniform_sample,
                 random_sample = args.random_sample,strided_sample = args.strided_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True) 

    test_loader = torch.utils.data.DataLoader(
        Action_dataset(test_path,num_class, args.test_list, num_segments=args.num_segments,
                   modality=args.modality,random_shift=False, test_mode=True,
                   input_size = input_size,
                   transform=torchvision.transforms.Compose([ 
                       ToTorchFormatTensor(div=1),
                       normalize]),dense_sample =False,uniform_sample=False,
                 random_sample = False,strided_sample = False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
  if args.dataset == 'ava':
    train_loader = torch.utils.data.DataLoader(Ava(cfg,split = 'train'))
          # prevent something not % n_GPU
  if args.cuda:
    cfg.CUDA = True

# initilize the network here.
  if args.net == 'vgg16':
   fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)  
  elif args.net == 'res101':
   fasterRCNN = resnet(imdb.classes, 101,modality=args.modality ,pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
   base_model ='resnet50'
   fasterRCNN = resnet(num_class,modality = 'RGB',num_layers =50, base_model ='resnet50', n_segments =8,
               n_div =args.shift_div , place = args.shift_place,temporal_pool = args.temporal_pool,
               pretrain = args.pretrain,shift = args.shift,
               class_agnostic=args.class_agnostic,loss_type =args.loss_type)
  elif args.net == 'res152':
   fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
   print("network is not defined")
   pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  BASE_LEARNING_RATE = lr
  
  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN.cuda()
  
  #define optimizer

  if args.optimizer == "adam":
     lr = lr * 0.1
     optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
     optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
     
  ## adding temporal shift pretrained kinetics weights

  if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = fasterRCNN.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict:
                replace_dict.append((k.replace('module.base_model.conv1',
                'RCNN_base.0').replace('module.base_model.bn1','RCNN_base.1')
                .replace('module.base_model.layer1.0','RCNN_base.4.0')
                .replace('module.base_model.layer1.1','RCNN_base.4.1')
                .replace('module.base_model.layer1.2','RCNN_base.4.2')
                .replace('module.base_model.layer2.0','RCNN_base.5.0')
                .replace('module.base_model.layer2.1','RCNN_base.5.1')
                .replace('module.base_model.layer2.2','RCNN_base.5.2')
                .replace('module.base_model.layer2.3','RCNN_base.5.3')
                .replace('module.base_model.layer3.0','RCNN_base.6.0')
                .replace('module.base_model.layer3.1','RCNN_base.6.1')
                .replace('module.base_model.layer3.2','RCNN_base.6.2')
                .replace('module.base_model.layer3.3','RCNN_base.6.3')
                .replace('module.base_model.layer3.4','RCNN_base.6.4')
                .replace('module.base_model.layer3.5','RCNN_base.6.5')
                .replace('module.base_model.layer4.0.conv1.net.weight','RCNN_top.0.0.conv1.weight')
                .replace('module.base_model.layer4.1.conv1.net','RCNN_top.0.1.conv1')
                .replace('module.base_model.layer4.2.conv1.net','RCNN_top.0.2.conv1')
                .replace('module.base_model.layer4.0.','RCNN_top.0.0.')
                .replace('module.base_model.layer4.1','RCNN_top.0.1')
                .replace('module.base_model.layer4.2','RCNN_top.0.2'),k))
                 
        for k_new, k in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
      
        model_dict.update(sd)
        fasterRCNN.load_state_dict(model_dict)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  
  if args.mGPUs:
    mGPUs = True
    fasterRCNN = nn.DataParallel(fasterRCNN)
  else:
    mGPUs = False
  
  train_iters_per_epoch = int(len(train_loader.dataset) / args.batch_size)
  val_iters_per_epoch = int(len(val_loader.dataset) / args.batch_size)
  test_iters_per_epoch = int(len(test_loader.dataset) / args.batch_size)
  session = args.session
  
  LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5)]

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")
  
  if args.evaluate:
    meanprec = validate(val_loader, fasterRCNN,0,val_iters_per_epoch,num_class, \
            args.num_segments,vis,session,args.batch_size)
    
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    
    
    
    '''warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
    
    if epoch <= warmup_end_epoch :
      startepoch = epoch - 1
      warmupepoch = startepoch + float(args.batch_size) / train_iters_per_epoch
      initial_decay = warmup_lr_multiplier * warmupepoch / warmup_end_epoch
      lr = BASE_LEARNING_RATE * initial_decay
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if epoch == warmup_end_epoch+1:
      lr = BASE_LEARNING_RATE
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr '''

    
    if epoch % (args.lr_decay_step + 1) == 0:
    #if meanap < best_meanap:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma
    #best_meanap = max(best_meanap,meanap) 
    train(train_loader, fasterRCNN,lr,optimizer,
    epoch,num_class,train_iters_per_epoch,session,mGPUs,logger,output_dir)
    
    # evaluate on validation set
    if epoch % 3 == 0:
      meanap = validate(val_loader, fasterRCNN,epoch,val_iters_per_epoch,num_class, \
              args.num_segments,vis,session,args.batch_size)
      is_best = meanap > best_meanap
      best_meanap = max(best_meanap,meanap)
      print(f"best mean ap : [{best_meanap}]")
    if epoch % 6 == 0:
       test_ap = validate(test_loader, fasterRCNN,epoch,test_iters_per_epoch,num_class, \
              args.num_segments,vis,session,args.batch_size)

def train(train_loader,fasterRCNN,lr,optimizer,epoch,num_class,train_iters_per_epoch,session,mGPUs,logger,output_dir):
       
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)


    batch_time = AverageMeter()
    data_time = AverageMeter()

    # setting to train mode
    fasterRCNN.train()
    #loss = 0
    loss_temp = 0
    end = time.time()
    data_iter = iter(train_loader)
    optimizer.zero_grad()
    fasterRCNN.zero_grad()
    for step in range (train_iters_per_epoch):
        data = next(data_iter)
        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
          im_data.resize_(data[0].size()).copy_(data[0])
          gt_boxes.resize_(data[1].size()).copy_(data[1])
          num_boxes.resize_(data[2].size()).copy_(data[2])
          im_info.resize_(data[3].size()).copy_(data[3])
        im_data = im_data.view(-1,im_data.size(2),im_data.size(3),im_data.size(4))
        im_info = im_info.view(-1,3)
        gt_boxes= gt_boxes.view(-1,15,num_class+4)
        num_boxes = num_boxes.view(-1)
        
        #fasterRCNN.zero_grad()
        # compute output
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data,im_info,gt_boxes,num_boxes)
         
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()
         
        # backward
        #normalise the loss i.e average the accumuated loss
        loss = loss / 64                   
        loss.backward()


        if (step+1)%64 == 0:   #wait for several backward step
          optimizer.step()     #update the model 
          fasterRCNN.zero_grad()#Reset the gradient tensor

        #if step > 0:
        #loss_temp /= (100 + 1)

        if mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (session, epoch, step, train_iters_per_epoch, loss, lr))
        output = ('\t\t\t Batch_Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data_Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      .format(batch_time=batch_time,data_time=data_time))
        print(output)
        print("\t\t\tfg/bg=(%d/%d)" % (fg_cnt, bg_cnt))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        #if args.use_tfboard:
        info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
        }
        logger.add_scalars("logs_s_{}/losses".format(session), info, (epoch - 1) * train_iters_per_epoch + step)
        loss_temp = 0
        logger.close()
    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(session, epoch, step))
    save_checkpoint({
      'session': session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      #'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))
        

def validate(val_loader,fasterRCNN,epoch,val_iters_per_epoch,
              num_class,num_segments,vis,session,batch_size):
    
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    #if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    data_iter = iter(val_loader)

    batch_time = AverageMeter()

    fasterRCNN.eval()
    all_boxes = [[[[]for _ in range(num_class)] for _ in range(batch_size *num_segments)]
               for _ in range(val_iters_per_epoch)]
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
      for step in range (val_iters_per_epoch):
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
        #batch_size = rois.shape[0]
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(scores.shape[0], -1, 4 * num_class)
        
       #transforms the image to x1,y1,x2,y2, format and clips the coord to images
        pred_boxes = bbox_transform_inv(boxes, box_deltas,scores.shape[0])
        pred_boxes = clip_boxes(pred_boxes, im_info.data,scores.shape[0])   

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
                .format(step,(val_iters_per_epoch), batch_time=batch_time))
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
    print(f"mAP for epoch [{epoch}]: {mean(ap)}")
    return np.mean(ap)

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
