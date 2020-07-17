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
from args import parse_args
import pprint
import pdb
import time
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision

from operator import itemgetter

from torch.utils.data.sampler import Sampler
from data_loader.data_loader import *

from torch.utils.data._utils.collate import default_collate

import multiprocessing
multiprocessing.set_start_method('spawn', True)
from model.utils.config import cfg,cfg_from_list,cfg_from_file
from model.utils.net_utils import *
from model.utils.confusion_matrix import *

from model.nms.nms import soft_nms,py_cpu_nms,avg_iou

from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv,cpu_bbox_overlaps_batch, \
          bbox_transform_batch,cpu_bbox_overlaps,clip_boxes

#from dataset_config.virat.data_load import activity2id,activity2id_hard
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16

def main():
  args = parse_args()
  print('Called with args:')
  print(args)
  
  best_meanap =0
  meanap = 0
  if args.dataset == "virat":args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 3]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '15']
  if args.dataset == "ava":args.set_cfgs = ['ANCHOR_SCALES', '[2 , 4, 6]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '15']
  if args.dataset == "ucfsport":args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,24,28]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '2']
  if args.dataset == "jhmdb":args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16,24,28]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '1']
  if args.dataset == "ucf24":args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,24]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '1']
  args.cfg_file = "cfgs/{}.yml".format(args.net)
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
  
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  
  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.USE_GPU_NMS = args.cuda
  if args.dataset == 'virat':
      num_class = cfg.VIRAT.NUM_CLASS
      output_dir = cfg.VIRAT.output_model_dir + "/" + args.net + "/" + args.dataset
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
  elif args.dataset =='ava':
      num_class = cfg.AVA.NUM_CLASSES
      output_dir = cfg.AVA.output_model_dir + "/" + args.net + "/" + args.dataset
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
  elif args.dataset =='ucfsport':
      num_class = cfg.UCFSPORT.NUM_CLASSES
      output_dir = cfg.UCFSPORT.output_model_dir + "/" + args.net + "/" + args.dataset
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
  elif args.dataset =='jhmdb':
      num_class = cfg.JHMDB.NUM_CLASSES
      output_dir = cfg.JHMDB.output_model_dir + "/" + args.net + "/" + args.dataset
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
  elif args.dataset =='ucf24':
      num_class = cfg.UCF24.NUM_CLASSES
      output_dir = cfg.UCF24.output_model_dir + "/" + args.net + "/" + args.dataset
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
  else:
    print("dataset is not defined ")
  
  #visualisation 
  vis = args.vis
  
  #log initialisation
  args.store_name = '_'.join(
        ['ACT_D', args.dataset,args.net,'segment%d' % args.num_segments,'e{}'.format(args.max_epochs),'session%d'%args.session])
  check_rootfolders(args.store_name,args.dataset)
  
  #logging
  log_training,logger = log_info(cfg,args.store_name,args.dataset,args= args)
  
  #dataloader

  train_loader = construct_loader(cfg,dataset = args.dataset,num_segments = args.num_segments,
        batch_size = args.batch_size,split = 'train',input_sampling = True,split_num=args.splits)
  val_loader = construct_loader(cfg,dataset = args.dataset,num_segments = args.num_segments,
        batch_size = args.batch_size,split = 'val',input_sampling = True,split_num=args.splits)
  if args.dataset == 'virat':
    test_loader = construct_loader(cfg,dataset = args.dataset,num_segments = args.num_segments,
        batch_size = args.batch_size,split = 'test',input_sampling = True,split_num=args.splits)
      
  #initialise meter here for AVA
  if args.dataset == 'ava':
      ava_val_meter = AVAMeter(len(val_loader), cfg, mode="val",
                             num_seg =args.num_segments)
  else : 
    ava_val_meter = None
          
      
  # prevent something not % n_GPU
  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(num_class,pretrained=True,class_agnostic=args.class_agnostic,
    loss_type = args.loss_type)
  elif args.net == 'res50':
   fasterRCNN = resnet(num_class,num_layers =50, base_model ='resnet50', n_segments =args.num_segments,
               n_div =args.shift_div , place = args.shift_place,
               pretrain = args.pretrain,shift = args.shift,
               class_agnostic=args.class_agnostic,loss_type =args.loss_type)
  
  else:
   print("network is not defined")
   pdb.set_trace()

  fasterRCNN.create_architecture()
  print("blocks fixed:%d"%(cfg.RESNET.FIXED_BLOCKS))
  
  lr = args.lr
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
                
                #.replace('module.base_model.layer1.0.conv1.net','RCNN_base.4.0.conv1')
                #.replace('module.base_model.layer1.0.conv2','RCNN_base.4.0.conv2.net')
                .replace('module.base_model.layer1.0','RCNN_base.4.0')
                
                #.replace('module.base_model.layer1.1.conv1.net','RCNN_base.4.1.conv1')
                #.replace('module.base_model.layer1.1.conv2','RCNN_base.4.1.conv2.net')
                .replace('module.base_model.layer1.1','RCNN_base.4.1')
                
                #.replace('module.base_model.layer1.2.conv1.net','RCNN_base.4.2.conv1')
                #.replace('module.base_model.layer1.2.conv2','RCNN_base.4.2.conv2.net')
                .replace('module.base_model.layer1.2','RCNN_base.4.2')

                
                #.replace('module.base_model.layer2.0.conv1.net','RCNN_base.5.0.conv1')
                #.replace('module.base_model.layer2.0.conv2','RCNN_base.5.0.conv2.net')
                .replace('module.base_model.layer2.0','RCNN_base.5.0')

                
                #.replace('module.base_model.layer2.1.conv1.net','RCNN_base.5.1.conv1')
                #.replace('module.base_model.layer2.1.conv2','RCNN_base.5.1.conv2.net')
                .replace('module.base_model.layer2.1','RCNN_base.5.1')
                   
                #.replace('module.base_model.layer2.2.conv1.net','RCNN_base.5.2.conv1')
                #.replace('module.base_model.layer2.2.conv2','RCNN_base.5.2.conv2.net') 
                .replace('module.base_model.layer2.2','RCNN_base.5.2')

                #.replace('module.base_model.layer2.3.conv1.net','RCNN_base.5.3.conv1')
                #.replace('module.base_model.layer2.3.conv2','RCNN_base.5.3.conv2.net')
                .replace('module.base_model.layer2.3','RCNN_base.5.3')

                
                #.replace('module.base_model.layer3.0.conv1.net','RCNN_base.6.0.conv1')
                #.replace('module.base_model.layer3.0.conv2','RCNN_base.6.0.conv2.net')
                .replace('module.base_model.layer3.0','RCNN_base.6.0')
                
                #.replace('module.base_model.layer3.1.conv1.net','RCNN_base.6.1.conv1')
                #.replace('module.base_model.layer3.1.conv2','RCNN_base.6.1.conv2.net')
                .replace('module.base_model.layer3.1','RCNN_base.6.1')
                
                #.replace('module.base_model.layer3.2.conv1.net','RCNN_base.6.2.conv1')
                #.replace('module.base_model.layer3.2.conv2','RCNN_base.6.2.conv2.net')
                .replace('module.base_model.layer3.2','RCNN_base.6.2')
                
                #.replace('module.base_model.layer3.3.conv1.net','RCNN_base.6.3.conv1')
                #.replace('module.base_model.layer3.3.conv2','RCNN_base.6.3.conv2.net')
                .replace('module.base_model.layer3.3','RCNN_base.6.3')
                
                #.replace('module.base_model.layer3.4.conv1.net','RCNN_base.6.4.conv1')
                #.replace('module.base_model.layer3.4.conv2','RCNN_base.6.4.conv2.net')
                .replace('module.base_model.layer3.4','RCNN_base.6.4')
                
                #.replace('module.base_model.layer3.5.conv1.net','RCNN_base.6.5.conv1')
                #.replace('module.base_model.layer3.5.conv2','RCNN_base.6.5.conv2.net')
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
  
  session = args.session
  
  LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5)]

  # initilize the tensor holder here.
  im_data = torch.cuda.FloatTensor(1)
  im_info = torch.cuda.FloatTensor(1)
  num_boxes = torch.cuda.LongTensor(1)
  gt_boxes = torch.cuda.FloatTensor(1)

  #input variable
  input_data = im_data,im_info,num_boxes,gt_boxes
  
  if args.evaluate:
    validate_voc(val_loader, fasterRCNN,args.start_epoch,num_class, \
             args.num_segments,vis,session,args.batch_size,input_data,\
             cfg,log_training,ava_val_meter,args.dataset)
    
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
    
    
    if args.dataset == 'virat':
      #dataloader 
      train(train_loader, fasterRCNN,lr,optimizer,
        epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
        input_data,cfg,args.acc_step,log_training)
      ava_val_meter = None
      
      # evaluate on validation set
      validate_virat(val_loader, fasterRCNN,epoch,num_class, \
      args.num_segments,vis,session,args.batch_size,input_data,cfg,\
      log_training,ava_val_meter,args.dataset)
      
      if epoch % 10== 0:
      # test_ap = 
        validate_virat(test_loader, fasterRCNN,epoch,num_class, \
              args.num_segments,vis,session,args.batch_size,input_data,cfg,
              log_training,ava_val_meter,args.dataset)
    
    elif args.dataset == 'ava':
          
      train(train_loader, fasterRCNN,lr,optimizer,
          epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
          input_data,cfg,args.acc_step,log_training)
      
      if epoch % 2== 0:
        validate(val_loader, fasterRCNN,epoch,num_class, \
              args.num_segments,vis,session,args.batch_size,input_data,
              cfg,log_training,ava_val_meter,args.dataset)
    
    elif args.dataset == 'ucfsport':
      
      ava_val_meter = None
      train(train_loader, fasterRCNN,lr,optimizer,
            epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
            input_data,cfg,args.acc_step,log_training)
      validate_voc(val_loader, fasterRCNN,epoch,num_class, \
              args.num_segments,vis,session,args.batch_size,input_data,
              cfg,log_training,ava_val_meter,args.dataset)
    
    elif args.dataset == 'jhmdb':
      
      ava_val_meter = None
      train(train_loader, fasterRCNN,lr,optimizer,
            epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
            input_data,cfg,args.acc_step,log_training)
      validate_voc(val_loader, fasterRCNN,epoch,num_class, \
              args.num_segments,vis,session,args.batch_size,input_data,
              cfg,log_training,ava_val_meter,args.dataset)
    
    elif args.dataset == 'ucf24':
      
      ava_val_meter = None
      train(train_loader, fasterRCNN,lr,optimizer,
            epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
            input_data,cfg,args.acc_step,log_training)
      validate_voc(val_loader, fasterRCNN,epoch,num_class, \
              args.num_segments,vis,session,args.batch_size,input_data,
              cfg,log_training,ava_val_meter,args.dataset)
 
 
def train(train_loader,fasterRCNN,lr,optimizer,epoch,num_class,batch_size,session,mGPUs,
          logger,output_dir,input_data,cfg,acc_step,log):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # setting to train mode
    fasterRCNN.train()
    #loss = 0
    loss_temp = 0
    end = time.time()
    im_data,im_info,num_boxes,gt_boxes = input_data
    train_iters_per_epoch = int(len(train_loader.dataset) / batch_size)
    optimizer.zero_grad()
    fasterRCNN.zero_grad()
    for step,data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
          im_data.resize_(data[0].size()).copy_(data[0])
          gt_boxes.resize_(data[1].size()).copy_(data[1])
          num_boxes.resize_(data[2].size()).copy_(data[2])
          im_info.resize_(data[3].size()).copy_(data[3])
        im_data = im_data.view(-1,im_data.size(2),im_data.size(3),im_data.size(4))
        im_info = im_info.view(-1,3)
        gt_boxes= gt_boxes.view(-1,cfg.MAX_NUM_GT_BOXES,num_class+4)
        num_boxes = num_boxes.view(-1)
        
        #fasterRCNN.zero_grad()
        # compute ouevatput
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data,im_info,gt_boxes,num_boxes)
         
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()
         
        # backward
        #normalise the loss i.e average the accumuated loss by accumulation steps
        loss = loss / acc_step                   
        loss.backward()


        if (step+1)%acc_step == 0:   #wait for several backward step
          optimizer.step()     #update the model 
          fasterRCNN.zero_grad()#Reset the gradient tensor

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

        output = ('[session {0}][epoch {1}][iter {2}/{3}] loss: {loss:.4f}, lr: {lr:.2e}\n '
                  '\t\t\t\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'
                  '\t\t\t\t fg/bg=({4}/{5}) \n'
                  '\t\t\t\t rpn_cls: {loss_rpn_cls:.4f}, rpn_box: {loss_rpn_box:.4f},rcnn_cls: {loss_rcnn_cls:.4f},rcnn_box {loss_rcnn_box:.4f}\n' 
               .format(session, epoch, step, train_iters_per_epoch,fg_cnt, bg_cnt,
               loss= loss,lr= lr,batch_time=batch_time,data_time=data_time,
                      loss_rpn_cls=loss_rpn_cls,loss_rpn_box= loss_rpn_box,
                      loss_rcnn_cls=loss_rcnn_cls,loss_rcnn_box= loss_rcnn_box))
        #print("With fixed layer")
        print(output)
        log.write(output + '\n')
        log.flush()
        
        
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
    logger.close()

@torch.no_grad()
def validate_voc(val_loader,fasterRCNN,epoch,num_class,num_segments,vis,session,
             batch_size,input_data,cfg,log,ava_val_meter,dataset):
    val_iters_per_epoch = int(np.round(len(val_loader)))
    im_data,im_info,num_boxes,gt_boxes = input_data
    fasterRCNN.eval()
    all_boxes = [[[[]for _ in range(num_class)] for _ in range(batch_size *num_segments)]
               for _ in range(val_iters_per_epoch)]
    bbox = [[[[]for _ in range(num_class)] for _ in range(batch_size *num_segments)]
               for _ in range(val_iters_per_epoch)]
    #limit the number of proposal per image across all the class
    max_per_image = cfg.MAX_DET_IMG

    #confusion matrix
    conf_mat = ConfusionMatrix(num_classes = num_class, CONF_THRESHOLD = 0, IOU_THRESHOLD = 0.5)
 
    num_gt = [0 for _ in range(num_class)]   
    #data_iter = iter(val_loader)   
    for step,data in enumerate(val_loader):
    #for step in range (10):
        #data = next(data_iter)
        
        im_data.resize_(data[0].size()).copy_(data[0])
        gt_boxes.resize_(data[1].size()).copy_(data[1])
        num_boxes.resize_(data[2].size()).copy_(data[2])
        im_info.resize_(data[3].size()).copy_(data[3])
        im_data = im_data.view(-1,im_data.size(2),im_data.size(3),im_data.size(4))
        im_info = im_info.view(-1,3)
        gt_boxes= gt_boxes.view(-1,cfg.MAX_NUM_GT_BOXES,num_class+4)
        num_boxes = num_boxes.view(-1)

        
        #evaluate /inference cpde
        #start_time = time.time()
        rois, cls_prob, bbox_pred = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        #torch.cuda.synchronize()
        #end_time = time.time() - start_time

        if dataset == 'ucfsport':
         class_dict = act2id
        elif dataset == 'jhmdb':
          class_dict = jhmdbact2id
        elif dataset == 'ucf24':
          class_dict = ucf24act2id
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

        #gt boxes 
        gtbb = gt_boxes[:,:,0:4]
        gtlabels = gt_boxes[:,:,4:]
        #pred_boxes /= data[3][0][1][2].item()
        #gtbb /= data[3][0][1][2].item()
      
        #move the groudtruth to cpu
        gtbb = gtbb.cpu().numpy()
        gtlabels = gtlabels.cpu().numpy()
        #count = 0 

        for image in range(pred_boxes.shape[0]):
          for class_id in range(1,num_class):
            inds = torch.nonzero(scores[image,:,class_id]>0).view(-1)
            # if there is det
            if inds.numel() > 0:
             cls_scores = scores[image,inds,class_id]
             #arranging in descending order
             _, order = torch.sort(cls_scores, 0, True)
             cls_boxes = pred_boxes[image,inds, class_id * 4:(class_id + 1) * 4]
             cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
             cls_dets = cls_dets[order,:]
             keep = nms(cls_boxes[order, :], cls_scores[order],cfg.TEST.NMS)
             cls_dets = cls_dets[keep.view(-1)]  
             all_boxes[step][image][class_id] = cls_dets.cpu().numpy()
              
               
            #collect groud truth boxes for the image
            index =np.unique(np.nonzero(gtbb[image])[0])
            gtbox = gtbb[image][index]
            label = gtlabels[image][index]
            
            #take groundtruth box only if the label =1 for that class
            bbox[step][image][class_id] = [gtbox[i] for i in range(len(label)) if label[i,class_id]]
            num_gt[class_id] += np.sum(len(bbox[step][image][class_id]))
            '''if bbox[step][image][class_id]:
                conf_mat.process_batch(all_boxes[step][image][class_id], bbox[step][image][class_id],class_id)'''
    
    #plot confusion matrix
    '''import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    matrix = conf_mat.return_matrix()
    List1 = ["BG","diving","golf","kicking","lifting","riding","run","skateboarding",
              "swing1","swing2","walk","Falsepos"]
    List2 = ["BG","diving","golf","kicking","lifting","riding","run","skateboarding",
              "swing1","swing2","walk","FalseNeg"]
    df_cm = pd.DataFrame(matrix, index = [i for i in List1],
                  columns = [i for i in List2])
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.2) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size

    plt.show()'''

    
    #conf_mat.print_matrix()
    ap = [None for _ in range(num_class)]
    #calculate fp anf tp for each detections
    for cls_id in range(1,num_class):
      
      tpfp = []
      class_det = []
      for video in range(len(all_boxes)):
        for batch in range(len(all_boxes[0])):
          tp_fp = (tpfp_default(all_boxes[video][batch][cls_id],\
             bbox[video][batch][cls_id],iou_thr=0.5))
          if (len(tp_fp)>0 and len(all_boxes[video][batch][cls_id])>0):
            tpfp.append(tp_fp)
            class_det.append(all_boxes[video][batch][cls_id])
      assert len(tpfp) == len(class_det)
      tp, fp = tuple(zip(*tpfp))
      # sort all det bboxes by score, also sort tp and fp
      cls_det = np.vstack(class_det)
      num_dets = cls_det.shape[0]
      sort_inds = np.argsort(-cls_det[:, -1])
      tp = np.hstack(tp)[:, sort_inds]
      fp = np.hstack(fp)[:, sort_inds]
      # calculate recall and precision with tp and fp
      tp = np.cumsum(tp, axis=1)
      fp = np.cumsum(fp, axis=1)
      eps = np.finfo(np.float32).eps
      recalls = tp / np.maximum(num_gt[cls_id], eps)
      precisions = tp / np.maximum((tp + fp), eps)
      ap[cls_id] = average_precision(recalls[0, :], precisions[0, :],mode ='area')
    for k,v in class_dict.items():
      #print("Average precision per class:")
      out =("class [{0}]:{1}   |gt:{2}".format(k,ap[v],num_gt[v]))
      print(out)
      log.write(out + '\n')
    mAP= ("mAP for epoch [{0}] is : {1}".format(epoch,mean(ap[1:])))
    print(mAP)
    log.write(mAP + '\n')
    log.flush()
    print("----------------------------------------------")   
          

  
@torch.no_grad()
def validate_virat(val_loader,fasterRCNN,epoch,num_class,num_segments,vis,session,
             batch_size,input_data,cfg,log,ava_val_meter,dataset):
    val_iters_per_epoch = int(np.round(len(val_loader)))
    im_data,im_info,num_boxes,gt_boxes = input_data
    fasterRCNN.eval()
    all_boxes = [[[[]for _ in range(num_class)] for _ in range(batch_size *num_segments)]
               for _ in range(val_iters_per_epoch)]
    #limit the number of proposal per image across all the class
    max_per_image = cfg.MAX_DET_IMG
 
    #dict  with matched detections and its score @class_idx
    eval_target = {one:1 for one in activity2id_person}
    e = {one:{} for one in eval_target} # cat_id -> imgid -> {"dm","dscores"}

    #unique image id
    imgid = 0     
    num_gt = [0 for _ in range(num_class)]   
    #data_iter = iter(val_loader)   
    for step,data in enumerate(val_loader):
    #for step in range (50):
        #data = next(data_iter)
        
        im_data.resize_(data[0].size()).copy_(data[0])
        gt_boxes.resize_(data[1].size()).copy_(data[1])
        num_boxes.resize_(data[2].size()).copy_(data[2])
        im_info.resize_(data[3].size()).copy_(data[3])
        im_data = im_data.view(-1,im_data.size(2),im_data.size(3),im_data.size(4))
        im_info = im_info.view(-1,3)
        gt_boxes= gt_boxes.view(-1,cfg.MAX_NUM_GT_BOXES,num_class+4)
        num_boxes = num_boxes.view(-1)

        
        #evaluate /inference cpde
        start = time.time()
        rois, cls_prob, bbox_pred = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        torch.cuda.synchronize()
        end_time = time.time() - start
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

        #gt boxes 
        gtbb = gt_boxes[:,:,0:4]
        gtlabels = gt_boxes[:,:,4:]
        #pred_boxes /= data[3][0][1][2].item()
        #gtbb /= data[3][0][1][2].item()
      
        #move the groudtruth to cpu
        gtbb = gtbb.cpu().numpy()
        gtlabels = gtlabels.cpu().numpy()
        #count = 0 

        
        for image in range(pred_boxes.shape[0]):
          box = [None for _ in range(num_class)]
          imgid += 1
          for class_id in range(1,num_class):
            inds = torch.nonzero(scores[image,:,class_id]>cfg.VIRAT.SCORE_THRES).view(-1)
            # if there is det
            if inds.numel() > 0:
             cls_scores = scores[image,inds,class_id]
             #arranging in descending order
             _, order = torch.sort(cls_scores, 0, True)
             cls_boxes = pred_boxes[image,inds, class_id * 4:(class_id + 1) * 4]
             cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
             cls_dets = cls_dets[order,:]
             keep = nms(cls_boxes[order, :], cls_scores[order],cfg.TEST.NMS)
             cls_dets = cls_dets[keep.view(-1)]  
             all_boxes[step][image][class_id] = cls_dets.cpu().numpy()
              
               
            #collect groud truth boxes for the image
            index =np.unique(np.nonzero(gtbb[image])[0])
            gtbox = gtbb[image][index]
            label = gtlabels[image][index]
            
            #take groundtruth box only if the label =1 for that class
            box[class_id] = [gtbox[i] for i in range(len(label)) if label[i,class_id]]
            num_gt[class_id] += np.sum(len(box[class_id]))
          match_dt_gt(e, imgid, all_boxes[step][image], box, activity2id_person)
          if (step+1) % 50 == 0:
            output = ('Test: [{0}/{1}]\t'
                .format(step,(val_iters_per_epoch)))
            print(output)
    
    aps = aggregate_eval(e, maxDet=max_per_image)
    mAP = (mean(aps[target] for target in aps.keys()))
    
    for k,v in aps.items():   
      output =('class: [{0}] - {1}'.format(k,v)) 
      log.write(output + '\n')
      print(output)
    mAPout = ('mAP at epoch {0}: {1}'.format(epoch,mAP)) 
    print('mAP at epoch {0}: {1} \n'.format(epoch,mAP)) 
    log.write( mAPout + '\n')
    log.flush()       

        
@torch.no_grad()
def validate(val_loader,fasterRCNN,epoch,num_class,num_segments,vis,session,
             batch_size,input_data,cfg,log,ava_val_meter,dataset):
    
    batch_time = AverageMeter()
    val_iters_per_epoch = int(np.round(len(val_loader)))
    im_data,im_info,num_boxes,gt_boxes = input_data
    fasterRCNN.eval()
    all_boxes = [[[[]for _ in range(num_class)] for _ in range(batch_size *num_segments)]
               for _ in range(val_iters_per_epoch)]
    #limit the number of proposal per image across all the class
    max_per_image = cfg.MAX_DET_IMG
    bins = 9 
    score_threshold = 0.1
    tp_labels = np.zeros((num_class,bins),dtype=int)
    fp_labels = np.zeros((num_class,bins),dtype=int)
    fn_labels = np.zeros((num_class,bins),dtype=int)
    all_pred_box,all_gtbb,all_gtlabels,all_scores=[],[],[],[]
    end = time.time()
    #data_iter = iter(val_loader)
    
    for step,data in enumerate(val_loader):
    #for step in range (500):
        #data = next(data_iter)
        im_data.resize_(data[0].size()).copy_(data[0])
        gt_boxes.resize_(data[1].size()).copy_(data[1])
        num_boxes.resize_(data[2].size()).copy_(data[2])
        im_info.resize_(data[3].size()).copy_(data[3])
        im_data = im_data.view(-1,im_data.size(2),im_data.size(3),im_data.size(4))
        im_info = im_info.view(-1,3)
        gt_boxes= gt_boxes.view(-1,cfg.MAX_NUM_GT_BOXES,num_class+4)
        num_boxes = num_boxes.view(-1)
        if ava_val_meter:
          meta = data[4]
          if isinstance(meta["metadata"], (list,)):
                for i in range(len(meta["metadata"])):
                    meta["metadata"][i] = meta["metadata"][i].cpu().numpy()
        start_time = time.time()
        rois, cls_prob, bbox_pred = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        torch.cuda.synchronize()
        end_time = time.time() - start_time

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

        if ava_val_meter:
           pred_boxes /=(im_data.size(2)-1)
        else:
          #convert the prediction boxes and gt_boxes to the image size
          gtbb = gt_boxes[:,:,0:4]
          gtlabels = gt_boxes[:,:,4:]
          #pred_boxes /= data[3][0][1][2].item()
          #gtbb /= data[3][0][1][2].item()
      
          #move the groudtruth to cpu
          gtbb = gtbb.cpu().numpy()
          gtlabels = gtlabels.cpu().numpy()
        count = 0 
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
            #take the image threshold value taking 5th or 15th entry  
              image_thresh = np.sort(image_scores)[-max_per_image]
              for class_id in range(1, num_class):
                if len(all_boxes[step][image][class_id]) > 0:
                  keep = np.where(all_boxes[step][image][class_id][:, -1] >= image_thresh)[0]
                  all_boxes[step][image][class_id] = all_boxes[step][image][class_id][keep, :]
                else :
                  all_boxes[step][image][class_id] = []
          #if all_boxes[step][image].any()  
          det = np.asarray(all_boxes[step][image]) 
          if det.size == 0:
            continue
          else:
            all_boxes[step][image] = avg_iou(all_boxes[step][image],0.7)

          if ava_val_meter:
            if ((image % num_segments) == 0):
              #if len(all_boxes[step][image]) > 5: #check for empty list
              #   if not any(all_boxes[step][image]):
              #      continue
                
             # else:
                preds = all_boxes[step][image]
                metadata = list(map(itemgetter(count),meta["metadata"]))
                ava_val_meter.update_stats(preds,metadata)
                count += 1

          else:  
            #calculate tp_fp_fn
            tp_labels,fp_labels,fn_labels =compute_tp_fp(all_boxes[step][image],gtlabels[image],gtbb[image], \
                      num_class,tp_labels,fp_labels,fn_labels,bins)
        output = ('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(step,(val_iters_per_epoch), batch_time=batch_time))
        print(output)
     
    if ava_val_meter:
       ava_val_meter.finalize_metrics(log)
       ava_val_meter.reset()
    else : 
       ap = precision_recall(tp_labels,fp_labels,fn_labels,num_class)
       if dataset == 'virat':
         dictio = activity2id
       else:
         dictio = class_dict
       print_ap(tp_labels,fp_labels,fn_labels,num_class,log,step,epoch,dictio,ap)
if __name__ == '__main__':
   main()    
