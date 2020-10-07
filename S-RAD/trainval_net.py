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
from lib.model.utils.config import cfg,cfg_from_list,cfg_from_file
from lib.model.nms.nms import soft_nms,py_cpu_nms,avg_iou
from validation.eval_net import *
from lib.model.roi_layers import nms

from lib.model.faster_rcnn.resnet import resnet
from lib.model.faster_rcnn.vgg16 import vgg16

def main():
  args = parse_args()
  print('Called with args:')
  print(args)
  
  best_meanap =0
  meanap = 0
  if args.dataset == "virat":args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 3]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '15']
  if args.dataset == "ucfsport":args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,24,28]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '2']
  if args.dataset == "urfall":args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,24,28]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '1']
  if args.dataset == "imfd":args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,24,28]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '1']
  if args.dataset == "jhmdb":args.set_cfgs = ['ANCHOR_SCALES', '[4, 8,16,24,28]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '1']
  if args.dataset == "ucf24":args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,24,28]', 'ANCHOR_RATIOS', 
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
  elif args.dataset =='ucfsport':
      num_class = cfg.UCFSPORT.NUM_CLASSES
      output_dir = cfg.UCFSPORT.output_model_dir + "/" + args.net + "/" + args.dataset
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
  elif args.dataset =='urfall':
      num_class = cfg.URFD.NUM_CLASSES
      output_dir = cfg.URFD.output_model_dir + "/" + args.net + "/" + args.dataset
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
  elif args.dataset =='imfd':
      num_class = cfg.IMFD.NUM_CLASSES
      output_dir = cfg.IMFD.output_model_dir + "/" + args.net + "/" + args.dataset
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
  
  #log initialisation
  args.store_name = '_'.join(
        ['S-RAD', args.dataset,args.net,'segment%d' % args.num_segments,'e{}'.format(args.max_epochs),'session%d'%args.session])
  check_rootfolders(args.store_name,args.dataset)
  
  #logging
  log_training,logger = log_info(cfg,args.store_name,args.dataset,args= args)
  
  #dataloader
  train_loader = construct_loader(cfg,dataset = args.dataset,num_segments = args.num_segments,interval=args.interval,
        batch_size = args.batch_size,split = 'train',input_sampling = True,split_num=args.splits,pathway= args.pathway)
  val_loader = construct_loader(cfg,dataset = args.dataset,num_segments = args.num_segments, interval = args.interval,
        batch_size = args.batch_size,split = 'val',input_sampling = True,split_num=args.splits,pathway= args.pathway)
  if args.dataset == 'virat':
    test_loader = construct_loader(cfg,dataset = args.dataset,num_segments = args.num_segments,interval= args.interval,
        batch_size = args.batch_size,split = 'test',input_sampling = True,split_num=args.splits,pathway= args.pathway)
      
  # prevent something not % n_GPU
  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    S_RAD = vgg16(num_class,pretrained=True,class_agnostic=args.class_agnostic,
    loss_type = args.loss_type)
  elif args.net == 'res50':
   S_RAD = resnet(num_class,num_layers =50, base_model ='resnet50', n_segments =args.num_segments,
               n_div =args.shift_div , place = args.shift_place,
               pretrain = args.pretrain,shift = args.shift,
               class_agnostic=args.class_agnostic,loss_type =args.loss_type,pathway=args.pathway)
  
  else:
   print("network is not defined")
   pdb.set_trace()

  #create the architecture
  S_RAD.create_architecture()
  
  #set the parameters
  lr = args.lr
  params = []
  for key, value in dict(S_RAD.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    S_RAD.cuda()
  
  #define optimizer
  if args.optimizer == "adam":
     lr = lr * 0.1
     optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
     optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  #adding UCF sport weights to the first branch base1
  #if args.pathway == "two_pathway":
  #if args.tune_from:
        
     
  ## adding temporal shift pretrained kinetics weights
  #if args.pathway =="naive":
  if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = S_RAD.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict:
                replace_dict.append((k.replace('module.base_model.conv1',
                'RCNN_base1.0').replace('module.base_model.bn1','RCNN_base1.1')
                .replace('module.base_model.layer1.0','RCNN_base1.4.0')
                .replace('module.base_model.layer1.1','RCNN_base1.4.1')
                .replace('module.base_model.layer1.2','RCNN_base1.4.2')
                .replace('module.base_model.layer2.0','RCNN_base1.5.0')
                .replace('module.base_model.layer2.1','RCNN_base1.5.1')
                .replace('module.base_model.layer2.2','RCNN_base1.5.2')
                .replace('module.base_model.layer2.3','RCNN_base1.5.3')
                .replace('module.base_model.layer3.0','RCNN_base1.6.0')
                .replace('module.base_model.layer3.1','RCNN_base1.6.1')
                .replace('module.base_model.layer3.2','RCNN_base1.6.2')
                .replace('module.base_model.layer3.3','RCNN_base1.6.3')
                .replace('module.base_model.layer3.4','RCNN_base1.6.4')
                .replace('module.base_model.layer3.5','RCNN_base1.6.5')
                .replace('module.base_model.layer4.0.','RCNN_top.0.0.')
                .replace('module.base_model.layer4.1','RCNN_top.0.1')
                .replace('module.base_model.layer4.2','RCNN_top.0.2')
                .replace('module.base_model.layer4.0.conv1.net','RCNN_top.0.0.conv1')
                .replace('module.base_model.layer4.1.conv1.net','RCNN_top.0.1.conv1')
                .replace('module.base_model.layer4.2.conv1.net','RCNN_top.0.2.conv1')
                .replace('RCNN_top.0.0.conv1.net','RCNN_top.0.0.conv1')
                .replace('RCNN_top.0.1.conv1.net','RCNN_top.0.1.conv1')
                .replace('RCNN_top.0.2.conv1.net','RCNN_top.0.2.conv1'),k))
                 
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
        S_RAD.load_state_dict(model_dict)
 
  if args.resume:
    load_name = os.path.join(output_dir,
      'S-RAD_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    
    sd = checkpoint['model']
    model_dict = S_RAD.state_dict()
    replace_dict = []
    for k, v in sd.items():
      if k not in model_dict:
        replace_dict.append((k.replace('RCNN_base',
             'RCNN_base1'),k))
    for k_new, k in replace_dict:
            sd[k_new] = sd.pop(k)
    keys1 = set(list(sd.keys()))
    keys2 = set(list(model_dict.keys()))
    set_diff = (keys1 - keys2) | (keys2 - keys1)
    print('#### Notice: keys that failed to load: {}'.format(set_diff))
    model_dict.update(sd)
    S_RAD.load_state_dict(model_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  
  if args.mGPUs:
    mGPUs = True
    S_RAD = nn.DataParallel(S_RAD)
  else:
    mGPUs = False
  
  session = args.session
  
  if args.evaluate:
    validate_voc(val_loader, S_RAD,args.start_epoch,num_class, \
             args.num_segments,session,args.batch_size,\
             cfg,log_training,args.dataset,args.pathway,args.eval_metrics)
    sys.exit
    
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    
    if epoch % (args.lr_decay_step + 1) == 0:
    
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma
     
    if args.dataset == 'virat':
      #dataloader 
      train(train_loader, S_RAD,lr,optimizer,
        epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
        data,cfg,args.acc_step,log_training)
       
      # evaluate on validation set
      validate_virat(val_loader, S_RAD,epoch,num_class, \
      args.num_segments,session,args.batch_size,data,cfg,\
      log_training,args.dataset)
      
      if epoch % 10== 0:
          validate_virat(test_loader, S_RAD,epoch,num_class, \
              args.num_segments,session,args.batch_size,data,cfg,
              log_training,args.dataset)
       
    elif args.dataset == 'ucfsport':
    
      train(train_loader, S_RAD,lr,optimizer,
            epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
            cfg,args.acc_step,log_training)
      validate_voc(val_loader, S_RAD,epoch,num_class, \
              args.num_segments,session,args.batch_size,
              cfg,log_training,args.dataset,args.pathway,args.eval_metrics)

    elif args.dataset == 'urfall':
      
      train(train_loader, S_RAD,lr,optimizer,
            epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
            cfg,args.acc_step,log_training)
      validate_voc(val_loader, S_RAD,epoch,num_class, \
             args.num_segments,session,args.batch_size,\
             cfg,log_training,args.dataset,args.pathway,args.eval_metrics)   
    
    elif args.dataset == 'imfd':
      
      train(train_loader, S_RAD,lr,optimizer,
            epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
            cfg,args.acc_step,log_training)
      validate_voc(val_loader, S_RAD,epoch,num_class, \
              args.num_segments,session,args.batch_size,
              cfg,log_training,args.dataset,args.pathway,args.eval_metrics)    
    
    elif args.dataset == 'jhmdb':
      
      train(train_loader, S_RAD,lr,optimizer,
            epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
            cfg,args.acc_step,log_training)
      if epoch % 2== 0:
        validate_voc(val_loader, S_RAD,epoch,num_class, \
              args.num_segments,session,args.batch_size,
              cfg,log_training,args.dataset,args.pathway,args.eval_metrics)
    
    elif args.dataset == 'ucf24':
      train(train_loader, S_RAD,lr,optimizer,
            epoch,num_class,args.batch_size,session,mGPUs,logger,output_dir,
            cfg,args.acc_step,log_training)
      validate_voc(val_loader, S_RAD,epoch,num_class, \
              args.num_segments,session,args.batch_size,
              cfg,log_training,args.dataset,args.pathway,args.eval_metrics)
 
 
def train(train_loader,S_RAD,lr,optimizer,epoch,num_class,batch_size,session,mGPUs,
          logger,output_dir,cfg,acc_step,log):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # setting to train mode
    S_RAD.train()
    #loss = 0
    loss_temp = 0
    end = time.time()
    train_iters_per_epoch = int(len(train_loader.dataset) / batch_size)
    optimizer.zero_grad()
    S_RAD.zero_grad()
    for step,data in enumerate(train_loader):
        # measure data loading time
               
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = S_RAD(data)
         
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()
         
        # backward
        #normalise the loss i.e average the accumulated loss by accumulation steps
        loss = loss / acc_step                   
        loss.backward()


        if (step+1)%acc_step == 0:   #wait for several backward step
          optimizer.step()           #update the model 
          S_RAD.zero_grad()          #Reset the gradient tensor

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
        
        
    save_name = os.path.join(output_dir, 'S-RAD_{}_{}_{}.pth'.format(session, epoch, step))
    save_checkpoint({
      'session': session,
      'epoch': epoch + 1,
      'model': S_RAD.module.state_dict() if mGPUs else S_RAD.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      #'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))
    logger.close()


if __name__ == '__main__':
   main()    
