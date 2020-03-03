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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision
from statistics import mean 

from torch.utils.data.sampler import Sampler
from data_loader.data_load import Action_dataset
from dataset_config import dataset
from dataset_config.transforms import *
from tensorboardX import SummaryWriter

import multiprocessing
multiprocessing.set_start_method('spawn', True)
#from roi_data_layer.roidb import combined_roidb
#from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.rpn.bbox_transform import clip_boxes
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

from model.nms.nms import soft_nms,py_cpu_nms

# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv,bbox_overlaps, bbox_transform_batch
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

  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  print('Called with args:')
  print(args)

  if args.dataset == "virat":args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 3]', 'ANCHOR_RATIOS', 
      '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  args.store_name = '_'.join(
        ['Action_proposal', args.dataset, args.modality, args.net, 'segment%d' % args.num_segments,
         'e{}'.format(args.max_epochs)])
  if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
  print('storing name: ' + args.store_name)
  
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
  #cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
          os.makedirs(output_dir)

  #dataloader
  #train_augmentation = fasterRCNN.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)
  input_size =600
  input_mean = [0.485, 0.456, 0.406]
  input_std = [0.229, 0.224, 0.225]
  normalize = GroupNormalize(input_mean, input_std)

  train_loader = torch.utils.data.DataLoader(
        Action_dataset(train_path, args.train_list, num_segments=args.num_segments,
                   modality=args.modality,random_shift=True, test_mode=False,
                   input_size = input_size,
                   transform=torchvision.transforms.Compose([ 
                       ToTorchFormatTensor(div=1),
                       normalize]),dense_sample =False,uniform_sample=False,
                 random_sample = False,strided_sample = False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU
  
  val_loader = torch.utils.data.DataLoader(
        Action_dataset(val_path, args.val_list, num_segments=args.num_segments,
                   modality=args.modality,random_shift=True, test_mode=False,
                   input_size = input_size,
                   transform=torchvision.transforms.Compose([ 
                       ToTorchFormatTensor(div=1),
                       normalize]),dense_sample =False,uniform_sample=False,
                 random_sample = False,strided_sample = False),
        batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        ) 
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
               class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
   fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
   print("network is not defined")
   pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
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
  
  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")
  
  if args.evaluate:
    validate(val_loader, fasterRCNN,0,val_iters_per_epoch,num_class,args.num_segments)
    #return
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    session = args.session
    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    train(train_loader, fasterRCNN,lr,optimizer,
    epoch,train_iters_per_epoch,session,mGPUs,logger,output_dir)
    
    # evaluate on validation set
    if epoch % 4 == 0:
      validate(val_loader, fasterRCNN,epoch,val_iters_per_epoch,num_class,args.num_segments)

def train(train_loader,fasterRCNN,lr,optimizer,epoch,train_iters_per_epoch,session,mGPUs,logger,output_dir):
       
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

    # setting to train mode
    fasterRCNN.train()
    #loss = 0
    loss_temp = 0
    data_iter = iter(train_loader)
    for step in range (train_iters_per_epoch):
        data = next(data_iter)
        
        with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              gt_boxes.resize_(data[1].size()).copy_(data[1])
              num_boxes.resize_(data[2].size()).copy_(data[2])
              im_info.resize_(data[3].size()).copy_(data[3])
              imgsize1 = im_data.size(3)
              imgsize2 = im_data.size(4)
              channel = im_data.size(2)
              im_data = im_data.view(-1,channel,imgsize1,imgsize2)
              im_info = im_info.view(-1,3)
              gt_boxes= gt_boxes.view(-1,15,44)
              num_boxes = num_boxes.view(-1)
        
        fasterRCNN.zero_grad()
        # compute output
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data,im_info,gt_boxes,num_boxes)
         
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()
         
        # backward
        optimizer.zero_grad()
        loss.backward()
        #if args.net == "vgg16":
        #  clip_gradient(fasterRCNN, 10.)
        optimizer.step()

        #if step % args.disp_interval == 0:
        # end = time.time()
        if step > 0:
          loss_temp /= (100 + 1)

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

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (session, epoch, step, train_iters_per_epoch, loss, lr))
    
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
        

def validate(val_loader,fasterRCNN,epoch,val_iters_per_epoch,num_class,num_segments):
    
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

    fasterRCNN.eval()
    final_score,tp,fp ,ap= [],[],[],[]
    num_gt = 0
    true_positive,false_positive = [],[]
    for step in range(val_iters_per_epoch):
      data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              gt_boxes.resize_(data[1].size()).copy_(data[1])
              num_boxes.resize_(data[2].size()).copy_(data[2])
              im_info.resize_(data[3].size()).copy_(data[3])
              imgsize1 = im_data.size(3)
              imgsize2 = im_data.size(4)
              channel = im_data.size(2)
              im_data = im_data.view(-1,channel,imgsize1,imgsize2)
              im_info = im_info.view(-1,3)
              gt_boxes= gt_boxes.view(-1,15,44)
              num_boxes = num_boxes.view(-1)
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      
      #get the rois label to index out the single bbox pred
      RCNN_proposal_target = _ProposalTargetLayer(num_class)
      rois_label = RCNN_proposal_target(rois, gt_boxes, num_boxes,val =1)
      #roi_data= roi_data.cpu().numpy()
      rois_label = rois_label.view(-1,40)

      #calculate the scores and boxes
      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]
      batch_size = rois.shape[0]
      box_deltas = bbox_pred.data
      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
      box_deltas = box_deltas.view(batch_size, -1, 4 * num_class)
      pred_boxes = bbox_transform_inv(boxes, box_deltas, 8)
      pred_boxes = clip_boxes(pred_boxes, im_info.data, 8)   

      #convert the prediction boxes and gt_boxes to the image size
      gtbb = gt_boxes[:,:,0:4]
      gtlabels = gt_boxes[:,:,4:]
      pred_boxes /= data[3][0][1][2].item()
      gtbb /= data[3][0][1][2].item()

      #move the items to cpu to save GPU memory
      rois_label = rois_label.cpu().numpy()
      proposal_num,class_num = np.nonzero(rois_label)
      
      pred_boxes = pred_boxes.cpu().numpy()
      gtbb = gtbb.cpu().numpy()
      gtlabels = gtlabels.cpu().numpy()
      scores = scores.cpu().numpy()
      #code to get single bbox detection from the 40 bbox detections for a single proposal
      pbox = pred_boxes.reshape(-1,pred_boxes.shape[2])
      bbox_pred_view = pbox.reshape(pbox.shape[0], int(pbox.shape[1] / 4), 4)
      bbox_pred_select = np.zeros((pbox.shape[0], 1, 4),dtype = float)
            
      for i in range (proposal_num.shape[0]):
                dup = np.argwhere(proposal_num == proposal_num[i])
                if (len(dup)) != 1:
                   bbox_pred_select[proposal_num[i]] = bbox_pred_view[proposal_num[i],class_num[dup],:].mean(0)
                    
                else:
                   bbox_pred_select[proposal_num[i]] = bbox_pred_view[proposal_num[i],class_num[i],:]
      bbox_pred = bbox_pred_select.squeeze(1)

      #calculate the scores ,tp,fp labels for the final proposals
      metrics = calc_proposal_after_nms(bbox_pred,scores,gtbb,num_class,gtlabels)
      fscore,tp1,fp1,gt = metrics
      num_gt += gt
      final_score.append(fscore)
      tp.extend(tp1)
      fp.extend(fp1)
    print(f"completed calculating tp/fp labels for step: {step}")
    
    #calculate precision for i'th class
    for i in range(num_class-1): #starts from 0 indices till 39
      s = []
      for video in range(val_iters_per_epoch):
       imd_score = final_score[video][:,i]
       s.extend(imd_score)
      ap += precision_recall(tp,fp, s, num_gt )
      for key, v in activity2id.items():
        if v == i :
          print(f"Class '{i}' ({key}) - AveragePrecision: {ap[i]}")
    print(f"mAP for epoch [{epoch}]: {mean(ap)}")


def calc_proposal_after_nms (pred_boxes,scores,boxes,num_class,labels):

  #take the non-.zero gt boxes and gt labels
  index=np.unique(np.nonzero(boxes)[1])
  gtbox = boxes[:,index,:]
  gtlabel = labels[:,index,:]
  num_gt = gtbox.shape[0]*gtbox.shape[1]
  score = scores.reshape(-1,40)
  pred_box = pred_boxes.reshape(-1,4)
  
  #apply nms to remove the extra proposals
  keep = py_cpu_nms(pred_box,score, 0.3)
  pred_box_after_nms = pred_box[keep,:]
  score_afer_nms = score[keep]       
  (score, tp_labels,fp_labels,num_gt) = compute_tp_fp(pred_box_after_nms,score_afer_nms,
                                                                  gtbox,gtlabel)
  
  return score,tp_labels,fp_labels,num_gt
          
def precision_recall(tp_labels,fp_labels,scores,num_gt):
  ap, p, r = [], [], []

  scores = np.asarray(scores)
  index = np.argsort(-scores)
  
  scores= scores[index]

  fp_labels = np.asarray(fp_labels)
  fp_labels = fp_labels[index]
  tp_labels = np.asarray(tp_labels)
  tp_labels = tp_labels[index]
  #if tp_labels == 0 and n_gt == 0:
  #  continue
  if np.sum(tp_labels) == 0 or num_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
  else:
   
   fpc = fp_labels.cumsum()
   tpc = tp_labels.cumsum()

   precision = tpc.astype(float) / (
        tpc + fpc + 0.00001
    )
   p.append(precision)
   recall = tpc.astype(float) / num_gt
   r.append(recall)
   ap.append(compute_ap(recall, precision))
  #print("Both positives and gt are zero")
  return ap

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

          
def compute_tp_fp(detected_boxes_at_ith_class,detected_scores_at_ith_class,
                                    box,label):
     
     gtbox = box.reshape(-1,4)
     gtlabel=label.reshape(-1,40)
     #detected_boxes_at_ith_class = detected_boxes_at_ith_class.view(-1,4)
     #detected_scores_at_ith_class = detected_scores_at_ith_class.view(-1,40)
     
     #compute IOU between predicted proposal and gt box
     overlaps = cpu_bbox_overlaps(detected_boxes_at_ith_class, gtbox)
     num_proposal = detected_boxes_at_ith_class.shape[0]
     num_boxes_per_img = gtbox.shape[0]
     
     # initilaise the tp,fp labels 
     tp_labels = np.zeros(num_proposal)
     fp_labels = np.zeros(num_proposal)
     is_gt_detected =[] 
     is_proposal_labeled =[]

     #calculate the tp,fp labels for each proposal and keep count of gt and detections match
     if (num_proposal>0) and (num_boxes_per_img !=0):
            match_table = np.zeros((num_proposal,num_boxes_per_img),dtype=float)
            for p in range(num_proposal):
                for g in range(num_boxes_per_img):
                    if overlaps[p,g] >= 0.5:
                        match_table[p,g] = overlaps[p,g]
                    
            best_match = match_table.max()
            while best_match >= 0.5:
                match_pos = np.where(match_table == best_match)
                p = match_pos[0][0]
                detect_label = detected_scores_at_ith_class[p,:]
                detect_pos = np.argwhere(detect_label > 0.6)
                g = match_pos[1][0]
                gt_label = gtlabel[g,:]
                gt_pos = np.argwhere(gt_label ==1 )
                is_gt_detected += [g]
                is_proposal_labeled+=[p]
                if len(gt_pos) == len(detect_pos) and np.equal(gt_pos,detect_pos).all():
                  tp_labels[p] = 1
                else:
                  fp_labels[p] = 1
                match_table[p,:] = np.zeros(match_table[p,:].shape)
                match_table[:,g] = np.zeros(match_table[:,g].shape)

                best_match = match_table.max()
     
     #count the left out proposals as fp      
     for i in range(num_proposal): 
        if i not in is_proposal_labeled:
          fp_labels[i] = 1


     
     tp_labels = tp_labels.reshape(-1)
     fp_labels = fp_labels.reshape(-1)
     return detected_scores_at_ith_class,tp_labels,fp_labels,num_boxes_per_img
def cpu_bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).reshape(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).reshape(N, 1)

    boxes = np.expand_dims(anchors,axis = 1)
    boxes = np.tile(boxes,(K,1))
    query_boxes = np.expand_dims(gt_boxes,axis=0)
    query_boxes = np.tile(query_boxes,(N,1,1))

    iw = (np.minimum(boxes[:,:,2], query_boxes[:,:,2]) -
        np.maximum(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (np.minimum(boxes[:,:,3], query_boxes[:,:,3]) -
        np.maximum(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps
     
if __name__ == '__main__':
   main()    
