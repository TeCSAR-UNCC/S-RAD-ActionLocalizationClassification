import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN
from lib.model.fuse.fuse_twopath import *

from lib.model.roi_layers import ROIAlign,ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from lib.model.utils.net_utils import *
from lib.model.utils.net_utils import _smooth_l1_loss

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic,loss_type,pathway):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.pathway = pathway
        self.n_classes = classes
        self.class_agnostic = class_agnostic
        self.loss_type = loss_type
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        #define fuse layer
        if pathway == "two_pathway":
            self.fuselayer = Fuse_twopath(self.dout_base_model)

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model,self.n_classes)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    #def forward(self, im_data, im_info, gt_boxes, num_boxes):
    def forward(self, data):
        #batch_size = im_data.size(0)
        if self.pathway == 'two_pathway':
            chan = data[1].shape[2]
            img_h = data[1].shape[3]
            img_w = data[1].shape[4]

            im_info = (data[0][3].view(-1,3)).to(device="cuda")
            gt_boxes= (data[0][1].view(-1,cfg.MAX_NUM_GT_BOXES,self.classes+4)).to(device="cuda")
            num_boxes = (data[0][2].view(-1)).to(device="cuda")


            im_data1 = (data[0][0].view(-1,chan,img_h,img_w)).to(device="cuda")
            batch_size=im_data1.shape[0]
            im_data2 = (data[1].view(-1,chan,img_h,img_w)).to(device="cuda")
            
            # feed image data to base model to obtain base feature map
            #slow TSM way
            base_feat1 = self.RCNN_base1(im_data1)
            #fast non TSM way
            base_feat2 = self.RCNN_base2(im_data2)

            #changes
            base_feat =self.fuselayer(base_feat1,base_feat2)

        else:
            chan = data[0].shape[2]
            height = data[0].shape[3]
            width = data[0].shape[4]
            

            im_info = (data[3].view(-1,3)).to(device="cuda")
            gt_boxes= (data[1].view(-1,cfg.MAX_NUM_GT_BOXES,self.classes+4)).to(device="cuda")
            num_boxes = (data[2].view(-1)).to(device="cuda")


            im_data = (data[0].view(-1,chan,height,width)).to(device="cuda")
            batch_size=im_data.shape[0]
            
            base_feat = self.RCNN_base1(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes,val=0)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        
        if cfg.POOLING_MODE == "align":
           pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == "crop": #TODO
           pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
           pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
          if self.loss_type == 'focal' or self.loss_type == 'sigmoid':
            # select the corresponding columns according to roi labels
            rois_label = Variable(rois_label.view(-1,self.classes))#.long()) #modified
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            proposal_num = torch.nonzero(rois_label)[:,0]
            class_num = torch.nonzero(rois_label)[:,1]
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = bbox_pred_view.new(bbox_pred.size(0), 1, 4).zero_()
            for i in range (proposal_num.shape[0]):
                dup = torch.nonzero(proposal_num == proposal_num[i])
                if (dup.shape[0] > 1):
                   bbox_pred_select[proposal_num[i]] = bbox_pred_view[proposal_num[i],class_num[dup],:].mean(0)
                    
                else:
                   bbox_pred_select[proposal_num[i]] = bbox_pred_view[proposal_num[i],class_num[i],:]
                
                    
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        if self.loss_type == "sigmoid":
           cls_prob = torch.sigmoid(cls_score)
        if self.loss_type == "softmax":
           cls_prob = torch.softmax(cls_score,1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        
        if self.training:
            # classification loss
            if self.loss_type == "sigmoid":
               RCNN_loss_cls = F.binary_cross_entropy_with_logits(cls_score, rois_label)
               
            elif self.loss_type == "softmax":
               rois_label = Variable(rois_label.view(-1,self.classes).long())
               rois_label_select = rois_label.new(rois_label.size(0)).zero_()
               proposal_num = torch.nonzero(rois_label)[:,0]
               class_num = (torch.nonzero(rois_label)[:,1])
               rois_label_select[proposal_num] = class_num
               

               rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
               rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
               rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
               
               bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
               bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label_select.view(rois_label_select.size(0), 1, 1).expand(rois_label_select.size(0), 1, 4))
               bbox_pred = bbox_pred_select.squeeze(1)
               RCNN_loss_cls = F.cross_entropy(cls_score, rois_label_select)
             
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.training:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
                RCNN_loss_cls, RCNN_loss_bbox, rois_label
        else :
            return rois,cls_prob,bbox_pred

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
