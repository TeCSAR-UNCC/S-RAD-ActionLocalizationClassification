import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from model.utils.config import cfg
import os
import cv2
import pdb
import random
from tensorboardX import SummaryWriter
from statistics import mean 
import operator
from operator import itemgetter

from model.rpn.bbox_transform import cpu_bbox_overlaps

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    per_entry_cross_ent = (F.binary_cross_entropy_with_logits(
        labels, logits))
    prediction_probabilities = torch.sigmoid(logits)
    p_t = ((labels * prediction_probabilities) +
           ((1 - labels) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if gamma:
      modulating_factor = torch.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
      alpha_weight_factor = (labels * alpha +
                             (1 - labels) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)
    return focal_cross_entropy_loss.mean() #* weights
    




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

def vis_detections(im, class_name, scores,pred_box,thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, scores.shape[0])):
        bbox = pred_box[i]
        index = np.nonzero(scores[i] > thresh)
        score = scores[i][index]
        if score.all() !=0:
            im = cv2.rectangle(im, (int(bbox[0]),int(bbox[1])), \
             (int(bbox[2]),int(bbox[3])), (0, 0, 204), 2)
            if len(index[0]) > 0:
                for ind in range(len(index[0])):
                    cv2.putText(im,"'{0}:{1:03f}'".format(class_name[index[0][ind]],score[ind]),
                (int(bbox[0]), int(bbox[1]) + 15*ind), cv2.FONT_HERSHEY_PLAIN,\
                        1.0, (0, 0, 255), thickness=1)
            '''else:
                cv2.putText(im, "'{0}:{1}'".format(class_name[index[0][0]]\
                    ,score),(int(bbox[0]), int(bbox[1]) + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)'''

        
    return im

def gt_visuals(im, class_name,pred_box,labels):
    """Visualizing the ground truth."""
    for i in range(labels.shape[0]):
        bbox = pred_box[i]
        im = cv2.rectangle(im, (int(bbox[0]),int(bbox[1])), \
             (int(bbox[2]),int(bbox[3])), (0, 204, 0), 2)
        index =np.where(labels[i]==1)
        for ind in range(len(index[0])):
            cv2.putText(im,"'{0}'".format(class_name[index[0][ind]]),
                (int(bbox[2]), int(bbox[3]) + 15*ind), cv2.FONT_HERSHEY_PLAIN,\
                        1.0, (0, 255, 0), thickness=1)
        
    return im

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def match_dt_gt(e, imgid, target_dt_boxes, gt_boxes, eval_target):
  for target_class,id in eval_target.items():
    #if len(gt_boxes[target_class]) == 0:
    #  continue
    # target_dt_boxes[id].sort(key=operator.itemgetter(1), reverse=True)
    if len(target_dt_boxes[id])>0:
      d = target_dt_boxes[id][:,0:4]
      dscores=target_dt_boxes[id][:,4:5]
    else:
      d=np.zeros((0,4))
      dscores=np.zeros((0,1))
    #d = [box for box, prob in target_dt_boxes[id]]
    #dscores = [prob for box, prob in target_dt_boxes[target_class]]
    if (gt_boxes[id]):
      g=np.vstack(gt_boxes[id])
    else:
      g = np.zeros((0,4))
    # len(D), len(G)
    iou = cpu_bbox_overlaps(d,g)

    dm, gm = match_detection(d, g, iou,iou_thres=0.5)
    
    e[target_class][imgid] = {
        "dscores": dscores,
        "dm": dm,
        "gt_num": len(g)}

def aggregate_eval(e, maxDet=100):
  aps = {}
  ars = {}
  for catId in e:
    e_c = e[catId]
    # put all detection scores from all image together
    dscores = np.concatenate([e_c[imageid]["dscores"][:maxDet]
                              for imageid in e_c])
    dscores = dscores.reshape(-1)                         
    # sort
    inds = np.argsort(-dscores, kind="mergesort")
    # dscores_sorted = dscores[inds]

    # put all detection annotation together based on the score sorting
    dm = np.concatenate([e_c[imageid]["dm"][:maxDet] for imageid in e_c])[inds]
    num_gt = np.sum([e_c[imageid]["gt_num"] for imageid in e_c])
    
    print("gt: class['{0}'] is :'{1}'\t".format(catId,num_gt))
    print("detections: class['{0}'] is :'{1}'".format(catId,(dscores.shape[0])))
    # here the average precision should also put the unmatched ground truth
    aps[catId] = computeAP_v2(dm, num_gt)
    #ars[catId] = computeAR_2(dm, num_gt)

  return aps


def computeAP_v2(lists, total_gt):

  rels = 0
  rank = 0
  
  score = 0.0
  for one in lists:
    rank += 1
    if one >= 0:
      rels += 1
      score += rels / float(rank)
  if total_gt != 0:
    score /= float(total_gt)
  return score

def match_detection(d, g, ious, iou_thres=0.5):
  D = len(d)
  G = len(g)
  # < 0 to note it is not matched, once matched will be the index of the d
  gtm = -np.ones((G)) # whether a gt box is matched
  dtm = -np.ones((D))

  # for each detection bounding box (ranked), will get the best IoU
  # matched ground truth box
  for didx, _ in enumerate(d):
    iou = iou_thres # the matched iou
    m = -1 # used to remember the matched gidx
    for gidx, _ in enumerate(g):
      # if this gt box is matched
      if gtm[gidx] >= 0:
        continue

      # the di,gi pair doesn"t have the required iou
      # or not better than before
      if ious[didx, gidx] < iou:
        continue

      # got one
      iou = ious[didx, gidx]
      m = gidx

    if m == -1:
      continue
    gtm[m] = didx
    dtm[didx] = m
  return dtm, gtm



def tpfp_default(det_bboxes, gt_bboxes, iou_thr):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): the detected bbox
        gt_bboxes (ndarray): ground truth bboxes of this image
        gt_ignore (ndarray): indicate if gts are ignored for evaluation or not
        iou_thr (float): the iou thresholds

    Returns:
        tuple: (tp, fp), two arrays whose elements are 0 and 1
    """
    if len(det_bboxes)>0:
      num_dets = det_bboxes.shape[0]
    else:
      det_bboxes = np.asarray(det_bboxes,dtype=np.float)
      num_dets = det_bboxes.shape[0]
    gt_bboxes = np.asarray(gt_bboxes,dtype=np.float)
    num_gts = gt_bboxes.shape[0]
    area_ranges = [(None, None)]
    #num_scales
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((1,num_dets), dtype=np.float32)
    fp = np.zeros((1,num_dets), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        return tp, fp
    if det_bboxes.shape[0] == 0:
      return tp,fp
        
    ious = cpu_bbox_overlaps(det_bboxes[:,:4], gt_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[0, i] = 1
                else:
                        fp[0, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            else:
                fp[0, i] = 1
            
    return tp, fp


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
    return tp_labels,fp_labels,fn_labels


def check_rootfolders(store_name,dataset):
    """Create log and model folder"""
    if dataset == 'virat':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.VIRAT_LOG_DIR
    elif dataset == 'ava':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.AVA_LOG_DIR
    elif dataset == 'ucfsport':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.UCFSPORT_LOG_DIR
    elif dataset == 'jhmdb':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.JHMDB_LOG_DIR
    elif dataset == 'ucf24':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.UCF24_LOG_DIR
    folders_util = [cfg.LOG.ROOT_LOG_DIR, os.path.join(cfg.LOG.ROOT_LOG_DIR, store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def precision_recall(tp_labels,fp_labels,fn_labels,num_class):
  ap = []
  for class_id in range(1,num_class):
    tp = tp_labels[class_id]
    fp = fp_labels[class_id]
    fn = fn_labels[class_id]

    '''if np.sum(tp) == 0:
            ap.append(0)
    else:'''
   
    precision = tp.astype(float) / (
    tp + fp + 0.00001
       )
    recall = tp.astype(float) / (tp + fn + 0.00001)
    ap.append(compute_ap(recall[::-1], precision[::-1]))
    #print("Both positives and gt are zero")
  return ap
    
def compute_ap(recall, precision):
    """Compute Average Precision according to the definition in VOCdevkit.
  Precision is modified to ensure that it does not decrease as recall
  decrease.
  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls
  Raises:
    ValueError: if the input is not of the correct format
  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.
  """
    if precision is None:
        if recall is not None:
            raise ValueError("If precision is None, recall must also be None")
        return np.NAN

    if not isinstance(precision, np.ndarray) or not isinstance(
        recall, np.ndarray
    ):
        raise ValueError("precision and recall must be numpy array")
    if precision.dtype != np.float or recall.dtype != np.float:
        raise ValueError("input must be float numpy array.")
    if len(precision) != len(recall):
        raise ValueError("precision and recall must be of the same size.")
    if not precision.size:
        return 0.0
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("Precision must be in the range of [0, 1].")
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall must be in the range of [0, 1].")
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError("recall must be a non-decreasing array")

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices]
    )
    return average_precision

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def save_checkpoint(state, filename):
    torch.save(state, filename)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)

    return crops, grid

def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid

def _affine_theta(rois, input_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([\
      (y2 - y1) / (height - 1),
      zero,
      (y1 + y2 - height + 1) / (height - 1),
      zero,
      (x2 - x1) / (width - 1),
      (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta

def log_info(cfg,store_name = None,dataset = None,args = None) :
  if dataset == 'ava':
    log_training = open(os.path.join(cfg.LOG.AVA_LOG_DIR, store_name, 'log.csv'), 'w')
    with open(os.path.join(cfg.LOG.AVA_LOG_DIR, store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    logger = SummaryWriter(log_dir=os.path.join(cfg.LOG.AVA_LOG_DIR, store_name))
  elif dataset == 'virat':
    log_training = open(os.path.join(cfg.LOG.VIRAT_LOG_DIR, store_name, 'log.csv'), 'w')
    with open(os.path.join(cfg.LOG.VIRAT_LOG_DIR, store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    logger = SummaryWriter(log_dir=os.path.join(cfg.LOG.VIRAT_LOG_DIR, store_name))
    
  elif dataset == 'ucfsport':
    log_training = open(os.path.join(cfg.LOG.UCFSPORT_LOG_DIR, store_name, 'log.csv'), 'w')
    with open(os.path.join(cfg.LOG.UCFSPORT_LOG_DIR, store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    logger = SummaryWriter(log_dir=os.path.join(cfg.LOG.UCFSPORT_LOG_DIR, store_name))
  elif dataset == 'jhmdb':
    log_training = open(os.path.join(cfg.LOG.JHMDB_LOG_DIR, store_name, 'log.csv'), 'w')
    with open(os.path.join(cfg.LOG.JHMDB_LOG_DIR, store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    logger = SummaryWriter(log_dir=os.path.join(cfg.LOG.JHMDB_LOG_DIR, store_name))
  elif dataset == 'ucf24':
    log_training = open(os.path.join(cfg.LOG.UCF24_LOG_DIR, store_name, 'log.csv'), 'w')
    with open(os.path.join(cfg.LOG.UCF24_LOG_DIR, store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    logger = SummaryWriter(log_dir=os.path.join(cfg.LOG.UCF24_LOG_DIR, store_name))
  
  return log_training,logger



def print_ap(tp_labels,fp_labels,fn_labels,num_class,log,step,epoch,dictio,ap):
  
      for n in range(1,num_class):
        for key, v in dictio.items(): #modfieed
          if v == n :
            print(f"Class '{n}' ({key}) - AveragePrecision: {ap[n-1]}")
            ap_out = ('Class {0} ({1}) - AveragePrecision: {2}').format(n,key,ap[n-1])
            log.write(ap_out + '\n')
      output = (' completed step:{0}\n'
              'tp_labels: {1}\n'
              'fp_labels: {2} \n' 
              'fn_labels: {3} \n'          
                           .format(step,tp_labels,fp_labels,fn_labels)) 
                                         
      #print the mean average precision      
      print(f"mAP for epoch [{epoch}]: {mean(ap)}")
      mean_outap = ('mAP for epoch [{0}]: {1}'.format(epoch,mean(ap)))
      log.write(output + '\n' + mean_outap + '\n')
      log.flush()
    