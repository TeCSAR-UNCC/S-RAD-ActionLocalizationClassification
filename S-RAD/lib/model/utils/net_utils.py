import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from lib.model.utils.config import cfg
import os
import cv2
import pdb
import random
from tensorboardX import SummaryWriter
from statistics import mean 
import operator
from operator import itemgetter

from lib.model.rpn.bbox_transform import cpu_bbox_overlaps

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

def check_rootfolders(store_name,dataset):
    """Create log and model folder"""
    if dataset == 'virat':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.VIRAT_LOG_DIR
    elif dataset == 'ucfsport':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.UCFSPORT_LOG_DIR
    elif dataset == 'jhmdb':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.JHMDB_LOG_DIR
    elif dataset == 'ucf24':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.UCF24_LOG_DIR
    elif dataset == 'urfall':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.URFD_LOG_DIR
    elif dataset == 'imfd':
      cfg.LOG.ROOT_LOG_DIR = cfg.LOG.IMFD_LOG_DIR
    
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
def log_info(cfg,store_name = None,dataset = None,args = None) :
  if dataset == 'virat':
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
  elif dataset == 'urfall':
    log_training = open(os.path.join(cfg.LOG.URFD_LOG_DIR, store_name, 'log.csv'), 'w')
    with open(os.path.join(cfg.LOG.URFD_LOG_DIR, store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    logger = SummaryWriter(log_dir=os.path.join(cfg.LOG.URFD_LOG_DIR, store_name))
  elif dataset == 'imfd':
    log_training = open(os.path.join(cfg.LOG.IMFD_LOG_DIR, store_name, 'log.csv'), 'w')
    with open(os.path.join(cfg.LOG.IMFD_LOG_DIR, store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    logger = SummaryWriter(log_dir=os.path.join(cfg.LOG.IMFD_LOG_DIR, store_name))
 
  return log_training,logger
