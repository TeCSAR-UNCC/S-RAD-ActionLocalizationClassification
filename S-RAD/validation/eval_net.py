import time
import cv2
import os

from lib.model.utils.net_utils import *
from lib.model.utils.confusion_matrix import *
from data_loader.data_loader import *

from lib.model.roi_layers import nms

from lib.model.rpn.bbox_transform import bbox_transform_inv,cpu_bbox_overlaps_batch, \
          bbox_transform_batch,cpu_bbox_overlaps,clip_boxes

@torch.no_grad()
def validate_voc(val_loader,S_RAD,epoch,num_class,num_segments,session,
             batch_size,cfg,log,dataset,pathway,eval_metrics):
    val_iters_per_epoch = int(np.round(len(val_loader)))
    S_RAD.eval()
    all_boxes = [[[[]for _ in range(num_class)] for _ in range(batch_size *num_segments)]
               for _ in range(val_iters_per_epoch)]
    bbox = [[[[]for _ in range(num_class)] for _ in range(batch_size *num_segments)]
               for _ in range(val_iters_per_epoch)]
    #limit the number of proposal per image across all the class
    max_per_image = cfg.MAX_DET_IMG

    #confusion matrix
    conf_mat = ConfusionMatrix(num_classes = num_class, CONF_THRESHOLD = 0.8, IOU_THRESHOLD = 0.2 ,dataset = dataset)
 
    num_gt = [0 for _ in range(num_class)] 
    
    #data_iter = iter(val_loader)   
    for step,data in enumerate(val_loader):
                   
        #evaluate /inference code
        #start_time = time.time()
        rois, cls_prob, bbox_pred = S_RAD(data)
        #torch.cuda.synchronize()
        #end_time = time.time() - start_time
        
        if dataset == 'ucfsport':
         class_dict = act2id
        elif dataset == 'jhmdb':
          class_dict = jhmdbact2id
        elif dataset == 'ucf24':
          class_dict = ucf24act2id
        elif dataset == 'urfall':
          class_dict = fallactivity2id
        elif dataset == 'imfd':
          class_dict = imfallactivity2id
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(scores.shape[0], -1, 4 * num_class)
        
       #transforms the image to x1,y1,x2,y2, format and clips the coord to images
        pred_boxes = bbox_transform_inv(boxes, box_deltas,scores.shape[0])
        if pathway =="two_pathway":
          im_info = data[0][3].view(-1,3).to(device="cuda")
          gt_boxes= (data[0][1].view(-1,cfg.MAX_NUM_GT_BOXES,num_class+4)).to(device="cuda")
        else:
          im_info = data[3].view(-1,3).to(device="cuda")
          gt_boxes= (data[1].view(-1,cfg.MAX_NUM_GT_BOXES,num_class+4)).to(device="cuda")
          pred_boxes = clip_boxes(pred_boxes, im_info.data,scores.shape[0]) 

        #gt boxes 
        gtbb = gt_boxes[:,:,0:4]
        gtlabels = gt_boxes[:,:,4:]
        
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
            if eval_metrics:
              if len(bbox[step][image][class_id])>0 and len(all_boxes[step][image][class_id])>0:
                conf_mat.process_batch(all_boxes[step][image], bbox[step][image])
    
    if eval_metrics:
      result = conf_mat.return_matrix()
      print(result)
      conf_mat.plot(result)
       
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
      
      #ROC curve visualisation
      if eval_metrics:
        import matplotlib.pyplot as plt
        colors = ['ac','navy','gold','turquoise', 'red','green','black',
             'brown','darkorange', 'cornflowerblue', 'teal']
        plt.plot(recalls[0, :],precisions[0, :],color=colors[cls_id],
              lw =2,label='class {}'.format(cls_id))
      
      ap[cls_id] = average_precision(recalls[0, :], precisions[0, :],mode ='area')
    
    #Plot ROC Curve
    if eval_metrics:
      fig = plt.gcf()
      fig.subplots_adjust(bottom=0.25)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('Extension of Precision-Recall curve to multi-class')
      plt.legend(loc="best")
      plt.show()

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
def validate_virat(val_loader,S_RAD,epoch,num_class,num_segments,vis,session,
             batch_size,input_data,cfg,log,dataset):
    val_iters_per_epoch = int(np.round(len(val_loader)))
    im_data,im_info,num_boxes,gt_boxes = input_data
    S_RAD.eval()
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
    for step,data in enumerate(val_loader):
       
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
        rois, cls_prob, bbox_pred = S_RAD(im_data, im_info, gt_boxes, num_boxes)
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
