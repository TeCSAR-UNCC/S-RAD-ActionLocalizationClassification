import numpy as np
from lib.model.rpn.bbox_transform import cpu_bbox_overlaps
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD = 0.3, IOU_THRESHOLD = 0.5 , dataset = 'None'):
        self.matrix = np.zeros((num_classes+1 , num_classes+1 ))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
        self.dataset = dataset
    
    def plot(self,matrix):
        '''
        plot confusion matrix
        '''
        if self.dataset == 'ucfsport':
            List1 = ["Diving","Golf","Kicking","Lifting","Riding","Run","Skate",
              "Swing1","Swing2","Walk"]
            List2 = ["Diving","Golf","Kicking","Lifting","Riding","Run","Skate",
              "Swing1","Swing2","Walk","FalseNeg"]
            df_cm = pd.DataFrame(matrix[1:self.num_classes,1:self.num_classes+1], index = [i for i in List1],
                  columns = [i for i in List2])
        elif self.dataset == 'urfall':
            List1 = ["Fall","Not-Fall"]
            List2 = ["Fall", "Not-Fall"]
            df_cm = pd.DataFrame(matrix[1:self.num_classes,1:self.num_classes], index = [i for i in List1],
                  columns = [i for i in List2])
        else :
            print("undefined dataset for eval metrics : available only for ucfsport and urfall")
        
        plt.figure(figsize = (10,7))
        sn.set(font_scale=1.6) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 20},cmap="Blues",linewidths=0.05, linecolor='black') # font size
        
        plt.ylabel("Detected action class")
        plt.xlabel("Groundtruth action class")
        plt.show()


    
    def process_batch(self, detections, labels):
        '''
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        '''
        assert len(detections)==len(labels)
        final_det = []
        final_gt =[]
        for i in range(1,(self.num_classes)):
            '''
            append classid to detections -x1,y1,x2,y2,scores,classid
            gt - x1,y1,x2,y2,classid
            '''
            if len(detections[i]):
              for det in detections[i]:
                detect_w_class = np.append(det,i)
                final_det.append(detect_w_class)
            if len(labels[i]): 
              for single_gt in labels[i]:            
                label_w_class = np.append(single_gt,i)
                final_gt.append(label_w_class)
        detection = np.vstack(final_det)
        gt = np.vstack(final_gt)
        gt_classes = gt[:, 4].astype(np.int16)
        if self.dataset == 'urfall':
            detection = detection[np.argmax(detection[:, 4]),np.newaxis]
            detection_classes = detection[:, 5].astype(np.int16)
            if gt_classes[0] == detection_classes[0]:
                self.matrix[(detection_classes[0]), gt_classes[0]] += 1
            else:
                self.matrix[(detection_classes[0]), gt_classes[0]] += 1
        else : 
            detection = detection[detection[:, 4] > self.CONF_THRESHOLD]
            detection_classes = detection[:, 5].astype(np.int16)
            all_ious = cpu_bbox_overlaps(detection[:, :4],gt[:, :4])
            #all_ious = np.transpose(all_ious)
            ious_max = all_ious.max(axis=1)
            ious_argmax = all_ious.argmax(axis=1)
            #sort_inds = np.argsort(-detection[:, 4])
            gt_covered = np.zeros(len(gt), dtype=bool)
            for i in range(detection.shape[0]):
                if ious_max[i] >= self.IOU_THRESHOLD:
                    matched_gt = ious_argmax[i]
                    if not gt_covered[matched_gt]:
                        if gt_classes[0] == detection_classes[i]:
                            gt_covered[matched_gt] = True
                            detection_class = detection_classes[i]
                            gt_class = gt_classes[0]
                            self.matrix[(gt_class), detection_class] += 1

                    else:
                    #False positive if belong to same class as gt
                    #it belongs to falsepos row
                    
                        detection_class = detection_classes[i]
                    #gt_class = gt_classes[i]
                        if gt_class == detection_class:
                                self.matrix[(self.num_classes), detection_class] += 1
                        else:
                                self.matrix[(gt_class), detection_class] += 1
                
                else:
                    detection_class = detection_classes[i]
                    gt_class = gt_classes[0]
                    self.matrix[(gt_class), detection_class] += 1
                                        
                if gt_covered.all() == 0:
                
                #Unmatched groundtruth i.e box not detected (FN)
                
                    for i in range(len(gt)):
                        if gt_covered[i] == 0:
                            gt_class = gt_classes[0]
                            self.matrix[(gt_class), self.num_classes] += 1

        
    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

