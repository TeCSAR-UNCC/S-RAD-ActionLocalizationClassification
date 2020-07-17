import numpy as np
from model.rpn.bbox_transform import cpu_bbox_overlaps

class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD = 0.3, IOU_THRESHOLD = 0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
    
    def process_batch(self, detections, labels,class_id):
        '''
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        '''
        detect_class = np.full((detections.shape[0],1),class_id)
        detections = np.concatenate((detections,detect_class),axis=1)
        if labels: 
            
            labels =np.array(labels)
            label_class = np.full((labels.shape[0],1),class_id)
            labels = np.concatenate((label_class,labels),axis=1)
        

        detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        gt_classes = labels[:, 0].astype(np.int16)
        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = cpu_bbox_overlaps(detections[:, :4],labels[:, 1:])
        all_ious = np.transpose(all_ious)
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = []
        for i in range(want_idx[0].shape[0]):
            all_matches.append([want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]])
        
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0: # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index = True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index = True)[1]]


        for i, label in enumerate(labels):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                gt_class = gt_classes[i]
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[(gt_class), detection_class] += 1
            else:
                gt_class = gt_classes[i]
                self.matrix[(gt_class), self.num_classes] += 1
        
        for i, detection in enumerate(detections):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.matrix[self.num_classes ,detection_class] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

