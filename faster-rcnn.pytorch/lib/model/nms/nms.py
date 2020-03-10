import numpy as np


def soft_nms(boxes,labels, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    N = boxes.shape[0]
    labels = labels.cpu().numpy()
    boxes = boxes.detach().cpu().numpy()
    pos = 0
    maxscore = []
    maxpos = 0
    ts = []
    
    for i in range(N):
        maxscore = np.max(labels[i, :])
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = labels[i,:]

        pos = i + 1
	# get max box
        '''while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1'''
        try:
            maxscore= np.max(labels[pos:,:])
        except ValueError:  #raised if `y` is empty.
            pass
        maxpos = np.where(labels == maxscore)[0]
        maxpos = maxpos[0]
	# add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        labels[i,:] = labels[maxpos,:]

	# swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        labels[maxpos,:] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = labels[i,:]

        pos = i + 1
	# NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = labels[pos, :]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    labels[pos, :] = weight*labels[pos, :]
		    
		    # if box score falls below threshold, discard the box by swapping with last box
		    # update N
                    if (labels[pos, :].all()) < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        labels[pos,:] = labels[N-1, :]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep



def py_cpu_nms(box,label, thresh):
  """Pure Python NMS baseline."""
  
  final_keep = []
  for j in range(box.shape[0]):
    x1 = box[j,:, 0]
    y1 = box[j,:, 1]
    x2 = box[j,:, 2]
    y2 = box[j,:, 3]
    scores = label[j,:, ]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sc =np.amax(scores,axis =1)
    order = sc.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        #find the overlap of ith proposal with other proposals
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
      
        #find the proposals that have IOU scores < 0.3
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    final_keep.append(keep)
  return final_keep