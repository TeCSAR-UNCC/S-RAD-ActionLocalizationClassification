
#import datasets
import os
from numpy.random import randint
import random
import numpy as np
import cv2 

from cv2 import imread
import torch.utils.data
from torch.utils.data import Dataset

act2id = {
    "BG": 0,  # background
    "Diving": 1,
    "Golf": 2,
    "Kicking": 3,
    "Lifting": 4,
    "Riding": 5,
    "Run":6,
    "SkateBoarding":7,
    "Swing1":8,
    "Swing2":9,
    "Walk":10
}



class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])
    
    @property
    def labels(self):
        return self._data[2]

class ucfsports(Dataset):
    def __init__(self,cfg, image_set, PHASE = 'train',num_segments = 8,interval = 3,dense_sample = False,
             uniform_sample=True,random_sample = False,strided_sample = False, is_input_sampling = True,
             pathway ='two_pathway',transform= None):
        
        self.cfg = cfg
        self.interval = interval
        self.pathway = pathway
        self.num_segments = num_segments
        self.dense_sample = dense_sample
        self.uniform_sample = uniform_sample
        self.random_sample= random_sample
        self.strided_sample = strided_sample
        self.is_input_sampling = is_input_sampling
        self.num_classes = cfg.UCFSPORT.NUM_CLASSES
        self.framelist = cfg.UCFSPORT.FRAME_LIST_DIR
        self.transform = transform
        
        if PHASE=='train':
            self.list_file = self.framelist + 'train_list.txt' # you need a full path for image list and data path
        else:
            self.list_file = self.framelist + 'test_list.txt'

        self._annot_path = cfg.UCFSPORT.ANNOTATION_DIR # you only have annotations in RGB data folder
        self._data_path = cfg.UCFSPORT.FRAME_DIR
        self._classes = ('__background__', 
                         'Diving', 'Golf', 'Kicking', 'Lifting', 'Riding', 
                         'Run', 'SkateBoarding', 'Swing1', 'Swing2', 'Walk')

        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        self._parse_list()
        
    def _parse_list(self):
        """
        Parse the video info from the list file
        """
        frame_path = [x.strip().split(' ') for x in open(self.list_file)]  
        self.video_list = [VideoRecord(item) for item in frame_path]
        print('Sequence number/ video number:%d' % (len(self.video_list)))
        
    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)+1
        elif self.uniform_sample:  # normal sample
            average_duration = (record.num_frames) // self.num_segments
            if self.pathway == "two_pathway":
                if record.num_frames < 80:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration)+average_duration
                    assert offsets[-1] <= record.num_frames
                    return offsets
                else:
                    mid_frame_ind = record.num_frames // 2
                    offsets = list(range(mid_frame_ind - 12, mid_frame_ind + 12, 3))
                    return offsets
            else:
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                       size=self.num_segments)
                    return offsets+1

        elif self.random_sample:
            offsets = np.sort(randint(record.num_frames + 1, size=self.num_segments))
            return offsets+1 
        elif self.strided_sample:
            average_duration = (record.num_frames) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + average_duration//2
            return offsets+1
        else:
            offsets = np.zeros((self.num_segments,))    
            return offsets+1  
    
    def get_all_frames(self,index,record):
        #initialise the gt,num_boxes,im_info and one hot labels
        gt = np.zeros((self.num_segments,self.cfg.MAX_NUM_GT_BOXES,(self.num_classes + 4)),
                  dtype=np.float32)
        num_boxes = np.ones((self.num_segments),dtype=np.float32)
        im_info = np.zeros((self.num_segments,3),dtype=np.float32)
        one_hot_labels = np.zeros((self.num_classes),dtype = np.float)
        count = 0  #to traverse between the segments

        #get the imageindex and labels from the framelist 
        im_split = (record.path).split('/')
        num_parts = len(im_split)
        im_ind = int(im_split[num_parts-1][0:6])

        #get the class label and convert to one hot
        class_label_id = act2id[record.labels]
        one_hot_labels[class_label_id] = 1
        
        #annotation file and image folder to get the clips
        ann_file =self._annot_path +im_split[5]+ '.txt'
        Lines = open(ann_file, 'r').readlines()
        img_folder = os.path.join(self._data_path, im_split[5])
        max_num = len(os.listdir(img_folder)) - 1
        
        clip = []

        d = self.interval
        path = []
        for i in reversed(range(self.num_segments)):
        # make it as a loop
           i_temp = im_ind - i * d
           while i_temp < 1:
            i_temp = max_num + i_temp
           while i_temp > max_num:
            i_temp = i_temp - max_num
           path.append(i_temp)
           
           #prepare the image/frame
           path_tmp = self._data_path + '/'+im_split[5]+'/'+('{:06d}.jpg'.format(i_temp))
           #print("image split0:{0} imagesplit1: {1} frameindex: {2}\n".format(im_split[0], im_split[1] ,'{:05d}.png'.format(i_temp)))
           #print("maximum_frames : {}".format(max_num))
           #print(path_tmp)
           im = imread(path_tmp)
           if im is None:
               print("caught")
           im = im[:,:,::-1].astype(np.float32, copy=False) #RGB
           height,width,_= im.shape 
           im_scale = float(self.cfg.TRAIN.TRIM_HEIGHT) / float(self.cfg.TRAIN.TRIM_WIDTH)
           im = cv2.resize(im, (400,300), fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
           im_scale1 = float(self.cfg.TRAIN.TRIM_HEIGHT) / height
           im_scale2 = float(self.cfg.TRAIN.TRIM_WIDTH) / width
           im_info[count,:]=im.shape[0],im.shape[1],im_scale
           
           #prepare the gt boxes 
           if len(Lines[0].split()) == 5:
            # gt boxes and labels per image
               x,y,w,h = [line.strip().split()[1:] for line in Lines if int((str(line).split())[0]) == im_ind][0]
               x2 = int(x)+ int(w)
               y2 = int(y) + int(h)
               y,y2 = int(y)*im_scale1,y2*im_scale1
               x,x2 = int(x)*im_scale2,x2*im_scale2
               gt[count,0,:4] = int(x),int(y),x2,y2
               gt[count,0,4:] = one_hot_labels
           else : 
               data1 =[(line.split())[1:5] for line in Lines if int((str(line).split())[0]) == im_ind][0]
               xf,yf,wf,hf = [int(tup) for tup in data1]
               data2 =[(line.split())[5:] for line in Lines if int((str(line).split())[0]) == im_ind][0]
               xs,ys,ws,hs = [int(tup) for tup in data2]
               gt[count,0,:4]= xf*im_scale2,yf*im_scale1,(wf+xf)*im_scale2,(yf+hf)*im_scale1
               gt[count,1,:4]= xs*im_scale2,ys*im_scale1,(ws+xs)*im_scale2,(ys+hs)*im_scale1
               num_boxes[count] *= 2
               gt[count,:,4:] = one_hot_labels
            #gt[count,:,:4] = gt[count,:,:4]*im_scale
           count += 1
           clip.append(im)
        
        max_shape = np.array([imz.shape for imz in clip]).max(axis=0)
        blob = np.zeros((len(clip), max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
        for i in range(len(clip)):
           blob[i,0:clip[i].shape[0], 0:clip[i].shape[1], :] = clip[i]

        process_data = self.transform(blob)
        return process_data,gt,num_boxes,im_info
   
        

    #def get(self,index,record,indices,num_segments):
    def get_input_keyframe(self,record,indices):
        """
        Extract the gt boxes,labels from the file list
        """
        video_id = str(record.path).strip().split('/Frames/')[1]
        ann_file = self._annot_path + video_id +'.txt' 
        
        
        gt = np.zeros((self.num_segments,self.cfg.MAX_NUM_GT_BOXES,(self.num_classes + 4)),
                  dtype=np.float32)
        num_boxes = np.ones((self.num_segments),dtype=np.float32)
        im_info = np.zeros((self.num_segments,3),dtype=np.float32)
                
        one_hot_labels = np.zeros((self.num_classes),dtype = np.float)
        count = 0
        images =[]

        labels = open(self.list_file, 'r').readlines()
        class_label =self._class_to_ind[([label.split()[2] for label in labels if label.split()[0].split('/Frames/')[1] == video_id])[0]]
        one_hot_labels[class_label] = 1
        Lines = open(ann_file, 'r').readlines() 
           
        for seg_ind in indices:

            #image information 
            image_path = os.path.join(record.path, '{:06d}.jpg'.format(seg_ind))
            im = imread(image_path)
            im = im[:,:,::-1].astype(np.float32, copy=False) #RGB
            height,width,_= im.shape 
            im_size_min= min(height,width)
            im_size_max=max(height,width)
            #im_scale1 = float(self.cfg.TRAIN.TRIM_HEIGHT) / float(im_size_min)
            #im_scale2 = float(self.cfg.TRAIN.TRIM_WIDTH) / float(im_size_max)
            im_scale = float(self.cfg.TRAIN.TRIM_HEIGHT) / float(self.cfg.TRAIN.TRIM_WIDTH)
            im = cv2.resize(im, (400,300), fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
            im_scale1 = float(self.cfg.TRAIN.TRIM_HEIGHT) / height
            im_scale2 = float(self.cfg.TRAIN.TRIM_WIDTH) / width
            
            #im = cv2.resize(im, None, None, fx=im_scale1, fy=im_scale2,
            #        interpolation=cv2.INTER_LINEAR)
            im_info[count,:]=self.cfg.TRAIN.TRIM_HEIGHT,len(im[2]),im_scale
            if len(Lines[0].split()) == 5:
            # gt boxes and labels per image
               x,y,w,h = [line.strip().split()[1:] for line in Lines if int((str(line).split())[0]) == seg_ind][0]
               x2 = int(x)+ int(w)
               y2 = int(y) + int(h)
               y,y2 = int(y)*im_scale1,y2*im_scale1
               x,x2 = int(x)*im_scale2,x2*im_scale2
               gt[count,0,:4] = int(x),int(y),x2,y2
               gt[count,0,4:] = one_hot_labels
            else : 
               data1 =[(line.split())[1:5] for line in Lines if int((str(line).split())[0]) == seg_ind][0]
               xf,yf,wf,hf = [int(tup) for tup in data1]
               data2 =[(line.split())[5:] for line in Lines if int((str(line).split())[0]) == seg_ind][0]
               xs,ys,ws,hs = [int(tup) for tup in data2]
               gt[count,0,:4]= xf*im_scale2,yf*im_scale1,(wf+xf)*im_scale2,(yf+hf)*im_scale1
               gt[count,1,:4]= xs*im_scale2,ys*im_scale1,(ws+xs)*im_scale2,(ys+hs)*im_scale1
               num_boxes[count] *= 2
               gt[count,:,4:] = one_hot_labels
            #gt[count,:,:4] = gt[count,:,:4]*im_scale
            count += 1
            images.append(im)
        
    
        max_shape = np.array([imz.shape for imz in images]).max(axis=0)
        blob = np.zeros((len(images), max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
        for i in range(len(images)):
           blob[i,0:images[i].shape[0], 0:images[i].shape[1], :] = images[i]

        process_data = self.transform(blob)
        return process_data,gt,num_boxes,im_info
   
    def __getitem__(self, index):
        record = self.video_list[index]
        #self.yaml_file(index)
        segment_indices = self._sample_indices(record)
        segment_indices = np.sort(segment_indices)
        '''Changes made here for slowfast'''
        if self.pathway == 'two_pathway':
          segment_adj_indices = self.get_adj_frames(record,segment_indices)
          #
          return self.two_branch_input(record,segment_indices,segment_adj_indices)
        else:
          return self.get_input_keyframe(record,segment_indices)

    def two_branch_input(self,record,segment_indices,segment_adj_indices):
        two_branch_list =[]
        two_branch_list.append(self.get_input_keyframe(record,segment_indices))
        two_branch_list.append(self.get_input_adjframe(record,segment_adj_indices))
        return two_branch_list
     
    def get_input_adjframe(self,record,indices):
        """
        Extract the gt boxes,labels from the file list
        """
        video_id = str(record.path).strip().split('/Frames/')[1]
        images =[]

        for seg_ind in indices:

            #image information 
            image_path = os.path.join(record.path, '{:06d}.jpg'.format(seg_ind))
            im = imread(image_path)
            im = im[:,:,::-1].astype(np.float32, copy=False) #RGB
            height,width,_= im.shape 
            im_size_min= min(height,width)
            im_size_max=max(height,width)
            im_scale = float(self.cfg.TRAIN.TRIM_HEIGHT) / float(self.cfg.TRAIN.TRIM_WIDTH)
            im = cv2.resize(im, (400,300), fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
            images.append(im)
        
    
        max_shape = np.array([imz.shape for imz in images]).max(axis=0)
        blob = np.zeros((len(images), max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
        for i in range(len(images)):
           blob[i,0:images[i].shape[0], 0:images[i].shape[1], :] = images[i]

        process_data = self.transform(blob)
        return process_data

    def get_adj_frames(self,record,segment_indices):
        new_adj_indices =[]
        average_duration = (record.num_frames) // self.num_segments
        #if average_duration < 3:
        #if record.num_frames > 80:
        for seg in segment_indices:
            new_adj_indices.append(np.abs(seg-2))
            new_adj_indices.append(np.abs(seg-1))
        new_indices = [1 if i == 0 else i for i in new_adj_indices]
        #else:
        ''''for ind in range(len(segment_indices)):
                if ind == 0:
                    new_adj_indices.extend(random.sample(list(range(segment_indices[ind])),2))
                    continue
                new_adj_indices.extend(random.sample(list(range(segment_indices[ind-1]+1,segment_indices[ind])),2))
            new_indices = [1 if i == 0 else i for i in new_adj_indices]'''

        return new_indices

               
    def __len__(self):
        return (len(self.video_list))

    def clip_boxes(boxes, im_shape):
     """
     Clip boxes to image boundaries.
     """
     
     boxes[boxes < 0] = 0
     batch_x = im_shape[:, 1] - 1
     batch_y = im_shape[:, 0] - 1

     boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
     boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
     boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
     boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

     return boxes