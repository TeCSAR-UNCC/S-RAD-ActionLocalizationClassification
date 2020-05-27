import torch.utils.data as data


import os
from os import listdir
from os.path import isfile, join

from PIL import Image
import numpy as np
from numpy.random import randint
import operator
import torch
import cv2
from cv2 import imread

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

activity2id_hard = {
    "BG": 0,  # background
    "activity_gesturing": 1,
    "Closing": 2,
    "Opening": 3,
    "Interacts": 4,
    "Exiting": 5,
    "Entering": 6,
    "Talking": 7,
    "Transport_HeavyCarry": 8,
    "Unloading": 9,
    "Pull": 10,
    "Loading": 11,
    "Open_Trunk": 12,
    "Closing_Trunk": 13,
    "Riding": 14,
    "specialized_texting_phone": 15,
    "Person_Person_Interaction": 16,
    "specialized_talking_phone": 17,
    "activity_running": 18,
    "PickUp": 19,
    "specialized_using_tool": 20,
    "SetDown": 21,
    "activity_crouching": 22,
    "activity_sitting": 23,
    "Object_Transfer": 24,
    "Push": 25,
    "PickUp_Person_Vehicle": 26
    }
    

activity2id = {
    "BG": 0,  # background
    "activity_walking": 1,
    "activity_standing": 2,
    "activity_carrying": 3,
    "activity_gesturing": 4,
    "Closing": 5,
    "Opening": 6,
    "Interacts": 7,
    "Exiting": 8,
    "Entering": 9,
    "Talking": 10,
    "Transport_HeavyCarry": 11,
    "Unloading": 12,
    "Pull": 13,
    "Loading": 14,
    "Open_Trunk": 15,
    "Closing_Trunk": 16,
    "Riding": 17,
    "specialized_texting_phone": 18,
    "Person_Person_Interaction": 19,
    "specialized_talking_phone": 20,
    "activity_running": 21,
    "PickUp": 22,
    "specialized_using_tool": 23,
    "SetDown": 24,
    "activity_crouching": 25,
    "activity_sitting": 26,
    "Object_Transfer": 27,
    "Push": 28,
    "PickUp_Person_Vehicle": 29,
    "vehicle_turning_right": 30,
    "vehicle_moving": 31,
    "vehicle_stopping" : 32,
    "vehicle_starting" :33,
    "vehicle_turning_left": 34,
    "vehicle_u_turn": 35,
    "specialized_miscellaneous": 36,
    "DropOff_Person_Vehicle" : 37,
    "Misc" : 38,
    "Drop" : 39}


class VIRAT_dataset(data.Dataset):
    def __init__(self, train_path,num_class,cfg,list_file,
                 num_segments=3,input_size = 600,transform=None,dense_sample=False,
                 uniform_sample=True,random_sample=False,strided_sample=False):
                 
        self.train_path = train_path
        self.num_class = num_class
        self.list_file = list_file
        self.num_segments = num_segments
        self.transform = transform
        self.new_size = input_size
        self.dense_sample = dense_sample
        self.uniform_sample = uniform_sample
        self.strided_sample = strided_sample
        self.random_sample = random_sample
        self.cfg = cfg
        self._parse_list()

    def _parse_list(self):
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
            return np.array(offsets)
        elif self.uniform_sample:  # normal sample
            average_duration = (record.num_frames) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                       size=self.num_segments)
            return offsets 
        elif self.random_sample:
            offsets = np.sort(randint(record.num_frames + 1, size=self.num_segments))
            return offsets 
        elif self.strided_sample:
            average_duration = (record.num_frames) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + average_duration//2
            return offsets
        else:
            offsets = np.zeros((self.num_segments,))    
            return offsets  

    def get(self,index,record, indices):
      
      sequence_path = str(record.path).strip().split('/frames/')[0]
      label = list()
      bbox = list()
      images = list()
      img_path = list()
      gt = np.zeros((len(indices),self.cfg.MAX_NUM_GT_BOXES,(self.num_class + 4)),
                  dtype=np.float32)
      num_boxes = np.zeros((self.num_segments),dtype=np.float32)
      im_info = np.zeros((self.num_segments,3),dtype=np.float32)
      npy_file = (os.path.join(str(sequence_path),'ground_truth.npy'))
      data = np.load(npy_file)
      frame = data[0][0]
      j =0 
      for seg_ind in indices: #iterate through every image
                    count = 0
                    
                    bboxes = np.zeros((self.cfg.MAX_NUM_GT_BOXES,(self.num_class + 4)),dtype= float)
                    p = int(seg_ind) + int(frame)
                    image_path = os.path.join(record.path, '{:06d}.jpg'.format(p))
                    im = imread(image_path)
                    im = im[:,:,::-1]
                    im = im.astype(np.float32, copy=False)
                    height,width,_= im.shape #h=1080,w=1920
                    im_size_min= min(height,width)
                    im_size_max = max(height,width)
                    im_scale = float(self.new_size) / float(im_size_min)
                    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
                    im_info[j,:]=self.new_size,len(im[2]),im_scale
                    img_path.append(image_path)
                    for i in data:
                        if i[0] == p:
                            bbox_new =[]
                            bbox = (i[2:6])*im_scale
                            bbox_new[0:4] = bbox
                            #change to train only hard class
                            bbox_new[4:] = i[6:7]
                            bbox_new[5:] = i[10:10+self.num_class -1]
                            #bbox_new[4:]=i[6:6+self.num_class]#change here to train only less hard class
                            bboxes[count,:]+=bbox_new
                            count+=1
                            
                    num_boxes[j,]+=count
                    gt[j,:,:] = bboxes
                    j = j+1     
                    images.append(im)
      
      max_shape = np.array([imz.shape for imz in images]).max(axis=0)
      num_images = len(images)
      blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
      for i in range(len(images)):
        im1 = images[i]
        blob[i,0:im1.shape[0], 0:im1.shape[1], :] = im1


      process_data = self.transform(blob)
      return process_data, gt, num_boxes , im_info ,img_path

                          
    def __getitem__(self, index):
        record = self.video_list[index]
        #self.yaml_file(index)
        segment_indices = self._sample_indices(record)
        segment_indices = np.sort(segment_indices)
        return self.get( index, record, segment_indices)
               
    def __len__(self):
        return (len(self.video_list))


#parse the yml file into the variables


