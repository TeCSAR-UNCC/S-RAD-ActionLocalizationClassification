
#import datasets
import os
from numpy.random import randint
import numpy as np
import cv2 

from cv2 import imread
import torch.utils.data
from torch.utils.data import Dataset

imfallactivity2id = {
    "BG": 0,  # background
    "Fall": 1,
    "NonFall": 2,
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

class imvia_fd(Dataset):
    def __init__(self,cfg, PHASE = 'train',num_segments = 8,interval = 3,dense_sample = False,
             uniform_sample=True,random_sample = False,strided_sample = False,
             transform= None):
        
        self.cfg = cfg
        self.interval = interval
        self.num_segments = num_segments
        self.dense_sample = dense_sample
        self.uniform_sample = uniform_sample
        self.random_sample= random_sample
        self.strided_sample = strided_sample
        self.num_classes = cfg.IMFD.NUM_CLASSES
        self.framelist = cfg.IMFD.FRAME_LIST_DIR
        self.transform = transform
        self.PHASE = PHASE
        
        if PHASE=='train':
            self.list_file = self.framelist + 'Trainlist.txt' # you need a full path for image list and data path
        else:
            self.list_file = self.framelist + 'Testlist.txt'

        self._data_path = cfg.IMFD.FRAME_DIR
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
            return np.array(offsets)-1
        elif self.uniform_sample:  # normal sample
            average_duration = (record.num_frames) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                       size=self.num_segments)
            return np.abs(offsets-1)
        elif self.random_sample:
            offsets = np.sort(randint(record.num_frames + 1, size=self.num_segments))
            return offsets -1
        elif self.strided_sample:
            average_duration = (record.num_frames) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + average_duration//2
            return offsets-1
        else:
            offsets = np.zeros((self.num_segments,))    
            return offsets-1
        
    def get(self,index,record, indices):
        """
        Extract the gt boxes,labels from the file list
        """
        gt = np.zeros((self.num_segments,self.cfg.MAX_NUM_GT_BOXES,(self.num_classes + 4)),
                  dtype=np.float32)
        num_boxes = np.ones((self.num_segments),dtype=np.float32)
        im_info = np.zeros((self.num_segments,3),dtype=np.float32)
        one_hot_labels = np.zeros((self.num_classes),dtype = np.float)
        count = 0
        images =[]

        labels = int(record.labels)
        one_hot_labels[labels] = 1
        ann_file = record.path + '/' + 'gt.txt'
        if record.path.split('/')[5] == 'Fall':  
             start_index = self.read_annot(ann_file)
        else:
              start_index = int((sorted(os.listdir(record.path))[0].split('_')[1]).split('.')[0])
 
        for seg_ind in indices:

            #image information 
            image_path = os.path.join(record.path, 'frame_{:05d}.jpg'.format(seg_ind+start_index))
            im = imread(image_path)
            im = im[:,:,::-1].astype(np.float32, copy=False) #RGB
            height,width,_= im.shape 
            im_scale = float(self.cfg.TRAIN.TRIM_HEIGHT) / float(self.cfg.TRAIN.TRIM_WIDTH)
            im = cv2.resize(im, (400,300), fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
            im_scale1 = float(self.cfg.TRAIN.TRIM_HEIGHT) / height
            im_scale2 = float(self.cfg.TRAIN.TRIM_WIDTH) / width
            im_info[count,:]=self.cfg.TRAIN.TRIM_HEIGHT,len(im[2]),im_scale
            x1,y1,x2,y2 = self.get_annot(ann_file,image_path)
            gt[count,:,:4] =x1*im_scale2,y1*im_scale1,x2*im_scale2,y2*im_scale1
            gt[count,:,4:] = one_hot_labels
            count += 1
            images.append(im)
        
    
        max_shape = np.array([imz.shape for imz in images]).max(axis=0)
        blob = np.zeros((len(images), max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
        for i in range(len(images)):
           blob[i,0:images[i].shape[0], 0:images[i].shape[1], :] = images[i]

        process_data = self.transform(blob)
        return process_data,gt,num_boxes,im_info
    
    def read_annot(self,ann_file):
        with open(ann_file,'r') as f1:
            line1=next(f1)
        return int(line1.strip())
    
    
    def get_annot(self,ann_file,image_path):
        with open (ann_file,"r") as f1:
         if self.PHASE == 'train' and image_path.split('/')[5] == 'Fall':
           for line in f1.readlines()[2:]:               
              if int(line.split(',')[0])==int(image_path.split('/')[7].split('_')[1].split('.')[0]):
                gt_boxes=np.asarray(line.strip().split(',')[2:],dtype=float)

         else:
            if image_path.split('/')[5] == 'Fall':
               for line in f1.readlines()[2:]: 
                   if line.split(' ')[0].split('/')[3] == image_path.split('/')[7]:
                        gt_boxes = np.asarray(line.strip().split(' ')[1:],dtype=float)
            else:
                for line in f1:
                    if line.split(' ')[0].split('/')[3] == image_path.split('/')[7]:
                        gt_boxes = np.asarray(line.strip().split(' ')[1:],dtype=float)
        return gt_boxes
   
    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record)
        segment_indices = np.sort(segment_indices)
        return self.get( index, record, segment_indices)
               
    def __len__(self):
        return (len(self.video_list))