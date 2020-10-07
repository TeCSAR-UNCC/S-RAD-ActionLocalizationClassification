from torch.utils.data import Dataset
from numpy.random import randint

import cv2 
from cv2 import imread

import os, copy
import numpy as np
import scipy.sparse

import pickle
import subprocess
import pdb

ucf24act2id = { 
    "__background__" : 0,
    "Basketball": 1,
    "BasketballDunk" : 2,
    "Biking" : 3,
    "CliffDiving" : 4,
    "CricketBowling" : 5,
    "Diving" : 6,
    "Fencing": 7,
    "FloorGymnastics" : 8,
    "GolfSwing" : 9,
    "HorseRiding" : 10,
    "IceDancing" : 11,
    "LongJump" : 12,
    "PoleVault" : 13,
    "RopeClimbing" : 14,
    "SalsaSpin": 15,
    "SkateBoarding" : 16,
    "Skiing" : 17,
    "Skijet" : 18,
    "SoccerJuggling" : 19,
    "Surfing" : 20,
    "TennisSwing" : 21,
    "TrampolineJumping" : 22,
    "VolleyballSpiking" : 23,
    "WalkingWithDog":24
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

        
class UCF101(Dataset):
    def __init__(self, cfg,
            PHASE = 'train',num_segments=8,dense_sample = False,
            uniform_sample=True,random_sample = False,
            strided_sample = False,is_input_sampling = True,
            transform=None):
        
        self.cfg = cfg
        self.transform = transform
        self.num_segments = num_segments
        self.dense_sample = dense_sample
        self.uniform_sample = uniform_sample
        self.random_sample= random_sample
        self.strided_sample = strided_sample
        self.is_input_sampling = is_input_sampling
        self.mat_annot_file = cfg.UCF24.ANNOTATION_DIR
        self._USE_MAT_GT = self.mat_annot_file!=None  
        self.split = PHASE 
        
        if PHASE=='train':
            self._image_set = cfg.UCF24.FRAME_LIST_DIR + 'trainlist01.txt' 
            
        else:
            self._image_set = cfg.UCF24.FRAME_LIST_DIR + 'testlist01.txt'
            
        if self.mat_annot_file:
             self._mat_gt_ = self.get_mat_gt(self.mat_annot_file)    
        
        self.num_class = cfg.UCF24.NUM_CLASSES     
        
        self._data_path = cfg.UCF24.FRAME_DIR
        
        self._classes = ('__background__', 
                         'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 
                         'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding',
                         'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin',
                         'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing',
                         'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')
        self._class_to_ind = dict(zip(self._classes, range(self.num_class)))
        
        self._parse_list()

        # Default to roidb handler
        #self._roidb_handler = self.gt_roidb
    
    def _parse_list(self):
        """
        Parse the video info from the list file
        """
        frame_path = [x.strip().split(' ') for x in open(self._image_set)]  
        self.video_list = [VideoRecord(item) for item in frame_path]
        print('Sequence number/ video number:%d' % (len(self.video_list)))
    
    
    def _sample_indices(self, num_frames):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        elif self.uniform_sample:  # normal sample
            if num_frames <= self.num_segments:
                offsets = list(range(num_frames))
                #offsets =[i+1 for i in offsets]
                diff = self.num_segments - num_frames
                add_offset = ([1] * diff)
                add_offset = [off * num_frames for off in add_offset]
                return np.array(offsets + add_offset)
            else:
                average_duration = (num_frames) // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                       size=self.num_segments)
                return offsets
        elif self.random_sample:
            offsets = np.sort(randint(num_frames + 1, size=self.num_segments))
            return offsets 
        elif self.strided_sample:
            average_duration = (num_frames) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + average_duration//2
            return offsets
        else:
            offsets = np.zeros((self.num_segments,))    
            return offsets 
    

    
    def get(self,index,record, indices):
        """
        Extract the gt boxes,labels from the file list
        """
        video_id = str(record.path).strip().split('/frames/')[1]
                  
        gt = np.zeros((self.num_segments,self.cfg.MAX_NUM_GT_BOXES,(self.num_class + 4)),
                  dtype=np.float32)
        num_boxes = np.ones((self.num_segments),dtype=np.float32)
        im_info = np.zeros((self.num_segments,3),dtype=np.float32)
        one_hot_labels = np.zeros((self.num_class),dtype = np.float)
        count = 0
        images =[]

        class_label =int(record.labels)
        one_hot_labels[class_label] = 1
        frame_index = list(self._mat_gt_[video_id].keys())       
           
        for seg_ind in indices:

            #image information 
            cur_frame = frame_index[0]+seg_ind
            image_path = os.path.join(record.path, '{:05d}.jpg'.format(cur_frame))
            im = imread(image_path)
            im = im[:,:,::-1].astype(np.float32, copy=False) #RGB
            height,width,_= im.shape 
            im_scale = float(self.cfg.TRAIN.TRIM_HEIGHT) / float(self.cfg.TRAIN.TRIM_WIDTH)
            im = cv2.resize(im, (400,300), fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
            im_scale1 = float(self.cfg.TRAIN.TRIM_HEIGHT) / height
            im_scale2 = float(self.cfg.TRAIN.TRIM_WIDTH) / width
            im_info[count,:]=self.cfg.TRAIN.TRIM_HEIGHT,len(im[2]),im_scale
            
            gt[count,0,:4] = self._load_UCF101_annotation(video_id,cur_frame,self._mat_gt_) 
            x1,y1,x2,y2 = gt[count,0,:4]
            y1,y2 = y1*im_scale1,y2*im_scale1
            x1,x2 = x1*im_scale2,x2*im_scale2
            gt[count,0,:4] = x1,y1,x2,y2
            #if gt[count,0,:4].any():
            gt[count,0,4:] = one_hot_labels
            #else:
            #    gt[count,0,4:] = np.zeros((1,self.cfg.MAX_NUM_GT_BOXES,self.num_class),dtype = float)
            
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
        #index = 711
        record = self.video_list[index]
        video_id = str(record.path).strip().split('/frames/')[1]
        ann_frame_id= list(self._mat_gt_[video_id].keys())
        average_dur = len(ann_frame_id)
        assert average_dur != 0
        if average_dur < 8:
            print("delete")
        segment_indices = self._sample_indices(average_dur)
        segment_indices = np.sort(segment_indices)
        return self.get( index, record, segment_indices)
               
    def __len__(self):
        return (len(self.video_list))

       
    def get_mat_gt(self, mat_annot_file):
        # parse annot from mat file
        import scipy.io as sio
        f = sio.loadmat(mat_annot_file)['annot'][0]

        mat_gt = {}
        n_total = f.shape[0]
        for i in range(n_total):
            videoname = str(f[i][1][0])
            mat_gt[videoname] = {}
            ef = f[i][2][0][0][0][0,0]
            sf = f[i][2][0][0][1][0,0]
            for framenr in range(sf, ef+1):
                mat_gt[videoname][framenr] = f[i][2][0][0][3][framenr-sf,:].astype(np.int32)

        return mat_gt


    def _load_UCF101_annotation(self, videoname=None,frm= 0,gtfile = None):
        #index = index.split(',')[-1] # to support 2 stream filelist input
        #videoname = os.path.dirname(index) 
        #frm = int(index.split('/')[-1].split('.')[0])
        if self._USE_MAT_GT:
            if videoname in gtfile and frm in gtfile[videoname]:
                boxes = gtfile[videoname][frm]
                if boxes.ndim==1:
                    boxes = boxes[np.newaxis, :]
                    boxes[:,2] += (boxes[:,0]-1)
                    boxes[:,3] += (boxes[:,1]-1)
                    if not (boxes[:, 2] >= boxes[:, 0]).all():
                        print("wrong")
                        print (index)
                        print (boxes)
                else:
                    pdb.set_trace()
            else:
                # print '{} {} has no box'.format(videoname, frm)
                boxes = np.zeros((4), dtype=np.int32) 
        return boxes

    