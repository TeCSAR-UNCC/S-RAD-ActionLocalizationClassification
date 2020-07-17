import sys, os
from PIL import Image
from scipy.io import loadmat
import torch.utils.data
from torch.utils.data import Dataset
from numpy.random import randint
import numpy as np
import scipy.sparse
import pickle
import pdb

import cv2 
from cv2 import imread

jhmdbact2id = { 
    "__background__": 0,
    "brush_hair" : 1,
    "catch" :2,
                "clap" :3,
                "climb_stairs" : 4,
                "golf" : 5, 
                "jump": 6,
                "kick_ball": 7,
                "pick" : 8,
                "pour" : 9,
                "pullup" : 10,
                "push" : 11,
                "run" : 12,
                "shoot_ball" : 13,
                "shoot_bow" : 14,
                "shoot_gun": 15,
                "sit": 16,
                "stand": 17,
                "swing_baseball": 18,
                "throw": 19,
                "walk": 20,
                "wave": 21
                
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



class JHMDB(Dataset):
    def __init__(self, cfg, image_set,
            PHASE = 'train',num_segments=8,dense_sample = False,
            uniform_sample=True,random_sample = False,
            strided_sample = False,is_input_sampling = True,
            transform=None):
        #imdb.__init__(self, image_set, PHASE)
        if PHASE=='train':
            self._image_set = cfg.JHMDB.FRAME_LIST_DIR + image_set + '.trainlist' # you need a full path for image list and data path
        else:
            self._image_set = cfg.JHMDB.FRAME_LIST_DIR + image_set + '.testlist'
        
        self.cfg = cfg
        self.num_segments = num_segments
        self.dense_sample = dense_sample
        self.uniform_sample = uniform_sample
        self.random_sample= random_sample
        self.strided_sample = strided_sample
        self.is_input_sampling = is_input_sampling
        self._annot_path = cfg.JHMDB.ANNOTATION_DIR
        self._SPLIT = int(image_set.split('_')[-1])
        #print("split used is {}".format(self._SPLIT))

        self.num_classes = cfg.JHMDB.NUM_CLASSES
        self.transform = transform
        self._data_path = cfg.JHMDB.FRAME_DIR
        
        
        self._classes = ('__background__', 
                         'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 
                         'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
                         'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
                         'stand', 'swing_baseball', 'throw', 'walk', 'wave') # 22
        self._class_to_ind = dict(zip(self._classes, range(cfg.JHMDB.NUM_CLASSES)))
        #self._image_index = self._load_image_set_index()

        #self.test_videos = self.get_test_videos(self._SPLIT)

        self._parse_list()
   
   
    def _parse_list(self):
        """
        Parse the video info from the list file
        """
        frame_path = [x.strip().split(' ') for x in open(self._image_set)]  
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
    

    
    def get(self,index,record, indices):
        """
        Extract the gt boxes,labels from the file list
        """
        video_id = str(record.path).strip().split('/frames/')[1]
        
        ann_file = self._annot_path + '/' + str(record.path).strip().split('/frames/')[1] + '/' + 'puppet_mask.mat' 
                
        gt = np.zeros((self.num_segments,self.cfg.MAX_NUM_GT_BOXES,(self.num_classes + 4)),
                  dtype=np.float32)
        num_boxes = np.ones((self.num_segments),dtype=np.float32)
        im_info = np.zeros((self.num_segments,3),dtype=np.float32)
        one_hot_labels = np.zeros((self.num_classes),dtype = np.float)
        count = 0
        images =[]

        
        class_label =int(record.labels)
        one_hot_labels[class_label] = 1
               
           
        for seg_ind in indices:

            #image information 
            image_path = os.path.join(record.path, '{:05d}.png'.format(seg_ind))
            im = imread(image_path)
            im = im[:,:,::-1].astype(np.float32, copy=False) #RGB
            height,width,_= im.shape 
            im_scale = float(self.cfg.TRAIN.TRIM_HEIGHT) / float(self.cfg.TRAIN.TRIM_WIDTH)
            im = cv2.resize(im, (400,300), fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
            im_scale1 = float(self.cfg.TRAIN.TRIM_HEIGHT) / height
            im_scale2 = float(self.cfg.TRAIN.TRIM_WIDTH) / width
            im_info[count,:]=self.cfg.TRAIN.TRIM_HEIGHT,len(im[2]),im_scale
            
            gt[count,0,:4] = self.get_annot_image_boxes(ann_file, seg_ind)
            x1,y1,x2,y2 = gt[count,0,:4]
            y1,y2 = y1*im_scale1,y2*im_scale1
            x1,x2 = x1*im_scale2,x2*im_scale2
            gt[count,0,:4] = x1,y1,x2,y2
            gt[count,0,4:] = one_hot_labels
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
        segment_indices = self._sample_indices(record)
        segment_indices = np.sort(segment_indices)
        return self.get( index, record, segment_indices)
               
    def __len__(self):
        return (len(self.video_list))

    # annotation: warning few images do not have annotation
    def _get_puppet_mask_file(self, videoname):
        return os.path.join(self._annot_path, videoname, "puppet_mask.mat")

    def get_annot_image_mask(self, videoname, n):
        assert os.path.exists(videoname)
        m = loadmat(videoname)["part_mask"]
        if n-1<m.shape[2]: return m[:,:,n-1]>0
        else: return m[:,:,-1]>0

    def get_annot_image_boxes(self, videoname, n):
#        pdb.set_trace()
        mask = self.get_annot_image_mask(videoname, n)
        m = self.mask_to_bbox(mask)
        if m is None:
            pdb.set_trace() 
            m = np.zeros((0,4), dtype=np.float32)
        if m.shape[0]>1:
            pdb.set_trace()
        return m

    def mask_to_bbox(self,mask):
         # you are aware that only 1 box for each frame
        return np.array(Image.fromarray(mask.astype(np.uint8)).getbbox(), dtype=np.float32).reshape(1,4)-np.array([0,0,1,1])

    