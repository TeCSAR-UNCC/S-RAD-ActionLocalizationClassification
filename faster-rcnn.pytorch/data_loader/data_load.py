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

    '''@property
    def label(self):
        return int(self._data[2])'''

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


class Action_dataset(data.Dataset):
    def __init__(self, train_path,list_file,
                 num_segments=3, modality='RGB',
                 transform=None,random_shift=True, test_mode=False,
                 dense_sample =False,uniform_sample=False,
                 random_sample = False,strided_sample = False, input_size = 600):
                 
        self.train_path = train_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.new_size = input_size
        #self.dense_sample = dense_sample
        self.dense_sample = False
        self.uniform_sample = True
        self.strided_sample = False
        self.random_sample = False
        self._parse_list()

    def _parse_list(self):
        frame_path = [x.strip().split(' ') for x in open(self.list_file)]  
        self.video_list = [VideoRecord(item) for item in frame_path]
        print('Sequence number/ video number:%d' % (len(self.video_list)))
#        for i in range(len(self.video_list)):
#            self.getitem(i)

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
            return np.array(offsets) + 1
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

    def _load_image(self, directory, filename):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(directory, filename)).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(directory, filename))
                return [Image.open(os.path.join(directory, filename)).convert('RGB')]   
    
    
    def get(self,index,record, indices,new_size):
      
      #sequence_path=list()
      #train_file = '/mnt/AI_RAID/VIRAT/actev-data-repo/dataset/train/train_list.txt'
      sequence_path = str(record.path).strip().split('/frames/')[0]
      label = list()
      bbox = list()
      #bbox_new=list()
      images = list()
      img_path = list()
      gt = np.zeros((len(indices),20,44),
                  dtype=np.float32)
      num_boxes = np.zeros((8),dtype=np.float32)
      im_info = np.zeros((8,3),dtype=np.float32)
      #for path in range(len(sequence_path)): #sequence level for loop 
      npy_file = (os.path.join(str(sequence_path),'ground_truth.npy'))
      data = np.load(npy_file)
          
      with open('ground.txt', 'w') as f:
                for n in data:
                    f.write("%s \n" % n)
                frame = data[0][0]
                j =0 
                for seg_ind in indices: #iterate through every image
                    count = 0
                    
                    bboxes = np.zeros((20,44),dtype= float)
                    p = int(seg_ind) + int(frame)
                    #seg_imgs = self._load_image(record.path, '{:06d}.jpg'.format(p))
                    image_path = os.path.join(record.path, '{:06d}.jpg'.format(p))
                    im = imread(image_path)
                    im = im[:,:,::-1]
                    im = im.astype(np.float32, copy=False)
                    height,width,_= im.shape #h=1080,w=1920
                    im_size_min= min(height,width)
                    im_size_max = max(height,width)
                    #new_w = float(im_size_max * new_size) / im_size_min
                    im_scale = float(new_size) / float(im_size_min)
                    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
                    im_info[j,:]=new_size,len(im[2]),im_scale
                    img_path.append(image_path)
                    for i in data:
                        if i[0] == p:
                            bbox_new =[]
                            bbox = (i[2:6])*im_scale
                            #y1,x1,y2,x2=bbox
                            bbox_new[0:4] = bbox
                            #differ=new_y2 - new_y1
                            #diff.append(differ)
                            #label=(i[6:]).tolist()
                            #bbox_new+=label
                            #bboxes+=[bbox_new]
                            #bbox_new[4:44]=label
                            bbox_new[4:]=i[6:]
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

      ''''for y in range(len(data["annotations"][seg_ind]['actions'])):#get the bbox in that frame
                key,value = data["annotations"][seg_ind]['actions'][y].items()
                
                for i in range(len(key[1])):
                    act_classid = activity2id[value[1]]
                    label=(act_classid)
                    #label.append(value[1])
                    out = key[1]
                    k,v = out[i].items()
                    bbox=(v[1][:4])     #somehow annotation is wrong here
                    x1,y1,x2,y2=bbox
                    new_x1= round(x1/height * new_size)
                    new_x2= round(x2/height * new_size)
                    new_y1= round(x1/width * new_size)
                    new_y2= round(x2/width * new_size)
                    bbox_new = (new_x1,new_y1,new_x2,new_y2)
                    assert(i < new_size for i in bbox_new)
                    #j = str(j)
                    #bbox = [j.split('[', 1)[1].split(']')[0]]
                    #bbox.append(split_results)
                    #bbox = [split_results]
                    bbox_new+=(label)

                    bboxes+=[bbox_new]
            gt += [bboxes]       
            #gt.update({count : [label,bbox]})
            #count+=1
            images.extend(seg_imgs)
      process_data = self.transform(images)
      #code to unroll the channel dimensrion to batch,channels
      process_data= np.array_split(process_data,8,axis =0)
      data = []
      dat = []
      for i in process_data:
        data = np.expand_dims(i, axis=0)
        dat += [data]
      image = np.concatenate(dat,axis=0)
      return image, gt'''
                    
    def __getitem__(self, index):
        record = self.video_list[index]
        #self.yaml_file(index)
        segment_indices = self._sample_indices(record)
        return self.get( index, record, segment_indices,self.new_size)
               
    def __len__(self):
        return (len(self.video_list))


#parse the yml file into the variables


