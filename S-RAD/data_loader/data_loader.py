"""Data loader."""

import itertools
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from dataset_config.virat.transforms import *
from dataset_config.virat.data_load import *
from dataset_config.UCF_Sports.ucfsports import *
from dataset_config.JHMDB.JHMDB import *
from dataset_config.UCF101.UCF101 import *

from dataset_config.imvia_fd.imvia_fd import *

from dataset_config.UR_falldataset.urfalldataset import *


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    images, gt, num_boxes, im_info = zip(*batch)
    max_shape = np.array([imz.shape for imz in images]).max(axis=0)
    blob =np.zeros((len(images), max_shape[0], max_shape[1], max_shape[2],max_shape[3]),
                    dtype=np.float32)
    for i in range(len(images)):
           blob[i,:,:,0:images[i].shape[2], 0:images[i].shape[3]] = images[i]
    blob = torch.FloatTensor(blob)
    gt = torch.FloatTensor(gt)
    num_boxes = torch.LongTensor(num_boxes)
    im_info = torch.FloatTensor(im_info)

    return blob, gt, num_boxes, im_info

def construct_loader(cfg,dataset = None,num_segments = 8,batch_size = 3,
                     split = None,input_sampling = True, split_num = None,
                     interval = 3,pathway = "two_pathway"):
    
    if dataset == 'virat':
        num_class = cfg.VIRAT.NUM_CLASS
        normalize = GroupNormalize(cfg.VIRAT.INPUT_MEAN, cfg.VIRAT.INPUT_STD)
        if split == 'train':
           data_path = cfg.VIRAT.TRAIN_DATA
           listfile_path = cfg.VIRAT.FRAMELIST_TRAIN
           shuffle = True
           drop_last = True
        if split == 'val':
           data_path = cfg.VIRAT.VAL_DATA
           listfile_path = cfg.VIRAT.FRAMELIST_VAL
           shuffle = False
           drop_last = False
        if split == 'test':
           data_path = cfg.VIRAT.TEST_DATA
           listfile_path = cfg.VIRAT.FRAMELIST_TEST
           shuffle = False
           drop_last = False
        data_loader = torch.utils.data.DataLoader(
          VIRAT_dataset(data_path,cfg.VIRAT.NUM_CLASS,cfg,listfile_path, 
          num_segments=num_segments,input_size = cfg.VIRAT.INPUT_SIZE,
          transform=torchvision.transforms.Compose([ToTorchFormatTensor(div=1),normalize]),
          dense_sample =cfg.DENSE_SAMPLE,uniform_sample=cfg.UNIFORM_SAMPLE,random_sample = cfg.RANDOM_SAMPLE,
          strided_sample = cfg.STRIDED_SAMPLE),
          batch_size=batch_size, shuffle=shuffle,
          num_workers=cfg.NUM_WORKERS, pin_memory=True,
          drop_last=drop_last)
        return data_loader
        
       
    elif dataset == 'ucfsport':

        num_class = cfg.UCFSPORT.NUM_CLASSES
        image_set = 'UCF-Sports_RGB_1_split_0'
        normalize = GroupNormalize(cfg.UCFSPORT.INPUT_MEAN, cfg.UCFSPORT.INPUT_STD)
        if split == 'train':
            shuffle = True      #modify to true after debug
            drop_last = True
        if split == 'val':
            shuffle = False
            drop_last = False
        
        data_loader = torch.utils.data.DataLoader(ucfsports(cfg,image_set,
            PHASE = split,num_segments=num_segments,interval = interval,dense_sample = cfg.DENSE_SAMPLE,
            uniform_sample=cfg.UNIFORM_SAMPLE,random_sample = cfg.RANDOM_SAMPLE,
            strided_sample = cfg.STRIDED_SAMPLE,is_input_sampling = input_sampling,
            pathway = pathway,
            transform=torchvision.transforms.Compose([ToTorchFormatTensor(div=1),normalize])),
            batch_size=batch_size,shuffle = shuffle,num_workers=cfg.NUM_WORKERS,pin_memory=True,
            drop_last = drop_last)#,collate_fn=detection_collate)  #change shuffle to true after debuging
        return data_loader
    
    elif dataset == 'urfall':

        num_class = cfg.URFD.NUM_CLASSES
        #image_set = 'UCF-Sports_RGB_1_split_0'
        normalize = GroupNormalize(cfg.URFD.INPUT_MEAN, cfg.URFD.INPUT_STD)
        if split == 'train':
            shuffle = True      #modify to true after debug
            drop_last = True
        if split == 'val':
            shuffle = False
            drop_last = False
        
        data_loader = torch.utils.data.DataLoader(urfalldataset(cfg,
            PHASE = split,num_segments=num_segments,interval = interval,dense_sample = cfg.DENSE_SAMPLE,
            uniform_sample=cfg.UNIFORM_SAMPLE,random_sample = cfg.RANDOM_SAMPLE,
            strided_sample = cfg.STRIDED_SAMPLE,
            transform=torchvision.transforms.Compose([ToTorchFormatTensor(div=1),normalize])),
            batch_size=batch_size,shuffle = shuffle,num_workers=cfg.NUM_WORKERS,pin_memory=True,
            drop_last = drop_last,collate_fn=detection_collate)  #change shuffle to true after debuging
        return data_loader
    
    elif dataset == 'imfd':

        num_class = cfg.IMFD.NUM_CLASSES
        #image_set = 'UCF-Sports_RGB_1_split_0'
        normalize = GroupNormalize(cfg.IMFD.INPUT_MEAN, cfg.IMFD.INPUT_STD)
        if split == 'train':
            shuffle = True      #modify to true after debug
            drop_last = True
        if split == 'val':
            shuffle = False
            drop_last = False
        
        data_loader = torch.utils.data.DataLoader(imvia_fd(cfg,
            PHASE = split,num_segments=num_segments,interval = interval,dense_sample = cfg.DENSE_SAMPLE,
            uniform_sample=cfg.UNIFORM_SAMPLE,random_sample = cfg.RANDOM_SAMPLE,
            strided_sample = cfg.STRIDED_SAMPLE,
            transform=torchvision.transforms.Compose([ToTorchFormatTensor(div=1),normalize])),
            batch_size=batch_size,shuffle = shuffle,num_workers=cfg.NUM_WORKERS,pin_memory=True,
            drop_last = drop_last,collate_fn=detection_collate)  #change shuffle to true after debuging
        return data_loader


    elif dataset == 'jhmdb':

        num_class = cfg.JHMDB.NUM_CLASSES
        image_set = split_num
        normalize = GroupNormalize(cfg.JHMDB.INPUT_MEAN, cfg.JHMDB.INPUT_STD)
        if split == 'train':
            shuffle = True      #modify to true after debug
            drop_last = True
        if split == 'val':
            shuffle = False
            drop_last = False
        
        data_loader = torch.utils.data.DataLoader(JHMDB(cfg,image_set,
            PHASE = split,num_segments=num_segments,dense_sample = cfg.DENSE_SAMPLE,
            uniform_sample=cfg.UNIFORM_SAMPLE,random_sample = cfg.RANDOM_SAMPLE,
            strided_sample = cfg.STRIDED_SAMPLE,is_input_sampling = input_sampling,
            transform=torchvision.transforms.Compose([ToTorchFormatTensor(div=1),normalize])),
            batch_size=batch_size,shuffle = shuffle,num_workers=cfg.NUM_WORKERS,pin_memory=True,
            drop_last = drop_last,collate_fn=detection_collate)  #change shuffle to true after debuging
        return data_loader

    elif dataset == 'ucf24':

        num_class = cfg.UCF24.NUM_CLASSES
        #image_set = split_num
        normalize = GroupNormalize(cfg.UCF24.INPUT_MEAN, cfg.UCF24.INPUT_STD)
        if split == 'train':
            shuffle = True      #modify to true after debug
            drop_last = True
        if split == 'val':
            shuffle = False
            drop_last = False
        
        data_loader = torch.utils.data.DataLoader(UCF101(cfg,
            PHASE = split,num_segments=num_segments,dense_sample = cfg.DENSE_SAMPLE,
            uniform_sample=cfg.UNIFORM_SAMPLE,random_sample = cfg.RANDOM_SAMPLE,
            strided_sample = cfg.STRIDED_SAMPLE,is_input_sampling = input_sampling,
            transform=torchvision.transforms.Compose([ToTorchFormatTensor(div=1),normalize])),
            batch_size=batch_size,shuffle = shuffle,num_workers=cfg.NUM_WORKERS,pin_memory=True,
            drop_last = drop_last,collate_fn=detection_collate)  #change shuffle to true after debuging
        return data_loader
          
    else: 
       print("dataset is not defined")
  