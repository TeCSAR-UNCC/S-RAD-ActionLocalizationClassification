#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch
import cv2 

from . import ava_helper as ava_helper
from . import meters as meters
#from . import utils as utils
from . import cv2_transform as cv2_transform
logger = logging.getLogger(__name__)

class Ava(torch.utils.data.Dataset):
    """
    AVA Dataset
    """

    def __init__(self, cfg,split,num_segments,dense_sample=True,uniform_sample= True,
                 random_sample = False,strided_sample = False):
        self.cfg = cfg
        self._split = split
        #self._sample_rate = cfg.AVA.SAMPLING_RATE 
        self.num_segments = num_segments
        #self._video_length = cfg.AVA.NUM_FRAMES
        #self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.AVA.NUM_CLASSES
        self.ground_truth = cfg.AVA.GT_BOXES
        self.fps = cfg.AVA.TARGET_FPS
        self.dense_sample = dense_sample
        self.uniform_sample = uniform_sample
        self.random_sample = random_sample
        self.strided_sample = strided_sample
        # Augmentation params.
        self._data_mean = cfg.AVA.MEAN
        self._data_std = cfg.AVA.STD
        self._use_bgr = cfg.AVA.BGR
        self.random_horizontal_flip = cfg.AVA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.AVA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.AVA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.AVA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.AVA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.AVA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.AVA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP


        self._load_data(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        self._image_paths, self._video_idx_to_name = ava_helper.load_image_lists(
            cfg, is_train=(self._split == "train")
        )

        # Loading annotations for boxes and labels.
        boxes_and_labels = ava_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        assert len(boxes_and_labels) == len(self._image_paths)

        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels)

        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        #self.print_summary()

    '''def print_summary(self):
        logger.info("=== AVA dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))'''

    def __len__(self):
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)
        #boxes = transforms.ava_clip_boxes_to_image(boxes, height, width)
        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )
        
        #im_size_min= min(height,width)
        #im_scale = float(self._img_scale[0]/im_size_min)
        #imgs = [cv2.resize(img, None, None, fx=im_scale, fy=im_scale,
        #            interpolation=cv2.INTER_LINEAR)for img in imgs]
        # Convert image to CHW 
        # keeping BGR order
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        '''imgs = [
            np.ascontiguousarray(img).astype(np.float32)for img in imgs
        ]'''

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(
            boxes, self._crop_size, self._crop_size
        )

        return imgs, boxes

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]

        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + self.fps - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            seq = [(idx * t_stride + start_idx) % (self.fps) for idx in range(self.num_segments)]
            seq = [element+center_idx for element in seq]       
        elif self.uniform_sample:  # normal sample
            average_duration = (self.fps) // self.num_segments
            if average_duration > 0:
                seq = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration,
                                       size=self.num_segments)
            seq = [element+center_idx for element in seq] 
        elif self.random_sample:
            seq = np.sort(np.random.randint(self.fps + 1, size=self.num_segments))
            seq = [element+center_idx for element in seq]
        elif self.strided_sample:
            average_duration = (self.fps) // self.num_segments
            if average_duration > 0:
                seq = np.multiply(list(range(self.num_segments)), average_duration) + average_duration//2
                seq = [element+center_idx for element in seq]
        else:
            seq = np.zeros((self.num_segments,)) 
            seq = [element+center_idx for element in seq]   
            
        #assert that seq stays within the sequence/clip selected
        for seq_idx in range(len(seq)):
         if seq[seq_idx] < center_idx:
            seq[seq_idx] = center_idx
         elif seq[seq_idx] >= (self.fps)+center_idx: #num_frames:
            seq[seq_idx] = ((self.fps)+center_idx) - 1

        # Get the frame idxs for current clip.
        #seq = utils.get_sequence(
        #    center_idx,
        #    self._seq_len // 2,
        #    self._sample_rate,
        #    num_frames=len(self._image_paths[video_idx]),
        #)

        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        for box_labels in clip_label_list:
            boxes.append(box_labels[0])
            labels.append(box_labels[1])
        boxes = np.array(boxes)
        # Score is not used.
        boxes = boxes[:, :4].copy()
        ori_boxes = boxes.copy()

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame-1] for frame in seq]
        imgs = meters.retry_load_images(
            image_paths, backend=self.cfg.AVA.IMG_PROC_BACKEND
        )
        
        if self.cfg.AVA.IMG_PROC_BACKEND == "pytorch":
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            imgs, boxes = self._images_and_boxes_preprocessing(
                imgs, boxes=boxes
            )
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                imgs, boxes=boxes
            )
            imgs = imgs.permute(1,0,2,3)
        # Construct label arrays.
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        for i, box_labels in enumerate(labels):
            # AVA label index starts from 1.
            for label in box_labels:
                if label == -1:
                    continue
                assert label >= 1 and label <= 80
            #    label_arrs[i][label - 1] = 1
                label_arrs[i][label] = 1
        #imgs = utils.pack_pathway_output(self.cfg, imgs)
        metadata = [video_idx, sec] #* len(boxes) #changed here
        
        gt = np.zeros(((self.ground_truth),self._num_classes + 4),dtype = np.float32)
        for i,labels in enumerate(label_arrs):
            gt[i][:4] = boxes[i]
            gt[i][4:] = label_arrs[i]
        gt = np.repeat((np.expand_dims(gt,axis=0)),imgs.shape[0],axis=0)
        im_info = np.zeros((imgs.shape[0],3),dtype=np.float32)
        num_boxes = np.zeros((imgs.shape[0]),dtype=np.float32)
        for i,image in enumerate(imgs):
            im_info[i,:] = (image.shape[1],image.shape[2],(image.shape[1] / image.shape[2]))
            num_boxes[i] = boxes.shape[0]
        extra_data = {
           # "boxes": boxes,
           # "ori_boxes": ori_boxes,
            "metadata": metadata,
        }

        return imgs, gt, num_boxes,im_info,extra_data 
