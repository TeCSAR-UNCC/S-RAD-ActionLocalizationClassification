#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
import cv2
#from fvcore.common.timer import Timer

import dataset_config.ava.ava_helper as ava_helper
#import slowfast.utils.logging as logging
import dataset_config.ava.metrics as metrics
#import slowfast.utils.misc as misc
from dataset_config.ava.ava_eval_helper import (
    evaluate_ava,
    read_csv,
    read_exclusions,
    read_labelmap,
)



def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret

def retry_load_images(image_paths, retry=10, backend="pytorch"):
        """
        This function is to load images with support of retrying for failed load.

        Args:
            image_paths (list): paths of images needed to be loaded.
            retry (int, optional): maximum time of loading retrying. Defaults to 10.
            backend (str): `pytorch` or `cv2`.

        Returns:
            imgs (list): list of loaded images.
        """
        for i in range(retry):
            imgs = [cv2.imread(image_path) for image_path in image_paths]

            if all(img is not None for img in imgs):
                if backend == "pytorch":
                    imgs = torch.as_tensor(np.stack(imgs))
                return imgs
            else:
                logger.warn("Reading failed. Will retry.")
                time.sleep(1.0)
            if i == retry - 1:
                raise Exception("Failed to load images {}".format(image_paths))


class AVAMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode,num_seg):
        """
            overall_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
            mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.lr = None
        #self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
        self.mode = mode
        self.num_segments = num_seg
        #self.iter_timer = Timer()
        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []
        self.overall_iters = overall_iters
        self.excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
        )
        gt_filename = os.path.join(
            cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE
        )
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist,False, num_seg =self.num_segments)
        self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)

        _, self.video_idx_to_name = ava_helper.load_image_lists(
            cfg, mode == "train"
       )

    

    def reset(self):
        """
        Reset the Meter.
        """
        #self.loss.reset()

        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []

    #def update_stats(self, preds, #ori_boxes, metadata, loss=None, lr=None):
    def update_stats(self, preds,metadata, loss=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            ori_boxes (tensor): original boxes (x1, y1, x2, y2).
            metadata (tensor): metadata of the AVA data.
            loss (float): loss value.
            lr (float): learning rate.
        """
        if self.mode in ["val", "test"]:
            self.all_preds.append(preds)
            #self.all_ori_boxes.append(ori_boxes)
            self.all_metadata.append(metadata)
        if loss is not None:
            self.loss.add_value(loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, log):
        """
        Calculate and log the final AVA metrics.
        """
        #all_preds = torch.cat(self.all_preds, dim=0)
        all_preds = [np.stack(self.all_preds[i],axis=0) for i in range(len(self.all_preds))]
        #all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
        #all_ori_boxes = np.stack(self.all_ori_boxes)
        # all_metadata = torch.cat(self.all_metadata, dim=0)
        all_metadata = np.stack(self.all_metadata)

        if self.mode == "test" or (self.full_ava_test and self.mode == "val"):
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth

        self.full_map = evaluate_ava(
            all_preds,
            #all_ori_boxes,
            all_metadata.tolist(),
            self.excluded_keys,
            self.class_whitelist,
            self.categories,
            groundtruth=groundtruth,
            video_idx_to_name=self.video_idx_to_name,log = log
        )
        self.log = log
        output = ('mAP for ava valid dataset: {0}'.format(self.full_map))
        print(output)
        log.write(output+ '\n')
        log.flush()
        '''if log:
            stats = {"mode": self.mode, "map": self.full_map}
            logging.log_json_stats(stats)'''

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            self.finalize_metrics(log=False)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "map": self.full_map,
            }
            logging.log_json_stats(stats)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

