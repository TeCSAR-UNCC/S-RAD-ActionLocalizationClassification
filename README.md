# S-RAD Single Run Action Detector

# Introduction 
Single  Run  Action  Detector  (S-RAD)is a real-time, privacy-preserving  action  detector that performs end-to-end action localization and classification. It is based on Faster-RCNN combined with  temporal shift  modeling  and  segment  based sampling to capture the human  actions. Results on UCF-Sports and UR Fall dataset present comparable accuracy to State-of-the-Art approaches with significantly lower model size and computation demand and the ability for real-time execution on edge embedded device (e.g. Nvidia Jetson Xavier). The repository involves the usage of following methods:
* [Temporal Shift Module](https://arxiv.org/abs/1811.08383)
* [Faster R-CNN](https://arxiv.org/abs/1506.01497)
* [Temporal Segment Sampling](https://arxiv.org/abs/1608.00859)

# Overview

We release the Pytorch Code of S-RAD
![Screenshot](single_scale_base_implementation.png)

# Table of Contents:
* [Preparation](#Preparation)
  * [Pre-Requisites](#Pre-Requisites) 
  * [Compilation](#Compilation)
* [Dataset Preparation](#Dataset-Preparation)
* [Training](#Training)
   * [UCF-Sports](#UCF-Sports) 
   * [UR-Fall](#UR-Fall) 
* [Testing](#Testing)
   * [UCF-Sports](#UCF-Sports) 
   * [UR-Fall](#UR-Fall) 
* [Results](#Results)
   * [UCF-Sports](#UCF-Sports) 
   * [UR-Fall](#UR-Fall) 
* [Citation](#Citation)


## Preparation

First of all, clone the code
```
https://github.com/TeCSAR-UNCC/S-RAD-ActionLocalizationClassification.git
```

Then, move to the folder:
```
cd S-RAD
```

### Pre-Requisites

* Python 3
* Pytorch 1.4 and higher 
* CUDA 10.0 or higher

### Compilation

Install all the python dependencies using pip:
```
pip3 install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
python3 setup.py build develop
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. 

## Dataset Preparation

 1. Modify the dataset path , log directory , model directory in the config file to the path you are using:
 
      ```
      S-RAD/lib/model/utils/config.py
      ```
    All the model and dataset related parameters can be updated here in the config file according to the dataset and model used.
    
 2. The framelist of the two datasets are provided in the below path :
      ```
      S-RAD/dataset_config/UCF_Sports/frames_list/
      S-RAD/dataset_config/UR_falldataset/frame_list/
      ```
      Frames lists are in the format videopath, #of frames, Class label
    
 3. Change the path of the video in the framelist.txt files for all the dataset with the location that the dataset is stored in your environment
 
 3. The annotations for UR-Fall dataset is derived from the COCO pretrained on mmdetection and we had provided the bounding box annotation in the following path:
    
     ```
      S-RAD/dataset_config/UR_falldataset/annotation/
      S-RAD/dataset_config/UCF_Sports/ucfsports-anno/
      ```

## Train

Before training, set the right directory to save and load the trained models in *S-RAD/lib/model/utils/config.py* and modify the number of workers according to the batch size in the config file.

### UCF-Sports:

To train on UCF-sport with resnet50 with 8 segment per clip, simply run:

```
python3 trainval_net.py --dataset ucfsport --net res50 --bs 3 --lr 0.01 --lr_decay_step 60 --cuda --num_segments 8 --acc_step 2  --s 16 --epochs 300 --loss_type softmax --shift --shift_div 8 --shift_place blockres --tune_from kinetics_resnet50.pth --pathway naive
```
where 'bs' is the batch size with default 1,'s' is the session number to differentiate the training session,'epochs' is the value the maximum epoch,loss type is sigmoid by default,acc_step is the accumulation step (gradient accumulation is implemented), 'tune_from' is the the checkpoint   **V100 GPU accomodated batch size of 3 (24 frames) at lr_rate 0.01 , lr_decay_step of 60** To obtain the result as reported in the paper freeze the first block of Resnet in RESNET.FIXED_BLOCKS_1 of config file at S-RAD/lib/model/utils/config.py

### UR_Fall Dataset:

To train on UR_Fall dataset with resnet50 with 8 segment per clip, simply run:
```
python3 trainval_net.py --dataset urfall --net res50 --bs 4 --lr 0.02 --lr_decay_step 20 --cuda --pathway naive --num_segments 8 --acc_step 3  --s 12 --epochs 80 --loss_type softmax --shift --shift_div 8 --shift_place blockres --tune_from kinetics_resnet50.pth
```
## Test

### UCF-Sports:

If you want to evaluate the detection performance of a pre-trained res50 model on UCF sports test set, simply run
```
python3 trainval_net.py --dataset ucfsport --net res50 --bs 3 --cuda --num_segments 8 --loss_type softmax --shift --shift_div 8 --shift_place blockres --checkpoint 37 --checksession 45 --checkepoch 3 --r True --evaluate --eval_metrics --pathway naive
```
Specify the specific model session, checkepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416 and set the num_of_workers = 0 in the config file. 

### UR-Fall :

If you want to evaluate the detection performance of a pre-trained res50 model on UR Fall test set, simply run

```
python3 trainval_net.py --dataset urfall --net res50 --bs 3 --cuda --num_segments 8 --loss_type softmax --shift --shift_div 8 --shift_place blockres --checkpoint 13 --checksession 13 --checkepoch 41 --r True --evaluate --eval_metrics --pathway naive
```

## Results

### UCF-Sports
### UR-Fall

## Citation


