# Action-Proposal-Networks

## Preparation


First of all, clone the code
```
https://github.com/samerogers/Action-Proposal-Networks.git
```

Then, move to the folder:
```
cd faster-rcnn.pytorch
```

### prerequisites

* Python 3
* Pytorch 1.4 and higher 
* CUDA 10.0 or higher

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
python setup.py build develop
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. 

## Dataset Preparation

 1. Modify the dataset path , log directory , model directory in the config file to the path you are using:
 
      ```
          faster-rcnn.pytorch/lib/model/utils/config.py
      ```
    All the model and dataset related parameters can be updated here in the config file according to the dataset and model used.
 2. Change the path to the video in the framelist.txt files for all the dataset


## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test.py to adapt to your environment.

To train on UCF-sport with resnet50 with 8 segment per clip, simply run:

The UCF-Sports dataset consists of 150 videos from 10 action classes.
All videos contain spatio-temporal annotations in the form of frame-level bounding boxes.

```
python3 trainval_net.py --dataset ucfsport --net res50 --bs 3 --lr 0.01 --lr_decay_step 60 --cuda --num_segments 8 --acc_step 2  --s 16 --epochs 300 --loss_type softmax --shift --shift_div 8 --shift_place blockres --tune_from kinetics_pretrained_segment8.pth
```
where 'bs' is the batch size with default 1,'s' is the session number to differentiate the training session,'epochs' is the value the maximum epoch,loss type is sigmoid by default,acc_step is the accumulation step (gradient accumulation is implemented), 'tune_from' is the the checkpoint   **V100 GPU accomodated batch size of 3 (24 frames) at lr_rate 0.01 , lr_decay_step of 60**

## Test

If you want to evaluate the detection performance of a pre-trained res50 model on UCF sports test set, simply run
```
python3 trainval_net.py --dataset ucfsport --net res50 --bs 3 --lr 0.01 --lr_decay_step 60 --cuda --num_segments 8 --acc_step 2  --epochs 300 --loss_type softmax --shift --shift_div 8 --shift_place blockres --checkpoint 37 --checksession 45 --checkepoch 3 --r True --evaluate
```
Specify the specific model session, checkepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416 and set the num_of_workers = 0 in the config file.


