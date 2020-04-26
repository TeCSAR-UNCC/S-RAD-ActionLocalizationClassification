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

* Python 2.7 or 3.6
* Pytorch 1.0 
* CUDA 8.0 or higher

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

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

Modify the dataset path , log directory , model directory in the config file to the path you are using:

```
cd faster-rcnn.pytorch/lib/model/utils/config.py
```
All the model and dataset related parameters can be updated here in the config file according to the dataset and model you are using.

## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test.py to adapt to your environment.

To train a faster R-CNN model with resnet50 on virat with 8 segment per clip, simply run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python3 trainval_net.py \
                   --dataset virat --net res50 --bs $BATCH_SIZE --nw $NUM_WORKERS --lr $LEARNING_RATE \
                   --lr_decay_step $DECAY_STEP --cuda --shift --shift_div 8 --shift_place blockres --num_segments 8 \
                   --s $SESSION_NUM --epochs $MAX_EPOCHS --dense_sample --acc_step $acc_step --loss_type $loss_type

```
where 'bs' is the batch size with default 1,'s' is the session number to differentiate the training session,'epochs' is the value the maximum epoch should be, 'vis' to enable visualisation on validation data ,'dense_sample' to enable dense sampling ,it can be uniform_sampling,strided_sampling,random sampling,loss type is sigmoid by default  **V100 GPU accomodated batch size of 3 (24 frames) at lr_rate 0.04 , lr_decay_step of 15**

## Test

If you want to evlauate the detection performance of a pre-trained res50 model on virat test set, simply run
```
python test.py --dataset virat --net res50 --num_segments 8 --nw $NUM_WORKERS --bs 1 \
                --shift --shift_div 8 --shift_place blockres --lr $LEARNING_RATE \
                --lr_decay_step $DECAY_STEP --cuda --epochs $MAX_EPOCHS \
                --dense_sample --checksession $CHECKSESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.


